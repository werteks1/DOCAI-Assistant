// Проверка консоли администратора: добавление, правка, переключение и удаление
// подключений. Запуск: node tests/settings_connections.mjs
// PLAYWRIGHT_MODULE может указывать на установленный playwright-core.
import assert from "node:assert/strict";
import http from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const settingsPage = fileURLToPath(new URL("../compiler/static/settings.html", import.meta.url));
const html = await readFile(settingsPage);

const server = http.createServer((_request, response) => {
  response.setHeader("Content-Type", "text/html; charset=utf-8");
  response.end(html);
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));

const state = {
  connections: [
    { id: "conn-1", name: "LM Studio", host: "http://127.0.0.1:1234", model: "qwen2.5-vl-7b", api_key_set: false, api_key_hint: "" },
    { id: "conn-2", name: "SiliconFlow", host: "https://api.siliconflow.com", model: "Qwen/Qwen2.5-VL-72B", api_key_set: true, api_key_hint: "sk…y" },
  ],
  active_connection_id: "conn-1",
  activeModels: ["qwen2.5-vl-7b"],
};
const calls = [];
const probeModels = [
  "qwen2.5-vl-7b", "qwen2.5-vl-32b", "qwen2.5-vl-72b", "llava-v1.6",
  "gemma-4-26b-a4b-it:free", "gemma-4-31b-it:free", "gemini-3-flash",
  "gpt-5-vision", "claude-haiku-4.5", "nova-premier-v1",
  "dots-3-note-preview:free", "inkling-small:free",
];
const activeConn = () => state.connections.find(conn => conn.id === state.active_connection_id) || {};
const payload = warning => ({
  exists: true,
  warning: warning || "",
  saved: {
    connections: state.connections.map(conn => ({ ...conn, active: conn.id === state.active_connection_id })),
    active_connection_id: state.active_connection_id,
    limit: 20,
    host: activeConn().host || "",
    model: activeConn().model || "",
    api_key_set: Boolean(activeConn().api_key_set),
    api_key_hint: activeConn().api_key_hint || "",
    params: { temperature: 0, max_tokens: 4096 },
  },
  active: {
    host: activeConn().host || "", backend: "LM Studio", model: activeConn().model || "",
    connection: activeConn().name || "", connection_id: activeConn().id || "",
    paddle_available: true, models: state.activeModels, connected: state.activeModels.length > 0,
    params: { temperature: 0, max_tokens: 4096 },
  },
});

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext();
const page = await context.newPage();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
page.on("dialog", dialog => dialog.accept());

await context.route("**/api/**", async route => {
  const request = route.request();
  const endpoint = new URL(request.url()).pathname;
  const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  if (endpoint === "/api/auth/me") return json({ user: { username: "admin", role: "admin", must_change: false } });
  if (endpoint === "/api/auth/logout") return json({ status: "ok" });
  if (endpoint === "/api/settings") {
    calls.push({ method: request.method(), path: endpoint, body: request.method() === "POST" ? request.postDataJSON() : null });
    return json(payload());
  }
  if (endpoint === "/api/connections/check") {
    calls.push({ method: "POST", path: endpoint, body: request.postDataJSON() });
    return json({ connected: true, host: "http://127.0.0.1:11434", backend: "Ollama", model: "qwen2.5-vl-7b", models: probeModels });
  }
  if (endpoint === "/api/connections") {
    const body = request.postDataJSON();
    calls.push({ method: "POST", path: endpoint, body });
    const id = `conn-${state.connections.length + 1}`;
    state.connections.push({
      id, name: body.name || "Новое", host: body.host, model: body.model || "",
      api_key_set: Boolean(body.api_key), api_key_hint: body.api_key ? "sk…w" : "",
    });
    if (body.activate) {
      state.active_connection_id = id;
      state.activeModels = probeModels;
    }
    return json(payload(body.activate ? "" : "Подключение сохранено, активно прежнее."));
  }
  const byId = endpoint.match(/^\/api\/connections\/([^/]+)$/);
  if (byId && request.method() === "POST") {
    const body = request.postDataJSON();
    calls.push({ method: "POST", path: endpoint, body });
    const conn = state.connections.find(item => item.id === byId[1]);
    if (body.name != null) conn.name = body.name;
    if (body.host) conn.host = body.host;
    if (body.api_key != null) {
      conn.api_key_set = Boolean(body.api_key);
      conn.api_key_hint = body.api_key ? "sk…n" : "";
    }
    return json(payload());
  }
  if (byId && request.method() === "DELETE") {
    calls.push({ method: "DELETE", path: endpoint });
    const id = byId[1];
    const wasActive = state.active_connection_id === id;
    state.connections = state.connections.filter(conn => conn.id !== id);
    let warning = "";
    if (wasActive) {
      state.active_connection_id = state.connections[0]?.id || "";
      if (!state.connections.length) {
        warning = "Ни одно из оставшихся подключений не ответило: включён сервер по умолчанию.";
      }
    }
    return json(payload(warning));
  }
  const activate = endpoint.match(/^\/api\/connections\/([^/]+)\/activate$/);
  if (activate) {
    calls.push({ method: "POST", path: endpoint, body: request.postDataJSON() });
    state.active_connection_id = activate[1];
    return json(payload());
  }
  if (endpoint === "/api/model") {
    calls.push({ method: "POST", path: endpoint, body: request.postDataJSON() });
    return json({ model: request.postDataJSON().model });
  }
  return json({ detail: "Not found" }, 404);
});
await page.addInitScript(() => { try { localStorage.setItem("docai_token", "t"); } catch { /* noop */ } });
await page.goto(`http://127.0.0.1:${server.address().port}/settings`, { waitUntil: "networkidle" });

const rows = page.locator("#connList .user-row");
await page.waitForSelector("#connList .user-row");
assert.equal(await rows.count(), 2, "список подключений должен показать оба сохранённых");
const listText = await page.locator("#connList").innerText();
assert.match(listText, /^.*LM Studio/m);
assert.match(listText, /активно/);
assert.match(listText, /ключ sk…y/);
assert.equal(await page.locator("#stConn").innerText(), "LM Studio");
assert.equal(await page.locator("#connList .flag-chip.on-chip").count(), 1);

// Выбор API: модели подтягиваются автоматически, без кнопки «Проверить».
assert.equal(await page.locator("#model option").count(), 1);
await page.locator("#seg button[data-prov='ollama']").click();
await page.waitForFunction(count =>
  document.querySelectorAll("#model option").length === count && !document.querySelector("#model").disabled,
  probeModels.length
);
assert.ok(calls.some(call => call.path === "/api/connections/check"), "проверка адреса уходит сама");
assert.match(await page.locator("#modelSource").innerText(), /127\.0\.0\.1:11434/);
assert.match(await page.locator("#msgForm").innerText(), /Список моделей обновлён/);

// Поиск по 400+ моделям: поле появляется на больших списках и фильтрует варианты.
assert.equal(await page.locator("#modelSearch").isVisible(), true);
await page.fill("#modelSearch", "qwen");
const qwenModels = probeModels.filter(m => m.includes("qwen"));
await page.waitForFunction(count => document.querySelectorAll("#model option").length === count, qwenModels.length);
assert.deepEqual(await page.locator("#model option").allInnerTexts(), qwenModels);
assert.match(await page.locator("#modelSource").innerText(), new RegExp(`Найдено ${qwenModels.length} из ${probeModels.length}`));
await page.fill("#modelSearch", "такой-модели-нет");
await page.waitForFunction(() => document.querySelector("#model").disabled === true);
assert.match(await page.locator("#model option").innerText(), /Ничего не найдено/);
await page.fill("#modelSearch", "");
await page.waitForFunction(count => document.querySelectorAll("#model option").length === count, probeModels.length);
assert.equal(await page.locator("#model").isDisabled(), false);
assert.deepEqual(await page.locator("#model option").allInnerTexts(), probeModels);

// Выбор модели для нового подключения запоминается до сохранения.
await page.selectOption("#model", "llava-v1.6");
assert.match(await page.locator("#msgMain").innerText(), /применится, когда подключение станет активным/);

// Добавление: пресет уже выбран, модель уходит вместе с подключением.
await page.fill("#connName", "Ollama кабинет");
await page.click("#btnAddConn");
await page.waitForFunction(() => document.querySelectorAll("#connList .user-row").length === 3);
const added = calls.find(call => call.path === "/api/connections");
assert.deepEqual(added.body, { name: "Ollama кабинет", host: "http://127.0.0.1:11434", api_key: "", activate: true, model: "llava-v1.6" });
assert.equal(await page.locator("#stConn").innerText(), "Ollama кабинет");
assert.match(await page.locator("#msgForm").innerText(), /добавлено/);

// Проверка адреса из формы.
await page.click("#btnCheck");
await page.waitForFunction(() => document.querySelector("#msgForm").textContent.includes("доступен"));
assert.match(await page.locator("#msgForm").innerText(), /Адрес доступен: Ollama/);

// Правка неактивного подключения: имя меняется, ключ не трогаем.
const siliconRow = page.locator("#connList .user-row", { hasText: "SiliconFlow" });
await siliconRow.getByRole("button", { name: "Изменить" }).click();
assert.equal(await page.locator("#connFormTitle").innerText(), "Изменить подключение");
assert.equal(await page.inputValue("#connName"), "SiliconFlow");
await page.fill("#connName", "SiliconFlow облако");
await page.click("#btnAddConn");
await page.waitForFunction(() => document.querySelector("#connFormTitle").textContent === "Добавить подключение");
const edited = calls.filter(call => call.path === "/api/connections/conn-2").at(-1);
assert.deepEqual(edited.body, { name: "SiliconFlow облако", host: "https://api.siliconflow.com" });
assert.match(await page.locator("#msgForm").innerText(), /Подключение сохранено/);
assert.match(await page.locator("#connList").innerText(), /SiliconFlow облако/);

// Переключение активного подключения.
await page.locator("#connList .user-row", { hasText: "LM Studio" }).getByRole("button", { name: "Подключить" }).click();
await page.waitForFunction(() => document.querySelector("#stConn").textContent === "LM Studio");
assert.ok(calls.some(call => call.path === "/api/connections/conn-1/activate"));

// Удаление: неактивное уходит молча, последнее активное — с предупреждением.
await page.locator("#connList .user-row", { hasText: "SiliconFlow облако" }).getByRole("button", { name: "Удалить" }).click();
await page.waitForFunction(() => document.querySelectorAll("#connList .user-row").length === 2);
assert.ok(calls.some(call => call.method === "DELETE" && call.path === "/api/connections/conn-2"));
assert.match(await page.locator("#msgConn").innerText(), /Подключение удалено/);
await page.locator("#connList .user-row", { hasText: "Ollama кабинет" }).getByRole("button", { name: "Удалить" }).click();
await page.waitForFunction(() => document.querySelectorAll("#connList .user-row").length === 1);
await page.locator("#connList .user-row", { hasText: "LM Studio" }).getByRole("button", { name: "Удалить" }).click();
await page.waitForFunction(() => document.querySelector("#stConn").textContent === "по умолчанию");
assert.match(await page.locator("#msgConn").innerText(), /Ни одно из оставшихся подключений не ответило/);
assert.match(await page.locator("#connList").innerText(), /Подключений пока нет/);

assert.deepEqual(errors, [], "в консоли не должно быть JS-ошибок");
await browser.close();
server.close();
console.log("PASS: подключения добавляются, правятся, переключаются и удаляются из консоли");
