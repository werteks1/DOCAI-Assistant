// Run with Playwright installed: node tests/session_isolation.mjs
// PLAYWRIGHT_MODULE may point to an existing playwright-core installation.
import assert from "node:assert/strict";
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const dist = fileURLToPath(new URL("../web/dist/", import.meta.url));
const server = http.createServer(async (request, response) => {
  const pathname = new URL(request.url, "http://localhost").pathname;
  const filename = path.resolve(dist, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!filename.startsWith(dist)) { response.writeHead(404).end(); return; }
  try {
    const body = await readFile(filename);
    response.setHeader("Content-Type", { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" }[path.extname(filename)] || "application/octet-stream");
    response.end(body);
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext();
const page = await context.newPage();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
const scan = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="white"/></svg>');
const file = name => ({ name, mimeType: "image/svg+xml", buffer: scan });
let extractionMode = "success";
let delayedExtraction;
let delayedPreview;
let holdPreview = false;
let extractionCalls = 0;
let cancelCalls = 0;
let pendingCancelledRequest;
let cancelBeforeRegistration = false;
let lastExportBody = null;

await context.route("**/api/**", async route => {
  const endpoint = new URL(route.request().url()).pathname;
  const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  if (endpoint === "/api/auth/login") {
    const { username } = route.request().postDataJSON();
    return json({ token: username, user: { username, role: "operator", must_change: false } });
  }
  if (endpoint === "/api/auth/me") {
    const username = route.request().headers().authorization?.slice(7);
    return json({ user: { username, role: "operator", must_change: false } });
  }
  if (endpoint === "/api/auth/logout") return json({ status: "ok" });
  if (endpoint === "/api/health") return json({ model: "test", backend: "test", models: ["test"] });
  if (endpoint === "/api/batch/duplicates") return json([[], []]);
  if (endpoint === "/api/export/excel") {
    lastExportBody = route.request().postDataJSON();
    return route.fulfill({ status: 200, contentType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", body: "fake-xlsx" });
  }
  if (/^\/api\/extract\/[^/]+\/cancel$/.test(endpoint)) {
    cancelCalls++;
    if (cancelBeforeRegistration) {
      cancelBeforeRegistration = false;
      return json({ detail: "Активный запрос не найден" }, 404);
    }
    if (delayedExtraction) pendingCancelledRequest = delayedExtraction;
    return json({ status: "cancelling" });
  }
  if (endpoint === "/api/preview") {
    if (holdPreview) { delayedPreview = route; return; }
    return route.fulfill({ contentType: "image/svg+xml", body: scan });
  }
  if (endpoint === "/api/extract") {
    extractionCalls++;
    if (extractionMode === "delay") { delayedExtraction = route; return; }
    if (extractionMode === "expired") return json({ detail: "Требуется вход" }, 401);
    if (extractionMode === "error") return json({ detail: "Не удалось распарсить ответ модели" }, 422);
    return json({ data: { "ФИО поступающего ученика": "Синтетический результат", "Контактный телефон": "9001234567" } });
  }
  return json({ detail: "Not found" }, 404);
});

await page.addInitScript(() => {
  window.testObjectUrls = new Set();
  const create = URL.createObjectURL.bind(URL);
  const revoke = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = blob => { const url = create(blob); window.testObjectUrls.add(url); return url; };
  URL.revokeObjectURL = url => { window.testObjectUrls.delete(url); return revoke(url); };
});

async function login(username) {
  await page.getByLabel("Логин", { exact: true }).fill(username);
  await page.getByLabel("Пароль", { exact: true }).fill("password");
  await page.getByRole("button", { name: "Войти", exact: true }).click();
  await page.getByRole("heading", { name: "Обработка документов" }).waitFor();
}

async function cleanWorkspace() {
  assert.deepEqual(await page.locator(".fields input").evaluateAll(inputs => inputs.map(input => input.value)), Array(10).fill(""));
  assert.equal(await page.locator(".document-image").count(), 0);
  assert.equal(await page.getByRole("button", { name: "Скачать результат в Excel" }).isDisabled(), true);
  await page.getByRole("button", { name: "Пакетная оцифровка", exact: true }).click();
  assert.equal(await page.locator(".batch-row").count(), 0);
  assert.equal(await page.locator("input[type=file][multiple]").evaluate(input => input.files.length), 0);
  await page.getByRole("button", { name: "Одиночный документ", exact: true }).click();
}

async function waitUntil(predicate) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  throw new Error("Timed out waiting for mocked request");
}

try {
  const origin = `http://127.0.0.1:${server.address().port}`;
  await page.goto(origin);
  await login("first");
  await page.locator("input[type=file]").setInputFiles(file("single.svg"));
  await page.getByRole("button", { name: "Распознать документ", exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll(".fields input")[1]?.value === "Синтетический результат");
  await page.getByRole("button", { name: "Пакетная оцифровка", exact: true }).click();
  await page.locator("input[type=file][multiple]").setInputFiles([file("batch.svg")]);
  await page.getByRole("button", { name: "Распознать папку" }).click();
  await page.locator(".batch-row .success").waitFor();
  await page.getByRole("button", { name: "Проверить ошибки" }).click();
  await page.locator(".review-doc img").waitFor();
  assert.equal(await page.evaluate(() => window.testObjectUrls.size), 1);
  await page.getByRole("button", { name: "Закрыть", exact: true }).click();
  await page.getByRole("button", { name: "Выйти", exact: true }).click();
  await login("second");
  await cleanWorkspace();
  assert.equal(await page.evaluate(() => window.testObjectUrls.size), 0);
  console.log("PASS: logout clears single document, batch, file inputs and preview URLs");

  extractionMode = "expired";
  await page.locator("input[type=file]").setInputFiles(file("expired.svg"));
  await page.getByRole("button", { name: "Распознать документ", exact: true }).click();
  await login("third");
  await cleanWorkspace();
  console.log("PASS: HTTP 401 clears document state before the next login");

  extractionMode = "delay";
  await page.getByRole("button", { name: "Пакетная оцифровка", exact: true }).click();
  await page.locator("input[type=file][multiple]").setInputFiles([file("old-one.svg"), file("old-two.svg")]);
  const previousCalls = extractionCalls;
  await page.getByRole("button", { name: "Распознать папку" }).click();
  await waitUntil(() => delayedExtraction);
  const otherTab = await context.newPage();
  await otherTab.goto(origin);
  await otherTab.evaluate(() => localStorage.removeItem("docai_token"));
  await login("fourth");
  await delayedExtraction.fulfill({ contentType: "application/json", body: JSON.stringify({ data: { "ФИО поступающего ученика": "Старый ответ" } }) }).catch(() => {});
  await cleanWorkspace();
  assert.equal(extractionCalls, previousCalls + 1);
  assert.equal(await page.evaluate(() => localStorage.getItem("docai_token")), "fourth");
  console.log("PASS: cross-tab logout isolates delayed responses and stops the old queue");

  extractionMode = "success";
  holdPreview = true;
  await page.getByRole("button", { name: "Пакетная оцифровка", exact: true }).click();
  await page.locator("input[type=file][multiple]").setInputFiles([file("delayed-preview.svg")]);
  await page.getByRole("button", { name: "Распознать папку" }).click();
  await page.locator(".batch-row .success").waitFor();
  await page.getByRole("button", { name: "Проверить ошибки" }).click();
  await waitUntil(() => delayedPreview);
  await otherTab.evaluate(() => localStorage.removeItem("docai_token"));
  await login("fifth");
  await delayedPreview.fulfill({ contentType: "image/svg+xml", body: scan });
  await cleanWorkspace();
  assert.equal(await page.evaluate(() => window.testObjectUrls.size), 0);
  console.log("PASS: a delayed preview cannot recreate document URLs after logout");

  extractionMode = "error";
  await page.getByRole("button", { name: "Пакетная оцифровка", exact: true }).click();
  await page.locator("input[type=file][multiple]").setInputFiles([file("invalid.svg")]);
  await page.getByRole("button", { name: "Распознать папку" }).click();
  await page.locator(".batch-row .failed").waitFor();
  assert.equal(await page.locator(".batch-row .success").count(), 0);
  assert.equal(await page.getByRole("button", { name: "Проверить ошибки" }).isDisabled(), true);
  console.log("PASS: recognition errors are displayed as failures instead of successful empty records");

  extractionMode = "delay";
  delayedExtraction = null;
  pendingCancelledRequest = null;
  cancelBeforeRegistration = true;
  const beforeCancel = cancelCalls;
  const beforeExtract = extractionCalls;
  await page.locator("input[type=file][multiple]").setInputFiles([file("cancel.svg"), file("queued.svg")]);
  await page.getByRole("button", { name: "Распознать папку" }).click();
  await waitUntil(() => delayedExtraction);
  await page.getByRole("button", { name: "Отменить обработку" }).click();
  await waitUntil(() => pendingCancelledRequest);
  assert.equal(cancelCalls, beforeCancel + 2);
  assert.equal(await page.getByRole("button", { name: "Отмена…", exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole("button", { name: "Обработка…", exact: true }).isDisabled(), true);
  await pendingCancelledRequest.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ detail: "Распознавание отменено" }) });
  await page.getByRole("button", { name: "Распознать папку" }).waitFor();
  assert.equal(extractionCalls, beforeExtract + 1);
  assert.equal(await page.locator(".batch-row .success").count(), 0);
  extractionMode = "success";
  await page.getByRole("button", { name: "Повторить неудавшиеся (2)" }).click();
  await page.waitForFunction(() => document.querySelectorAll(".batch-row .success").length === 2);
  console.log("PASS: cancellation retries early 404, waits for server completion, stops queue and permits retry");

  await page.getByRole("button", { name: "Проверить ошибки" }).click();
  await page.locator(".review-modal").waitFor();
  assert.equal(await page.locator(".field-group").count(), 3);
  assert.equal(await page.locator(".review-field").count(), 10);
  assert.equal(await page.locator(".review-progress").innerText(), "Проверено 0 из 2");
  await page.getByRole("button", { name: /Требуют сверки/ }).click();
  assert.equal(await page.locator(".review-field").count(), 3);
  await page.getByRole("button", { name: "Все поля" }).click();
  await page.getByRole("button", { name: "Документ 1 из 2" }).click();
  assert.equal(await page.locator(".review-files.open").count(), 1);
  await page.getByRole("button", { name: /queued\.svg/ }).click();
  assert.equal(await page.locator(".review-files.open").count(), 0);
  await page.getByRole("button", { name: "Документ 2 из 2" }).waitFor();
  await page.getByRole("button", { name: "Подтвердить и завершить" }).click();
  assert.equal(await page.locator(".review-modal").count(), 0);
  assert.equal(await page.locator(".verified-badge").count(), 1);

  await page.getByRole("button", { name: "Проверить ошибки" }).click();
  await page.locator(".review-modal").waitFor();
  assert.equal(await page.locator(".review-progress").innerText(), "Проверено 1 из 2");
  await page.getByRole("button", { name: "Подтвердить и перейти →" }).click();
  assert.equal(await page.locator(".review-progress.complete").count(), 1);
  assert.equal(await page.getByRole("button", { name: "Проверено ✓" }).isDisabled(), true);
  await page.keyboard.press("Escape");
  assert.equal(await page.locator(".review-modal").count(), 0);
  assert.equal(await page.locator(".verified-badge").count(), 2);
  await page.getByRole("button", { name: "Скачать результат в Excel" }).click();
  await waitUntil(() => lastExportBody);
  assert.deepEqual(lastExportBody.records.map(record => record["Статус проверки"]), ["Проверено", "Проверено"]);
  console.log("PASS: review confirms files, filters warnings, closes with Escape and exports verified status");
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
