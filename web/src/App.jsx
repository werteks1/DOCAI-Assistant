import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// LAN-режим: фронтенд отдаёт сам компилятор (same-origin), поэтому путь пустой.
// Dev-режим: Vite-прокси направляет /api и /settings на компилятор.
const API = import.meta.env.VITE_API_URL || "";
const TOKEN_KEY = "docai_token";

const FIELDS = [
  "Дата подачи заявления", "ФИО поступающего ученика", "Дата рождения ребенка",
  "Класс / профиль обучения", "ФИО родителя / заявителя", "Паспортные данные",
  "Адрес регистрации / проживания", "Контактный телефон", "СНИЛС поступающего",
  "Особые отметки / льготы"
];

function getToken() { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } }
function setToken(token) { try { localStorage.setItem(TOKEN_KEY, token); } catch { /* приватный режим */ } }
function clearToken() { try { localStorage.removeItem(TOKEN_KEY); } catch { /* noop */ } }

function ProgressModal({ open, mode, job, onCancel, cancelling }) {
  if (!open) return null;
  if (mode === "single") {
    return <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Выполняется распознавание">
      <div className="progress-modal"><div className="loader-orbit"><span /></div><div className="modal-brand">DOCAI COMPILER</div><h2>Распознавание документа</h2><p>Распознавание полей…</p><div className="progress-meta"><span>VLM-модель читает рукописный бланк</span><strong>…</strong></div><small>Не закрывайте страницу до завершения операции</small></div>
    </div>;
  }
  const percent = job && job.total ? Math.round((job.completed / job.total) * 100) : 0;
  return <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Пакетная обработка">
    <div className="progress-modal"><div className="loader-orbit"><span /></div><div className="modal-brand">DOCAI COMPILER</div><h2>Пакетная оцифровка</h2><p>{job?.current_message || "Подготовка…"}</p><div className="progress-track"><span style={{ transform: `scaleX(${percent / 100})` }} /></div><div className="progress-meta"><span>Обработано {job?.completed ?? 0} из {job?.total ?? 0}</span><strong>{percent}%</strong></div>{onCancel && <button className="secondary cancel-batch" disabled={cancelling} onClick={onCancel}>{cancelling ? "Отмена…" : "Отменить обработку"}</button>}<small>Не закрывайте страницу до завершения операции</small></div>
  </div>;
}

const NUMBER_FIELDS = new Set(["Дата подачи заявления", "Дата рождения ребенка", "Паспортные данные", "Контактный телефон", "СНИЛС поступающего"]);

function needsReview(field, value, data) {
  const status = data?._fields?.[field]?.status;
  if (status === "unclear" || status === "needs_review") return true;
  if (field.includes("СНИЛС")) return String(value || "").replace(/\D/g, "").length !== 11;
  if (field.includes("телефон")) return String(value || "").replace(/\D/g, "").length < 10;
  if (field.includes("Дата")) return String(value || "").replace(/\D/g, "").length < 8;
  return false;
}

const FIELD_GROUPS = [
  { title: "Заявление", fields: ["Дата подачи заявления", "Особые отметки / льготы"] },
  { title: "Ученик", fields: ["ФИО поступающего ученика", "Дата рождения ребенка", "Класс / профиль обучения", "СНИЛС поступающего"] },
  { title: "Заявитель", fields: ["ФИО родителя / заявителя", "Паспортные данные", "Адрес регистрации / проживания", "Контактный телефон"] },
];

const MULTILINE_FIELDS = new Set(["Адрес регистрации / проживания", "Паспортные данные", "Особые отметки / льготы"]);

function pluralize(count, one, few, many) {
  const mod100 = count % 100;
  const mod10 = count % 10;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

function warningCount(data) {
  return FIELDS.filter(field => needsReview(field, data?.[field] || "", data)).length;
}

function ReviewModal({ results, loadPreview, index, onClose, onSelect, onChange, onConfirm }) {
  const item = results[index];
  const [zoom, setZoom] = useState(1);
  const [page, setPage] = useState(0);
  const [filesOpen, setFilesOpen] = useState(false);
  const [onlyWarnings, setOnlyWarnings] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const [preview, setPreview] = useState({ status: "loading", url: null });
  const viewportRef = useRef(null);
  const dragRef = useRef(null);
  const pages = item?.data?._pdf_pages || 1;

  useEffect(() => { setZoom(1); setPage(0); setFilesOpen(false); setOnlyWarnings(false); }, [index]);

  useEffect(() => {
    let cancelled = false;
    setPreview({ status: "loading", url: null });
    if (item?.sourceIndex != null) {
      loadPreview(item.sourceIndex, page)
        .then(url => { if (!cancelled) setPreview(url ? { status: "ready", url } : { status: "error", url: null }); })
        .catch(() => { if (!cancelled) setPreview({ status: "error", url: null }); });
    }
    return () => { cancelled = true; };
  }, [item?.sourceIndex, page, retryKey, loadPreview]);

  useEffect(() => {
    function onKey(event) { if (event.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!item?.data) return null;
  const duplicates = item.data._duplicates || [];
  const warnings = FIELDS.filter(field => needsReview(field, item.data[field] || "", item.data));
  const verifiedCount = results.filter(entry => entry.verified).length;

  function startPan(event) {
    if (zoom <= 1 || !viewportRef.current) return;
    dragRef.current = { pointerX: event.clientX, pointerY: event.clientY, scrollLeft: viewportRef.current.scrollLeft, scrollTop: viewportRef.current.scrollTop };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function movePan(event) {
    const drag = dragRef.current;
    if (!drag || !viewportRef.current) return;
    viewportRef.current.scrollLeft = drag.scrollLeft - (event.clientX - drag.pointerX);
    viewportRef.current.scrollTop = drag.scrollTop - (event.clientY - drag.pointerY);
  }

  function endPan() { dragRef.current = null; }

  function confirmAndAdvance() {
    onConfirm(item.sourceIndex);
    if (index < results.length - 1) onSelect(index + 1);
    else onClose();
  }

  return <div className="modal-backdrop review-layer" role="dialog" aria-modal="true" aria-label="Проверка результатов">
    <div className="review-modal">
      <div className="review-head">
        <div className="review-title"><div className="modal-brand">ПРОВЕРКА ПЕРЕД ЭКСПОРТОМ</div><h2 title={item.filename}>{item.filename}</h2></div>
        <div className="review-head-actions">
          <span className={`review-progress ${verifiedCount === results.length ? "complete" : ""}`}>Проверено {verifiedCount} из {results.length}</span>
          <button type="button" className="files-toggle" aria-expanded={filesOpen} onClick={() => setFilesOpen(open => !open)}>Документ {index + 1} из {results.length}</button>
          <button className="icon-button" onClick={onClose} aria-label="Закрыть">×</button>
        </div>
      </div>
      <div className="review-layout">
        {filesOpen && <button type="button" className="files-backdrop" aria-label="Закрыть список документов" onClick={() => setFilesOpen(false)} />}
        <nav className={`review-files ${filesOpen ? "open" : ""}`} aria-label="Файлы пачки">
          {results.map((entry, fileIndex) => {
            const count = warningCount(entry.data);
            return <button className={`${fileIndex === index ? "active" : ""} ${entry.verified ? "verified" : ""}`} key={entry.sourceIndex} onClick={() => { onSelect(fileIndex); setFilesOpen(false); }}>
              <span>{entry.filename}</span>
              <small>{entry.verified ? "Проверено" : count ? `${count} ${pluralize(count, "замечание", "замечания", "замечаний")}` : "Без замечаний"}</small>
            </button>;
          })}
        </nav>
        <figure className="review-doc">
          <div className="preview-toolbar">
            {pages > 1
              ? <div className="page-nav"><button type="button" aria-label="Предыдущая страница" disabled={page === 0} onClick={() => setPage(current => Math.max(0, current - 1))}>‹</button><span>{page + 1} / {pages}</span><button type="button" aria-label="Следующая страница" disabled={page === pages - 1} onClick={() => setPage(current => Math.min(pages - 1, current + 1))}>›</button></div>
              : <span className="preview-toolbar-label">Скан документа</span>}
            <div className="preview-zoom">
              <button type="button" onClick={() => setZoom(current => Math.max(0.5, Number((current - 0.25).toFixed(2))))} aria-label="Уменьшить скан">−</button>
              <span>{Math.round(zoom * 100)}%</span>
              <button type="button" onClick={() => setZoom(current => Math.min(3, Number((current + 0.25).toFixed(2))))} aria-label="Увеличить скан">+</button>
              <button type="button" className="zoom-reset" onClick={() => setZoom(1)}>По ширине</button>
            </div>
          </div>
          {preview.status === "ready"
            ? <div className={`preview-viewport ${zoom > 1 ? "pannable" : ""}`} ref={viewportRef} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan}><img src={preview.url} alt={`Скан документа ${item.filename}`} style={{ transform: `scale(${zoom})` }} draggable={false} /></div>
            : preview.status === "error"
              ? <div className="preview-viewport preview-error" role="status"><span>Не удалось загрузить скан</span><button type="button" className="secondary" onClick={() => setRetryKey(key => key + 1)}>Повторить</button></div>
              : <div className="preview-viewport preview-loading" role="status">Загрузка скана…</div>}
          <figcaption>{item.filename}{pages > 1 ? ` · страница ${page + 1} из ${pages}` : " · первая страница"}</figcaption>
        </figure>
        <div className="review-fields">
          <div className="fields-head" role="group" aria-label="Фильтр полей">
            <button type="button" className={onlyWarnings ? "" : "active"} onClick={() => setOnlyWarnings(false)}>Все поля</button>
            <button type="button" className={onlyWarnings ? "active" : ""} onClick={() => setOnlyWarnings(true)}>Требуют сверки{warnings.length ? ` · ${warnings.length}` : ""}</button>
          </div>
          {duplicates.length > 0 && <div className="duplicate-banner" role="alert"><strong>Возможный дубль в пачке</strong><span>{duplicates.map(duplicate => duplicate.reason).join(" · ")}</span><small>Совпадает с ранее обработанным файлом. Убедитесь, что это не повторный ввод одного ученика.</small></div>}
          {FIELD_GROUPS.map(group => {
            const visible = group.fields.filter(field => !onlyWarnings || warnings.includes(field));
            if (!visible.length) return null;
            return <section className="field-group" key={group.title}>
              <h3>{group.title}</h3>
              {visible.map(field => {
                const value = item.data[field] || "";
                const warning = warnings.includes(field);
                const alternative = item.data?._fields?.[field]?.alternative;
                const numeric = NUMBER_FIELDS.has(field);
                return <label className={`review-field ${numeric ? "numeric-field" : ""} ${warning ? "warning-field" : ""}`} key={field}>
                  <span>{field}{warning && <b>Проверить</b>}</span>
                  {MULTILINE_FIELDS.has(field)
                    ? <textarea rows={2} value={value} onChange={event => onChange(index, field, event.target.value)} />
                    : <input value={value} onChange={event => onChange(index, field, event.target.value)} />}
                  {alternative && <small>Проверочный фрагмент: {alternative}</small>}
                </label>;
              })}
            </section>;
          })}
          {onlyWarnings && warnings.length === 0 && <p className="fields-empty">Замечаний нет — все поля выглядят корректно.</p>}
        </div>
      </div>
      <div className="review-actions">
        <div className="review-nav">
          <button className="secondary" disabled={index === 0} onClick={() => onSelect(index - 1)}>‹ Назад</button>
          <button className="secondary" disabled={index === results.length - 1} onClick={() => onSelect(index + 1)}>Далее ›</button>
        </div>
        <button className="primary" disabled={Boolean(item.verified)} onClick={confirmAndAdvance}>{item.verified ? "Проверено ✓" : index === results.length - 1 ? "Подтвердить и завершить" : "Подтвердить и перейти →"}</button>
      </div>
    </div>
  </div>;
}

function LoginScreen({ busy, error, onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  function submit(event) { event.preventDefault(); onLogin(username.trim(), password); }
  return <main className="shell">
    <div className="auth-card"><div className="auth-brand">DOCAI · ВХОД</div><h1>Вход в систему</h1><p className="auth-note">Обработка документов и Excel-отчётность доступны только после входа.</p>
      <form onSubmit={submit}>
        <label className="auth-label">Логин<input autoFocus value={username} onChange={event => setUsername(event.target.value)} autoComplete="username" spellCheck="false" /></label>
        <label className="auth-label">Пароль<input type="password" value={password} onChange={event => setPassword(event.target.value)} autoComplete="current-password" /></label>
        {error && <div className="auth-error" role="alert">{error}</div>}
        <button className="primary auth-submit" disabled={busy || !username || !password}>{busy ? "Вход…" : "Войти"}</button>
      </form>
    </div>
  </main>;
}

function ChangePasswordScreen({ busy, error, onSave, onLogout }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [localError, setLocalError] = useState("");
  function submit(event) {
    event.preventDefault(); setLocalError("");
    if (next.length < 4) { setLocalError("Пароль слишком короткий — минимум 4 символа."); return; }
    if (next !== confirm) { setLocalError("Новые пароли не совпадают."); return; }
    if (next === current) { setLocalError("Новый пароль совпадает с текущим."); return; }
    onSave(current, next);
  }
  return <main className="shell">
    <div className="auth-card"><div className="auth-brand">DOCAI · ПЕРВЫЙ ВХОД</div><h1>Смените пароль</h1><p className="auth-note">Для этой учётной записи установлен временный пароль. Придумайте постоянный — вход без смены пароля невозможен.</p>
      <form onSubmit={submit}>
        <label className="auth-label">Текущий пароль<input type="password" value={current} onChange={event => setCurrent(event.target.value)} autoComplete="current-password" /></label>
        <label className="auth-label">Новый пароль<input type="password" value={next} onChange={event => setNext(event.target.value)} autoComplete="new-password" /></label>
        <label className="auth-label">Повторите новый пароль<input type="password" value={confirm} onChange={event => setConfirm(event.target.value)} autoComplete="new-password" /></label>
        {(error || localError) && <div className="auth-error" role="alert">{error || localError}</div>}
        <button className="primary auth-submit" disabled={busy}>{busy ? "Сохраняем…" : "Сохранить новый пароль"}</button>
        <button type="button" className="auth-ghost" onClick={onLogout}>Выйти</button>
      </form>
    </div>
  </main>;
}

export default function App() {
  const [sessionVersion, setSessionVersion] = useState(0);
  return <SessionApp key={sessionVersion} onSessionEnd={() => setSessionVersion(version => version + 1)} />;
}

function SessionApp({ onSessionEnd }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [zoom, setZoom] = useState(1);
  const [data, setData] = useState({});
  const [corrections, setCorrections] = useState({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [ocrPriority, setOcrPriority] = useState("auto");
  const [detector, setDetector] = useState("PP-OCRv6_medium_det");
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchMessage, setBatchMessage] = useState("");
  const [batchCancelling, setBatchCancelling] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [reviewIndex, setReviewIndex] = useState(0);
  const [activeTab, setActiveTab] = useState("single");

  // --- Авторизация -----------------------------------------------------
  const [user, setUser] = useState(null);            // null — не вошёл
  const [checking, setChecking] = useState(true);    // проверка токена при загрузке
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  // Пакетная очередь: результаты и отмена живут в ref, чтобы последовательные
  // запросы видели актуальное состояние без гонок со setState.
  const batchResultsRef = useRef([]);
  const cancelFlagRef = useRef(false);
  const abortRef = useRef(null);
  const operationRef = useRef(null);
  const cancelTimerRef = useRef(null);
  const previewCacheRef = useRef({});
  const activeRef = useRef(true);
  const sessionTokenRef = useRef(getToken());

  function sessionIsActive() {
    return activeRef.current && getToken() === sessionTokenRef.current;
  }

  function endSession() {
    if (!activeRef.current) return;
    activeRef.current = false;
    cancelFlagRef.current = true;
    stopCancelPolling();
    const operation = operationRef.current;
    if (operation) fetch(`${API}/api/extract/${operation.id}/cancel`, { method: "POST", headers: { Authorization: `Bearer ${sessionTokenRef.current}` } }).catch(() => {});
    abortRef.current?.abort();
    Object.values(previewCacheRef.current).forEach(url => URL.revokeObjectURL(url));
    previewCacheRef.current = {};
    // Старый запрос не должен удалить токен новой сессии в другой вкладке.
    if (getToken() === sessionTokenRef.current) clearToken();
    onSessionEnd();
  }

  useEffect(() => {
    function sessionChanged(event) {
      if ((event.key === TOKEN_KEY || event.key === null) && getToken() !== sessionTokenRef.current) endSession();
    }
    window.addEventListener("storage", sessionChanged);
    return () => {
      activeRef.current = false;
      cancelFlagRef.current = true;
      stopCancelPolling();
      abortRef.current?.abort();
      Object.values(previewCacheRef.current).forEach(url => URL.revokeObjectURL(url));
      window.removeEventListener("storage", sessionChanged);
    };
  }, []);

  useEffect(() => {
    const token = getToken();
    if (!token) { setChecking(false); return; }
    fetch(`${API}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async response => {
        if (!sessionIsActive()) return;
        if (response.status === 401) endSession();
        else { const payload = await response.json(); if (sessionIsActive()) setUser(payload.user || null); }
      })
      .catch(() => { if (sessionIsActive()) endSession(); })
      .finally(() => setChecking(false));
  }, []);

  // Индикатор подключения и список моделей — только для вошедшего (public /api/health).
  useEffect(() => {
    if (!user) return;
    fetch(`${API}/api/health`).then(r => r.json()).then(info => { setHealth(info); setModels(info.models || []); setModel(info.model || ""); }).catch(() => setHealth(null));
  }, [user]);

  const entries = useMemo(() => FIELDS.map(field => [field, data[field] || ""]), [data]);
  const hasExtractedData = entries.some(([, value]) => String(value || "").trim() !== "");

  // Все хуки обязаны вызываться до ранних return экранов авторизации,
  // иначе React получит разное число хуков между рендерами и упадёт.
  const loadPreview = useCallback(async (sourceIndex, page = 0) => {
    if (!sessionIsActive()) return null;
    const cacheKey = `${sourceIndex}:${page}`;
    if (previewCacheRef.current[cacheKey]) return previewCacheRef.current[cacheKey];
    const selected = batchFiles[sourceIndex];
    if (!selected) return null;
    const image_base64 = await readFileAsDataUrl(selected);
    const response = await authedFetch("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_base64, page }) });
    if (!response.ok) return null;
    const blob = await response.blob();
    if (!sessionIsActive()) return null;
    const url = URL.createObjectURL(blob);
    previewCacheRef.current[cacheKey] = url;
    return url;
  }, [batchFiles]);

  async function authedFetch(path, options = {}) {
    if (!sessionIsActive()) throw new DOMException("Сессия завершена", "AbortError");
    const headers = new Headers(options.headers || {});
    const token = sessionTokenRef.current;
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(`${API}${path}`, { ...options, headers });
    if (!sessionIsActive()) throw new DOMException("Сессия завершена", "AbortError");
    if (response.status === 401) {
      endSession();
      throw new DOMException("Сессия завершена", "AbortError");
    }
    return response;
  }

  async function doLogin(username, password) {
    setAuthBusy(true); setAuthError("");
    try {
      const response = await fetch(`${API}/api/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) });
      const payload = await response.json().catch(() => ({}));
      if (!sessionIsActive()) return;
      if (!response.ok) throw new Error(payload.detail || "Не удалось войти в систему.");
      setToken(payload.token); sessionTokenRef.current = payload.token;
      setUser(payload.user); setAuthError("");
    } catch (reason) { setAuthError(reason.message); }
    finally { setAuthBusy(false); }
  }

  async function doChangePassword(current, next) {
    setAuthBusy(true); setAuthError("");
    try {
      const response = await authedFetch("/api/auth/change-password", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ current_password: current, new_password: next }) });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Не удалось сменить пароль.");
      setUser(current => ({ ...current, must_change: false }));
    } catch (reason) { setAuthError(reason.message); }
    finally { setAuthBusy(false); }
  }

  async function doLogout() {
    const token = sessionTokenRef.current;
    endSession();
    try {
      await fetch(`${API}/api/auth/logout`, { method: "POST", headers: { Authorization: `Bearer ${token}` } });
    } catch { /* локальная сессия уже очищена */ }
  }

  // --- Экраны авторизации ---------------------------------------------
  if (checking) {
    return <main className="shell"><div className="auth-card"><div className="auth-brand">DOCAI</div><p className="auth-note">Проверка сессии…</p></div></main>;
  }
  if (!user) {
    return <LoginScreen busy={authBusy} error={authError} onLogin={doLogin} />;
  }
  if (user.must_change) {
    return <ChangePasswordScreen busy={authBusy} error={authError} onSave={doChangePassword} onLogout={doLogout} />;
  }

  // --- Рабочий экран (оператор и администратор) -----------------------
  const isAdmin = user.role === "admin";

  function chooseFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected); setError(""); setData({}); setCorrections({}); setZoom(1);
    const reader = new FileReader(); reader.onload = () => setPreview(reader.result); reader.readAsDataURL(selected);
  }

  async function extract() {
    if (!preview) return;
    setBusy(true); setError("");
    try {
      const response = await authedFetch("/api/extract", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: file?.name, image_base64: preview, target_columns: FIELDS, ocr_priority: ocrPriority, detector, model: isAdmin ? (model || null) : null }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Ошибка компилятора");
      setData(payload.data || {}); setCorrections(payload.data?._reference_corrections || {});
    } catch (reason) { setError(reason.message); }
    finally { setBusy(false); }
  }

  function readFileAsDataUrl(selected) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error(`Не удалось прочитать ${selected.name}`));
      reader.readAsDataURL(selected);
    });
  }

  function setResults(next) {
    if (!sessionIsActive()) return;
    batchResultsRef.current = next;
    setBatchResults(next);
  }

  function updateItem(sourceIndex, patch) {
    setResults(batchResultsRef.current.map(item => item.sourceIndex === sourceIndex ? { ...item, ...patch } : item));
  }

  async function refreshDuplicates() {
    // Дубли пересчитываются после каждого успешного файла: одна лёгкая запись на документ.
    const done = batchResultsRef.current.filter(item => item.status === "done" && item.data);
    if (done.length < 2) return;
    const records = done.map(item => Object.fromEntries(Object.entries(item.data).filter(([key]) => !key.startsWith("_"))));
    try {
      const response = await authedFetch("/api/batch/duplicates", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records, filenames: done.map(item => item.filename) }) });
      if (!response.ok) return;
      const found = await response.json();
      let doneIndex = 0;
      setResults(batchResultsRef.current.map(item => {
        if (item.status === "done" && item.data) {
          const duplicates = found[doneIndex++] || [];
          return { ...item, data: { ...item.data, _duplicates: duplicates } };
        }
        return item;
      }));
    } catch { /* дубли не критичны для результата */ }
  }

  async function processOne(sourceIndex) {
    const selected = batchFiles[sourceIndex];
    if (!selected) return;
    updateItem(sourceIndex, { status: "processing", error: null, message: "" });
    setBatchMessage(`Файл ${sourceIndex + 1} из ${batchFiles.length}: ${selected.name}`);
    try {
      const image_base64 = await readFileAsDataUrl(selected);
      if (cancelFlagRef.current || !sessionIsActive()) {
        updateItem(sourceIndex, { status: "pending", message: "" });
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      const operation = { id: Array.from(crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, "0")).join("") };
      operationRef.current = operation;
      const response = await authedFetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: operation.id, filename: selected.name, image_base64, target_columns: FIELDS, ocr_priority: ocrPriority, detector, model: isAdmin ? (model || null) : null, source: "batch" }),
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Ошибка компилятора");
      updateItem(sourceIndex, { status: "done", data: payload.data || {}, verified: false, error: null, message: "" });
      await refreshDuplicates();
    } catch (reason) {
      const aborted = reason?.name === "AbortError";
      updateItem(sourceIndex, { status: "error", data: null, message: "", error: aborted ? "Обработка отменена" : (reason?.message || "Не удалось выполнить запрос") });
    } finally {
      stopCancelPolling();
      operationRef.current = null;
      abortRef.current = null;
    }
  }

  async function runQueue(indexes) {
    setBatchRunning(true);
    setBatchCancelling(false);
    cancelFlagRef.current = false;
    for (const sourceIndex of indexes) {
      if (cancelFlagRef.current || !sessionIsActive()) break;
      await processOne(sourceIndex);
    }
    setBatchRunning(false);
    setBatchMessage("");
  }

  async function processBatch() {
    if (!batchFiles.length || batchRunning) return;
    setError("");
    Object.values(previewCacheRef.current).forEach(url => URL.revokeObjectURL(url));
    previewCacheRef.current = {};
    setResults(batchFiles.map((selected, index) => ({ filename: selected.name, sourceIndex: index, status: "pending", message: "", error: null, verified: false, data: null })));
    await runQueue(batchFiles.map((_, index) => index));
  }

  function retryItem(sourceIndex) {
    if (batchRunning) return;
    runQueue([sourceIndex]);
  }

  function retryFailed() {
    if (batchRunning) return;
    const targets = batchResultsRef.current.filter(item => item.status === "error" || item.status === "pending").map(item => item.sourceIndex);
    if (targets.length) runQueue(targets);
  }

  function stopCancelPolling() {
    if (cancelTimerRef.current) clearTimeout(cancelTimerRef.current);
    cancelTimerRef.current = null;
  }

  async function requestCancellation(operation) {
    if (!sessionIsActive() || operationRef.current !== operation) return;
    try {
      const response = await authedFetch(`/api/extract/${operation.id}/cancel`, { method: "POST" });
      if (response.ok) return;
    } catch { /* повторяем, пока исходный запрос не завершится */ }
    if (sessionIsActive() && operationRef.current === operation) {
      // Отмена может прийти раньше, чем сервер зарегистрирует загрузку.
      cancelTimerRef.current = setTimeout(() => requestCancellation(operation), 250);
    }
  }

  function cancelBatch() {
    if (!batchRunning || batchCancelling) return;
    setBatchCancelling(true);
    cancelFlagRef.current = true;
    if (operationRef.current) requestCancellation(operationRef.current);
  }

  async function exportExcel() {
    const records = batchResults.filter(item => item.data).map(item => {
      const duplicates = item.data._duplicates || [];
      const status = item.verified ? (duplicates.length ? "Проверено · возможный дубль" : "Проверено") : duplicates.length ? "Возможный дубль" : "Требует проверки";
      return { ...item.data, "Имя файла источника": item.filename, "Статус проверки": status };
    });
    if (!records.length) return;
    const response = await authedFetch("/api/export/excel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records }) });
    if (!response.ok) { setError("Не удалось сформировать Excel-файл"); return; }
    const blob = await response.blob();
    if (!sessionIsActive()) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = "docai_results.xlsx"; link.click();
    URL.revokeObjectURL(url);
  }

  async function exportSingleExcel() {
    if (!hasExtractedData) return;
    const record = { ...data, "Имя файла источника": file?.name || "", "Статус проверки": "Требует проверки" };
    const response = await authedFetch("/api/export/excel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records: [record] }) });
    if (!response.ok) { setError("Не удалось сформировать Excel-файл"); return; }
    const blob = await response.blob();
    if (!sessionIsActive()) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = "docai_results.xlsx"; link.click();
    URL.revokeObjectURL(url);
  }

  function updateBatchField(index, field, value) {
    // Ручная правка снимает подтверждение: документ снова требует сверки.
    setResults(batchResultsRef.current.map((item, itemIndex) => itemIndex === index ? { ...item, data: { ...item.data, [field]: value }, verified: false } : item));
  }

  function confirmItem(sourceIndex) {
    setResults(batchResultsRef.current.map(item => item.sourceIndex === sourceIndex ? { ...item, verified: true } : item));
  }

  async function acceptCorrection(field, change) {
    const category = change.field_part || (field.includes("Паспорт") ? "organizations" : field.includes("Адрес") ? "cities" : "surnames");
    await authedFetch("/api/reference/corrections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category, wrong: change.from, right: change.to }) });
    setCorrections(current => ({ ...current, [field]: (current[field] || []).filter(item => item !== change) }));
  }

  const isPdf = file?.type === "application/pdf";
  const reviewResults = batchResults.filter(item => item.data);
  const hasReviewableResults = reviewResults.length > 0;
  const verifiedCount = reviewResults.filter(item => item.verified).length;
  const duplicateCount = batchResults.reduce((sum, item) => sum + ((item.data?._duplicates || []).length ? 1 : 0), 0);
  const unfinishedCount = batchResults.filter(item => item.status === "error" || item.status === "pending").length;
  const completedCount = batchResults.filter(item => item.status === "done" || item.status === "error").length;
  const batchProgress = { total: batchFiles.length, completed: completedCount, current_message: batchMessage };
  return <main className="shell">
    <header><div><h1>Обработка документов</h1><p className="subtitle">Рукописный бланк → проверенные данные → Excel</p></div><div className="header-actions">{isAdmin && <a className="settings-link" href={`${API}/settings`} target="_blank" rel="noreferrer" title="Настройки сервера ИИ, пользователи и статистика (администратор)">⚙ Администрирование</a>}<div className="health">{health ? `● ${health.connection ? `${health.connection} · ` : ""}${health.model} · ${health.backend || "ИИ"}` : "Компилятор не подключён"}</div><span className="user-chip" title={`Роль: ${isAdmin ? "администратор" : "пользователь"}`}>{user.username}</span><button className="logout-button" onClick={doLogout} title="Завершить сеанс">Выйти</button></div></header>
    <nav className="mode-switch" aria-label="Режим оцифровки"><span className={`mode-slider ${activeTab === "batch" ? "batch-active" : ""}`} aria-hidden="true" /><button className={activeTab === "single" ? "active" : ""} onClick={() => setActiveTab("single")} aria-selected={activeTab === "single"}>Одиночный документ</button><button className={activeTab === "batch" ? "active" : ""} onClick={() => setActiveTab("batch")} aria-selected={activeTab === "batch"}>Пакетная оцифровка</button></nav>
    {activeTab === "single" && <section className="workspace">
      <aside className="preview"><label className="upload"><input type="file" accept="image/*,.pdf" onChange={chooseFile}/><span>Выбрать документ</span><small>{file?.name || "PDF или изображение"}</small></label>{preview && (isPdf ? <div className="pdf-preview">PDF выбран<br/><small>Предпросмотр появится после распознавания</small></div> : <><div className="zoom-toolbar" aria-label="Масштаб документа"><button type="button" onClick={() => setZoom(current => Math.max(.5, Number((current - .25).toFixed(2))))} aria-label="Уменьшить">−</button><input className="zoom-range" type="range" min="50" max="300" step="10" value={Math.round(zoom * 100)} onChange={event => setZoom(Number(event.target.value) / 100)} aria-label="Масштаб в процентах"/><span>{Math.round(zoom * 100)}%</span><button type="button" onClick={() => setZoom(current => Math.min(3, Number((current + .25).toFixed(2))))} aria-label="Увеличить">+</button><button className="zoom-reset" type="button" onClick={() => setZoom(1)}>Сбросить</button></div><div className="image-viewport"><img className="document-image" src={preview} alt="Предпросмотр документа" style={{ transform: `scale(${zoom})` }}/></div></>)}</aside>
      <section className="panel"><div className="panel-head"><div><h2>Поля документа</h2><p className="panel-note">Проверьте распознанные значения перед экспортом</p></div><button className="primary" disabled={!preview || busy} onClick={extract}>{busy ? "Распознавание…" : "Распознать документ"}</button></div><div className="controls"><label>Режим проверки<select value={ocrPriority} onChange={e => setOcrPriority(e.target.value)}><option value="auto">VLM + проверка важных полей</option><option value="paddle">VLM + проверка всех полей</option><option value="vlm">Только полный проход VLM</option></select></label><label>Детектор полей<select value={detector} onChange={e => setDetector(e.target.value)}><option>PP-OCRv6_medium_det</option><option>PP-OCRv6_small_det</option></select></label>{isAdmin && <label className="model-control">Модель ИИ<select value={model} onChange={e => setModel(e.target.value)} disabled={!models.length}>{models.length ? models.map(item => <option key={item} value={item}>{item}</option>) : <option>Подключите сервер ИИ</option>}</select></label>}</div>{error && <div className="error">{error}</div>}<div className="fields">{entries.map(([field, value]) => { const alternative = data?._fields?.[field]?.alternative; return <label className="field" key={field}><span>{field}</span><input value={value} onChange={e => setData({...data, [field]: e.target.value})}/>{alternative && <small className="verification-alternative">Проверочный фрагмент: {alternative}</small>}</label>; })}</div><button className="export" disabled={!hasExtractedData || busy} onClick={exportSingleExcel}>Скачать результат в Excel</button></section>
    </section>}
    {activeTab === "batch" && <section className="batch panel"><div className="panel-head"><div><h2>Пакетная оцифровка</h2><p className="panel-note">До 50 файлов — каждое заявление отправляется отдельным запросом</p></div><button className="primary" disabled={!batchFiles.length || batchRunning} onClick={processBatch}>{batchRunning ? "Обработка…" : "Распознать папку"}</button></div><label className="batch-upload"><input type="file" accept="image/*,.pdf" multiple onChange={e => { setBatchFiles(Array.from(e.target.files || [])); setResults([]); }}/><span>{batchFiles.length ? `Выбрано файлов: ${batchFiles.length}` : "Выбрать несколько PDF или сканов"}</span></label>{batchResults.length > 0 && <><div className="review-callout"><div><strong>Проверьте результат перед экспортом</strong><span>{reviewResults.length > 0 ? `Проверено ${verifiedCount} из ${reviewResults.length}. ` : ""}{duplicateCount ? `Найдено возможных дублей: ${duplicateCount}. ` : ""}Особое внимание уделите СНИЛС, телефону, датам и паспортным цифрам.</span></div><div className="callout-actions">{unfinishedCount > 0 && !batchRunning && <button className="secondary retry-failed" onClick={retryFailed}>Повторить неудавшиеся ({unfinishedCount})</button>}<button className="secondary" disabled={!hasReviewableResults} onClick={() => { setReviewIndex(0); setReviewOpen(true); }}>Проверить ошибки</button></div></div><div className="batch-results">{batchResults.map(item => { const duplicates = (item.data?._duplicates) || []; const statusText = item.status === "processing" ? "Распознавание…" : item.status === "pending" ? (batchRunning ? "В очереди" : "Не обработан") : null; return <div className={`batch-row ${item.status === "processing" ? "row-processing" : ""}`} key={item.sourceIndex}><span className="batch-file">{item.filename}{item.verified && <small className="verified-badge">Проверено</small>}{duplicates.length > 0 && <small className="dup-badge">Возможный дубль</small>}</span><strong className={item.status === "error" ? "failed" : item.status === "done" ? "success" : "muted"}>{item.status === "error" ? <>{`Ошибка: ${item.error}`}<button className="row-review row-retry" disabled={batchRunning} onClick={() => retryItem(item.sourceIndex)}>Повторить</button></> : item.status === "done" ? <button className="row-review" onClick={() => { setReviewIndex(reviewResults.findIndex(entry => entry.sourceIndex === item.sourceIndex)); setReviewOpen(true); }}>Проверить</button> : statusText}</strong></div>; })}</div><button className="export" disabled={!hasReviewableResults} onClick={exportExcel}>Скачать результат в Excel</button></>}</section>}
    {Object.keys(corrections).length > 0 && <section className="suggestions"><h2>Предложения справочника</h2>{Object.entries(corrections).flatMap(([field, items]) => items.map(change => <div className="suggestion" key={`${field}-${change.from}-${change.to}`}><span>{change.from} → <strong>{change.to}</strong><small>{field}</small></span><button onClick={() => acceptCorrection(field, change)}>Добавить</button></div>))}</section>}
    <ProgressModal open={busy || batchRunning} mode={busy ? "single" : "batch"} job={batchProgress} onCancel={batchRunning ? cancelBatch : null} cancelling={batchCancelling} />
    {reviewOpen && <ReviewModal results={reviewResults} loadPreview={loadPreview} index={reviewIndex} onClose={() => setReviewOpen(false)} onSelect={setReviewIndex} onConfirm={confirmItem} onChange={(index, field, value) => { updateBatchField(reviewResults[index]?.sourceIndex, field, value); }} />}
  </main>;
}
