import { useEffect, useMemo, useRef, useState } from "react";

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

function ReviewModal({ results, index, onClose, onSelect, onChange }) {
  const item = results[index];
  if (!item?.data) return null;
  const duplicates = item.data._duplicates || [];
  return <div className="modal-backdrop review-layer" role="dialog" aria-modal="true" aria-label="Проверка результатов">
    <div className="review-modal"><div className="review-head"><div><div className="modal-brand">ПРОВЕРКА ПЕРЕД ЭКСПОРТОМ</div><h2>Проверьте цифры и спорные поля</h2></div><button className="icon-button" onClick={onClose} aria-label="Закрыть">×</button></div><div className="review-layout"><nav className="review-files">{results.map((entry, fileIndex) => <button className={fileIndex === index ? "active" : ""} key={entry.filename} onClick={() => onSelect(fileIndex)}><span>{entry.filename}</span><small>{entry.data ? "Готово" : "Ошибка"}</small></button>)}</nav><div className="review-fields">{duplicates.length > 0 && <div className="duplicate-banner" role="alert"><strong>Возможный дубль в пачке</strong><span>{duplicates.map(duplicate => duplicate.reason).join(" · ")}</span><small>Совпадает с ранее обработанным файлом. Убедитесь, что это не повторный ввод одного ученика.</small></div>}<p className="review-hint">Жёлтым отмечены поля, которые стоит перепроверить вручную.</p>{FIELDS.map(field => { const value = item.data[field] || ""; const numeric = NUMBER_FIELDS.has(field); const warning = needsReview(field, value, item.data); return <label className={`review-field ${numeric ? "numeric-field" : ""} ${warning ? "warning-field" : ""}`} key={field}><span>{field}{warning && <b>Проверить</b>}</span><input value={value} onChange={event => onChange(index, field, event.target.value)} /></label>; })}</div></div><div className="review-actions"><button className="secondary" onClick={onClose}>Вернуться к списку</button><button className="primary" onClick={() => index < results.length - 1 ? onSelect(index + 1) : onClose}>{index < results.length - 1 ? "Следующий файл" : "Завершить проверку"}</button></div></div>
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
  const [batchJob, setBatchJob] = useState(null);
  const [batchCancelling, setBatchCancelling] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [reviewIndex, setReviewIndex] = useState(0);
  const [activeTab, setActiveTab] = useState("single");

  // --- Авторизация -----------------------------------------------------
  const [user, setUser] = useState(null);            // null — не вошёл
  const [checking, setChecking] = useState(true);    // проверка токена при загрузке
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");

  const jobIdRef = useRef(null);
  const pollTimerRef = useRef(null);

  useEffect(() => {
    const token = getToken();
    if (!token) { setChecking(false); return; }
    fetch(`${API}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async response => {
        if (response.status === 401) { clearToken(); setUser(null); }
        else { const payload = await response.json(); setUser(payload.user || null); }
      })
      .catch(() => clearToken())
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => () => { if (pollTimerRef.current) clearTimeout(pollTimerRef.current); }, []);

  // Индикатор подключения и список моделей — только для вошедшего (public /api/health).
  useEffect(() => {
    if (!user) return;
    fetch(`${API}/api/health`).then(r => r.json()).then(info => { setHealth(info); setModels(info.models || []); setModel(info.model || ""); }).catch(() => setHealth(null));
  }, [user]);

  const entries = useMemo(() => FIELDS.map(field => [field, data[field] || ""]), [data]);

  async function authedFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(`${API}${path}`, { ...options, headers });
    if (response.status === 401) {
      clearToken(); setUser(null); setChecking(false);
    }
    return response;
  }

  async function doLogin(username, password) {
    setAuthBusy(true); setAuthError("");
    try {
      const response = await fetch(`${API}/api/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Не удалось войти в систему.");
      setToken(payload.token); setUser(payload.user); setAuthError("");
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
    try { await authedFetch("/api/auth/logout", { method: "POST" }); } catch { /* сессия уже недействительна */ }
    clearToken(); setUser(null);
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

  function stopPolling() {
    if (pollTimerRef.current) { clearTimeout(pollTimerRef.current); pollTimerRef.current = null; }
  }

  function finishBatch(state) {
    stopPolling();
    jobIdRef.current = null;
    setBatchResults(state.items || []);
    setBatchJob(null);
    setBatchCancelling(false);
    if (state.status === "cancelled") setError("Пакетная обработка отменена. Показаны уже распознанные файлы.");
    else if (state.status === "error") setError(state.error || "Пакетная обработка завершилась ошибкой.");
  }

  function pollBatch(jobId) {
    stopPolling();
    authedFetch(`/api/batch/${jobId}`)
      .then(async response => {
        const state = await response.json();
        if (!response.ok) throw new Error(state.detail || "Ошибка получения статуса");
        if (state.status === "running") { setBatchJob(state); pollTimerRef.current = setTimeout(() => pollBatch(jobId), 700); }
        else finishBatch(state);
      })
      .catch(() => { pollTimerRef.current = setTimeout(() => pollBatch(jobId), 1500); });
  }

  async function cancelBatch() {
    if (!jobIdRef.current || batchCancelling) return;
    setBatchCancelling(true);
    try { await authedFetch(`/api/batch/${jobIdRef.current}/cancel`, { method: "POST" }); }
    catch { /* следующий poll сам переведёт джоб в cancelled */ }
  }

  async function processBatch() {
    if (!batchFiles.length) return;
    setError(""); setBatchResults([]); setBatchCancelling(false);
    setBatchJob({ status: "running", total: batchFiles.length, completed: 0, current_message: "Чтение файлов…" });
    try {
      const documents = await Promise.all(batchFiles.map(async selected => ({ filename: selected.name, image_base64: await readFileAsDataUrl(selected) })));
      const response = await authedFetch("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ documents, target_columns: FIELDS, ocr_priority: ocrPriority, detector, model: isAdmin ? (model || null) : null }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Ошибка запуска пакетной обработки");
      jobIdRef.current = payload.job_id;
      pollBatch(payload.job_id);
    } catch (reason) { setBatchJob(null); setError(reason.message); }
  }

  async function exportExcel() {
    const records = batchResults.filter(item => item.data).map(item => {
      const duplicates = item.data._duplicates || [];
      return { ...item.data, "Имя файла источника": item.filename, "Статус проверки": duplicates.length ? "Возможный дубль" : "Требует проверки" };
    });
    if (!records.length) return;
    const response = await authedFetch("/api/export/excel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records }) });
    if (!response.ok) { setError("Не удалось сформировать Excel-файл"); return; }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = "docai_results.xlsx"; link.click();
    URL.revokeObjectURL(url);
  }

  function updateBatchField(index, field, value) {
    setBatchResults(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, data: { ...item.data, [field]: value } } : item));
  }

  async function acceptCorrection(field, change) {
    const category = change.field_part || (field.includes("Паспорт") ? "organizations" : field.includes("Адрес") ? "cities" : "surnames");
    await authedFetch("/api/reference/corrections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category, wrong: change.from, right: change.to }) });
    setCorrections(current => ({ ...current, [field]: (current[field] || []).filter(item => item !== change) }));
  }

  const isPdf = file?.type === "application/pdf";
  const batchRunning = Boolean(batchJob && batchJob.status === "running");
  const reviewResults = batchResults.filter(item => item.data);
  const hasReviewableResults = reviewResults.length > 0;
  const duplicateCount = batchResults.reduce((sum, item) => sum + ((item.data?._duplicates || []).length ? 1 : 0), 0);
  return <main className="shell">
    <header><div><h1>Обработка документов</h1><p className="subtitle">Рукописный бланк → проверенные данные → Excel</p></div><div className="header-actions">{isAdmin && <a className="settings-link" href={`${API}/settings`} target="_blank" rel="noreferrer" title="Настройки сервера ИИ, пользователи и статистика (администратор)">⚙ Администрирование</a>}<div className="health">{health ? `● ${health.model} · ${health.backend || "ИИ"}` : "Компилятор не подключён"}</div><span className="user-chip" title={`Роль: ${isAdmin ? "администратор" : "пользователь"}`}>{user.username}</span><button className="logout-button" onClick={doLogout} title="Завершить сеанс">Выйти</button></div></header>
    <nav className="mode-switch" aria-label="Режим оцифровки"><span className={`mode-slider ${activeTab === "batch" ? "batch-active" : ""}`} aria-hidden="true" /><button className={activeTab === "single" ? "active" : ""} onClick={() => setActiveTab("single")} aria-selected={activeTab === "single"}>Одиночный документ</button><button className={activeTab === "batch" ? "active" : ""} onClick={() => setActiveTab("batch")} aria-selected={activeTab === "batch"}>Пакетная оцифровка</button></nav>
    {activeTab === "single" && <section className="workspace">
      <aside className="preview"><label className="upload"><input type="file" accept="image/*,.pdf" onChange={chooseFile}/><span>Выбрать документ</span><small>{file?.name || "PDF или изображение"}</small></label>{preview && (isPdf ? <div className="pdf-preview">PDF выбран<br/><small>Предпросмотр появится после распознавания</small></div> : <><div className="zoom-toolbar" aria-label="Масштаб документа"><button type="button" onClick={() => setZoom(current => Math.max(.5, Number((current - .25).toFixed(2))))} aria-label="Уменьшить">−</button><input className="zoom-range" type="range" min="50" max="300" step="10" value={Math.round(zoom * 100)} onChange={event => setZoom(Number(event.target.value) / 100)} aria-label="Масштаб в процентах"/><span>{Math.round(zoom * 100)}%</span><button type="button" onClick={() => setZoom(current => Math.min(3, Number((current + .25).toFixed(2))))} aria-label="Увеличить">+</button><button className="zoom-reset" type="button" onClick={() => setZoom(1)}>Сбросить</button></div><div className="image-viewport"><img className="document-image" src={preview} alt="Предпросмотр документа" style={{ transform: `scale(${zoom})` }}/></div></>)}</aside>
      <section className="panel"><div className="panel-head"><div><h2>Поля документа</h2><p className="panel-note">Проверьте распознанные значения перед экспортом</p></div><button className="primary" disabled={!preview || busy} onClick={extract}>{busy ? "Распознавание…" : "Распознать документ"}</button></div><div className="controls"><label>Приоритет OCR<select value={ocrPriority} onChange={e => setOcrPriority(e.target.value)}><option value="auto">Авто: OCR + VLM</option><option value="paddle">PaddleOCR-полосы</option><option value="vlm">Только VLM</option></select></label><label>Детектор<select value={detector} onChange={e => setDetector(e.target.value)}><option>PP-OCRv6_medium_det</option><option>PP-OCRv6_small_det</option></select></label>{isAdmin && <label className="model-control">Модель ИИ<select value={model} onChange={e => setModel(e.target.value)} disabled={!models.length}>{models.length ? models.map(item => <option key={item} value={item}>{item}</option>) : <option>Подключите сервер ИИ</option>}</select></label>}</div>{error && <div className="error">{error}</div>}<div className="fields">{entries.map(([field, value]) => <label className="field" key={field}><span>{field}</span><input value={value} onChange={e => setData({...data, [field]: e.target.value})}/></label>)}</div></section>
    </section>}
    {activeTab === "batch" && <section className="batch panel"><div className="panel-head"><div><h2>Пакетная оцифровка</h2><p className="panel-note">Обработайте до 50 файлов одной очередью</p></div><button className="primary" disabled={!batchFiles.length || batchRunning} onClick={processBatch}>{batchRunning ? "Обработка…" : "Распознать папку"}</button></div><label className="batch-upload"><input type="file" accept="image/*,.pdf" multiple onChange={e => { setBatchFiles(Array.from(e.target.files || [])); setBatchResults([]); setBatchJob(null); }}/><span>{batchFiles.length ? `Выбрано файлов: ${batchFiles.length}` : "Выбрать несколько PDF или сканов"}</span></label>{batchResults.length > 0 && <><div className="review-callout"><div><strong>Проверьте результат перед экспортом</strong><span>{duplicateCount ? `Найдено возможных дублей: ${duplicateCount}. ` : ""}Особое внимание уделите СНИЛС, телефону, датам и паспортным цифрам.</span></div><button className="secondary" disabled={!hasReviewableResults} onClick={() => { setReviewIndex(0); setReviewOpen(true); }}>Проверить ошибки</button></div><div className="batch-results">{batchResults.map(item => { const duplicates = (item.data?._duplicates) || []; return <div className="batch-row" key={item.filename}><span className="batch-file">{item.filename}{duplicates.length > 0 && <small className="dup-badge">Возможный дубль</small>}</span><strong className={item.error ? "failed" : item.data ? "success" : ""}>{item.error ? `Ошибка: ${item.error}` : item.data ? <button className="row-review" onClick={() => { setReviewIndex(reviewResults.findIndex(entry => entry.filename === item.filename)); setReviewOpen(true); }}>Проверить</button> : (item.message || "Не обработан")}</strong></div>; })}</div><button className="export" disabled={!hasReviewableResults} onClick={exportExcel}>Скачать результат в Excel</button></>}</section>}
    {Object.keys(corrections).length > 0 && <section className="suggestions"><h2>Предложения справочника</h2>{Object.entries(corrections).flatMap(([field, items]) => items.map(change => <div className="suggestion" key={`${field}-${change.from}-${change.to}`}><span>{change.from} → <strong>{change.to}</strong><small>{field}</small></span><button onClick={() => acceptCorrection(field, change)}>Добавить</button></div>))}</section>}
    <ProgressModal open={busy || batchRunning} mode={busy ? "single" : "batch"} job={batchJob} onCancel={batchRunning ? cancelBatch : null} cancelling={batchCancelling} />
    {reviewOpen && <ReviewModal results={reviewResults} index={reviewIndex} onClose={() => setReviewOpen(false)} onSelect={setReviewIndex} onChange={(index, field, value) => { const originalIndex = batchResults.findIndex(item => item.filename === reviewResults[index]?.filename); updateBatchField(originalIndex, field, value); }} />}
  </main>;
}
