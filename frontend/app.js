const SESSION_STORAGE_KEY = "shared-fitness.session-id";
const USER_STORAGE_KEY = "shared-fitness.user-id";
const SESSIONS_STORAGE_KEY = "shared-fitness.sessions";

function readSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(SESSIONS_STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.filter((item) => item && item.id).map((item) => ({ ...item, busy: false })) : [];
  } catch {
    return [];
  }
}

const state = {
  sessionId: localStorage.getItem(SESSION_STORAGE_KEY),
  userId: localStorage.getItem(USER_STORAGE_KEY) || "demo-user",
  sessions: readSessions(),
  mode: "stream",
  busy: false,
  lastMessage: "",
  lastTrace: [],
  safetyUsed: false,
  parallelUsed: false,
  handoffUsed: false,
  authToken: null,
  requests: new Map()
};

const $ = (selector) => document.querySelector(selector);
const messageList = $("#message-list");
const input = $("#message-input");
const sendButton = $("#send-button");
const statusNode = $("#connection-status");

const labels = {
  router_agent: "请求路由",
  venue_agent: "场馆 Agent",
  reservation_agent: "预约 Agent",
  device_agent: "设备 Agent",
  support_agent: "运维 Agent",
  general_agent: "通用 Agent",
  safety_agent: "安全 Agent",
  response_agent: "回复 Agent",
  task_planner_agent: "任务规划 Agent",
  context_agent: "共享上下文 Agent",
  risk_agent: "风险评估 Agent",
  domain_agent: "领域执行 Agent",
  conflict_resolution_agent: "冲突解决 Agent",
  human_handoff_agent: "人工接管 Agent"
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[character]));
}

function formatContent(value) {
  return escapeHtml(value).replace(/\n/g, "<br>");
}

function timeLabel() {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date());
}

function shortSession(sessionId) {
  return sessionId ? `SESSION / ${sessionId.slice(0, 8)}` : "正在创建...";
}

function persistSession() {
  if (state.sessionId) localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
  else localStorage.removeItem(SESSION_STORAGE_KEY);
}

function persistSessions() {
  localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(state.sessions.slice(0, 12)));
}

function getSession(sessionId = state.sessionId) {
  return state.sessions.find((item) => item.id === sessionId);
}

function ensureSession() {
  if (!state.sessionId) {
    state.sessionId = crypto.randomUUID();
    state.sessions.unshift({ id: state.sessionId, title: "新会话", createdAt: Date.now(), messages: [], pendingAction: null, busy: false });
    persistSession();
    persistSessions();
  } else if (!getSession()) {
    state.sessions.unshift({ id: state.sessionId, title: "已恢复会话", createdAt: Date.now(), messages: [], pendingAction: null, busy: false });
    persistSessions();
  }
  return getSession();
}

function saveSessionMessage(role, content, sessionId = state.sessionId) {
  const session = getSession(sessionId) || (sessionId === state.sessionId ? ensureSession() : null);
  if (!session) return;
  session.messages = session.messages || [];
  session.messages.push({ role, content, at: Date.now() });
  session.messages = session.messages.slice(-40);
  if (role === "user" && session.title === "新会话") session.title = content.slice(0, 22);
  persistSessions();
  renderSessionList();
}

function savePendingAction(pendingAction, sessionId = state.sessionId) {
  const session = getSession(sessionId);
  if (!session) return;
  session.pendingAction = pendingAction || null;
  persistSessions();
}

function renderSessionList() {
  const list = $("#session-list");
  if (!list) return;
  list.innerHTML = "";
  state.sessions.slice(0, 8).forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${session.id === state.sessionId ? " is-active" : ""}${session.busy ? " is-busy" : ""}`;
    const status = session.busy ? "后台处理中" : shortSession(session.id);
    button.innerHTML = `<span class="session-item-dot"></span><span class="session-item-copy"><strong>${escapeHtml(session.title || "未命名会话")}</strong><span>${escapeHtml(status)}</span></span>`;
    button.addEventListener("click", () => switchSession(session.id));
    list.appendChild(button);
  });
}

function renderSessionMessages(session) {
  messageList.innerHTML = "";
  if (!session || !session.messages?.length) {
    addMessage("assistant", "新会话已准备好。预约类操作会在创建前请求确认。", {});
  } else {
    session.messages.forEach((item) => addMessage(item.role, item.content));
    if (session.pendingAction && !session.busy) {
      const last = messageList.lastElementChild;
      if (last && session.messages[session.messages.length - 1].role === "assistant") {
        const action = session.pendingAction.type === "reservation_update" ? "确认修改" : "确认预约";
        addAction(last, action, action);
      }
    }
  }
  if (session?.busy) {
    const loadingArticle = addMessage("assistant", "", { loading: true });
    const request = state.requests.get(session.id);
    if (request) request.article = loadingArticle;
  }
}

function switchSession(sessionId) {
  if (sessionId === state.sessionId) return;
  const session = getSession(sessionId);
  if (!session) return;
  state.sessionId = sessionId;
  persistSession();
  renderSessionMessages(session);
  resetInspector();
  $("#session-short").textContent = shortSession(sessionId);
  $("#session-note").textContent = session.busy ? "该会话正在后台处理" : "已切换到本地会话";
  updateComposerState();
  renderSessionList();
  showToast(session.busy ? "已切换，会话将在后台继续处理" : "已切换会话，原会话内容已保留");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 2600);
}

function setConnection(healthy, model = "deepseek-r1:8b", mode = "simulator") {
  statusNode.classList.toggle("is-error", healthy === false);
  $("#status-label").textContent = healthy === null ? "检查服务中..." : healthy ? "服务在线" : "服务不可用";
  $("#model-name").textContent = `${model} / ${mode}`;
}

function updateComposerState() {
  const session = getSession();
  state.busy = Boolean(session?.busy);
  sendButton.disabled = state.busy;
  input.disabled = state.busy;
  sendButton.querySelector("span").textContent = state.busy ? "处理中..." : "发送请求";
}

async function ensureAuthToken() {
  if (state.authToken) return state.authToken;
  const query = encodeURIComponent(state.userId);
  const response = await fetch(`/auth/demo-token?user_id=${query}`, { method: "POST", headers: { "X-Demo-Login-Key": "local-demo-login" } });
  if (!response.ok) throw new Error("本地身份令牌获取失败");
  const data = await response.json();
  state.authToken = data.access_token;
  return state.authToken;
}

function addMessage(role, content = "", options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  if (options.loading) article.dataset.loading = "true";
  article.innerHTML = `
    <div class="avatar ${role === "user" ? "user-avatar" : "assistant-avatar"}">${role === "user" ? "YOU" : "SF"}</div>
    <div class="message-body">
      <div class="message-meta"><strong>${role === "user" ? "你" : "Fitness Orchestrator"}</strong><span>${timeLabel()}</span></div>
      <div class="message-content ${options.loading ? "thinking" : ""}">${options.loading ? "正在协调专业 Agent" : formatContent(content)}</div>
    </div>`;
  messageList.appendChild(article);
  messageList.scrollTop = messageList.scrollHeight;
  return article;
}

function updateAssistant(article, content, loading = false) {
  if (!article?.isConnected) return;
  const node = article.querySelector(".message-content");
  node.classList.toggle("thinking", loading);
  node.innerHTML = loading ? "正在协调专业 Agent" : formatContent(content);
  messageList.scrollTop = messageList.scrollHeight;
}

function addAction(article, label, message) {
  if (!article?.isConnected) return;
  const action = document.createElement("button");
  action.type = "button";
  action.className = "message-action";
  action.textContent = label;
  action.addEventListener("click", () => sendMessage(message));
  article.querySelector(".message-body").appendChild(action);
}

function resetInspector() {
  $("#route-name").textContent = "等待请求";
  $("#route-reason").textContent = "系统会根据问题和上下文选择合适的专业 Agent。";
  $("#route-caption").textContent = "等待请求";
  $("#fact-empty").classList.remove("is-hidden");
  $("#fact-content").classList.add("is-hidden");
  $("#activity-list").innerHTML = '<div class="empty-activity">发送请求后，这里会显示 Agent 和工具事件。</div>';
  state.safetyUsed = false;
  state.parallelUsed = false;
  state.handoffUsed = false;
  setRisk("low", "等待安全评估", "每次请求先检查当前表达中的风险信号；未授权的信息不会被推断。");
  document.querySelectorAll(".agent-row").forEach((row) => {
    row.classList.remove("is-active", "is-done");
    row.classList.add("is-idle");
    row.querySelector("small").textContent = "等待任务";
  });
}

function setRoute(route, reason = "") {
  const routeNames = { venue: "共享资源", reservation: "预约事务", device: "设备状态", support: "设备运维", general: "通用问答", safety: "安全分流" };
  $("#route-name").textContent = routeNames[route] || route || "智能路由";
  $("#route-reason").textContent = reason || "已匹配专业协同角色。";
  $("#route-caption").textContent = `${routeNames[route] || "SMART"} / ACTIVE`;
}

function updateAgent(agent, details = "") {
  const rows = [...$("#agent-stack").querySelectorAll(".agent-row")];
  const displayAgent = ["context_agent", "risk_agent", "domain_agent"].includes(agent)
    ? "parallel_agents" : agent;
  const ordered = ["router_agent", "task_planner_agent", "parallel_agents", "conflict_resolution_agent", "human_handoff_agent", "response_agent"];
  if (displayAgent === "parallel_agents") state.parallelUsed = true;
  if (agent === "safety_agent") state.safetyUsed = true;
  if (agent === "human_handoff_agent") state.handoffUsed = true;
  const current = ordered.indexOf(displayAgent);
  rows.forEach((row) => {
    const name = row.querySelector("strong").textContent;
    const index = ordered.indexOf(name);
    row.classList.toggle("is-active", name === displayAgent);
    row.classList.toggle("is-done", index >= 0 && index < current && (name !== "human_handoff_agent" || state.handoffUsed));
    if (name === displayAgent) row.querySelector("small").textContent = details || "执行中";
    else if (index >= 0 && index < current) row.querySelector("small").textContent = "已完成";
  });
}

function setRisk(level = "low", name = "未评估", reason = "") {
  const labelsByLevel = { low: "LOW", medium: "MEDIUM", high: "HIGH" };
  const label = $("#risk-label");
  label.textContent = labelsByLevel[level] || String(level).toUpperCase();
  label.className = `risk-label risk-${level}`;
  $("#risk-name").textContent = name;
  $("#risk-reason").textContent = reason || "暂无安全说明";
}

function addActivity(agent, details) {
  const list = $("#activity-list");
  const empty = list.querySelector(".empty-activity");
  if (empty) empty.remove();
  const item = document.createElement("div");
  item.className = "activity-item";
  item.innerHTML = `<div><strong>${escapeHtml(labels[agent] || agent)}</strong><span>${escapeHtml(details || "完成工作")}</span></div>`;
  list.appendChild(item);
  list.scrollTop = list.scrollHeight;
}

function showFact(toolName, result, args = {}) {
  $("#fact-empty").classList.add("is-hidden");
  $("#fact-content").classList.remove("is-hidden");
  $("#fact-tool").textContent = toolName || "business_tool";
  const argsText = Object.keys(args).length ? `\n\n调用参数：${JSON.stringify(args, null, 2)}` : "";
  $("#fact-result").textContent = `${result || "暂无工具结果"}${argsText}`;
}

function handleDone(data, assistantArticle, sessionId) {
  const session = getSession(sessionId);
  if (!session) return;
  session.busy = false;
  savePendingAction(data.pending_action, sessionId);
  if (data.response) {
    saveSessionMessage("assistant", data.response, sessionId);
    if (sessionId === state.sessionId) updateAssistant(assistantArticle, data.response, false);
  }
  if (sessionId === state.sessionId) {
    if (data.route) setRoute(data.route, data.route_reason);
    if (data.risk_level) {
      const riskNames = { low: "可继续处理", medium: "谨慎处理", high: "停止常规运动建议" };
      setRisk(data.risk_level, riskNames[data.risk_level] || "已评估", data.safety_notice || data.risk_reason);
    }
    if (data.agent_trace?.length) {
      state.lastTrace = data.agent_trace;
      data.agent_trace.forEach((item) => updateAgent(item.agent, item.details));
    }
    if (data.tool_results?.length) {
      const latest = data.tool_results[data.tool_results.length - 1];
      showFact(latest.tool_name, latest.result, latest.args);
    }
    if (data.pending_action) {
      const action = data.pending_action.type === "reservation_update" ? "确认修改" : "确认预约";
      addAction(assistantArticle, action, action);
    }
    $("#session-note").textContent = "会话已绑定当前演示用户";
    updateComposerState();
  } else {
    showToast(`${session.title || "后台会话"} 已处理完成`);
  }
  persistSessions();
  renderSessionList();
}

async function readSse(response, request) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  request.responseText = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const eventLine = frame.split("\n").find((line) => line.startsWith("event:"));
      const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;
      const event = eventLine.slice(6).trim();
      let data = {};
      try { data = JSON.parse(dataLine.slice(5).trim()); } catch { continue; }
      if (event === "agent" && request.sessionId === state.sessionId) {
        updateAgent(data.agent, data.details);
        addActivity(data.agent, data.details);
        if (data.agent === "safety_agent") setRisk("high", "停止常规运动建议", data.details);
      } else if (event === "tool_result" && request.sessionId === state.sessionId) {
        showFact(data.tool_name, data.result, data.args || {});
      } else if (event === "token") {
        request.responseText += data.content || "";
        if (request.sessionId === state.sessionId) updateAssistant(request.article, request.responseText, false);
      } else if (event === "done") {
        handleDone(data, request.article, request.sessionId);
      } else if (event === "error") {
        throw new Error(data.message || "流式请求失败");
      }
    }
  }
}

async function sendMessage(message) {
  const text = message.trim();
  if (!text) return;
  const session = ensureSession();
  if (session.busy) return;
  state.lastMessage = text;
  session.busy = true;
  updateComposerState();
  renderSessionList();
  resetInspector();
  addMessage("user", text);
  saveSessionMessage("user", text, session.id);
  const assistantArticle = addMessage("assistant", "", { loading: true });
  const request = { sessionId: session.id, article: assistantArticle, responseText: "" };
  state.requests.set(session.id, request);
  try {
    const authToken = await ensureAuthToken();
    const endpoint = state.mode === "stream" ? "/stream" : "/chat";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({ message: text, user_id: state.userId, session_id: session.id })
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `请求失败：${response.status}`);
    }
    if (state.mode === "stream") {
      await readSse(response, request);
    } else {
      const data = await response.json();
      handleDone(data, request.article, request.sessionId);
    }
  } catch (error) {
    session.busy = false;
    const errorText = `连接出现问题：${error.message}`;
    saveSessionMessage("assistant", errorText, session.id);
    if (session.id === state.sessionId) {
      updateAssistant(request.article, errorText, false);
      addAction(request.article, "重试", state.lastMessage);
      showToast(error.message);
      updateComposerState();
    } else {
      showToast(`${session.title || "后台会话"} 请求失败`);
    }
    persistSessions();
    renderSessionList();
  } finally {
    session.busy = false;
    state.requests.delete(session.id);
    persistSessions();
    renderSessionList();
    if (session.id === state.sessionId) {
      updateComposerState();
      $("#session-short").textContent = shortSession(session.id);
      input.focus();
    }
  }
}

async function loadSystem(manual = false) {
  const refreshButton = $("#refresh-status");
  refreshButton.classList.add("is-loading");
  refreshButton.setAttribute("aria-busy", "true");
  setConnection(null);
  if (manual) showToast("正在检查 Ollama 和 Agent 服务...");
  try {
    const [healthResponse, agentsResponse] = await Promise.all([fetch("/health"), fetch("/agents"), ensureAuthToken()]);
    if (!healthResponse.ok) throw new Error(`健康检查失败：${healthResponse.status}`);
    const health = await healthResponse.json();
    const agents = await agentsResponse.json();
    setConnection(health.status === "healthy", health.model, health.device_data_mode);
    $("#agent-count").textContent = agents.agents?.length || "--";
    if (manual) showToast(`服务正常 · ${health.model} · ${health.device_data_mode}`);
  } catch (error) {
    setConnection(false);
    showToast(`服务检查失败：${error.message || "无法连接后端"}`);
  } finally {
    refreshButton.classList.remove("is-loading");
    refreshButton.removeAttribute("aria-busy");
  }
}

$("#composer").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(input.value); input.value = ""; input.style.height = "auto"; });
input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#composer").requestSubmit(); } });
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 120)}px`; });
document.querySelectorAll(".prompt-chip").forEach((chip) => chip.addEventListener("click", () => { input.value = chip.dataset.prompt; input.focus(); }));
$("#voice-input")?.addEventListener("click", () => {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return showToast("当前浏览器不支持语音输入，请使用文字输入");
  const recognition = new Recognition();
  recognition.lang = "zh-CN";
  recognition.interimResults = false;
  recognition.onstart = () => showToast("正在听，请说出你的需求");
  recognition.onresult = (event) => { input.value = event.results[0][0].transcript; input.focus(); };
  recognition.onerror = () => showToast("语音识别失败，请重试或使用文字输入");
  recognition.start();
});
$("#large-text")?.addEventListener("click", () => {
  document.body.classList.toggle("is-large-text");
  const enabled = document.body.classList.contains("is-large-text");
  localStorage.setItem("shared-fitness.large-text", enabled ? "1" : "0");
  showToast(enabled ? "大字模式已开启" : "大字模式已关闭");
});
document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => { state.mode = button.dataset.mode; document.querySelectorAll(".mode-button").forEach((item) => item.classList.toggle("is-active", item === button)); }));
$("#new-session").addEventListener("click", () => {
  const previousSession = getSession();
  state.sessionId = crypto.randomUUID();
  state.sessions.unshift({ id: state.sessionId, title: "新会话", createdAt: Date.now(), messages: [], pendingAction: null, busy: false });
  state.sessions = state.sessions.slice(0, 12);
  persistSession();
  persistSessions();
  renderSessionMessages(getSession());
  resetInspector();
  $("#session-short").textContent = shortSession(state.sessionId);
  $("#session-note").textContent = previousSession?.busy ? "原会话仍在后台处理" : "新会话已创建，原会话已保留";
  updateComposerState();
  renderSessionList();
  showToast(previousSession?.busy ? "新会话已创建，原会话继续后台处理" : "已创建新会话，原会话仍可从列表切换");
});
$("#clear-activity").addEventListener("click", () => { $("#activity-list").innerHTML = '<div class="empty-activity">运行记录已清空。</div>'; });
$("#refresh-status").addEventListener("click", () => loadSystem(true));
$("#copy-session").addEventListener("click", async () => {
  if (!state.sessionId) return showToast("当前会话还没有 ID");
  if (!navigator.clipboard?.writeText) return showToast("当前浏览器不支持复制，请手动记录会话 ID");
  try {
    await navigator.clipboard.writeText(state.sessionId);
    showToast("会话 ID 已复制");
  } catch {
    showToast("复制失败，请检查浏览器剪贴板权限");
  }
});

if (!state.sessionId && state.sessions.length) {
  state.sessionId = state.sessions[0].id;
  persistSession();
}
if (state.sessionId && !getSession()) ensureSession();
const initialSession = getSession();
if (initialSession) {
  $("#session-short").textContent = shortSession(initialSession.id);
  $("#session-note").textContent = "已从本地恢复会话 ID";
}
renderSessionMessages(initialSession);
renderSessionList();
updateComposerState();
if (localStorage.getItem("shared-fitness.large-text") === "1") document.body.classList.add("is-large-text");
loadSystem();
