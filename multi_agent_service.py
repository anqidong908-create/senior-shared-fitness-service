"""FastAPI service exposing the shared-fitness multi-agent graph."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import (
    APP_ENV,
    CORS_ORIGINS,
    DEVICE_DATA_MODE,
    MAX_HISTORY_MESSAGES,
    MAX_SESSIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SESSION_TTL_SECONDS,
    SESSION_DB_PATH,
    AUTH_SECRET,
    DEMO_LOGIN_KEY,
)
from multi_agent_runtime import build_graph, extract_result, make_state
from ollama_client import OllamaChatClient, OllamaConnectionError
from business_tools import TOOLS, resource_recommendation_data
from auth import issue_token, verify_token


logger = logging.getLogger("shared_fitness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="用户问题")
    user_id: str = Field(default="demo-user", min_length=1, max_length=100)
    session_id: Optional[str] = Field(default=None, max_length=100)

    @field_validator("message", "user_id", "session_id")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("字段不能是空白字符串")
        return value


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    route: str
    route_reason: str
    response: str
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    risk_level: str = "low"
    risk_reason: str = ""
    safety_notice: str = ""
    context_metadata: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    task_plan: list[str] = Field(default_factory=list)
    conflict_result: str = ""
    handoff_result: str = ""


class ResourceRecommendRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    activity_type: str = Field(default="", max_length=100)
    max_distance_km: float = Field(default=5.0, ge=0.1, le=50)


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="demo-user", min_length=1, max_length=100)
    resource_id: str = Field(default="", max_length=100)
    comfort: str = Field(default="舒适", max_length=50)
    venue_feedback: str = Field(default="", max_length=500)
    save_preference: bool = False


class CheckInRequest(BaseModel):
    reservation_id: str = Field(min_length=4, max_length=100)
    user_id: str = Field(default="demo-user", min_length=1, max_length=100)
    method: str = Field(default="manual", max_length=30)


class PermissionRequest(BaseModel):
    user_id: str = Field(default="demo-user", min_length=1, max_length=100)
    permission: str = Field(min_length=1, max_length=50)
    enabled: bool


class SessionAccessError(Exception):
    """Raised when a session belongs to another user."""


class SessionStore:
    """SQLite-backed session store with per-session locking and ownership."""

    def __init__(self, db_path: str = SESSION_DB_PATH) -> None:
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db_lock = threading.RLock()
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_locks_lock = threading.Lock()
        with self._db_lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, history TEXT NOT NULL, pending_action TEXT, memory TEXT NOT NULL, last_access REAL NOT NULL)"
            )
            self._db.commit()

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._session_locks_lock:
            return self._session_locks.setdefault(session_id, threading.RLock())

    @contextmanager
    def claim(self, session_id: str, user_id: str):
        lock = self._session_lock(session_id)
        with lock:
            yield

    def _prune_locked(self, now: float) -> None:
        self._db.execute("DELETE FROM sessions WHERE last_access < ?", (now - SESSION_TTL_SECONDS,))
        count = self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count >= MAX_SESSIONS:
            remove_count = count - MAX_SESSIONS + 1
            self._db.execute(
                "DELETE FROM sessions WHERE session_id IN (SELECT session_id FROM sessions ORDER BY last_access ASC LIMIT ?)",
                (remove_count,),
            )

    def load(self, session_id: str, user_id: str) -> tuple[list[dict[str, str]], dict[str, Any] | None, dict[str, Any]]:
        now = time.time()
        with self._db_lock:
            self._prune_locked(now)
            row = self._db.execute("SELECT user_id, history, pending_action, memory FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row and row[0] != user_id:
                raise SessionAccessError("session does not belong to this user")
            if not row:
                self._db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)", (session_id, user_id, "[]", None, "{}", now))
                self._db.commit()
                return [], None, {}
            self._db.execute("UPDATE sessions SET last_access = ? WHERE session_id = ?", (now, session_id))
            self._db.commit()
            return json.loads(row[1])[-MAX_HISTORY_MESSAGES:], json.loads(row[2]) if row[2] else None, json.loads(row[3] or "{}")

    def save_result(self, session_id: str, user_id: str, user_message: str, assistant_message: str, pending_action: dict[str, Any] | None, memory_patch: dict[str, Any]) -> None:
        with self._db_lock:
            row = self._db.execute("SELECT user_id, history, memory FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not row or row[0] != user_id:
                raise SessionAccessError("session does not belong to this user")
            history = json.loads(row[1])
            history.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": assistant_message}])
            memory = json.loads(row[2] or "{}")
            memory.update(memory_patch)
            self._db.execute(
                "UPDATE sessions SET history = ?, pending_action = ?, memory = ?, last_access = ? WHERE session_id = ?",
                (json.dumps(history[-MAX_HISTORY_MESSAGES:], ensure_ascii=False), json.dumps(pending_action, ensure_ascii=False) if pending_action else None, json.dumps(memory, ensure_ascii=False), time.time(), session_id),
            )
            self._db.commit()

    def close(self) -> None:
        """Release the SQLite connection during service shutdown and tests."""

        with self._db_lock:
            self._db.close()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parts(text: str) -> list[str]:
    return [part for part in re.split(r"([，。！？\n])", text) if part.strip()]


def create_app() -> FastAPI:
    client = OllamaChatClient(OLLAMA_BASE_URL, OLLAMA_MODEL)
    graph = build_graph(client)
    sessions = SessionStore()

    app = FastAPI(
        title="共享健身 Multi-Agent API",
        description="基于 LangGraph、Ollama 开源模型和 FastAPI 的共享健身服务原型",
        version="1.2.0",
    )
    app.state.graph = graph
    app.state.sessions = sessions
    app.state.ollama_client = client

    @app.on_event("shutdown")
    def close_sessions() -> None:
        sessions.close()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type", "Authorization"],
    )

    def authenticated_user(user_id: str, authorization: str | None) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要 Bearer 登录令牌。")
        subject = verify_token(authorization[7:].strip(), AUTH_SECRET)
        if not subject or subject != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="身份与请求用户不匹配。")
        return subject

    def load_session(request: ChatRequest, session_id: str) -> tuple[list[dict[str, str]], dict[str, Any] | None, dict[str, Any]]:
        try:
            return sessions.load(session_id, request.user_id)
        except SessionAccessError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该会话。") from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        try:
            await run_in_threadpool(client.check_ready)
        except HTTPException:
            raise
        except OllamaConnectionError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return {
            "status": "healthy",
            "service": "shared-fitness-multi-agent",
            "model": OLLAMA_MODEL,
            "environment": APP_ENV,
            "device_data_mode": DEVICE_DATA_MODE,
        }

    @app.get("/agents")
    async def agents() -> dict[str, Any]:
        return {
            "orchestrator": "router_agent",
            "agents": [
                {"name": "task_planner_agent", "responsibility": "将请求拆分为并行任务"},
                {"name": "context_agent", "responsibility": "读取确认档案与会话共享记忆"},
                {"name": "risk_agent", "responsibility": "安全风险分级与硬约束"},
                {"name": "domain_agent", "responsibility": "执行场馆、预约、设备或运维领域任务"},
                {"name": "conflict_resolution_agent", "responsibility": "合并并行结果并解决约束冲突"},
                {"name": "human_handoff_agent", "responsibility": "创建人工接管任务"},
                {"name": "response_agent", "responsibility": "统一回复与事实一致性"},
            ],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, authorization: str | None = Header(default=None)) -> ChatResponse:
        session_id = request.session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        authenticated_user(request.user_id, authorization)
        logger.info("request_id=%s user_id=%s session_id=%s mode=chat", request_id, request.user_id, session_id)
        try:
            with sessions.claim(session_id, request.user_id):
                history, pending_action, memory = load_session(request, session_id)
                state = make_state(request.message, request.user_id, session_id, history, pending_action, memory)
                result = await run_in_threadpool(graph.invoke, state)
                data = extract_result(result)
                memory_patch = {"last_route": data.get("route", ""), "last_risk_level": data.get("risk_level", "low"), "last_request_id": request_id, "updated_at": time.time()}
                sessions.save_result(session_id, request.user_id, request.message, data["response"], data.get("pending_action"), memory_patch)
                data["memory"] = {**memory, **memory_patch}
        except HTTPException:
            raise
        except OllamaConnectionError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("request_id=%s graph_failed", request_id)
            raise HTTPException(status_code=500, detail="Agent 处理失败，请稍后重试。") from exc

        return ChatResponse(request_id=request_id, session_id=session_id, **data)

    @app.post("/auth/demo-token")
    async def demo_token(user_id: str = "demo-user", x_demo_login_key: str | None = Header(default=None)) -> dict[str, str]:
        if x_demo_login_key != DEMO_LOGIN_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Demo 登录密钥不正确。")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", user_id):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="用户标识格式不正确。")
        return {"access_token": issue_token(user_id, AUTH_SECRET), "token_type": "bearer", "user_id": user_id}

    @app.get("/profiles/{user_id}")
    async def profile(user_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """查看演示档案；真实部署需接入身份、授权和最小化数据访问。"""

        authenticated_user(user_id, authorization)
        raw = TOOLS["get_user_profile"].invoke({"user_id": user_id})
        return json.loads(raw)

    @app.post("/resources/recommend")
    async def recommend_resources(request: ResourceRecommendRequest) -> dict[str, Any]:
        items = resource_recommendation_data(request.query, request.activity_type, request.max_distance_km)
        return {
            "items": items,
            "count": len(items),
            "source": "社区资源演示目录",
            "note": "推荐结果需以场馆实时容量、开放规则和用户确认权限为准。",
        }

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
        authenticated_user(request.user_id, authorization)
        result = TOOLS["submit_feedback"].invoke(request.model_dump())
        return {"status": "recorded", "result": result}

    @app.put("/permissions")
    async def permissions(request: PermissionRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
        authenticated_user(request.user_id, authorization)
        result = TOOLS["update_user_permission"].invoke(request.model_dump())
        return {"status": "processed", "result": result}

    @app.post("/check-in")
    async def check_in(request: CheckInRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
        authenticated_user(request.user_id, authorization)
        result = TOOLS["record_check_in"].invoke(request.model_dump())
        return {"status": "processed", "result": result}

    def stream_events(request: ChatRequest, session_id: str, request_id: str) -> Iterator[str]:
        with sessions.claim(session_id, request.user_id):
            history, pending_action, memory = sessions.load(session_id, request.user_id)
            state = make_state(request.message, request.user_id, session_id, history, pending_action, memory)
            yield _sse("start", {"request_id": request_id, "session_id": session_id})
            try:
                result: dict[str, Any] = dict(state)
                emitted_tools: set[str] = set()
                for update in graph.stream(state, stream_mode="updates"):
                    for node_name, node_update in update.items():
                        result.update(node_update)
                        trace = node_update.get("agent_trace", []) or node_update.get("parallel_trace", [])
                        if trace:
                            item = trace[-1]
                            yield _sse("agent", {"agent": item["agent"], "details": item["details"]})
                        for tool_result in node_update.get("tool_results", []) + node_update.get("parallel_tool_results", []):
                            tool_key = json.dumps(tool_result, ensure_ascii=False, sort_keys=True)
                            if tool_key in emitted_tools:
                                continue
                            emitted_tools.add(tool_key)
                            yield _sse("tool_result", {"tool_name": tool_result["tool_name"], "args": tool_result.get("args", {}), "result": tool_result["result"]})
                        if node_name == "response_agent":
                            for part in _parts(str(node_update.get("final_response", ""))):
                                yield _sse("token", {"content": part})

                data = extract_result(result)
                memory_patch = {"last_route": data.get("route", ""), "last_risk_level": data.get("risk_level", "low"), "last_request_id": request_id, "updated_at": time.time()}
                sessions.save_result(session_id, request.user_id, request.message, data["response"], data.get("pending_action"), memory_patch)
                data["memory"] = {**memory, **memory_patch}
                yield _sse("done", {"request_id": request_id, "session_id": session_id, **data})
            except OllamaConnectionError as exc:
                logger.exception("request_id=%s ollama_failed", request_id)
                yield _sse("error", {"request_id": request_id, "code": "ollama_unavailable", "message": str(exc)})
            except Exception as exc:
                logger.exception("request_id=%s stream_failed", request_id)
                yield _sse("error", {"request_id": request_id, "code": "agent_failed", "message": "Agent 处理失败，请稍后重试。"})

    @app.post("/stream")
    async def stream(request: ChatRequest, authorization: str | None = Header(default=None)) -> StreamingResponse:
        session_id = request.session_id or str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        authenticated_user(request.user_id, authorization)
        load_session(request, session_id)
        logger.info("request_id=%s user_id=%s session_id=%s mode=stream", request_id, request.user_id, session_id)
        return StreamingResponse(
            stream_events(request, session_id, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
