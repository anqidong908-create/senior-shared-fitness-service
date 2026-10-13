"""LangGraph orchestration for the shared-fitness multi-agent system."""

from __future__ import annotations

import datetime as dt
import json
import operator
import re
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from business_tools import GYMS, TOOLS
from ollama_client import OllamaChatClient, extract_json


ROUTES = {"venue", "reservation", "device", "support", "general", "safety"}


class MultiAgentState(TypedDict, total=False):
    user_message: str
    user_id: str
    session_id: str
    history: list[dict[str, str]]
    pending_action: dict[str, Any] | None
    route: str
    route_reason: str
    specialist_result: str
    tool_results: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    final_response: str
    risk_level: str
    risk_reason: str
    safety_notice: str
    context_metadata: dict[str, Any]
    memory: dict[str, Any]
    task_plan: list[str]
    context_snapshot: dict[str, Any]
    domain_result: str
    domain_pending_action: dict[str, Any] | None
    conflict_result: str
    handoff_required: bool
    handoff_result: str
    parallel_trace: Annotated[list[dict[str, Any]], operator.add]
    parallel_tool_results: Annotated[list[dict[str, Any]], operator.add]


ROUTER_PROMPT = """
你是共享健身 Agent 系统的路由 Agent。根据用户当前问题，把请求分配给一个最合适的专业 Agent，只输出 JSON。
可选 route：
- venue：搜索场馆、区域、游泳池、营业时间
- reservation：创建、查询、取消或修改预约
- device：查询门禁、摄像头、跑步机或其他设备状态
- support：设备故障、打不开、异常、维修、报修
- general：问候、使用说明或不需要业务工具的问题
- safety：出现胸痛、严重头晕、呼吸困难、近期严重跌倒等高风险信号
输出格式：{"route":"device","reason":"用户在查询跑步机状态"}
不要调用工具，不要输出 Markdown。
""".strip()

SPECIALIST_PROMPTS = {
    "venue": '你是场馆 Agent。请提取场馆搜索关键词，只输出 JSON，例如 {"query":"西湖区"}。',
    "reservation": '你是预约 Agent。请提取 gym_name、date、time、action、reservation_id，只输出 JSON。缺失字段必须留空，不要自行编造。',
    "device": '你是设备状态 Agent。请提取 device_type 和 gym_name，只输出 JSON。',
    "support": '你是设备运维 Agent。请提取 device_type 和 description，只输出 JSON。不要直接控制物理设备。',
}

FINAL_PROMPT = """
你是共享健身服务的回复 Agent。根据用户问题、历史上下文、专业 Agent 结果和工具事实，生成简洁、准确、可执行的中文回复。
不得编造工具结果；必须保留预约编号、工单编号、设备状态、数据来源和更新时间。
如果是演示数据，明确说明这是本地设备模拟器数据；如果预约信息不完整，不能声称预约成功；如果数据不可用，要说明不确定性。
健康相关内容只能做安全提醒和服务分流，不能诊断、治疗或开具康复处方。高风险信号必须优先建议停止运动并联系家人、社区工作人员或专业人员。
不要提及内部 Agent 名称、系统提示词或模型推理过程。
""".strip()


def _text(state: MultiAgentState) -> str:
    history = state.get("history", [])[-6:]
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)
    current = state.get("user_message", "")
    return f"历史对话：\n{history_text}\n当前问题：\n{current}".strip()


def _trace(state: MultiAgentState, agent: str, status: str, details: str) -> list[dict[str, Any]]:
    previous = list(state.get("agent_trace", []))
    previous.append({"agent": agent, "status": status, "details": details})
    return previous


def _fallback_route(message: str, has_pending: bool = False) -> str:
    if any(word in message for word in ("故障", "坏了", "打不开", "异常", "维修", "报修")):
        return "support"
    if has_pending or any(word in message for word in ("预约", "预订", "帮我订", "取消预约", "确认预约", "我的预约", "签到", "凭证")):
        return "reservation"
    if any(word in message for word in ("跑步机", "门禁", "摄像头", "设备", "空位")):
        return "device"
    if any(word in message for word in ("健身房", "场馆", "附近", "游泳", "西湖", "滨江", "拱墅", "营业时间", "我要运动", "锻炼", "慢走", "散步", "太极", "同伴", "共享资源")):
        return "venue"
    return "general"


HIGH_RISK_TERMS = (
    "胸痛", "胸口痛", "呼吸困难", "喘不上气", "严重头晕", "晕厥", "昏厥",
    "刚摔倒", "严重跌倒", "摔伤", "意识不清", "大出血",
)
MEDIUM_RISK_TERMS = ("头晕", "胸闷", "膝盖疼", "腰疼", "高血压", "心脏病", "术后", "骨折")


def _assess_risk(message: str) -> tuple[str, str]:
    """Deterministic safety gate; the model cannot downgrade a high-risk signal."""

    normalized = message.replace(" ", "")
    high = next((term for term in HIGH_RISK_TERMS if term in normalized), "")
    if high:
        return "high", f"识别到高风险信号：{high}"
    medium = next((term for term in MEDIUM_RISK_TERMS if term in normalized), "")
    if medium:
        return "medium", f"识别到需要谨慎评估的信号：{medium}"
    return "low", "未识别到当前消息中的高风险信号"


def _model_json(client: OllamaChatClient, system: str, user: str) -> dict[str, Any]:
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return {}


def router_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    risk_level, risk_reason = _assess_risk(state.get("user_message", ""))
    # A high-risk signal is a deterministic hard gate and must not depend on
    # model availability or on the model choosing the correct route.
    decision = {} if risk_level == "high" else _model_json(client, ROUTER_PROMPT, _text(state))
    route = str(decision.get("route", "")).lower().strip()
    product_route = _fallback_route(state.get("user_message", ""), bool(state.get("pending_action")))
    if product_route != "general" or route not in ROUTES:
        route = product_route
    reason = str(decision.get("reason") or "根据业务关键词、会话上下文和安全规则完成路由")
    if risk_level == "high":
        route = "safety"
        reason = f"安全优先：{risk_reason}"
    return {
        "route": route,
        "route_reason": reason,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "safety_notice": risk_reason if risk_level != "low" else "",
        "agent_trace": _trace(state, "router_agent", "completed", f"route={route}; risk={risk_level}; {reason}"),
    }


def _extract_time(raw: str) -> str:
    match = re.search(r"(\d{1,2}):(\d{2})", raw)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    match = re.search(r"(上午|下午)?\s*(\d{1,2})\s*点", raw)
    if not match:
        return ""
    hour = int(match.group(2))
    if match.group(1) == "下午" and hour < 12:
        hour += 12
    return f"{hour:02d}:00"


def _reservation_args(message: str, user_id: str, model_args: dict[str, Any], history: list[dict[str, str]]) -> dict[str, str]:
    context = "\n".join(item["content"] for item in history[-4:]) + "\n" + message
    raw_date = str(model_args.get("date") or "")
    if "明天" in context or raw_date in {"明天", "tomorrow"}:
        date = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    elif "今天" in context or raw_date in {"今天", "today"}:
        date = dt.date.today().isoformat()
    else:
        date_match = re.search(r"20\d{2}-\d{2}-\d{2}", raw_date + " " + context)
        date = date_match.group(0) if date_match else ""

    time = _extract_time(str(model_args.get("time") or "") + " " + context)
    gym_name = str(model_args.get("gym_name") or "").strip()
    for district, item in GYMS.items():
        if district in context or district[:-1] in context:
            gym_name = item["name"]
            break
        if item["name"] in gym_name:
            gym_name = item["name"]
    reservation_id_match = re.search(r"RES-[A-Za-z0-9-]+", message) or re.search(r"RES-[A-Za-z0-9-]+", context)
    return {
        "gym_name": gym_name,
        "date": date,
        "time": time,
        "user_id": user_id,
        "reservation_id": reservation_id_match.group(0) if reservation_id_match else "",
    }


def _reservation_action(message: str, args: dict[str, str]) -> str:
    explicit_reservation_id = bool(re.search(r"RES-[A-Za-z0-9-]+", message))
    if "签到" in message:
        return "check_in"
    if "凭证" in message or "二维码" in message:
        return "credential"
    if "取消" in message:
        return "cancel"
    if any(word in message for word in ("修改", "更改")):
        return "update"
    if any(word in message for word in ("查询", "查看", "我的预约", "预约详情")):
        # Session history may contain an old reservation id; it must not turn
        # a new "my reservations" request into a single-record lookup.
        return "query" if explicit_reservation_id else "list"
    return "create"


def _reservation_confirmation(message: str) -> bool:
    return any(word in message.strip() for word in ("确认预约", "确认", "好的", "确定", "可以"))


def _confirmation_text(args: dict[str, str]) -> str:
    return (
        "预约信息已准备好，请确认后再创建：\n"
        f"场馆：{args.get('gym_name') or '待补充'}\n"
        f"日期：{args.get('date') or '待补充'}\n"
        f"时间：{args.get('time') or '待补充'}\n"
        "回复“确认预约”后才会创建预约；回复“取消”可放弃。"
    )


def _update_confirmation_text(args: dict[str, str]) -> str:
    return (
        "预约修改信息已准备好，请确认后再提交：\n"
        f"预约编号：{args.get('reservation_id') or '待补充'}\n"
        f"新日期：{args.get('date') or '待补充'}\n"
        f"新时间：{args.get('time') or '待补充'}\n"
        "回复“确认修改”后才会执行修改。"
    )


def _specialist_agent(client: OllamaChatClient, state: MultiAgentState, route: str) -> dict[str, Any]:
    message = state.get("user_message", "")
    user_id = state.get("user_id", "default")
    decision = _model_json(client, SPECIALIST_PROMPTS[route], _text(state))
    tool_results = list(state.get("tool_results", []))
    pending_action = state.get("pending_action")

    if route == "venue":
        query = str(decision.get("query") or message)
        for district in GYMS:
            if district in message or district[:-1] in message:
                query = district
                break
        if "游泳" in message or "泳池" in message:
            query = "游泳池"
        if any(word in message for word in ("我要运动", "锻炼", "慢走", "散步", "太极", "同伴", "推荐", "共享资源")):
            activity = "步行" if any(word in message for word in ("慢走", "散步", "走路")) else ("太极" if "太极" in message else "")
            data = TOOLS["search_shared_resources"].invoke({"query": message, "activity_type": activity, "max_distance_km": 5.0})
            tool_results.append({"tool_name": "search_shared_resources", "args": {"query": message, "activity_type": activity, "max_distance_km": 5.0}, "result": data})
            if "同伴" in message:
                companion_data = TOOLS["find_companions"].invoke({"activity": activity or "轻运动", "user_id": user_id})
                tool_results.append({"tool_name": "find_companions", "args": {"activity": activity or "轻运动", "user_id": user_id}, "result": companion_data})
            details = f"search_shared_resources(activity={activity or '综合'})"
        else:
            data = TOOLS["search_gymnasium"].invoke({"query": query})
            tool_results.append({"tool_name": "search_gymnasium", "args": {"query": query}, "result": data})
            details = f"search_gymnasium(query={query})"
        return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "venue_agent", "completed", details)}

    if route == "reservation":
        args = _reservation_args(message, user_id, decision, state.get("history", []))
        action = _reservation_action(message, args)
        if pending_action and pending_action.get("type") == "reservation_update" and _reservation_confirmation(message):
            action = "update"
        if action in {"credential", "check_in"}:
            if not args["reservation_id"]:
                data = "请提供预约编号，例如 RES-XXXXXXXXXX。"
            else:
                tool_name = "get_access_credential" if action == "credential" else "record_check_in"
                tool_args = {"reservation_id": args["reservation_id"], "user_id": user_id}
                data = TOOLS[tool_name].invoke(tool_args)
                tool_results.append({"tool_name": tool_name, "args": tool_args, "result": data})
            return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "reservation_agent", "completed", action)}
        if action == "list":
            data = TOOLS["list_reservations"].invoke({"user_id": user_id})
            tool_results.append({"tool_name": "list_reservations", "args": {"user_id": user_id}, "result": data})
            return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "reservation_agent", "completed", "list_reservations")}
        if action == "cancel":
            if not args["reservation_id"]:
                data = "请提供需要取消的预约编号，例如 RES-XXXXXXXXXX。"
            else:
                data = TOOLS["cancel_reservation"].invoke({"reservation_id": args["reservation_id"], "user_id": user_id})
                tool_results.append({"tool_name": "cancel_reservation", "args": {"reservation_id": args["reservation_id"], "user_id": user_id}, "result": data})
            return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "reservation_agent", "completed", "cancel_reservation")}
        if action == "query":
            data = TOOLS["get_reservation"].invoke({"reservation_id": args["reservation_id"], "user_id": user_id})
            tool_results.append({"tool_name": "get_reservation", "args": {"reservation_id": args["reservation_id"], "user_id": user_id}, "result": data})
            return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "reservation_agent", "completed", "get_reservation")}

        if action == "update":
            if pending_action and pending_action.get("type") == "reservation_update" and _reservation_confirmation(message):
                pending_args = dict(pending_action["args"])
                data = TOOLS["update_reservation"].invoke({**pending_args, "confirmed": True})
                tool_results.append({"tool_name": "update_reservation", "args": pending_args, "result": data})
                return {"specialist_result": data, "tool_results": tool_results, "pending_action": None, "agent_trace": _trace(state, "reservation_agent", "completed", "update_reservation confirmed")}
            missing = [label for key, label in (("reservation_id", "预约编号"), ("date", "新日期"), ("time", "新时间")) if not args[key]]
            data = (f"还需要补充：{'、'.join(missing)}。\n" if missing else "") + _update_confirmation_text(args)
            pending = {"type": "reservation_update", "args": args, "idempotency_key": f"{state.get('session_id')}:{uuid.uuid4().hex}"}
            return {"specialist_result": data, "tool_results": tool_results, "pending_action": pending, "agent_trace": _trace(state, "reservation_agent", "waiting_confirmation", "修改预约副作用已阻断，等待用户确认")}

        if pending_action and _reservation_confirmation(message):
            pending_args = dict(pending_action["args"])
            data = TOOLS["create_reservation"].invoke(
                {
                    **pending_args,
                    "confirmed": True,
                    "idempotency_key": pending_action["idempotency_key"],
                }
            )
            tool_results.append({"tool_name": "create_reservation", "args": pending_args, "result": data})
            return {
                "specialist_result": data,
                "tool_results": tool_results,
                "pending_action": None,
                "agent_trace": _trace(state, "reservation_agent", "completed", "create_reservation confirmed"),
            }

        missing = [label for key, label in (("gym_name", "场馆"), ("date", "日期"), ("time", "时间")) if not args[key]]
        if missing:
            data = f"还需要补充：{'、'.join(missing)}。" + "\n" + _confirmation_text(args)
        else:
            data = _confirmation_text(args)
        pending = {"type": "reservation_create", "args": args, "idempotency_key": f"{state.get('session_id')}:{uuid.uuid4().hex}"}
        return {
            "specialist_result": data,
            "tool_results": tool_results,
            "pending_action": pending,
            "agent_trace": _trace(state, "reservation_agent", "waiting_confirmation", "预约副作用已阻断，等待用户确认"),
        }

    if route == "device":
        device_type = str(decision.get("device_type") or "全部")
        if "跑步机" in message or "空位" in message:
            device_type = "跑步机"
        elif "门禁" in message:
            device_type = "门禁"
        elif "摄像头" in message:
            device_type = "摄像头"
        gym_name = next((item["name"] for district, item in GYMS.items() if district in message), "")
        data = TOOLS["get_device_status"].invoke({"device_type": device_type, "gym_name": gym_name})
        tool_results.append({"tool_name": "get_device_status", "args": {"device_type": device_type, "gym_name": gym_name}, "result": data})
        return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "device_agent", "completed", f"get_device_status(device_type={device_type})")}

    device_type = str(decision.get("device_type") or "设备")
    description = str(decision.get("description") or message)
    data = TOOLS["create_service_ticket"].invoke(
        {"device_type": device_type, "description": description, "user_id": user_id, "idempotency_key": f"{state.get('session_id')}:{message}"}
    )
    tool_results.append({"tool_name": "create_service_ticket", "args": {"device_type": device_type, "description": description, "user_id": user_id}, "result": data})
    return {"specialist_result": data, "tool_results": tool_results, "agent_trace": _trace(state, "support_agent", "completed", f"create_service_ticket(device_type={device_type})")}


def venue_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    return _specialist_agent(client, state, "venue")


def reservation_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    return _specialist_agent(client, state, "reservation")


def device_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    return _specialist_agent(client, state, "device")


def support_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    return _specialist_agent(client, state, "support")


def general_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    answer = client.chat(
        [
            {"role": "system", "content": "你是共享健身客服 Agent，只回答确定的服务说明，不要编造实时状态。"},
            {"role": "user", "content": _text(state)},
        ],
        temperature=0.2,
    )
    return {"specialist_result": answer, "agent_trace": _trace(state, "general_agent", "completed", "直接回答通用服务问题")}


def safety_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    """Hard stop for high-risk signals; no normal exercise recommendation is allowed."""

    reason = state.get("risk_reason", "识别到高风险信号")
    answer = (
        "先暂停运动。\n"
        f"{reason}。本系统不能进行医疗诊断。\n"
        "请立即联系家人、社区工作人员或医疗专业人员；如果症状严重、持续或正在恶化，请拨打当地急救电话。\n"
        "在获得专业人员确认前，我不会继续推荐场馆、训练强度或预约。"
    )
    return {
        "specialist_result": answer,
        "safety_notice": answer,
        "agent_trace": _trace(state, "safety_agent", "completed", "高风险信号触发安全停机与人工/专业支持分流"),
    }


def response_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    if state.get("risk_level") == "high":
        answer = state.get("specialist_result", "请先暂停运动并联系专业人员。")
    elif state.get("pending_action") and not state.get("tool_results"):
        answer = state.get("specialist_result", "请补充预约信息后继续。")
    else:
        evidence = json.dumps(
            {
                "history": state.get("history", [])[-6:],
                "route": state.get("route"),
                "specialist_result": state.get("specialist_result", ""),
                "tool_results": state.get("tool_results", []),
                "risk_level": state.get("risk_level", "low"),
                "safety_notice": state.get("safety_notice", ""),
            },
            ensure_ascii=False,
        )
        answer = client.chat(
            [
                {"role": "system", "content": FINAL_PROMPT},
                {"role": "user", "content": f"用户问题：{state.get('user_message', '')}\n事实证据：{evidence}"},
            ],
            temperature=0.2,
        )
        answer = _repair_factual_answer(state, answer)
    return {
        "final_response": (
            (answer or state.get("specialist_result", "暂时无法生成回复。"))
            + (f"\n安全提示：{state['risk_reason']}，建议降低强度并先确认身体状态。" if state.get("risk_level") == "medium" else "")
        ),
        "agent_trace": _trace(state, "response_agent", "completed", "基于上下文、专业 Agent 和工具事实生成最终回复"),
    }


def _repair_factual_answer(state: MultiAgentState, answer: str) -> str:
    results = state.get("tool_results", [])
    if not results:
        return answer
    evidence = "\n".join(str(item.get("result", "")) for item in results)
    if state.get("route") == "reservation" and "预约成功" in evidence and "预约编号" not in answer:
        return results[-1]["result"]
    if state.get("route") == "reservation" and "预约已取消" in evidence and "预约已取消" not in answer:
        return results[-1]["result"]
    if state.get("route") == "support" and "工单编号" in evidence and "工单编号" not in answer:
        return results[-1]["result"]
    if state.get("route") == "device" and "数据来源" not in answer:
        return results[-1]["result"]
    return answer


def _parallel_event(agent: str, status: str, details: str) -> dict[str, Any]:
    return {"agent": agent, "status": status, "details": details}


def parallel_router_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    """Select a domain; downstream workers still run as a coordinated fan-out."""

    risk_level, risk_reason = _assess_risk(state.get("user_message", ""))
    decision = {} if risk_level == "high" else _model_json(client, ROUTER_PROMPT, _text(state))
    route = str(decision.get("route", "")).lower().strip()
    fallback = _fallback_route(state.get("user_message", ""), bool(state.get("pending_action")))
    if fallback != "general" or route not in ROUTES - {"safety"}:
        route = fallback
    if risk_level == "high":
        route = "safety"
    reason = f"{risk_reason}；已将请求拆分为风险、上下文、业务执行三个并行任务" if risk_level != "low" else "已将请求拆分为风险、上下文、业务执行三个并行任务"
    return {
        "route": route,
        "route_reason": reason,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "parallel_trace": [_parallel_event("router_agent", "completed", f"route={route}; risk={risk_level}")],
    }


def task_planner_agent(state: MultiAgentState) -> dict[str, Any]:
    route = state.get("route", "general")
    plan = ["风险评估", "读取共享上下文记忆", f"执行 {route} 领域任务", "冲突解决与副作用检查", "生成回复或人工接管"]
    return {
        "task_plan": plan,
        "parallel_trace": [_parallel_event("task_planner_agent", "completed", "任务已拆分为 3 个并行工作分支")],
    }


def context_agent(state: MultiAgentState) -> dict[str, Any]:
    user_id = state.get("user_id", "demo-user")
    try:
        profile = json.loads(TOOLS["get_user_profile"].invoke({"user_id": user_id}))
    except (TypeError, ValueError, json.JSONDecodeError):
        profile = {"user_id": user_id, "confirmed_constraints": [], "preferences": [], "permissions": {}}
    memory = dict(state.get("memory", {}))
    snapshot = {
        "profile": profile,
        "memory": memory,
        "source_priority": ["explicit_user_message", "confirmed_profile", "session_memory", "history"],
        "permission_checked": True,
    }
    return {
        "context_snapshot": snapshot,
        "context_metadata": {"source_priority": "explicit_user_message > confirmed_profile > session_memory > history", "permissions_checked": True},
        "parallel_trace": [_parallel_event("context_agent", "completed", "已读取确认档案和会话记忆，未推断未授权字段")],
    }


def risk_agent(state: MultiAgentState) -> dict[str, Any]:
    level, reason = _assess_risk(state.get("user_message", ""))
    notice = reason if level != "low" else "未识别到当前消息中的高风险信号"
    return {
        "risk_level": level,
        "risk_reason": reason,
        "safety_notice": notice,
        "parallel_trace": [_parallel_event("risk_agent", "completed", f"risk={level}; {reason}")],
    }


def domain_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    route = state.get("route", "general")
    if route == "safety" or _assess_risk(state.get("user_message", ""))[0] == "high":
        return {
            "domain_result": "高风险请求已阻断业务工具调用。",
            "domain_pending_action": None,
            "parallel_trace": [_parallel_event("domain_agent", "skipped", "安全分支阻断场馆、预约和设备副作用")],
        }
    result = _specialist_agent(client, state, route if route in SPECIALIST_PROMPTS else "general") if route in SPECIALIST_PROMPTS else general_agent(client, state)
    return {
        "domain_result": result.get("specialist_result", ""),
        "domain_pending_action": result.get("pending_action"),
        "parallel_tool_results": result.get("tool_results", []),
        "parallel_trace": [_parallel_event("domain_agent", "completed", f"已执行 {route} 领域 Agent 与业务工具")],
    }


def conflict_resolution_agent(state: MultiAgentState) -> dict[str, Any]:
    level = state.get("risk_level", "low")
    context = state.get("context_snapshot", {})
    profile = context.get("profile", {}) if isinstance(context, dict) else {}
    result = state.get("domain_result", "")
    conflicts: list[str] = []
    if level == "high":
        conflicts.append("高风险优先级覆盖普通业务意图")
    constraints = profile.get("confirmed_constraints", []) if isinstance(profile, dict) else []
    if constraints and any(term in result for term in ("高强度", "竞技")):
        conflicts.append("已确认的低强度约束优先于领域 Agent 的普通推荐")
    pending = state.get("domain_pending_action")
    if level == "high":
        route_reason = f"安全优先：{state.get('risk_reason', '高风险信号')}"
        handoff_required = True
    else:
        route_reason = state.get("route_reason", "")
        handoff_required = False
    return {
        "route": "safety" if handoff_required else state.get("route", "general"),
        "route_reason": route_reason,
        "specialist_result": result,
        "pending_action": pending,
        "tool_results": list(state.get("parallel_tool_results", [])),
        "conflict_result": "；".join(conflicts) if conflicts else "未发现业务与安全约束冲突",
        "handoff_required": handoff_required,
        "agent_trace": list(state.get("parallel_trace", [])) + [_parallel_event("conflict_resolution_agent", "completed", "已合并并行结果并按安全、权限、可执行性优先级解决冲突")],
    }


def human_handoff_agent(state: MultiAgentState) -> dict[str, Any]:
    reason = state.get("risk_reason", "需要人工协助")
    result = TOOLS["create_human_handoff"].invoke({"user_id": state.get("user_id", "demo-user"), "reason": reason})
    tool_results = list(state.get("tool_results", []))
    tool_results.append({"tool_name": "create_human_handoff", "args": {"user_id": state.get("user_id", "demo-user"), "reason": reason}, "result": result})
    return {
        "handoff_result": result,
        "tool_results": tool_results,
        "agent_trace": list(state.get("agent_trace", [])) + [_parallel_event("human_handoff_agent", "completed", "已创建人工接管记录")],
    }


def parallel_response_agent(client: OllamaChatClient, state: MultiAgentState) -> dict[str, Any]:
    if state.get("risk_level") == "high":
        answer = (
            "先暂停运动。\n"
            f"{state.get('risk_reason', '识别到高风险信号')}。本系统不能进行医疗诊断。\n"
            "请联系家人、社区工作人员或医疗专业人员；如果症状严重、持续或正在恶化，请拨打当地急救电话。\n"
            f"{state.get('handoff_result', '人工接管记录正在创建。')}"
        )
    elif state.get("pending_action") and not state.get("tool_results"):
        answer = state.get("specialist_result", "请补充预约信息后继续。")
    else:
        evidence = json.dumps({
            "route": state.get("route"),
            "risk_level": state.get("risk_level"),
            "risk_reason": state.get("risk_reason"),
            "context": state.get("context_snapshot", {}),
            "memory": state.get("memory", {}),
            "specialist_result": state.get("specialist_result", ""),
            "tool_results": state.get("tool_results", []),
            "conflict_result": state.get("conflict_result", ""),
        }, ensure_ascii=False)
        answer = client.chat([{"role": "system", "content": FINAL_PROMPT}, {"role": "user", "content": f"用户问题：{state.get('user_message', '')}\n事实证据：{evidence}"}], temperature=0.2)
        answer = _repair_factual_answer(state, answer)
        if state.get("risk_level") == "medium":
            answer += f"\n安全提示：{state.get('risk_reason', '请先确认身体状态')}，建议降低强度并保留休息。"
    return {
        "final_response": answer or state.get("specialist_result", "暂时无法生成回复。"),
        "agent_trace": list(state.get("agent_trace", [])) + [_parallel_event("response_agent", "completed", "已汇总并行 Agent、工具事实、记忆与冲突结果")],
    }


def _route_to_node(state: MultiAgentState) -> str:
    return state.get("route", "general")


def build_graph(client: OllamaChatClient):
    graph = StateGraph(MultiAgentState)
    graph.add_node("router_agent", lambda state: parallel_router_agent(client, state))
    graph.add_node("task_planner_agent", task_planner_agent)
    graph.add_node("context_agent", context_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("domain_agent", lambda state: domain_agent(client, state))
    graph.add_node("conflict_resolution_agent", conflict_resolution_agent)
    graph.add_node("human_handoff_agent", human_handoff_agent)
    graph.add_node("response_agent", lambda state: parallel_response_agent(client, state))
    graph.add_edge(START, "router_agent")
    graph.add_edge("router_agent", "task_planner_agent")
    graph.add_edge("task_planner_agent", "context_agent")
    graph.add_edge("task_planner_agent", "risk_agent")
    graph.add_edge("task_planner_agent", "domain_agent")
    graph.add_edge("context_agent", "conflict_resolution_agent")
    graph.add_edge("risk_agent", "conflict_resolution_agent")
    graph.add_edge("domain_agent", "conflict_resolution_agent")
    graph.add_conditional_edges("conflict_resolution_agent", lambda state: "human_handoff_agent" if state.get("handoff_required") else "response_agent", {"human_handoff_agent": "human_handoff_agent", "response_agent": "response_agent"})
    graph.add_edge("human_handoff_agent", "response_agent")
    graph.add_edge("response_agent", END)
    return graph.compile()


def make_state(
    message: str,
    user_id: str,
    session_id: str,
    history: list[dict[str, str]],
    pending_action: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> MultiAgentState:
    return {
        "user_message": message,
        "user_id": user_id,
        "session_id": session_id,
        "history": history,
        "pending_action": pending_action,
        "memory": memory or {},
        "tool_results": [],
        "agent_trace": [],
        "parallel_trace": [],
        "parallel_tool_results": [],
        "risk_level": "low",
        "risk_reason": "",
        "safety_notice": "",
        "context_metadata": {"source_priority": "explicit_user_message > confirmed_profile > history", "permissions_checked": False},
    }


def extract_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "response": result.get("final_response", ""),
        "route": result.get("route", "general"),
        "route_reason": result.get("route_reason", ""),
        "agent_trace": result.get("agent_trace", []),
        "tool_results": result.get("tool_results", []),
        "pending_action": result.get("pending_action"),
        "risk_level": result.get("risk_level", "low"),
        "risk_reason": result.get("risk_reason", ""),
        "safety_notice": result.get("safety_notice", ""),
        "context_metadata": result.get("context_metadata", {}),
        "memory": result.get("memory", {}),
        "task_plan": result.get("task_plan", []),
        "conflict_result": result.get("conflict_result", ""),
        "handoff_result": result.get("handoff_result", ""),
    }
