"""Business tools for the shared-fitness demo.

The default implementation is an in-memory simulator. Each result declares
its source so the UI and portfolio documentation cannot present fixture data
as live hardware data.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import uuid

from langchain_core.tools import tool

from config import DEVICE_DATA_MODE


GYMS: dict[str, dict[str, str]] = {
    "西湖区": {
        "gym_id": "gym-xihu",
        "name": "西湖区健身中心",
        "address": "西湖区文一路123号",
        "facilities": "游泳池、瑜伽室、力量区",
        "hours": "06:00-22:00",
        "capacity": "12/50",
    },
    "滨江区": {
        "gym_id": "gym-binjiang",
        "name": "滨江体育中心",
        "address": "滨江区江南大道56号",
        "facilities": "篮球馆、羽毛球馆、跑步机区",
        "hours": "07:00-21:00",
        "capacity": "8/40",
    },
    "拱墅区": {
        "gym_id": "gym-gongshu",
        "name": "拱墅区全民健身中心",
        "address": "拱墅区莫干山路89号",
        "facilities": "恒温泳池、体操房、器械区",
        "hours": "06:30-22:30",
        "capacity": "20/60",
    },
}

# These records make the PRD workflow executable without pretending that a
# real community platform or hardware gateway is connected.
RESOURCE_CATALOG = [
    {
        "resource_id": "res-xihu-walk",
        "gym_id": "gym-xihu",
        "name": "西湖区健身中心·长者步行区",
        "activity": "慢走、拉伸",
        "distance_km": 0.8,
        "available_time": "今日 16:00-18:00",
        "elderly_facilities": "无障碍入口、休息座椅、饮水点",
        "capacity": "12/50",
        "companion_needed": "可独行，也可匹配同伴",
        "risk_note": "建议低强度开始，途中可随时休息",
    },
    {
        "resource_id": "res-binjiang-bike",
        "gym_id": "gym-binjiang",
        "name": "滨江体育中心·器械区",
        "activity": "室内单车、跑步机",
        "distance_km": 2.4,
        "available_time": "今日 18:30-20:00",
        "elderly_facilities": "电梯、工作人员值守、应急呼叫点",
        "capacity": "8/40",
        "companion_needed": "建议首次使用时有人陪同",
        "risk_note": "不建议直接进行高强度训练",
    },
    {
        "resource_id": "res-gongshu-chair",
        "gym_id": "gym-gongshu",
        "name": "拱墅区全民健身中心·轻运动房",
        "activity": "坐姿操、太极、轻器械",
        "distance_km": 1.6,
        "available_time": "明日 09:00-10:30",
        "elderly_facilities": "防滑地面、座椅、志愿者引导",
        "capacity": "20/60",
        "companion_needed": "可参加社区小组活动",
        "risk_note": "适合需要分段休息的人群",
    },
]

USER_PROFILES = {
    "demo-user": {
        "user_id": "demo-user",
        "community": "西湖区翠苑社区",
        "confirmed_constraints": ["偏好低强度", "膝部不适时需要休息"],
        "preferences": ["优先附近", "熟悉场馆", "可匹配同伴"],
        "digital_ability": "普通",
        "language": "普通话/可使用短句",
        "permissions": {"location": False, "health": True, "family_assistance": False},
        "source": "演示用户主动确认的档案字段",
    }
}
_feedback_events: list[dict[str, str]] = []
_checkins: dict[str, dict[str, str]] = {}

DEVICE_STATES = {
    "门禁": "门禁：正常运行\n位置：主入口\n状态：在线\n今日通行次数：156次",
    "摄像头": "摄像头：正常运行\n位置：大厅、走廊、器械区\n状态：在线\n存储剩余空间：85%",
    "跑步机": "跑步机：部分使用中\n总数：10台\n使用中：6台\n空闲：4台\n状态：全部在线",
    "全部": "设备总览\n门禁：在线\n摄像头：在线\n跑步机：6/10使用中\n空调系统：正常",
}

_lock = threading.Lock()
_reservations: dict[str, dict[str, str]] = {}
_reservation_by_key: dict[str, str] = {}
_tickets: dict[str, dict[str, str]] = {}
_ticket_by_key: dict[str, str] = {}


def _source_line() -> str:
    if DEVICE_DATA_MODE == "simulator":
        return "数据来源：本地设备模拟器（演示数据，非真实硬件）"
    return f"数据来源：{DEVICE_DATA_MODE}适配器"


def _find_gym(gym_name: str) -> dict[str, str] | None:
    value = gym_name.strip()
    for district, item in GYMS.items():
        if value in {district, district[:-1], item["name"], item["gym_id"]}:
            return item
    return None


def _gym_text(item: dict[str, str]) -> str:
    return (
        f"场馆：{item['name']}\n"
        f"场馆 ID：{item['gym_id']}\n"
        f"地址：{item['address']}\n"
        f"设施：{item['facilities']}\n"
        f"营业时间：{item['hours']}\n"
        f"当前人数：{item['capacity']}\n"
        f"数据来源：场馆演示数据"
    )


def _resource_text(item: dict[str, object], rank: int) -> str:
    return (
        f"推荐 {rank}：{item['name']}\n"
        f"资源 ID：{item['resource_id']}\n"
        f"活动：{item['activity']}\n"
        f"距离：{item['distance_km']} km\n"
        f"可用时段：{item['available_time']}\n"
        f"适老设施：{item['elderly_facilities']}\n"
        f"当前容量：{item['capacity']}\n"
        f"同伴/志愿者：{item['companion_needed']}\n"
        f"匹配理由：附近、可休息、与需求强度相符\n"
        f"风险提示：{item['risk_note']}\n"
        "状态来源：社区资源演示目录"
    )


def resource_recommendation_data(
    query: str = "",
    activity_type: str = "",
    max_distance_km: float = 5.0,
) -> list[dict[str, object]]:
    """Return hard-filtered shared resources for API and agent reuse."""

    text = f"{query} {activity_type}".lower()
    result = [dict(item) for item in RESOURCE_CATALOG if float(item["distance_km"]) <= max_distance_km]
    activity_words = {
        "步行": ("慢走", "太极"),
        "散步": ("慢走", "太极"),
        "跑步": ("跑步机", "室内单车"),
        "器械": ("器械", "室内单车"),
        "太极": ("太极",),
        "拉伸": ("拉伸",),
    }
    wanted = next((words for key, words in activity_words.items() if key in text), ())
    if wanted:
        filtered = [item for item in result if any(word in str(item["activity"]) for word in wanted)]
        result = filtered
    observed_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    for item in result:
        item["availability_status"] = "demo_available"
        item["source"] = "社区资源演示目录"
        item["observed_at"] = observed_at
        item["hard_filters"] = ["距离范围", "活动类型"]
    return sorted(result, key=lambda item: float(item["distance_km"]))[:3]


@tool
def search_shared_resources(query: str = "", activity_type: str = "", max_distance_km: float = 5.0) -> str:
    """搜索适老共享运动资源，返回 2-3 个带理由和风险提示的候选。"""

    items = resource_recommendation_data(query, activity_type, max_distance_km)
    if not items:
        return "未找到满足距离和活动条件的共享资源。建议扩大距离或改选低强度活动。"
    return "\n\n".join(_resource_text(item, index) for index, item in enumerate(items, 1))


@tool
def get_user_profile(user_id: str = "demo-user") -> str:
    """读取演示用户已确认的偏好与权限，不推断未授权的健康或定位信息。"""

    profile = dict(USER_PROFILES.get(user_id, {
        "user_id": user_id,
        "community": "未设置",
        "confirmed_constraints": [],
        "preferences": [],
        "digital_ability": "未设置",
        "language": "普通话",
        "permissions": {"location": False, "health": False, "family_assistance": False},
        "source": "未建立档案",
    }))
    return json.dumps(profile, ensure_ascii=False)


@tool
def update_user_permission(user_id: str = "demo-user", permission: str = "family_assistance", enabled: bool = False) -> str:
    """显式更新演示权限；权限不能由对话内容或历史行为推断。"""

    allowed = {"location", "health", "family_assistance", "emergency_notification"}
    if permission not in allowed:
        return f"权限更新失败：仅支持 {', '.join(sorted(allowed))}。"
    with _lock:
        profile = USER_PROFILES.setdefault(user_id, {
            "user_id": user_id,
            "community": "未设置",
            "confirmed_constraints": [],
            "preferences": [],
            "digital_ability": "未设置",
            "language": "普通话",
            "permissions": {},
            "source": "用户显式授权",
        })
        profile["permissions"][permission] = bool(enabled)
    return f"权限已{'开启' if enabled else '关闭'}：{permission}\n提示：家庭协助不等于家庭控制，用户可以随时撤销。"


@tool
def find_companions(activity: str = "轻运动", date: str = "", time: str = "", user_id: str = "demo-user") -> str:
    """返回经过脱敏的演示同伴候选，真实产品应接入授权和社区匹配服务。"""

    return (
        "同伴候选（演示数据，需双方授权后才能看到联系方式）\n"
        "1. 王阿姨｜翠苑社区｜偏好：慢走｜可约：工作日 16:00 后\n"
        "2. 李叔叔｜翠苑社区｜偏好：太极｜可约：周末上午\n"
        f"匹配活动：{activity}｜日期：{date or '待确认'}｜时间：{time or '待确认'}\n"
        "隐私提示：当前只展示脱敏标签，不能把 AI 推荐当作对方已同意。"
    )


@tool
def get_access_credential(reservation_id: str, user_id: str = "demo-user") -> str:
    """生成预约凭证演示信息，不执行开门或控制真实设备。"""

    with _lock:
        item = _reservations.get(reservation_id.strip())
    if not item or item["user_id"] != user_id or item["status"] != "confirmed":
        return "无法生成凭证：未找到当前用户的有效预约。"
    return (
        "到场凭证已生成（演示）\n"
        f"预约编号：{reservation_id}\n"
        f"场馆：{item['gym_name']}\n"
        f"日期：{item['date']}\n"
        f"时间：{item['time']}\n"
        f"到场方式：二维码 / 社区实体服务卡 / 人工核验\n"
        "有效性：以场馆实时预约系统为准\n"
        "设备边界：本系统不直接执行开门。"
    )


@tool
def record_check_in(reservation_id: str, user_id: str = "demo-user", method: str = "manual") -> str:
    """记录签到事件，供演示闭环使用；不代表真实门禁已打开。"""

    with _lock:
        item = _reservations.get(reservation_id.strip())
        if not item or item["user_id"] != user_id or item["status"] != "confirmed":
            return "签到失败：未找到当前用户的有效预约。"
        if reservation_id in _checkins:
            return f"签到已存在（演示）\n预约编号：{reservation_id}\n状态：已签到"
        event = {"reservation_id": reservation_id, "user_id": user_id, "method": method, "status": "checked_in", "checked_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        _checkins[reservation_id] = event
    return f"签到记录已提交（演示）\n预约编号：{reservation_id}\n核验方式：{method}\n状态：已签到\n提示：门禁结果需由场馆系统确认。"


@tool
def submit_feedback(user_id: str = "demo-user", resource_id: str = "", comfort: str = "舒适", venue_feedback: str = "", save_preference: bool = False) -> str:
    """记录活动反馈，并明确区分事件记录与待确认偏好。"""

    event = {
        "event_id": f"EVT-{uuid.uuid4().hex[:10].upper()}",
        "user_id": user_id,
        "resource_id": resource_id or "未指定",
        "comfort": comfort,
        "venue_feedback": venue_feedback or "未填写",
        "preference_status": "待用户确认" if save_preference else "仅记录本次事件",
    }
    with _lock:
        _feedback_events.append(event)
    return "反馈已记录\n" + "\n".join(f"{key}：{value}" for key, value in event.items())


@tool
def list_reservations(user_id: str = "demo-user") -> str:
    """列出当前用户的预约，避免把“我的预约”误判成创建请求。"""

    with _lock:
        items = [item.copy() for item in _reservations.values() if item["user_id"] == user_id]
    if not items:
        return "当前没有预约记录。"
    return "我的预约\n" + "\n\n".join(
        "\n".join(f"{key}：{value}" for key, value in item.items()) for item in items[-10:]
    )


@tool
def update_reservation(reservation_id: str, date: str, time: str, user_id: str = "demo-user", confirmed: bool = False) -> str:
    """更新当前用户预约；保留二次确认和营业时间校验。"""

    if not confirmed:
        return "修改待确认：请先展示新的日期和时间，并获得用户明确确认。"
    parsed_date, parsed_time, error = _parse_slot(date, time)
    if error:
        return error
    with _lock:
        item = _reservations.get(reservation_id.strip())
        if not item or item["user_id"] != user_id:
            return "未找到属于当前用户的预约。"
        gym = _find_gym(item["gym_id"])
        opening, closing = (dt.datetime.strptime(value, "%H:%M").time() for value in gym["hours"].split("-"))
        if not opening <= parsed_time <= closing:
            return f"业务失败：该场馆营业时间为 {gym['hours']}，请重新选择时间。"
        item["date"] = date
        item["time"] = time
    return f"预约已修改\n预约编号：{reservation_id}\n日期：{date}\n时间：{time}\n状态：已确认"


@tool
def create_human_handoff(user_id: str = "demo-user", reason: str = "需要人工协助") -> str:
    """创建人工接管事件；高风险和识别失败都应能进入该节点。"""

    ticket_id = f"HANDOFF-{uuid.uuid4().hex[:10].upper()}"
    return (
        "人工接管已创建\n"
        f"接管编号：{ticket_id}\n"
        f"用户：{user_id}\n"
        f"原因：{reason}\n"
        "状态：待社区工作人员或专业人员处理\n"
        "提示：AI 不会代替人工做医疗或现场安全决定。"
    )


@tool
def search_gymnasium(query: str) -> str:
    """按区域或设施关键词搜索共享健身场馆。"""

    query = query.strip()
    found: dict[str, dict[str, str]] = {}
    for district, item in GYMS.items():
        if district in query or district[:-1] in query or query in district:
            found[item["name"]] = item
    if "游泳" in query or "泳池" in query:
        found[GYMS["西湖区"]["name"]] = GYMS["西湖区"]
        found[GYMS["拱墅区"]["name"]] = GYMS["拱墅区"]
    if not found:
        return "未找到匹配场馆。可尝试：西湖区、滨江区、拱墅区、游泳池。"
    return "\n\n".join(_gym_text(item) for item in found.values())


def _parse_slot(date: str, time: str) -> tuple[dt.date | None, dt.time | None, str | None]:
    try:
        parsed_date = dt.datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return None, None, f"业务失败：日期格式错误 {date}，请使用 YYYY-MM-DD。"
    try:
        parsed_time = dt.datetime.strptime(time, "%H:%M").time()
    except ValueError:
        return None, None, f"业务失败：时间格式错误 {time}，请使用 HH:MM。"
    if parsed_date < dt.date.today():
        return None, None, "业务失败：不能预约过去的日期。"
    return parsed_date, parsed_time, None


@tool
def create_reservation(
    gym_name: str,
    date: str,
    time: str,
    user_id: str = "default",
    confirmed: bool = False,
    idempotency_key: str = "",
) -> str:
    """创建预约；必须显式 confirmed=True，重复请求使用幂等键。"""

    gym = _find_gym(gym_name)
    if not gym:
        return "业务失败：未找到对应场馆，请先选择有效场馆。"
    parsed_date, parsed_time, error = _parse_slot(date, time)
    if error:
        return error
    opening, closing = (dt.datetime.strptime(value, "%H:%M").time() for value in gym["hours"].split("-"))
    if not opening <= parsed_time <= closing:
        return f"业务失败：该场馆营业时间为 {gym['hours']}，请重新选择时间。"
    if not confirmed:
        return "预约待确认：请先向用户展示场馆、日期和时间，并获得明确确认。"

    key = idempotency_key.strip() or f"{user_id}|{gym['gym_id']}|{date}|{time}"
    with _lock:
        existing_id = _reservation_by_key.get(key)
        if existing_id:
            existing = _reservations[existing_id]
            return f"预约已存在\n预约编号：{existing_id}\n场馆：{existing['gym_name']}\n日期：{date}\n时间：{time}\n状态：{existing['status']}"
        slot_count = sum(
            1
            for item in _reservations.values()
            if item["gym_id"] == gym["gym_id"] and item["date"] == date and item["time"] == time and item["status"] == "confirmed"
        )
        if slot_count >= 20:
            return "业务失败：该时段预约名额已满，请选择其他时间。"
        reservation_id = f"RES-{uuid.uuid4().hex[:10].upper()}"
        _reservations[reservation_id] = {
            "reservation_id": reservation_id,
            "gym_id": gym["gym_id"],
            "gym_name": gym["name"],
            "date": date,
            "time": time,
            "user_id": user_id,
            "status": "confirmed",
        }
        _reservation_by_key[key] = reservation_id
    return (
        "预约成功\n"
        f"预约编号：{reservation_id}\n"
        f"场馆：{gym['name']}\n"
        f"日期：{date}\n"
        f"时间：{time}\n"
        f"用户：{user_id}\n"
        "状态：已确认\n"
        "提示：请提前10分钟到场签到。"
    )


@tool
def get_reservation(reservation_id: str, user_id: str = "default") -> str:
    """查询当前用户的预约。"""

    with _lock:
        item = _reservations.get(reservation_id.strip())
    if not item or item["user_id"] != user_id:
        return "未找到属于当前用户的预约。"
    return "预约详情\n" + "\n".join(f"{key}：{value}" for key, value in item.items())


@tool
def cancel_reservation(reservation_id: str, user_id: str = "default") -> str:
    """取消当前用户的预约。"""

    with _lock:
        item = _reservations.get(reservation_id.strip())
        if not item or item["user_id"] != user_id:
            return "未找到属于当前用户的预约。"
        if item["status"] == "cancelled":
            return f"预约 {reservation_id} 已经取消，无需重复操作。"
        item["status"] = "cancelled"
    return f"预约已取消\n预约编号：{reservation_id}\n场馆：{item['gym_name']}\n日期：{item['date']}\n时间：{item['time']}"


@tool
def get_device_status(device_type: str, gym_name: str = "") -> str:
    """查询设备状态；默认返回本地设备模拟器数据并标记观察时间。"""

    normalized = device_type.strip()
    if normalized not in DEVICE_STATES:
        if "跑步" in normalized or "空位" in normalized:
            normalized = "跑步机"
        elif "门" in normalized:
            normalized = "门禁"
        elif "摄像" in normalized:
            normalized = "摄像头"
        else:
            normalized = "全部"
    observed_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    location = gym_name.strip() or "西湖区健身中心（演示场馆）"
    return f"{DEVICE_STATES[normalized]}\n场馆：{location}\n{_source_line()}\n观测时间：{observed_at}\n数据新鲜度：演示数据"


@tool
def create_service_ticket(
    device_type: str,
    description: str,
    user_id: str = "default",
    idempotency_key: str = "",
) -> str:
    """创建运维工单，不直接控制物理设备。"""

    key = idempotency_key.strip() or f"{user_id}|{device_type}|{description}"
    with _lock:
        existing_id = _ticket_by_key.get(key)
        if existing_id:
            return _ticket_text(_tickets[existing_id])
        ticket_id = f"TICKET-{uuid.uuid4().hex[:10].upper()}"
        item = {
            "ticket_id": ticket_id,
            "device_type": device_type,
            "description": description,
            "user_id": user_id,
            "status": "open",
        }
        _tickets[ticket_id] = item
        _ticket_by_key[key] = ticket_id
    return _ticket_text(item)


def _ticket_text(item: dict[str, str]) -> str:
    return (
        "报修工单已创建\n"
        f"工单编号：{item['ticket_id']}\n"
        f"设备：{item['device_type']}\n"
        f"问题：{item['description']}\n"
        f"提交用户：{item['user_id']}\n"
        f"状态：{item['status']}\n"
        "处理方式：已转交场馆运维人员，请保留工单编号。"
    )


TOOLS = {
    "search_gymnasium": search_gymnasium,
    "create_reservation": create_reservation,
    "get_reservation": get_reservation,
    "cancel_reservation": cancel_reservation,
    "get_device_status": get_device_status,
    "create_service_ticket": create_service_ticket,
    "search_shared_resources": search_shared_resources,
    "get_user_profile": get_user_profile,
    "update_user_permission": update_user_permission,
    "find_companions": find_companions,
    "get_access_credential": get_access_credential,
    "record_check_in": record_check_in,
    "submit_feedback": submit_feedback,
    "list_reservations": list_reservations,
    "update_reservation": update_reservation,
    "create_human_handoff": create_human_handoff,
}
