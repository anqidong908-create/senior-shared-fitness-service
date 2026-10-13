"""Deterministic product-boundary tests that do not call Ollama."""

from __future__ import annotations

from business_tools import TOOLS
from multi_agent_runtime import _assess_risk, _fallback_route, _reservation_action
from ollama_client import extract_json


def main() -> None:
    cases = {
        "找西湖区附近的健身房": "venue",
        "跑步机现在有空位吗": "device",
        "帮我预约明天下午3点西湖区健身中心": "reservation",
        "储物柜打不开，帮我报修": "support",
        "你好，怎么使用这个系统": "general",
    }
    for message, expected in cases.items():
        assert _fallback_route(message) == expected, message

    assert _assess_risk("我胸痛，还能去运动吗")[0] == "high"
    assert _assess_risk("我膝盖疼，想慢走")[0] == "medium"
    assert _assess_risk("我想找附近的场馆")[0] == "low"
    assert _reservation_action("我的预约", {"reservation_id": ""}) == "list"
    assert _reservation_action("修改预约时间", {"reservation_id": ""}) == "update"

    assert "确认" in TOOLS["create_reservation"].invoke(
        {"gym_name": "西湖区健身中心", "date": "2099-01-01", "time": "10:00", "user_id": "eval"}
    )
    reservation_args = {
        "gym_name": "西湖区健身中心",
        "date": "2099-01-02",
        "time": "10:00",
        "user_id": "eval",
        "confirmed": True,
        "idempotency_key": "evaluation-key-01",
    }
    first = TOOLS["create_reservation"].invoke(reservation_args)
    second = TOOLS["create_reservation"].invoke(reservation_args)
    assert "预约成功" in first
    assert "预约已存在" in second
    reservation_id = first.split("预约编号：", 1)[1].split("\n", 1)[0]
    update_preview = TOOLS["update_reservation"].invoke({
        "reservation_id": reservation_id,
        "date": "2099-01-03",
        "time": "11:00",
        "user_id": "eval",
    })
    assert "修改待确认" in update_preview
    update_confirmed = TOOLS["update_reservation"].invoke({
        "reservation_id": reservation_id,
        "date": "2099-01-03",
        "time": "11:00",
        "user_id": "eval",
        "confirmed": True,
    })
    assert "预约已修改" in update_confirmed
    assert "我的预约" in TOOLS["list_reservations"].invoke({"user_id": "eval"})
    assert "本地设备模拟器" in TOOLS["get_device_status"].invoke({"device_type": "跑步机"})
    resources = TOOLS["search_shared_resources"].invoke({"query": "附近慢走资源", "activity_type": "步行"})
    assert "适老设施" in resources and "风险提示" in resources
    assert "双方授权" in TOOLS["find_companions"].invoke({"activity": "慢走"})
    permission = TOOLS["update_user_permission"].invoke({"user_id": "eval", "permission": "family_assistance", "enabled": True})
    assert "家庭协助不等于家庭控制" in permission
    assert extract_json('```json\n{"route":"device"}\n```')["route"] == "device"
    assert extract_json('{"reason":"括号 { 在字符串中"}') ["reason"] == "括号 { 在字符串中"
    print("evaluation tests passed")


if __name__ == "__main__":
    main()
