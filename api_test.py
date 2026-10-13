"""Real HTTP tests against the local Ollama-backed multi-agent service."""

from __future__ import annotations

import os
import re

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8010")
DEMO_LOGIN_KEY = os.getenv("DEMO_LOGIN_KEY", "local-demo-login")


def auth_headers(user_id: str) -> dict[str, str]:
    token = requests.post(f"{BASE_URL}/auth/demo-token", params={"user_id": user_id}, headers={"X-Demo-Login-Key": DEMO_LOGIN_KEY}, timeout=15)
    token.raise_for_status()
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


def main() -> None:
    health = requests.get(f"{BASE_URL}/health", timeout=20)
    health.raise_for_status()
    headers = auth_headers("api-test")
    assert health.json()["model"].startswith("deepseek")
    assert health.json()["device_data_mode"] == "simulator"

    agents = requests.get(f"{BASE_URL}/agents", timeout=15)
    agents.raise_for_status()
    assert len(agents.json()["agents"]) == 7

    resources = requests.post(
        f"{BASE_URL}/resources/recommend",
        json={"query": "附近慢走资源", "activity_type": "步行", "max_distance_km": 5},
        timeout=15,
    )
    resources.raise_for_status()
    assert resources.json()["count"] >= 1
    assert "elderly_facilities" in resources.json()["items"][0]
    assert "observed_at" in resources.json()["items"][0]

    unauthorized = requests.get(f"{BASE_URL}/profiles/api-test", timeout=15)
    assert unauthorized.status_code == 401

    profile = requests.get(f"{BASE_URL}/profiles/api-test", headers=headers, timeout=15)
    profile.raise_for_status()
    assert profile.json()["permissions"]["family_assistance"] is False

    device = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "跑步机现在有空位吗", "user_id": "api-test"}, headers=headers,
        timeout=240,
    )
    device.raise_for_status()
    device_data = device.json()
    assert device_data["route"] == "device"
    assert device_data["tool_results"][0]["tool_name"] == "get_device_status"
    assert "4" in device_data["response"]
    assert "本地设备模拟器" in device_data["response"]
    agent_names = {item["agent"] for item in device_data["agent_trace"]}
    assert {"task_planner_agent", "context_agent", "risk_agent", "domain_agent", "conflict_resolution_agent", "response_agent"}.issubset(agent_names)
    assert device_data["memory"]["last_route"] == "device"
    assert len(device_data["task_plan"]) >= 5
    assert "未发现业务与安全约束冲突" in device_data["conflict_result"]

    high_risk = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "我胸痛，还能去运动吗", "user_id": "api-test"}, headers=headers,
        timeout=60,
    )
    high_risk.raise_for_status()
    high_risk_data = high_risk.json()
    assert high_risk_data["route"] == "safety"
    assert high_risk_data["risk_level"] == "high"
    assert any(item["tool_name"] == "create_human_handoff" for item in high_risk_data["tool_results"])
    assert all(item["tool_name"] == "create_human_handoff" for item in high_risk_data["tool_results"])
    assert high_risk_data["handoff_result"].startswith("人工接管已创建")
    assert "暂停运动" in high_risk_data["response"]

    reservation = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "帮我预约明天下午3点西湖区健身中心", "user_id": "student-09"}, headers=auth_headers("student-09"),
        timeout=240,
    )
    reservation.raise_for_status()
    reservation_data = reservation.json()
    assert reservation_data["route"] == "reservation"
    assert reservation_data["tool_results"] == []
    assert reservation_data["pending_action"]["type"] == "reservation_create"
    assert "确认预约" in reservation_data["response"]

    confirmed = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "确认预约",
            "user_id": "student-09",
            "session_id": reservation_data["session_id"],
        }, headers=auth_headers("student-09"),
        timeout=240,
    )
    confirmed.raise_for_status()
    confirmed_data = confirmed.json()
    assert confirmed_data["tool_results"][0]["tool_name"] == "create_reservation"
    assert confirmed_data["tool_results"][0]["args"]["user_id"] == "student-09"
    assert "预约成功" in confirmed_data["response"]
    assert confirmed_data["pending_action"] is None

    reservation_id = re.search(r"RES-[A-Z0-9]+", confirmed_data["response"])
    assert reservation_id
    cancelled = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": f"取消预约 {reservation_id.group(0)}",
            "user_id": "student-09",
            "session_id": reservation_data["session_id"],
        }, headers=auth_headers("student-09"),
        timeout=240,
    )
    cancelled.raise_for_status()
    assert "预约已取消" in cancelled.json()["response"]

    listed = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "查看我的预约",
            "user_id": "student-09",
            "session_id": reservation_data["session_id"],
        }, headers=auth_headers("student-09"),
        timeout=240,
    )
    listed.raise_for_status()
    listed_data = listed.json()
    assert listed_data["tool_results"][0]["tool_name"] == "list_reservations"
    assert listed_data["pending_action"] is None

    permission_unauthorized = requests.put(
        f"{BASE_URL}/permissions",
        json={"user_id": "student-09", "permission": "family_assistance", "enabled": True},
        timeout=15,
    )
    assert permission_unauthorized.status_code == 401

    forbidden = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "查看我的预约",
            "user_id": "another-user",
            "session_id": reservation_data["session_id"],
        }, headers=auth_headers("another-user"),
        timeout=15,
    )
    assert forbidden.status_code == 403

    invalid = requests.post(f"{BASE_URL}/chat", json={"message": "   "}, headers=auth_headers("api-test"), timeout=15)
    assert invalid.status_code == 422

    with requests.post(
        f"{BASE_URL}/stream",
        json={"message": "储物柜打不开，帮我报修", "user_id": "api-test"}, headers=headers,
        stream=True,
        timeout=240,
    ) as stream:
        stream.raise_for_status()
        payload = "".join(f"{line}\n" for line in stream.iter_lines(decode_unicode=True) if line is not None)
    assert "event: agent" in payload
    assert "domain_agent" in payload
    assert "event: tool_result" in payload
    assert "event: token" in payload
    assert "event: done" in payload
    print("multi-agent API end-to-end test passed")


if __name__ == "__main__":
    main()
