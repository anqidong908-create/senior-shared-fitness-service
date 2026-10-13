"""Structural checks without calling the local model."""

from __future__ import annotations

from business_tools import TOOLS
from multi_agent_service import app


def main() -> None:
    paths = {route.path for route in app.routes}
    assert {"/health", "/agents", "/chat", "/stream", "/docs"}.issubset(paths)
    graph_nodes = set(app.state.graph.get_graph().nodes)
    expected_nodes = {
        "__start__",
        "router_agent",
        "task_planner_agent",
        "context_agent",
        "risk_agent",
        "domain_agent",
        "conflict_resolution_agent",
        "human_handoff_agent",
        "response_agent",
        "__end__",
    }
    assert expected_nodes.issubset(graph_nodes)
    assert {
        "search_gymnasium",
        "create_reservation",
        "get_reservation",
        "cancel_reservation",
        "get_device_status",
        "create_service_ticket",
        "search_shared_resources",
        "get_user_profile",
        "update_user_permission",
        "find_companions",
        "get_access_credential",
        "record_check_in",
        "submit_feedback",
        "list_reservations",
        "update_reservation",
        "create_human_handoff",
    }.issubset(set(TOOLS))
    result = TOOLS["get_device_status"].invoke({"device_type": "跑步机"})
    assert "空闲：4台" in result
    assert "本地设备模拟器" in result
    pending = TOOLS["create_reservation"].invoke(
        {"gym_name": "西湖区健身中心", "date": "2099-01-01", "time": "10:00", "user_id": "smoke"}
    )
    assert "待确认" in pending
    print("multi-agent smoke test passed")


if __name__ == "__main__":
    main()
