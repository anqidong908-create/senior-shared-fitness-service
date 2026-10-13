from __future__ import annotations

import unittest

from multi_agent_runtime import build_graph, make_state


class FakeOllamaClient:
    """Deterministic model double for graph tests; no network calls."""

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
        system = messages[0]["content"]
        if "路由 Agent" in system:
            return '{"route":"device","reason":"用户在查询设备状态"}'
        if "设备状态 Agent" in system:
            return '{"device_type":"跑步机","gym_name":""}'
        return "设备状态已读取，请以工具事实为准。"


class GraphUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_graph(FakeOllamaClient())

    def test_parallel_device_workflow(self) -> None:
        result = self.graph.invoke(make_state("跑步机现在有空位吗", "graph-user", "graph-device", []))
        names = {item["agent"] for item in result["agent_trace"]}
        self.assertEqual(result["route"], "device")
        self.assertTrue({"task_planner_agent", "context_agent", "risk_agent", "domain_agent", "conflict_resolution_agent", "response_agent"}.issubset(names))
        self.assertEqual(result["tool_results"][0]["tool_name"], "get_device_status")
        self.assertEqual(len(result["task_plan"]), 5)

    def test_high_risk_skips_domain_and_hands_off(self) -> None:
        result = self.graph.invoke(make_state("我胸痛，还能去运动吗", "graph-risk-user", "graph-risk", []))
        names = {item["agent"] for item in result["agent_trace"]}
        self.assertEqual(result["route"], "safety")
        self.assertEqual(result["risk_level"], "high")
        self.assertIn("human_handoff_agent", names)
        self.assertEqual(result["tool_results"][0]["tool_name"], "create_human_handoff")
        self.assertNotIn("create_reservation", {item["tool_name"] for item in result["tool_results"]})


if __name__ == "__main__":
    unittest.main()
