from __future__ import annotations

import datetime as dt
import tempfile
import unittest
import uuid
from pathlib import Path

from auth import issue_token, verify_token
from business_tools import TOOLS
from multi_agent_runtime import _assess_risk, _fallback_route, _reservation_action
from multi_agent_service import SessionAccessError, SessionStore
from ollama_client import extract_json


class CoreUnitTests(unittest.TestCase):
    def test_signed_token_tamper_and_expiry(self) -> None:
        token = issue_token("unit-user", "unit-secret")
        self.assertEqual(verify_token(token, "unit-secret"), "unit-user")
        self.assertIsNone(verify_token(token + "x", "unit-secret"))
        expired = issue_token("unit-user", "unit-secret", ttl_seconds=-1)
        self.assertIsNone(verify_token(expired, "unit-secret"))

    def test_route_and_risk_boundaries(self) -> None:
        self.assertEqual(_fallback_route("找西湖区附近的健身房"), "venue")
        self.assertEqual(_fallback_route("跑步机现在有空位吗"), "device")
        self.assertEqual(_fallback_route("储物柜打不开，帮我报修"), "support")
        self.assertEqual(_assess_risk("我胸痛，还能运动吗")[0], "high")
        self.assertEqual(_assess_risk("我膝盖疼，想慢走")[0], "medium")

    def test_my_reservations_ignores_historical_id(self) -> None:
        self.assertEqual(_reservation_action("我的预约", {"reservation_id": "RES-OLD"}), "list")
        self.assertEqual(_reservation_action("查看 RES-OLD", {"reservation_id": "RES-OLD"}), "query")

    def test_resource_filter_has_provenance(self) -> None:
        result = TOOLS["search_shared_resources"].invoke({"query": "附近慢走资源", "activity_type": "步行"})
        self.assertIn("适老设施", result)
        self.assertIn("状态来源：社区资源演示目录", result)

    def test_reservation_confirmation_idempotency_and_ownership(self) -> None:
        user_id = f"unit-{uuid.uuid4().hex[:8]}"
        reservation_id_key = f"unit-key-{uuid.uuid4().hex}"
        args = {
            "gym_name": "西湖区健身中心",
            "date": (dt.date.today() + dt.timedelta(days=30)).isoformat(),
            "time": "10:00",
            "user_id": user_id,
            "idempotency_key": reservation_id_key,
        }
        self.assertIn("预约待确认", TOOLS["create_reservation"].invoke(args))
        created = TOOLS["create_reservation"].invoke({**args, "confirmed": True})
        repeated = TOOLS["create_reservation"].invoke({**args, "confirmed": True})
        self.assertIn("预约成功", created)
        self.assertIn("预约已存在", repeated)
        reservation_id = created.split("预约编号：", 1)[1].split("\n", 1)[0]
        self.assertIn("未找到属于当前用户", TOOLS["get_reservation"].invoke({"reservation_id": reservation_id, "user_id": "other-user"}))
        self.assertIn("预约已取消", TOOLS["cancel_reservation"].invoke({"reservation_id": reservation_id, "user_id": user_id}))

    def test_session_store_persists_and_enforces_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sessions.sqlite3")
            store = SessionStore(path)
            reloaded = None
            try:
                session_id = "session-unit"
                history, pending, memory = store.load(session_id, "owner")
                self.assertEqual((history, pending, memory), ([], None, {}))
                store.save_result(session_id, "owner", "你好", "已收到", {"type": "reservation_create"}, {"last_route": "general"})
                reloaded = SessionStore(path)
                history, pending, memory = reloaded.load(session_id, "owner")
                self.assertEqual(history[-2:], [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "已收到"}])
                self.assertEqual(pending["type"], "reservation_create")
                self.assertEqual(memory["last_route"], "general")
                with self.assertRaises(SessionAccessError):
                    reloaded.load(session_id, "other-user")
            finally:
                store.close()
                if reloaded:
                    reloaded.close()

    def test_json_extraction_handles_fenced_response(self) -> None:
        self.assertEqual(extract_json('```json\n{"route":"device"}\n```')["route"], "device")


if __name__ == "__main__":
    unittest.main()
