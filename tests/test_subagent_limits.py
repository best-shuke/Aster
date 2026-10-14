import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import http_transport
from config import MAX_SUBAGENTS, Settings
from reasoning import ReasoningTools


class SubagentLimitTests(unittest.TestCase):
    def test_default_configuration_allows_twenty_subagents(self):
        with (
            patch("config.load_env"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}, clear=True),
        ):
            self.assertEqual(MAX_SUBAGENTS, 20)
            self.assertEqual(Settings.from_env().max_subagents, 20)

    def test_configuration_preserves_lower_limits_and_clamps_to_twenty(self):
        for configured, expected in [
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 2),
            (8, 8),
            (19, 19),
            (20, 20),
            (21, 20),
            (100, 20),
        ]:
            with (
                self.subTest(configured=configured),
                patch("config.load_env"),
                patch.dict(
                    "os.environ",
                    {
                        "DEEPSEEK_API_KEY": "test",
                        "DEEPSEEK_MAX_SUBAGENTS": str(configured),
                    },
                    clear=True,
                ),
            ):
                self.assertEqual(Settings.from_env().max_subagents, expected)

    def test_reasoning_runner_uses_twenty_as_default_and_hard_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            default = ReasoningTools(Mock(), Mock(), [], workspace=workspace)
            self.assertEqual(default.max_workers, 20)
            for configured, expected in [(0, 1), (2, 2), (20, 20), (21, 20)]:
                with self.subTest(configured=configured):
                    reasoning = ReasoningTools(
                        Mock(), Mock(), [], max_workers=configured, workspace=workspace
                    )
                    self.assertEqual(reasoning.max_workers, expected)
                    self.assert_slot_capacity(reasoning.slots, expected)

    def test_http_transport_has_twenty_bounded_worker_slots(self):
        self.assert_slot_capacity(http_transport._SLOTS, 20)

    def assert_slot_capacity(self, semaphore, capacity):
        acquired = 0
        try:
            for _ in range(capacity):
                self.assertTrue(semaphore.acquire(blocking=False))
                acquired += 1
            extra = semaphore.acquire(blocking=False)
            acquired += int(extra)
            self.assertFalse(extra)
        finally:
            for _ in range(acquired):
                semaphore.release()
