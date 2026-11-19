import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config import Settings
from reasoning import ReasoningTools


class SubagentLimitTests(unittest.TestCase):
    def test_task_deadline_defaults_to_unlimited_and_respects_explicit_settings(self):
        for configured, expected in ((None, None), ("60", 60), ("unlimited", None), ("0", None)):
            settings = {"DEEPSEEK_API_KEY": "test"}
            if configured is not None:
                settings["DEEPSEEK_MAX_TASK_SECONDS"] = configured
            with (
                self.subTest(configured=configured),
                patch("config.load_env"),
                patch.dict("os.environ", settings, clear=True),
            ):
                self.assertEqual(Settings.from_env().max_task_seconds, expected)

    def test_default_configuration_leaves_subagents_and_output_unlimited(self):
        with (
            patch("config.load_env"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}, clear=True),
        ):
            settings = Settings.from_env()
            self.assertIsNone(settings.max_subagents)
            self.assertIsNone(settings.max_output_tokens)

    def test_configuration_honors_explicit_limits_or_unlimited(self):
        for configured, expected in [
            (-1, 1),
            (0, None),
            ("unlimited", None),
            (1, 1),
            (2, 2),
            (8, 8),
            (19, 19),
            (20, 20),
            (21, 21),
            (100, 100),
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

    def test_malformed_numbers_name_their_variable(self):
        for name, value in (
            ("DEEPSEEK_MAX_SUBAGENTS", "abc"),
            ("DEEPSEEK_MAX_TASK_SECONDS", "soon"),
            ("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "1m"),
        ):
            with (
                self.subTest(name=name),
                patch("config.load_env"),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test", name: value}, clear=True),
                self.assertRaisesRegex(ValueError, f"{name}.*{value!r}"),
            ):
                Settings.from_env()

    def test_reasoning_runner_only_enforces_explicit_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            default = ReasoningTools(Mock(), Mock(), [], workspace=workspace)
            self.assertIsNone(default.slots)
            for configured, expected in [(0, 1), (2, 2), (20, 20), (21, 21)]:
                with self.subTest(configured=configured):
                    reasoning = ReasoningTools(
                        Mock(), Mock(), [], max_workers=configured, workspace=workspace
                    )
                    self.assertEqual(reasoning.max_workers, expected)
                    self.assert_slot_capacity(reasoning.slots, expected)

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
