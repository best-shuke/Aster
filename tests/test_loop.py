import tempfile
import unittest
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from commands import CommandName, CommandResult, parse_command, parse_loop
from main import handle_command, main, use_loop
from tests.support import make_settings
from tests.test_conversation import FakeClient
from ui import Console


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.console = Mock(spec=Console)
        self.console.interactive = True
        self.console.interrupt_on_escape.return_value = nullcontext()
        self.conversation = Mock(last_outcome={"status": "completed"})

    def test_loop_preserves_quoted_and_multiline_task(self):
        task = 'Run `rg "ERROR details" logs`\nReport new matches.'
        command = parse_command(f"/loop 5m --count 3 {task}")
        self.assertIs(command.name, CommandName.LOOP)
        self.assertEqual(parse_loop(command.arguments[0]), (300, 3, task))

    def test_rejects_invalid_interval_count_and_missing_task(self):
        for source in (
            "", "1s", "0s task", "-1m task", "5x task", "9999999d task",
            "1s --count", "1s --count 2", "1s --count 0 task", "1s --count x task",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_loop(source)

    def test_interactive_dispatch_runs_immediately_and_waits_only_between_rounds(self):
        events = []

        def run(conversation, console, prompt, **options):
            self.assertIs(conversation, self.conversation)
            self.assertIn("fresh observations", options["task_contract"].description)
            events.append(prompt)

        with (
            patch("main.ask", side_effect=run),
            patch("command_ui.time.sleep", side_effect=events.append),
        ):
            result = handle_command(self.conversation, self.console, "/loop 2s --count 3 inspect")
        self.assertIs(result, CommandResult.HANDLED)
        self.assertEqual(events, ["inspect", 2, "inspect", 2, "inspect"])
        self.console.interrupt_on_escape.assert_called_once_with()

    def test_unbounded_loop_stops_when_interrupted_between_turns(self):
        with (
            patch("main.ask") as ask,
            patch("command_ui.time.sleep", side_effect=[None, KeyboardInterrupt]) as sleep,
        ):
            result = use_loop(self.conversation, self.console, ["1h inspect"])
        self.assertEqual(result, 130)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.console.info.call_args.args, ("循环已停止。",))

    def test_interrupt_during_turn_skips_wait_and_remaining_rounds(self):
        with (
            patch("main.ask", side_effect=KeyboardInterrupt) as ask,
            patch("command_ui.time.sleep") as sleep,
        ):
            result = use_loop(self.conversation, self.console, ["1s --count 2 inspect"])
        self.assertEqual(result, 130)
        ask.assert_called_once()
        sleep.assert_not_called()

    def test_incomplete_round_counts_towards_limit(self):
        self.conversation.last_outcome = {"status": "incomplete"}
        with patch("main.ask") as ask, patch("command_ui.time.sleep"):
            result = use_loop(self.conversation, self.console, ["1s --count 2 inspect"])
        self.assertEqual(result, 1)
        self.assertEqual(ask.call_count, 2)

    def test_cli_rejects_invalid_arguments_before_loading_settings(self):
        with (
            patch("sys.argv", ["agent", "/loop", "0s", "inspect"]),
            patch("sys.stderr", new_callable=StringIO) as stderr,
            patch("main.load_settings", return_value=None) as load_settings,
        ):
            self.assertEqual(main(), 2)

        load_settings.assert_not_called()
        self.assertIn("循环间隔须大于零", stderr.getvalue())

    def test_cli_round_value_error_exits_with_runtime_failure(self):
        self.console.interactive = False
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = make_settings(
                workspace=workspace,
                session_file=workspace / "session.json",
                api_key="test",
                base_url="https://example.invalid",
                model="test",
                max_context_chars=120_000,
                keep_recent_turns=6,
                max_task_seconds=None,
                max_subagents=None,
                skills_enabled=False,
                mcp_enabled=False,
                websearch_enabled=False,
            )
            client = FakeClient()
            with (
                patch("main.load_settings", return_value=settings),
                patch("main.Console", return_value=self.console),
                patch("main.model_client", return_value=client),
                patch("command_ui.time.sleep") as sleep,
                patch("sys.argv", ["agent", "/loop", "1s", "--count", "2", "/skill:missing inspect"]),
            ):
                self.assertEqual(main(), 1)

        self.console.error.assert_called_once()
        self.assertIn("Skills 已关闭", str(self.console.error.call_args.args[0]))
        self.assertEqual(client.requests, [])
        sleep.assert_not_called()

    def test_cli_runs_multiple_turns_in_same_session(self):
        self.console.interactive = False
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = make_settings(
                workspace=workspace,
                session_file=workspace / "session.json",
                api_key="test",
                base_url="https://example.invalid",
                model="test",
                max_context_chars=120_000,
                keep_recent_turns=6,
                max_task_seconds=None,
                max_subagents=None,
                mcp_enabled=False,
                websearch_enabled=False,
            )
            task = 'Explain `rg "ERROR details" logs`.'
            client = FakeClient(
                {"role": "assistant", "content": "First result."},
                {"role": "assistant", "content": "Second result."},
            )
            with (
                patch("main.load_settings", return_value=settings),
                patch("main.Console", return_value=self.console),
                patch("main.model_client", return_value=client),
                patch("command_ui.time.sleep") as sleep,
                patch("sys.argv", ["agent", "/loop", "1s", "--count", "2", task]),
            ):
                self.assertEqual(main(), 0)
            self.assertTrue(settings.session_file.is_file())
        sleep.assert_called_once_with(1)
        self.assertEqual(len(client.requests), 2)
        messages = client.requests[1]["messages"]
        self.assertEqual([item["content"] for item in messages if item["role"] == "user"], [task] * 2)
        self.assertIn("First result.", [item["content"] for item in messages])
