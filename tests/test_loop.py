import json
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from commands import CommandName, CommandResult, LoopCommand, parse_command, parse_loop
from conversation import Conversation
from main import handle_command, main, use_loop
from model_request import CompletionError
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tool_runtime import PAGE_CHARS
from tools import CodingTools, ToolPolicy
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
        self.assertEqual(parse_loop(command.arguments[0]), LoopCommand(300, 3, task))

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


class LoopCheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.console = Mock(spec=Console)
        self.console.interactive = False
        self.settings = make_settings(
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
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

    def run_cli(self, client, *arguments):
        with (
            patch("sys.argv", ["agent", "/loop", "1s", *arguments]),
            patch("main.load_settings", return_value=self.settings),
            patch("main.Console", return_value=self.console),
            patch("main.model_client", return_value=client),
            patch("command_ui.time") as clock,
        ):
            status = main()
        self.console.error.assert_not_called()
        return status, clock.sleep

    def evidence(self, request):
        self.assertFalse(request["tools"])
        prompt = next(
            item["content"] for item in reversed(request["messages"]) if item["role"] == "user"
        )
        return json.loads(prompt.split("\n", 1)[1])

    def test_parse_check_preserves_literal_shell_remainder(self):
        shell = 'printf "%s\\n" "--timeout 2" > "check log"\ntest -s "check log"'
        command = parse_command(f"/loop 2m --count 4 --check --timeout 5 {shell}")
        self.assertEqual(parse_loop(*command.arguments), LoopCommand(120, 4, shell, True, 5))
        self.assertEqual(parse_loop(f"1s --check {shell}"), LoopCommand(1, None, shell, True, 30))

    def test_parse_check_preserves_leading_shell_whitespace(self):
        shell = " \t\n\u00a0:"
        for options in ("--check", "--check --timeout 5"):
            with self.subTest(options=options):
                command = parse_command(f"/loop 1s {options} {shell}")
                self.assertEqual(parse_loop(*command.arguments).task, shell)

    def test_rejects_missing_check_command_and_invalid_timeout(self):
        for source in (
            "1s --check", "1s --count 2 --check", "1s --check --timeout",
            "1s --check --timeout 1", "1s --check --timeout 0 true",
            "1s --check --timeout -1 true", "1s --check --timeout 1.5 true",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_loop(source)

    def test_healthy_shell_checks_run_each_round_without_requesting_model(self):
        client = FakeClient()
        shell = 'if read -r line; then exit 9; fi\nprintf "%s\\n" "--timeout 2" >> "check log"'

        status, sleep = self.run_cli(client, "--count", "2", "--check", shell)

        self.assertEqual(status, 0)
        self.assertEqual((self.workspace / "check log").read_text(), "--timeout 2\n" * 2)
        self.assertEqual(client.requests, [])
        sleep.assert_called_once_with(1)

    def test_cli_preserves_a_trailing_escaped_space_in_check_command(self):
        client = FakeClient()
        shell = "printf '%s' > escaped value\\ "

        status, _ = self.run_cli(client, "--count", "1", "--check", shell)

        self.assertEqual(status, 0)
        self.assertEqual((self.workspace / "escaped").read_text(), "value ")
        command = parse_command(f"/loop 1s --check {shell}")
        self.assertEqual(parse_loop(*command.arguments).task, shell)
        self.assertEqual(client.requests, [])

    def test_nonzero_check_reports_bounded_output_and_fails_even_after_recovery(self):
        client = FakeClient({"role": "assistant", "content": "Check exited with an error."})
        shell = (
            "if test -f attempted; then touch recovered; exit 0; fi; touch attempted; "
            f"printf 'start-of-output'; printf '%*s' {2 * PAGE_CHARS} ''; "
            "printf 'diagnostic-tail\\n' >&2; exit 7"
        )

        status, sleep = self.run_cli(client, "--count", "2", "--check", shell)

        self.assertEqual(status, 1)
        self.assertTrue((self.workspace / "recovered").is_file())
        sleep.assert_called_once_with(1)
        self.assertEqual(len(client.requests), 1)
        evidence = self.evidence(client.requests[0])
        self.assertEqual(evidence["command"], shell)
        self.assertEqual(evidence["exit_code"], 7)
        self.assertTrue(evidence["output_truncated"])
        self.assertLessEqual(len(evidence["output"]), PAGE_CHARS)
        self.assertTrue(evidence["output"].endswith("diagnostic-tail\n"))
        self.assertNotIn("start-of-output", evidence["output"])

    def test_timed_out_check_sends_captured_output_to_model(self):
        client = FakeClient({"role": "assistant", "content": "Check timed out."})
        shell = "printf 'before timeout\\n' >&2; exec sleep 10"

        status, sleep = self.run_cli(client, "--count", "1", "--check", "--timeout", "1", shell)

        self.assertEqual(status, 1)
        sleep.assert_not_called()
        self.assertEqual(len(client.requests), 1)
        evidence = self.evidence(client.requests[0])
        self.assertIn("超过 1 秒", evidence["reason"])
        self.assertEqual(evidence["output"], "before timeout\n")
        self.assertFalse(evidence["output_truncated"])

    def test_completion_error_does_not_stop_subsequent_checks(self):
        client = FakeClient(
            CompletionError("fixture", "analysis failed"),
            {"role": "assistant", "content": "Second check failed."},
        )
        shell = "printf 'round\\n' >> rounds; wc -l < rounds; exit 3"

        status, sleep = self.run_cli(client, "--count", "2", "--check", shell)

        self.assertEqual(status, 1)
        sleep.assert_called_once_with(1)
        outputs = [self.evidence(request)["output"].strip() for request in client.requests]
        self.assertEqual(outputs, ["1", "2"])
        warnings = [call.args[0] for call in self.console.warning.call_args_list]
        self.assertTrue(any("analysis failed" in warning for warning in warnings))

    def test_unsolicited_tool_call_cannot_run_in_anomaly_analysis(self):
        client = FakeClient(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "unsolicited-exec",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": json.dumps({"command": "touch model-executed"}),
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Check failed."},
        )

        status, _ = self.run_cli(client, "--count", "1", "--check", "exit 4")

        self.assertFalse((self.workspace / "model-executed").exists())
        self.assertEqual(status, 1)
        self.assertGreaterEqual(len(client.requests), 1)
        self.assertTrue(all(not request["tools"] for request in client.requests))

    def test_anomaly_analysis_does_not_resume_team_work(self):
        client = FakeClient({"role": "assistant", "content": "Check failed."})
        conversation = Conversation(self.settings, client, CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        team = Mock(main_instructions=Mock(return_value="Team instructions."))
        team.turn.return_value = nullcontext()
        team.turn_context.return_value = ""
        team.member_names.return_value = ()
        team.source_workers_at_takeover.return_value = (False, 0)
        team.final_answer_problem.return_value = None
        team.continues_in_background.return_value = False
        conversation.team = team

        status = use_loop(conversation, self.console, ["1s --count 1 --check exit 4"])

        self.assertEqual(status, 1)
        self.assertEqual(len(client.requests), 1)
        self.evidence(client.requests[0])
        team.turn.assert_not_called()

    def test_environment_approval_is_rechecked_between_rounds(self):
        client = FakeClient()
        conversation = Conversation(self.settings, client, CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        environment = Mock()
        conversation.active_environment = environment
        command = "1s --count 2 --check printf 'round\\n' >> rounds"
        with (
            patch.object(conversation, "environments") as store,
            patch("command_ui.time.sleep"),
        ):
            store.require_approved.side_effect = [None, ValueError("Approval revoked")]
            with self.assertRaisesRegex(ValueError, "Approval revoked"):
                use_loop(conversation, self.console, [command])

        self.assertEqual((self.workspace / "rounds").read_text(), "round\n")
        store.verify.assert_called_once_with(environment)
        self.assertEqual(client.requests, [])

    def test_check_requires_exec_permission_before_running_shell(self):
        tools = CodingTools(self.workspace)
        self.addCleanup(tools.close)
        tools.policy = ToolPolicy(deny=frozenset({"exec"}))
        conversation = Mock(tools=tools)
        with patch("main.ask") as ask, self.assertRaises(PermissionError):
            use_loop(conversation, self.console, ["1s --count 1 --check touch check-executed"])

        ask.assert_not_called()
        self.assertFalse((self.workspace / "check-executed").exists())

    def test_interrupt_during_check_reaps_shell_and_skips_model_and_next_round(self):
        tools = CodingTools(self.workspace)
        self.addCleanup(tools.close)
        conversation = Mock(tools=tools)
        wait = subprocess.Popen.wait
        interrupted = []

        def interrupt_wait(process, timeout=None):
            if timeout is not None:
                interrupted.append(process)
                raise KeyboardInterrupt
            return wait(process, timeout=5)

        with (
            patch.object(subprocess.Popen, "wait", interrupt_wait),
            patch("main.ask") as ask,
            patch("command_ui.time") as clock,
        ):
            status = use_loop(conversation, self.console, ["1s --count 2 --check exec sleep 10"])

        self.assertEqual(status, 130)
        self.assertEqual(len(interrupted), 1)
        self.assertIsNotNone(interrupted[0].returncode)
        ask.assert_not_called()
        clock.sleep.assert_not_called()
