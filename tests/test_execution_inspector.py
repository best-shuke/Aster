import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from conversation import Conversation
from execution_history import ArtifactPages, ExecutionEntry, execution_entries
from main import ask, handle_command
from tests import test_terminal_input as terminal_tests
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tools import CodingTools
from trace_ui import TraceInspector
from ui import Console


def entry(number=1, *, command="echo example", status="ok", artifact=None, task="task-1"):
    return ExecutionEntry(
        number,
        {
            "tool": "exec",
            "execution_id": f"execution-{number}",
            "state": "completed",
            "arguments": json.dumps({"command": command, "cwd": "."}),
            "trace": {"task_id": task},
            "task_title": "检查测试",
            "result": json.dumps(
                {
                    "status": status,
                    "data": {"output": "example\n", "exit_code": 0},
                    "artifact_path": str(artifact) if artifact else None,
                }
            ),
        },
    )


class ExecutionHistoryTests(unittest.TestCase):
    def test_legacy_calls_are_matched_within_their_assistant_batch(self):
        messages = []
        runs = []
        for command in ("first", "second"):
            index = len(messages)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "reused",
                                "function": {
                                    "name": "exec",
                                    "arguments": json.dumps({"command": command}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "reused",
                        "content": json.dumps({"status": "ok", "data": {"output": command}}),
                    },
                ]
            )
            runs.append({"assistant_index": index, "tool_call_id": "reused", "tool": "exec"})
        records = execution_entries(runs, messages)
        self.assertEqual([record.arguments["command"] for record in records], ["first", "second"])
        self.assertEqual(
            [record.result["data"]["output"] for record in records], ["first", "second"]
        )
        self.assertNotIn("arguments", runs[0])

    def test_details_retain_multiline_commands_and_render_real_output_lines(self):
        record = entry(command="printf 'first\\n'\nprintf 'second\\n'")
        record.run["result"] = json.dumps({"status": "ok", "data": {"output": "first\nsecond\n"}})
        detail = record.detail([record])
        self.assertIn("first\nsecond\n", detail)
        self.assertIn("printf 'first\\n'\nprintf 'second\\n'", detail)

    def test_missing_legacy_arguments_are_reported_without_guessing(self):
        record = execution_entries([{"assistant_index": 999, "tool": "exec"}], [])[0]
        self.assertIn("旧记录未保留参数", record.detail([record]))

    def test_worker_details_link_to_parent_and_keep_call_numbers(self):
        parent, worker = entry(1), entry(2)
        worker.run["trace"].update(agent_id="A1", parent_execution_id="execution-1")
        self.assertIn("来自：#1", worker.detail([parent, worker]))
        self.assertIn("子调用：#2 A1", parent.detail([parent, worker]))

    def test_artifacts_are_paged_without_losing_unicode_or_loading_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            text = "中文🙂\n" * 100
            path.write_text(text, encoding="utf-8")
            pages = ArtifactPages(path, page_chars=17)
            self.assertEqual(len(pages.text), 17)
            content = [pages.text]
            while pages.move(1):
                self.assertLessEqual(len(pages.text), 17)
                content.append(pages.text)
            self.assertEqual("".join(content), text)
            self.assertTrue(pages.move(-1))
            self.assertEqual(pages.text, content[-2])

    def test_empty_artifact_and_terminal_control_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            path.write_text("\x1b[2J\x00safe\n", encoding="utf-8")
            pages = ArtifactPages(path)
            self.assertEqual(pages.text, "safe\n")
            self.assertFalse(pages.move(1))
            path.write_text("", encoding="utf-8")
            self.assertEqual(ArtifactPages(path).text, "")


class InspectorInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def inspect(self, records, keys, **kwargs):
        with create_pipe_input() as pipe:
            with patch(
                "trace_ui.Application", wraps=__import__("prompt_toolkit").Application
            ) as factory:
                factory.side_effect = lambda **options: __import__("prompt_toolkit").Application(
                    **options, input=pipe, output=DummyOutput()
                )
                inspector = TraceInspector(records, **kwargs)
            pipe.send_text(keys)
            await inspector.application.run_async()
        return inspector

    async def test_navigation_opens_the_selected_call_and_returns(self):
        inspector = await self.inspect([entry(1), entry(2)], "\x1b[A\r\x1b")
        self.assertEqual(inspector.entry.number, 1)
        self.assertIn("#1", inspector.details.text)

    async def test_filters_preserve_absolute_numbers_and_support_errors(self):
        inspector = await self.inspect([entry(1), entry(2, status="error")], "e\x1b")
        self.assertEqual([record.number for record in inspector.filtered], [2])

    async def test_rows_mark_status_and_the_heading_counts_failures(self):
        inspector = await self.inspect([entry(1), entry(2, status="error")], "\x1b")
        heading = "".join(text for _, text in inspector._heading())
        rows = "".join(text for _, text in inspector._rows())
        self.assertIn("2/2 次调用 · 1 次异常", heading)
        self.assertIn("✓ #1  exec", rows)
        self.assertIn("✗ #2  exec", rows)
        self.assertIn("失败 · echo example", rows)
        self.assertNotIn("主代理", rows)

    async def test_search_filter_and_current_task_can_be_cleared(self):
        inspector = await self.inspect(
            [entry(1, command="first", task="old"), entry(2)], "/first\rtc\x1b"
        )
        self.assertEqual(len(inspector.filtered), 2)
        self.assertEqual(inspector.filter_buffer.text, "")

    async def test_detail_search_returns_to_details_without_closing_inspector(self):
        inspector = await self.inspect([entry()], "\r/example\r\x1b")
        self.assertEqual(inspector.details.control.search_state.text, "example")

    async def test_missing_artifact_is_visible_and_keeps_the_record(self):
        inspector = await self.inspect([entry(artifact=Path("/missing-synthetic-output"))], "o\x1b")
        self.assertIn("无法打开保留文件", inspector.notice)
        self.assertIn("#1", inspector.details.text)

    async def test_artifact_controls_load_the_next_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"
            path.write_text("first-page\n" + "x" * 32000 + "SECOND_PAGE", encoding="utf-8")
            inspector = await self.inspect([entry(artifact=path)], "o]\x1b")
            self.assertEqual(inspector.artifact.index, 1)
            self.assertIn("SECOND_PAGE", inspector.details.text)
            self.assertNotIn("first-page", inspector.details.text)

    async def test_narrow_layout_keeps_navigation_and_exit_hints_visible(self):
        class Output(DummyOutput):
            def get_size(self):
                return Size(rows=24, columns=38)

        with (
            create_pipe_input() as pipe,
            patch(
                "trace_ui.Application",
                side_effect=lambda **options: Application(**options, input=pipe, output=Output()),
            ),
        ):
            inspector = TraceInspector([entry()], colors=False)
            frames = []

            def capture(app):
                screen = app.renderer.last_rendered_screen
                if screen and not frames:
                    frames.append(
                        "\n".join(
                            "".join(screen.data_buffer[y][x].char for x in range(38))
                            for y in range(24)
                        )
                    )
                    app.exit()

            inspector.application.after_render += capture
            await inspector.application.run_async()
        self.assertIn("Enter/Tab", frames[0])
        self.assertIn("Esc 返回", frames[0])
        self.assertIn("echo example", frames[0])


class PersistedInspectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.settings = make_settings(
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=2,
        )

    def test_commands_and_results_survive_message_compaction_and_restart(self):
        call = {
            "id": "test-call",
            "type": "function",
            "function": {
                "name": "exec",
                "arguments": json.dumps({"command": "printf 'saved-output\\n'"}),
            },
        }
        client = FakeClient(
            {"role": "assistant", "tool_calls": [call]}, {"role": "assistant", "content": "done"}
        )
        console = Console(workspace=self.workspace)
        conversation = Conversation(
            self.settings, client, CodingTools(self.workspace, on_tool=console.tool)
        )
        self.addCleanup(conversation.close)
        with patch("sys.stdout", io.StringIO()) as output:
            ask(conversation, console, "inspect output")
        self.assertIn("#1", output.getvalue())
        self.assertIn("exit 0", output.getvalue())
        conversation.messages = [{"role": "user", "content": "compacted task"}]
        conversation.save()
        restored = Conversation(self.settings, FakeClient(), CodingTools(self.workspace))
        self.addCleanup(restored.close)
        record = execution_entries(restored.execution_snapshot(), restored.messages)[0]
        self.assertIn("saved-output", record.arguments["command"])
        self.assertIn("saved-output", record.result["data"]["output"])
        self.assertGreaterEqual(record.run["duration_seconds"], 0)
        with patch("sys.stdout", io.StringIO()) as output:
            handle_command(restored, console, "/tool 1")
        self.assertIn("saved-output", output.getvalue())

    def test_parallel_worker_results_are_retained_in_the_session(self):
        (self.workspace / "sample.txt").write_text("worker evidence", encoding="utf-8")

        class Client:
            def complete(self, messages, tools=None, **kwargs):
                if any(message["role"] == "tool" for message in messages):
                    return {"role": "assistant", "content": "inspection complete"}
                main = any(tool["function"]["name"] == "delegate" for tool in tools)
                arguments = (
                    {
                        "tasks": [
                            {
                                "task": f"inspect {index}",
                                "mode": "inspect",
                                "expected_output": "report",
                            }
                            for index in range(2)
                        ]
                    }
                    if main
                    else {"path": "sample.txt"}
                )
                return {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {
                                "name": "delegate" if main else "read",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }

        conversation = Conversation(self.settings, Client(), CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        conversation.ask("inspect with workers")
        records = execution_entries(conversation.sessions.read()["tool_runs"], [])
        self.assertEqual(len(records), 3)
        workers = [record for record in records if record.agent != "主代理"]
        self.assertEqual(len(workers), 2)
        for record in workers:
            self.assertEqual(record.result["data"]["content"], "worker evidence")
            self.assertIn("来自：#1", record.detail(records))

    def test_interrupted_worker_records_restore_without_repeating_the_tool(self):
        conversation = Conversation(self.settings, FakeClient(), CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        worker = entry().run
        worker["trace"].update(agent_id="A1")
        worker["state"] = "running"
        del worker["result"]
        conversation.messages = [{"role": "user", "content": "unfinished task"}]
        conversation.tool_runs = [worker]
        conversation.save()
        restored = Conversation(self.settings, FakeClient(), CodingTools(self.workspace))
        self.addCleanup(restored.close)
        record = execution_entries(restored.execution_snapshot(), restored.messages)[0]
        self.assertEqual(record.status, "unknown")
        self.assertIn("worker stopped", record.result["error"]["message"])

    def test_trace_filters_and_invalid_numbers_do_not_execute_tools(self):
        conversation = Conversation(self.settings, FakeClient(), CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        conversation.tool_runs = [entry(1).run, entry(2, status="error").run]
        console = Console()
        with patch("sys.stdout", io.StringIO()) as output:
            handle_command(conversation, console, "/trace errors")
        self.assertNotIn("#1", output.getvalue())
        self.assertIn("#2", output.getvalue())
        with patch.object(console, "error") as error, patch.object(console, "trace") as trace:
            handle_command(conversation, console, "/tool 999")
        error.assert_called_once()
        trace.assert_not_called()

    def test_live_command_output_is_bounded_and_points_to_the_full_result(self):
        output = "\n".join(f"output-{index}" for index in range(100))
        with patch("sys.stdout", io.StringIO()) as stream:
            console = Console()
            console.tool("exec", {"command": "long output"})
            console.tool_result(
                "exec", {"status": "ok", "data": {"output": output, "exit_code": 0}}
            )
        rendered = stream.getvalue()
        self.assertIn("output-0", rendered)
        self.assertIn("output-99", rendered)
        self.assertNotIn("output-50", rendered)
        self.assertIn("/tool 1", rendered)
        self.assertLess(len(rendered), 500)


@unittest.skipUnless(terminal_tests.HAS_TERMINAL_EDITOR, "PTY input is unavailable")
class TerminalTraceTests(unittest.TestCase):
    read_keys = terminal_tests.TerminalInputTests.read_keys

    def test_ctrl_t_inspects_history_and_restores_the_unsubmitted_draft(self):
        code = """
import json, sys, termios
from execution_history import ExecutionEntry
from ui import Console
console = Console()
console.execution_provider = lambda: [ExecutionEntry(1, {
    'tool': 'exec', 'state': 'completed', 'arguments': '{"command":"SAVED_COMMAND"}',
    'result': '{"status":"ok","data":{"output":"SAVED_RESULT"}}'
})]
trace = console.trace
def inspected():
    trace()
    print('TRACE_CLOSED', flush=True)
console.trace = inspected
def terminal_state():
    state = termios.tcgetattr(sys.stdin.fileno())
    state[3] &= ~getattr(termios, 'PENDIN', 0)
    return state
before = terminal_state()
with console.session_screen():
    value = console.prompt()
    screen_active = console._screen_active
after = terminal_state()
print('RESULT:' + json.dumps({'value': value, 'screen_active': screen_active,
      'terminal_restored': before == after}), flush=True)
"""
        for width in (38, 120):
            with self.subTest(width=width):
                result = self.read_keys(
                    [b"draft\x14", ("执行轨迹".encode(), b"\x1b"), (b"TRACE_CLOSED", b"\r")],
                    width=width,
                    code=code,
                )
                self.assertEqual(
                    result, {"value": "draft", "screen_active": True, "terminal_restored": True}
                )
