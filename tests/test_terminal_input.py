"""Exercise real terminal key sequences without calling the model or tools."""

import json
import os
import select
import subprocess
import sys
import time
import unittest
from collections.abc import Callable
from pathlib import Path

try:
    import fcntl
    import pty
    import readline  # noqa: F401 -- skip PTY checks on unsupported Python builds
    import struct
    import termios

    HAS_TERMINAL_EDITOR = True
except ImportError:
    HAS_TERMINAL_EDITOR = False

ROOT = Path(__file__).resolve().parents[1]
CHILD_CODE = """
import json, sys
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
try:
    value = console.prompt()
except EOFError:
    print('RESULT:' + json.dumps({'eof': True}), flush=True)
except KeyboardInterrupt:
    print('RESULT:' + json.dumps({'interrupt': True}), flush=True)
else:
    print('RESULT:' + json.dumps({'value': value}, ensure_ascii=False), flush=True)
"""

SELECT_CODE = """
import json, sys
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
value = console.select_option('SELECT_READY', [
    {'label': 'A', 'description': 'First option'},
    {'label': 'B', 'description': 'Second option'},
    {'label': 'C', 'description': 'Third option'}], default_index=1)
print('RESULT:' + json.dumps({'value': value,
    'history_untouched': console._prompt_session is None}), flush=True)
"""

ASK_CODE = """
import json, sys
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
stdout = sys.stdout
def terminal_state():
    state = termios.tcgetattr(sys.stdin.fileno())
    state[3] &= ~getattr(termios, 'PENDIN', 0)
    return state
before = terminal_state()
console.task_start('Waiting for a choice')
try:
    with console.interrupt_on_escape():
        result = console.ask_user('ASK_READY', [
            {'label': 'A', 'description': 'First option'},
            {'label': 'B', 'description': 'Second option'}])
except KeyboardInterrupt:
    result = {'paused': True}
live_restored = console._live is not None
console.task_end('paused' not in result)
print('RESULT:' + json.dumps({'result': result, 'live_restored': live_restored,
    'stdout_restored': sys.stdout is stdout,
    'history_untouched': console._prompt_session is None,
    'terminal_restored': terminal_state() == before}), flush=True)
"""

EFFORT_CODE = """
import json, sys
from types import SimpleNamespace
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from main import update_effort
from ui import Console
console = Console()
settings = SimpleNamespace(reasoning_effort='high')
conversation = SimpleNamespace(settings=settings,
    set_reasoning_effort=lambda value: setattr(settings, 'reasoning_effort', value))
update_effort(conversation, console, [])
value = console._read_input('AFTER_EFFORT')
print('RESULT:' + json.dumps({'effort': settings.reasoning_effort, 'value': value}), flush=True)
"""

WORKFLOW_CODE = """
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from main import handle_command
from commands import CommandResult
from ui import Console
from workflows import WorkflowStore
console = Console()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    store = WorkflowStore(root / 'workflows')
    store.create('Review', '1. Inspect\\n2. Report')
    conversation = SimpleNamespace(settings=SimpleNamespace(workspace=root), workflows=store)
    prompts = []
    with patch('main.ask', side_effect=lambda chat, output, prompt: prompts.append(prompt)):
        handled = handle_command(conversation, console, '/workflow')
    print('RESULT:' + json.dumps({'handled': handled is CommandResult.HANDLED, 'prompts': prompts,
        'saved': {workflow.name: workflow.instructions for workflow in store.list()},
        'history_untouched': console._prompt_session is None}, ensure_ascii=False), flush=True)
"""

STREAM_CODE = """
import json, os, sys, tempfile, time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from tests.support import make_settings
from conversation import Conversation
from deepseek_client import DeepSeekClient
from main import ask
from ui import Console
def event(delta):
    return ('data: ' + json.dumps({'choices': [{'delta': delta}]}) + '\\n').encode()
@contextmanager
def open_response(*args, **kwargs):
    def parts():
        yield event({'reasoning_content': 'FIRST_THOUGHT'})
        yield event({'content': 'FIRST_ANSWER'})
        time.sleep(1)
        yield event({'content': ' FINAL_ANSWER'})
        yield b'data: [DONE]\\n'
    yield parts()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    settings = make_settings(api_key='test', base_url='https://example.invalid', model='fake',
        session_file=root / 'session.json', workspace=root, max_context_chars=120000,
        keep_recent_turns=6, max_task_seconds=60, max_subagents=1)
    conversation = Conversation(settings, DeepSeekClient(settings), None)
    os.environ['NO_COLOR'] = '1'
    console = Console('fake')
    try:
        with patch('http_transport.open_url', side_effect=open_response):
            answer = ask(conversation, console, 'Explain the approach')
    except KeyboardInterrupt:
        answer = None
    print('RESULT:' + json.dumps({'answer': answer, 'live_closed': console._live is None,
        'ui_status': console._task_state,
        'status': conversation.last_outcome['status'],
        'checkpointed': bool(list(root.glob('.agent-artifacts/*.interrupted.json')))}), flush=True)
"""

CHOICE_COMMAND_CODE = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from main import handle_command
from commands import CommandResult
from skills import Skill
from ui import Console
console = Console()
limits, notices = [], []
original_history, original_info = console.history, console.info
def history(messages, limit):
    limits.append(limit)
    original_history(messages, limit)
def info(message):
    notices.append(message)
    original_info(message)
console.history, console.info = history, info
conversation = SimpleNamespace(messages=[
    {'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}],
    skills=SimpleNamespace(enabled=True, reload=lambda: None, diagnostics=[], skills=[
        Skill('alpha', 'Inspect API', Path('alpha/SKILL.md'), Path('alpha')),
        Skill('beta', 'Inspect tests', Path('beta/SKILL.md'), Path('beta'))]))
conversation.refresh_skills = conversation.skills.reload
command = '/history'
handled = handle_command(conversation, console, command)
print('RESULT:' + json.dumps({'handled': handled is CommandResult.HANDLED, 'limits': limits, 'notices': notices,
    'message_count': len(conversation.messages),
    'history_untouched': console._prompt_session is None}), flush=True)
"""


@unittest.skipUnless(HAS_TERMINAL_EDITOR, "requires a Unix PTY and readline")
class TerminalInputTests(unittest.TestCase):
    def read_keys(
        self,
        keys: bytes | list[bytes | tuple[bytes, bytes]],
        width: int = 32,
        *,
        code: str = CHILD_CODE,
        ready_marker: bytes = "You  ›".encode(),
        on_ready: Callable[[bytes], None] | None = None,
    ) -> dict:
        master, slave = pty.openpty()
        child = None
        output = bytearray()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
            child = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=ROOT,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                env=dict(os.environ, TERM="xterm-256color", INPUTRC=os.devnull),
            )
            os.close(slave)
            slave = None

            def read_until(marker: bytes) -> None:
                deadline = time.monotonic() + 5
                while marker not in output and time.monotonic() < deadline:
                    if select.select([master], [], [], 0.1)[0]:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError:
                            break
                        if not chunk:
                            break
                        output.extend(chunk)
                self.assertIn(marker, output, repr(output))

            read_until(ready_marker)
            if on_ready:
                on_ready(bytes(output))
            for step in [keys] if isinstance(keys, bytes) else keys:
                if isinstance(step, tuple):
                    marker, chunk = step
                    read_until(marker)
                else:
                    chunk = step
                os.write(master, chunk)
                time.sleep(0.01)
            read_until(b"RESULT:")
            # Keep draining while the child exits; a large JSON result can fill
            # the PTY output buffer and otherwise deadlock child.wait().
            deadline = time.monotonic() + 5
            while child.poll() is None and time.monotonic() < deadline:
                if not select.select([master], [], [], 0.1)[0]:
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
            # The final bytes may become readable just after process exit.
            while select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
            value = output.split(b"RESULT:", 1)[1].split(b"\r\n", 1)[0]
            return json.loads(value)
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            os.close(master)
            if slave is not None:
                os.close(slave)

    def test_backspace_in_the_middle_after_left_arrows(self):
        keys = b"abcXdef" + b"\x1b[D" * 3 + b"\x7f\r"
        self.assertEqual(self.read_keys(keys), {"value": "abcdef"})

    def test_interactive_launch_enters_a_fresh_screen_and_restores_the_terminal(self):
        code = """
import json, sys, tempfile, termios
from pathlib import Path
from unittest.mock import patch
import main
from tests.support import make_settings
from tests.test_conversation import FakeClient

class RecordingOutput:
    def __init__(self, output):
        self.output = output
        self.parts = []
    def write(self, value):
        self.parts.append(value)
        return self.output.write(value)
    def flush(self):
        self.output.flush()
    def isatty(self):
        return self.output.isatty()
    def fileno(self):
        return self.output.fileno()

before = termios.tcgetattr(sys.stdin.fileno())
output = RecordingOutput(sys.stdout)
with tempfile.TemporaryDirectory() as temporary:
    workspace = Path(temporary)
    settings = make_settings(api_key='test', base_url='https://example.invalid',
        model='test-model', workspace=workspace, session_file=workspace / 'session.json',
        max_context_chars=120000, keep_recent_turns=6, max_task_seconds=60,
        max_subagents=1, mcp_enabled=False, websearch_enabled=False, skills_enabled=False)
    with patch('main.load_settings', return_value=settings), \\
            patch('main.DeepSeekClient', return_value=FakeClient()), \\
            patch('sys.argv', ['main.py']), patch('sys.stdout', output):
        status = main.main()
captured = ''.join(output.parts)
after = termios.tcgetattr(sys.stdin.fileno())
before[3] &= ~getattr(termios, 'PENDIN', 0)
after[3] &= ~getattr(termios, 'PENDIN', 0)
print('RESULT:' + json.dumps({'status': status,
    'entered_once': captured.count('\\x1b[?1049h') == 1,
    'exited_once': captured.count('\\x1b[?1049l') == 1,
    'restored_at_exit': captured.endswith('\\x1b[0m\\x1b[?25h\\x1b[?1049l'),
    'terminal_restored': after == before}), flush=True)
"""

        def verify_fresh_screen(output):
            self.assertTrue(output.startswith(b"\x1b[?1049h\x1b[2J\x1b[H"))
            self.assertIn(b"ASTER", output)

        result = self.read_keys(b"/exit\r", code=code, width=80, on_ready=verify_fresh_screen)
        self.assertEqual(
            result,
            {
                "status": 0,
                "entered_once": True,
                "exited_once": True,
                "restored_at_exit": True,
                "terminal_restored": True,
            },
        )

    def test_forward_delete_in_the_middle(self):
        keys = b"abcXdef" + b"\x1b[D" * 4 + b"\x1b[3~\r"
        self.assertEqual(self.read_keys(keys), {"value": "abcdef"})

    def test_both_backspace_encodings(self):
        self.assertEqual(self.read_keys(b"abXY\x08\x7fcd\r"), {"value": "abcd"})

    def test_command_suggestions_appear_while_typing_and_tab_completes(self):
        result = self.read_keys([b"/eff", (b"/effort", b"\t\r")], width=80)
        self.assertEqual(result, {"value": "/effort"})

    def test_tab_completes_a_command_on_a_narrow_terminal(self):
        self.assertEqual(self.read_keys(b"/eff\t\r", width=24), {"value": "/effort"})

    def test_skill_suggestions_appear_and_tab_preserves_task_arguments(self):
        code = CHILD_CODE.replace(
            "console = Console()",
            "console = Console()\nconsole.skill_provider = lambda: {'pdf-tools': 'Extract PDF tables.'}",
        )
        result = self.read_keys(
            [b"/skill:p", (b"/skill:pdf-tools", b"\t extract report.pdf\r")],
            code=code,
            width=80,
        )
        self.assertEqual(result, {"value": "/skill:pdf-tools extract report.pdf"})

    def test_skills_argument_tab_completion_preserves_the_command(self):
        code = CHILD_CODE.replace(
            "console = Console()",
            "console = Console()\nconsole.skill_provider = lambda: {'beta': 'Inspect tests.'}",
        )
        self.assertEqual(self.read_keys(b"/skills be\t\r", code=code), {"value": "/skills beta"})

    def test_fast_tab_completion_can_cycle_before_the_menu_is_rendered(self):
        for keys, expected in (
            (b"/e\t\t\r", "/env"),
            (b"/e\t\t\t\r", "/environment"),
            (b"/e\x1b[Z\r", "/exit"),
        ):
            with self.subTest(keys=keys):
                self.assertEqual(self.read_keys(keys, width=24), {"value": expected})

    def test_completion_does_not_submit_or_modify_an_unselected_prefix(self):
        for keys, expected in ((b"\r", "/eff"), (b"\t max\r", "/effort max")):
            with self.subTest(keys=keys):
                result = self.read_keys([b"/eff", (b"/effort", keys)], width=80)
                self.assertEqual(result, {"value": expected})

    def test_tab_shift_tab_and_arrows_navigate_command_candidates(self):
        for keys, expected in (
            (b"\t\t\r", "/env"),
            (b"\t\t\x1b[Z\r", "/effort"),
            (b"\x1b[B\x1b[B\r", "/env"),
            (b"\x1b[B\x1b[B\x1b[A\r", "/effort"),
        ):
            with self.subTest(keys=keys):
                result = self.read_keys([b"/e", (b"/effort", keys)], width=80)
                self.assertEqual(result, {"value": expected})

    def test_escape_dismisses_command_completion_without_exiting(self):
        result = self.read_keys([b"/eff", (b"/effort", b"\x1b\r")], width=80)
        self.assertEqual(result, {"value": "/eff"})

    def test_custom_answer_editor_does_not_complete_commands(self):
        code = CHILD_CODE.replace("console.prompt()", "console.read_value('  You  › ')")
        self.assertEqual(self.read_keys(b"/eff\t\r", code=code), {"value": "/eff"})

    def test_arrow_selection_and_confirmation_use_the_current_default(self):
        for keys, expected in [(b"\r", 1), (b"\x1b[A\r", 0), (b"\x1b[B\r", 2)]:
            with self.subTest(keys=keys):
                result = self.read_keys(keys, code=SELECT_CODE, ready_marker=b"SELECT_READY")
                self.assertEqual(result, {"value": expected, "history_untouched": True})

    def test_up_and_down_wrap_at_the_ends_of_the_menu(self):
        for keys, expected in [(b"\x1b[A\x1b[A\r", 2), (b"\x1b[B\x1b[B\r", 0)]:
            with self.subTest(keys=keys):
                result = self.read_keys(keys, code=SELECT_CODE, ready_marker=b"SELECT_READY")
                self.assertEqual(result["value"], expected)

    def test_escape_ctrl_c_and_ctrl_d_dismiss_the_menu(self):
        for keys in (b"\x1b", b"\x03", b"\x04"):
            with self.subTest(keys=keys):
                result = self.read_keys(keys, code=SELECT_CODE, ready_marker=b"SELECT_READY")
                self.assertIsNone(result["value"])
                self.assertTrue(result["history_untouched"])

    def test_arrow_choice_resumes_an_active_task_and_restores_the_terminal(self):
        result = self.read_keys(b"\x1b[B\r", code=ASK_CODE, ready_marker=b"ASK_READY")
        self.assertEqual(
            result,
            {
                "result": {
                    "answer": "B",
                    "selected_index": 2,
                    "selected_label": "B",
                    "custom": False,
                },
                "live_restored": True,
                "stdout_restored": True,
                "history_untouched": True,
                "terminal_restored": True,
            },
        )

    def test_last_arrow_choice_opens_the_custom_answer_editor(self):
        result = self.read_keys(
            [b"\x1b[B\x1b[B\r", ("请输入你的答案".encode(), "我的方案\r".encode())],
            code=ASK_CODE,
            ready_marker=b"ASK_READY",
        )
        self.assertEqual(result["result"]["answer"], "我的方案")
        self.assertEqual(result["result"]["selected_index"], 3)
        self.assertTrue(result["result"]["custom"])
        self.assertTrue(result["live_restored"])
        self.assertTrue(result["stdout_restored"])
        self.assertTrue(result["history_untouched"])
        self.assertTrue(result["terminal_restored"])

    def test_escape_in_the_custom_editor_cancels_the_active_question(self):
        result = self.read_keys(
            [b"\x1b[B\x1b[B\r", ("请输入你的答案".encode(), b"\x1b")],
            code=ASK_CODE,
            ready_marker=b"ASK_READY",
        )
        self.assertEqual(result["result"], {"paused": True})
        self.assertTrue(result["live_restored"])
        self.assertTrue(result["history_untouched"])
        self.assertTrue(result["terminal_restored"])

    def test_escape_on_an_active_question_pauses_without_leaking_terminal_state(self):
        result = self.read_keys(b"\x1b", code=ASK_CODE, ready_marker=b"ASK_READY")
        self.assertEqual(
            result,
            {
                "result": {"paused": True},
                "live_restored": True,
                "stdout_restored": True,
                "history_untouched": True,
                "terminal_restored": True,
            },
        )

    def test_effort_selection_and_cancellation_return_to_a_clean_chat_editor(self):
        for keys, expected in [(b"\x1b[B\r", "low"), (b"\x1b", "high")]:
            with self.subTest(keys=keys):
                result = self.read_keys(
                    [keys, (b"AFTER_EFFORT", b"\x1b[Ahello\r")],
                    code=EFFORT_CODE,
                    ready_marker="选择推理强度".encode(),
                )
                self.assertEqual(result, {"effort": expected, "value": "hello"})

    def test_workflow_menu_confirms_running_with_optional_task_context(self):
        result = self.read_keys(
            [b"\r", ("本次任务补充".encode(), "检查登录\r".encode())],
            code=WORKFLOW_CODE,
            width=80,
            ready_marker="选择 workflow".encode(),
        )

        self.assertTrue(result["handled"])
        self.assertEqual(len(result["prompts"]), 1)
        self.assertIn("1. Inspect\n2. Report", result["prompts"][0])
        self.assertIn("检查登录", result["prompts"][0])
        self.assertTrue(result["history_untouched"])

    def test_workflow_menu_can_create_multiline_steps_without_running(self):
        instructions = "1. 阅读代码\n2. 检查风险\n3. 总结问题"
        result = self.read_keys(
            [
                b"\x1b[B\r",
                ("Workflow 名称".encode(), "代码审查\r".encode()),
                (
                    "Workflow 步骤".encode(),
                    b"\x1b[200~" + instructions.encode() + b"\x1b[201~\r",
                ),
            ],
            code=WORKFLOW_CODE,
            width=80,
            ready_marker="选择 workflow".encode(),
        )

        self.assertTrue(result["handled"])
        self.assertEqual(result["prompts"], [])
        self.assertEqual(result["saved"]["代码审查"], instructions)
        self.assertTrue(result["history_untouched"])

    def test_workflow_menu_cancellation_and_escape_in_creation_do_not_run(self):
        for keys in (
            b"\x1b",
            [b"\x1b[B\r", ("Workflow 名称".encode(), b"\x1b")],
            [
                b"\x1b[B\r",
                ("Workflow 名称".encode(), b"New\r"),
                ("Workflow 步骤".encode(), b"\x1b"),
            ],
            [b"\r", ("本次任务补充".encode(), b"\x1b")],
        ):
            with self.subTest(keys=keys):
                result = self.read_keys(
                    keys,
                    code=WORKFLOW_CODE,
                    width=80,
                    ready_marker="选择 workflow".encode(),
                )
                self.assertEqual(result["prompts"], [])
                self.assertEqual(result["saved"], {"Review": "1. Inspect\n2. Report"})
                self.assertTrue(result["history_untouched"])

    def test_workflow_command_can_be_tab_completed(self):
        self.assertEqual(self.read_keys(b"/work\t\r"), {"value": "/workflow"})

    def test_streaming_answer_is_visible_before_the_model_finishes(self):
        result = self.read_keys(
            b"continue\r", width=80, code=STREAM_CODE, ready_marker=b"FIRST_ANSWER"
        )
        self.assertEqual(result["answer"], "FIRST_ANSWER FINAL_ANSWER")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["live_closed"])

    def test_escape_during_a_stream_preserves_the_partial_response(self):
        result = self.read_keys(b"\x1b", width=80, code=STREAM_CODE, ready_marker=b"FIRST_ANSWER")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["ui_status"], "paused")
        self.assertTrue(result["live_closed"])
        self.assertTrue(result["checkpointed"])

    def test_arrow_keys_do_not_interrupt_a_streaming_response(self):
        result = self.read_keys(
            b"\x1b[A\x1b[B\x1bOA\x1bOB\x1b[1;3A\x1b[1;5B",
            width=80,
            code=STREAM_CODE,
            ready_marker=b"FIRST_ANSWER",
        )
        self.assertEqual(result["answer"], "FIRST_ANSWER FINAL_ANSWER")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["live_closed"])
        self.assertFalse(result["checkpointed"])

    def test_fragmented_arrow_sequences_do_not_become_escape_interruptions(self):
        result = self.read_keys(
            [b"\x1b", b"[", b"B", b"\x1b", b"O", b"A"],
            width=80,
            code=STREAM_CODE,
            ready_marker=b"FIRST_ANSWER",
        )
        self.assertEqual(result["answer"], "FIRST_ANSWER FINAL_ANSWER")
        self.assertEqual(result["status"], "completed")

    def test_markdown_formats_before_the_model_finishes_with_and_without_color(self):
        first = "## FIRST_ANSWER\n\n- **STREAM_BOLD**\n\n```python\nprint('STREAM_CODE')\n"
        last = "```\n\nFINAL_ANSWER"
        code = STREAM_CODE.replace("'FIRST_ANSWER'", repr(first)).replace(
            "' FINAL_ANSWER'", repr(last)
        )

        def assert_formatted(output):
            self.assertIn(b"STREAM_BOLD", output)
            self.assertIn(b"STREAM_CODE", output)
            self.assertNotIn(b"**STREAM_BOLD**", output)
            self.assertNotIn(b"## FIRST_ANSWER", output)
            self.assertNotIn(b"```python", output)
            self.assertNotIn(b"FINAL_ANSWER", output)

        for color in (False, True):
            with self.subTest(color=color):
                source = (
                    code.replace("os.environ['NO_COLOR'] = '1'", "os.environ.pop('NO_COLOR', None)")
                    if color
                    else code
                )
                result = self.read_keys(
                    b"continue\r",
                    width=80,
                    code=source,
                    ready_marker=b"STREAM_CODE",
                    on_ready=assert_formatted,
                )
                self.assertEqual(result["answer"], first + last)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["live_closed"])

    def test_history_command_accepts_an_arrow_selected_range(self):
        result = self.read_keys(
            b"\x1b[B\r", width=80, code=CHOICE_COMMAND_CODE, ready_marker="选择历史范围".encode()
        )
        self.assertTrue(result["handled"])
        self.assertEqual(result["limits"], [20])
        self.assertEqual(result["message_count"], 2)
        self.assertTrue(result["history_untouched"])

    def test_history_custom_count_accepts_or_cancels_without_touching_chat_history(self):
        for keys, expected in [(b"37\r", [37]), (b"\x1b", [])]:
            with self.subTest(keys=keys):
                result = self.read_keys(
                    [b"\x1b[B" * 4 + b"\r", ("对话条数".encode(), keys)],
                    width=80,
                    code=CHOICE_COMMAND_CODE,
                    ready_marker="选择历史范围".encode(),
                )
                self.assertTrue(result["handled"])
                self.assertEqual(result["limits"], expected)
                self.assertTrue(result["history_untouched"])

    def test_skills_command_accepts_an_arrow_selected_skill_or_cancellation(self):
        code = CHOICE_COMMAND_CODE.replace("command = '/history'", "command = '/skills'")
        for keys, expected in [
            (b"\x1b[B\x1b[B\r", "beta: Inspect tests"),
            (b"\x1b", "已取消查看技能。"),
        ]:
            with self.subTest(keys=keys):
                result = self.read_keys(keys, width=80, code=code, ready_marker="选择技能".encode())
                self.assertTrue(result["handled"])
                self.assertIn(expected, result["notices"])
                self.assertEqual(result["message_count"], 2)
                self.assertTrue(result["history_untouched"])

    def test_chinese_backspace_across_wrapped_lines(self):
        text = "这是检查长段中文退格删除是否正常的一段文字。" * 5
        keys = text.encode() + b"\x7f" * 35 + "改好了".encode() + b"\r"
        self.assertEqual(self.read_keys(keys), {"value": text[:-35] + "改好了"})

    def test_chinese_edit_in_the_middle(self):
        keys = "你好错世界".encode() + b"\x1b[D" * 2 + b"\x7f" + "新".encode() + b"\r"
        self.assertEqual(self.read_keys(keys), {"value": "你好新世界"})

    def test_home_end_and_delete(self):
        keys = b"XhelloY\x1b[H\x1b[3~\x1b[F\x7f\r"
        self.assertEqual(self.read_keys(keys), {"value": "hello"})

    def test_long_ascii_can_be_cleared_and_retyped(self):
        keys = b"long-text-" * 30 + b"\x7f" * 300 + b"retyped\r"
        self.assertEqual(self.read_keys(keys, width=24), {"value": "retyped"})

    def test_multiline_bracketed_paste_is_submitted_as_one_prompt(self):
        pasted = "第一行\nsecond line\n最后一行"
        keys = b"\x1b[200~" + pasted.encode() + b"\x1b[201~\r"

        self.assertEqual(self.read_keys(keys), {"value": pasted})

    def test_trailing_backslash_continues_the_prompt_on_a_new_line(self):
        keys = [b"first \\", (b"first", b"\r"), (b"\\", b"second\r")]
        self.assertEqual(self.read_keys(keys, width=60), {"value": "first \nsecond"})

    def test_answers_and_paths_keep_a_trailing_backslash(self):
        code = CHILD_CODE.replace("console.prompt()", "console.read_value('  You  › ')")
        self.assertEqual(self.read_keys(b"C:\\work\\\r", code=code), {"value": "C:\\work\\"})

    def test_chunked_bracketed_paste_is_not_truncated(self):
        pasted = "\n".join(f"第 {index} 行：" + "内容" * 20 for index in range(24))
        payload = pasted.encode()
        keys = [
            b"\x1b[20",
            b"0~" + payload[:137],
            payload[137:913],
            payload[913:] + b"\x1b[20",
            b"1~\r",
        ]

        self.assertEqual(self.read_keys(keys, width=40), {"value": pasted})

    def test_ctrl_d_still_exits(self):
        self.assertEqual(self.read_keys(b"\x04"), {"eof": True})

    def test_ctrl_c_still_interrupts(self):
        self.assertEqual(self.read_keys(b"not submitted\x03"), {"interrupt": True})

    def test_ctrl_c_returns_to_the_chat_prompt_without_exiting(self):
        code = """
import json, sys, termios
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
if sys.stdin.isatty():
    import fcntl
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from main import run_interactive
from ui import Console
console = Console()
before = termios.tcgetattr(sys.stdin.fileno())
original_prompt = console.prompt
prompt_count = 0
def prompt():
    global prompt_count
    prompt_count += 1
    if prompt_count > 1:
        print('CHAT_REOPENED', flush=True)
    return original_prompt()
console.prompt = prompt
conversation = SimpleNamespace(settings=SimpleNamespace(session_file=Path('session.json')),
    messages=[], tools=None, skills=None, last_environment_id=None,
    environment_id=None, recovery_notice=None, context_status=lambda: {})
with patch('main.sync_environment_ui'), patch('main.show_skill_diagnostics'):
    run_interactive(conversation, console)
after = termios.tcgetattr(sys.stdin.fileno())
before[3] &= ~getattr(termios, 'PENDIN', 0)
after[3] &= ~getattr(termios, 'PENDIN', 0)
print('RESULT:' + json.dumps({'prompt_count': prompt_count,
    'terminal_restored': after == before}), flush=True)
"""
        result = self.read_keys([b"not submitted\x03", (b"CHAT_REOPENED", b"/exit\r")], code=code)
        self.assertEqual(result, {"prompt_count": 2, "terminal_restored": True})

    def test_ctrl_c_can_interrupt_while_escape_pause_is_enabled(self):
        code = """
import json, signal, sys, termios, time
if sys.stdin.isatty():
    import fcntl
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
before = termios.tcgetattr(sys.stdin.fileno())
handler = signal.getsignal(signal.SIGQUIT)
paused = False
try:
    with console.interrupt_on_escape():
        print('TASK_READY', flush=True)
        time.sleep(10)
except KeyboardInterrupt:
    paused = True
after = termios.tcgetattr(sys.stdin.fileno())
before[3] &= ~getattr(termios, 'PENDIN', 0)
after[3] &= ~getattr(termios, 'PENDIN', 0)
print('RESULT:' + json.dumps({'paused': paused,
    'terminal_restored': after == before,
    'handler_restored': signal.getsignal(signal.SIGQUIT) == handler}), flush=True)
"""
        result = self.read_keys(b"\x03", code=code, ready_marker=b"TASK_READY")
        self.assertEqual(
            result, {"paused": True, "terminal_restored": True, "handler_restored": True}
        )

    def test_task_navigation_is_ignored_and_chat_history_remains_editable(self):
        code = """
import json, sys, time
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
first = console.prompt()
paused = False
try:
    with console.interrupt_on_escape():
        print('TASK_READY', flush=True)
        time.sleep(0.5)
except KeyboardInterrupt:
    paused = True
print('TASK_DONE', flush=True)
previous = console.prompt()
print('HISTORY_READY', flush=True)
draft = console.prompt()
print('RESULT:' + json.dumps({'first': first, 'previous': previous,
    'draft': draft, 'paused': paused}), flush=True)
"""
        result = self.read_keys(
            [
                b"/status\r",
                (b"TASK_READY", b"\x1b[A\x1b[B"),
                (b"TASK_DONE", b"\x1b[A\r"),
                (b"HISTORY_READY", b"draft\x1b[A\x1b[B\r"),
            ],
            code=code,
        )
        self.assertEqual(
            result, {"first": "/status", "previous": "/status", "draft": "draft", "paused": False}
        )

    def test_live_task_renders_todos_and_agents_and_restores_stdout(self):
        code = """
import json, sys
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
console.prompt()
stdout = sys.stdout
console.task_start('检查界面')
console.task_update({'phase': 'plan', 'title': '检查界面', 'steps': [
    {'step': '检查状态', 'status': 'in_progress'}
]})
console.task_update({'phase': 'subagent', 'agent_id': 'A1', 'title': '读取界面',
    'state': 'running', 'activity': 'tool', 'tool': 'read', 'arguments': {'path': 'ui.py'}})
console.assistant_start()
console.stream('主代理结果')
console.assistant_end()
console.task_update({'phase': 'subagent', 'agent_id': 'A1', 'title': '读取界面',
    'state': 'completed', 'activity': 'finished', 'tool_calls': 1, 'model_requests': 2})
console.task_end(True)
print('RESULT:' + json.dumps({'stdout_restored': sys.stdout is stdout,
    'live_closed': console._live is None, 'agent_state': console._agents['A1']['state'],
    'todo_state': console._todos[0]['status']}), flush=True)
"""
        self.assertEqual(
            self.read_keys(b"start\r", width=80, code=code),
            {
                "stdout_restored": True,
                "live_closed": True,
                "agent_state": "completed",
                "todo_state": "in_progress",
            },
        )

    def test_new_session_in_a_terminal_discards_input_history(self):
        code = """
import json, sys
from pathlib import Path
if sys.stdin.isatty():
    import fcntl, termios
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
console.prompt()
old = console._prompt_session
console.reset_session(Path('session.json'))
value = console._read_input('NEW_HISTORY_READY')
print('RESULT:' + json.dumps({'value': value, 'new_input_session': console._prompt_session is not old}), flush=True)
"""
        self.assertEqual(
            self.read_keys([b"old conversation\r", b"\x1b[Anew\r"], code=code),
            {
                "value": "new",
                "new_input_session": True,
            },
        )

    def test_escape_interrupts_active_task_and_restores_ctrl_c(self):
        code = """
import json, sys, termios, time
if sys.stdin.isatty():
    import fcntl
    fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
from ui import Console
console = Console()
fd = sys.stdin.fileno()
console.prompt()
before = termios.tcgetattr(fd)[6][termios.VINTR]
paused = False
console.task_start('可暂停任务')
try:
    with console.interrupt_on_escape():
        print('TASK_READY', flush=True)
        time.sleep(10)
except KeyboardInterrupt:
    paused = True
finally:
    console.task_end(False)
after = termios.tcgetattr(fd)[6][termios.VINTR]
def byte(value):
    return value[0] if isinstance(value, bytes) else value
print('RESULT:' + json.dumps({'paused': paused, 'restored': byte(before) == byte(after),
    'live_closed': console._live is None}), flush=True)
"""
        master, slave = pty.openpty()
        child = None
        output = bytearray()
        try:
            child = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=ROOT,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                env=dict(os.environ, TERM="xterm-256color"),
            )
            os.close(slave)
            slave = None
            deadline = time.monotonic() + 5
            while "You  ›".encode() not in output and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    output.extend(os.read(master, 65536))
            self.assertIn("You  ›".encode(), output, repr(output))
            os.write(master, b"start\r")

            deadline = time.monotonic() + 5
            while b"TASK_READY" not in output and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    output.extend(os.read(master, 65536))
            self.assertIn(b"TASK_READY", output, repr(output))

            os.write(master, b"\x1b")
            deadline = time.monotonic() + 5
            while b"RESULT:" not in output and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        output.extend(os.read(master, 65536))
                    except OSError:
                        break
            self.assertIn(b"RESULT:", output, repr(output))
            child.wait(timeout=5)
            value = output.split(b"RESULT:", 1)[1].split(b"\r\n", 1)[0]
            self.assertEqual(
                json.loads(value), {"paused": True, "restored": True, "live_closed": True}
            )
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            os.close(master)
            if slave is not None:
                os.close(slave)


class RedirectedInputTests(unittest.TestCase):
    def test_piped_input_still_works(self):
        result = subprocess.run(
            [sys.executable, "-c", CHILD_CODE],
            cwd=ROOT,
            input="  通过管道输入中文  \n",
            text=True,
            capture_output=True,
            check=True,
        )
        value = json.loads(result.stdout.split("RESULT:", 1)[1])
        self.assertEqual(value, {"value": "通过管道输入中文"})


if __name__ == "__main__":
    unittest.main()
