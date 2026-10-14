import io
import json
import socketserver
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import http_transport
from config import Settings
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from main import run_interactive
from reasoning import ReasoningTools
from task_acceptance import TaskContract
from tests.support import make_settings
from tool_runtime import ExecutionContext, ToolResult
from tools import CodingTools
from ui import Console


def call(name, arguments, identity="call-1"):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def stream(delta, finish="stop"):
    return io.BytesIO(
        (
            "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n"
        ).encode()
    )


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = make_settings(
            api_key="test-placeholder",
            base_url="https://example.invalid",
            model="test",
            session_file=self.root / "session.json",
            workspace=self.root,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            request_retries=1,
            retry_backoff_seconds=0,
        )

    def test_interactive_client_retries_503_with_identical_messages(self):
        error = urllib.error.HTTPError(
            "https://example.invalid", 503, "test", {}, io.BytesIO(b"secret body")
        )
        with patch(
            "http_transport.open_url", side_effect=[error, stream({"content": "done"})]
        ) as opened:
            answer = Conversation(self.settings, DeepSeekClient(self.settings), None).ask("task")
        self.assertEqual(answer, "done")
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(
            opened.call_args_list[0].args[0].data, opened.call_args_list[1].args[0].data
        )

    def test_backoff_can_be_cancelled(self):
        cancelled = threading.Event()
        timer = threading.Timer(0.05, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        client = DeepSeekClient(replace(self.settings, retry_backoff_seconds=5))
        started = time.monotonic()
        with (
            patch.object(
                client,
                "_complete_once",
                side_effect=CompletionError("http_503", "temporary", retryable=True),
            ) as request,
            self.assertRaises(CompletionError) as caught,
        ):
            client.complete([], cancelled=cancelled)
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(request.call_count, 1)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_cancellation_closes_a_connecting_socket_before_sending_credentials(self):
        cancelled = threading.Event()
        released = threading.Event()
        fake_socket = Mock()
        fake_socket.connect.side_effect = lambda _: released.wait(2)
        fake_socket.shutdown.side_effect = lambda _: released.set()
        timer = threading.Timer(0.05, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with (
            patch(
                "http_transport.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 1))]
            ),
            patch("http_transport.socket.socket", return_value=fake_socket),
            self.assertRaises(http_transport.TransportCancelled),
        ):
            http_transport.open_url(
                urllib.request.Request("http://example.invalid"), cancelled=cancelled
            )
        self.assertTrue(released.is_set())
        fake_socket.sendall.assert_not_called()
        self.assertLess(time.monotonic() - started, 0.5)

    def test_timed_out_batches_do_not_create_more_active_workers(self):
        release = threading.Event()
        finished = threading.Event()
        calls = []

        class Client:
            def complete(inner, *args, **kwargs):
                calls.append(kwargs)
                try:
                    release.wait(2)
                    return {"role": "assistant", "content": "done"}
                finally:
                    finished.set()

        tools = CodingTools(self.root)
        tools.attach_reasoning(Client(), max_workers=1)
        request = call("delegate", {"tasks": [{"task": "one", "expected_output": "report"}]})
        try:
            results = [
                json.loads(tools.execute(request, deadline=time.monotonic() + 0.06))
                for _ in range(3)
            ]
            self.assertEqual([result["status"] for result in results], ["timeout"] * 3)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]["cancelled"].is_set())
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_completed_report_survives_deadline_during_sibling_submission(self):
        reasoning = ReasoningTools(Mock(), Mock(), [], max_workers=2, workspace=self.root)
        completed = Future()
        completed.set_result(ToolResult(data={"report": "verified finding"}))
        tasks = [
            {"task": "fast", "expected_output": "report"},
            {"task": "waiting", "expected_output": "report"},
        ]
        with patch.object(reasoning, "submit", side_effect=[completed, TimeoutError("deadline")]):
            result = reasoning.delegate(tasks, _context=ExecutionContext(time.monotonic() + 10))
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["reports"][0]["data"]["report"], "verified finding")
        self.assertEqual(result.data["reports"][1]["status"], "timeout")

    def test_restart_closes_unknown_call_without_replaying_side_effect(self):
        client = Mock(deadline=None, recorder=None)
        client.complete.return_value = {
            "role": "assistant",
            "tool_calls": [call("write", {"path": "effect.txt", "content": "once"})],
        }

        class CrashTools(CodingTools):
            def execute(inner, *args, **kwargs):
                super().execute(*args, **kwargs)
                raise SystemExit("crash after side effect")

        with self.assertRaises(SystemExit):
            Conversation(self.settings, client, CrashTools(self.root)).ask("task")
        reloaded = Conversation(self.settings, client, CodingTools(self.root))
        self.assertEqual((self.root / "effect.txt").read_text(), "once")
        self.assertEqual(reloaded.messages[-1]["role"], "tool")
        self.assertEqual(
            json.loads(reloaded.messages[-1]["content"])["error"]["code"], "execution_unknown"
        )
        self.assertEqual(reloaded.tool_runs[-1]["state"], "unknown")
        self.assertEqual(client.complete.call_count, 1)
        again = Conversation(self.settings, client, CodingTools(self.root))
        self.assertEqual(again.messages, reloaded.messages)

    def test_pending_calls_are_recorded_as_not_executed(self):
        tool = call("write", {"path": "never.txt", "content": "x"})
        self.settings.session_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "messages": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "tool_calls": [tool]},
                    ],
                    "tool_runs": [
                        {"assistant_index": 1, "tool_call_id": "call-1", "state": "pending"}
                    ],
                }
            )
        )
        conversation = Conversation(self.settings, Mock(), CodingTools(self.root))
        self.assertEqual(
            json.loads(conversation.messages[-1]["content"])["error"]["code"], "not_executed"
        )
        self.assertFalse((self.root / "never.txt").exists())

    def test_ctrl_c_saves_partial_work_and_returns_to_prompt(self):
        class InterruptedStream:
            def __enter__(inner):
                return inner

            def __exit__(inner, *args):
                pass

            def __iter__(inner):
                yield stream({"reasoning": "useful partial work"}, None).getvalue()
                raise KeyboardInterrupt

        console = Mock(spec=Console)
        console.prompt.side_effect = ["task", "/exit"]
        conversation = Conversation(self.settings, DeepSeekClient(self.settings), None)
        with patch("http_transport.open_url", return_value=InterruptedStream()):
            run_interactive(conversation, console)
        self.assertEqual(console.prompt.call_count, 2)
        self.assertIn("useful partial work", self.settings.session_file.read_text())
        self.assertEqual(conversation.last_outcome["status"], "incomplete")

    def test_promise_only_response_is_repaired_before_publication(self):
        client = Mock()
        client.complete.side_effect = [
            {"role": "assistant", "content": "我会先检查文件，然后修改代码并运行测试。"},
            {"role": "assistant", "tool_calls": [call("ls", {})]},
            {
                "role": "assistant",
                "content": "工作区没有待修复的源文件，无法完成修改；请提供目标文件。",
            },
        ]
        # Mock attributes otherwise masquerade as deadlines/recorders.
        client.deadline = None
        client.recorder = None
        output = []
        conversation = Conversation(self.settings, client, CodingTools(self.root))
        answer = conversation.ask("修复代码", on_content=output.append)
        self.assertEqual(output, [answer])
        self.assertEqual(client.complete.call_count, 3)
        self.assertEqual(conversation.last_outcome["status"], "blocked")

    def test_task_check_prevents_false_completion_and_accepts_actual_fix(self):
        target = self.root / "answer.txt"

        def verify_file():
            return (
                None
                if target.exists() and target.read_text() == "42"
                else "answer.txt must contain 42"
            )

        client = Mock(deadline=None, recorder=None)
        client.complete.side_effect = [
            {
                "role": "assistant",
                "tool_calls": [call("write", {"path": "answer.txt", "content": "wrong"})],
            },
            {"role": "assistant", "content": "done"},
            {
                "role": "assistant",
                "tool_calls": [call("write", {"path": "answer.txt", "content": "42"}, "call-2")],
            },
            {"role": "assistant", "content": "fixed"},
        ]
        conversation = Conversation(self.settings, client, CodingTools(self.root))
        output = []
        conversation.ask(
            "write answer",
            on_content=output.append,
            task_contract=TaskContract("change", (verify_file,), "Create answer.txt containing 42"),
        )
        self.assertEqual(output, ["fixed"])
        self.assertEqual(conversation.last_outcome["status"], "completed")
        self.assertTrue(conversation.last_outcome["checks"][0]["passed"])

    def test_translation_of_a_future_sentence_is_an_ordinary_answer(self):
        client = Mock(deadline=None, recorder=None)
        client.complete.return_value = {
            "role": "assistant",
            "content": "I will fix the code tomorrow.",
        }
        conversation = Conversation(self.settings, client, CodingTools(self.root))
        self.assertEqual(
            conversation.ask("翻译这句话：我明天会修复代码。"), "I will fix the code tomorrow."
        )
        self.assertEqual(client.complete.call_count, 1)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.release = threading.Event()
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append(("redirected", self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                owner.requests.append((self.path, self.headers.get("Authorization")))
                if self.path.startswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location", f"http://localhost:{self.server.server_port}/sink")
                    self.end_headers()
                    return
                if self.path.startswith("/headers"):
                    owner.release.wait(2)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    if self.path.startswith("/cancel"):
                        self.wfile.write(
                            stream({"reasoning": "partial evidence " * 30}, None).getvalue()
                        )
                        self.wfile.flush()
                        owner.release.wait(2)
                    else:
                        while not owner.release.wait(0.01):
                            self.wfile.write(b":")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.addCleanup(self.cleanup_server)
        self.settings = Settings(
            api_key="fake-private-key",
            base_url=self.url("redirect"),
            model="test",
            session_file=Path("unused"),
            workspace=Path("."),
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            request_retries=0,
        )

    def url(self, route):
        return f"http://127.0.0.1:{self.server.server_port}/{route}"

    def cleanup_server(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def test_redirect_is_rejected_without_forwarding_credentials(self):
        with self.assertRaises(CompletionError) as caught:
            DeepSeekClient(self.settings).complete([])
        self.assertEqual(caught.exception.code, "http_302")
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("fake-private-key", str(caught.exception))

    def test_deadline_interrupts_headers_and_unterminated_stream_lines(self):
        for route in ("headers", "trickle"):
            with self.subTest(route=route):
                started = time.monotonic()
                with self.assertRaises(CompletionError) as caught:
                    DeepSeekClient(replace(self.settings, base_url=self.url(route))).complete(
                        [], on_content=lambda _: None, deadline=started + 0.1
                    )
                self.assertEqual(caught.exception.code, "task_deadline")
                self.assertLess(time.monotonic() - started, 0.4)

    def test_cancellation_closes_stream_and_keeps_partial_evidence(self):
        cancelled = threading.Event()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        events = []
        with self.assertRaises(CompletionError) as caught:
            DeepSeekClient(replace(self.settings, base_url=self.url("cancel"))).complete(
                [], on_content=lambda _: None, cancelled=cancelled, on_progress=events.append
            )
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertIn("partial evidence", caught.exception.partial_message["reasoning"])
        self.assertTrue(any(event.get("received_bytes", 0) > 0 for event in events))

    def test_deadline_interrupts_tls_handshake(self):
        release = self.release

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                release.wait(2)

        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(CompletionError) as caught:
                DeepSeekClient(
                    replace(self.settings, base_url=f"https://127.0.0.1:{server.server_address[1]}")
                ).complete([], on_content=lambda _: None, deadline=started + 0.1)
            self.assertEqual(caught.exception.code, "task_deadline")
            self.assertLess(time.monotonic() - started, 0.4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(1)


if __name__ == "__main__":
    unittest.main()
