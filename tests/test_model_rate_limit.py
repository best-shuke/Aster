import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from email.utils import formatdate
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_client import CompletionError, DeepSeekClient
from model_rate_limit import RateLimitGate, retry_after_seconds
from reasoning import SubagentRunner
from task_runtime import TaskRuntime
from tests.support import make_settings
from tool_runtime import ExecutionContext, ToolResult
from worker_checkpoint import WorkerCheckpoint


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AdvancingEvent:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.clock.advance(seconds)
        return self.stopped


def settings(**changes):
    return SimpleNamespace(
        **{
            "base_url": "https://rate-limit-fixture.invalid/v1",
            "api_key": "synthetic-rate-limit-key",
            "model": "fixture",
            "max_output_tokens": 8192,
            "request_retries": 3,
            "retry_backoff_seconds": 2,
            **changes,
        }
    )


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gate = RateLimitGate(clock=self.clock)

    def test_retry_after_supports_seconds_and_http_dates_without_accepting_bad_values(self):
        self.assertEqual(retry_after_seconds("120"), 120)
        self.assertEqual(retry_after_seconds(formatdate(1300, usegmt=True), now=1000), 300)
        self.assertEqual(retry_after_seconds(formatdate(900, usegmt=True), now=1000), 0)
        for value in (None, "", "-1", "1.5", "nan", "inf", "junk", "9" * 400, "１２", 30):
            with self.subTest(value=value):
                self.assertIsNone(retry_after_seconds(value))

    def test_healthy_requests_are_admitted_without_delay_or_a_concurrency_cap(self):
        for _ in range(25):
            admission, delay, probe = self.gate.try_start()
            self.assertIsNotNone(admission)
            self.assertEqual(delay, 0)
            self.assertFalse(probe)

    def test_burst_failures_share_one_cooldown_and_only_one_recovery_probe(self):
        admissions = [self.gate.try_start()[0] for _ in range(13)]
        for admission in admissions:
            self.gate.limited(admission, backoff=2)
        self.assertEqual(self.gate.try_start()[1], 2)
        self.clock.advance(2)
        probe = self.gate.try_start()[0]
        self.assertIsNotNone(probe)
        for _ in range(12):
            admission, _, waiting_for_probe = self.gate.try_start()
            self.assertIsNone(admission)
            self.assertTrue(waiting_for_probe)
        self.gate.accepted(probe)
        self.assertIsNone(self.gate.try_start()[0])
        self.clock.advance(2)
        resumed = self.gate.try_start()[0]
        self.assertIsNotNone(resumed)
        self.assertIsNone(self.gate.try_start()[0])
        self.gate.accepted(resumed)

    def test_repeated_failed_probes_expand_cooldown_but_stale_success_does_not_clear_it(self):
        failed, stale = self.gate.try_start()[0], self.gate.try_start()[0]
        self.gate.limited(failed, backoff=2)
        self.gate.accepted(stale)
        self.assertEqual(self.gate.try_start()[1], 2)
        self.clock.advance(2)
        probe = self.gate.try_start()[0]
        self.gate.limited(probe, backoff=2)
        self.assertEqual(self.gate.try_start()[1], 4)
        self.gate.accepted(probe)
        self.assertEqual(self.gate.try_start()[1], 4)

    def test_long_server_cooldown_is_not_clipped_and_late_headers_cannot_shorten_it(self):
        first, later = self.gate.try_start()[0], self.gate.try_start()[0]
        self.gate.limited(first, retry_after=1200)
        self.clock.advance(10)
        self.gate.limited(later, retry_after=5)
        self.assertEqual(self.gate.try_start()[1], 1190)

    def test_releasing_an_old_probe_cannot_release_a_new_probe(self):
        self.gate.limited(self.gate.try_start()[0], retry_after=0)
        self.clock.advance(2)
        old = self.gate.try_start()[0]
        self.gate.release(old)
        current = self.gate.try_start()[0]
        self.gate.release(old)
        self.assertTrue(self.gate.try_start()[2])
        self.gate.accepted(current)
        self.clock.advance(2)
        self.assertIsNotNone(self.gate.try_start()[0])

    def test_accepted_recovery_requests_gradually_restore_unrestricted_starts(self):
        self.gate.limited(self.gate.try_start()[0], retry_after=0)
        self.clock.advance(2)
        self.gate.accepted(self.gate.try_start()[0])
        for _ in range(20):
            self.clock.advance(2)
            admission = self.gate.try_start()[0]
            self.assertIsNotNone(admission)
            self.gate.accepted(admission)
        for _ in range(25):
            self.assertIsNotNone(self.gate.try_start()[0])

    def test_forks_share_account_limits_but_other_endpoints_and_credentials_are_isolated(self):
        client = DeepSeekClient(settings())
        clone = client.fork(settings=settings(model="other-model"))
        self.assertIs(client.request_gate, clone.request_gate)
        other_key = client.fork(settings=settings(api_key="another-synthetic-key"))
        other_endpoint = client.fork(settings=settings(base_url="https://other-fixture.invalid"))
        self.assertIsNot(client.request_gate, other_key.request_gate)
        self.assertIsNot(client.request_gate, other_endpoint.request_gate)

    def test_native_http_error_retains_safe_retry_after_without_reading_or_logging_body(self):
        body = io.BytesIO(b"private fixture error body")
        error = urllib.error.HTTPError(
            "https://fixture.invalid", 429, "fixture", {"Retry-After": "120"}, body
        )
        with (
            patch("http_transport.open_url", side_effect=error),
            self.assertRaises(CompletionError) as caught,
        ):
            DeepSeekClient(settings())._complete_once([])
        self.assertEqual(caught.exception.retry_after, 120)
        self.assertTrue(body.closed)
        self.assertNotIn("private", str(caught.exception))

    def test_stream_rate_limit_after_http_acceptance_is_classified_and_keeps_partial(self):
        for code in (429, "429", "rate_limit", "rate_limit_exceeded"):
            with self.subTest(code=code):
                response = io.BytesIO(
                    b'data: {"choices": [{"delta": {"content": "partial evidence"}}]}\n'
                    + (
                        "data: "
                        + json.dumps(
                            {
                                "choices": [
                                    {
                                        "finish_reason": "error",
                                        "error": {"code": code, "message": "private fixture"},
                                    }
                                ]
                            }
                        )
                        + "\n"
                    ).encode()
                )
                with self.assertRaises(CompletionError) as caught:
                    DeepSeekClient._read_stream(response, lambda _: None)
                self.assertEqual(caught.exception.code, "http_429")
                self.assertEqual(caught.exception.partial_message["content"], "partial evidence")
                self.assertTrue(caught.exception.retryable)
                self.assertNotIn("private", str(caught.exception))

    def test_nonstream_rate_limit_is_classified_even_with_http_200(self):
        payloads = [
            {"error": {"code": "rate_limit_exceeded"}},
            {"choices": [{"finish_reason": "error", "error": {"code": "rate_limit"}}]},
        ]
        for payload in payloads:
            with (
                self.subTest(payload=payload),
                patch(
                    "http_transport.open_url", return_value=io.BytesIO(json.dumps(payload).encode())
                ),
                self.assertRaises(CompletionError) as caught,
            ):
                DeepSeekClient(settings())._complete_once([])
            self.assertEqual(caught.exception.code, "http_429")

    def test_rate_limit_retry_preserves_request_body_and_does_not_charge_waiting_as_api_calls(self):
        messages = [
            {"role": "system", "content": "fixed cached instructions"},
            {"role": "user", "content": "fixture"},
        ]
        events = []
        response = io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "prompt_cache_hit_tokens": 90,
                    },
                }
            ).encode()
        )
        error = urllib.error.HTTPError(
            "https://fixture.invalid", 429, "fixture", {"Retry-After": "5"}, io.BytesIO()
        )
        client = DeepSeekClient(settings())
        with (
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch("deepseek_client.time.monotonic", self.clock),
            patch("http_transport.open_url", side_effect=[error, response]) as opened,
        ):
            answer = client.complete(
                messages, cancelled=AdvancingEvent(self.clock), on_progress=events.append
            )
        self.assertEqual(answer["content"], "done")
        self.assertGreaterEqual(self.clock.now, 5)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(
            opened.call_args_list[0].args[0].data, opened.call_args_list[1].args[0].data
        )
        self.assertEqual(client.cache_stats.snapshot()["calls"], 2)
        self.assertTrue(any(event["phase"] == "rate_limit" for event in events))

    def test_parallel_workers_wait_together_and_recover_without_synchronized_retries(self):
        gate = RateLimitGate()
        clients = [
            DeepSeekClient(settings(request_retries=1, retry_backoff_seconds=0)) for _ in range(3)
        ]
        barrier = threading.Barrier(3)
        release, probing = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        attempts, retries = [0, 0, 0], []

        def complete_once(index, *args, **kwargs):
            attempts[index] += 1
            if attempts[index] == 1:
                barrier.wait(2)
                raise CompletionError("http_429", "fixture", retryable=True, retry_after=0.03)
            retries.append(time.monotonic())
            probing.set()
            release.wait(2)
            return {"role": "assistant", "content": "done"}

        started = time.monotonic()
        with patch("deepseek_client.shared_gate", return_value=gate), ThreadPoolExecutor(3) as pool:
            futures = []
            for index, client in enumerate(clients):
                client._complete_once = lambda *args, index=index, **kwargs: complete_once(
                    index, *args, **kwargs
                )
                futures.append(pool.submit(client.complete, [], deadline=started + 6))
            try:
                self.assertTrue(probing.wait(2))
                self.assertEqual(len(retries), 1)
            finally:
                release.set()
            self.assertEqual([future.result(6)["content"] for future in futures], ["done"] * 3)
        self.assertEqual(attempts, [2, 2, 2])
        self.assertGreater(retries[1] - retries[0], 0.8)
        self.assertGreater(retries[2] - retries[1], 0.6)

    def test_wait_can_be_cancelled_without_starting_a_request(self):
        self.gate.limited(self.gate.try_start()[0], retry_after=120)
        client = DeepSeekClient(settings())
        cancelled = threading.Event()
        timer = threading.Timer(0.03, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with (
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch.object(client, "_complete_once") as request,
            self.assertRaises(CompletionError) as caught,
        ):
            client.complete([], cancelled=cancelled)
        self.assertEqual(caught.exception.code, "cancelled")
        request.assert_not_called()
        self.assertLess(time.monotonic() - started, 0.5)

    def test_shared_cooldown_honors_task_and_generation_deadlines_without_a_request(self):
        for deadline_kind in ("deadline", "generation_timeout"):
            with self.subTest(deadline_kind=deadline_kind):
                self.gate.limited(self.gate.try_start()[0], retry_after=120)
                client = DeepSeekClient(settings())
                kwargs = {deadline_kind: 0.03}
                if deadline_kind == "deadline":
                    kwargs[deadline_kind] += time.monotonic()
                with (
                    patch("deepseek_client.shared_gate", return_value=self.gate),
                    patch.object(client, "_complete_once") as request,
                    self.assertRaises(CompletionError) as caught,
                ):
                    client.complete([], **kwargs)
                self.assertEqual(
                    caught.exception.code,
                    "task_deadline" if deadline_kind == "deadline" else "generation_budget",
                )
                if deadline_kind == "generation_timeout":
                    self.assertTrue(caught.exception.rate_limited)
                request.assert_not_called()
                self.gate = RateLimitGate(clock=self.clock)


class RateLimitTaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.clock = Clock()
        self.cancelled = AdvancingEvent(self.clock)
        self.gate = RateLimitGate(clock=self.clock)
        self.settings = make_settings(
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
            api_key="synthetic-recovery-key",
            base_url="https://recovery-fixture.invalid/v1",
            model="fixture",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=1000,
            max_subagents=20,
            max_output_tokens=16000,
            reasoning_effort="max",
            request_retries=3,
            retry_backoff_seconds=2,
            generation_timeout_seconds=None,
        )

    def test_native_recovery_crosses_transport_retry_limit_with_append_only_model_feedback(self):
        errors = [
            urllib.error.HTTPError("https://fixture.invalid", 429, "fixture", {}, io.BytesIO())
            for _ in range(6)
        ]
        accepted = io.BytesIO(
            b'data: {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}], '
            b'"usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 90}}\n'
            b"data: [DONE]\n"
        )
        client = DeepSeekClient(self.settings)
        events = []
        messages = [{"role": "user", "content": "Complete the remaining work"}]
        definitions = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
        with (
            patch("deepseek_client.time.monotonic", self.clock),
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch("http_transport.open_url", side_effect=[*errors, accepted]) as opened,
        ):
            runtime = TaskRuntime(client, self.settings, 1000, cancelled=self.cancelled)
            response = runtime.generate(
                messages,
                {"role": "system", "content": "fixed instructions"},
                definitions,
                on_progress=events.append,
            )
        self.assertEqual(response["content"], "done")
        self.assertEqual(opened.call_count, 7)
        self.assertEqual(runtime.requests, 2)
        self.assertEqual(client.cache_stats.snapshot()["calls"], 7)
        bodies = [json.loads(call.args[0].data) for call in opened.call_args_list]
        first = bodies[0]
        self.assertEqual(bodies[:4], [first] * 4)
        for body in bodies[4:]:
            self.assertEqual(body["messages"][: len(first["messages"])], first["messages"])
            self.assertEqual(
                {key: value for key, value in body.items() if key != "messages"},
                {key: value for key, value in first.items() if key != "messages"},
            )
            self.assertIn('"code": "http_429"', body["messages"][-1]["content"])
        self.assertGreaterEqual(self.clock.now, 122)
        self.assertTrue(any(event["phase"] == "rate_limit" for event in events))

    def test_short_generation_budget_during_shared_cooldown_keeps_task_alive(self):
        self.gate.limited(self.gate.try_start()[0], retry_after=8)
        client = DeepSeekClient(self.settings)
        self.settings = replace(self.settings, generation_timeout_seconds=0.5)
        client.settings = self.settings
        with (
            patch("deepseek_client.time.monotonic", self.clock),
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch.object(client, "_complete_once", return_value={"content": "done"}) as request,
        ):
            runtime = TaskRuntime(client, self.settings, 60, cancelled=self.cancelled)
            messages = [{"role": "user", "content": "task"}]
            response = runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertEqual(response["content"], "done")
        request.assert_called_once()
        self.assertGreater(runtime.requests, 1)
        self.assertEqual(sum("Runtime execution result:" in m["content"] for m in messages), 1)
        self.assertIn('"rate_limited": true', messages[-1]["content"])

    def test_long_native_retry_after_waits_until_original_deadline_with_failure_retained(self):
        client = DeepSeekClient(self.settings)
        error = urllib.error.HTTPError(
            "https://fixture.invalid", 429, "fixture", {"Retry-After": "1200"}, io.BytesIO()
        )
        messages = [{"role": "user", "content": "task"}]
        with (
            patch("deepseek_client.time.monotonic", self.clock),
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch("http_transport.open_url", side_effect=error) as opened,
        ):
            runtime = TaskRuntime(client, self.settings, 3, cancelled=self.cancelled)
            with self.assertRaises(CompletionError) as caught:
                runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertEqual(caught.exception.code, "task_deadline")
        self.assertEqual(self.clock.now, 3)
        opened.assert_called_once()
        self.assertIn('"code": "http_429"', messages[-1]["content"])

    def test_native_midstream_limit_recovers_from_partial_without_replaying_calls(self):
        partial = "Retained unverified evidence. " * 20
        stream = io.BytesIO(
            (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": partial,
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "unfinished",
                                            "type": "function",
                                            "function": {"name": "write", "arguments": "{"},
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
                + "\n"
                + 'data: {"choices": [{"finish_reason": "error", "error": {"code": "rate_limit"}}]}\n'
            ).encode()
        )
        accepted = io.BytesIO(
            b'data: {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}\ndata: [DONE]\n'
        )
        client = DeepSeekClient(self.settings)
        messages = [{"role": "user", "content": "task"}]
        with (
            patch("deepseek_client.time.monotonic", self.clock),
            patch("deepseek_client.shared_gate", return_value=self.gate),
            patch("http_transport.open_url", side_effect=[stream, accepted]) as opened,
        ):
            runtime = TaskRuntime(client, self.settings, 60, cancelled=self.cancelled)
            answer = runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertEqual(answer["content"], "done")
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(runtime.requests, 2)
        self.assertFalse(answer.get("tool_calls"))
        self.assertFalse(any(message.get("role") == "tool" for message in messages))
        last_request = json.loads(opened.call_args.args[0].data)
        self.assertIn(partial, str(last_request["messages"]))
        self.assertIn('"code": "http_429"', last_request["messages"][-1]["content"])
        self.assertTrue(list((self.workspace / ".agent-artifacts").glob("*.unfinished.json")))

    def test_persistent_limit_respects_deadline_without_spinning_or_growing_prompt(self):
        calls = []

        class Client:
            def complete(inner, messages, **kwargs):
                calls.append(json.loads(json.dumps(messages)))
                raise CompletionError("http_429", "fixture", retryable=True)

        with patch("task_runtime.time.monotonic", self.clock):
            runtime = TaskRuntime(Client(), self.settings, 30, cancelled=self.cancelled)
            messages = [{"role": "user", "content": "task"}]
            with self.assertRaises(CompletionError) as caught:
                runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertEqual(caught.exception.code, "task_deadline")
        self.assertEqual(self.clock.now, 30)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(messages), 2)
        self.assertEqual(calls[1:], [calls[1]] * 3)

    def test_retry_after_wait_can_be_cancelled_without_another_model_request(self):
        calls, events = [], []

        class Client:
            def complete(inner, messages, **kwargs):
                calls.append(messages)
                raise CompletionError("http_429", "fixture", retryable=True, retry_after=1200)

        def cancel(event):
            events.append(event)
            if event["phase"] == "rate_limit":
                self.cancelled.set()

        with patch("task_runtime.time.monotonic", self.clock):
            runtime = TaskRuntime(Client(), self.settings, 60, cancelled=self.cancelled)
            with self.assertRaises(CompletionError) as caught:
                runtime.generate([], {"role": "system", "content": "fixed"}, [], on_progress=cancel)
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[-1]["delay_seconds"], 1200)

    def test_keyboard_interrupt_during_recovery_sets_cancellation_and_keeps_observation(self):
        calls = []

        class Client:
            def complete(inner, messages, **kwargs):
                calls.append(messages)
                raise CompletionError("http_429", "fixture", retryable=True)

        def interrupt(event):
            if event.get("phase") == "rate_limit":
                raise KeyboardInterrupt

        messages = [{"role": "user", "content": "task"}]
        with patch("task_runtime.time.monotonic", self.clock):
            runtime = TaskRuntime(Client(), self.settings, 60, cancelled=self.cancelled)
            with self.assertRaises(KeyboardInterrupt):
                runtime.generate(
                    messages, {"role": "system", "content": "fixed"}, [], on_progress=interrupt
                )
        self.assertTrue(self.cancelled.is_set())
        self.assertEqual(len(calls), 1)
        self.assertIn('"code": "http_429"', messages[-1]["content"])

    def test_authentication_and_non_rate_generation_failures_remain_terminal(self):
        for code in ("http_401", "http_402", "http_403", "generation_budget"):
            with self.subTest(code=code):
                error = CompletionError(code, "fixture")

                class Client:
                    def complete(inner, *args, error=error, **kwargs):
                        raise error

                messages = [{"role": "user", "content": "task"}]
                with patch("task_runtime.time.monotonic", self.clock):
                    runtime = TaskRuntime(Client(), self.settings, 60, cancelled=self.cancelled)
                    with self.assertRaises(CompletionError) as caught:
                        runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
                self.assertIs(caught.exception, error)
                self.assertEqual(len(messages), 1)

    def test_model_limit_partial_is_retained_without_executing_its_calls(self):
        partial = {
            "content": "Unverified partial finding",
            "tool_calls": [{"id": "unexecuted", "function": {"name": "write"}}],
        }
        calls = []

        class Client:
            def complete(inner, messages, **kwargs):
                calls.append(json.loads(json.dumps(messages)))
                if len(calls) == 1:
                    raise CompletionError(
                        "http_429", "fixture", retryable=True, partial_message=partial
                    )
                return {"content": "done"}

        with patch("task_runtime.time.monotonic", self.clock):
            runtime = TaskRuntime(Client(), self.settings, 60, cancelled=self.cancelled)
            messages = [{"role": "user", "content": "task"}]
            self.assertEqual(
                runtime.generate(messages, {"role": "system", "content": "fixed"}, [])["content"],
                "done",
            )
        self.assertIn("Unverified partial finding", str(calls[1]))
        self.assertIn("checkpoint tool calls were not executed", str(calls[1]))
        self.assertFalse(any(message.get("role") == "tool" for message in messages))
        artifact = next((self.workspace / ".agent-artifacts").glob("*.unfinished.json"))
        self.assertEqual(json.loads(artifact.read_text()), partial)

    def test_worker_recovers_multiple_limits_without_replaying_completed_tool(self):
        call = {
            "id": "completed-read",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "source.txt"}'},
        }
        requests, executed, events = [], [], []

        class Client:
            settings = self.settings

            def complete(inner, messages, **kwargs):
                requests.append(json.loads(json.dumps(messages)))
                if len(requests) == 1:
                    return {"role": "assistant", "tool_calls": [call]}
                if len(requests) <= 5:
                    raise CompletionError("http_429", "fixture", retryable=True)
                return {"role": "assistant", "content": "Report from retained evidence"}

        def execute(call, execution):
            executed.append(call["id"])
            return json.dumps(ToolResult(data={"content": "verified source evidence"}).as_dict())

        definitions = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
        runner = SubagentRunner(Client(), execute, definitions, self.workspace)
        with patch("task_runtime.time.monotonic", self.clock):
            report = runner.run(
                "Inspect source",
                "context",
                "report",
                ExecutionContext(60, self.cancelled),
                on_progress=events.append,
            )
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.data["tool_calls"], 1)
        self.assertEqual(report.data["model_requests"], 6)
        self.assertEqual(report.data["model_recovery"]["status"], "recovered")
        self.assertEqual(report.data["model_recovery"]["recovery_cycles"], 4)
        self.assertEqual(executed, ["completed-read"])
        self.assertIn("verified source evidence", str(requests[-1]))
        self.assertEqual(
            sum("Runtime execution result:" in m.get("content", "") for m in requests[-1]), 1
        )
        self.assertTrue(any(event.get("model_phase") == "rate_limit" for event in events))
        self.assertFalse(
            list((self.workspace / ".agent-artifacts").glob("*.worker-checkpoint.json"))
        )

    def test_waiting_worker_checkpoint_already_contains_model_failure_observation(self):
        checkpoint = WorkerCheckpoint(self.workspace)
        states = []

        class Client:
            settings = self.settings

            def complete(inner, *args, **kwargs):
                raise CompletionError("http_429", "fixture", retryable=True, retry_after=120)

        def progress(event):
            if event.get("model_phase") == "rate_limit" and not states:
                # The delegate may freeze its checkpoint while the worker is
                # still waiting; the recovery note must already be captured.
                stopped = checkpoint.failure("deadline_exceeded", "fixture timeout")
                states.append(json.loads(Path(stopped.data["checkpoint_path"]).read_text()))
                self.cancelled.set()

        with patch("task_runtime.time.monotonic", self.clock):
            runner = SubagentRunner(Client(), lambda *args: None, [], self.workspace)
            runner.run(
                "Inspect",
                "context",
                "report",
                ExecutionContext(60, self.cancelled),
                on_progress=progress,
                checkpoint=checkpoint,
            )
        self.assertEqual(len(states), 1)
        self.assertIn('"code": "http_429"', str(states[0]["messages"]))
        self.assertEqual(states[0]["data"]["model_recovery"]["status"], "waiting")

    def test_rate_limit_observation_survives_compaction_that_omits_it(self):
        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(json.loads(json.dumps(messages)))
                if len(requests) == 1:
                    raise CompletionError("http_429", "fixture", retryable=True)
                return {"content": "done"}

        def compact(messages, *args, **kwargs):
            if len(requests) == 1:
                messages[:] = [
                    {"role": "user", "content": "task"},
                    {"role": "system", "content": "summary without failure"},
                ]
                return True
            return False

        with (
            patch("task_runtime.time.monotonic", self.clock),
            patch("task_runtime.compact_messages", side_effect=compact),
        ):
            runtime = TaskRuntime(Client(), self.settings, 60, cancelled=self.cancelled)
            runtime.generate(
                [{"role": "user", "content": "task"}], {"role": "system", "content": "fixed"}, []
            )
        self.assertIn('"code": "http_429"', requests[-1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
