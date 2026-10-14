import io
import json
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_client import CompletionError, DeepSeekClient
from generation_metrics import GenerationMetrics
from usage import CacheStats


def event(delta):
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n".encode()


class StreamTests(unittest.TestCase):
    def client(self):
        return DeepSeekClient(
            SimpleNamespace(
                model="test",
                api_key="test",
                base_url="https://example.invalid",
                max_output_tokens=131072,
                request_retries=0,
            )
        )

    def test_partial_response_is_not_accepted_as_complete(self):
        response = io.BytesIO(event({"content": "unfinished"}))
        with self.assertRaises(CompletionError) as raised:
            DeepSeekClient._read_stream(response, lambda _: None)
        self.assertEqual(raised.exception.code, "incomplete_stream")
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.partial_message["content"], "unfinished")

    def test_display_callback_failures_do_not_retry_a_completed_model_request(self):
        client = self.client()
        response = {"role": "assistant", "content": "accepted answer"}

        def fail(*args):
            raise RuntimeError("display fixture")

        def complete_once(*args, **kwargs):
            kwargs["on_progress"]({"phase": "receiving"})
            kwargs["on_content"](response["content"])
            return response

        with patch.object(client, "_complete_once", side_effect=complete_once) as request:
            accepted = client.complete([], on_content=fail, on_progress=fail)
        self.assertEqual(accepted, response)
        request.assert_called_once()

    def test_preview_deltas_arrive_before_the_stream_finishes(self):
        client = self.client()
        previews, accepted = [], []

        class StreamingResponse(io.BytesIO):
            def __iter__(response):
                yield event({"reasoning_content": "先检查"})
                self.assertEqual(previews[-1]["reasoning"], "先检查")
                self.assertEqual(accepted, [])
                yield event({"content": "第一段"})
                self.assertEqual(previews[-1]["content"], "第一段")
                self.assertEqual(accepted, [])
                yield event({"content": "第二段"})
                yield b"data: [DONE]\n"

        with patch("http_transport.open_url", return_value=StreamingResponse()):
            answer = client.complete([], on_content=accepted.append, on_progress=previews.append)

        self.assertEqual(answer["content"], "第一段第二段")
        self.assertEqual(accepted, ["第一段", "第二段"])
        deltas = [preview for preview in previews if preview["phase"] == "response_delta"]
        self.assertEqual([delta["content"] for delta in deltas], ["", "第一段", "第二段"])

    def test_failed_attempt_previews_are_reset_before_a_retry(self):
        client = self.client()
        client.settings.request_retries = 1
        client.settings.retry_backoff_seconds = 0
        previews, accepted = [], []
        failed = io.BytesIO(event({"content": "discard this partial answer"}))
        success = io.BytesIO(event({"content": "valid answer"}) + b"data: [DONE]\n")
        with patch("http_transport.open_url", side_effect=[failed, success]):
            answer = client.complete([], on_content=accepted.append, on_progress=previews.append)

        self.assertEqual(answer["content"], "valid answer")
        self.assertEqual(accepted, ["valid answer"])
        response_events = [
            preview for preview in previews if preview["phase"].startswith("response_")
        ]
        self.assertEqual(
            [preview["phase"] for preview in response_events],
            [
                "response_reset",
                "response_delta",
                "response_reset",
                "response_reset",
                "response_delta",
            ],
        )

    def test_summary_requests_do_not_publish_chat_previews(self):
        client = self.client()
        previews = []
        response = io.BytesIO(event({"content": "summary"}) + b"data: [DONE]\n")
        with patch("http_transport.open_url", return_value=response):
            client.complete(
                [], on_content=lambda _: None, on_progress=previews.append, purpose="summary"
            )
        self.assertFalse(any(preview["phase"].startswith("response_") for preview in previews))

    def test_preview_reports_tool_argument_progress_without_raw_arguments(self):
        previews = []
        response = io.BytesIO(
            event({"role": "assistant"})
            + event({"reasoning": "same", "reasoning_content": "same"})
            + event(
                {
                    "tool_calls": [
                        {"index": 0, "id": "call-1", "function": {"name": "ls", "arguments": "{}"}}
                    ]
                }
            )
            + b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n'
            + b"data: [DONE]\n"
        )
        DeepSeekClient._read_stream(response, lambda _: None, on_delta=previews.append)
        self.assertEqual(
            previews,
            [
                {"phase": "response_delta", "content": "", "reasoning": "same"},
                {"phase": "tool_call_delta", "tool": "ls", "argument_chars": 2},
            ],
        )

    def test_late_request_usage_does_not_enter_new_session_stats(self):
        client = self.client()
        previous_stats = client.cache_stats

        def finish_old_request(*args):
            client.cache_stats = CacheStats()
            return {"role": "assistant", "content": "old response"}

        with patch.object(client, "_perform_request", side_effect=finish_old_request):
            client._complete_once([{"role": "user", "content": "old task"}])
        self.assertEqual(previous_stats.snapshot()["calls"], 1)
        self.assertEqual(client.cache_stats.snapshot()["calls"], 0)

    def test_stream_performance_uses_output_interval_and_survives_usage_attachment(self):
        client = self.client()
        events = []
        timestamps = iter([0, 10, 12])

        def metrics(usage, on_progress):
            return GenerationMetrics(usage, on_progress, clock=lambda: next(timestamps))

        response = io.BytesIO(
            event({"content": "first"})
            + event({"content": "second"})
            + b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 120}}\n'
            + b"data: [DONE]\n"
        )
        with (
            patch("http_transport.open_url", return_value=response) as opened,
            patch("deepseek_client.GenerationMetrics", side_effect=metrics),
        ):
            answer = client.complete([], on_content=lambda _: None, on_progress=events.append)
        performance = answer["_harness"]["performance"]
        self.assertEqual(performance["tps"], 60)
        self.assertEqual(performance["excluded_wait_seconds"], 10)
        self.assertEqual(answer["_harness"]["usage"]["completion_tokens"], 120)
        self.assertTrue(events[-1]["finished"])
        self.assertNotIn("performance", opened.call_args.args[0].data.decode())

    def test_gateway_error_is_not_silently_returned_as_empty(self):
        response = io.BytesIO(
            b'data: {"error":{"code":"gateway_stream_terminated"},"choices":[{"delta":{},"finish_reason":"error"}]}\n'
        )
        with self.assertRaises(CompletionError) as raised:
            DeepSeekClient._read_stream(response, lambda _: None)
        self.assertEqual(raised.exception.code, "provider_error")
        self.assertTrue(raised.exception.retryable)

    def test_output_limit_is_not_a_retryable_network_failure(self):
        response = io.BytesIO(
            event({"content": "partial answer"})
            + b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n'
        )
        with self.assertRaises(CompletionError) as raised:
            DeepSeekClient._read_stream(response, lambda _: None)
        self.assertEqual(raised.exception.code, "output_limit")
        self.assertFalse(raised.exception.retryable)

    def test_terminal_chunk_without_done_is_sufficient(self):
        response = io.BytesIO(
            event({"content": "complete"})
            + b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        )
        self.assertEqual(
            DeepSeekClient._read_stream(response, lambda _: None)["content"], "complete"
        )

    def test_reasoning_survives_for_subsequent_tool_turns(self):
        response = io.BytesIO(
            event({"reasoning": "first ", "reasoning_content": "native "})
            + event({"reasoning": "second", "reasoning_content": "reasoning", "content": "answer"})
            + b"data: [DONE]\n"
        )
        message = DeepSeekClient._read_stream(response, lambda _: None)
        self.assertEqual(message["reasoning"], "first second")
        self.assertEqual(message["reasoning_content"], "native reasoning")

    def test_wrapped_nonstream_response_and_explicit_token_budget(self):
        payload = {
            "data": {
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
                ]
            }
        }
        with patch(
            "http_transport.open_url", return_value=io.BytesIO(json.dumps(payload).encode())
        ) as opened:
            answer = self.client().complete([{"role": "user", "content": "question"}])
        self.assertEqual(answer["content"], "answer")
        self.assertEqual(json.loads(opened.call_args.args[0].data)["max_tokens"], 131072)

    def test_nonstream_malformed_shapes_are_structured_errors(self):
        for payload in ([], {"choices": "wrong"}, {"choices": [None]}):
            with (
                self.subTest(payload=payload),
                patch(
                    "http_transport.open_url", return_value=io.BytesIO(json.dumps(payload).encode())
                ),
                self.assertRaises(CompletionError) as raised,
            ):
                self.client().complete([])
            self.assertEqual(raised.exception.code, "invalid_response")
            self.assertTrue(raised.exception.retryable)

    def test_stream_malformed_shapes_are_structured_errors(self):
        payloads = (
            [],
            {"choices": "wrong"},
            {"choices": [None]},
            {"choices": [{"delta": "wrong"}]},
            {"choices": [{"delta": {"tool_calls": [{}]}}]},
        )
        for payload in payloads:
            response = io.BytesIO(f"data: {json.dumps(payload)}\n".encode())
            with self.subTest(payload=payload), self.assertRaises(CompletionError) as raised:
                DeepSeekClient._read_stream(response, lambda _: None)
            self.assertEqual(raised.exception.code, "invalid_response")
            self.assertTrue(raised.exception.retryable)
            self.assertEqual(raised.exception.partial_message["role"], "assistant")

    def test_summary_output_override_does_not_change_solver_budget(self):
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
            ]
        }
        client = self.client()
        with patch(
            "http_transport.open_url",
            side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
        ) as opened:
            client.complete([], max_output_tokens=8192)
            client.complete([])
        self.assertEqual(
            [json.loads(call.args[0].data)["max_tokens"] for call in opened.call_args_list],
            [8192, 131072],
        )

    def test_length_keeps_reasoning_and_partial_tool_arguments_for_recovery(self):
        for streamed in (False, True):
            with self.subTest(streamed=streamed):
                partial = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "known result = 42",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "write", "arguments": '{"path":'},
                        }
                    ],
                }
                if streamed:
                    payload = (
                        event(partial)
                        + b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n'
                    )
                else:
                    payload = json.dumps(
                        {"choices": [{"finish_reason": "length", "message": partial}]}
                    ).encode()
                with (
                    patch("http_transport.open_url", return_value=io.BytesIO(payload)),
                    self.assertRaises(CompletionError) as raised,
                ):
                    self.client().complete([], on_content=(lambda _: None) if streamed else None)
                self.assertEqual(raised.exception.code, "output_limit")
                self.assertEqual(
                    raised.exception.partial_message["reasoning_content"], "known result = 42"
                )
                self.assertEqual(
                    raised.exception.partial_message["tool_calls"][0]["function"]["arguments"],
                    '{"path":',
                )

    def test_auth_errors_are_not_retryable_but_rate_limits_are(self):
        for status, retryable in [(401, False), (400, False), (429, True), (503, True)]:
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://example.invalid", status, "test", {}, io.BytesIO(b"test")
                )
                with (
                    patch("http_transport.open_url", side_effect=error),
                    self.assertRaises(CompletionError) as raised,
                ):
                    self.client().complete([])
                self.assertEqual(raised.exception.retryable, retryable)

    def test_active_stream_respects_absolute_task_deadline(self):
        with (
            patch("deepseek_client.time.monotonic", return_value=2),
            self.assertRaises(CompletionError) as raised,
        ):
            DeepSeekClient._read_stream(
                io.BytesIO(event({"reasoning": "still thinking"})), lambda _: None, deadline=1
            )
        self.assertEqual(raised.exception.code, "task_deadline")

    def test_stream_assembles_content_and_fragmented_tool_calls(self):
        response = io.BytesIO(
            b"".join(
                [
                    event({"content": "hello "}),
                    event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-",
                                    "function": {"name": "re", "arguments": '{"pa'},
                                }
                            ]
                        }
                    ),
                    event(
                        {
                            "content": "world",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "1",
                                    "function": {"name": "ad", "arguments": 'th":"x"}'},
                                }
                            ],
                        }
                    ),
                    b"data: [DONE]\n",
                ]
            )
        )
        streamed = []

        message = DeepSeekClient._read_stream(response, streamed.append)

        self.assertEqual("".join(streamed), "hello world")
        self.assertEqual(message["content"], "hello world")
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")
        self.assertEqual(
            message["tool_calls"][0]["function"], {"name": "read", "arguments": '{"path":"x"}'}
        )


if __name__ == "__main__":
    unittest.main()
