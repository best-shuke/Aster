import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from anthropic_client import AnthropicClient
from auth import AuthStore
from chatgpt_client import ChatGPTClient
from config import Login, Settings
from context_budget import LOCAL_METADATA, request_fingerprint
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from model_api import model_endpoint, parse_api_compat, parse_model_api
from model_clients import model_client
from model_messages import protocol_messages
from responses_client import ResponsesClient
from tests.support import make_settings
from tools import CodingTools
from usage import UsageAccumulator

APIS = ("openai-responses", "anthropic-messages")
CALL = {
    "id": "call_read",
    "type": "function",
    "function": {"name": "read", "arguments": '{"path":"note.txt"}'},
}


def payload(api, *, tool=False, reason=None):
    if api == "openai-responses":
        output = (
            [{"type": "function_call", "id": "fc_read", "call_id": CALL["id"], **CALL["function"]}]
            if tool
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ]
        )
        return {
            "status": "incomplete" if reason else "completed",
            "incomplete_details": {"reason": reason},
            "output": output,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 7},
            },
        }
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": CALL["id"], "name": "read", "input": {"path": "note.txt"}}
        ]
        if tool
        else [{"type": "text", "text": "answer"}],
        "stop_reason": "max_tokens" if reason else "tool_use" if tool else "end_turn",
        "usage": {
            "input_tokens": 3,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 2,
            "output_tokens": 5,
        },
    }


def events(api, value):
    if api == "openai-responses":
        result = [{"type": "response.created", "response": {}}]
        for index, item in enumerate(value["output"]):
            start = {**item, "arguments": ""} if item["type"] == "function_call" else item
            result.append(
                {"type": "response.output_item.added", "output_index": index, "item": start}
            )
            if item["type"] == "function_call":
                result.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "item_id": item["id"],
                        "delta": item["arguments"],
                    }
                )
            elif item["type"] == "message":
                result.append(
                    {
                        "type": "response.output_text.delta",
                        "output_index": index,
                        "delta": item["content"][0]["text"],
                    }
                )
            result.append(
                {"type": "response.output_item.done", "output_index": index, "item": item}
            )
        return [
            *result,
            {
                "type": "response.incomplete"
                if value["status"] == "incomplete"
                else "response.completed",
                "response": value,
            },
        ]
    result = [
        {"type": "message_start", "message": {"usage": {**value["usage"], "output_tokens": 0}}}
    ]
    for index, block in enumerate(value["content"]):
        if block["type"] == "tool_use":
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start = {**block, "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        result.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": start},
                {"type": "content_block_delta", "index": index, "delta": delta},
                {"type": "content_block_stop", "index": index},
            ]
        )
    return [
        *result,
        {
            "type": "message_delta",
            "delta": {"stop_reason": value["stop_reason"]},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]


def encode_sse(rows):
    return b": heartbeat\n\n" + b"".join(
        f"event: {row['type']}\ndata: {json.dumps(row)}\n\n".encode() for row in rows
    )


class ModelApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests, self.replies = [], []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, dict(self.headers), body))
                status, content, data = owner.replies.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", content)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_port}"

    def settings(self, api, **values):
        return make_settings(
            api_key="synthetic-key",
            base_url=self.base,
            model="test-model",
            workspace=self.root,
            session_file=self.root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            api=api,
            reasoning_effort="off",
            request_retries=0,
            **values,
        )

    def queue(self, api, *, streaming=False, tool=False, reason=None):
        value = payload(api, tool=tool, reason=reason)
        data = encode_sse(events(api, value)) if streaming else json.dumps(value).encode()
        self.replies.append((200, "text/event-stream" if streaming else "application/json", data))

    def test_factory_separates_authentication_from_protocol_and_auto_detection(self):
        for api, expected in [
            ("openai-completions", DeepSeekClient),
            ("openai-responses", ResponsesClient),
            ("anthropic-messages", AnthropicClient),
        ]:
            self.assertIsInstance(model_client(self.settings(api)), expected)
        for endpoint, expected in [
            ("https://api.deepseek.com", "openai-completions"),
            ("https://api.openai.com/v1", "openai-responses"),
            ("https://api.anthropic.com", "anthropic-messages"),
            ("https://api.deepseek.com/anthropic", "anthropic-messages"),
            ("https://proxy.invalid/v1", "openai-completions"),
        ]:
            self.assertEqual(
                replace(self.settings("auto"), base_url=endpoint).resolved_api, expected
            )

    def test_sync_and_stream_outputs_usage_auth_and_deadlines_share_one_contract(self):
        for api in APIS:
            for streaming in (False, True):
                with self.subTest(api=api, streaming=streaming):
                    self.queue(api, streaming=streaming)
                    client = model_client(self.settings(api))
                    content, progress = [], []
                    result = client.complete(
                        [{"role": "user", "content": "task"}],
                        on_content=content.append if streaming else None,
                        on_progress=progress.append,
                    )
                    self.assertEqual(result["content"], "answer")
                    usage = result[LOCAL_METADATA]["usage"]
                    self.assertEqual(usage["prompt_tokens"], 12)
                    self.assertEqual(usage["completion_tokens"], 5)
                    self.assertEqual(usage["total_tokens"], 17)
                    self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 7)
                    self.assertEqual(client.cache_stats.snapshot()["calls"], 1)
                    if streaming:
                        self.assertEqual("".join(content), "answer")
                        self.assertTrue(
                            any(event["phase"] == "response_delta" for event in progress)
                        )
                    route, headers, body = self.requests[-1]
                    self.assertEqual(
                        route, "/responses" if api == "openai-responses" else "/v1/messages"
                    )
                    self.assertEqual(body["stream"], streaming)
                    if api == "anthropic-messages":
                        self.assertEqual(headers["X-Api-Key"], "synthetic-key")
                        self.assertNotIn("Authorization", headers)
                    else:
                        self.assertEqual(headers["Authorization"], "Bearer synthetic-key")

    def test_real_http_tool_loop_returns_results_to_each_protocol(self):
        (self.root / "note.txt").write_text("fixture text")
        for api in APIS:
            for streaming in (False, True):
                with self.subTest(api=api, streaming=streaming):
                    # TaskRuntime streams internally even without a display callback.
                    self.queue(api, streaming=True, tool=True)
                    self.queue(api, streaming=True)
                    settings = replace(
                        self.settings(api), session_file=self.root / f"{api}-{streaming}.json"
                    )
                    tools = CodingTools(self.root)
                    conversation = Conversation(settings, model_client(settings), tools)
                    self.addCleanup(conversation.close)
                    answer = conversation.ask(
                        "read note.txt", on_content=(lambda _: None) if streaming else None
                    )
                    self.assertEqual(answer, "answer")
                    body = self.requests[-1][2]
                    if api == "openai-responses":
                        result = next(
                            item for item in body["input"] if item["type"] == "function_call_output"
                        )
                        self.assertEqual(result["call_id"], CALL["id"])
                        self.assertIn("fixture text", result["output"])
                        self.assertFalse(body["tools"][0]["strict"])
                    else:
                        result = body["messages"][-1]["content"][0]
                        self.assertEqual(result["tool_use_id"], CALL["id"])
                        self.assertIn("fixture text", result["content"])
                        self.assertIn("input_schema", body["tools"][0])

    def test_truncated_and_unterminated_streams_never_become_successful_answers(self):
        for api in APIS:
            for streaming in (False, True):
                with self.subTest(api=api, streaming=streaming):
                    self.queue(api, streaming=streaming, reason="max_output_tokens")
                    with self.assertRaises(CompletionError) as raised:
                        model_client(self.settings(api)).complete(
                            [], on_content=(lambda _: None) if streaming else None
                        )
                    self.assertEqual(raised.exception.code, "output_limit")
                    self.assertEqual(raised.exception.partial_message["content"], "answer")
            data = encode_sse(events(api, payload(api))[:-1])
            self.replies.append((200, "text/event-stream", data))
            with self.assertRaises(CompletionError) as raised:
                model_client(self.settings(api)).complete([], on_content=lambda _: None)
            self.assertEqual(raised.exception.code, "incomplete_stream")

    def test_http_errors_are_sanitized_and_401_is_not_retried(self):
        for api in APIS:
            self.replies.append((401, "application/json", b'{"error":{"message":"synthetic-key"}}'))
            with self.assertRaises(CompletionError) as raised:
                model_client(self.settings(api)).complete([])
            self.assertEqual(raised.exception.code, "http_401")
            self.assertNotIn("synthetic-key", str(raised.exception))

    def test_deepseek_compatibility_omits_unsupported_responses_fields(self):
        settings = replace(
            self.settings("openai-responses"),
            base_url="https://api.deepseek.com",
            reasoning_effort="max",
        )
        body = model_client(settings)._request_body(
            [{"role": "user", "content": "task"}, {"role": "system", "content": "runtime note"}],
            None,
            True,
            200,
        )
        self.assertNotIn("store", body)
        self.assertNotIn("include", body)
        self.assertEqual(body["input"][-1]["role"], "system")
        self.assertEqual(body["reasoning"], {"effort": "max"})
        self.assertEqual(
            model_endpoint("https://api.deepseek.com/v1", "anthropic-messages"),
            "https://api.deepseek.com/anthropic/v1/messages",
        )

    def test_native_thinking_replay_is_scoped_to_model_protocol_endpoint_and_message(self):
        for api in APIS:
            settings = self.settings(api)
            client = model_client(settings)
            value = payload(api)
            if api == "openai-responses":
                value["output"].insert(
                    0, {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
                )
            else:
                value["content"].insert(
                    0, {"type": "thinking", "thinking": "plan", "signature": "signed"}
                )
            self.replies.append((200, "application/json", json.dumps(value).encode()))
            message = client.complete([])
            body = client._request_body([message], None, False, 200)
            self.assertIn("opaque" if api == "openai-responses" else "signed", json.dumps(body))
            for changed in [
                replace(settings, model="other"),
                replace(settings, base_url="https://other.invalid"),
            ]:
                body = model_client(changed)._request_body([message], None, False, 200)
                self.assertNotIn(
                    "opaque" if api == "openai-responses" else "signed", json.dumps(body)
                )
            changed_message = copy.deepcopy(message)
            changed_message["content"] = "edited answer"
            body = client._request_body([changed_message], None, False, 200)
            self.assertIn("edited answer", json.dumps(body))
            self.assertNotIn("opaque" if api == "openai-responses" else "signed", json.dumps(body))

    def test_protocol_and_compatibility_affect_cache_identity(self):
        settings = self.settings("openai-completions")
        original = request_fingerprint([], None, settings)
        self.assertNotEqual(
            original, request_fingerprint([], None, replace(settings, api="openai-responses"))
        )
        self.assertNotEqual(
            original,
            request_fingerprint([], None, replace(settings, api_compat={"supports_store": False})),
        )

    def test_cross_protocol_tool_ids_are_normalized_without_changing_saved_history(self):
        invalid_id = "fc_" + "x" * 80 + "|native-id"
        history = [
            {"role": "assistant", "content": None, "tool_calls": [{**CALL, "id": invalid_id}]},
            {"role": "tool", "tool_call_id": invalid_id, "content": "result"},
        ]
        original = copy.deepcopy(history)
        for api in APIS:
            messages = protocol_messages(history, self.settings(api))
            normalized = messages[0]["tool_calls"][0]["id"]
            self.assertLessEqual(len(normalized), 64)
            self.assertNotIn("|", normalized)
            self.assertEqual(normalized, messages[1]["tool_call_id"])
            self.assertEqual(history, original)

    def test_malformed_content_and_provider_errors_are_structured(self):
        for api in APIS:
            invalid = payload(api)
            if api == "openai-responses":
                invalid["output"][0]["content"] = [None]
            else:
                invalid["content"] = [None]
            for bad in [[], invalid]:
                with self.subTest(api=api, value=bad):
                    self.replies.append((200, "application/json", json.dumps(bad).encode()))
                    with self.assertRaises(CompletionError) as raised:
                        model_client(self.settings(api)).complete([])
                    self.assertEqual(raised.exception.code, "invalid_response")
            error = {"type": "error", "error": {"type": "api_error", "message": "synthetic-key"}}
            self.replies.append((200, "application/json", json.dumps(error).encode()))
            with self.assertRaises(CompletionError) as raised:
                model_client(self.settings(api)).complete([])
            self.assertEqual(raised.exception.code, "provider_error")
            self.assertNotIn("synthetic-key", str(raised.exception))

    def test_cancellation_and_expired_deadline_do_not_send_a_request(self):
        for api in APIS:
            cancelled = threading.Event()
            cancelled.set()
            for options, code in [
                ({"cancelled": cancelled}, "cancelled"),
                ({"deadline": time.monotonic() - 1}, "task_deadline"),
            ]:
                with self.assertRaises(CompletionError) as raised:
                    model_client(self.settings(api)).complete([], **options)
                self.assertEqual(raised.exception.code, code)
        self.assertEqual(self.requests, [])

    def test_anthropic_requires_closed_blocks_and_preserves_empty_tool_arguments(self):
        value = payload("anthropic-messages", tool=True)
        value["content"][0]["input"] = {}
        rows = events("anthropic-messages", value)
        rows = [row for row in rows if row["type"] != "content_block_delta"]
        result = AnthropicClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(result["tool_calls"][0]["function"]["arguments"], "{}")
        rows = [row for row in rows if row["type"] != "content_block_stop"]
        with self.assertRaises(CompletionError) as raised:
            AnthropicClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "incomplete_stream")

    def test_responses_rejects_final_tool_arguments_that_differ_from_stream(self):
        rows = events("openai-responses", payload("openai-responses", tool=True))
        for row in rows:
            if row["type"] == "response.output_item.done":
                row["item"] = {**row["item"], "arguments": '{"path":"other.txt"}'}
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_unfinalized_call_without_done_event_is_rejected(self):
        rows = events("openai-responses", payload("openai-responses", tool=True))
        rows = [row for row in rows if row["type"] != "response.output_item.done"]
        for row in rows:
            if row["type"] == "response.completed":
                row["response"]["output"] = []
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_changed_function_name_at_done_is_rejected(self):
        rows = events("openai-responses", payload("openai-responses", tool=True))
        for row in rows:
            if row["type"] == "response.output_item.done":
                row["item"] = {**row["item"], "name": "other_func"}
            if row["type"] == "response.completed":
                row["response"]["output"][0]["name"] = "other_func"
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_changed_call_id_at_done_is_rejected(self):
        rows = events("openai-responses", payload("openai-responses", tool=True))
        for row in rows:
            if row["type"] == "response.output_item.done":
                row["item"] = {**row["item"], "call_id": "call_different"}
            if row["type"] == "response.completed":
                row["response"]["output"][0]["call_id"] = "call_different"
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_completed_output_mismatch_is_rejected(self):
        rows = events("openai-responses", payload("openai-responses", tool=True))
        for row in rows:
            if row["type"] == "response.completed":
                row["response"]["output"] = []
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

        rows2 = events("openai-responses", payload("openai-responses", tool=True))
        for row in rows2:
            if row["type"] == "response.completed":
                extra = copy.deepcopy(row["response"]["output"][0])
                extra["id"] = "fc_extra"
                extra["call_id"] = "call_extra"
                row["response"]["output"].append(extra)
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(rows2)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_positive_multi_call_interleaved_and_done_only(self):
        call1 = {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "c1",
            "name": "read",
            "arguments": '{"path":"1.txt"}',
        }
        call2 = {
            "id": "fc_2",
            "type": "function_call",
            "call_id": "c2",
            "name": "write",
            "arguments": '{"path":"2.txt"}',
        }
        call3 = {
            "id": "fc_3",
            "type": "function_call",
            "call_id": "c3",
            "name": "edit",
            "arguments": '{"path":"3.txt"}',
        }
        rows = [
            {"type": "response.created", "response": {}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**call1, "arguments": ""},
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**call2, "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_1",
                "delta": '{"path":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_2",
                "delta": '{"path":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_1",
                "delta": '"1.txt"}',
            },
            {"type": "response.output_item.done", "output_index": 0, "item": call1},
            {"type": "response.output_item.done", "output_index": 1, "item": call2},
            {"type": "response.output_item.done", "output_index": 2, "item": call3},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [call1, call2, call3],
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                },
            },
        ]
        message = ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
        self.assertEqual(len(message["tool_calls"]), 3)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"path":"1.txt"}')
        self.assertEqual(message["tool_calls"][1]["function"]["name"], "write")
        self.assertEqual(message["tool_calls"][1]["function"]["arguments"], '{"path":"2.txt"}')
        self.assertEqual(message["tool_calls"][2]["function"]["name"], "edit")
        self.assertEqual(message["tool_calls"][2]["function"]["arguments"], '{"path":"3.txt"}')

    def test_responses_rejects_restarted_calls_and_arguments_before_start(self):
        first = {
            "type": "function_call",
            "id": "fc_same",
            "call_id": "call_same",
            "name": "wr",
            "arguments": '{"path":"unexpected.txt","content":',
        }
        final = {**first, "name": "ite", "arguments": '"fixture"}'}
        added = "response.output_item.added"
        done = "response.output_item.done"
        starts = [
            [
                {"type": added, "output_index": 0, "item": first},
                {"type": added, "output_index": 0, "item": final},
            ],
            [
                {"type": done, "output_index": 0, "item": first},
                {"type": added, "output_index": 0, "item": final},
            ],
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": first["id"],
                    "output_index": 0,
                    "delta": first["arguments"],
                },
                {"type": added, "output_index": 0, "item": {**final, "name": "write"}},
            ],
        ]
        for index, start in enumerate(starts):
            rows = [
                *start,
                {"type": done, "output_index": 0, "item": final},
                {"type": "response.completed", "response": {"output": [final]}},
            ]
            with self.subTest(sequence=index), self.assertRaises(CompletionError) as raised:
                ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
            self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_requires_complete_tool_identity_at_done_and_completion(self):
        for kind in ("response.output_item.done", "response.completed"):
            for field in ("id", "call_id", "name"):
                rows = events("openai-responses", payload("openai-responses", tool=True))
                for row in rows:
                    if row["type"] == kind:
                        if "item" in row:
                            row["item"] = dict(row["item"])
                            row["item"].pop(field)
                        else:
                            row["response"]["output"] = [dict(row["response"]["output"][0])]
                            row["response"]["output"][0].pop(field)
                with (
                    self.subTest(event=kind, field=field),
                    self.assertRaises(CompletionError) as raised,
                ):
                    ResponsesClient._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
                self.assertEqual(raised.exception.code, "invalid_tool_calls")

    def test_responses_refusal_streaming_and_nonstreaming(self):
        refusal_content = [{"type": "refusal", "refusal": "I cannot fulfill this request."}]
        refusal_rows = [
            {"type": "response.created", "response": {}},
            {"type": "response.refusal.delta", "output_index": 0, "delta": "I cannot fulfill "},
            {"type": "response.refusal.delta", "output_index": 0, "delta": "this request."},
            {
                "type": "response.refusal.done",
                "output_index": 0,
                "refusal": "I cannot fulfill this request.",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": refusal_content},
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "message", "role": "assistant", "content": refusal_content}
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                },
            },
        ]
        msg = ResponsesClient._read_stream(io.BytesIO(encode_sse(refusal_rows)), lambda _: None)
        self.assertEqual(msg["content"], "I cannot fulfill this request.")

        payload = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I refuse this task."}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
        client = ResponsesClient(self.settings("openai-responses"))
        obs = UsageAccumulator()
        stream = io.BytesIO(json.dumps(payload).encode())
        msg_nonstream = client._read_nonstream_response(stream, obs)
        self.assertEqual(msg_nonstream["content"], "I refuse this task.")

    def test_responses_malformed_refusal_is_rejected(self):
        bad_delta = [
            {"type": "response.refusal.delta", "output_index": 0, "delta": 123},
        ]
        with self.assertRaises(CompletionError) as raised:
            ResponsesClient._read_stream(io.BytesIO(encode_sse(bad_delta)), lambda _: None)
        self.assertEqual(raised.exception.code, "invalid_response")

        payload = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": ["not", "a", "string"]}],
                }
            ],
        }
        client = ResponsesClient(self.settings("openai-responses"))
        with self.assertRaises(CompletionError) as raised:
            stream = io.BytesIO(json.dumps(payload).encode())
            client._read_nonstream_response(stream, UsageAccumulator())
        self.assertEqual(raised.exception.code, "invalid_response")

    def test_login_switch_updates_search_and_keeps_policy_restrictions(self):
        settings = self.settings("openai-completions")
        tools = CodingTools(self.root)
        conversation = Conversation(settings, model_client(settings), tools)
        self.addCleanup(conversation.close)
        login = Login(
            "api",
            "deepseek-flash",
            "https://api.deepseek.com",
            "deepseek-key",
            api="anthropic-messages",
        )
        conversation.switch_login(login, model_client(settings.with_login(login)))
        self.assertEqual(conversation.settings.resolved_api, "anthropic-messages")
        self.assertIsInstance(conversation.client, AnthropicClient)
        search = tools.registry["websearch"].handler.__self__
        self.assertEqual(search.settings.resolved_websearch_api_key, "deepseek-key")
        self.assertEqual(search.settings.resolved_websearch_provider, "deepseek")
        self.assertIn("websearch", conversation._baseline_registry)
        other = Login("api", "other", "https://other.invalid", "other-key")
        conversation.switch_login(other, model_client(settings.with_login(other)))
        self.assertNotIn("websearch", tools.registry)
        self.assertNotIn("websearch", conversation._baseline_registry)


class ResponsesIdentityTests(unittest.TestCase):
    clients = (ResponsesClient, ChatGPTClient)

    def test_reversed_done_only_calls_match_final_output_by_identity(self):
        calls = [
            {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}",
             "name": "read", "arguments": json.dumps({"path": f"{i}.txt"})}
            for i in range(2)
        ]
        rows = [
            {"type": "response.output_item.done", "output_index": i, "item": calls[i]}
            for i in (1, 0)
        ]
        rows.append({"type": "response.completed", "response": {"output": calls}})
        for client in self.clients:
            with self.subTest(client=client.__name__):
                answer = client._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
                self.assertEqual([call["id"] for call in answer["tool_calls"]],
                                 ["call_0", "call_1"])

    def test_terminal_duplicate_identity_does_not_hide_a_missing_call(self):
        calls = [
            {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}",
             "name": "read", "arguments": "{}"}
            for i in range(2)
        ]
        rows = [
            {"type": "response.output_item.done", "output_index": i, "item": call}
            for i, call in enumerate(calls)
        ]
        rows.append({"type": "response.completed", "response": {"output": [calls[0]] * 2}})
        for client in self.clients:
            with self.subTest(client=client.__name__), self.assertRaises(CompletionError) as error:
                client._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
            self.assertEqual(error.exception.code, "invalid_tool_calls")

    def test_refusal_aliases_render_each_content_block_once(self):
        texts = ["No.", "Ask something else."]
        item = {"type": "message", "id": "msg", "role": "assistant", "content": [
            {"type": "refusal", "refusal": text} for text in texts
        ]}
        for bridge in ("done", "added"):
            rows = [{"type": "response.refusal.delta", "output_index": 0,
                     "content_index": 0, "delta": texts[0]}]
            if bridge == "added":
                rows.extend([
                    {"type": "response.output_item.added", "output_index": 0, "item": item},
                    {"type": "response.refusal.delta", "item_id": "msg",
                     "content_index": 1, "delta": texts[1]},
                ])
            rows.extend([
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"output": [item]}},
            ])
            for client in self.clients:
                shown = []
                with self.subTest(bridge=bridge, client=client.__name__):
                    answer = client._read_stream(io.BytesIO(encode_sse(rows)), shown.append)
                    self.assertEqual(answer["content"], "".join(texts))
                    self.assertEqual("".join(shown), answer["content"])

    def test_malformed_refusal_identifiers_are_structured_errors(self):
        cases = [(field, value) for field in ("content_index", "output_index")
                 for value in ({}, [], True, -1, "0", None)]
        cases.extend(("item_id", value) for value in ({}, [], 0, True, "", None))
        for field, value in cases:
            row = {"type": "response.refusal.delta", "item_id": "msg",
                   "output_index": 0, "content_index": 0, "delta": "No.", field: value}
            for client in self.clients:
                with self.subTest(field=field, value=value, client=client.__name__):
                    with self.assertRaises(CompletionError) as error:
                        client._read_stream(io.BytesIO(encode_sse([row])), lambda _: None)
                    self.assertEqual(error.exception.code, "invalid_response")

    def test_mixed_text_and_refusal_share_one_message_identity(self):
        for refusal_first in (False, True):
            parts = [
                {"type": "output_text", "text": "Prefix "},
                {"type": "refusal", "refusal": "No."},
            ]
            if refusal_first:
                parts.reverse()
            expected = "".join(part.get("text", part.get("refusal", "")) for part in parts)
            item = {"type": "message", "id": "msg", "role": "assistant", "content": parts}
            for text_id in ({"item_id": "msg"}, {"output_index": 0}):
                for refusal_delta in (False, True):
                    rows = [{
                        "type": "response.output_item.added", "output_index": 0,
                        "item": {**item, "content": []},
                    }]
                    for index, part in enumerate(parts):
                        if part["type"] == "output_text":
                            rows.append({
                                "type": "response.output_text.delta", **text_id,
                                "content_index": index, "delta": part["text"],
                            })
                        elif refusal_delta:
                            rows.append({
                                "type": "response.refusal.delta", "item_id": "msg",
                                "content_index": index, "delta": part["refusal"],
                            })
                        else:
                            rows.append({
                                "type": "response.refusal.done", "item_id": "msg",
                                "content_index": index, "refusal": part["refusal"],
                            })
                    rows.extend([
                        {"type": "response.output_item.done", "output_index": 0, "item": item},
                        {"type": "response.completed", "response": {"output": [item]}},
                    ])
                    for client in self.clients:
                        with self.subTest(
                            client=client.__name__, first=refusal_first,
                            text_id=text_id, delta=refusal_delta,
                        ):
                            shown = []
                            result = client._read_stream(io.BytesIO(encode_sse(rows)), shown.append)
                            self.assertEqual(result["content"], expected)
                            self.assertEqual("".join(shown), expected)
            payload = {"status": "completed", "output": [item]}
            result = object.__new__(ResponsesClient)._read_nonstream_response(
                io.BytesIO(json.dumps(payload).encode()), UsageAccumulator()
            )
            self.assertEqual(result["content"], expected)

    def test_mixed_messages_keep_paragraph_boundaries_and_unidentified_text(self):
        rows = [
            {"type": "response.output_text.delta", "item_id": "msg1", "delta": "Prefix "},
            {"type": "response.refusal.delta", "item_id": "msg1", "delta": "No."},
            {"type": "response.output_text.delta", "item_id": "msg2", "delta": "Next."},
            {"type": "response.completed", "response": {"output": []}},
        ]
        for client in self.clients:
            with self.subTest(client=client.__name__):
                result = client._read_stream(io.BytesIO(encode_sse(rows)), lambda _: None)
                self.assertEqual(result["content"], "Prefix No.\n\nNext.")
                anonymous = [
                    {"type": "response.output_text.delta", "delta": "Text."}, rows[-1]
                ]
                result = client._read_stream(io.BytesIO(encode_sse(anonymous)), lambda _: None)
                self.assertEqual(result["content"], "Text.")

    def test_nonstream_refusal_ids_use_the_same_validated_keys(self):
        for identity in ("msg", None, {}, [], 0):
            item = {"type": "message", "role": "assistant", "content": [
                {"type": "refusal", "refusal": "No."}
            ]}
            if identity is not None:
                item["id"] = identity
            data = io.BytesIO(json.dumps({"status": "completed", "output": [item]}).encode())
            client = object.__new__(ResponsesClient)
            with self.subTest(identity=identity):
                if identity in ("msg", None):
                    result = client._read_nonstream_response(data, UsageAccumulator())
                    self.assertEqual(result["content"], "No.")
                else:
                    with self.assertRaises(CompletionError) as error:
                        client._read_nonstream_response(data, UsageAccumulator())
                    self.assertEqual(error.exception.code, "invalid_response")


class ModelApiConfigTests(unittest.TestCase):
    def test_env_and_saved_login_protocols_and_compatibility_are_independent(self):
        with (
            patch("config.load_env"),
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "key",
                    "DEEPSEEK_API": "anthropic-messages",
                    "DEEPSEEK_API_COMPAT": '{"supports_store":false}',
                },
                clear=True,
            ),
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.resolved_api, "anthropic-messages")
            self.assertFalse(settings.model_compat.supports_store)
            saved = Settings.from_env(
                Login("api", "model", "https://other.invalid", "key", api="openai-responses")
            )
            self.assertEqual(saved.resolved_api, "openai-responses")
            self.assertTrue(saved.model_compat.supports_store)
        for invalid in ["wrong", "anthropic"]:
            with self.assertRaises(ValueError):
                parse_model_api(invalid)
        for invalid in ["[]", '{"unknown":true}', '{"supports_store":"false"}', "{broken"]:
            with self.assertRaises(ValueError):
                parse_api_compat(invalid)

    def test_auth_store_preserves_protocol_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AuthStore(Path(directory) / "auth.json")
            store.update(
                provider="api",
                api={"base_url": "https://proxy.invalid", "api_key": "key", "model": "model"},
            )
            self.assertEqual(store.login().api, "auto")
            store.update(
                api={
                    "base_url": "https://proxy.invalid",
                    "api_key": "key",
                    "model": "model",
                    "api": "anthropic-messages",
                    "api_compat": {"adaptive_thinking": True},
                }
            )
            self.assertEqual(store.login().api, "anthropic-messages")
            self.assertEqual(store.login().api_compat, {"adaptive_thinking": True})

    def test_anthropic_usage_deltas_do_not_reset_cached_input(self):
        from anthropic_client import anthropic_usage

        observed = UsageAccumulator()
        observed.add(
            {
                "usage": anthropic_usage(
                    {
                        "input_tokens": 3,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 2,
                        "output_tokens": 0,
                    }
                )
            }
        )
        observed.add({"usage": anthropic_usage({"output_tokens": 5})})
        self.assertEqual(observed.result()["prompt_tokens"], 12)
        self.assertEqual(observed.result()["completion_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
