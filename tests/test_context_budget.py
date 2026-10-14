import copy
import io
import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import Settings, environment_context
from context_budget import (
    context_snapshot,
    context_window,
    estimate_context,
    estimate_tokens,
    input_budget,
    wire_messages,
)
from context_management import compact_messages
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from task_runtime import RUNTIME_INSTRUCTIONS
from tests.support import make_settings
from ui import Console
from usage import CacheStats, UsageAccumulator


def stream(usage=None, finish="stop"):
    chunks = [
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "checked", "content": "done"},
                    "finish_reason": finish,
                }
            ]
        }
    ]
    if usage is not None:
        chunks += [
            {"choices": [], "usage": usage},
            {"choices": [], "usage": {"completion_tokens": None}},
        ]
    return io.BytesIO(
        b"".join(("data: " + json.dumps(c) + "\n").encode() for c in chunks) + b"data: [DONE]\n"
    )


class BudgetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.root / "session.json",
            workspace=self.root,
            max_context_chars=None,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=2,
            context_window_tokens=64000,
            context_reserve_tokens=8192,
            request_retries=0,
        )
        self.system = {"role": "system", "content": "Fixed instructions"}

    def response(self, messages, prompt=20000, output=1000, tools=None, client=None):
        client = client or DeepSeekClient(self.settings)
        with patch(
            "http_transport.open_url",
            return_value=stream(
                {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "prompt_cache_hit_tokens": prompt - 100,
                }
            ),
        ):
            return client.complete(messages, tools=tools, on_content=lambda _: None)

    def test_defaults_use_tokens_and_verified_route_capacity_without_network(self):
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test",
                    "DEEPSEEK_BASE_URL": "https://api.cline.bot/api/v1",
                    "DEEPSEEK_MODEL": "cline-pass/deepseek-v4.1-flash",
                },
                clear=True,
            ),
            patch("config.load_env"),
        ):
            settings = Settings.from_env()
        self.assertIsNone(settings.max_context_chars)
        self.assertEqual(context_window(settings), 1000000)
        self.assertEqual(input_budget(settings), 983616)
        self.assertEqual(settings.reasoning_effort, "max")
        self.assertEqual(
            context_window(replace(settings, base_url="https://custom.example/v1")), 131072
        )
        self.assertEqual(context_window(replace(settings, context_window_tokens=32000)), 32000)
        with self.assertRaises(ValueError):
            input_budget(replace(settings, context_window_tokens=1000))

    def test_usage_only_chunk_is_recorded_once_and_metadata_never_reaches_provider(self):
        client = DeepSeekClient(self.settings)
        initial = [self.system, {"role": "user", "content": "question"}]
        answer = self.response(initial, client=client)
        self.assertEqual(answer["_harness"]["usage"]["completion_tokens"], 1000)
        with patch("http_transport.open_url", return_value=stream()) as opened:
            client.complete(
                [*initial, answer, {"role": "user", "content": "continue"}],
                on_content=lambda _: None,
            )
        body = json.loads(opened.call_args.args[0].data)
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertEqual(body["messages"][2], wire_messages([answer])[0])
        self.assertNotIn("_harness", json.dumps(body))
        self.assertEqual(body["reasoning_effort"], "max")
        stats = client.cache_stats.snapshot()
        self.assertEqual(
            (stats["calls"], stats["observed_calls"], stats["unknown_calls"]), (2, 1, 1)
        )
        self.assertEqual(stats["hit_rate"], 19900 / 20000)
        self.assertIsNone(stats["latest"]["hit_rate"])

    def test_nonstream_wrapped_usage_is_retained(self):
        payload = {
            "data": {
                "choices": [
                    {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 0,
                },
            }
        }
        client = DeepSeekClient(self.settings)
        with patch(
            "http_transport.open_url", return_value=io.BytesIO(json.dumps(payload).encode())
        ):
            answer = client.complete([self.system, {"role": "user", "content": "question"}])
        self.assertEqual(answer["_harness"]["usage"]["prompt_tokens"], 500)
        self.assertEqual(client.cache_stats.snapshot()["hit_rate"], 0)

    def test_usage_anchor_counts_output_and_new_tool_result_without_counting_cache_twice(self):
        messages = [self.system, {"role": "user", "content": "x" * 120000}]
        answer = self.response(messages)
        trailing = {"role": "tool", "tool_call_id": "1", "content": "最新结果" * 100}
        restored = json.loads(json.dumps([*messages, answer, trailing]))
        estimate = estimate_context(restored, [], self.settings)
        self.assertEqual(estimate["source"], "usage")
        self.assertEqual(estimate["tokens"], 21000 + estimate_tokens(trailing))

    def test_edit_compaction_and_model_or_tool_changes_invalidate_old_usage(self):
        original = [self.system, {"role": "user", "content": "question"}]
        original.append(self.response(original))
        for kind in ("user", "assistant", "system", "model", "effort", "tools", "compaction"):
            with self.subTest(kind=kind):
                messages, settings, tools = copy.deepcopy(original), self.settings, []
                if kind in ("user", "assistant", "system"):
                    messages[{"system": 0, "user": 1, "assistant": 2}[kind]]["content"] += " edited"
                elif kind == "model":
                    settings = replace(settings, model="other")
                elif kind == "effort":
                    settings = replace(settings, reasoning_effort="high")
                elif kind == "tools":
                    tools = [{"type": "function", "function": {"name": "new", "parameters": {}}}]
                else:
                    messages.insert(1, {"role": "system", "content": "summary"})
                self.assertEqual(estimate_context(messages, tools, settings)["source"], "estimate")

    def test_child_response_cannot_anchor_parent_context(self):
        parent = [self.system, {"role": "user", "content": "parent"}]
        child = [self.system, {"role": "user", "content": "child"}]
        answer = self.response(child)
        self.assertEqual(
            estimate_context([*parent, answer], [], self.settings)["source"], "estimate"
        )
        self.assertEqual(estimate_context([*child, answer], [], self.settings)["source"], "usage")

    def test_zero_usage_falls_back_and_unknown_usage_can_use_an_earlier_anchor(self):
        messages = [self.system, {"role": "user", "content": "question"}]
        answer = self.response(messages, prompt=0, output=0)
        self.assertEqual(
            estimate_context([*messages, answer], [], self.settings)["source"], "estimate"
        )
        answer = self.response(messages)
        suffix = [{"role": "assistant", "content": "unmetered answer"}]
        estimate = estimate_context([*messages, answer, *suffix], [], self.settings)
        self.assertEqual(estimate["tokens"], 21000 + estimate_tokens(suffix[0]))

    def test_old_character_threshold_does_not_compact_a_fitting_token_context(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "x" * 150000},
        ]
        settings = replace(self.settings, context_window_tokens=1000000)
        before = copy.deepcopy(messages)
        client = DeepSeekClient(settings)
        with patch.object(client, "complete", side_effect=AssertionError("Unnecessary summary")):
            self.assertFalse(compact_messages(messages, self.system, [], client, settings))
        self.assertEqual(messages, before)

    def test_usage_threshold_triggers_compaction_and_preserves_task_and_tool_group(self):
        context = {
            "role": "system",
            "content": environment_context(self.root) + "\nAcceptance: keep interfaces",
        }
        messages = [
            {"role": "user", "content": "current task"},
            context,
            {"role": "assistant", "content": "large old evidence " * 10000},
        ]
        call = {"id": "last", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "last", "content": "recent evidence"},
        ]
        client = DeepSeekClient(self.settings)
        with patch.object(
            client, "complete", return_value={"role": "assistant", "content": "Verified summary"}
        ) as summary:
            self.assertTrue(
                compact_messages(
                    messages, self.system, [], client, self.settings, time.monotonic() + 60
                )
            )
        self.assertIn(context, messages)
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], messages[-1]["tool_call_id"])
        self.assertLess(
            estimate_context([self.system, *messages], [], self.settings)["tokens"],
            input_budget(self.settings),
        )
        self.assertGreater(summary.call_count, 0)
        before = copy.deepcopy(messages)
        with patch.object(client, "complete", side_effect=AssertionError("Repeated summary")):
            self.assertFalse(compact_messages(messages, self.system, [], client, self.settings))
        self.assertEqual(messages, before)

    def test_real_usage_can_trigger_compaction_even_when_character_estimate_is_small(self):
        messages = [
            {"role": "user", "content": "current task"},
            {"role": "assistant", "content": "evidence " * 3000},
        ]
        messages.append(self.response([self.system, *messages], prompt=55000, output=2000))
        client = DeepSeekClient(self.settings)
        with patch.object(
            client, "complete", return_value={"role": "assistant", "content": "Verified summary"}
        ):
            self.assertTrue(compact_messages(messages, self.system, [], client, self.settings))
        self.assertEqual(
            estimate_context([self.system, *messages], [], self.settings)["source"], "estimate"
        )

    def test_context_reserve_includes_pending_runtime_notes_and_tool_definitions(self):
        settings = replace(self.settings, context_window_tokens=16000, context_reserve_tokens=4000)
        messages = [
            {"role": "user", "content": "current task"},
            {"role": "assistant", "content": "x" * 30000},
        ]
        definitions = [
            {"type": "function", "function": {"name": "tool", "description": "x" * 1000}}
        ]
        client = DeepSeekClient(settings)
        with patch.object(
            client, "complete", return_value={"role": "assistant", "content": "summary"}
        ):
            self.assertTrue(
                compact_messages(
                    messages, self.system, definitions, client, settings, reserve_tokens=2000
                )
            )
        self.assertLess(
            estimate_context([self.system, *messages], definitions, settings)["tokens"] + 2000,
            input_budget(settings),
        )

    def test_output_budget_fits_remaining_window_without_changing_effort(self):
        messages = [self.system, {"role": "user", "content": "question"}]
        messages.append(self.response(messages, prompt=40000, output=2000))
        client = DeepSeekClient(self.settings)
        with patch("http_transport.open_url", return_value=stream()) as opened:
            client.complete(messages, on_content=lambda _: None)
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], 64000 - 42000 - 256)
        self.assertEqual(body["reasoning_effort"], "max")

    def test_stats_survive_session_restart_and_clear_without_entering_prompt(self):
        client = DeepSeekClient(self.settings)
        conversation = Conversation(self.settings, client, None)
        with patch(
            "http_transport.open_url",
            return_value=stream(
                {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_cache_hit_tokens": 900}
            ),
        ):
            conversation.ask("question")
        restored = Conversation(self.settings, DeepSeekClient(self.settings), None)
        self.assertEqual(restored.client.cache_stats.snapshot()["hit_rate"], 0.9)
        self.assertEqual(
            estimate_context(
                [
                    {
                        **restored.system_message,
                        "content": restored.system_message["content"]
                        + "\n\n"
                        + RUNTIME_INSTRUCTIONS,
                    },
                    *restored.messages,
                ],
                [],
                self.settings,
            )["source"],
            "usage",
        )
        restored.clear()
        self.assertEqual(restored.client.cache_stats.snapshot()["calls"], 0)

    def test_ui_context_snapshot_reuses_provider_usage_and_updates_after_clear(self):
        conversation = Conversation(self.settings, DeepSeekClient(self.settings), None)
        with patch(
            "http_transport.open_url",
            return_value=stream(
                {
                    "prompt_tokens": 20000,
                    "completion_tokens": 1000,
                }
            ),
        ):
            conversation.ask("What is 1 plus 1?")
        snapshot = conversation.context_status()
        self.assertEqual(snapshot["tokens"], 21000)
        self.assertEqual(snapshot["source"], "usage")
        self.assertEqual(snapshot["window_tokens"], 64000)
        self.assertEqual(snapshot["budget_tokens"], 55808)
        conversation.clear()
        cleared = conversation.context_status()
        self.assertEqual(cleared["source"], "estimate")
        self.assertLess(cleared["tokens"], snapshot["tokens"])

    def test_context_snapshot_reports_character_mode_without_changing_token_display(self):
        settings = replace(self.settings, max_context_chars=120000)
        snapshot = context_snapshot([self.system], [], settings)
        self.assertEqual(snapshot["max_chars"], 120000)
        self.assertGreater(snapshot["chars"], 0)
        self.assertEqual(snapshot["window_tokens"], 64000)


class UsageTests(unittest.TestCase):
    def test_gateway_metadata_zero_and_trailing_nulls(self):
        usage = UsageAccumulator()
        usage.add(
            {
                "choices": [
                    {
                        "delta": {
                            "provider_metadata": {
                                "deepseek": {"promptCacheHitTokens": 0, "promptCacheMissTokens": 20}
                            }
                        }
                    }
                ]
            }
        )
        usage.add({"usage": {"completion_tokens": 5}})
        usage.add({"usage": {"completion_tokens": None}})
        self.assertEqual(usage.result()["prompt_tokens"], 20)
        self.assertEqual(usage.result()["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(usage.result()["completion_tokens"], 5)

    def test_stats_are_weighted_and_thread_safe(self):
        stats = CacheStats()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda _: stats.record(
                        {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 900}},
                        "solve",
                    ),
                    range(100),
                )
            )
        stats.record(
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}, "summary"
        )
        summary = stats.snapshot()
        self.assertEqual(summary["calls"], 101)
        self.assertEqual(summary["hit_rate"], 90000 / 100100)
        self.assertEqual(summary["latest"]["hit_rate"], 0)
        output = io.StringIO()
        with patch("sys.stdout", output):
            Console().cache(summary)
        self.assertIn("最近一次（摘要） 0.00%", output.getvalue())
        self.assertIn("会话累计 89.91%", output.getvalue())

    def test_failed_and_retried_attempts_are_counted_without_inventing_cache_misses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = make_settings(
                api_key="test",
                base_url="https://example.invalid",
                model="test",
                session_file=root / "session.json",
                workspace=root,
                max_context_chars=None,
                keep_recent_turns=6,
                max_task_seconds=60,
                max_subagents=1,
                request_retries=1,
                retry_backoff_seconds=0,
            )
            client = DeepSeekClient(settings)
            with patch(
                "http_transport.open_url",
                side_effect=[
                    io.BytesIO(b""),
                    stream({"prompt_tokens": 100, "prompt_cache_hit_tokens": 80}),
                ],
            ):
                client.complete([], on_content=lambda _: None)
            stats = client.cache_stats.snapshot()
            self.assertEqual(
                (stats["calls"], stats["observed_calls"], stats["unknown_calls"]), (2, 1, 1)
            )
            self.assertEqual(stats["hit_rate"], 0.8)
            with (
                patch(
                    "http_transport.open_url",
                    return_value=stream(
                        {"prompt_tokens": 100, "prompt_cache_hit_tokens": 90}, finish="length"
                    ),
                ),
                self.assertRaises(CompletionError),
            ):
                client.complete([], on_content=lambda _: None)
            self.assertEqual(client.cache_stats.snapshot()["calls"], 3)


if __name__ == "__main__":
    unittest.main()
