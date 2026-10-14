import copy
import io
import json
import os
import random
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import Settings, environment_context
from context_budget import (
    WireCache,
    context_snapshot,
    context_window,
    estimate_context,
    estimate_tokens,
    fingerprint,
    input_budget,
    request_fingerprint,
    wire_messages,
)
from context_management import compact_messages
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from task_runtime import RUNTIME_INSTRUCTIONS
from tests.support import make_settings
from tools import CodingTools
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


def tool_stream(call_id, usage):
    call = {"index": 0, "id": call_id, "function": {"name": "ls", "arguments": "{}"}}
    chunks = [
        {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": usage},
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

    def test_task_rounds_share_the_conversation_cache_without_changing_any_value(self):
        conversation = Conversation(
            self.settings, DeepSeekClient(self.settings), CodingTools(self.root)
        )
        cache = conversation._wire_cache
        sources = []

        def checked_snapshot(messages, tools, settings, wire_cache=None):
            self.assertIs(wire_cache, cache)
            expected = context_snapshot(messages, tools, settings)
            self.assertEqual(context_snapshot(messages, tools, settings, wire_cache), expected)
            sources.append(expected["source"])
            return expected

        def checked_estimate(messages, tools, settings, wire_cache=None):
            self.assertIs(wire_cache, cache)
            expected = estimate_context(messages, tools, settings)
            self.assertEqual(estimate_context(messages, tools, settings, wire_cache), expected)
            return expected

        usage = {"prompt_tokens": 3000, "completion_tokens": 40}
        responses = [tool_stream(f"call-{index}", usage) for index in range(3)] + [stream(usage)]
        with (
            patch("http_transport.open_url", side_effect=responses),
            patch("task_runtime.context_snapshot", checked_snapshot),
            patch("conversation.context_snapshot", checked_snapshot),
            patch("context_management.estimate_context", checked_estimate),
        ):
            answer = conversation.ask("List the workspace.", on_progress=lambda _: None)
        self.assertEqual(answer, "done")
        # Per-round snapshots from TaskRuntime plus one after each tool result and answer.
        self.assertGreaterEqual(len(sources), 8)
        self.assertIn("usage", sources)


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


class WireCacheTests(unittest.TestCase):
    def _settings(self):
        return make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=Path(tempfile.gettempdir()) / "unused-session.json",
            workspace=Path(tempfile.gettempdir()),
            max_context_chars=None,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=2,
            context_window_tokens=64000,
            context_reserve_tokens=8192,
        )

    def _messages(self):
        return [
            {"role": "system", "content": '系统 prompt with "quotes" and \\slash'},
            {"role": "user", "content": "hello\nworld"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "exec", "arguments": '{"x":1}'}}],
                "_harness": {"usage": {"prompt_tokens": 10}},
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true,"emoji":"✅"}'},
            {"role": "assistant", "content": "done", "reasoning_content": "思考"},
            {"role": "user", "content": ""},
        ]

    def test_request_fingerprint_is_identical_with_and_without_cache(self):
        settings = self._settings()
        messages = self._messages()
        cache = WireCache()
        for tools in (None, [], [{"type": "function", "function": {"name": "exec"}}]):
            for index in range(len(messages) + 1):
                prefix = messages[:index]
                self.assertEqual(
                    request_fingerprint(prefix, tools, settings),
                    request_fingerprint(prefix, tools, settings, cache),
                )

    def test_context_snapshot_is_identical_with_and_without_cache(self):
        settings = self._settings()
        messages = self._messages()
        for tools in (None, [], [{"type": "function", "function": {"name": "exec"}}]):
            self.assertEqual(
                context_snapshot(messages, tools, settings),
                context_snapshot(messages, tools, settings, WireCache()),
            )

    def test_pretty_list_len_matches_default_separator_dump(self):
        cache = WireCache()
        messages = self._messages()
        for subset in ([], messages[:1], messages[:3], messages):
            self.assertEqual(
                cache.pretty_list_len(subset),
                len(json.dumps(wire_messages(subset), ensure_ascii=False)),
            )

    def test_cache_tracks_appends_replacements_and_prunes_dead_entries(self):
        settings = self._settings()
        cache = WireCache()
        messages = self._messages()
        context_snapshot(messages, None, settings, cache)
        messages.append({"role": "user", "content": "新消息"})
        self.assertEqual(
            request_fingerprint(messages, None, settings),
            request_fingerprint(messages, None, settings, cache),
        )
        # A full replacement (as compaction produces) must not reuse stale serializations.
        replaced = [{"role": "system", "content": "新的 system"}, {"role": "user", "content": "q"}]
        self.assertEqual(
            request_fingerprint(replaced, None, settings),
            request_fingerprint(replaced, None, settings, cache),
        )
        context_snapshot(replaced, None, settings, cache)
        self.assertLessEqual(len(cache._entries), len(replaced))

    def _anchor(self, messages, tools, settings, response, prompt_tokens=None):
        # The metadata DeepSeekClient._attach_usage records for a provider response.
        observed = UsageAccumulator()
        prompt = 500 + len(messages) if prompt_tokens is None else prompt_tokens
        observed.add({"usage": {"prompt_tokens": prompt, "completion_tokens": 7}})
        client = SimpleNamespace(settings=settings)
        return DeepSeekClient._attach_usage(
            client, response, wire_messages(messages), tools, observed
        )

    def test_sizes_tokens_and_message_fingerprints_match_uncached_for_unusual_text(self):
        cache = WireCache()
        messages = [*self._messages(), *UNUSUAL_MESSAGES, LONE_SURROGATE]
        for start in range(len(messages)):
            for end in range(start, len(messages) + 1):
                subset = messages[start:end]
                visible = wire_messages(subset)
                self.assertEqual(
                    cache.pretty_list_len(subset), len(json.dumps(visible, ensure_ascii=False))
                )
                self.assertEqual(cache.list_tokens(subset), estimate_tokens(visible))
        for message in messages:
            self.assertEqual(
                cache.message_tokens(message), estimate_tokens(wire_messages([message])[0])
            )
        for message in messages[:-1]:
            self.assertEqual(
                cache.message_fingerprint(message), fingerprint(wire_messages([message])[0])
            )

    def test_prefix_fingerprint_matches_uncached_across_extension_divergence_and_head_changes(
        self,
    ):
        settings = self._settings()
        tools = [{"type": "function", "function": {"name": "exec"}}]
        messages = [*self._messages(), *UNUSUAL_MESSAGES]
        cache = WireCache()

        def check(prefix, tools=tools, settings=settings):
            self.assertEqual(
                request_fingerprint(prefix, tools, settings, cache),
                request_fingerprint(prefix, tools, settings),
            )

        for index in range(len(messages) + 1):  # A growing session extends the hashed prefix.
            check(messages[:index])
        for index in range(len(messages), -1, -1):  # Shorter prefixes reuse stored states.
            check(messages[:index])
        check([*messages[:3], {"role": "user", "content": "diverged"}, *messages[4:]])
        check(messages)
        check([dict(messages[0]), *messages[1:]])  # Equal content in a new object.
        check(messages, tools=None)
        check(messages, settings=replace(settings, model="other"))
        check(messages)
        unencodable = [*messages[:2], LONE_SURROGATE]
        for memo in (None, cache):
            with self.assertRaises(UnicodeEncodeError):
                request_fingerprint(unencodable, tools, settings, memo)
        check(messages)  # A failed extension leaves the stored states consistent.
        check(messages[:2])

    def test_estimate_context_matches_uncached_for_anchor_and_fallback_paths(self):
        settings = self._settings()
        tools = [{"type": "function", "function": {"name": "exec"}}]
        call = {"id": "c1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
        history = [
            {"role": "system", "content": "系统 instructions"},
            {"role": "user", "content": "q"},
        ]
        history.append(
            self._anchor(
                history,
                tools,
                settings,
                {"role": "assistant", "content": None, "tool_calls": [call]},
            )
        )
        history.append({"role": "tool", "tool_call_id": "c1", "content": "结果 ✅"})
        history.append(
            self._anchor(history, tools, settings, {"role": "assistant", "content": "答"})
        )
        history.append({"role": "user", "content": "follow-up after the anchor"})
        unmetered = self._anchor(history, tools, settings, {"role": "assistant", "content": "n/a"})
        unmetered["_harness"]["usage"].pop("completion_tokens")
        edited = {**history[4], "content": "edited after anchoring"}
        scenarios = {
            "latest anchor": ([*history], 4),
            "anchor without completion tokens": ([*history, unmetered], 6),
            "edited response falls back to an earlier anchor": (
                [*history[:4], edited, history[5]],
                2,
            ),
            "no anchor": (wire_messages(history), None),
        }
        cache = WireCache()
        for name, (messages, anchor_index) in scenarios.items():
            expected = estimate_context(messages, tools, settings)
            self.assertEqual(expected["anchor_index"], anchor_index, name)
            for tool_set in (tools, None):
                with self.subTest(name, tools=tool_set):
                    self.assertEqual(
                        estimate_context(messages, tool_set, settings, cache),
                        estimate_context(messages, tool_set, settings),
                    )

    def test_sync_bounds_entries_and_hash_states_to_the_live_history(self):
        settings = self._settings()
        messages = [*self._messages(), *UNUSUAL_MESSAGES]
        cache = WireCache()
        context_snapshot(messages, None, settings, cache)
        request_fingerprint(messages, None, settings, cache)
        self.assertEqual(len(cache._chain_parts), len(messages))
        cache.sync(messages[:3])  # A shorter live history drops the hashed tail.
        self.assertEqual(len(cache._chain_parts), 3)
        self.assertEqual(len(cache._chain_states), 4)
        request_fingerprint(messages, None, settings, cache)
        # Compaction: a rebuilt system message with equal content, a summary, a recent tail.
        live = [dict(messages[0]), {"role": "system", "content": "Summary"}, *messages[-2:]]
        cache.sync(live)
        self.assertLessEqual(set(cache._entries), {id(message) for message in live})
        self.assertEqual(len(cache._chain_parts), 1)
        self.assertEqual(len(cache._chain_states), 2)
        self.assertIs(cache._chain_parts[0], cache._entries[id(live[0])][1])
        self.assertEqual(
            request_fingerprint(live, None, settings, cache),
            request_fingerprint(live, None, settings),
        )

    def test_random_session_histories_match_uncached(self):
        # Differential check over appends, anchors, edits, compaction-style replacement,
        # truncation, repeated objects, metadata-only updates and settings/tool changes.
        rng = random.Random(20261014)
        base = self._settings()
        settings_choices = [base, replace(base, reasoning_effort="high"), replace(base, model="m")]
        tool_choices = [None, [], [{"type": "function", "function": {"name": "exec"}}]]
        texts = ["plain", "中文内容", "emoji 🙂", "naïve", 'q"b\\s', "\x00ctl", "x" * 300, ""]
        settings, tools, system_text = base, tool_choices[2], "System 系统"
        history: list = [{"role": "user", "content": "task"}]
        cache = WireCache()
        for step in range(400):
            op = rng.randrange(10)
            # A fresh system object each step, as Conversation.context_status builds one.
            system = {"role": "system", "content": system_text}
            if op == 0:
                role = rng.choice(["user", "tool", "system"])
                history.append({"role": role, "content": rng.choice(texts)})
            elif op == 1:
                response = {"role": "assistant", "content": rng.choice(texts)}
                history.append(self._anchor([system, *history], tools, settings, response))
            elif op == 2 and history:
                index = rng.randrange(len(history))
                history[index] = {**history[index], "content": rng.choice(texts) + "!"}
            elif op == 3 and len(history) > 3:
                tail = history[-rng.randint(1, 3) :]
                history[:] = [{"role": "system", "content": f"Summary {step}"}, *tail]
            elif op == 4 and history:
                del history[rng.randrange(len(history)) :]
            elif op == 5:
                tools = rng.choice(tool_choices)
            elif op == 6:
                settings = rng.choice(settings_choices)
            elif op == 7 and history:
                history.append(rng.choice(history))
            elif op == 8:
                system_text = rng.choice(["System 系统", "Reloaded prompt"])
            elif op == 9 and history:
                rng.choice(history).setdefault("_harness", {})["performance"] = step
            live = [system, *history]
            with self.subTest(step=step, op=op):
                self.assertEqual(
                    context_snapshot(live, tools, settings, cache),
                    context_snapshot(live, tools, settings),
                )
                end = rng.randint(0, len(live))
                self.assertEqual(
                    request_fingerprint(live[:end], tools, settings, cache),
                    request_fingerprint(live[:end], tools, settings),
                )
                start = rng.randint(0, len(live))
                part = live[start : rng.randint(start, len(live))]
                self.assertEqual(cache.list_tokens(part), estimate_tokens(wire_messages(part)))
                self.assertLessEqual(set(cache._entries), {id(message) for message in live})
                self.assertLessEqual(len(cache._chain_parts), len(live))
                self.assertEqual(len(cache._chain_states), len(cache._chain_parts) + 1)

    def _add_rounds(self, messages, system, definitions, settings, turn):
        # A user turn of bulky tool rounds, each response anchored with realistic usage.
        messages.append({"role": "user", "content": f"task {turn}"})
        for round_index in range(8):
            call_id = f"c{turn}-{round_index}"
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": "exec", "arguments": ""},
            }
            prefix = [system, *messages]
            response = {"role": "assistant", "content": None, "tool_calls": [call]}
            prompt = estimate_context(prefix, definitions, settings)["tokens"]
            messages.append(self._anchor(prefix, definitions, settings, response, prompt))
            content = f"evidence {turn}/{round_index} 证据 " * 300
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def test_compaction_matches_uncached_and_keeps_the_cache_within_the_live_history(self):
        system = {"role": "system", "content": "Fixed instructions 系统"}
        definitions = [{"type": "function", "function": {"name": "exec", "description": "run"}}]
        base = self._settings()
        modes = {
            "tokens": replace(base, context_window_tokens=40000, context_reserve_tokens=1000),
            "characters": replace(base, max_context_chars=60000),
        }
        for mode, settings in modes.items():
            original: list = []
            for turn in range(3):
                self._add_rounds(original, system, definitions, settings, turn)
            runs = []
            for cache in (None, WireCache()):
                messages, events, compacted = copy.deepcopy(original), [], []
                for compaction in range(2):
                    for turn in range(7, 10) if compaction else ():
                        # Grow again; the second pass reuses the same cache.
                        self._add_rounds(messages, system, definitions, settings, turn)
                    with patch(
                        "context_management.summarize",
                        side_effect=lambda _c, _s, source, *_: f"Summary of {len(source)}",
                    ):
                        compacted.append(
                            compact_messages(
                                messages,
                                system,
                                definitions,
                                SimpleNamespace(),
                                settings,
                                on_progress=events.append,
                                reserve_chars=100,
                                reserve_tokens=100,
                                cache=cache,
                            )
                        )
                    if cache is not None:
                        live = [system, *messages]
                        self.assertLessEqual(set(cache._entries), {id(m) for m in live})
                        self.assertLessEqual(len(cache._chain_parts), len(live))
                runs.append((compacted, wire_messages(messages), events))
            with self.subTest(mode):
                self.assertEqual(runs[0][0], [True, True])
                self.assertEqual(runs[1], runs[0])


UNUSUAL_MESSAGES = [
    {"role": "user", "content": "naïve café — ünïcödé"},
    {"role": "user", "content": "emoji 🙂👍🏽, 中文, and a   line separator"},
    {"role": "tool", "tool_call_id": "t", "content": 'ctrl \x00\x1f\t\n quote " back \\ slash'},
    {"role": "assistant", "content": None, "n": 1, "f": 1.5, "b": True, "z": None, "x": [["é"]]},
    {"content": "same keys, different order", "role": "user"},
    {},
    {"role": "user", "content": ""},
]
# Counted like any other character, but not UTF-8 encodable: fingerprinting must fail
# the same way with and without a cache.
LONE_SURROGATE = {"role": "user", "content": "broken \ud800 escape"}


if __name__ == "__main__":
    unittest.main()
