import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from context_management import OutputRecovery
from conversation import Conversation
from deepseek_client import CompletionError
from skills import SkillCatalog
from tests.support import make_settings
from tools import CodingTools
from usage import CacheStats


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        self.requests.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "tools": tools,
                "timeout": timeout,
                **kwargs,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if on_content and response.get("content"):
            on_content(response["content"])
        return response


class InterruptingTools(CodingTools):
    def execute(self, tool_call, max_timeout=None, *, deadline=None):
        raise KeyboardInterrupt


def tool_call(call_id="call-1", arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ls", "arguments": arguments},
    }


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def settings(self, **overrides):
        values = {
            "api_key": "test",
            "base_url": "https://example.invalid",
            "model": "test",
            "session_file": self.workspace / "state" / "session.json",
            "max_context_chars": 120_000,
            "keep_recent_turns": 6,
            "max_task_seconds": 3_600,
            "max_subagents": 4,
            "workspace": self.workspace,
        }
        values.update(overrides)
        return make_settings(**values)

    def test_busy_operations_do_not_run_or_release_another_operation(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        operations = (
            ("ask", "_ask", ("task",), "任务或环境切换正在运行"),
            ("resume", "_resume", (conversation.session_id,), "不能恢复会话"),
            ("apply_environment", "_apply_environment", (None,), "不能切换环境"),
        )
        conversation._operation.acquire()
        try:
            for method, implementation, arguments, error in operations:
                with (
                    self.subTest(operation=method),
                    patch.object(conversation, implementation) as handler,
                ):
                    with self.assertRaisesRegex(RuntimeError, error):
                        getattr(conversation, method)(*arguments)
                    handler.assert_not_called()
                    self.assertTrue(conversation._operation.locked())
            self.assertEqual(conversation.messages, [])
            self.assertFalse(conversation.settings.session_file.exists())
        finally:
            conversation._operation.release()

    def test_operation_lock_is_released_after_failure_or_interruption(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        operations = (
            ("ask", "_ask", ("task",), "done"),
            ("resume", "_resume", (conversation.session_id,), None),
            ("apply_environment", "_apply_environment", (None,), []),
        )
        for method, implementation, arguments, result in operations:
            for error in (RuntimeError, KeyboardInterrupt):
                with self.subTest(operation=method, error=error.__name__):
                    failure = error("operation failed")
                    with patch.object(conversation, implementation, side_effect=failure):
                        with self.assertRaises(error) as raised:
                            getattr(conversation, method)(*arguments)
                        self.assertIs(raised.exception, failure)
                    self.assertFalse(conversation._operation.locked())
                    with patch.object(conversation, implementation, return_value=result):
                        self.assertEqual(getattr(conversation, method)(*arguments), result)

    def test_tool_result_callbacks_observe_persisted_batch_order(self):
        client = FakeClient(
            {
                "role": "assistant",
                "tool_calls": [tool_call("call-1"), tool_call("call-2")],
            },
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        observed = []

        def record(name, result):
            data = conversation.sessions.read()
            message = data["messages"][-1]
            self.assertEqual(result, json.loads(message["content"]))
            self.assertEqual(data["last_outcome"], conversation.last_outcome)
            observed.append(
                (name, [run["state"] for run in data["tool_runs"]], message["tool_call_id"])
            )

        self.assertEqual(conversation.ask("go", on_tool_result=record), "done")
        self.assertEqual(
            observed,
            [
                ("ls", ["completed", "pending"], "call-1"),
                ("ls", ["completed", "completed"], "call-2"),
            ],
        )

    def test_tool_callback_failure_does_not_interrupt_persisted_batch(self):
        client = FakeClient(
            {
                "role": "assistant",
                "tool_calls": [tool_call("call-1"), tool_call("call-2")],
            },
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))

        def fail(name, result):
            raise RuntimeError("callback failed")

        self.assertEqual(conversation.ask("go", on_tool_result=fail), "done")
        self.assertFalse(conversation._operation.locked())
        self.assertEqual(
            [run["state"] for run in conversation.sessions.read()["tool_runs"]],
            ["completed", "completed"],
        )
        restored = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        self.assertEqual([run["state"] for run in restored.tool_runs], ["completed", "completed"])
        results = [
            json.loads(message["content"])
            for message in restored.messages
            if message["role"] == "tool"
        ]
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["status"], "ok")

    def test_tool_callback_keyboard_interrupt_keeps_completed_and_pending_results(self):
        client = FakeClient(
            {"role": "assistant", "tool_calls": [tool_call("call-1"), tool_call("call-2")]}
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))

        def pause(name, result):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            conversation.ask("go", on_tool_result=pause)
        restored = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        self.assertEqual(
            [run["state"] for run in restored.tool_runs], ["completed", "not_executed"]
        )

    def test_display_failures_cannot_fail_or_replay_a_successful_write(self):
        operation = {
            "id": "write-once",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps({"path": "effect.txt", "content": "once"}),
            },
        }
        client = FakeClient(
            {"role": "assistant", "tool_calls": [operation]},
            {"role": "assistant", "content": "saved", "reasoning": "fixture reasoning"},
        )
        failed = []

        def callback(label):
            def display(*args):
                failed.append(label)
                raise RuntimeError("display fixture")

            return display

        tools = CodingTools(self.workspace, on_tool=callback("tool_start"))
        conversation = Conversation(self.settings(), client, tools)
        self.assertEqual(
            conversation.ask(
                "Save fixture",
                on_content=callback("content"),
                on_reasoning=callback("reasoning"),
                on_response_start=callback("start"),
                on_response_end=callback("end"),
                on_progress=callback("progress"),
                on_tool_result=callback("result"),
            ),
            "saved",
        )
        self.assertEqual((self.workspace / "effect.txt").read_text(), "once")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(conversation.tool_runs), 1)
        self.assertEqual(json.loads(conversation.tool_runs[0]["result"])["status"], "ok")
        self.assertEqual(
            set(failed),
            {"tool_start", "content", "reasoning", "start", "end", "progress", "result"},
        )

    def test_plan_survives_restart_and_clear_removes_all_task_state(self):
        plan = {"title": "检查任务", "steps": [{"step": "读取目录", "status": "in_progress"}]}
        client = FakeClient(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "plan",
                        "type": "function",
                        "function": {
                            "name": "update_plan",
                            "arguments": json.dumps(plan),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        conversation.ask("task")
        restored = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        self.assertEqual(restored.tools.task_plan, plan)
        self.assertEqual(restored.tools.task_title, plan["title"])
        restored.clear()
        self.assertEqual(restored.messages, [])
        self.assertEqual(restored.tool_runs, [])
        self.assertIsNone(restored.last_outcome)
        self.assertIsNone(restored.recovery_notice)
        self.assertIsNone(restored.tools.task_plan)
        self.assertEqual(restored.tools.task_title, "")
        self.assertFalse(restored.settings.session_file.exists())

    def test_new_turn_replaces_previous_task_title_and_plan(self):
        client = FakeClient({"role": "assistant", "content": "done"})
        tools = CodingTools(self.workspace)
        conversation = Conversation(self.settings(), client, tools)
        tools.task_title = "旧任务"
        tools.task_plan = {"title": "旧任务", "steps": [{"step": "旧步骤", "status": "pending"}]}
        conversation.ask("新的\n任务")
        self.assertEqual(tools.task_title, "新的 任务")
        self.assertIsNone(tools.task_plan)

    def test_malformed_arguments_receive_a_matching_tool_response(self):
        client = FakeClient(
            {"role": "assistant", "content": None, "tool_calls": [tool_call(arguments="{bad")]},
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        observed_results = []

        self.assertEqual(
            conversation.ask(
                "go", on_tool_result=lambda name, result: observed_results.append((name, result))
            ),
            "done",
        )
        tool_message = next(
            message for message in conversation.messages if message["role"] == "tool"
        )
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertEqual(json.loads(tool_message["content"])["error"]["code"], "invalid_arguments")
        self.assertEqual(observed_results[0][0], "ls")
        self.assertEqual(observed_results[0][1]["error"]["code"], "invalid_arguments")

    def test_user_selection_returns_to_the_model_and_conversation_continues(self):
        options = [
            {"label": "SQLite", "description": "无需独立服务。"},
            {"label": "PostgreSQL", "description": "适合多用户部署。"},
        ]
        question_call = {
            "id": "call-question",
            "type": "function",
            "function": {
                "name": "ask_user",
                "arguments": json.dumps({"question": "选择数据库", "options": options}),
            },
        }
        client = FakeClient(
            {"role": "assistant", "content": None, "tool_calls": [question_call]},
            {"role": "assistant", "content": "将使用 PostgreSQL。"},
        )
        tools = CodingTools(self.workspace)
        tools.attach_user_input(
            lambda question, choices: {
                "answer": choices[1]["label"],
                "selected_index": 2,
                "selected_label": choices[1]["label"],
                "custom": False,
            }
        )

        answer = Conversation(self.settings(), client, tools).ask("创建应用")

        self.assertEqual(answer, "将使用 PostgreSQL。")
        result = json.loads(client.requests[1]["messages"][-1]["content"])
        self.assertEqual(result["data"]["answer"], "PostgreSQL")

    def test_tools_remain_available_beyond_former_limit(self):
        client = FakeClient(
            *[
                {"role": "assistant", "content": None, "tool_calls": [tool_call(f"call-{i}")]}
                for i in range(45)
            ],
            {"role": "assistant", "content": "finished"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("go"), "finished")
        self.assertEqual(len([m for m in conversation.messages if m["role"] == "tool"]), 45)
        self.assertTrue(all(r["tools"] for r in client.requests))

    def test_unlimited_task_can_finish_after_the_default_deadline(self):
        client = FakeClient(
            {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
            {"role": "assistant", "content": "finished"},
        )
        conversation = Conversation(
            self.settings(max_task_seconds=None), client, CodingTools(self.workspace)
        )
        complete = client.complete
        with patch("time.monotonic", return_value=100) as clock:

            def advance_clock(*args, **kwargs):
                result = complete(*args, **kwargs)
                clock.return_value += 10000
                return result

            with patch.object(client, "complete", side_effect=advance_clock):
                self.assertEqual(conversation.ask("inspect the directory"), "finished")
        self.assertTrue(all(r["deadline"] == float("inf") for r in client.requests))
        self.assertTrue(all(r["purpose"] == "solve" for r in client.requests))
        self.assertEqual(conversation.last_outcome["status"], "completed")

    def test_interruption_closes_every_tool_call_in_the_session(self):
        client = FakeClient(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call("call-1"), tool_call("call-2")],
            }
        )
        conversation = Conversation(self.settings(), client, InterruptingTools(self.workspace))

        with self.assertRaises(KeyboardInterrupt):
            conversation.ask("go")

        tool_messages = [message for message in conversation.messages if message["role"] == "tool"]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages], ["call-1", "call-2"]
        )
        self.assertIn("interrupted", tool_messages[0]["content"])
        self.assertIn("skipped", tool_messages[1]["content"])

    def test_tool_results_reach_the_model_as_valid_json_with_complete_artifact(self):
        client = FakeClient(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "large-output",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": json.dumps({"command": "printf '%020000d' 0"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        conversation.ask("go")
        tool_message = next(
            message for message in client.requests[-1]["messages"] if message["role"] == "tool"
        )
        result = json.loads(tool_message["content"])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["truncated"])
        self.assertEqual(Path(result["artifact_path"]).read_text(), "0" * 20000)

    def test_prompt_and_api_share_the_actual_registry(self):
        client = FakeClient({"role": "assistant", "content": "done"})
        conversation = Conversation(
            self.settings(max_subagents=1), client, CodingTools(self.workspace)
        )
        conversation.ask("go")
        definitions = {d["function"]["name"]: d["function"] for d in client.requests[0]["tools"]}
        self.assertEqual(set(definitions), set(conversation.tools.registry))
        self.assertEqual(
            definitions["delegate"]["parameters"]["properties"]["tasks"]["maxItems"], 1
        )
        advertised = (
            conversation.system_message["content"].split("Available tools: ")[1].split(".\n")[0]
        )
        self.assertEqual(set(advertised.split(", ")), set(definitions))

    def test_skill_metadata_is_advertised_and_explicit_command_is_expanded(self):
        skills_dir = self.workspace / "skills"
        skill_dir = skills_dir / "review"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: review\ndescription: Review code changes.\n---\n\n"
            "Inspect the diff before responding.\n",
            encoding="utf-8",
        )
        client = FakeClient({"role": "assistant", "content": "done"})
        conversation = Conversation(
            self.settings(),
            client,
            CodingTools(self.workspace),
            skills=SkillCatalog(self.workspace, skills_dir),
        )

        conversation.ask("/skill:review focus on correctness")

        system = client.requests[0]["messages"][0]["content"]
        user = next(
            message for message in client.requests[0]["messages"] if message["role"] == "user"
        )
        self.assertIn("<available_skills>", system)
        self.assertIn(str(skill_file.resolve()), system)
        self.assertNotIn("Inspect the diff before responding.", system)
        self.assertIn("Inspect the diff before responding.", user["content"])
        self.assertIn("User request:\nfocus on correctness", user["content"])

    def test_save_is_loadable_and_leaves_no_temporary_file(self):
        settings = self.settings()
        conversation = Conversation(settings, FakeClient(), CodingTools(self.workspace))
        conversation.messages = [{"role": "user", "content": "hello"}]
        conversation.save()

        self.assertEqual(
            json.loads(settings.session_file.read_text())["messages"], conversation.messages
        )
        self.assertEqual(list(settings.session_file.parent.glob("*.tmp")), [])

    def test_invalid_session_has_a_clear_error(self):
        settings = self.settings()
        settings.session_file.parent.mkdir(parents=True)
        settings.session_file.write_text("{bad")
        with self.assertRaisesRegex(RuntimeError, "Cannot load session file"):
            Conversation(settings, FakeClient(), CodingTools(self.workspace))

    def test_resume_restores_history_task_journal_outcome_and_cache_without_mixing_sessions(self):
        client = FakeClient()
        client.cache_stats = CacheStats()
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        old_messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答", "reasoning_content": "旧思考"},
        ]
        plan = {"title": "旧任务", "steps": [{"step": "检查", "status": "completed"}]}
        conversation.messages.extend(old_messages)
        conversation.tools.task_title = "旧任务"
        conversation.tools.task_plan = plan
        conversation.tool_runs = [{"tool_call_id": "old-tool", "state": "completed"}]
        conversation.last_outcome = {"status": "completed"}
        client.cache_stats.record(
            {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}, "solve"
        )
        old_id = conversation.session_id
        conversation.save()
        conversation.clear()
        self.assertFalse(conversation.settings.session_file.exists())
        self.assertNotEqual(conversation.session_id, old_id)
        new_id = conversation.session_id
        conversation.messages.append({"role": "user", "content": "新问题"})
        conversation.tools.task_title = "新任务"
        client.cache_stats.record(
            {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 20}}, "solve"
        )
        conversation.save()

        conversation.resume(old_id)

        self.assertEqual(conversation.messages, old_messages)
        self.assertEqual(conversation.tools.task_title, "旧任务")
        self.assertEqual(conversation.tools.task_plan, plan)
        self.assertEqual(
            conversation.tool_runs, [{"tool_call_id": "old-tool", "state": "completed"}]
        )
        self.assertEqual(conversation.last_outcome, {"status": "completed"})
        self.assertEqual(client.cache_stats.snapshot()["prompt_tokens"], 10)
        self.assertEqual(len(conversation.sessions.list()), 2)
        restarted = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        self.assertEqual(restarted.session_id, old_id)
        self.assertEqual(restarted.messages, old_messages)
        conversation.resume(new_id)
        self.assertEqual(conversation.messages, [{"role": "user", "content": "新问题"}])
        self.assertIsNone(conversation.tools.task_plan)
        self.assertEqual(conversation.tool_runs, [])
        self.assertIsNone(conversation.last_outcome)
        self.assertEqual(client.cache_stats.snapshot()["prompt_tokens"], 50)

    def test_resume_saves_unsaved_current_messages_before_switching(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        conversation.messages.append({"role": "user", "content": "旧问题"})
        old_id = conversation.session_id
        conversation.clear()
        new_id = conversation.session_id
        current = [{"role": "user", "content": "尚未保存的新问题"}]
        conversation.messages.extend(current)

        conversation.resume(old_id)
        self.assertEqual(conversation.messages, [{"role": "user", "content": "旧问题"}])
        conversation.resume(new_id)
        self.assertEqual(conversation.messages, current)

    def test_resuming_current_session_keeps_unsaved_changes(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        conversation.messages.append({"role": "user", "content": "旧问题"})
        conversation.save()
        conversation.messages.append({"role": "user", "content": "最新问题"})

        conversation.resume(conversation.session_id)

        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.sessions.read()["messages"], conversation.messages)
        self.assertEqual(len(conversation.sessions.list()), 1)

    def test_legacy_session_is_automatically_archived_with_a_stable_id(self):
        settings = self.settings()
        settings.session_file.parent.mkdir(parents=True)
        messages = [{"role": "user", "content": "旧版对话"}]
        settings.session_file.write_text(json.dumps({"version": 1, "messages": messages}))

        conversation = Conversation(settings, FakeClient(), CodingTools(self.workspace))
        restarted = Conversation(settings, FakeClient(), CodingTools(self.workspace))

        self.assertEqual(conversation.messages, messages)
        self.assertEqual(restarted.session_id, conversation.session_id)
        self.assertEqual(len(conversation.sessions.list()), 1)
        self.assertEqual(conversation.sessions.read()["session_id"], conversation.session_id)

    def test_invalid_resume_does_not_modify_current_history_task_or_cache(self):
        client = FakeClient()
        client.cache_stats = CacheStats()
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        conversation.messages.append({"role": "user", "content": "旧问题"})
        old_id = conversation.session_id
        conversation.clear()
        conversation.messages.append({"role": "user", "content": "当前问题"})
        conversation.tools.task_title = "当前任务"
        client.cache_stats.record({"prompt_tokens": 10}, "solve")
        conversation.save()
        current_id = conversation.session_id
        current_file = conversation.settings.session_file.read_text()
        stats = client.cache_stats.snapshot()
        path = conversation.sessions.snapshot_path(old_id)
        data = conversation.sessions.read(path)
        data["task_state"] = {"title": "坏任务", "plan": {"steps": "invalid"}}
        path.write_text(json.dumps(data))

        with self.assertRaisesRegex(RuntimeError, "Cannot load session file"):
            conversation.resume(old_id)

        self.assertEqual(conversation.session_id, current_id)
        self.assertEqual(conversation.messages, [{"role": "user", "content": "当前问题"}])
        self.assertEqual(conversation.tools.task_title, "当前任务")
        self.assertEqual(client.cache_stats.snapshot(), stats)
        self.assertEqual(conversation.settings.session_file.read_text(), current_file)

    def test_resume_repairs_pending_tools_without_reexecuting_them(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        conversation.messages.extend(
            [
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "tool_calls": [tool_call()]},
            ]
        )
        conversation.tool_runs = [
            {"assistant_index": 1, "tool_call_id": "call-1", "state": "pending"}
        ]
        old_id = conversation.session_id
        conversation.clear()

        with patch.object(conversation.tools, "execute") as execute:
            conversation.resume(old_id)

        execute.assert_not_called()
        self.assertEqual(conversation.messages[-1]["role"], "tool")
        self.assertEqual(conversation.messages[-1]["tool_call_id"], "call-1")
        self.assertIsNotNone(conversation.recovery_notice)
        self.assertEqual(conversation.sessions.read()["messages"], conversation.messages)

    def test_resume_keeps_completed_tool_results_without_reexecuting_them(self):
        conversation = Conversation(self.settings(), FakeClient(), CodingTools(self.workspace))
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "tool_calls": [tool_call()]},
            {"role": "tool", "tool_call_id": "call-1", "content": "已保存的执行结果"},
            {"role": "assistant", "content": "旧回答"},
        ]
        conversation.messages.extend(messages)
        old_id = conversation.session_id
        conversation.clear()

        with patch.object(conversation.tools, "execute") as execute:
            conversation.resume(old_id)

        execute.assert_not_called()
        self.assertEqual(conversation.messages, messages)
        self.assertIsNone(conversation.recovery_notice)

    def test_followup_after_resume_uses_the_restored_conversation(self):
        client = FakeClient({"role": "assistant", "content": "继续回答"})
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        conversation.messages.extend(
            [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"}]
        )
        old_id = conversation.session_id
        conversation.clear()
        conversation.messages.append({"role": "user", "content": "其他会话的问题"})
        conversation.resume(old_id)

        self.assertEqual(conversation.ask("继续聊刚才的问题"), "继续回答")
        request = client.requests[-1]["messages"]
        self.assertIn({"role": "user", "content": "旧问题"}, request)
        self.assertIn({"role": "assistant", "content": "旧回答"}, request)
        self.assertNotIn({"role": "user", "content": "其他会话的问题"}, request)

    def test_compaction_counts_tools_and_keeps_the_latest_turn(self):
        client = FakeClient(*[{"role": "assistant", "content": "summary"}] * 4)
        conversation = Conversation(
            self.settings(max_context_chars=30_000, keep_recent_turns=2),
            client,
            CodingTools(self.workspace),
        )
        for index in range(5):
            conversation.messages += [
                {"role": "user", "content": f"user-{index}-" + "u" * 4_000},
                {"role": "assistant", "content": f"answer-{index}-" + "a" * 4_000},
            ]

        conversation.compact()

        self.assertLessEqual(
            conversation._request_size([conversation.system_message, *conversation.messages]),
            conversation.settings.max_context_chars,
        )
        self.assertIn("summary", conversation.messages[0]["content"])
        self.assertTrue(
            any("user-4-" in (message.get("content") or "") for message in conversation.messages)
        )

    def test_compaction_obeys_remaining_task_budget(self):
        client = FakeClient({"role": "assistant", "content": "summary"})
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertTrue(
            conversation._summarize(
                [{"role": "user", "content": "task"}], time.monotonic() + 0.5
            ).startswith("summary")
        )
        self.assertLessEqual(client.requests[0]["timeout"], 0.5)
        with self.assertRaises(TimeoutError):
            conversation._summarize([{"role": "user", "content": "task"}], time.monotonic() - 1)
        self.assertEqual(len(client.requests), 1)

    def test_output_limit_continues_from_reasoning_without_unnecessary_summary(self):
        partial = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Established n = 42; only units remain.",
        }
        client = FakeClient(
            CompletionError("output_limit", "cut off", partial_message=partial),
            {"role": "assistant", "content": "Answer: 42 metres"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("Compute n and include units."), "Answer: 42 metres")
        self.assertEqual(len(client.requests), 2)
        resumed = client.requests[1]["messages"][-1]["content"]
        self.assertIn("Established n = 42", resumed)
        self.assertTrue(
            any(
                m.get("content") == "Compute n and include units."
                for m in client.requests[1]["messages"]
            )
        )
        self.assertNotEqual(client.requests[0]["messages"], client.requests[1]["messages"])
        (artifact,) = (self.workspace / ".agent-artifacts").glob("*.output-limit.json")
        self.assertEqual(json.loads(artifact.read_text()), partial)
        self.assertEqual(
            json.loads(conversation.settings.session_file.read_text())["messages"],
            conversation.messages,
        )

    def test_large_truncated_reasoning_that_fits_keeps_its_middle_without_summary(self):
        partial = {
            "role": "assistant",
            "reasoning": "a" * 24000 + "KEY INTERMEDIATE VALUE 42" + "z" * 24000,
        }
        client = FakeClient(
            CompletionError("output_limit", "cut off", partial_message=partial),
            {"role": "assistant", "content": "Answer: 42 metres"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("Find n."), "Answer: 42 metres")
        self.assertEqual(len(client.requests), 2)
        resumed = client.requests[-1]["messages"][-1]["content"]
        self.assertIn(partial["reasoning"], resumed)
        (artifact,) = (self.workspace / ".agent-artifacts").glob("*.output-limit.json")
        self.assertEqual(json.loads(artifact.read_text()), partial)

    def test_truncated_summary_falls_back_to_original_source_without_recursive_generation(self):
        partial = {
            "role": "assistant",
            "reasoning": "IMPORTANT FIRST FACT " + "x" * 140000 + " IMPORTANT LAST FACT",
        }
        client = FakeClient(
            CompletionError("output_limit", "cut off", partial_message=partial),
            CompletionError(
                "output_limit",
                "summary cut off",
                partial_message={"reasoning": "invented summary reasoning"},
            ),
            {"role": "assistant", "content": "finished"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("original task"), "finished")
        self.assertEqual(len(client.requests), 3)
        resumed = str(client.requests[-1]["messages"])
        self.assertIn("IMPORTANT FIRST FACT", resumed)
        self.assertIn("IMPORTANT LAST FACT", resumed)
        self.assertNotIn("invented summary reasoning", resumed)

    def test_output_recovery_never_executes_partial_calls_or_repeats_completed_tools(self):
        def write(name):
            return {
                "id": name,
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"path": name, "content": "42"}),
                },
            }

        client = FakeClient(
            {"role": "assistant", "content": None, "tool_calls": [write("completed.txt")]},
            CompletionError(
                "output_limit",
                "cut off",
                partial_message={
                    "role": "assistant",
                    "reasoning_content": "Previous file already written",
                    "tool_calls": [write("unfinished.txt")],
                },
            ),
            {"role": "assistant", "content": "finished"},
        )
        tools = CodingTools(self.workspace)
        conversation = Conversation(self.settings(), client, tools)
        with patch.object(tools, "execute", wraps=tools.execute) as executed:
            self.assertEqual(conversation.ask("write a file"), "finished")
        self.assertEqual(executed.call_count, 1)
        self.assertTrue((self.workspace / "completed.txt").exists())
        self.assertFalse((self.workspace / "unfinished.txt").exists())
        self.assertIsNotNone(client.requests[-1]["tools"])
        tool_messages = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["completed.txt"])

    def test_recovery_budget_does_not_reset_after_a_successful_tool_round(self):
        error = CompletionError(
            "output_limit", "cut off", partial_message={"reasoning": "partial progress"}
        )
        client = FakeClient(
            error, {"role": "assistant", "content": None, "tool_calls": [tool_call()]}, error
        )
        conversation = Conversation(
            self.settings(max_output_continuations=1), client, CodingTools(self.workspace)
        )
        with self.assertRaises(CompletionError) as raised:
            conversation.ask("finish task")
        self.assertEqual(raised.exception.code, "output_limit")
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(
            len(list((self.workspace / ".agent-artifacts").glob("*.output-limit.json"))), 2
        )

    def test_disabled_recovery_and_unusable_partial_response_never_start_another_request(self):
        for code, partial, limit in [
            ("output_limit", {"reasoning": "progress"}, 0),
            ("output_limit", {"content": None}, 2),
            ("transport_error", {"content": "partial"}, 2),
        ]:
            with self.subTest(code=code, partial=partial, limit=limit):
                client = FakeClient(CompletionError(code, "failure", partial_message=partial))
                conversation = Conversation(
                    self.settings(max_output_continuations=limit),
                    client,
                    CodingTools(self.workspace),
                )
                with self.assertRaises(CompletionError):
                    conversation.ask("go")
                self.assertEqual(len(client.requests), 1)

    def test_expired_recovery_budget_keeps_checkpoint_without_model_call(self):
        client = FakeClient()
        recovery = OutputRecovery(client, self.settings(), time.monotonic() - 1)
        partial = {"reasoning": "last known progress"}
        with self.assertRaises(CompletionError) as raised:
            recovery.recover(
                CompletionError("output_limit", "cut off", partial_message=partial),
                [{"role": "user", "content": "task"}],
            )
        self.assertEqual(raised.exception.code, "task_deadline")
        self.assertEqual(client.requests, [])
        (artifact,) = (self.workspace / ".agent-artifacts").glob("*.output-limit.json")
        self.assertEqual(json.loads(artifact.read_text()), partial)

    def test_timed_out_summary_still_allows_continuation_with_remaining_task_time(self):
        client = FakeClient(
            CompletionError(
                "output_limit", "cut off", partial_message={"reasoning": "progress " * 20000}
            ),
            CompletionError("task_deadline", "summary deadline"),
            {"role": "assistant", "content": "done"},
        )
        conversation = Conversation(self.settings(), client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("task"), "done")
        self.assertIn("Source excerpts", str(client.requests[-1]["messages"]))


if __name__ == "__main__":
    unittest.main()
