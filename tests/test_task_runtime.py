import errno
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Settings
from context_management import summarize
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient, clean_completion_suffix
from task_runtime import TaskProgress, TaskRuntime
from tests.support import make_settings
from tools import CodingTools


class Responses:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def tool(name="ls", arguments=None):
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {})},
    }


def chunk(delta, finish=None):
    return (
        "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n"
    ).encode()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="fake",
            session_file=self.workspace / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            workspace=self.workspace,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_tool_count_environment_settings_cannot_restore_limits(self):
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test",
                    "DEEPSEEK_MAX_TOOL_ROUNDS": "1",
                    "DEEPSEEK_MAX_SUBAGENT_TOOL_ROUNDS": "1",
                },
            ),
            patch("config.load_env"),
        ):
            settings = Settings.from_env()
        self.assertFalse(hasattr(settings, "max_tool_rounds"))
        self.assertFalse(hasattr(settings, "max_subagent_tool_rounds"))

    def test_task_time_defaults_to_unlimited_and_preserves_explicit_limits(self):
        with (
            patch("config.load_env"),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True),
        ):
            self.assertIsNone(Settings.from_env().max_task_seconds)
            with patch.dict(os.environ, {"DEEPSEEK_MAX_TASK_SECONDS": " unlimited "}):
                self.assertIsNone(Settings.from_env().max_task_seconds)
            with patch.dict(os.environ, {"DEEPSEEK_MAX_TASK_SECONDS": "150"}):
                self.assertEqual(Settings.from_env().max_task_seconds, 150)

    def test_orphan_protocol_suffix_is_removed_without_losing_long_report_or_metadata(self):
        body = (
            "TARGET: report.md\n<<<MARKDOWN>>>\n"
            + "正文与证据\n" * 3000
            + "<<<END>>>\n[[[REPORT]]]\n"
        )
        response = {
            "role": "assistant",
            "content": body
            + "</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>",
            "_harness": {"usage": {"completion_tokens": 9000}},
        }
        client = Responses(response)
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        accepted = runtime.generate(
            [{"role": "user", "content": "report"}], {"role": "system", "content": "fixed"}, []
        )
        self.assertEqual(accepted["content"], body)
        self.assertEqual(accepted["_harness"], response["_harness"])
        self.assertIn("DSML", response["content"])
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            list((self.workspace / ".agent-artifacts").glob("*.invalid-response.json")), []
        )

    def test_suffix_cleanup_preserves_embedded_tags_and_rejects_open_tool_markup(self):
        for content in [
            "A closing tag </｜｜DSML｜｜ calls> followed by explanatory text.",
            '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="write">fake</｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>',
        ]:
            with self.subTest(content=content):
                self.assertEqual(clean_completion_suffix(content), content)
        bad = {"role": "assistant", "content": "<｜｜DSML｜｜ calls>fake</｜｜DSML｜｜ calls>"}
        client = Responses(bad, bad, bad)
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        with self.assertRaises(CompletionError) as raised:
            runtime.generate(
                [{"role": "user", "content": "task"}], {"role": "system", "content": "fixed"}, []
            )
        self.assertEqual(raised.exception.code, "invalid_final_response")

    def test_closing_tags_alone_cannot_become_an_accepted_answer(self):
        bad = {"role": "assistant", "content": "\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>"}
        runtime = TaskRuntime(Responses(bad, bad, bad), self.settings, time.monotonic() + 60)
        with self.assertRaises(CompletionError) as raised:
            runtime.generate([], {"role": "system", "content": "fixed"}, [])
        self.assertEqual(raised.exception.code, "invalid_final_response")

    def test_protocol_repair_points_to_full_artifact_and_only_appends_request_notes(self):
        body = "BEGIN\n" + "完整段落\n" * 1800 + "MIDDLE-EVIDENCE\n" + "完整段落\n" * 1800
        rejected = {
            "role": "assistant",
            "content": body + "<｜｜DSML｜｜ calls>fake</｜｜DSML｜｜ calls>",
        }
        client = Responses(rejected, {"role": "assistant", "content": "repaired"})
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        runtime.generate(
            [{"role": "user", "content": "task"}],
            {"role": "system", "content": "fixed"},
            [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        )
        artifact = next((self.workspace / ".agent-artifacts").glob("*.invalid-response.json"))
        self.assertEqual(json.loads(artifact.read_text()), rejected)
        note = client.requests[1]["messages"][-1]["content"]
        self.assertIn(str(artifact), note)
        self.assertNotIn("MIDDLE-EVIDENCE", note)
        before, after = client.requests[0]["messages"], client.requests[1]["messages"]
        self.assertEqual(before, after[: len(before)])
        self.assertEqual(client.requests[0]["tools"], client.requests[1]["tools"])

    def test_protocol_repair_write_failure_preserves_full_response_in_appended_note(self):
        response = {"role": "assistant", "content": "full original response " * 1000}
        runtime = TaskRuntime(Responses(), self.settings, time.monotonic() + 60)
        with patch("task_runtime.atomic_write_text", side_effect=OSError("disk fixture")):
            note = runtime._repair_instruction(response, "format fixture", [])
        self.assertIn(response["content"], note)
        self.assertIn("Recovery artifact unavailable", note)

    def test_repair_without_file_tools_keeps_full_report_and_request_prefix(self):
        body = "begin " * 1000 + "MIDDLE-EVIDENCE" + " end" * 1000
        client = Responses(
            {"role": "assistant", "content": body},
            {"role": "assistant", "content": "valid"},
        )
        runtime = TaskRuntime(
            client,
            self.settings,
            time.monotonic() + 60,
            final_validator=lambda content: None if content == "valid" else "Expected final format",
        )
        messages = [{"role": "user", "content": "format the report"}]
        response = runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertEqual(response["content"], "valid")
        note = client.requests[1]["messages"][-1]["content"]
        self.assertIn(body, note)
        self.assertIn("MIDDLE-EVIDENCE", note)
        before, after = client.requests[0]["messages"], client.requests[1]["messages"]
        self.assertEqual(before, after[: len(before)])
        self.assertEqual(len(client.requests), 2)

    def test_terminal_checkpoint_write_failure_preserves_response_without_retry(self):
        partial = {"role": "assistant", "content": "retained final draft"}
        client = Responses(CompletionError("task_deadline", "stopped", partial_message=partial))
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        messages = [{"role": "user", "content": "task"}]
        with (
            patch("context_management.ArtifactStore.new_path", side_effect=OSError("disk fixture")),
            self.assertRaises(CompletionError),
        ):
            runtime.generate(messages, {"role": "system", "content": "fixed"}, [])
        self.assertIn("retained final draft", messages[-1]["content"])
        self.assertIn("Recovery artifact unavailable", messages[-1]["content"])
        self.assertEqual(len(client.requests), 1)

    def test_pseudo_tool_calls_are_repaired_not_executed_or_completed(self):
        dsml = '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="write">must-not-execute</｜｜DSML｜｜ invoke>'
        client = Responses(
            {"role": "assistant", "content": dsml},
            {"role": "assistant", "content": None, "tool_calls": [tool()]},
            {"role": "assistant", "content": "finished"},
        )
        tools = CodingTools(self.workspace)
        with patch.object(tools, "execute", wraps=tools.execute) as executed:
            result = Conversation(self.settings, client, tools).ask("work")
        self.assertEqual(result, "finished")
        self.assertEqual(executed.call_count, 1)
        self.assertEqual(executed.call_args.args[0]["function"]["name"], "ls")
        self.assertTrue(all(r["tools"] for r in client.requests))
        self.assertIn("not executed", client.requests[1]["messages"][-1]["content"])
        self.assertEqual(client.requests[0]["messages"][0], client.requests[1]["messages"][0])

    def test_ui_publishes_accepted_content_before_ending_the_response(self):
        events = []
        client = Responses(
            {"role": "assistant", "reasoning_content": "checked the evidence", "content": "done"}
        )
        Conversation(self.settings, client, None).ask(
            "task",
            on_content=lambda value: events.append(value),
            on_reasoning=lambda value: events.append(f"reasoning:{value}"),
            on_response_start=lambda: events.append("start"),
            on_response_end=lambda: events.append("end"),
        )
        self.assertEqual(events, ["start", "reasoning:checked the evidence", "done", "end"])

    def test_ui_closes_a_recovered_request_before_starting_its_retry(self):
        events = []
        client = Responses(
            CompletionError(
                "generation_budget",
                "slice",
                partial_message={"reasoning": "useful intermediate result " * 20},
            ),
            {"role": "assistant", "content": "done"},
        )
        Conversation(self.settings, client, None).ask(
            "task",
            on_content=lambda value: events.append(value),
            on_reasoning=lambda value: events.append(f"reasoning:{value}"),
            on_response_start=lambda: events.append("start"),
            on_response_end=lambda: events.append("end"),
        )
        self.assertEqual(events, ["start", "end", "start", "done", "end"])

    def test_main_task_recovers_model_rate_limit_after_tool_without_replaying_it(self):
        client = Responses(
            {
                "role": "assistant",
                "tool_calls": [tool("write", {"path": "result.txt", "content": "completed"})],
            },
            CompletionError("http_429", "fixture", retryable=True),
            {"role": "assistant", "content": "done"},
        )
        tools = CodingTools(self.workspace)
        self.addCleanup(tools.processes.close)
        conversation = Conversation(self.settings, client, tools)
        with patch.object(tools, "execute", wraps=tools.execute) as executed:
            self.assertEqual(conversation.ask("write result"), "done")
        self.assertEqual(executed.call_count, 1)
        self.assertEqual((self.workspace / "result.txt").read_text(), "completed")
        before, after = client.requests[1]["messages"], client.requests[2]["messages"]
        self.assertEqual(after[: len(before)], before)
        self.assertIn('"code": "http_429"', after[-1]["content"])
        self.assertEqual(client.requests[1]["tools"], client.requests[2]["tools"])

    def test_parent_model_receives_successful_workers_rate_limit_recovery(self):
        client = Responses(
            {
                "role": "assistant",
                "tool_calls": [
                    tool("delegate", {"tasks": [{"task": "Inspect", "expected_output": "report"}]})
                ],
            },
            CompletionError("http_429", "fixture", retryable=True),
            {"role": "assistant", "content": "Worker report"},
            {"role": "assistant", "content": "done"},
        )
        tools = CodingTools(self.workspace)
        self.addCleanup(tools.processes.close)
        conversation = Conversation(self.settings, client, tools)
        self.assertEqual(conversation.ask("Inspect and report"), "done")
        result = json.loads(
            next(
                message["content"]
                for message in client.requests[-1]["messages"]
                if message["role"] == "tool"
            )
        )
        report = result["data"]["reports"][0]
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["data"]["model_recovery"]["status"], "recovered")
        self.assertIn("further delegation", report["data"]["model_recovery"]["guidance"])

    def test_repeated_invalid_tool_text_is_an_explicit_protocol_failure(self):
        bad = {"role": "assistant", "content": "<｜｜DSML｜｜ calls>fake</｜｜DSML｜｜ calls>"}
        client = Responses(bad, bad, bad)
        with self.assertRaises(CompletionError) as caught:
            Conversation(self.settings, client, None).ask("give an answer")
        self.assertEqual(caught.exception.code, "invalid_final_response")
        self.assertEqual(
            len(list((self.workspace / ".agent-artifacts").glob("*.invalid-response.json"))), 3
        )

    def test_continuations_can_cross_old_recovery_limit_without_growing_user_turns(self):
        errors = [
            CompletionError(
                "generation_budget", "slice", partial_message={"reasoning": f"new progress {i}"}
            )
            for i in range(5)
        ]
        client = Responses(*errors, {"role": "assistant", "content": "done"})
        conversation = Conversation(self.settings, client, CodingTools(self.workspace))
        self.assertEqual(conversation.ask("original question"), "done")
        last = client.requests[-1]["messages"]
        self.assertEqual(sum(m["role"] == "user" for m in last), 1)
        self.assertEqual(sum("<unfinished_work>" in (m.get("content") or "") for m in last), 5)
        for i in range(5):
            self.assertIn(f"new progress {i}", str(last))
        self.assertIn("new progress 4", last[-1]["content"])
        self.assertTrue(all(r["tools"] for r in client.requests))

    def test_failed_observation_survives_a_summary_that_omits_it(self):
        client = Responses(
            *[{"role": "assistant", "content": "Concise task summary"}] * 3,
            {"role": "assistant", "content": "done"},
        )
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        call = tool("exec", {"command": "timeout 5 python3 solve.py"})
        runtime.progress.observe(
            call,
            {
                "status": "error",
                "error": {"code": "nonzero_exit", "message": "127"},
                "data": {"output": "/bin/sh: timeout: command not found\n"},
            },
        )
        messages = [
            {"role": "user", "content": "Solve the task"},
            {"role": "assistant", "reasoning": "x" * 130000},
        ]
        response = runtime.generate(messages, {"role": "system", "content": "Task solver"}, [])
        self.assertEqual(response["content"], "done")
        state = client.requests[-1]["messages"][-1]["content"]
        self.assertIn("timeout", state)
        self.assertIn("unresolved_failures", state)
        self.assertEqual(runtime.summary_requests, 3)
        self.assertEqual(json.loads(runtime.progress.path.read_text())["status"], "error")

    def test_repetition_ignores_duration_but_changed_output_counts_as_new(self):
        progress = TaskProgress(self.workspace)
        for duration, output in [(1, "same"), (2, "same"), (3, "new")]:
            progress.observe(
                tool("exec", {"command": "check"}),
                {"status": "ok", "data": {"duration_seconds": duration, "output": output}},
            )
            record = next(iter(progress.observations.values()))
            self.assertEqual(record["same_result_count"], 2 if duration == 2 else 1)
        self.assertEqual(len(progress.path.read_text().splitlines()), 3)

    def test_progress_recreates_a_removed_directory_without_losing_memory(self):
        progress = TaskProgress(self.workspace)
        progress.observe(
            tool("read", {"path": "missing.txt"}),
            {"status": "error", "error": {"code": "not_found"}},
        )
        path = progress.path
        shutil.rmtree(path.parent)

        notice = progress.observe(tool(), {"status": "ok", "data": {"entries": []}})

        self.assertEqual(progress.path, path)
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text())["status"], "ok")
        self.assertEqual(notice["code"], "progress_log_recreated")
        self.assertEqual(len(progress.observations), 2)
        self.assertEqual(len(progress.failures), 1)
        self.assertIn("unresolved_failures", progress.guidance())
        self.assertIn("Partial log", progress.guidance())
        self.assertNotIn("Full log", progress.guidance())

    def test_progress_recreates_a_removed_file_and_marks_history_incomplete(self):
        progress = TaskProgress(self.workspace)
        progress.observe(tool(), {"status": "ok"})
        progress.path.unlink()

        notice = progress.observe(tool(), {"status": "ok"})

        self.assertEqual(notice["code"], "progress_log_recreated")
        self.assertTrue(progress.log_incomplete)
        self.assertIsNone(progress.log_error)
        record = json.loads(progress.path.read_text())
        self.assertEqual(record["same_result_count"], 2)
        self.assertIn("do not repeat completed tools", progress.guidance())

    def test_progress_creation_failure_warns_once_and_later_recovers(self):
        progress = TaskProgress(self.workspace)
        result = {"status": "error", "error": {"code": "not_found"}}
        with patch("task_runtime.ArtifactStore.new_path", side_effect=PermissionError("denied")):
            first = progress.observe(tool(), result)
            repeated = progress.observe(tool(), result)

        self.assertEqual(first["code"], "progress_log_unavailable")
        self.assertIsNone(repeated)
        self.assertEqual(len(progress.failures), 1)
        self.assertIn("unavailable", progress.guidance())
        self.assertNotIn("Full log", progress.guidance())
        self.assertIsNone(progress.path)

        restored = progress.observe(tool(), result)
        self.assertEqual(restored["code"], "progress_log_restored")
        self.assertIsNone(progress.log_error)
        self.assertTrue(progress.log_incomplete)
        self.assertEqual(json.loads(progress.path.read_text())["same_result_count"], 3)

    def test_progress_write_failure_preserves_status_and_does_not_flood_warnings(self):
        progress = TaskProgress(self.workspace)
        progress.observe(tool(), {"status": "ok"})
        with patch.object(Path, "open", side_effect=OSError(errno.ENOSPC, "disk full")):
            first = progress.observe(tool(), {"status": "ok"})
            repeated = progress.observe(tool(), {"status": "ok"})

        self.assertEqual(first["code"], "progress_log_unavailable")
        self.assertIsNone(repeated)
        self.assertEqual(progress.failures, {})
        self.assertEqual(next(iter(progress.observations.values()))["status"], "ok")
        self.assertIn("Unavailable log", progress.guidance())

        restored = progress.observe(tool(), {"status": "ok"})
        self.assertEqual(restored["code"], "progress_log_restored")
        self.assertIsNone(progress.log_error)
        records = [json.loads(line) for line in progress.path.read_text().splitlines()]
        self.assertEqual([record["same_result_count"] for record in records], [1, 4])
        self.assertIn("Partial log", progress.guidance())

    def test_progress_retries_a_directory_removal_between_mkdir_and_open(self):
        progress = TaskProgress(self.workspace)
        progress.observe(tool(), {"status": "ok"})
        original_open = Path.open
        attempts = []

        def open_log(path, *args, **kwargs):
            attempts.append(path)
            if len(attempts) == 1:
                shutil.rmtree(path.parent)
                raise FileNotFoundError(errno.ENOENT, "missing directory", str(path))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=open_log):
            notice = progress.observe(tool(), {"status": "ok"})

        self.assertEqual(attempts, [progress.path, progress.path])
        self.assertEqual(notice["code"], "progress_log_recreated")
        self.assertEqual(json.loads(progress.path.read_text())["same_result_count"], 2)

    def test_progress_missing_path_retries_are_bounded_and_nonfatal(self):
        progress = TaskProgress(self.workspace)
        progress.observe(tool(), {"status": "ok"})
        with patch.object(Path, "open", side_effect=FileNotFoundError("missing")) as opened:
            notice = progress.observe(tool(), {"status": "ok"})

        self.assertEqual(opened.call_count, 2)
        self.assertEqual(notice["code"], "progress_log_unavailable")
        self.assertEqual(next(iter(progress.observations.values()))["same_result_count"], 2)

    def test_progress_does_not_swallow_user_interruption(self):
        progress = TaskProgress(self.workspace)
        with (
            patch.object(Path, "open", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            progress.observe(tool(), {"status": "ok"})

    def test_conversation_continues_after_journal_failure_without_replaying_tools(self):
        events, results = [], []
        client = Responses(
            {"role": "assistant", "content": None, "tool_calls": [tool()]},
            {"role": "assistant", "content": "Inspection finished"},
        )
        tools = CodingTools(self.workspace)
        conversation = Conversation(self.settings, client, tools)
        with (
            patch("task_runtime.ArtifactStore.new_path", side_effect=PermissionError("denied")),
            patch.object(tools, "execute", wraps=tools.execute) as executed,
        ):
            answer = conversation.ask(
                "Inspect the directory",
                on_progress=events.append,
                on_tool_result=lambda name, result: results.append((name, result)),
            )

        self.assertEqual(answer, "Inspection finished")
        self.assertEqual(executed.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1]["status"], "ok")
        self.assertEqual(conversation.last_outcome["status"], "completed")
        self.assertEqual(conversation.tool_runs[0]["state"], "completed")
        saved = json.loads(self.settings.session_file.read_text())
        tool_messages = [message for message in saved["messages"] if message["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(json.loads(tool_messages[0]["content"])["status"], "ok")
        warnings = [event for event in events if event["phase"] == "warning"]
        self.assertEqual([event["code"] for event in warnings], ["progress_log_unavailable"])

    def test_conversation_continues_when_a_tool_removes_the_progress_directory(self):
        events = []
        first_call, second_call = tool(), tool("ls", {"path": "."})
        second_call["id"] = "call-2"
        client = Responses(
            {"role": "assistant", "content": None, "tool_calls": [first_call]},
            {"role": "assistant", "content": None, "tool_calls": [second_call]},
            {"role": "assistant", "content": "Inspection finished"},
        )
        tools = CodingTools(self.workspace)
        original_execute = tools.execute

        def execute_and_remove(call, **kwargs):
            output = original_execute(call, **kwargs)
            if call["id"] == "call-2":
                shutil.rmtree(self.workspace / ".agent-artifacts")
            return output

        conversation = Conversation(self.settings, client, tools)
        with patch.object(tools, "execute", side_effect=execute_and_remove) as executed:
            answer = conversation.ask("Inspect the directory", on_progress=events.append)

        self.assertEqual(answer, "Inspection finished")
        self.assertEqual(executed.call_count, 2)
        self.assertTrue(all(run["state"] == "completed" for run in conversation.tool_runs))
        self.assertEqual(conversation.last_outcome["status"], "completed")
        self.assertTrue(any(event.get("code") == "progress_log_recreated" for event in events))

    def test_missing_software_survives_successful_shell_exit_and_can_be_refreshed(self):
        progress = TaskProgress(self.workspace)
        progress.observe(
            tool("exec", {"command": "which sage; which gp; which python3; python3 --version"}),
            {"status": "ok", "data": {"exit_code": 0, "output": "/usr/bin/python3\nPython 3.13\n"}},
        )
        self.assertEqual(progress.missing_commands, {"sage", "gp"})
        progress.observe(
            tool("exec", {"command": "which sage"}),
            {"status": "ok", "data": {"exit_code": 0, "output": "/opt/bin/sage\n"}},
        )
        self.assertEqual(progress.missing_commands, {"gp"})
        progress.observe(
            tool("exec", {"command": "false && which magma"}),
            {"status": "error", "data": {"output": ""}},
        )
        self.assertNotIn("magma", progress.missing_commands)

    def test_summary_chunks_full_reasoning_with_middle_evidence_and_accumulated_state(self):
        client = Responses(
            *[{"role": "assistant", "content": "Accumulated finding: n=42, unverified."}] * 7
        )
        source = [
            {
                "role": "assistant",
                "reasoning": "first fact "
                + "x" * 200000
                + "CRITICAL MIDDLE"
                + "y" * 200000
                + " last fact",
            }
        ]
        summarize(client, self.settings, source, time.monotonic() + 60)
        self.assertEqual(len(client.requests), 7)
        self.assertTrue(
            any("CRITICAL MIDDLE" in r["messages"][1]["content"] for r in client.requests)
        )
        for index, request in enumerate(client.requests):
            self.assertLessEqual(len(request["messages"][1]["content"]), 73000)
            self.assertEqual(request["purpose"], "summary")
            self.assertEqual(request["max_output_tokens"], 4096)
            if index:
                self.assertIn("Accumulated finding: n=42", request["messages"][1]["content"])
        (archived,) = (self.workspace / ".agent-artifacts").glob("*.context.json")
        self.assertEqual(json.loads(archived.read_text()), source)

    def test_tool_deadline_reserves_time_without_removing_tools(self):
        now = time.monotonic()
        runtime = TaskRuntime(Responses(), self.settings, now + 60)
        self.assertLessEqual(runtime.tool_deadline(), now + 52)
        self.assertGreater(runtime.tool_deadline(), now + 50)

    def test_task_deadline_cannot_accept_a_late_completed_response(self):
        client = Responses({"role": "assistant", "content": "late"})
        runtime = TaskRuntime(client, self.settings, 10)
        messages = [{"role": "user", "content": "task"}]
        with (
            patch("task_runtime.time.monotonic", side_effect=[1, 1, 11]),
            self.assertRaises(CompletionError) as caught,
        ):
            runtime.generate(messages, {"role": "system", "content": "solver"}, [])
        self.assertEqual(caught.exception.code, "task_deadline")
        checkpoint = next((self.workspace / ".agent-artifacts").glob("*.unfinished.json"))
        self.assertEqual(json.loads(checkpoint.read_text())["content"], "late")
        self.assertIn("Stopped generation", messages[-1]["content"])
        self.assertEqual(len(client.requests), 1)


class StreamProgressTests(unittest.TestCase):
    def test_reasoning_slice_keeps_partial_progress(self):
        raw = io.BytesIO(chunk({"reasoning": "working result " * 30}))
        with self.assertRaises(CompletionError) as caught:
            DeepSeekClient._read_stream(raw, lambda _: None, max_reasoning_chars=100)
        self.assertEqual(caught.exception.code, "generation_budget")
        self.assertIn("working result", caught.exception.partial_message["reasoning"])

    def test_generation_deadline_is_recoverable_and_distinct_from_task_deadline(self):
        raw = io.BytesIO(chunk({"reasoning": "first result"}) + chunk({"reasoning": "next"}))
        with (
            patch("deepseek_client.time.monotonic", side_effect=[1, 3]),
            self.assertRaises(CompletionError) as caught,
        ):
            DeepSeekClient._read_stream(raw, lambda _: None, deadline=10, generation_deadline=2)
        self.assertEqual(caught.exception.code, "generation_budget")
        self.assertEqual(caught.exception.partial_message["reasoning"], "first result")

    def test_decode_failure_and_transport_failure_preserve_reasoning(self):
        def failing_stream():
            yield chunk({"reasoning": "known intermediate value"})
            raise OSError("connection reset")

        for raw, code in [
            (
                io.BytesIO(chunk({"reasoning": "known intermediate value"}) + b"data: {broken\n"),
                "invalid_response",
            ),
            (failing_stream(), "transport_error"),
        ]:
            with self.subTest(code=code), self.assertRaises(CompletionError) as caught:
                DeepSeekClient._read_stream(raw, lambda _: None)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(
                caught.exception.partial_message["reasoning"], "known intermediate value"
            )

    def test_repeated_long_paragraphs_trigger_a_checkpoint_not_a_fake_answer(self):
        paragraph = "A repeated speculative argument with no new observation. " * 5
        raw = io.BytesIO(chunk({"reasoning": (paragraph + "\n\n") * 3}))
        with self.assertRaises(CompletionError) as caught:
            DeepSeekClient._read_stream(raw, lambda _: None)
        self.assertEqual(caught.exception.code, "reasoning_stalled")
        self.assertIsNone(caught.exception.partial_message["content"])


if __name__ == "__main__":
    unittest.main()
