import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_client import CompletionError
from reasoning import RolePromptCache
from tool_runtime import DELEGATE_RESULT_CHARS
from tools import CodingTools


def call(name, arguments):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def task(name="inspect", **kwargs):
    return {
        "task": name,
        "expected_output": "Evidence, findings and remaining uncertainty",
        **kwargs,
    }


class ReportingClient:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []

    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        with self.lock:
            self.requests.append({"messages": list(messages), "tools": tools, "timeout": timeout})
        assignment = messages[1]["content"].split("Assignment:\n", 1)[-1].splitlines()[0]
        return {"role": "assistant", "content": f"report for {assignment}"}


class InspectingClient:
    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        tool_message = next((message for message in messages if message["role"] == "tool"), None)
        if tool_message is None:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [call("read", {"path": "result.txt"})],
            }
        result = json.loads(tool_message["content"])
        verdict = (
            "PASS: result.txt:1 contains expected value"
            if result["data"]["content"] == "expected value"
            else "FAIL"
        )
        return {"role": "assistant", "content": verdict}


class ParallelClient:
    def __init__(self, workers=2):
        self.barrier = threading.Barrier(workers)

    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        self.barrier.wait(timeout=5)
        return {"role": "assistant", "content": "parallel report"}


class EditingClient:
    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        completed = sum(message["role"] == "tool" for message in messages)
        operations = [
            call("read", {"path": "result.txt"}),
            call(
                "edit",
                {
                    "path": "result.txt",
                    "edits": [{"oldText": "original", "newText": "updated"}],
                },
            ),
            call("write", {"path": "summary.txt", "content": "worker changes"}),
        ]
        if completed < len(operations):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [operations[completed]],
            }
        return {"role": "assistant", "content": "Updated result.txt and created summary.txt"}


class TimingOutClient:
    def complete(self, *args, **kwargs):
        raise TimeoutError("model request timed out")


class ReasoningToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.tools = CodingTools(self.workspace)

    def tearDown(self):
        self.temporary.cleanup()

    def delegate(self, tasks=None, **kwargs):
        return json.loads(
            self.tools.execute(call("delegate", {"tasks": tasks or [task()], **kwargs}))
        )

    def test_registry_reflects_actual_handlers_and_configured_worker_count(self):
        self.assertNotIn("delegate", {d["function"]["name"] for d in self.tools.definitions})
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=1)
        definitions = {d["function"]["name"]: d["function"] for d in self.tools.definitions}
        self.assertEqual(
            set(definitions),
            {
                "read",
                "write",
                "edit",
                "grep",
                "find",
                "ls",
                "exec",
                "monitor",
                "update_plan",
                "delegate",
            },
        )
        self.assertEqual(
            definitions["delegate"]["parameters"]["properties"]["tasks"]["maxItems"], 1
        )
        result = self.delegate([task("one"), task("two")])
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertEqual(client.requests, [])
        self.assertEqual(self.delegate()["status"], "ok")

    def test_delegate_returns_parallel_reports_in_task_order(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=2)
        result = self.delegate([task("inspect API"), task("inspect tests")])
        self.assertEqual(result["status"], "ok")
        reports = result["data"]["reports"]
        self.assertEqual([report["task"] for report in reports], ["inspect API", "inspect tests"])
        self.assertTrue(all(report["status"] == "ok" for report in reports))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual([report["data"]["model_requests"] for report in reports], [1, 1])

    def test_default_capacity_is_twenty_without_starting_unused_workers(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client)
        definition = self.tools.registry["delegate"].definition()["function"]
        self.assertEqual(definition["parameters"]["properties"]["tasks"]["maxItems"], 20)
        self.assertEqual(self.delegate([task("single investigation")])["status"], "ok")
        self.assertEqual(len(client.requests), 1)

    def test_worker_capacity_is_clamped_at_twenty_and_rejects_larger_batches(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=21)
        definition = self.tools.registry["delegate"].definition()["function"]
        self.assertEqual(definition["parameters"]["properties"]["tasks"]["maxItems"], 20)
        result = self.delegate([task(f"investigation {index}") for index in range(21)])
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertEqual(client.requests, [])

    def test_default_capacity_runs_twenty_independent_tasks_concurrently(self):
        self.tools.attach_reasoning(ParallelClient(workers=20))
        result = self.delegate([task(f"investigation {index}") for index in range(20)])
        self.assertEqual(result["status"], "ok", result)

    def test_delegate_really_runs_independent_tasks_concurrently(self):
        self.tools.attach_reasoning(ParallelClient(), max_workers=2)
        result = self.delegate([task("first"), task("second")])
        self.assertEqual(result["status"], "ok", result)

    def test_delegate_supplies_shared_context_task_context_and_expected_output(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client)
        self.delegate(
            [task(context="task evidence", expected_output="minimal counterexample")],
            context="shared evidence",
        )
        prompt = client.requests[0]["messages"][1]["content"]
        for expected in ("shared evidence", "task evidence", "minimal counterexample"):
            self.assertIn(expected, prompt)
        self.assertEqual(
            {d["function"]["name"] for d in client.requests[0]["tools"]},
            {"read", "grep", "find", "ls"},
        )

    def test_delegate_records_actual_workspace_checks(self):
        (self.workspace / "result.txt").write_text("expected value")
        self.tools.attach_reasoning(InspectingClient())
        result = self.delegate()
        report = result["data"]["reports"][0]
        self.assertEqual(report["status"], "ok")
        self.assertIn("PASS", report["data"]["report"])
        self.assertEqual(report["data"]["model_requests"], 2)
        self.assertEqual(report["data"]["checks"][0]["tool"], "read")
        self.assertEqual(report["data"]["checks"][0]["status"], "ok")

    def test_delegate_can_edit_and_write_when_worker_tools_are_granted(self):
        (self.workspace / "result.txt").write_text("original")
        self.tools.worker_allow = frozenset({"write", "edit"})
        self.tools.attach_reasoning(EditingClient())
        result = self.delegate([task("Update result.txt and create summary.txt", mode="inspect")])
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual((self.workspace / "result.txt").read_text(), "updated")
        self.assertEqual((self.workspace / "summary.txt").read_text(), "worker changes")
        checks = result["data"]["reports"][0]["data"]["checks"]
        self.assertEqual([check["tool"] for check in checks], ["read", "edit", "write"])
        self.assertTrue(all(check["status"] == "ok" for check in checks))

    def test_subagent_journal_failure_warns_without_losing_the_tool_result(self):
        (self.workspace / "result.txt").write_text("expected value")
        events = []
        self.tools.on_task_update = events.append
        self.tools.attach_reasoning(InspectingClient())
        with patch("task_runtime.ArtifactStore.new_path", side_effect=PermissionError("denied")):
            result = self.delegate()

        self.assertEqual(result["status"], "ok")
        report = result["data"]["reports"][0]
        self.assertEqual(report["status"], "ok")
        self.assertIn("PASS", report["data"]["report"])
        self.assertEqual(report["data"]["tool_calls"], 1)
        self.assertEqual(report["data"]["checks"][0]["status"], "ok")
        warnings = [event for event in events if event.get("activity") == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["agent_id"], "A1")
        self.assertEqual(warnings[0]["state"], "running")
        self.assertIn("任务继续执行", warnings[0]["warning"])
        self.assertEqual(events[-1]["state"], "completed")

    def test_subagent_progress_is_delivered_on_the_calling_thread(self):
        (self.workspace / "result.txt").write_text("expected value")
        events = []
        main_thread = threading.get_ident()

        def record(event):
            self.assertEqual(threading.get_ident(), main_thread)
            events.append(event)

        self.tools.on_task_update = record
        self.tools.attach_reasoning(InspectingClient())
        self.assertEqual(self.delegate([task(title="检查结果文件")])["status"], "ok")
        self.assertEqual(events[0]["state"], "pending")
        self.assertEqual(events[0]["title"], "检查结果文件")
        self.assertTrue(all(event["agent_id"] == "A1" for event in events))
        reading = next(event for event in events if event.get("activity") == "tool")
        self.assertEqual(reading["tool"], "read")
        self.assertEqual(reading["arguments"], {"path": "result.txt"})
        self.assertEqual(events[-1]["state"], "completed")
        self.assertEqual(events[-1]["tool_calls"], 1)
        self.assertEqual(events[-1]["model_requests"], 2)
        self.assertNotIn("report", events[-1])

    def test_progress_keeps_parallel_agents_distinct_and_surfaces_errors(self):
        class MixedClient(ReportingClient):
            def complete(self, messages, **kwargs):
                if "Assignment:\nfails" in messages[1]["content"]:
                    raise RuntimeError("model unavailable")
                return super().complete(messages, **kwargs)

        events = []
        self.tools.on_task_update = events.append
        self.tools.attach_reasoning(MixedClient())
        result = self.delegate([task("works"), task("fails")])
        self.assertEqual(result["status"], "partial")
        terminal = {event["agent_id"]: event for event in events if event["activity"] == "finished"}
        self.assertEqual(terminal["A1"]["state"], "completed")
        self.assertEqual(terminal["A2"]["state"], "error")
        self.assertIn("model unavailable", terminal["A2"]["error"]["message"])

    def test_timeout_drops_late_progress_after_a_fresh_task(self):
        release = threading.Event()
        returned = threading.Event()
        events = []

        class StalledClient:
            def complete(self, messages, **kwargs):
                release.wait(2)
                if kwargs.get("on_progress"):
                    kwargs["on_progress"]({"phase": "receiving"})
                returned.set()
                return {"role": "assistant", "content": "late report"}

        self.tools.on_task_update = events.append
        self.tools.attach_reasoning(StalledClient())
        try:
            result = json.loads(
                self.tools.execute(
                    call("delegate", {"tasks": [task()]}), deadline=time.monotonic() + 0.1
                )
            )
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(events[-1]["state"], "timeout")
            self.tools.reset_task()
            events.clear()
        finally:
            release.set()
        self.assertTrue(returned.wait(1))
        self.assertEqual(events, [])

    def test_subagent_tools_remain_available_beyond_former_limit(self):
        class LongInvestigation:
            def __init__(self):
                self.requests = 0
                self.tool_availability = []

            def complete(self, messages, tools=None, **kwargs):
                self.requests += 1
                self.tool_availability.append(bool(tools))
                if self.requests <= 25:
                    return {"role": "assistant", "tool_calls": [call("ls", {})]}
                return {"role": "assistant", "content": "report"}

        client = LongInvestigation()
        self.tools.attach_reasoning(client)
        result = self.delegate()
        self.assertEqual(result["status"], "ok")
        report = result["data"]["reports"][0]
        self.assertEqual(report["data"]["tool_calls"], 25)
        self.assertTrue(all(client.tool_availability))

    def test_subagent_timeout_is_never_ok(self):
        self.tools.attach_reasoning(TimingOutClient())
        result = self.delegate()
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["data"]["reports"][0]["status"], "timeout")

    def test_recall_and_explicit_reasoning_tasks_have_no_file_tools(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client)
        for assignment in (
            task("From memory only: recall the surname"),
            task("Compare these hypotheses", mode="reason"),
        ):
            result = self.delegate([assignment])
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(client.requests[-1]["tools"])
            self.assertIn("No tools are available", client.requests[-1]["messages"][0]["content"])

    def test_reasoning_mode_rejects_invented_file_calls(self):
        class CallingClient:
            def complete(self, *args, **kwargs):
                return {"role": "assistant", "tool_calls": [call("read", {"path": "question.txt"})]}

        self.tools.attach_reasoning(CallingClient())
        result = self.delegate([task("Recall a fact", mode="reason")])
        self.assertEqual(result["status"], "error")
        report = result["data"]["reports"][0]
        self.assertEqual(report["data"]["tool_calls"], 0)
        self.assertEqual(report["error"]["code"], "invalid_final_response")

    def test_invalid_delegation_mode_is_rejected_before_model_call(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client)
        self.assertEqual(
            self.delegate([task(mode="unknown")])["error"]["code"], "invalid_arguments"
        )
        self.assertEqual(client.requests, [])

    def test_subagent_recovers_truncated_reasoning_and_counts_summary_requests(self):
        class RecoveringClient:
            def __init__(inner):
                inner.settings = SimpleNamespace(
                    workspace=self.workspace,
                    max_context_chars=120000,
                    max_output_continuations=2,
                    request_timeout_seconds=180,
                    max_output_tokens=131072,
                )
                inner.requests = []

            def complete(inner, messages, **kwargs):
                inner.requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
                if len(inner.requests) == 1:
                    raise CompletionError(
                        "output_limit",
                        "cut off",
                        partial_message={"reasoning": "evidence " * 20000},
                    )
                if kwargs.get("purpose") == "summary":
                    return {
                        "role": "assistant",
                        "content": "Known finding: boundary case needs review.",
                    }
                return {"role": "assistant", "content": "Report: boundary case needs review."}

        client = RecoveringClient()
        self.tools.attach_reasoning(client)
        result = self.delegate()
        report = result["data"]["reports"][0]
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["data"]["model_requests"], len(client.requests))
        self.assertGreater(sum(r.get("purpose") == "summary" for r in client.requests), 1)
        self.assertEqual(report["data"]["output_continuations"], 1)
        self.assertEqual(report["data"]["tool_calls"], 0)
        self.assertIn("Known finding", str(client.requests[-1]["messages"]))
        self.assertEqual(client.requests[0]["tools"], client.requests[-1]["tools"])

    def test_mixed_reports_preserve_success_and_failure(self):
        class MixedClient(ReportingClient):
            def complete(self, messages, **kwargs):
                if "Assignment:\nfails" in messages[1]["content"]:
                    raise RuntimeError("model unavailable")
                return super().complete(messages, **kwargs)

        self.tools.attach_reasoning(MixedClient())
        result = self.delegate([task("works"), task("fails")])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(
            [report["status"] for report in result["data"]["reports"]], ["ok", "error"]
        )
        self.assertIn("report for works", result["data"]["reports"][0]["data"]["report"])

    def test_shared_deadline_returns_without_waiting_for_stalled_model_and_prevents_late_tools(
        self,
    ):
        release = threading.Event()
        returned = threading.Event()
        executed = threading.Event()

        class StalledClient:
            def complete(self, *args, **kwargs):
                release.wait(timeout=2)
                returned.set()
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call("read", {"path": "x"})],
                }

        class TrackingTools(CodingTools):
            def execute_workspace(self, call, context):
                executed.set()
                return super().execute_workspace(call, context)

        tools = TrackingTools(self.workspace)
        tools.attach_reasoning(StalledClient())
        started = time.monotonic()
        try:
            result = json.loads(
                tools.execute(call("delegate", {"tasks": [task()]}), deadline=started + 0.1)
            )
            self.assertEqual(result["status"], "timeout")
            self.assertLess(time.monotonic() - started, 1)
        finally:
            release.set()
        self.assertTrue(returned.wait(timeout=1))
        self.assertFalse(executed.wait(timeout=0.1))

    def test_several_long_reports_all_stay_visible_to_the_parent(self):
        class LongClient:
            def complete(self, messages, *args, **kwargs):
                number = messages[1]["content"].split("Assignment:\n", 1)[1].split()[1]
                return {
                    "role": "assistant",
                    "content": f"start{number} " + "发现 " * 8000 + f"end{number}",
                }

        self.tools.attach_reasoning(LongClient())
        tasks = [task(f"inspect {n} " + "context " * 200) for n in range(3)]
        raw = self.tools.execute(call("delegate", {"tasks": tasks}))
        self.assertLessEqual(len(raw), DELEGATE_RESULT_CHARS)
        result = json.loads(raw)
        self.assertTrue(result["truncated"])
        reports = result["data"]["reports"]
        for number, entry in enumerate(reports):
            self.assertTrue(entry["task"].startswith(f"inspect {number} "))
            self.assertLessEqual(len(entry["task"]), 300)
            self.assertTrue(entry["data"]["report"].startswith(f"start{number} "))
            self.assertTrue(entry["data"]["report"].endswith(f"end{number}"))

    def test_worker_checks_echo_only_a_brief_form_of_written_content(self):
        content = "line of generated content\n" * 2000

        class WritingClient:
            def complete(self, messages, *args, **kwargs):
                if any(message["role"] == "tool" for message in messages):
                    return {"role": "assistant", "content": "Wrote out.txt"}
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call("write", {"path": "out.txt", "content": content})],
                }

        self.tools.worker_allow = frozenset({"write"})
        self.tools.attach_reasoning(WritingClient())
        result = self.delegate([task("write out.txt", mode="inspect")])
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual((self.workspace / "out.txt").read_text(), content)
        check = result["data"]["reports"][0]["data"]["checks"][0]
        self.assertEqual(check["tool"], "write")
        self.assertLessEqual(len(check["arguments"]), 300)
        self.assertTrue(check["arguments"].startswith('{"path": "out.txt"'))
        self.assertFalse(result["truncated"])

    def test_many_worker_calls_are_digested_with_failures_kept(self):
        (self.workspace / "result.txt").write_text("expected value")

        class BusyClient:
            def complete(self, messages, *args, **kwargs):
                done = sum(message["role"] == "tool" for message in messages)
                if done == 20:
                    return {"role": "assistant", "content": "Checked everything"}
                path = "missing.txt" if done == 3 else "result.txt"
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call("read", {"path": path, "offset": 1})],
                }

        self.tools.attach_reasoning(BusyClient())
        data = self.delegate()["data"]["reports"][0]["data"]
        self.assertEqual(data["tool_calls"], 20)
        self.assertEqual(len(data["checks"]), 6)
        self.assertEqual(data["checks_omitted"], 14)
        self.assertEqual(data["check_counts"], {"read:ok": 19, "read:error": 1})
        # The failure is kept; the rest are the latest calls, in call order.
        self.assertEqual(data["checks"][0]["status"], "error")
        self.assertIn("missing.txt", data["checks"][0]["arguments"])
        self.assertTrue(all(check["status"] == "ok" for check in data["checks"][1:]))

    def test_long_reports_are_saved_without_cutting_json_or_dropping_evidence(self):
        report = "evidence " * (DELEGATE_RESULT_CHARS // 4)

        class LongClient:
            def complete(self, *args, **kwargs):
                return {"role": "assistant", "content": report}

        self.tools.attach_reasoning(LongClient())
        raw = self.tools.execute(call("delegate", {"tasks": [task()]}))
        self.assertLessEqual(len(raw), DELEGATE_RESULT_CHARS)
        result = json.loads(raw)
        self.assertTrue(result["truncated"])
        full = json.loads(Path(result["artifact_path"]).read_text())
        self.assertEqual(full["data"]["reports"][0]["data"]["report"], report)

    def test_delegate_role_uses_role_prompt_instead_of_lead_persona(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=1)
        runner = self.tools.registry["delegate"].handler.__self__.runner
        runner.environment_instructions = "LEAD PERSONA"
        runner.worker_skills = "SKILL LIST"
        runner.worker_roles = {"recon": "RECON ROLE INSTRUCTIONS"}
        result = self.delegate([task("do recon", role="recon")])
        self.assertEqual(result["status"], "ok", result)
        system = client.requests[0]["messages"][0]["content"]
        self.assertIn("RECON ROLE INSTRUCTIONS", system)
        self.assertNotIn("LEAD PERSONA", system)
        self.assertIn("SKILL LIST", system)
        self.assertIn("independent investigator", system)

    def test_delegate_parses_redteamRole_first_line(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=1)
        runner = self.tools.registry["delegate"].handler.__self__.runner
        runner.worker_roles = {"vuln-scan": "SCAN ROLE"}
        result = self.delegate([task("redteamRole: vuln-scan\n扫这个资产")])
        self.assertEqual(result["status"], "ok", result)
        self.assertIn("SCAN ROLE", client.requests[0]["messages"][0]["content"])
        self.delegate([task("plain assignment")])
        self.assertNotIn("SCAN ROLE", client.requests[1]["messages"][0]["content"])

    def test_delegate_unknown_role_fails_without_model_request(self):
        client = ReportingClient()
        self.tools.attach_reasoning(client, max_workers=1)
        runner = self.tools.registry["delegate"].handler.__self__.runner
        runner.worker_roles = {"recon": "R"}
        result = self.delegate([task("x", role="exploit")])
        report = result["data"]["reports"][0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["code"], "unknown_role")
        self.assertEqual(client.requests, [])

    def test_dynamic_role_is_used_before_static_role(self):
        client = ReportingClient()
        self.tools.worker_roles = {"recon": "Synthetic static role"}
        self.tools.role_loader = RolePromptCache(lambda role, execution: "Synthetic dynamic role")
        self.tools.attach_reasoning(client, max_workers=1)
        result = self.delegate([task("inspect", role="recon")])
        self.assertEqual(result["data"]["reports"][0]["data"]["role_source"], "dynamic")
        system = client.requests[0]["messages"][0]["content"]
        self.assertIn("Synthetic dynamic role", system)
        self.assertNotIn("Synthetic static role", system)

    def test_dynamic_role_cache_reuses_one_fetch_for_two_assignments(self):
        client = ReportingClient()
        calls = []

        def fetch(role, execution):
            calls.append(role)
            return "Synthetic cached role"

        self.tools.role_loader = RolePromptCache(fetch)
        self.tools.attach_reasoning(client, max_workers=1)
        self.delegate([task("first", role="recon")])
        self.delegate([task("second", role="recon")])
        self.assertEqual(calls, ["recon"])

    def test_dynamic_role_failure_marks_static_fallback(self):
        client = ReportingClient()
        self.tools.worker_roles = {"recon": "Synthetic fallback role"}

        def fetch(role, execution):
            raise TimeoutError("Synthetic source timeout")

        self.tools.role_loader = RolePromptCache(fetch)
        self.tools.attach_reasoning(client, max_workers=1)
        result = self.delegate([task("inspect", role="recon")])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["reports"][0]["data"]["role_source"], "static-fallback")
        self.assertIn("Synthetic fallback role", client.requests[0]["messages"][0]["content"])

    def test_dynamic_role_failure_without_fallback_skips_model(self):
        client = ReportingClient()

        def fetch(role, execution):
            raise RuntimeError("Synthetic source failure")

        self.tools.role_loader = RolePromptCache(fetch)
        self.tools.attach_reasoning(client, max_workers=1)
        result = self.delegate([task("inspect", role="recon")])
        self.assertEqual(result["data"]["reports"][0]["error"]["code"], "role_unavailable")
        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
