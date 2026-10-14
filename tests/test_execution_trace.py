import inspect
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from client_runtime import (
    ClientState,
    client_settings,
    configure_client,
    fork_client,
    snapshot_client,
)
from contracts import CompletionClient, ConfigurableCompletionClient
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from execution_trace import (
    TraceContext,
    current_sink,
    current_trace,
    display_callback,
    trace_scope,
    traced_progress,
)
from generation_metrics import GenerationMetrics
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tool_runtime import ExecutionContext
from tools import CodingTools
from usage import CacheStats, UsageAccumulator


def tool_call(name, arguments):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ExecutionTraceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.workspace / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=2,
            workspace=self.workspace,
            request_retries=0,
        )

    def test_nested_scopes_restore_context_and_sink_after_exceptions(self):
        root = TraceContext.task("session")
        events = []
        with trace_scope(root, events.append):
            with self.assertRaises(RuntimeError), trace_scope(root.request()):
                self.assertEqual(current_trace().task_id, root.task_id)
                self.assertIsNotNone(current_trace().request_id)
                raise RuntimeError("scope failure")
            self.assertEqual(current_trace(), root)
            self.assertEqual(current_sink(), events.append)
        self.assertEqual(current_trace().as_dict(), {})
        self.assertIsNone(current_sink())

    def test_child_contexts_share_cancellation_and_preserve_trace_and_sink(self):
        cancelled = threading.Event()
        root = TraceContext.task("session").request().tool("delegate", "execution")
        sink = Mock()
        parent = ExecutionContext(time.monotonic() + 30, cancelled, root, sink)
        child = parent.child(10)
        self.assertLessEqual(child.deadline, parent.deadline)
        self.assertIs(child.cancelled, cancelled)
        self.assertIs(child.trace, root)
        self.assertIs(child.event_sink, sink)
        agent = child.for_agent("A1")
        self.assertEqual(agent.trace.task_id, root.task_id)
        self.assertEqual(agent.trace.agent_id, "A1")
        self.assertEqual(agent.trace.parent_execution_id, "execution")
        self.assertIsNone(agent.trace.request_id)
        self.assertIsNone(agent.trace.execution_id)
        cancelled.set()
        with self.assertRaises(TimeoutError):
            agent.remaining()

    def test_progress_wrappers_preserve_an_explicit_worker_trace(self):
        events = []
        root = TraceContext.task("session")
        worker = root.agent("A1").request()
        with trace_scope(root):
            publish = traced_progress(events.append)
            publish({"phase": "subagent", "trace": worker.as_dict()})
            publish({"phase": "context"})
        self.assertEqual(events[0]["trace"], worker.as_dict())
        self.assertEqual(events[1]["trace"], root.as_dict())

    def test_display_failure_records_only_callback_type_and_does_not_escape(self):
        events = []

        def fail(event):
            raise RuntimeError("private callback fixture details")

        with trace_scope(TraceContext.task("session"), events.append):
            traced_progress(fail)({"phase": "receiving"})
        self.assertEqual(events[0]["event"], "display_callback_failed")
        self.assertEqual(events[0]["data"], {"callback": "progress", "error_type": "RuntimeError"})
        self.assertNotIn("private", json.dumps(events))

    def test_display_callback_does_not_swallow_user_interruption_or_process_exit(self):
        for exception in [KeyboardInterrupt, SystemExit]:
            with self.subTest(exception=exception):

                def interrupt(error=exception):
                    raise error

                with self.assertRaises(exception):
                    display_callback(interrupt)()

    def test_conversation_correlates_model_tools_progress_and_saved_runs(self):
        events, progress, callback_traces = [], [], []
        client = FakeClient(
            {"role": "assistant", "content": None, "tool_calls": [tool_call("ls", {})]},
            {"role": "assistant", "content": "directory inspected"},
        )
        tools = CodingTools(self.workspace)
        conversation = Conversation(self.settings, client, tools, event_sink=events.append)
        self.addCleanup(conversation.close)
        answer = conversation.ask(
            "list this directory",
            on_progress=progress.append,
            on_tool_result=lambda name, result: callback_traces.append(current_trace()),
        )
        self.assertEqual(answer, "directory inspected")
        root = conversation.last_task_trace
        self.assertEqual(root.session_id, conversation.session_id)
        self.assertTrue(all(event["trace"]["task_id"] == root.task_id for event in events))
        models = [event for event in events if event["event"] == "model_call_started"]
        self.assertEqual(len(models), 2)
        self.assertNotEqual(models[0]["trace"]["request_id"], models[1]["trace"]["request_id"])
        tool_events = [event for event in events if event["event"] == "tool_started"]
        self.assertEqual(tool_events[0]["trace"]["request_id"], models[0]["trace"]["request_id"])
        run = conversation.tool_runs[0]
        self.assertEqual(run["trace"], tool_events[0]["trace"])
        self.assertEqual(callback_traces[0].as_dict(), run["trace"])
        stored = json.loads(self.settings.session_file.read_text())
        self.assertEqual(stored["tool_runs"][0]["trace"], run["trace"])
        journal_path = next(self.workspace.glob(".agent-artifacts/*.progress.jsonl"))
        journal = json.loads(journal_path.read_text().splitlines()[0])
        self.assertEqual(journal["trace"], run["trace"])
        self.assertTrue(all(event["trace"]["task_id"] == root.task_id for event in progress))
        for request in client.requests:
            self.assertTrue(all("trace" not in message for message in request["messages"]))
        self.assertEqual(current_trace().as_dict(), {})

    def test_parallel_workers_keep_parent_links_and_distinct_request_ids(self):
        events, progress = [], []
        (self.workspace / "result.txt").write_text("synthetic content")

        class InspectingClient:
            def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
                if not any(message["role"] == "tool" for message in messages):
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call("read", {"path": "result.txt"})],
                    }
                return {"role": "assistant", "content": "inspection complete"}

        tools = CodingTools(self.workspace, on_task_update=traced_progress(progress.append))
        tools.attach_reasoning(InspectingClient(), max_workers=2)
        root = TraceContext.task("session").request()
        with trace_scope(root, events.append):
            output = tools.execute(
                tool_call(
                    "delegate",
                    {
                        "tasks": [
                            {"task": "inspect one", "expected_output": "report", "mode": "inspect"},
                            {"task": "inspect two", "expected_output": "report", "mode": "inspect"},
                        ]
                    },
                )
            )
        self.assertEqual(json.loads(output)["status"], "ok")
        parent = next(
            event
            for event in events
            if event["event"] == "tool_started" and event["data"]["tool"] == "delegate"
        )
        agents = [event for event in events if event["event"] == "agent_started"]
        self.assertEqual({event["trace"]["agent_id"] for event in agents}, {"A1", "A2"})
        for event in agents:
            self.assertEqual(event["trace"]["parent_execution_id"], parent["trace"]["execution_id"])
            self.assertEqual(event["trace"]["task_id"], root.task_id)
        requests = [event for event in events if event["event"] == "model_call_started"]
        self.assertEqual(len(requests), 4)
        self.assertEqual(len({event["trace"]["request_id"] for event in requests}), 4)
        worker_tools = [
            event
            for event in events
            if event["event"] == "tool_started" and event["data"]["worker"]
        ]
        self.assertEqual(len(worker_tools), 2)
        model_ids = {event["trace"]["request_id"] for event in requests}
        self.assertTrue(all(event["trace"]["request_id"] in model_ids for event in worker_tools))
        for event in progress:
            self.assertEqual(event["trace"]["agent_id"], event["agent_id"])
            self.assertEqual(event["trace"]["task_id"], root.task_id)
        journals = [
            json.loads(path.read_text().splitlines()[0])
            for path in self.workspace.glob(".agent-artifacts/*.progress.jsonl")
        ]
        self.assertEqual(
            {item["trace"]["execution_id"] for item in journals},
            {event["trace"]["execution_id"] for event in worker_tools},
        )

    def test_distinct_tasks_in_one_session_do_not_reuse_ids(self):
        client = FakeClient(
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
        )
        conversation = Conversation(self.settings, client, None)
        self.addCleanup(conversation.close)
        conversation.ask("first")
        first = conversation.last_task_trace
        conversation.ask("second")
        second = conversation.last_task_trace
        self.assertEqual(first.session_id, second.session_id)
        self.assertNotEqual(first.task_id, second.task_id)

    def test_contexts_in_concurrent_threads_do_not_cross_contaminate(self):
        barrier = threading.Barrier(2)
        observed = []

        def run(session):
            trace = TraceContext.task(session)
            with trace_scope(trace):
                barrier.wait(timeout=2)
                observed.append((session, current_trace()))

        threads = [threading.Thread(target=run, args=(session,)) for session in ("one", "two")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(session == trace.session_id for session, trace in observed))
        self.assertEqual(current_trace().as_dict(), {})

    def test_model_metrics_and_recorder_share_the_same_request_context(self):
        client = DeepSeekClient(self.settings)
        recorder = Mock()
        client.recorder = recorder
        events, progress = [], []
        root = TraceContext.task("session").request()

        def complete_once(messages, **kwargs):
            metrics = GenerationMetrics(UsageAccumulator(), kwargs["on_progress"])
            metrics.publish(finished=True)
            return {"role": "assistant", "content": "answer"}

        with (
            trace_scope(root, events.append),
            patch.object(client, "_complete_once", side_effect=complete_once),
        ):
            client.complete(
                [],
                on_content=lambda text: None,
                on_progress=progress.append,
                trace_context=root,
            )
        self.assertEqual(progress[-1]["request_id"], root.request_id)
        self.assertEqual(progress[-1]["trace"], root.as_dict())
        self.assertTrue(all(event["trace"] == root.as_dict() for event in events))
        self.assertTrue(
            all(call.kwargs["trace"] == root.as_dict() for call in recorder.event.call_args_list)
        )

    def test_failed_tasks_emit_a_terminal_event_and_restore_scope(self):
        events = []
        client = FakeClient(CompletionError("failure", "synthetic failure"))
        conversation = Conversation(self.settings, client, None, event_sink=events.append)
        self.addCleanup(conversation.close)
        with self.assertRaises(CompletionError):
            conversation.ask("task")
        self.assertEqual(events[-1]["event"], "task_failed")
        self.assertEqual(current_trace().as_dict(), {})

    def test_direct_model_calls_in_a_request_scope_receive_distinct_request_ids(self):
        client = DeepSeekClient(self.settings)
        requests = []
        root = TraceContext.task("session").request()

        def complete_once(messages, **kwargs):
            requests.append(current_trace())
            return {"role": "assistant", "content": "answer"}

        with trace_scope(root), patch.object(client, "_complete_once", side_effect=complete_once):
            client.complete([])
            client.complete([])
            self.assertEqual(current_trace(), root)
        self.assertEqual({item.task_id for item in requests}, {root.task_id})
        self.assertEqual(len({item.request_id for item in requests}), 2)
        self.assertNotIn(root.request_id, {item.request_id for item in requests})

    def test_event_sink_failure_does_not_interrupt_execution(self):
        sink = Mock(side_effect=OSError("optional sink unavailable"))
        conversation = Conversation(
            self.settings,
            FakeClient({"role": "assistant", "content": "answer"}),
            None,
            event_sink=sink,
        )
        self.addCleanup(conversation.close)
        self.assertEqual(conversation.ask("task"), "answer")
        self.assertGreater(sink.call_count, 0)

    def test_recorder_failure_does_not_mask_model_success_or_failure(self):
        client = DeepSeekClient(self.settings)
        client.recorder = Mock()
        client.recorder.event.side_effect = OSError("optional recorder unavailable")
        with patch.object(
            client, "_complete_once", return_value={"role": "assistant", "content": "answer"}
        ):
            self.assertEqual(client.complete([])["content"], "answer")
        with (
            patch.object(
                client,
                "_complete_once",
                side_effect=CompletionError("failure", "synthetic failure"),
            ),
            self.assertRaises(CompletionError) as raised,
        ):
            client.complete([])
        self.assertEqual(raised.exception.code, "failure")

    def test_real_client_has_an_explicit_configuration_and_fork_contract(self):
        client = DeepSeekClient(self.settings)
        self.assertIsInstance(client, ConfigurableCompletionClient)
        expected = inspect.signature(CompletionClient.complete).parameters
        self.assertEqual(
            set(inspect.signature(client.complete).parameters), set(expected) - {"self"}
        )
        client.recorder = Mock()
        client.deadline = time.monotonic() + 30
        stats = CacheStats()
        changed = replace(self.settings, reasoning_effort="low")
        candidate = fork_client(client, changed, stats)
        self.assertIsNot(candidate, client)
        self.assertIs(candidate.settings, changed)
        self.assertIs(candidate.cache_stats, stats)
        self.assertIs(candidate.recorder, client.recorder)
        self.assertEqual(candidate.deadline, client.deadline)
        self.assertIs(client.settings, self.settings)
        snapshot = snapshot_client(client)
        self.assertIsNot(snapshot, client)
        self.assertIs(snapshot.cache_stats, client.cache_stats)

    def test_injected_clients_are_adapted_at_one_configuration_boundary(self):
        client = FakeClient({"role": "assistant", "content": "answer"})
        state = ClientState.capture(client)
        candidate = fork_client(client, self.settings, CacheStats())
        self.assertIsNot(candidate, client)
        self.assertIs(candidate.settings, self.settings)
        self.assertFalse(hasattr(client, "settings"))
        configure_client(client, self.settings, CacheStats())
        state.restore(client)
        self.assertFalse(hasattr(client, "settings"))
        self.assertFalse(hasattr(client, "cache_stats"))
        fallback = client_settings(client, self.workspace)
        self.assertEqual(fallback.workspace, self.workspace)
        self.assertEqual(fallback.request_timeout_seconds, 180)


if __name__ == "__main__":
    unittest.main()
