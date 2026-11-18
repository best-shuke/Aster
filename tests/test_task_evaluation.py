import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from context_budget import LOCAL_METADATA
from conversation import Conversation
from tests.support import make_settings
from tests.team_support import Steps, TeamScript, say, team, use
from tests.test_conversation import FakeClient
from tests.test_team_runtime import TeamTestCase
from tests.test_tools import python_command
from tools import CodingTools


class EvaluationClient(FakeClient):
    def complete(self, *args, **kwargs):
        if kwargs.get("purpose") == "summary":
            self.responses.insert(0, say("Earlier evidence."))
        result = super().complete(*args, **kwargs)
        self.requests[-1]["usage"] = (result.get(LOCAL_METADATA) or {}).get("usage")
        return result


class EvaluationTeam(TeamScript):
    def __init__(self, **behaviors):
        super().__init__(**behaviors)
        self.model_calls = []

    def complete(self, *args, **kwargs):
        result = super().complete(*args, **kwargs)
        with self.lock:
            self.model_calls.append({
                "purpose": kwargs.get("purpose"),
                "usage": (result.get(LOCAL_METADATA) or {}).get("usage"),
            })
        return result


def report(case, start, conversation, requests, events, *, member_calls=0):
    if os.environ.get("ASTER_EVAL_METRICS") != "1":
        return
    print(json.dumps({
        "case": case,
        "status": conversation.last_outcome["status"],
        "accepted": int(conversation.last_outcome["status"] == "completed"),
        "requests": len(requests),
        "summary_requests": sum(request.get("purpose") == "summary" for request in requests),
        "compactions": sum(event["event"] == "context_compacted" for event in events),
        "tool_calls": len(conversation.execution_snapshot()) + member_calls,
        "repairs": sum(event["event"] == "protocol_repair" for event in events),
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "reported_usage": [request["usage"] for request in requests if request.get("usage")],
    }, ensure_ascii=False))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.events = []

    def conversation(self, *script, max_context_chars=120000):
        settings = make_settings(
            api_key="fixture", model="fixture", base_url="https://example.invalid",
            workspace=self.workspace, session_file=self.workspace / "state" / "session.json",
            max_context_chars=max_context_chars, keep_recent_turns=2, max_task_seconds=60,
            max_subagents=1, mcp_enabled=False, websearch_enabled=False,
        )
        conversation = Conversation(
            settings, EvaluationClient(*script), CodingTools(self.workspace),
            event_sink=self.events.append,
        )
        self.addCleanup(conversation.close)
        return conversation

    def measure(self, case, conversation, prompt, expected, *, repairs=0, previous=(), start=None):
        start = time.perf_counter() if start is None else start
        published = []
        self.assertEqual(conversation.ask(prompt, on_content=published.append), expected)
        self.assertEqual(published, [expected])
        self.assertEqual(conversation.last_outcome["status"], "completed")
        self.assertEqual(sum(event["event"] == "protocol_repair" for event in self.events), repairs)
        report(case, start, conversation, [*previous, *conversation.client.requests], self.events)

    def test_bug_fix_requires_repair(self):
        (self.workspace / "units.py").write_text("SCALE = 2\n")
        conversation = self.conversation(
            use("edit", path="units.py", edits=[{"oldText": "2", "newText": "3"}]),
            say("Fixed without checking."),
            use("exec", command=python_command("from units import SCALE; assert SCALE == 3")),
            {**say("Fixed and checked."), LOCAL_METADATA: {"usage": {"completion_tokens": 4}}},
        )
        self.measure("bug_fix", conversation, "Analyze and fix units.py", "Fixed and checked.", repairs=1)
        self.assertEqual((self.workspace / "units.py").read_text(), "SCALE = 3\n")

    def test_research_only(self):
        source = self.workspace / "facts.txt"
        source.write_text("Temperature: 17 C\n")
        conversation = self.conversation(use("read", path="facts.txt"), say("17 C"))
        self.measure("research", conversation, "Extract the temperature from facts.txt", "17 C")
        self.assertEqual(source.read_text(), "Temperature: 17 C\n")
        self.assertEqual(conversation.last_outcome["code_revision"], 0)

    def test_multi_file_change(self):
        conversation = self.conversation(
            use("write", path="scale.py", content="def scale(value):\n    return value * 2\n"),
            use("write", path="app.py", content="from scale import scale\nRESULT = scale(4)\n"),
            use("exec", command=python_command("from app import RESULT; assert RESULT == 8")),
            say("Integrated and checked."),
        )
        self.measure("multi_file", conversation, "Implement the two-file calculator", "Integrated and checked.")
        self.assertTrue((self.workspace / "scale.py").exists())
        outcome = conversation.last_outcome
        self.assertEqual(outcome["verified_revision"], outcome["code_revision"])

    def test_compaction_keeps_current_constraint(self):
        constraint = "Copy exactly: RED-42"
        conversation = self.conversation(say("RED-42"), max_context_chars=26000)
        conversation.messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": "old " * 1300}
            for index in range(12)
        ]
        start = time.perf_counter()
        answer = conversation.ask(constraint)
        requests = conversation.client.requests
        final_request = next(request for request in requests if request.get("purpose") == "solve")
        self.assertTrue(any(request.get("purpose") == "summary" for request in requests))
        self.assertIn(constraint, [message["content"] for message in final_request["messages"]])
        self.assertEqual(answer, "RED-42")
        self.assertEqual(conversation.last_outcome["status"], "completed")
        report("compaction", start, conversation, requests, self.events)

    def test_interruption_then_recovery(self):
        conversation = self.conversation(use("write", path="draft.txt", content="saved\n"))
        start = time.perf_counter()

        def interrupt(*_arguments):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            conversation.ask("Write draft.txt", on_tool_result=interrupt)
        self.assertEqual(conversation.sessions.read()["last_outcome"]["status"], "cancelled")
        restored = self.conversation(use("read", path="draft.txt"), say("Recovered draft."))
        self.measure(
            "recovery", restored, "Inspect the saved draft without repeating the write",
            "Recovered draft.", previous=conversation.client.requests, start=start,
        )
        self.assertEqual((self.workspace / "draft.txt").read_text(), "saved\n")
        self.assertEqual(sum(run["tool"] == "write" for run in restored.execution_snapshot()), 1)


class TeamEvaluationTests(TeamTestCase):
    def test_team_integration(self):
        discovery = patch("skills.skill_roots", return_value=[])
        discovery.start()
        self.addCleanup(discovery.stop)
        check = python_command("from result import ANSWER; assert ANSWER == 42")
        script = EvaluationTeam(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Save result.py"),
                team("wait", timeout=15),
                say("Saved 42."),
                use("exec", command=check),
                say("Integrated and checked."),
            ),
            engineer=Steps(
                use("write", path="result.py", content="ANSWER = 42\n"),
                use("exec", command=check),
                say("Saved 42."),
            ),
        )
        events = []
        conversation = self.team_conversation(script)
        conversation.event_sink = events.append
        start = time.perf_counter()
        self.assertEqual(conversation.ask("Analyze and implement result.py"), "Integrated and checked.")
        self.assertEqual(conversation.last_outcome["status"], "completed")
        self.assertEqual((self.workspace / "result.py").read_text(), "ANSWER = 42\n")
        self.assertEqual(conversation.last_outcome["code_revision"], conversation.last_outcome["verified_revision"])
        self.assertEqual(sum(event["event"] == "protocol_repair" for event in events), 1)
        member_calls = conversation.team.state.agents["engineer"]["turn"]["tool_calls"]
        self.assertEqual(member_calls, 2)
        report("team_integration", start, conversation, script.model_calls, events, member_calls=member_calls)
