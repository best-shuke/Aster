import io
import json
import sys
import threading
import time
from dataclasses import replace
from typing import ClassVar
from unittest.mock import patch

from context_budget import LOCAL_METADATA
from conversation import Conversation
from deepseek_client import CompletionError
from environment_definitions import EnvironmentEditor
from environment_spec import UNLIMITED_LIMITS
from environments import EnvironmentStore
from main import handle_command
from skills import SkillCatalog
from team_state import TeamStore
from team_tools import duration_text
from tests.team_support import (
    ANSWER,
    Steps,
    TeamScript,
    arrived,
    last_tool,
    last_user,
    say,
    team,
    turn_messages,
    use,
)
from tests.test_environments import ROOT, EnvironmentTestCase
from tools import CodingTools
from ui import Console
from usage import uncached_tokens

MEMBERS = {
    "researcher": {
        "description": "Finds facts in the workspace.",
        "instructions": "You research.",
        "tools": ["read", "grep", "find", "ls"],
    },
    "engineer": {
        "description": "Writes files and runs commands.",
        "instructions": "You build.",
        "tools": ["read", "write", "edit", "exec", "ls"],
    },
    "critic": {
        "description": "Reviews work.",
        "instructions": "You review.",
        "tools": ["read", "ls"],
    },
}


class TeamTestCase(EnvironmentTestCase):
    limits: ClassVar[dict] = {}
    members = MEMBERS

    def setUp(self):
        super().setUp()
        self.settings = replace(self.settings, max_task_seconds=120)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        self.define()

    def define(self, name="crew", **values):
        self.definition(
            name,
            type="multi-agent",
            agents=values.pop("agents", self.members),
            limits={**self.limits, **values.pop("limits", {})},
            **values,
        )
        environment = self.store.resolve(f"program:{name}")
        self.store.approve(environment)
        return environment

    def team_conversation(self, script, name="crew"):
        script.settings = self.settings
        tools = CodingTools(self.workspace)
        conversation = Conversation(
            self.settings,
            script,
            tools,
            skills=SkillCatalog(self.workspace, self.settings.skills_dir),
        )
        conversation.environments = EnvironmentStore(
            self.settings, conversation.templates, user_dir=self.user
        )
        self.addCleanup(conversation.close)
        warnings = conversation.apply_environment(
            conversation.environments.resolve(f"program:{name}")
        )
        self.assertIsNotNone(conversation.team, warnings)
        return conversation

    def ask(self, conversation, prompt="Do the team task."):
        return conversation.ask(prompt, task_contract=ANSWER)

    def events(self, conversation, kind=None):
        events = conversation.team.state.data["events"]
        return [event for event in events if kind is None or event["type"] == kind]


class DirectCollaborationTests(TeamTestCase):
    """Acceptance test 4: a member proactively asks another member, who really works."""

    def test_member_asks_member_who_executes_work_and_returns_results(self):
        def researcher_reads_and_answers(messages, tools):
            # The engineer's reply arrived in a new activation; verify by reading the file.
            reply = last_user(messages)
            assert "reply to" in reply and "wrote answer.txt" in reply, reply
            return use("read", path="answer.txt")

        script = TeamScript(
            main=Steps(
                team(
                    "send_message",
                    to=["researcher"],
                    kind="request",
                    content="What is 6*7? Get it computed and verified.",
                ),
                team("wait", timeout=30),
                lambda messages, tools: (
                    team("wait", timeout=30)
                    if "verified: 42" not in json.dumps(last_tool(messages))
                    else say("The team computed and verified 42.")
                ),
                say("The team computed and verified 42."),
            ),
            researcher=Steps(
                team(
                    "send_message",
                    to=["engineer"],
                    kind="request",
                    content="Please compute 6*7 and write it to answer.txt",
                ),
                say("Asked the engineer; waiting for the result."),
                researcher_reads_and_answers,
                lambda messages, tools: say(
                    "verified: " + last_tool(messages)["data"]["content"].strip()
                ),
            ),
            engineer=Steps(
                use("write", path="answer.txt", content="42\n"),
                say("Done: wrote answer.txt containing 42."),
            ),
        )
        conversation = self.team_conversation(script)
        answer = self.ask(conversation, "Compute 6*7 with the team")
        self.assertEqual(answer, "The team computed and verified 42.")
        self.assertEqual((self.workspace / "answer.txt").read_text(), "42\n")
        # The engineer was woken by the researcher's request, not by main.
        engineer_inbox = script.requests["engineer"][0]["messages"][-1]["content"]
        self.assertIn("request (reply expected) from researcher", engineer_inbox)
        self.assertIn("compute 6*7", engineer_inbox)
        self.assertNotIn("What is 6*7", script.seen("engineer"))
        # The researcher's reply reached main through the runtime, not main's own context.
        sent = self.events(conversation, "message_sent")
        flow = [(event["actor"], event["to"], event["kind"]) for event in sent]
        self.assertIn(("researcher", ["engineer"], "request"), flow)
        self.assertIn(("engineer", ["researcher"], "reply"), flow)
        self.assertIn(("researcher", ["main"], "reply"), flow)
        main_seen = script.seen("main")
        self.assertIn("verified: 42", main_seen)
        self.assertNotIn("You build.", main_seen)
        # Members ran with separate model contexts and their own system prompts.
        self.assertIn(
            'You are "engineer"', script.requests["engineer"][0]["messages"][0]["content"]
        )
        self.assertNotIn("Compute 6*7 with the team", script.seen("researcher"))
        team_state = conversation.team.state.data
        self.assertEqual(team_state["agents"]["engineer"]["activations_total"], 1)
        self.assertEqual(team_state["agents"]["researcher"]["activations_total"], 2)


class BackwardCompatibilityTests(TeamTestCase):
    """Acceptance test 1: without a custom environment nothing about the agent changes."""

    def test_default_agent_has_no_team_runtime_tool_or_prompt(self):
        script = TeamScript(main=Steps(say("plain answer")))
        script.settings = self.settings
        conversation = Conversation(self.settings, script, CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        self.assertIsNone(conversation.team)
        self.assertEqual(conversation.ask("hello", task_contract=ANSWER), "plain answer")
        self.assertNotIn("team", script.tool_names("main"))
        self.assertNotIn("Team environment", script.requests["main"][0]["messages"][0]["content"])

    def test_leaving_a_team_environment_restores_the_default_agent(self):
        script = TeamScript(main=Steps(say("team answer"), say("default answer")))
        conversation = self.team_conversation(script)
        self.ask(conversation)
        self.assertIn("team", script.tool_names("main"))
        conversation.apply_environment(None)
        self.assertIsNone(conversation.team)
        self.assertEqual(self.ask(conversation), "default answer")
        self.assertNotIn("team", script.tool_names("main"))


class SingleAgentTests(TeamTestCase):
    """Acceptance test 3: an environment works with one agent; teams are optional."""

    def test_single_agent_environment_runs_without_team(self):
        self.definition("solo", type="single-agent", skills={})
        self.store.approve(self.store.resolve("program:solo"))
        script = TeamScript(main=Steps(say("solo answer")))
        script.settings = self.settings
        conversation = Conversation(self.settings, script, CodingTools(self.workspace))
        conversation.environments = EnvironmentStore(
            self.settings, conversation.templates, user_dir=self.user
        )
        self.addCleanup(conversation.close)
        conversation.apply_environment(conversation.environments.resolve("program:solo"))
        self.assertIsNone(conversation.team)
        self.assertEqual(self.ask(conversation), "solo answer")
        self.assertNotIn("team", script.tool_names("main"))

    def test_main_agent_may_solve_a_simple_task_alone_in_a_team(self):
        script = TeamScript(main=Steps(say("4")))
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation, "What is 2+2?"), "4")
        agents = conversation.team.state.data["agents"]
        self.assertTrue(all(record["activations_total"] == 0 for record in agents.values()))
        self.assertEqual(set(script.requests), {"main"})

    def test_team_with_a_single_member(self):
        self.define("pair", agents={"researcher": MEMBERS["researcher"]})
        script = TeamScript(
            main=Steps(
                team("send_message", to=["researcher"], kind="request", content="Check README"),
                team("wait", timeout=30),
                say("done"),
            ),
            researcher=Steps(say("README checked")),
        )
        conversation = self.team_conversation(script, "pair")
        self.assertEqual(self.ask(conversation), "done")
        self.assertIn("README checked", script.seen("main"))


class DefaultTeamTemplateTests(TeamTestCase):
    def test_research_team_shares_evidence_writes_and_gets_a_dynamic_specialist_review(self):
        EnvironmentEditor(self.store).create("research", template="multi-agent", scope="program")
        self.resource(
            self.workspace / "papers.txt",
            "Study A: observational, n=24. Study B: observational, n=8. "
            "Neither establishes causality.\n",
        )
        review = (
            "# 文献综述\n\nStudy A（n=24）与 Study B（n=8）均为观察性研究，"
            "提供的材料不足以得出因果结论。\n"
        )
        assignments = [
            team(
                "spawn_agent",
                name="methodologist",
                description="Independently checks study design and causal claims.",
                instructions="Check the draft against the provided study notes.",
                tools=["read"],
            ),
            team("create_task", title="Read the study notes", assignee="researcher"),
            team(
                "create_task",
                title="Write literature-review.md using evidence artifact a1",
                assignee="writer",
                depends_on=["t1"],
            ),
            team(
                "create_task",
                title="Review the draft against papers.txt",
                assignee="methodologist",
                depends_on=["t2"],
            ),
        ]

        def main(messages, tools):
            index = len(script.requests["main"]) - 1
            if index < len(assignments):
                return assignments[index]
            if "Evidence check complete" in json.dumps(arrived(messages)):
                return say("综述已保存，样本量和因果表述已经独立核查。")
            return team("wait", timeout=30)

        script = TeamScript(
            main=main,
            researcher=Steps(
                use("read", path="papers.txt"),
                team(
                    "share_artifact",
                    title="Study evidence",
                    content="papers.txt: Study A n=24, Study B n=8; observational, no causal proof.",
                ),
                say("Evidence is available in artifact a1, based on papers.txt."),
            ),
            writer=Steps(
                team("get_artifact", artifact_id="a1"),
                use("write", path="literature-review.md", content=review),
                say("Draft saved to literature-review.md using evidence a1."),
            ),
            methodologist=Steps(
                use("read", path="papers.txt"),
                use("read", path="literature-review.md"),
                say("Evidence check complete: sample sizes agree; no causal conclusion is claimed."),
            ),
        )
        conversation = self.team_conversation(script, "research")
        answer = conversation.ask("根据提供的论文材料写一份文献综述，并独立核查方法与结论。")
        self.assertEqual(answer, "综述已保存，样本量和因果表述已经独立核查。")
        self.assertEqual((self.workspace / "literature-review.md").read_text(), review)
        self.assertEqual(script.tool_names("methodologist"), {"read", "team"})
        agents = conversation.team.state.data["agents"]
        self.assertEqual(agents["methodologist"]["dynamic"]["spawned_by"], "main")
        self.assertTrue(
            all(
                record["activations_total"] == 0
                for name, record in agents.items()
                if name not in {"researcher", "writer", "methodologist"}
            )
        )
        events = self.events(conversation)
        written = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "task_updated"
            and event["task_id"] == "t2"
            and event["status"] == "completed"
        )
        reviewed = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "activation_started" and event["actor"] == "methodologist"
        )
        self.assertLess(written, reviewed)

    def test_specialists_have_distinct_permissions_and_activate_only_when_addressed(self):
        EnvironmentEditor(self.store).create(
            "specialists", template="coding-team", scope="program"
        )

        def main(messages, tools):
            if len(script.requests["main"]) == 1:
                return team(
                    "send_message",
                    to=["tester", "security"],
                    kind="request",
                    content="Check your assigned capabilities and report the result.",
                )
            if sum(item["type"] == "reply" for item in arrived(messages)) >= 2:
                return say("Specialists checked.")
            return team("wait", timeout=30)

        def tester_report(messages, tools):
            result = last_tool(messages)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["data"]["exit_code"], 0)
            self.assertEqual(result["data"]["output"].strip(), "42")
            return say("Verified 6*7=42 with a local command.")

        script = TeamScript(
            main=main,
            tester=Steps(
                use("exec", command=f'"{sys.executable}" -c "print(6 * 7)"'), tester_report
            ),
            security=Steps(
                use("write", path="forbidden.txt", content="must not be written"),
                say("Write was rejected; the review remains read-only."),
            ),
        )
        conversation = self.team_conversation(script, "specialists")
        self.assertEqual(self.ask(conversation), "Specialists checked.")
        self.assertTrue({"write", "edit", "exec"} <= script.tool_names("tester"))
        self.assertFalse({"write", "edit", "exec"} & script.tool_names("security"))
        self.assertFalse((self.workspace / "forbidden.txt").exists())
        self.assertIn("Requested a tool that is not available", script.seen("security"))
        self.assertIn("Verified 6*7=42", script.seen("main"))
        self.assertTrue(
            all(
                record["activations_total"] == 0
                for name, record in conversation.team.state.data["agents"].items()
                if name not in {"tester", "security"}
            )
        )


class InformalDiscussionTests(TeamTestCase):
    """Acceptance test 5: members organize a discussion with plain messages and a group."""

    def test_members_discuss_in_a_group_without_any_protocol(self):
        def researcher(messages, tools):
            current = turn_messages(messages)
            if "request (reply expected) from main" in current[0]["content"]:
                if not any(message["role"] == "tool" for message in current):
                    return team("create_group", name="naming", members=["critic", "engineer"])
                group = last_tool(messages)["data"]
                if "message_id" not in group:
                    return team(
                        "send_message",
                        to=[group["address"]],
                        kind="request",
                        content="Proposal: call the module `ledger`. Objections?",
                    )
                return say("Asked the group for opinions.")
            seen = json.dumps(messages)
            if "critic" in seen and "engineer" in seen and seen.count("reply to m") >= 2:
                if last_tool(messages).get("data", {}).get("decision") is None:
                    return team(
                        "record_decision",
                        title="Module name",
                        decision="Use `ledger_core`",
                        rationale="critic objected to `ledger`; engineer agreed",
                    )
                return say("Decision: use `ledger_core` (agreed with critic and engineer).")
            return say("Still collecting opinions.")

        script = TeamScript(
            main=Steps(
                team(
                    "send_message",
                    to=["researcher"],
                    kind="request",
                    content="Agree on a module name with the others.",
                ),
                *[
                    lambda messages, tools: (
                        say("Agreed name: ledger_core")
                        if "ledger_core" in json.dumps(last_tool(messages))
                        else team("wait", timeout=30)
                    )
                ]
                * 4,
            ),
            researcher=researcher,
            critic=Steps(
                say("Objection: `ledger` collides with an existing package; prefer `ledger_core`.")
            ),
            engineer=Steps(say("Fine with `ledger_core`.")),
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "Agreed name: ledger_core")
        data = conversation.team.state.data
        self.assertEqual(data["protocols"], {})
        self.assertEqual(len(data["groups"]), 1)
        group = next(iter(data["groups"].values()))
        self.assertEqual(set(group["members"]), {"researcher", "critic", "engineer"})
        for member in ("critic", "engineer"):
            self.assertIn(
                f"in group {group['id']}", script.requests[member][0]["messages"][-1]["content"]
            )
        self.assertEqual(data["decisions"][0]["decision"], "Use `ledger_core`")
        # The researcher answered main only after both opinions arrived.
        deferred = [event for event in self.events(conversation, "obligations_deferred")]
        self.assertTrue(deferred)
        reply = [
            message
            for message in data["messages"].values()
            if message["sender"] == "researcher" and message["recipients"] == ["main"]
        ]
        self.assertEqual(len(reply), 1)
        self.assertIn("ledger_core", reply[0]["content"])


class MeetingProtocolTests(TeamTestCase):
    """Acceptance test 6: a member starts a formal meeting; members produce an outcome."""

    def test_member_initiated_meeting_runs_rounds_and_records_decisions(self):
        def critic(messages, tools):
            prompt = turn_messages(messages)[0]["content"]
            if "request (reply expected) from main" in prompt:
                if last_tool(messages).get("data", {}).get("protocol") == "meeting":
                    return say("Meeting p1 started.")
                return team(
                    "start_protocol",
                    protocol="meeting",
                    topic="SQLite or Postgres",
                    participants=["researcher", "engineer"],
                    rounds=2,
                    goal="Pick a database for the prototype",
                )
            if "All rounds are complete" in prompt:
                if last_tool(messages).get("data", {}).get("status") == "concluded":
                    return say("Concluded: SQLite for the prototype.")
                return team(
                    "conclude",
                    protocol_id="p1",
                    summary="SQLite now; revisit Postgres when concurrency matters.",
                    decisions=["Use SQLite for the prototype"],
                )
            return say("Meeting finished: SQLite for the prototype.")

        def researcher(messages, tools):
            prompt = turn_messages(messages)[0]["content"]
            if "Round 1 of 2" in prompt:
                if last_tool(messages).get("data", {}).get("id") == "p1":
                    return say("Contributed.")
                return team(
                    "contribute",
                    protocol_id="p1",
                    content="Research: SQLite handles the prototype's load.",
                )
            assert "SQLite handles" in prompt and "Postgres scales" in prompt, prompt
            return say("Agree on SQLite; ready.")

        def engineer(messages, tools):
            prompt = turn_messages(messages)[0]["content"]
            if "Round 1 of 2" in prompt:
                return say("Engineering: Postgres scales better but adds ops cost.")
            return say("SQLite is fine for now.")

        script = TeamScript(
            main=Steps(
                team(
                    "send_message",
                    to=["critic"],
                    kind="request",
                    content="Decide the database; organize a discussion if useful.",
                ),
                *[
                    lambda messages, tools: (
                        say("Team decided: SQLite.")
                        if "SQLite for the prototype" in json.dumps(last_tool(messages))
                        else team("wait", timeout=30)
                    )
                ]
                * 6,
            ),
            critic=critic,
            researcher=researcher,
            engineer=engineer,
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "Team decided: SQLite.")
        data = conversation.team.state.data
        meeting = data["protocols"]["p1"]
        self.assertEqual(meeting["status"], "concluded")
        self.assertEqual(meeting["initiator"], "critic")
        rounds = meeting["contributions"]
        self.assertEqual(set(rounds), {"1", "2"})
        self.assertEqual(set(rounds["1"]), {"researcher", "engineer"})
        self.assertFalse(rounds["1"]["researcher"]["auto"])
        self.assertTrue(rounds["1"]["engineer"]["auto"])
        self.assertEqual(data["decisions"][0]["decision"], "Use SQLite for the prototype")
        self.assertEqual(data["decisions"][0]["source"], "p1")
        artifact = data["artifacts"][meeting["outcome"]["artifact"]]
        self.assertIn("Postgres scales", artifact["content"])
        self.assertIn("SQLite now", artifact["content"])
        # Participants were independent agents with their own model requests.
        self.assertGreaterEqual(len(script.requests["researcher"]), 2)
        self.assertGreaterEqual(len(script.requests["engineer"]), 2)
        self.assertNotIn("ops cost", script.requests["researcher"][0]["messages"][-1]["content"])

    def test_independent_research_isolates_participants_until_findings_are_revealed(self):
        def researcher(messages, tools):
            current = turn_messages(messages)
            if len(current) == 1:
                return team("send_message", to=["critic"], content="What did you find?")
            if last_tool(messages)["error"] and last_tool(messages)["error"]["code"] == "isolated":
                return say("Finding A: the cache hit rate is 90%.")
            raise AssertionError("isolation was not enforced")

        script = TeamScript(
            main=Steps(
                team(
                    "start_protocol",
                    protocol="independent_research",
                    topic="Why is the service slow?",
                    participants=["researcher", "critic"],
                ),
                *[
                    lambda messages, tools: (
                        say("Both findings collected.")
                        if "Finding B" in json.dumps(last_tool(messages))
                        else team("wait", timeout=30)
                    )
                ]
                * 4,
            ),
            researcher=researcher,
            critic=Steps(say("Finding B: database locks under load.")),
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "Both findings collected.")
        record = conversation.team.state.data["protocols"]["p1"]
        self.assertEqual(record["status"], "concluded")
        self.assertNotIn("Finding A", script.seen("critic"))
        self.assertNotIn("Finding B", script.seen("researcher"))
        self.assertIn("Finding A", script.seen("main"))
        self.assertIn("Finding B", script.seen("main"))


class TaskDelegationTests(TeamTestCase):
    """Acceptance test 7: create, accept, execute and coordinate tasks, then integrate."""

    def test_dependent_tasks_run_in_order_and_results_are_integrated(self):
        def engineer(messages, tools):
            current = turn_messages(messages)
            prompt = current[0]["content"]
            if "t2" in prompt and "assigned to you" in prompt:
                step = sum(message["role"] == "tool" for message in current)
                if step == 0:
                    return team("get_task", task_id="t1")
                if step == 1:
                    facts = last_tool(messages)["data"]["result"]
                    return use("write", path="report.md", content=f"# Report\n{facts}\n")
                if step == 2:
                    return team(
                        "create_task",
                        title="Find one more fact",
                        assignee="researcher",
                        description="Need a third fact for the report",
                    )
                return say("Waiting for the extra fact.")
            # Woken by the subtask result; finish t2 explicitly.
            if "t3" in prompt and "completed" in prompt:
                if last_tool(messages).get("data", {}).get("status") == "completed":
                    return say("Report finished.")
                return team(
                    "update_task",
                    task_id="t2",
                    status="completed",
                    result="report.md written with 3 facts",
                )
            return say("Nothing to do.")

        def researcher(messages, tools):
            prompt = turn_messages(messages)[0]["content"]
            if "t1" in prompt and "assigned to you" in prompt:
                if last_tool(messages).get("data", {}).get("status") == "completed":
                    return say("Facts delivered.")
                return team(
                    "update_task", task_id="t1", status="completed", result="fact A; fact B"
                )
            return say("fact C")

        def main(messages, tools):
            requests = len(script.requests["main"])
            if requests == 1:
                return team("create_task", title="Collect facts", assignee="researcher")
            if requests == 2:
                return team(
                    "create_task",
                    title="Write report",
                    assignee="engineer",
                    depends_on=["t1"],
                    description="Use t1's facts",
                )
            seen = json.dumps(messages)
            if "report.md written with 3 facts" in seen:
                return say("Report delivered: report.md")
            return team("wait", timeout=30)

        script = TeamScript(main=main, researcher=researcher, engineer=engineer)
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "Report delivered: report.md")
        tasks = conversation.team.state.data["tasks"]
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertEqual(tasks["t2"]["status"], "completed")
        self.assertFalse(tasks["t2"]["auto_completed"])
        self.assertEqual(tasks["t3"]["creator"], "engineer")
        self.assertEqual(tasks["t3"]["depth"], 2)
        self.assertTrue(tasks["t3"]["auto_completed"])
        self.assertEqual(tasks["t3"]["result"], "fact C")
        self.assertIn("fact A; fact B", (self.workspace / "report.md").read_text())
        events = conversation.team.state.data["events"]
        t1_done = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "task_updated" and event["task_id"] == "t1"
        )
        engineer_start = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "activation_started" and event["actor"] == "engineer"
        )
        self.assertLess(t1_done, engineer_start)

    def test_claimed_task_is_still_owed_and_keeps_the_report_if_a_later_turn_fails(self):
        # Reproduces a live run: the researcher marks its task in_progress, asks the
        # engineer, ends its turn with a report, then fails in the next activation.
        def researcher_fails(messages, tools):
            raise CompletionError("provider_error", "synthetic outage")

        script = TeamScript(
            main=Steps(
                team("create_task", title="Gather precedent", assignee="researcher"),
                *[
                    lambda messages, tools: (
                        say("got it")
                        if any(item["type"] == "task_update" for item in arrived(messages))
                        else team("wait", timeout=30)
                    )
                ]
                * 4,
            ),
            researcher=Steps(
                team("update_task", task_id="t1", status="in_progress"),
                team("send_message", to=["engineer"], kind="request", content="run the probe"),
                say("Report: Excel returns #DIV/0!, statistics raises."),
                researcher_fails,
            ),
            engineer=Steps(say("probe output: ok")),
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "got it")
        task = conversation.team.state.data["tasks"]["t1"]
        self.assertEqual(task["status"], "blocked")
        self.assertIn("Latest report: Report: Excel returns #DIV/0!", task["notes"][-1])
        self.assertIn("Excel returns #DIV/0!", script.seen("main"))

    def test_review_requests_create_review_tasks_for_reviewers(self):
        script = TeamScript(
            main=Steps(
                team("share_artifact", title="Plan", content="Step 1: build. Step 2: ship."),
                team(
                    "request_review", subject="a1", reviewers=["critic"], instructions="Be strict"
                ),
                team("wait", timeout=30),
                say("Reviewed."),
            ),
            critic=Steps(
                lambda messages, tools: team("get_artifact", artifact_id="a1"),
                lambda messages, tools: say(
                    "Review: missing tests in " + last_tool(messages)["data"]["content"]
                ),
            ),
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "Reviewed.")
        task = conversation.team.state.data["tasks"]["t1"]
        self.assertEqual(task["kind"], "review")
        self.assertEqual(task["status"], "completed")
        self.assertIn("missing tests in Step 1", task["result"])
        self.assertIn("missing tests", script.seen("main"))


class RecoveryTests(TeamTestCase):
    """Acceptance test 8: interrupted sessions restore tasks, messages and completed work."""

    def test_interrupted_member_work_resumes_after_restart(self):
        started = threading.Event()

        def engineer_blocks(messages, tools):
            started.set()
            script.options()["cancelled"].wait(10)
            raise CompletionError("cancelled", "Model request cancelled")

        def main_turn_one(messages, tools):
            started.wait(10)
            return say("The engineer is still building; I will continue next turn.")

        script = TeamScript(
            main=Steps(
                team("create_task", title="Collect facts", assignee="researcher"),
                team("wait", timeout=30),
                team("send_message", to=["engineer"], kind="request", content="Build the widget"),
                main_turn_one,
                main_turn_one,
            ),
            researcher=Steps(
                team("update_task", task_id="t1", status="completed", result="facts: A, B"),
                say("Facts done."),
            ),
            engineer=engineer_blocks,
        )
        conversation = self.team_conversation(script)
        self.ask(conversation, "Build the widget with the team; don't wait for the engineer")
        session_id = conversation.session_id
        engineer = conversation.team.state.data["agents"]["engineer"]
        self.assertEqual(engineer["status"], "idle")
        mailbox = conversation.team.state.data["mailboxes"]["engineer"]
        self.assertTrue(mailbox and mailbox[0]["redelivered"])
        self.assertTrue(self.events(conversation, "activation_cancelled"))
        conversation.close()

        # A new process: the active session file is reloaded, then resumed explicitly.
        resumed_script = TeamScript(
            main=Steps(
                team("wait", timeout=30),
                lambda messages, tools: say(
                    "Resumed: " + last_tool(messages)["data"]["items"][0]["content"]
                ),
            ),
            engineer=Steps(say("Widget built after resuming.")),
        )
        resumed_script.settings = self.settings
        restarted = Conversation(self.settings, resumed_script, CodingTools(self.workspace))
        restarted.environments = EnvironmentStore(
            self.settings, restarted.templates, user_dir=self.user
        )
        self.addCleanup(restarted.close)
        restarted.resume(session_id)
        self.assertEqual(restarted.session_id, session_id)
        state = restarted.team.state.data
        self.assertEqual(state["tasks"]["t1"]["status"], "completed")
        self.assertEqual(state["tasks"]["t1"]["result"], "facts: A, B")
        self.assertIn("Build the widget", json.dumps(state["messages"]))
        answer = self.ask(restarted, "continue")
        self.assertEqual(answer, "Resumed: Widget built after resuming.")
        prompt = resumed_script.requests["engineer"][0]["messages"][-1]["content"]
        self.assertIn("redelivered after an interruption", prompt)
        self.assertIn("Build the widget", prompt)
        # The engineer's private context from before the restart was preserved.
        self.assertEqual(
            sum(
                "Build the widget" in (message.get("content") or "")
                for message in resumed_script.requests["engineer"][0]["messages"]
            ),
            2,
        )

    def test_crash_mid_activation_requeues_items_and_repairs_tool_calls(self):
        script = TeamScript(
            main=Steps(
                team("send_message", to=["critic"], kind="request", content="Review README"),
                team("wait", timeout=30),
                say("first turn done"),
            ),
            critic=Steps(say("Reviewed once.")),
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        runtime = conversation.team
        session_id = conversation.session_id
        # Simulate a crash: the critic was running with a tool call whose result was lost.
        with runtime.lock:
            data = runtime.state.data
            message = runtime._post(
                "main", ["critic"], "Second review", kind="request", hop=1, depth=0
            )
            data["mailboxes"]["critic"].clear()
            data["agents"]["critic"]["status"] = "running"
            data["agents"]["critic"]["inflight"] = [
                {"kind": "request", "ref": message["id"], "sender": "main", "hop": 1, "depth": 0}
            ]
            context = runtime.store.load_context("critic")
            context.append(use("read", path="README.md"))
            runtime.store.save_context("critic", context)
            runtime._persist()
        runtime.store.close()
        runtime.store = None
        conversation.team = None
        conversation.close()

        recovered_script = TeamScript(
            main=Steps(team("wait", timeout=30), say("second turn done")),
            critic=Steps(say("Reviewed again after recovery.")),
        )
        recovered_script.settings = self.settings
        restarted = Conversation(self.settings, recovered_script, CodingTools(self.workspace))
        restarted.environments = EnvironmentStore(
            self.settings, restarted.templates, user_dir=self.user
        )
        self.addCleanup(restarted.close)
        restarted.resume(session_id)
        self.assertEqual(restarted.team.state.data["agents"]["critic"]["status"], "idle")
        self.assertEqual(len(restarted.team.state.data["mailboxes"]["critic"]), 1)
        self.ask(restarted, "continue")
        critic_messages = recovered_script.requests["critic"][0]["messages"]
        self.assertTrue(
            any(
                message["role"] == "tool" and "execution_unknown" in message["content"]
                for message in critic_messages
            )
        )
        self.assertIn("Reviewed again after recovery.", recovered_script.seen("main"))


class ResourceLimitTests(TeamTestCase):
    """Acceptance test 9: loops, budgets, timeouts and concurrency stay bounded."""

    limits: ClassVar[dict] = {"max_message_hops": 4, "max_activations_per_agent": 6}

    def test_unlimited_budgets_keep_working_past_the_previous_caps(self):
        self.define("unlimited", limits=dict.fromkeys(UNLIMITED_LIMITS))
        self.resource(self.workspace / "evidence.txt", "available evidence\n")

        def begin_with_large_usage(messages, tools):
            with conversation.team.lock:
                data = conversation.team.state.data
                data["turn"]["messages"] = 10_000_000
                data["agents"]["engineer"]["turn"].update(
                    activations=1_000_000, tool_calls=1_000_000_000, tokens=1_000_000_000_000
                )
            return team("send_message", to=["engineer"], kind="request", content="Read evidence")

        script = TeamScript(
            main=Steps(begin_with_large_usage, team("wait", timeout=30), say("Complete.")),
            engineer=Steps(use("read", path="evidence.txt"), say("Evidence checked.")),
        )
        conversation = self.team_conversation(script, "unlimited")
        self.assertEqual(self.ask(conversation), "Complete.")
        self.assertIn("Evidence checked.", script.seen("main"))
        usage = conversation.team.state.data["agents"]["engineer"]["turn"]
        self.assertGreater(usage["activations"], 1_000_000)
        self.assertGreater(usage["tool_calls"], 1_000_000_000)
        self.assertGreater(usage["tokens"], 1_000_000_000_000)
        self.assertFalse(self.events(conversation, "budget_exhausted"))

    def test_circular_requests_terminate(self):
        def ping(other):
            def behavior(messages, tools):
                if len(turn_messages(messages)) == 1:
                    return team("send_message", to=[other], kind="request", content="ping")
                return say("pong")

            return behavior

        def main(messages, tools):
            result = json.dumps(last_tool(messages))
            if len(script.requests["main"]) == 1:
                return team("send_message", to=["researcher"], kind="request", content="go")
            if '"reason": "idle"' in result:
                return say("loop ended")
            return team("wait", timeout=30)

        script = TeamScript(main=main, researcher=ping("critic"), critic=ping("researcher"))
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "loop ended")
        data = conversation.team.state.data
        self.assertLess(len(data["messages"]), 20)
        self.assertIn("hop_limit", script.seen("researcher") + script.seen("critic"))
        replies_to_main = [
            message
            for message in data["messages"].values()
            if message["recipients"] == ["main"] and message["reply_to"] == "m1"
        ]
        self.assertEqual(len(replies_to_main), 1)
        for record in data["agents"].values():
            self.assertLessEqual(record["turn"]["activations"], 6)

    def test_activation_limit_answers_requesters_instead_of_running(self):
        self.define("tight", limits={"max_activations_per_agent": 1})
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="first"),
                team("wait", timeout=30),
                team("send_message", to=["engineer"], kind="request", content="second"),
                team("wait", timeout=30),
                say("done"),
                team("wait", timeout=30),
                lambda messages, tools: say(
                    "next turn: " + last_tool(messages)["data"]["items"][0]["content"]
                ),
            ),
            engineer=Steps(say("first handled"), say("second handled")),
        )
        conversation = self.team_conversation(script, "tight")
        self.ask(conversation)
        self.assertIn("first handled", script.seen("main"))
        self.assertIn("reached its limit of 1 activations", script.seen("main"))
        self.assertEqual(len(script.requests["engineer"]), 1)
        self.assertTrue(self.events(conversation, "budget_exhausted"))
        # The interim notice did not settle the request: the next turn delivers the answer.
        self.assertEqual(self.ask(conversation, "continue"), "next turn: second handled")

    def test_activation_timeout_reports_failure_to_requester(self):
        def slow(messages, tools):
            deadline = script.options()["deadline"]
            time.sleep(max(0, deadline - time.monotonic()) + 0.05)
            raise CompletionError("task_deadline", "Task deadline reached")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="slow job"),
                team("wait", timeout=30),
                say("reported"),
            ),
            engineer=slow,
        )
        conversation = self.team_conversation(script)
        conversation.team.limits = replace(conversation.team.limits, activation_timeout_seconds=1)
        started = time.monotonic()
        self.ask(conversation)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn("(no report: timeout", script.seen("main"))
        failed = self.events(conversation, "activation_failed")
        self.assertEqual(failed[0]["status"], "timeout")

    def test_token_budget_counts_uncached_prompt_and_output_tokens(self):
        usage = {
            "prompt_tokens": 50_000,
            "completion_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 48_000},
        }
        script = TeamScript(
            main=Steps(
                team("send_message", to=["critic"], kind="request", content="review"),
                team("wait", timeout=30),
                say("done"),
            ),
            critic=Steps({**say("reviewed"), LOCAL_METADATA: {"usage": usage}}),
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        turn = conversation.team.state.data["agents"]["critic"]["turn"]
        self.assertEqual((turn["tokens"], turn["model_requests"]), (2_400, 1))

    def test_uncached_tokens_prefers_reported_usage(self):
        self.assertEqual(
            uncached_tokens(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 90},
                }
            ),
            17,
        )
        self.assertEqual(uncached_tokens({"prompt_tokens": 100, "completion_tokens": 7}), 107)
        self.assertEqual(
            uncached_tokens({"total_tokens": 50, "prompt_tokens_details": {"cached_tokens": 30}}),
            20,
        )
        self.assertIsNone(uncached_tokens({"prompt_tokens_details": "?"}))
        self.assertIsNone(uncached_tokens(None))

    def test_concurrency_limit_queues_members(self):
        self.define("serial", limits={"max_concurrent_agents": 1})
        lock = threading.Lock()
        active = {"now": 0, "peak": 0}

        def member(messages, tools):
            with lock:
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
            time.sleep(0.05)
            with lock:
                active["now"] -= 1
            return say("done")

        def main(messages, tools):
            if len(script.requests["main"]) == 1:
                return team("send_message", to=["all"], kind="request", content="report in")
            if sum(item["type"] == "reply" for item in arrived(messages)) >= 3:
                return say("all reported")
            return team("wait", timeout=30)

        script = TeamScript(main=main, researcher=member, engineer=member, critic=member)
        conversation = self.team_conversation(script, "serial")
        self.assertEqual(self.ask(conversation), "all reported")
        self.assertEqual(active["peak"], 1)
        self.assertEqual(
            {name for name in ("researcher", "engineer", "critic") if script.requests[name]},
            {"researcher", "engineer", "critic"},
        )

    def test_default_capacity_runs_twenty_members_and_queues_the_rest(self):
        self.settings = replace(self.settings, max_subagents=20)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        members = {f"worker-{index}": {"description": "Reports in."} for index in range(128)}
        self.define("large", agents=members)
        barrier = threading.Barrier(20)
        lock = threading.Lock()
        active = {"now": 0, "peak": 0, "started": 0}

        def member(messages, tools):
            with lock:
                active["now"] += 1
                active["started"] += 1
                active["peak"] = max(active["peak"], active["now"])
                first_batch = active["started"] <= 20
            try:
                if first_batch:
                    barrier.wait(timeout=10)
                return say("done")
            finally:
                with lock:
                    active["now"] -= 1

        def main(messages, tools):
            if len(script.requests["main"]) == 1:
                return team("send_message", to=["all"], kind="request", content="report in")
            if sum(item["type"] == "reply" for item in arrived(messages)) == len(members):
                return say("all reported")
            return team("wait", timeout=30)

        script = TeamScript(main=main, **dict.fromkeys(members, member))
        conversation = self.team_conversation(script, "large")
        self.assertEqual(self.ask(conversation), "all reported")
        self.assertEqual(active["peak"], 20)
        self.assertTrue(all(len(script.requests[name]) == 1 for name in members))
        self.assertFalse(self.events(conversation, "budget_exhausted"))

    def editors(self, right_edit, *, after_left=None):
        """left and right read shared.txt; left edits first, then right edits its stale read."""
        self.define(
            "editors",
            agents={
                "left": {"description": "edits", "tools": ["read", "edit"]},
                "right": {"description": "edits", "tools": ["read", "edit", "write"]},
            },
        )
        (self.workspace / "shared.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        right_read, left_done = threading.Event(), threading.Event()

        def left(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            if step == 0:
                return use("read", path="shared.txt")
            if step == 1:
                right_read.wait(5)
                return use(
                    "edit", path="shared.txt", edits=[{"oldText": "alpha", "newText": "ALPHA"}]
                )
            if after_left is not None:
                after_left()
            left_done.set()
            return say("left edited")

        def right(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            if step == 0:
                return use("read", path="shared.txt")
            if step == 1:
                right_read.set()
                left_done.wait(5)
                return right_edit
            if step == 2 and last_tool(messages)["status"] != "ok":
                return use("read", path="shared.txt")
            if step == 3:
                return use(
                    "edit", path="shared.txt", edits=[{"oldText": "gamma", "newText": "GAMMA"}]
                )
            return say("right edited")

        def main(messages, tools):
            if len(script.requests["main"]) == 1:
                return team("send_message", to=["left", "right"], kind="request", content="edit")
            if sum(item["type"] == "reply" for item in arrived(messages)) >= 2:
                return say("both edited")
            return team("wait", timeout=30)

        script = TeamScript(main=main, left=left, right=right)
        conversation = self.team_conversation(script, "editors")
        self.assertEqual(self.ask(conversation), "both edited")
        right_results = [
            json.loads(message["content"])
            for message in script.requests["right"][-1]["messages"]
            if message["role"] == "tool"
        ]
        return (self.workspace / "shared.txt").read_text(), right_results[1]

    def test_concurrent_edits_to_one_file_merge_automatically(self):
        content, result = self.editors(
            use("edit", path="shared.txt", edits=[{"oldText": "beta", "newText": "BETA"}])
        )
        self.assertEqual(content, "ALPHA\nBETA\ngamma\n")
        self.assertEqual(result["status"], "ok")
        self.assertIn("merged with theirs", result["data"]["merged"])

    def test_full_file_write_from_a_stale_read_keeps_a_teammates_edit(self):
        content, result = self.editors(
            use("write", path="shared.txt", content="alpha\nbeta\ngamma\ndelta\n")
        )
        self.assertEqual(content, "ALPHA\nbeta\ngamma\ndelta\n")
        self.assertIn("merged", result["data"])

    def test_overlapping_concurrent_edits_conflict_and_rebase(self):
        content, result = self.editors(
            use("edit", path="shared.txt", edits=[{"oldText": "alpha", "newText": "Alpha"}])
        )
        self.assertEqual(result["error"]["code"], "file_changed")
        self.assertIn("changes conflict", result["error"]["message"])
        self.assertEqual(content, "ALPHA\nbeta\nGAMMA\n")

    def test_changes_from_outside_the_team_are_never_merged(self):
        def user_edits():
            path = self.workspace / "shared.txt"
            path.write_text(path.read_text() + "user line\n", encoding="utf-8")

        content, result = self.editors(
            use("edit", path="shared.txt", edits=[{"oldText": "beta", "newText": "BETA"}]),
            after_left=user_edits,
        )
        self.assertEqual(result["error"]["code"], "file_changed")
        self.assertEqual(content, "ALPHA\nbeta\nGAMMA\nuser line\n")


class PermissionTests(TeamTestCase):
    """Acceptance test 9: private state, tool scopes and cross-environment isolation."""

    def test_members_get_only_their_declared_tools(self):
        script = TeamScript(
            main=Steps(
                team("send_message", to=["critic"], kind="request", content="try to write"),
                team("wait", timeout=30),
                say("done"),
            ),
            critic=Steps(use("write", path="x.txt", content="nope"), say("I cannot write.")),
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        self.assertEqual(script.tool_names("critic"), {"read", "ls", "team"})
        # An undeclared tool is rejected before execution and never touches the workspace.
        self.assertFalse((self.workspace / "x.txt").exists())
        self.assertIn("Requested a tool that is not available", script.seen("critic"))
        self.assertIn("I cannot write.", script.seen("main"))

    def test_members_cannot_read_private_session_state(self):
        secret = "PRIVATE-MAIN-PROMPT-7731"

        def researcher(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            state_file = str(conversation.team.store.state_file)
            calls = [
                use("read", path=state_file),
                use("read", path="session.json"),
                use("grep", pattern=secret, path="."),
            ]
            if step < len(calls):
                return calls[step]
            return say("probe finished")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["researcher"], kind="request", content="probe"),
                team("wait", timeout=30),
                say("done"),
            ),
            researcher=researcher,
        )
        conversation = self.team_conversation(script)
        self.ask(conversation, f"Use the team. {secret}")
        tool_results = [
            json.loads(message["content"])
            for message in script.requests["researcher"][-1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(tool_results[0]["error"]["code"], "PermissionError")
        self.assertEqual(tool_results[1]["error"]["code"], "PermissionError")
        self.assertEqual(tool_results[2]["data"]["matches"], [])
        self.assertIn(secret, (self.workspace / "session.json").read_text())

    def test_members_cannot_see_other_agents_private_messages(self):
        def critic(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            if step == 0:
                return team("get_state")
            if step == 1:
                return team("send_message", reply_to="m1", content="I read your DM")
            return say("isolation checked")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], content="secret plan for engineer"),
                team("send_message", to=["critic"], kind="request", content="check state"),
                team("wait", timeout=30),
                say("done"),
            ),
            engineer=Steps(say("noted")),
            critic=critic,
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        self.assertNotIn("secret plan", script.seen("critic"))
        results = [
            json.loads(message["content"])
            for message in script.requests["critic"][-1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertNotIn("messages", results[0]["data"])
        self.assertEqual(results[1]["error"]["code"], "not_found")

    def test_delegate_workers_of_a_member_inherit_its_restrictions(self):
        self.define(
            "delegating",
            agents={
                "lead": {
                    "description": "delegates",
                    "tools": ["read", "delegate"],
                    "write_paths": ["out"],
                }
            },
        )
        conversation = self.team_conversation(TeamScript(main=Steps(say("ok"))), "delegating")
        member = conversation.team.member("lead")
        delegate = member.tools.registry["delegate"].handler.__self__
        workers = delegate.runner.execute_workspace.__self__
        self.assertEqual(workers.protected_paths, conversation.team.protected_paths)
        self.assertEqual(workers.write_roots, ((self.workspace / "out").resolve(),))
        with self.assertRaises(PermissionError):
            workers.path(str(self.settings.session_file))

    def test_task_depth_limit_stops_recursive_delegation(self):
        self.define("shallow", limits={"max_task_depth": 1})
        script = TeamScript(
            main=Steps(
                team("create_task", title="Build", assignee="engineer"),
                team("wait", timeout=30),
                say("done"),
            ),
            engineer=Steps(
                team("create_task", title="Sub-build", assignee="researcher"),
                lambda messages, tools: say("depth: " + last_tool(messages)["error"]["code"]),
            ),
        )
        conversation = self.team_conversation(script, "shallow")
        self.ask(conversation)
        self.assertIn("depth: task_depth", script.seen("main"))
        self.assertEqual(list(conversation.team.state.data["tasks"]), ["t1"])

    def test_wait_and_cancel_are_main_only(self):
        script = TeamScript(
            main=Steps(
                team("send_message", to=["critic"], kind="request", content="try"),
                team("wait", timeout=30),
                say("done"),
            ),
            critic=Steps(
                team("wait", timeout=1),
                lambda messages, tools: say("wait: " + last_tool(messages)["error"]["code"]),
            ),
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        self.assertIn("wait: main_only", script.seen("main"))

    def test_sessions_of_different_environments_never_share_state(self):
        script = TeamScript(
            main=Steps(team("share_artifact", title="crew note", content="crew only"), say("ok"))
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        other = self.define("other")
        store = TeamStore(self.settings.environment_state_dir, other, conversation.session_id)
        data, _ = store.open(self.workspace)
        store.close()
        self.assertEqual(data["artifacts"], {})
        # A crew state file copied under another environment is refused, not loaded.
        store.directory.mkdir(parents=True, exist_ok=True)
        (store.directory / "state.json").write_text(
            conversation.team.store.state_file.read_text(), encoding="utf-8"
        )
        with self.assertRaises(RuntimeError):
            store.open(self.workspace)

    def test_one_process_holds_an_environment_session(self):
        script = TeamScript(main=Steps(say("ok")))
        conversation = self.team_conversation(script)
        store = TeamStore(
            self.settings.environment_state_dir,
            conversation.active_environment,
            conversation.session_id,
        )
        with self.assertRaises(RuntimeError):
            store.open(self.workspace)

    def test_new_session_starts_isolated_team_state(self):
        script = TeamScript(
            main=Steps(team("record_decision", title="T", decision="first"), say("ok"))
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        first = conversation.session_id
        conversation.clear()
        self.assertNotEqual(conversation.session_id, first)
        self.assertEqual(conversation.team.state.data["decisions"], [])


class DynamicAgentTests(TeamTestCase):
    def test_spawned_agent_never_exceeds_its_creators_permissions(self):
        self.define("growing", collaboration={"dynamic_agents": True})

        def researcher(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            if step == 0:
                return team(
                    "spawn_agent",
                    name="helper",
                    description="Helps with lookups",
                    tools=["read", "write", "exec"],
                )
            if step == 1:
                return team("send_message", to=["helper"], kind="request", content="look up X")
            return say("asked helper")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["researcher"], kind="request", content="get help"),
                *[
                    lambda messages, tools: (
                        say("finished")
                        if "helper says X=1" in json.dumps(last_tool(messages))
                        else team("wait", timeout=30)
                    )
                ]
                * 4,
            ),
            researcher=Steps(
                researcher,
                researcher,
                researcher,
                lambda messages, tools: say("helper says X=1"),
            ),
            helper=Steps(say("X=1")),
        )
        conversation = self.team_conversation(script, "growing")
        self.assertEqual(self.ask(conversation), "finished")
        self.assertEqual(script.tool_names("helper"), {"read", "team"})
        spawn = json.dumps(script.requests["researcher"][1]["messages"])
        self.assertIn('\\"denied\\": [\\"exec\\", \\"write\\"]', spawn)
        self.assertIn("helper", conversation.team.state.data["agents"])

    def test_spawning_is_disabled_by_default(self):
        script = TeamScript(
            main=Steps(
                team("spawn_agent", name="helper", description="x"),
                lambda messages, tools: say(last_tool(messages)["error"]["code"]),
            )
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "dynamic_agents_disabled")


class TeamMcpTests(TeamTestCase):
    """Acceptance test 9: MCP access is the environment's scope intersected with the agent's."""

    def setUp(self):
        EnvironmentTestCase.setUp(self)
        server = {
            "command": sys.executable,
            "args": [str(ROOT / "tests" / "fixtures_mcp_server.py")],
            "timeout": 10,
        }
        config = self.resource(
            self.root / "mcp.json", json.dumps({"mcpServers": {"alpha": server, "beta": server}})
        )
        self.settings = replace(
            self.settings, max_task_seconds=120, mcp_enabled=True, mcp_config_file=config
        )
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        self.define(
            mcp={"alpha": {"tools": ["echo", "add"]}},
            agents={
                "researcher": {
                    "description": "Uses search tools",
                    "tools": ["read"],
                    "mcp": {"alpha": ["echo"], "beta": "*"},
                },
                "critic": {"description": "No external tools", "tools": ["read"]},
            },
        )

    def test_member_mcp_calls_are_limited_to_the_intersection(self):
        def researcher(messages, tools):
            step = sum(message["role"] == "tool" for message in turn_messages(messages))
            calls = [
                use("mcp", action="call", tool="alpha/echo", arguments={"text": "hi"}),
                use("mcp", action="call", tool="alpha/add", arguments={"left": 1, "right": 2}),
                use("mcp", action="search", query="*"),
                use("mcp", action="call", tool="beta/echo", arguments={"text": "x"}),
            ]
            return calls[step] if step < len(calls) else say("probed")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["researcher", "critic"], kind="request", content="probe"),
                *[
                    lambda messages, tools: (
                        say("done")
                        if sum(item["type"] == "reply" for item in arrived(messages)) >= 2
                        else team("wait", timeout=30)
                    )
                ]
                * 4,
            ),
            researcher=researcher,
            critic=Steps(say("no mcp here")),
        )
        conversation = self.team_conversation(script)
        self.assertEqual(self.ask(conversation), "done")
        results = [
            json.loads(message["content"])
            for message in script.requests["researcher"][-1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["error"]["code"], "permission_denied")
        self.assertEqual([item["path"] for item in results[2]["data"]["matches"]], ["alpha/echo"])
        # beta is outside the environment's selection, so the agent's own grant is void.
        self.assertEqual(results[3]["status"], "error")
        self.assertIn("mcp", script.tool_names("researcher"))
        self.assertNotIn("mcp", script.tool_names("critic"))
        self.assertEqual(set(conversation.mcp_manager.servers), {"alpha"})


class TeamCommandTests(TeamTestCase):
    def test_status_and_log_show_team_state(self):
        script = TeamScript(
            main=Steps(
                team("create_task", title="Collect facts", assignee="researcher"),
                team("wait", timeout=30),
                say("done"),
            ),
            researcher=Steps(say("facts")),
        )
        conversation = self.team_conversation(script)
        self.ask(conversation)
        console = Console()
        console.interactive = False
        output = io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            handle_command(conversation, console, "/env status")
            handle_command(conversation, console, "/env log 50")
        text = output.getvalue()
        self.assertIn("成员 researcher", text)
        self.assertIn("任务 t1 · completed", text)
        self.assertIn("task_created", text)
        self.assertIn("activation_finished", text)
        # Event details read as key=value pairs, not a Python dict.
        self.assertNotIn("{'", text)
        self.assertNotIn(" {}", text)


class TeamProgressTests(TeamTestCase):
    def test_dashboard_events_keep_titles_and_report_counts(self):
        events = []
        script = TeamScript(
            main=Steps(
                team("create_task", title="Build widget", assignee="engineer"),
                team("wait", timeout=30),
                say("done"),
            ),
            engineer=Steps(use("ls", path="."), say("built")),
        )
        script.settings = self.settings
        conversation = Conversation(
            self.settings, script, CodingTools(self.workspace, on_task_update=events.append)
        )
        conversation.environments = EnvironmentStore(
            self.settings, conversation.templates, user_dir=self.user
        )
        self.addCleanup(conversation.close)
        conversation.apply_environment(conversation.environments.resolve("program:crew"))
        self.ask(conversation)
        engineer = [event for event in events if event.get("agent_id") == "engineer"]
        self.assertTrue(engineer)
        self.assertTrue(all(event["title"] == "t1 Build widget" for event in engineer))
        finished = [event for event in engineer if event.get("activity") == "finished"]
        self.assertEqual(finished[-1]["model_requests"], 2)
        self.assertEqual(finished[-1]["tool_calls"], 1)
        detail = Console._tool_detail("team", {"action": "send_message", "to": ["critic", "main"]})
        self.assertEqual(detail, "send_message → critic, main")


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class BackgroundWorkTests(TeamTestCase):
    """Interactive sessions: members keep working between the user's messages."""

    def background_conversation(self, script, events=None, name="crew"):
        script.settings = self.settings
        conversation = Conversation(
            self.settings,
            script,
            CodingTools(
                self.workspace, on_task_update=(events.append if events is not None else None)
            ),
        )
        conversation.environments = EnvironmentStore(
            self.settings, conversation.templates, user_dir=self.user
        )
        self.addCleanup(conversation.close)
        conversation.apply_environment(conversation.environments.resolve(f"program:{name}"))
        conversation.team_background = True
        return conversation

    def background(self, conversation):
        return conversation.team.state.data.get("background") or {}

    def ask(self, conversation, prompt="Do the team task; don't wait for it."):
        # The user hands the work off, so main may answer while members still work.
        return super().ask(conversation, prompt)

    def test_main_waits_for_this_turns_work_unless_the_user_hands_it_off(self):
        release = threading.Event()

        def engineer(messages, tools):
            release.wait(10)
            return say("Widget built.")

        def wait(messages, tools):
            release.set()
            return team("wait", timeout=30)

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build the widget"),
                say("The engineer is building it."),
                # A tool call in between, so the next early answer is held again.
                team("wait", timeout=0),
                say("Still building."),
                wait,
                lambda messages, tools: say(
                    "Result: " + last_tool(messages)["data"]["items"][0]["content"]
                ),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        self.assertEqual(self.ask(conversation, "Build the widget."), "Result: Widget built.")
        self.assertEqual(len(script.requests["main"]), 6)
        self.assertFalse(conversation.team.went_background)
        self.assertIn("did not ask you to leave it running", script.seen("main"))
        self.assertEqual(conversation.last_outcome["status"], "completed")

    def test_answering_twice_in_a_row_hands_the_work_off_instead_of_failing(self):
        # The model fails a turn after a few rejected answers in a row: hold only once.
        release = threading.Event()
        self.addCleanup(release.set)
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build the widget"),
                say("Building."),
                say("Building."),
            ),
            engineer=lambda messages, tools: (release.wait(10), say("Built."))[1],
        )
        conversation = self.background_conversation(script)
        self.assertEqual(self.ask(conversation, "Build the widget."), "Building.")
        self.assertTrue(conversation.team.in_background)

    def test_main_is_not_held_in_the_turns_final_reserve(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def answer_near_the_deadline(messages, tools):
            conversation.team.turn_deadline = time.monotonic() + 1
            return say("Partial result.")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build the widget"),
                answer_near_the_deadline,
                say("Partial result."),
            ),
            engineer=lambda messages, tools: (release.wait(10), say("Built."))[1],
        )
        conversation = self.background_conversation(script)
        self.assertEqual(self.ask(conversation, "Build the widget."), "Partial result.")
        self.assertNotIn("did not ask you to leave it running", script.seen("main"))

    def test_new_work_is_held_even_when_an_earlier_turn_left_work_running(self):
        left, fresh = threading.Event(), threading.Event()
        self.addCleanup(left.set)

        def wait_for_fresh(messages, tools):
            fresh.set()
            return team("wait", timeout=30)

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                say("Building in the background."),
                say("Building in the background."),
                # Next turn: new work for the researcher while the engineer still works.
                team("send_message", to=["researcher"], kind="request", content="Find facts"),
                say("The researcher is on it."),
                wait_for_fresh,
                lambda messages, tools: say(
                    "Facts: " + last_tool(messages)["data"]["items"][0]["content"]
                ),
            ),
            engineer=lambda messages, tools: (left.wait(10), say("Built."))[1],
            researcher=lambda messages, tools: (fresh.wait(10), say("A and B."))[1],
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        self.assertTrue(conversation.team.in_background)
        self.assertEqual(self.ask(conversation, "Find the facts."), "Facts: A and B.")
        self.assertIn("did not ask you to leave it running", script.seen("main"))

    def test_unlimited_background_and_activation_have_no_expiry_timer(self):
        release = threading.Event()
        deadlines = []

        def engineer(messages, tools):
            deadlines.append(script.options()["deadline"])
            release.wait(10)
            return say("Background result ready.")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Continue work"),
                say("Working in the background."),
                say("Working in the background."),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        try:
            self.ask(conversation)
            self.assertTrue(conversation.team.in_background)
            self.assertEqual(deadlines, [float("inf")])
            self.assertIsNone(conversation.team._background_timer)
            self.assertIsNone(self.background(conversation)["seconds"])
            self.assertIn("without a fixed time limit", script.seen("main"))
        finally:
            release.set()
        self.assertTrue(wait_until(lambda: not conversation.team.in_background))
        self.assertIn("Background result ready.", "\n".join(conversation.team.result_lines()))
        self.assertFalse(self.events(conversation, "activation_cancelled"))

    def test_members_keep_working_after_main_answers_and_results_reach_the_next_turn(self):
        release, events = threading.Event(), []

        def engineer(messages, tools):
            release.wait(10)
            return say("Widget built and tested.")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build the widget"),
                say("The engineer is building it in the background."),
                # Nudged once while the engineer still works; main chooses to answer anyway.
                say("The engineer is building it in the background."),
                team("wait", timeout=30),
                lambda messages, tools: say(
                    "Result: " + last_tool(messages)["data"]["items"][0]["content"]
                ),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script, events)
        self.assertEqual(self.ask(conversation), "The engineer is building it in the background.")
        self.assertIn("keep working in the background", script.seen("main"))
        runtime = conversation.team
        self.assertTrue(runtime.in_background)
        self.assertIn("engineer", runtime.activations)
        self.assertEqual(self.background(conversation)["status"], "running")
        self.assertIn("后台：进行中", "\n".join(runtime.result_lines()))

        release.set()
        self.assertTrue(wait_until(lambda: self.background(conversation)["status"] == "finished"))
        self.assertFalse(runtime.in_background)
        lines = "\n".join(runtime.result_lines())
        self.assertIn("engineer ok", lines)
        self.assertIn("Widget built and tested.", lines)
        self.assertIn("发给主代理的 1 项结果", lines)
        notices = [event["notice"] for event in events if event.get("notice")]
        self.assertTrue(any("engineer 完成" in notice for notice in notices))
        self.assertTrue(any("后台工作已完成" in notice for notice in notices))

        # Viewing the results did not consume them: main reads them next turn.
        self.assertEqual(
            self.ask(conversation, "What happened?"), "Result: Widget built and tested."
        )
        self.assertIn("worked in the background", script.seen("main"))

    def test_background_work_pauses_at_the_time_limit_and_resumes_next_turn(self):
        self.define("brief", limits={"background_seconds": 1})
        calls = []

        def engineer(messages, tools):
            calls.append(1)
            if len(calls) == 1:
                script.options()["cancelled"].wait(10)
                raise CompletionError("cancelled", "Model request cancelled")
            return say("Finished after resuming.")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Long job"),
                say("Started."),
                say("Started."),
                team("wait", timeout=30),
                lambda messages, tools: say(last_tool(messages)["data"]["items"][0]["content"]),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script, name="brief")
        self.ask(conversation)
        self.assertTrue(wait_until(lambda: self.background(conversation)["status"] == "expired"))
        self.assertTrue(wait_until(lambda: not conversation.team.activations))
        mailbox = conversation.team.state.data["mailboxes"]["engineer"]
        self.assertTrue(mailbox and mailbox[0]["redelivered"])
        self.assertIn("1 秒时限后暂停", "\n".join(conversation.team.result_lines()))
        self.assertEqual(self.ask(conversation, "continue"), "Finished after resuming.")

    def test_next_turn_takes_over_work_still_running_in_the_background(self):
        release = threading.Event()

        def engineer(messages, tools):
            release.wait(10)
            return say("Done while you were typing.")

        def release_then_wait(messages, tools):
            release.set()
            return team("wait", timeout=30)

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                say("Building."),
                say("Building."),
                release_then_wait,
                lambda messages, tools: say(last_tool(messages)["data"]["items"][0]["content"]),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        self.assertTrue(conversation.team.in_background)
        self.assertEqual(self.ask(conversation, "Any news?"), "Done while you were typing.")
        self.assertEqual(self.background(conversation)["status"], "resumed")
        self.assertEqual(len(script.requests["engineer"]), 1)

    def test_interrupted_turns_still_pause_members(self):
        started = threading.Event()

        def engineer(messages, tools):
            started.set()
            script.options()["cancelled"].wait(10)
            raise CompletionError("cancelled", "Model request cancelled")

        def interrupt(messages, tools):
            started.wait(10)
            raise KeyboardInterrupt

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                interrupt,
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        with self.assertRaises(KeyboardInterrupt):
            self.ask(conversation)
        self.assertFalse(conversation.team.in_background)
        self.assertFalse(conversation.team.activations)
        self.assertTrue(self.events(conversation, "activation_cancelled"))
        self.assertNotIn("background", conversation.team.state.data)

    def test_a_background_period_cut_short_by_the_process_is_recovered(self):
        release = threading.Event()
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                say("Building."),
                say("Building."),
            ),
            engineer=lambda messages, tools: (release.wait(10), say("built"))[1],
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        session_id = conversation.session_id
        runtime = conversation.team
        # Simulate a crash: the state file still says the background period runs.
        with runtime.lock:
            runtime._background = None
            runtime.accepting = False
            runtime._persist()
        release.set()
        self.assertTrue(wait_until(lambda: not runtime.activations))
        with runtime.lock:
            runtime.state.data["background"]["status"] = "running"
            runtime._persist()
        runtime.store.close()
        runtime.store = None
        conversation.team = None
        conversation.close()

        restarted = Conversation(self.settings, TeamScript(), CodingTools(self.workspace))
        restarted.environments = EnvironmentStore(
            self.settings, restarted.templates, user_dir=self.user
        )
        self.addCleanup(restarted.close)
        restarted.resume(session_id)
        self.assertEqual(restarted.team.state.data["background"]["status"], "stopped")
        self.assertIn("后台工作", restarted.recovery_notice or "")

    def test_results_command_shows_background_work_without_consuming_it(self):
        release = threading.Event()
        script = TeamScript(
            main=Steps(
                team("create_task", title="Collect facts", assignee="researcher"),
                say("Researcher is on it."),
                say("Researcher is on it."),
            ),
            researcher=lambda messages, tools: (release.wait(10), say("facts: A, B"))[1],
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        self.assertTrue(conversation.team.in_background)
        release.set()
        self.assertTrue(wait_until(lambda: not conversation.team.in_background))
        console = Console()
        console.interactive = False
        output = io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            handle_command(conversation, console, "/env results")
            handle_command(conversation, console, "/env status")
        text = output.getvalue()
        self.assertIn("researcher ok · t1 Collect facts — facts: A, B", text)
        self.assertIn("任务 completed · researcher · Collect facts：facts: A, B", text)
        self.assertIn("团队已空闲", text)
        self.assertEqual(len(conversation.team.state.mailbox("main")), 1)

    def test_main_may_hand_off_a_change_whose_edits_are_still_in_progress(self):
        release = threading.Event()
        (self.workspace / "app.py").write_text("answer = 41\n", encoding="utf-8")

        def engineer(messages, tools):
            if not turn_messages(messages)[1:]:
                return use("write", path="app.py", content="answer = 42\n")
            release.wait(10)
            return say("Fixed and tested.")

        def write_started(messages, tools):
            # Answer only once the engineer's first edit has landed.
            wait_until(lambda: "42" in (self.workspace / "app.py").read_text())
            return say("已安排 engineer 修复，它在后台继续。")

        script = TeamScript(
            main=Steps(
                team("create_task", title="Fix app.py", assignee="engineer"),
                write_started,
                say("已安排 engineer 修复，它在后台继续。"),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        conversation.ask("修复 app.py：交给 engineer，不要等待它完成")
        self.assertEqual(conversation.last_outcome["status"], "background")
        self.assertTrue(conversation.team.in_background)
        release.set()
        self.assertTrue(wait_until(lambda: not conversation.team.in_background))

    def test_work_that_may_continue_in_the_background_outlives_the_turn_deadline(self):
        release, deadlines, close = threading.Event(), [], []

        def engineer(messages, tools):
            deadlines.append(script.options()["deadline"])
            release.wait(10)
            return say("Done after the turn deadline.")

        def send_with_a_close_deadline(messages, tools):
            # A turn deadline closer than the activation timeout no longer bounds the work.
            close.append(time.monotonic() + 0.2)
            conversation.team.turn_deadline = close[0]
            return team("send_message", to=["engineer"], kind="request", content="Build")

        script = TeamScript(
            main=Steps(send_with_a_close_deadline, say("Building."), say("Building.")),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        self.assertTrue(wait_until(lambda: time.monotonic() > close[0] + 0.1))
        release.set()
        self.assertTrue(wait_until(lambda: not conversation.team.in_background))
        self.assertGreater(deadlines[0], close[0] + 5)
        self.assertFalse(self.events(conversation, "activation_cancelled"))
        self.assertEqual(len(script.requests["engineer"]), 1)
        lines = "\n".join(conversation.team.result_lines())
        self.assertIn("Done after the turn deadline.", lines)
        self.assertNotIn("no report", lines)

    def test_work_finishing_while_the_next_turn_starts_belongs_to_that_turn(self):
        release = threading.Event()

        def engineer(messages, tools):
            release.wait(10)
            return say("Done during compaction.")

        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                say("Building."),
                say("Building."),
                team("wait", timeout=30),
                lambda messages, tools: say(last_tool(messages)["data"]["items"][0]["content"]),
            ),
            engineer=engineer,
        )
        conversation = self.background_conversation(script)
        self.ask(conversation)
        self.assertTrue(conversation.team.in_background)
        compact = conversation.compact

        def compact_while_the_engineer_finishes(*args, **kwargs):
            release.set()
            self.assertTrue(wait_until(lambda: not conversation.team.activations))
            return compact(*args, **kwargs)

        conversation.compact = compact_while_the_engineer_finishes
        self.assertEqual(self.ask(conversation, "Any news?"), "Done during compaction.")
        # The turn took the work over before its note, so it did not finish unreported in
        # the background between the note and the turn.
        self.assertEqual(self.background(conversation)["status"], "resumed")
        self.assertIn("that work continues in this turn", script.seen("main"))
        self.assertIn("Members working now: engineer", script.seen("main"))

    def test_members_of_turns_that_cannot_continue_stay_bound_to_the_turn_deadline(self):
        self.define("bound", limits={"background_seconds": 0})
        deadlines, close = [], []

        def send_with_a_close_deadline(messages, tools):
            close.append(time.monotonic() + 30)
            conversation.team.turn_deadline = close[0]
            return team("send_message", to=["engineer"], kind="request", content="Build")

        def engineer(messages, tools):
            deadlines.append(script.options()["deadline"])
            return say("Built.")

        script = TeamScript(
            main=Steps(send_with_a_close_deadline, team("wait", timeout=30), say("Done.")),
            engineer=engineer,
        )
        conversation = self.background_conversation(script, name="bound")
        self.ask(conversation)
        self.assertLessEqual(deadlines[0], close[0])

    def test_limits_read_the_same_to_main_and_to_the_user(self):
        for seconds, english, chinese in (
            (None, "unlimited", "不限时"),
            (60, "1 minute", "1 分钟"),
            (1800, "30 minutes", "30 分钟"),
            (90, "90 seconds", "90 秒"),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(duration_text(seconds), english)
                self.assertEqual(duration_text(seconds, chinese=True), chinese)

    def test_background_can_be_disabled_per_environment(self):
        self.define("bound", limits={"background_seconds": 0})
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Build"),
                say("Paused."),
                say("Paused."),
            ),
            engineer=lambda messages, tools: (
                script.options()["cancelled"].wait(10),
                say("late"),
            )[1],
        )
        conversation = self.background_conversation(script, name="bound")
        self.ask(conversation)
        self.assertFalse(conversation.team.in_background)
        self.assertNotIn("keep working in the background", script.seen("main"))


class MemberSelfCheckTests(TeamTestCase):
    """Members follow the environment's rules and check their own code changes."""

    CHECK = f'"{sys.executable}" -c "import app; assert app.answer == 42"'

    def delegate(self, *engineer):
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Fix app.py"),
                team("wait", timeout=30),
                lambda messages, tools: say(last_tool(messages)["data"]["items"][0]["content"]),
            ),
            engineer=Steps(*engineer),
        )
        conversation = self.team_conversation(script)
        return script, self.ask(conversation, "Fix app.py with the team")

    def test_members_start_from_the_environments_working_rules(self):
        script, _ = self.delegate(
            use("write", path="notes.txt", content="not code\n"), say("Wrote notes.")
        )
        member = script.requests["engineer"][0]["messages"][0]["content"]
        rules = self.store.resolve("program:crew").prompt_text
        self.assertTrue(member.startswith(rules))
        self.assertIn("You are a member, not main", member)
        self.assertIn('You are "engineer"', member)
        self.assertIn("run a relevant standalone check", member)
        # Main's prompt starts the same way up to the tool list, so the prefix is shared.
        main = script.requests["main"][0]["messages"][0]["content"]
        self.assertEqual(main.split("Available tools:")[0], member.split("Available tools:")[0])
        # A document edit needs no check.
        self.assertEqual(len(script.requests["engineer"]), 2)

    def test_a_member_is_asked_once_to_check_its_code_change(self):
        script, answer = self.delegate(
            use("write", path="app.py", content="answer = 42\n"),
            say("Fixed."),
            use("exec", command=self.CHECK),
            say("Fixed app.py; the assertion passed."),
        )
        self.assertEqual(answer, "Fixed app.py; the assertion passed.")
        self.assertIn("the change is not verified yet", script.seen("engineer"))
        self.assertEqual(len(script.requests["engineer"]), 4)

    def test_a_member_that_checks_its_change_is_not_asked_again(self):
        script, answer = self.delegate(
            use("write", path="app.py", content="answer = 42\n"),
            use("exec", command=self.CHECK),
            say("Fixed and checked."),
        )
        self.assertEqual(answer, "Fixed and checked.")
        self.assertNotIn("not verified yet", script.seen("engineer"))

    def test_an_unchecked_change_is_reported_as_unchecked(self):
        script, answer = self.delegate(
            use("write", path="app.py", content="answer = 42\n"),
            say("Fixed."),
            say("Fixed; I did not run anything."),
        )
        self.assertTrue(answer.startswith("Fixed; I did not run anything."))
        self.assertIn("engineer changed code", answer)
        self.assertIn("no passing standalone check followed its last edit", answer)
        self.assertEqual(len(script.requests["engineer"]), 3)

    def test_a_failing_check_without_an_edit_is_no_unchecked_change(self):
        # A tester whose check finds the bug changed nothing.
        script, answer = self.delegate(
            use("exec", command=f'"{sys.executable}" -c "assert 1 == 2"'),
            say("Found it: the assertion fails."),
        )
        self.assertEqual(answer, "Found it: the assertion fails.")
        self.assertEqual(len(script.requests["engineer"]), 2)

    def test_an_earlier_failing_check_is_named_as_the_reason(self):
        _, answer = self.delegate(
            use("exec", command=f'"{sys.executable}" -c "import app; assert app.answer == 42"'),
            use("write", path="app.py", content="answer = 42\n"),
            use("exec", command=f'"{sys.executable}" -c "import app; assert app.answer"'),
            say("Fixed."),
            say("Fixed; the second check passed."),
        )
        self.assertIn("a check it ran failed and was not rerun successfully", answer)
        self.assertNotIn("no passing standalone check", answer)

    def test_a_finished_answer_survives_a_failed_follow_up_request(self):
        self.define("tight", limits={"token_budget_per_agent": 1000})
        small = {"prompt_tokens": 100, "completion_tokens": 10}
        usage = {"prompt_tokens": 2000, "completion_tokens": 10}
        script = TeamScript(
            main=Steps(
                team("send_message", to=["engineer"], kind="request", content="Fix app.py"),
                team("wait", timeout=30),
                lambda messages, tools: say(last_tool(messages)["data"]["items"][0]["content"]),
            ),
            engineer=Steps(
                {
                    **use("write", path="app.py", content="answer = 42\n"),
                    LOCAL_METADATA: {"usage": small},
                },
                # The nudge's follow-up request finds the budget spent.
                {**say("Fixed app.py."), LOCAL_METADATA: {"usage": usage}},
            ),
        )
        answer = self.ask(self.team_conversation(script, "tight"), "Fix app.py with the team")
        self.assertTrue(answer.startswith("Fixed app.py."), answer)
        self.assertIn("no passing standalone check followed its last edit", answer)
        self.assertEqual(len(script.requests["engineer"]), 2)

    def test_members_without_exec_and_data_files_need_no_check(self):
        self.define(
            "writers",
            agents={
                **self.members,
                "writer": {
                    "description": "Writes pages.",
                    "instructions": "You write.",
                    "tools": ["read", "write", "edit"],
                },
            },
        )
        script = TeamScript(
            main=Steps(
                team("send_message", to=["writer", "engineer"], kind="request", content="Go"),
                team("wait", timeout=30),
                lambda messages, tools: (
                    say("done")
                    if sum(item["type"] == "reply" for item in arrived(messages)) >= 2
                    else team("wait", timeout=30)
                ),
                say("done"),
            ),
            writer=Steps(
                use("write", path="report.html", content="<p>report</p>\n"), say("Wrote it.")
            ),
            engineer=Steps(
                use("write", path="results.json", content="{}\n"), say("Saved results.")
            ),
        )
        self.ask(self.team_conversation(script, "writers"))
        self.assertNotIn("[runtime]", script.seen("main"))
        self.assertEqual(len(script.requests["writer"]), 2)
        self.assertEqual(len(script.requests["engineer"]), 2)
        writer = script.requests["writer"][0]["messages"][0]["content"]
        self.assertNotIn("run a relevant standalone check", writer)


class AcceptanceAttributionTests(TeamTestCase):
    def test_reporting_a_members_limitation_does_not_block_mains_answer(self):
        script = TeamScript(
            main=Steps(
                team("create_task", title="Review", assignee="critic"),
                team("wait", timeout=30),
                say("已修复并验证。critic 只有只读工具，无法执行测试，我重新运行了测试。"),
            ),
            critic=Steps(say("Looks right; I could not run the tests.")),
        )
        conversation = self.team_conversation(script)
        conversation.ask("检查团队结果")
        self.assertEqual(conversation.last_outcome["status"], "completed")
