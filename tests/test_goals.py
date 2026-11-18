import json
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from command_ui import use_goal
from commands import CommandName, parse_command
from conversation import Conversation
from deepseek_client import CompletionError
from goal_runtime import GoalRuntime, prepare_goal
from goals import GoalCommand, GoalState, parse_goal
from main import handle_command
from request_budget import budget_scope, reserve_request
from task_acceptance import TaskContract
from tests.support import make_settings
from tests.team_support import Steps, TeamScript, say, team, use
from tests.test_conversation import FakeClient
from tests.test_environments import EnvironmentTestCase
from tests.test_team_runtime import TeamTestCase
from tests.test_tools import python_command
from tool_runtime import ExecutionContext
from tools import CodingTools
from ui import Console


def assessment(
    status="completed",
    *,
    summary="已核对成果。",
    next_step="",
    criteria_met=(),
    evidence=("已检查交付物",),
):
    return say(
        json.dumps(
            {
                "status": status,
                "summary": summary,
                "next_step": next_step,
                "criteria_met": list(criteria_met),
                "evidence": list(evidence),
            },
            ensure_ascii=False,
        )
    )


class GoalCommandTests(unittest.TestCase):
    def test_goal_is_registered_and_preserves_quoted_criteria(self):
        command = parse_command('/goal "完成一份报告" --criterion "包含引用" --max-rounds 3')
        self.assertIs(command.name, CommandName.GOAL)
        parsed = parse_goal(list(command.arguments))
        self.assertEqual(parsed.text, "完成一份报告")
        self.assertEqual(parsed.criteria, ("包含引用",))
        self.assertEqual(parsed.max_rounds, 3)
        self.assertIsNotNone(parse_command('/goal "unfinished').error)

    def test_reserved_words_can_be_used_as_objectives_with_start(self):
        self.assertEqual(parse_goal(["start", "status"]).text, "status")
        self.assertEqual(parse_goal(["--", "--an-objective"]).text, "--an-objective")
        self.assertEqual(parse_goal([]).action, "status")

    def test_invalid_flags_budgets_and_empty_objectives_are_rejected(self):
        for args in (
            ["start"],
            [""],
            ["task", "--unknown"],
            ["task", "--max-rounds"],
            ["task", "--max-rounds", "0"],
            ["task", "--max-seconds", "-1"],
            ["task", "--max-rounds", "2.5"],
            ["task", "--max-rounds", "1000001"],
            ["task", "--max-rounds", "2", "--max-rounds", "3"],
            ["task", "--criterion", " "],
            ["resume", "--criterion", "new"],
            ["status", "extra"],
            ["pause", "extra"],
            ["cancel", "extra"],
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                parse_goal(args)

    def test_saved_goal_validation_rejects_bad_types_and_unknown_status(self):
        for changes in (
            {"rounds": True},
            {"status": "unknown"},
            {"max_seconds": 0},
            {"criteria": [None]},
            {"elapsed_ms": -1},
            {"unexpected": "value"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GoalState.from_dict({**GoalState("task").to_dict(), **changes})


class GoalRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            workspace=self.workspace,
            session_file=self.workspace / "state" / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=2,
            mcp_enabled=False,
            websearch_enabled=False,
        )

    def conversation(self, *responses, tools=False):
        conversation = Conversation(
            self.settings,
            FakeClient(*responses),
            CodingTools(self.workspace) if tools else None,
        )
        self.addCleanup(conversation.close)
        return conversation

    def start(self, conversation, text="撰写一份报告", **options):
        return prepare_goal(conversation, GoalCommand("start", text, **options))

    def test_token_budget_is_restored_without_dispatching_an_impossible_request(self):
        self.settings = replace(self.settings, goal_token_budget=100)
        first = self.conversation()
        goal = self.start(first)
        goal.token_budget = {"limit": 100, "measured": 100, "estimated": 0, "unknown": 0, "requests": 1}
        first.save()
        first.close()
        resumed = self.conversation()
        result = GoalRuntime(resumed).run()
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.token_budget["measured"], 100)
        self.assertEqual(result.token_budget["requests"], 1)
        self.assertEqual(resumed.client.requests, [])
        self.assertIn("已报告 100", "\n".join(result.status_lines()))

    def test_continues_until_entire_goal_and_all_criteria_are_met(self):
        conversation = self.conversation(
            say("报告初稿"),
            assessment("continue", next_step="补充引用", criteria_met=[False]),
            say("报告已补充引用"),
            assessment(criteria_met=[True]),
        )
        self.start(conversation, criteria=("包含引用",))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "completed")
        self.assertEqual(goal.rounds, 2)
        self.assertEqual(goal.criteria_met, [True])
        self.assertEqual(conversation.sessions.read()["goal"]["status"], "completed")
        requests = conversation.client.requests
        self.assertIn("补充引用", requests[2]["messages"][-2]["content"])
        self.assertIn("撰写一份报告", requests[2]["messages"][-2]["content"])
        self.assertEqual(len(requests), 4)
        self.assertIsNone(requests[1]["tools"])
        self.assertIsNone(requests[3]["tools"])

    def test_late_goal_usage_survives_resume_and_is_persisted(self):
        self.settings = replace(self.settings, goal_token_budget=100)
        conversation = self.conversation()
        goal = self.start(conversation)
        ledger = conversation._restore_goal_budget(goal)
        with budget_scope(ledger), reserve_request(10, 50) as reservation:
            reservation.dispatched = True
            conversation.save()
            prepare_goal(conversation, GoalCommand("resume"))
            self.assertIs(conversation._restore_goal_budget(conversation.goal), ledger)
            self.assertEqual(ledger.remaining(), 40)
            reservation.measured = 60
        conversation._save_background_budget((ledger,))
        self.assertEqual(conversation.sessions.read()["goal"]["token_budget"]["measured"], 60)
        conversation.close()
        restored = self.conversation()
        self.assertEqual(restored._restore_goal_budget(restored.goal).remaining(), 40)

    def test_late_goal_only_usage_updates_its_archived_session(self):
        self.settings = replace(self.settings, goal_token_budget=100)
        conversation = self.conversation()
        goal = self.start(conversation)
        old_session = conversation.session_id
        ledger = conversation._restore_goal_budget(goal)
        with budget_scope(ledger), reserve_request(10, 50) as reservation:
            reservation.dispatched = True
            conversation.clear()
            self.start(conversation, "new goal")
            reservation.measured = 60
        conversation._save_background_budget((ledger,))
        archived = conversation.sessions.read(conversation.sessions.snapshot_path(old_session))
        self.assertEqual(archived["goal"]["token_budget"]["measured"], 60)
        self.assertIsNone(conversation.sessions.read()["goal"].get("token_budget"))

    def test_code_goal_runs_real_tools_and_verifies_before_completion(self):
        conversation = self.conversation(
            use("write", path="feature.py", content="answer = 42\n"),
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已实现并验证。"),
            assessment(criteria_met=[True]),
            tools=True,
        )
        self.start(conversation, "实现 feature.py", criteria=("answer 等于 42",))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "completed")
        self.assertEqual((self.workspace / "feature.py").read_text(), "answer = 42\n")
        self.assertEqual(conversation.last_outcome["kind"], "change")
        self.assertEqual(
            conversation.last_outcome["code_revision"],
            conversation.last_outcome["verified_revision"],
        )
        self.assertTrue(
            any(item.get("verification") for item in conversation.last_outcome["evidence"])
        )

    def test_failed_code_check_must_be_repaired_before_final_answer_is_accepted(self):
        check = python_command(
            "namespace = {}; exec(open('feature.py').read(), namespace); assert namespace['answer'] == 42"
        )
        conversation = self.conversation(
            use("write", path="feature.py", content="answer = 41\n"),
            use("exec", command=check),
            say("已完成。"),
            use("edit", path="feature.py", edits=[{"oldText": "41", "newText": "42"}]),
            use("exec", command=check),
            say("已修复并通过验证。"),
            assessment(),
            tools=True,
        )
        self.start(conversation, "实现 feature.py 并验证")
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertIn("42", (self.workspace / "feature.py").read_text())
        self.assertEqual(len(conversation.client.requests), 7)

    def test_goal_retries_an_incomplete_round_before_accepting_completion(self):
        conversation = self.conversation(
            use("write", path="feature.py", content="answer = 42\n"),
            say("无法验证。"),
            say("无法执行测试。"),
            say("无法验证。"),
            assessment(),
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已通过验证。"),
            assessment(),
            tools=True,
        )
        self.start(conversation, "分析并修复 feature.py")
        observed = []
        runtime = GoalRuntime(
            conversation,
            on_review=lambda _goal: observed.append(conversation.last_outcome["status"]),
        )
        self.assertEqual(runtime.run().status, "completed")
        self.assertEqual(observed, ["incomplete", "completed"])

    def test_analysis_goal_round_cannot_accept_an_unverified_source_change(self):
        conversation = self.conversation(
            use("write", path="feature.py", content="answer = 42\n"),
            say("已完成。"),
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已修复并通过验证。"),
            assessment(),
            tools=True,
        )
        self.start(conversation, "先分析问题再修复 feature.py")
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertEqual(conversation.last_outcome["kind"], "answer")
        self.assertEqual(len(conversation.client.requests), 5)

    def test_later_goal_round_preserves_source_outside_the_workspace(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        external = Path(temporary.name).resolve() / "feature.py"
        conversation = self.conversation(
            use("write", path=str(external), content="answer = 42\n"),
            use("exec", cwd=str(external.parent), command=python_command(
                "from feature import answer; assert answer == 42"
            )),
            say("外部模块已验证。"),
            assessment("continue", next_step="补全本地模块"),
            use("write", path="local.py", content="answer = 42\n"),
            use("exec", command=python_command("from local import answer; assert answer == 42")),
            say("本地模块已验证。"),
            assessment(),
            tools=True,
        )
        self.start(conversation, "实现两个模块", max_rounds=2)

        def regress_after_second_check(goal):
            if goal.rounds == 2:
                external.write_text("answer = 0\n")

        goal = GoalRuntime(conversation, on_review=regress_after_second_check).run()
        self.assertEqual(goal.status, "budget_exhausted")
        self.assertTrue(goal.verification_required)
        self.assertIn(str(external), goal.source_scope["paths"])
        self.assertIn("源码在验证后发生变化", goal.next_step)

    def test_resumed_goal_requires_check_for_code_written_in_interrupted_round(self):
        conversation = self.conversation(use("write", path="feature.py", content="answer = 42\n"), tools=True)

        def interrupt(*_arguments):
            raise KeyboardInterrupt

        def ask_and_interrupt(prompt, **kwargs):
            return conversation.ask(prompt, on_tool_result=interrupt, **kwargs)

        self.start(conversation, "先分析再修复 feature.py")
        with self.assertRaises(KeyboardInterrupt):
            GoalRuntime(conversation, ask_turn=ask_and_interrupt).run()
        self.assertTrue(conversation.sessions.read()["goal"]["verification_required"])
        restored = self.conversation(
            say("已完成。"),
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已验证。"),
            assessment(),
            tools=True,
        )
        self.assertEqual(GoalRuntime(restored).run().status, "completed")
        self.assertEqual(len(restored.client.requests), 4)
        self.assertFalse(restored.goal.verification_required)

    def test_resumed_review_does_not_accept_a_changed_source_revision(self):
        def interrupt(_goal):
            raise KeyboardInterrupt

        conversation = self.conversation(
            use("write", path="feature.py", content="answer = 42\n"),
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已验证。"),
            tools=True,
        )
        self.start(conversation, "分析并修复 feature.py")
        with self.assertRaises(KeyboardInterrupt):
            GoalRuntime(conversation, on_review=interrupt).run()
        self.assertEqual(conversation.goal.phase, "review")
        self.assertIsNotNone(conversation.sessions.read()["goal"]["verified_source"])
        (self.workspace / "feature.py").write_text("answer = 43\n")
        restored = self.conversation(
            say("已完成。"),
            use("exec", command=python_command("from feature import answer; assert answer == 43")),
            say("已验证新版本。"),
            assessment(),
            tools=True,
        )
        self.assertEqual(GoalRuntime(restored).run().status, "completed")
        self.assertEqual(restored.goal.rounds, 2)
        self.assertEqual(len(restored.client.requests), 4)

    def test_interrupted_goal_stops_only_its_owned_background_process(self):
        feature = self.workspace / "feature.py"
        feature.write_text("answer = 42\n")
        command = python_command(
            "input(); from pathlib import Path; Path('feature.py').write_text('answer = 0\\n')"
        )
        conversation = self.conversation(use("exec", command=command, background=True), tools=True)
        other = conversation.tools.processes.start(command, self.workspace, 60)
        sessions = []

        def interrupt(_name, result):
            sessions.append(result["data"]["session_id"])
            raise KeyboardInterrupt

        def ask_and_interrupt(prompt, **kwargs):
            return conversation.ask(prompt, on_tool_result=interrupt, **kwargs)

        self.start(conversation, "先分析再修复 feature.py")
        with self.assertRaises(KeyboardInterrupt):
            GoalRuntime(conversation, ask_turn=ask_and_interrupt).run()
        self.assertTrue(conversation.goal.verification_required)
        self.assertTrue(conversation.tools.processes.lookup(sessions[0]).done.is_set())
        self.assertFalse(conversation.tools.processes.lookup(other.data["session_id"]).done.is_set())
        conversation.client.responses.extend([
            use("exec", command=python_command("from feature import answer; assert answer == 42")),
            say("已验证。"),
            assessment(),
        ])
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")

    def test_round_budget_exhaustion_and_explicit_extension(self):
        conversation = self.conversation(
            say("部分成果"),
            assessment("continue", next_step="完成剩余内容"),
            say("完整成果"),
            assessment(),
        )
        self.start(conversation, max_rounds=1)
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "budget_exhausted")
        before = goal.to_dict()
        with self.assertRaises(ValueError):
            prepare_goal(conversation, GoalCommand("resume", max_seconds=8000))
        self.assertEqual(conversation.goal.to_dict(), before)
        prepare_goal(conversation, GoalCommand("resume", max_rounds=2))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "completed")
        self.assertEqual(goal.rounds, 2)

    def test_shared_time_budget_is_passed_to_work_and_review(self):
        conversation = self.conversation(say("成果"), assessment())
        goal = self.start(conversation, max_seconds=40)
        started = time.monotonic()
        GoalRuntime(conversation).run()
        self.assertGreaterEqual(goal.elapsed_ms, 0)
        work, review = conversation.client.requests
        self.assertAlmostEqual(work["deadline"], review["deadline"], places=3)
        self.assertLessEqual(work["deadline"], started + 41)

    def test_time_exhaustion_stops_without_requesting_another_round(self):
        conversation = self.conversation()
        self.start(conversation, max_seconds=1)
        now = [100.0]

        def work(*args, **kwargs):
            now[0] += 2
            conversation.last_outcome = {"status": "completed"}
            conversation.messages.append(say("部分成果"))

        with patch("goal_runtime.time.monotonic", side_effect=lambda: now[0]):
            goal = GoalRuntime(conversation, ask_turn=work).run()
        self.assertEqual(goal.status, "budget_exhausted")
        self.assertEqual(goal.rounds, 1)
        self.assertEqual(goal.elapsed_ms, 2000)
        self.assertEqual(conversation.client.requests, [])

    def test_waiting_for_user_resumes_with_their_answer(self):
        conversation = self.conversation(
            say("需要确定报告的读者。"),
            assessment("waiting_user", summary="需要确定读者", next_step="请说明目标读者"),
            say("已完成面向开发者的报告"),
            assessment(),
        )
        self.start(conversation)
        self.assertEqual(GoalRuntime(conversation).run().status, "waiting_user")
        prepare_goal(conversation, GoalCommand("resume", "目标读者是开发者"))
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertIn(
            "目标读者是开发者", conversation.client.requests[2]["messages"][-2]["content"]
        )
        self.assertEqual(conversation.goal.notes, ["目标读者是开发者"])

    def test_three_identical_rounds_are_blocked_instead_of_looping_forever(self):
        responses = []
        for _ in range(3):
            responses.extend([say("同样的结果"), assessment("continue", next_step="继续调查")])
        conversation = self.conversation(*responses)
        self.start(conversation)
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "blocked")
        self.assertEqual(goal.rounds, 3)
        self.assertIn("连续三轮", goal.reason)

    def test_same_summary_with_different_tool_work_is_not_blocked(self):
        responses = []
        for content in ("第一部分", "第一部分、第二部分", "完整内容"):
            responses.extend(
                [
                    use("write", path="report.md", content=content),
                    say("本阶段已完成"),
                    assessment("continue", next_step="继续补充报告"),
                ]
            )
        conversation = self.conversation(*responses, tools=True)
        self.start(conversation, max_rounds=3)
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "budget_exhausted")
        self.assertEqual((self.workspace / "report.md").read_text(), "完整内容")
        self.assertEqual(goal.repeated_rounds, 1)

    def test_interrupt_during_work_pauses_and_preserves_checkpoint(self):
        conversation = self.conversation()
        self.start(conversation)
        with self.assertRaises(KeyboardInterrupt):
            GoalRuntime(conversation, ask_turn=Mock(side_effect=KeyboardInterrupt)).run()
        saved = conversation.sessions.read()
        self.assertEqual(saved["goal"]["status"], "paused")
        self.assertEqual(saved["goal"]["rounds"], 1)
        self.assertEqual(saved["goal"]["phase"], "work")
        self.assertEqual(len(conversation.sessions.list()), 1)

    def test_interrupt_during_review_resumes_review_without_repeating_work(self):
        conversation = self.conversation(say("报告已生成"))
        self.start(conversation, max_rounds=1)
        with (
            patch.object(GoalRuntime, "_assessment", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            GoalRuntime(conversation).run()
        self.assertEqual(conversation.goal.phase, "review")
        restored = self.conversation(assessment())
        prepare_goal(restored, GoalCommand("resume"))
        goal = GoalRuntime(restored).run()
        self.assertEqual(goal.status, "completed")
        self.assertEqual(goal.rounds, 1)
        self.assertEqual(len(restored.client.requests), 1)

    def test_review_resume_after_another_task_rechecks_goal_instead_of_using_unrelated_outcome(
        self,
    ):
        conversation = self.conversation(
            say("报告初稿"), say("另一个问题的回答"), say("报告已核对"), assessment()
        )
        self.start(conversation)
        with (
            patch.object(GoalRuntime, "_assessment", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            GoalRuntime(conversation).run()
        conversation.ask("另一个问题")
        prepare_goal(conversation, GoalCommand("resume"))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "completed")
        self.assertEqual(goal.rounds, 2)
        self.assertIn("重新核对目标", conversation.client.requests[2]["messages"][-2]["content"])

    def test_invalid_review_cannot_complete_goal_and_can_be_retried(self):
        invalid = assessment(criteria_met=[False])
        conversation = self.conversation(say("初稿"), invalid, invalid, invalid)
        self.start(conversation, criteria=("必须有引用",))
        with self.assertRaises(CompletionError):
            GoalRuntime(conversation).run()
        self.assertEqual(conversation.goal.status, "paused")
        self.assertEqual(conversation.goal.phase, "review")
        self.assertEqual(conversation.goal.rounds, 1)

    def test_reported_success_cannot_override_blocked_task_acceptance(self):
        conversation = self.conversation(
            say("我无法继续执行，需要你提供材料。"),
            assessment(),
        )
        self.start(conversation, max_rounds=1)
        # Give the turn a real tool result so its blocker is a work-task outcome.
        conversation.last_outcome = {"status": "blocked"}

        def work(*args, **kwargs):
            conversation.messages.append(say("缺少材料"))
            conversation.last_outcome = {"status": "blocked"}

        with patch.object(
            GoalRuntime, "_assessment", return_value=json.loads(assessment()["content"])
        ):
            goal = GoalRuntime(conversation, ask_turn=work).run()
        self.assertEqual(goal.status, "budget_exhausted")
        self.assertNotEqual(goal.status, "completed")

    def test_incomplete_plan_prevents_goal_completion(self):
        conversation = self.conversation(tools=True)
        self.start(conversation, max_rounds=1)
        conversation.tools.task_plan = {
            "title": "报告",
            "steps": [{"step": "核对", "status": "pending"}],
        }

        def work(*args, **kwargs):
            conversation.messages.append(say("部分成果"))
            conversation.last_outcome = {"status": "completed"}

        with patch.object(
            GoalRuntime, "_assessment", return_value=json.loads(assessment()["content"])
        ):
            self.assertEqual(
                GoalRuntime(conversation, ask_turn=work).run().status, "budget_exhausted"
            )

    def test_restart_restores_active_goal_as_paused_and_preserves_objective(self):
        conversation = self.conversation()
        goal = self.start(conversation)
        goal.status = "active"
        conversation.save()
        restored = self.conversation()
        self.assertEqual(restored.goal.status, "paused")
        self.assertEqual(restored.goal.objective, goal.objective)
        self.assertIn("/goal resume", restored.goal.reason)
        self.assertEqual(restored.client.requests, [])

    def test_new_and_messages_only_resume_do_not_inherit_goal(self):
        conversation = self.conversation(say("报告"), assessment())
        self.start(conversation)
        GoalRuntime(conversation).run()
        old_id = conversation.session_id
        conversation.clear()
        self.assertIsNone(conversation.goal)
        conversation.resume(old_id, messages_only=True)
        self.assertIsNone(conversation.goal)
        conversation.resume(old_id)
        self.assertEqual(conversation.goal.status, "completed")

    def test_goal_only_session_is_archived_and_can_be_resumed(self):
        conversation = self.conversation()
        goal = self.start(conversation)
        old_id = conversation.session_id
        conversation.clear()
        self.assertIsNone(conversation.goal)
        conversation.resume(old_id)
        self.assertEqual(conversation.goal.goal_id, goal.goal_id)

    def test_cannot_replace_unfinished_goal_or_resume_cancelled_goal(self):
        conversation = self.conversation()
        self.start(conversation)
        with self.assertRaises(ValueError):
            self.start(conversation, "另一个目标")
        prepare_goal(conversation, GoalCommand("cancel"))
        self.assertEqual(conversation.goal.status, "cancelled")
        with self.assertRaises(ValueError):
            prepare_goal(conversation, GoalCommand("resume"))
        self.start(conversation, "另一个目标")
        self.assertEqual(conversation.goal.objective, "另一个目标")

    def test_programmatic_zero_budget_is_rejected_without_creating_a_goal(self):
        conversation = self.conversation()
        with self.assertRaises(ValueError):
            prepare_goal(conversation, GoalCommand("start", "报告", max_rounds=0))
        self.assertIsNone(conversation.goal)

    def test_command_save_failure_preserves_existing_goal(self):
        conversation = self.conversation()
        before = self.start(conversation).to_dict()
        with (
            patch.object(conversation, "save", side_effect=OSError("disk unavailable")),
            self.assertRaises(OSError),
        ):
            prepare_goal(conversation, GoalCommand("cancel"))
        self.assertEqual(conversation.goal.to_dict(), before)

    def test_main_command_dispatch_and_status_do_not_call_model(self):
        conversation = self.conversation()
        console = Mock(spec=Console)
        console.interactive = False
        self.assertEqual(handle_command(conversation, console, "/goal").value, "handled")
        self.assertIn("没有目标", console.info.call_args.args[0])
        self.start(conversation)
        handle_command(conversation, console, "/goal status")
        handle_command(conversation, console, "/goal pause")
        self.assertEqual(conversation.goal.status, "paused")
        handle_command(conversation, console, "/goal cancel")
        self.assertEqual(conversation.goal.status, "cancelled")
        self.assertEqual(conversation.client.requests, [])

    def test_ui_uses_injected_task_runner_and_reports_goal_result(self):
        conversation = self.conversation(assessment())
        console = Mock(spec=Console)
        console.interactive = False

        def run(conv, output, prompt, **options):
            conv.last_outcome = {"status": "completed"}
            conv.messages.append(say("完整报告"))

        self.assertTrue(use_goal(conversation, console, ["写一份报告"], ask_task=run))
        self.assertEqual(conversation.goal.status, "completed")
        self.assertTrue(any("已完成" in call.args[0] for call in console.info.call_args_list))

    def test_interrupted_goal_tells_an_interactive_user_how_to_resume(self):
        conversation = self.conversation(assessment())
        console = Mock(spec=Console)
        console.interactive = True
        console.interrupt_on_escape.return_value = nullcontext()

        def run(conv, output, prompt, **options):
            self.assertEqual(options["title"], "目标 · 写一份报告")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            use_goal(conversation, console, ["写一份报告"], ask_task=run)
        self.assertEqual(conversation.goal.status, "paused")
        self.assertIn("/goal resume", console.info.call_args.args[0])

    def test_one_shot_cli_returns_goal_exit_status_and_preserves_quoted_arguments(self):
        from main import main

        for response, expected in (
            (assessment(criteria_met=[True]), 0),
            (assessment("waiting_user", next_step="需要材料", criteria_met=[False]), 1),
        ):
            with self.subTest(expected=expected):
                self.settings.session_file.unlink(missing_ok=True)
                client = FakeClient(say("报告内容"), response)
                console = Mock(spec=Console)
                console.interactive = False
                with (
                    patch("main.load_settings", return_value=self.settings),
                    patch("main.Console", return_value=console),
                    patch("main.model_client", return_value=client),
                    patch(
                        "sys.argv",
                        [
                            "main.py",
                            "/goal",
                            "start",
                            "撰写报告",
                            "--criterion",
                            "包含 引用",
                            "--max-rounds",
                            "1",
                        ],
                    ),
                ):
                    self.assertEqual(main(), expected)


class GoalEnvironmentTests(EnvironmentTestCase):
    def test_environment_keep_preserves_goal_and_new_session_archives_it(self):
        first = self.environment("first")
        second = self.environment("second")
        conversation = self.conversation()
        goal = prepare_goal(conversation, GoalCommand("start", "整理材料"))
        conversation.apply_environment(first, keep=True)
        self.assertEqual(conversation.goal.goal_id, goal.goal_id)
        old_id = conversation.session_id
        conversation.apply_environment(second)
        self.assertIsNone(conversation.goal)
        archived = conversation.sessions.read(conversation.sessions.snapshot_path(old_id))
        self.assertEqual(archived["goal"]["goal_id"], goal.goal_id)

    def test_failed_environment_switch_restores_goal(self):
        environment = self.environment()
        conversation = self.conversation()
        goal = prepare_goal(conversation, GoalCommand("start", "整理材料"))
        with (
            patch.object(
                conversation.sessions, "transition", side_effect=OSError("disk unavailable")
            ),
            self.assertRaises(OSError),
        ):
            conversation.apply_environment(environment)
        self.assertIs(conversation.goal, goal)
        self.assertEqual(conversation.sessions.read()["goal"]["goal_id"], goal.goal_id)


class GoalTeamTests(TeamTestCase):
    def test_idle_team_preserves_an_owned_process_handoff(self):
        script = TeamScript(main=Steps(
            use("exec", command=python_command("import time; time.sleep(30)"), background=True),
            say("The command is still running in the background."),
        ))
        conversation = self.team_conversation(script)
        answer = conversation.ask("Start the command and do not wait for it.")
        self.assertIn("still running", answer)
        self.assertFalse(conversation.team.went_background)
        self.assertEqual(conversation.last_outcome["status"], "background")
        self.assertIsNotNone(conversation.last_outcome["handoff_source"])

    def test_finished_background_member_still_requires_integration_check(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def write_after_handoff(_messages, _tools):
            self.assertTrue(release.wait(5))
            return use("write", path="feature.py", content="answer = 42\n")

        check = python_command("from feature import answer; assert answer == 42")
        script = TeamScript(
            main=Steps(say("后台成员已完成。"), team("wait", timeout=5), say("已完成。"), use("exec", command=check), say("已验证。")),
            engineer=Steps(write_after_handoff, use("exec", command=check), say("已验证。")),
        )
        conversation = self.team_conversation(script)
        conversation.team.begin_turn(time.monotonic() + 60, background=True)
        conversation.team.act_send_message(
            "main", None, to=["engineer"], kind="request", content="生成 feature.py",
            _context=ExecutionContext(time.monotonic() + 10),
        )
        conversation.team.end_turn()
        release.set()
        with conversation.team.changed:
            self.assertTrue(conversation.team.changed.wait_for(
                lambda: not conversation.team.activations, timeout=5,
            ))
        self.assertEqual(
            conversation.ask("汇报后台动态", task_contract=TaskContract(kind="answer")),
            "后台成员已完成。",
        )
        answer = conversation.ask("修复 feature.py", task_contract=TaskContract(kind="change"))
        self.assertEqual(answer, "已验证。")
        self.assertEqual(len(script.requests["main"]), 5)

    def test_member_write_before_inherited_baseline_needs_verification(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def write_when_resumed(_messages, _tools):
            self.assertTrue(release.wait(5))
            return use("write", path="feature.py", content="answer = 42\n")

        check = python_command("from feature import answer; assert answer == 42")
        script = TeamScript(
            main=Steps(say("已完成。"), use("exec", command=check), say("已验证。"), assessment()),
            engineer=Steps(write_when_resumed, use("exec", command=check), say("已验证。")),
        )
        conversation = self.team_conversation(script)
        conversation.team.begin_turn(time.monotonic() + 60, background=True)
        conversation.team.act_send_message(
            "main", None, to=["engineer"], kind="request", content="生成 feature.py",
            _context=ExecutionContext(time.monotonic() + 10),
        )
        conversation.team.end_turn()
        self.assertTrue(conversation.team.in_background)
        from task_acceptance import source_snapshot

        def snapshot_after_write(*args, **kwargs):
            release.set()
            with conversation.team.changed:
                self.assertTrue(conversation.team.changed.wait_for(
                    lambda: not conversation.team.activations, timeout=5,
                ))
            return source_snapshot(*args, **kwargs)

        prepare_goal(conversation, GoalCommand("start", "先分析再修复 feature.py"))
        with patch("task_acceptance.source_snapshot", side_effect=snapshot_after_write):
            self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertEqual(len(script.requests["main"]), 4)
        self.assertGreater(conversation.last_outcome["code_revision"], 0)

    def test_inherited_member_cannot_outlive_an_integrated_answer_after_one_nudge(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def engine(_messages, _tools):
            self.assertTrue(release.wait(5))
            return use("write", path="member.py", content="answer = 42\n")

        def verify_after_member(_messages, _tools):
            release.set()
            with conversation.team.changed:
                self.assertTrue(conversation.team.changed.wait_for(
                    lambda: not conversation.team.activations, timeout=5,
                ))
            return use("exec", command=python_command("from member import answer; assert answer == 42"))

        script = TeamScript(
            main=Steps(say("已完成。"), say("已完成。"), verify_after_member, say("已验证。"), assessment()),
            engineer=Steps(engine, say("已写入。")),
        )
        conversation = self.team_conversation(script)
        conversation.team.act_send_message(
            "main", None, to=["engineer"], kind="request", content="生成 member.py",
            _context=ExecutionContext(time.monotonic() + 10),
        )
        prepare_goal(conversation, GoalCommand("start", "先分析再修复 member.py"))
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertEqual(len(script.requests["main"]), 5)
        self.assertGreater(conversation.last_outcome["code_revision"], 0)

    def test_inherited_member_edits_require_goal_verification(self):
        def wait_for_member(_messages, _tools):
            with conversation.team.changed:
                self.assertTrue(conversation.team.changed.wait_for(
                    lambda: not conversation.team.activations, timeout=10,
                ))
            return say("已收到结果。")

        script = TeamScript(
            main=Steps(
                wait_for_member,
                use("exec", command=python_command("from feature import answer; assert answer == 42")),
                say("已验证成员的改动。"),
                assessment(),
            ),
            engineer=Steps(
                use("write", path="feature.py", content="answer = 42\n"),
                use("exec", command=python_command("from feature import answer; assert answer == 42")),
                say("已完成。"),
            ),
        )
        conversation = self.team_conversation(script)
        conversation.team.act_send_message(
            "main", None, to=["engineer"], kind="request", content="修复 feature.py",
            _context=ExecutionContext(time.monotonic() + 10),
        )
        prepare_goal(conversation, GoalCommand("start", "先分析再修复 feature.py"))
        self.assertEqual(GoalRuntime(conversation).run().status, "completed")
        self.assertEqual(len(script.requests["main"]), 4)
        self.assertGreater(conversation.last_outcome["code_revision"], 0)

    def test_goal_collects_team_results_and_restores_interactive_background_setting(self):
        script = TeamScript(
            main=Steps(
                team("create_task", title="搜集材料", assignee="researcher"),
                team("wait", timeout=3),
                say("已整理研究员提供的材料。"),
                assessment(),
            ),
            researcher=lambda messages, tools: say("已核对本地材料，提供研究结果。"),
        )
        conversation = self.team_conversation(script)
        conversation.team_background = True
        prepare_goal(conversation, GoalCommand("start", "调研材料并形成报告"))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "completed")
        self.assertTrue(conversation.team_background)
        self.assertFalse(conversation.team.in_background)
        tasks = conversation.team.state.data["tasks"]
        self.assertEqual(tasks["t1"]["status"], "completed")
        self.assertIn("研究结果", json.dumps(script.requests["main"][-1], ensure_ascii=False))

    def test_unfinished_team_task_prevents_completion(self):
        script = TeamScript(
            main=Steps(
                team("create_task", title="未完成的调查"),
                say("阶段性结果"),
                assessment(),
            )
        )
        conversation = self.team_conversation(script)
        prepare_goal(conversation, GoalCommand("start", "调研材料", max_rounds=1))
        goal = GoalRuntime(conversation).run()
        self.assertEqual(goal.status, "budget_exhausted")
        self.assertIn("未完成任务", goal.next_step)


if __name__ == "__main__":
    unittest.main()
