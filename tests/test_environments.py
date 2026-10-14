import io
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from commands import CommandResult
from conversation import Conversation
from environments import EnvironmentStore
from main import handle_command, sync_environment_ui, use_environment
from mcp_client import McpManager, McpServerConfig
from sessions import SessionConflictError
from skills import SkillCatalog
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tool_runtime import ExecutionContext, ToolResult
from tools import CodingTools, ToolPolicy
from ui import Console
from usage import CacheStats

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.program = self.root / "environments"
        self.program.mkdir()
        self.user = self.root / "user-environments"
        self.user.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.workspace / "session.json",
            workspace=self.workspace,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=4,
            skills_dir=self.root / "default-skills",
            templates_dir=self.root / "templates",
            mcp_enabled=False,
            environments_dir=self.program,
            environment_state_dir=self.root / "environment-state",
        )
        self.store = EnvironmentStore(self.settings, user_dir=self.user)

    def definition(self, name="review", *, directory=None, **values):
        directory = directory or self.program
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"version": 1, "name": name, **values}), encoding="utf-8")
        return path

    def resource(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def environment(self, name="review", **values):
        self.definition(name, **values)
        environment = self.store.resolve(f"program:{name}")
        self.store.approve(environment)
        return environment

    def conversation(self, settings=None, manager=None):
        settings = settings or self.settings
        client = FakeClient({"role": "assistant", "content": "done"})
        client.settings = settings
        client.cache_stats = CacheStats()
        tools = CodingTools(self.workspace)
        tools.attach_websearch(Mock())
        if manager is not None and manager.server_count:
            tools.attach_mcp(manager)
        conversation = Conversation(
            settings,
            client,
            tools,
            skills=SkillCatalog(self.workspace, settings.skills_dir),
            mcp_manager=manager,
        )
        conversation.environments = EnvironmentStore(
            settings, conversation.templates, user_dir=self.user
        )
        self.addCleanup(conversation.close)
        return conversation


class EnvironmentTests(EnvironmentTestCase):
    def test_read_only_review_environment_is_ready(self):
        self.resource(self.program / "review.md", "Synthetic review instructions")
        self.definition(
            prompt_file="review.md",
            reasoning_effort="high",
            websearch=False,
            mcp_config_file=None,
            tools={"deny": ["exec", "write", "edit", "delegate"]},
        )
        environment = self.store.resolve("program:review")
        self.assertEqual(environment.prompt_text, "Synthetic review instructions")
        self.assertEqual(environment.settings.reasoning_effort, "high")
        self.assertFalse(environment.settings.websearch_enabled)
        self.assertFalse(environment.settings.mcp_enabled)
        self.assertFalse(environment.policy.permits("exec"))
        self.assertFalse(environment.policy.permits("write"))
        self.assertFalse(environment.policy.permits("edit"))
        self.assertFalse(environment.policy.permits("delegate"))

    def test_strict_schema_rejects_invalid_or_process_only_fields(self):
        cases = [
            {"version": True},
            {"version": 2},
            {"name": "wrong"},
            {"api_key": "secret"},
            {"base_url": "http://bad"},
            {"model": "bad"},
            {"session_file": "bad"},
            {"guardrails": {}},
            {"websearch": "false"},
            {"skills_dirs": "skills"},
            {"reasoning_effort": "medium"},
            {"reasoning_effort": False},
            {"tools": {"handler": "bad"}},
            {"tools": {"deny": ["not-a-tool"]}},
            {"tools": {"allow": False}},
            {"max_subagents": True},
            {"max_subagents": 5},
            {"description": []},
            {"prompt_file": "missing"},
            {"opening_template": "missing"},
        ]
        for values in cases:
            with self.subTest(values=values):
                self.definition(**values)
                with self.assertRaises(ValueError):
                    self.store.resolve("review")

    def test_duplicate_json_keys_are_rejected(self):
        path = self.definition()
        path.write_text('{"version":1,"name":"review","websearch":true,"websearch":false}')
        with self.assertRaisesRegex(ValueError, "重复"):
            self.store.resolve("review")

    def test_relative_paths_are_anchored_and_cannot_escape(self):
        prompt = self.resource(self.program / "prompts" / "review.md", "Review only")
        self.definition(prompt_file="prompts/review.md")
        environment = self.store.resolve("review")
        self.assertEqual(environment.settings.prompt_file, prompt)
        outside = self.resource(self.root / "outside.md", "outside")
        for value in ("../outside.md", "prompts/../../outside.md"):
            self.definition(prompt_file=value)
            with self.assertRaisesRegex(ValueError, "逃逸"):
                self.store.resolve("review")
        link = self.program / "linked.md"
        link.symlink_to(outside)
        self.definition(prompt_file="linked.md")
        with self.assertRaisesRegex(ValueError, "逃逸"):
            self.store.resolve("review")
        self.definition(prompt_file=str(outside))
        self.assertIn(str(outside), self.store.summary(self.store.resolve("review")))

    def test_discovery_does_not_resolve_workspace_resources_and_names_are_qualified(self):
        self.definition()
        workspace_dir = self.workspace / ".agent" / "environments"
        self.definition(directory=workspace_dir, mcp_config_file="missing")
        self.definition(directory=self.user)
        self.assertEqual(
            [entry.source for entry in self.store.list()], ["program", "user", "workspace"]
        )
        with self.assertRaisesRegex(ValueError, "重名"):
            self.store.resolve("review")
        self.assertEqual(self.store.resolve("program:review").entry.source, "program")

    def test_symlinked_definition_or_skill_resource_is_rejected(self):
        outside = self.definition(directory=self.user)
        (self.program / "linked.json").symlink_to(outside)
        self.assertNotIn("linked", [entry.name for entry in self.store.list()])
        skill = self.resource(
            self.program / "skills" / "one" / "SKILL.md", "---\ndescription: one\n---\n"
        )
        skill.with_name("ref.md").symlink_to(ROOT / "README.md")
        self.definition(skills_dirs=["skills"])
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.store.resolve("program:review")

    def test_workspace_trust_hashes_prompt_and_all_skill_resources(self):
        directory = self.workspace / ".agent" / "environments"
        prompt = self.resource(directory / "prompt.md", "Review")
        reference = self.resource(
            directory / "skills" / "one" / "references" / "ref.md", "reference"
        )
        self.definition(directory=directory, prompt_file="prompt.md", skills_dirs=["skills"])
        environment = self.store.resolve("review")
        self.assertFalse(self.store.approved(environment))
        self.store.approve(environment)
        self.assertTrue(EnvironmentStore(self.settings, user_dir=self.user).approved(environment))
        for resource in (prompt, reference):
            resource.write_text(resource.read_text() + " changed")
            current = self.store.resolve("review")
            self.assertFalse(self.store.approved(current))
            self.assertNotEqual(current.fingerprint, environment.fingerprint)
            self.store.approve(current)
            environment = current
        self.resource(directory / "skills" / "one" / "new.md", "new resource")
        self.assertFalse(self.store.approved(self.store.resolve("review")))

    def test_deny_and_corrupt_trust_fail_closed(self):
        environment = self.environment()
        self.store.deny("review")
        self.assertFalse(self.store.approved(environment))
        self.assertTrue(self.store.denied(environment))
        self.store.approve(environment)
        self.assertFalse(self.store.denied(environment))
        self.store.trust_file.write_text("not json")
        with self.assertRaises(ValueError):
            self.store.approved(environment)

    def test_mcp_expansion_and_local_script_are_fingerprinted_without_exposing_secrets(self):
        script = self.resource(self.program / "server.py", "print('server')")
        config = {
            "mcpServers": {
                "local": {
                    "command": sys.executable,
                    "args": ["server.py", "--token", "${ENV_TEST_TOKEN}"],
                    "env": {"TOKEN": "${ENV_TEST_TOKEN}"},
                },
                "remote": {
                    "url": "https://user:secret@example.invalid/mcp?token=secret",
                    "headers": {"Authorization": "${ENV_TEST_TOKEN}"},
                },
            }
        }
        self.resource(self.program / "mcp.json", json.dumps(config))
        self.definition(mcp_config_file="mcp.json")
        with patch.dict(os.environ, {"ENV_TEST_TOKEN": "private-token"}):
            first = self.store.resolve("review")
            self.store.approve(first)
            summary = self.store.summary(first)
            self.assertNotIn("private-token", summary)
            self.assertNotIn("secret", summary)
            self.assertEqual(first.servers["local"].cwd, self.program)
            script.write_text("changed script")
            self.assertNotEqual(first.fingerprint, self.store.resolve("review").fingerprint)
        with patch.dict(os.environ, {"ENV_TEST_TOKEN": "changed-token"}):
            second = self.store.resolve("review")
            self.assertFalse(self.store.approved(second))

    def test_websearch_and_worker_limits_cannot_expand_startup_permissions(self):
        store = EnvironmentStore(
            replace(self.settings, websearch_enabled=False), user_dir=self.user
        )
        self.definition(websearch=True)
        with self.assertRaisesRegex(ValueError, "启动"):
            store.resolve("review")
        self.definition(max_subagents=5)
        with self.assertRaisesRegex(ValueError, "启动上限"):
            self.store.resolve("review")

    def test_inherited_mcp_handshake_mode_changes_environment_approval(self):
        self.definition()
        settings = replace(self.settings, mcp_enabled=True)
        server = McpServerConfig("local", command=sys.executable)
        auto_store = EnvironmentStore(
            settings, user_dir=self.user, baseline_servers={"local": server}
        )
        first = auto_store.resolve("review")
        auto_store.approve(first)
        legacy_store = EnvironmentStore(
            settings,
            user_dir=self.user,
            baseline_servers={"local": replace(server, mode="legacy")},
        )
        second = legacy_store.resolve("review")

        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertFalse(legacy_store.approved(second))

    def test_mcp_executable_and_path_resolution_are_fingerprinted(self):
        executable = self.resource(
            self.root / "bin" / "environment-test-command", "#!/bin/sh\nexit 0\n"
        )
        executable.chmod(0o700)
        self.resource(
            self.program / "mcp.json",
            '{"mcpServers":{"local":{"command":"environment-test-command"}}}',
        )
        self.definition(mcp_config_file="mcp.json")
        with patch.dict(os.environ, {"PATH": str(executable.parent)}):
            first = self.store.resolve("review")
            self.assertEqual(first.servers["local"].command, str(executable))
            self.store.approve(first)
            executable.write_text("#!/bin/sh\nexit 1\n")
            self.assertFalse(self.store.approved(self.store.resolve("review")))

    def test_summary_escapes_terminal_control_characters(self):
        self.definition(description="safe\n\x1b[2Jfake summary")
        summary = self.store.summary(self.store.resolve("review"))
        self.assertNotIn("\x1b", summary)
        self.assertIn(r"\x0a\x1b", summary)

    def test_switch_new_keep_and_off_use_startup_baseline(self):
        prompt = self.resource(self.program / "prompt.md", "REVIEW ENVIRONMENT")
        skill = self.resource(
            self.program / "skills" / "review" / "SKILL.md",
            "---\nname: review\ndescription: Review\n---\nReview body",
        )
        self.resource(
            self.workspace / ".agent" / "skills" / "bad" / "SKILL.md",
            "---\nname: bad\ndescription: Workspace\n---\n",
        )
        first = self.environment(
            prompt_file="prompt.md",
            skills_dirs=["skills"],
            reasoning_effort="low",
            websearch=False,
            tools={"deny": ["delegate", "exec"]},
            max_subagents=1,
        )
        second = self.environment("plain")
        conversation = self.conversation()
        original_tools = conversation.tools
        original_roots = conversation.skills.roots
        conversation.messages.append({"role": "user", "content": "old"})
        old_id = conversation.session_id
        conversation.apply_environment(first)
        self.assertIs(conversation.tools, original_tools)
        self.assertEqual(conversation.environment_id, "program:review")
        self.assertNotEqual(conversation.session_id, old_id)
        self.assertEqual(conversation.messages, [])
        self.assertEqual(conversation.settings.prompt_file, prompt)
        self.assertEqual(conversation.skills.skills[0].file_path, skill)
        self.assertEqual(conversation.skills.roots, [self.program / "skills"])
        self.assertEqual(conversation.settings.reasoning_effort, "low")
        self.assertNotIn("exec", [item["function"]["name"] for item in conversation.definitions])
        self.assertNotIn("websearch", conversation.tools.registry)
        self.assertEqual(conversation.sessions.read()["environment_id"], first.environment_id)
        conversation.messages.append({"role": "user", "content": "keep me"})
        kept_id = conversation.session_id
        conversation.apply_environment(second, keep=True)
        self.assertEqual(conversation.messages[-1]["content"], "keep me")
        self.assertEqual(conversation.session_id, kept_id)
        self.assertEqual(conversation.settings.reasoning_effort, self.settings.reasoning_effort)
        self.assertEqual(conversation.settings.max_subagents, 4)
        self.assertTrue(conversation.settings.websearch_enabled)
        self.assertEqual(conversation.skills.roots, [])
        conversation.apply_environment(None)
        self.assertIsNone(conversation.environment_id)
        self.assertEqual(conversation.settings, self.settings)
        self.assertEqual(conversation.skills.roots, original_roots)
        self.assertEqual(conversation.tools.policy, ToolPolicy())
        self.assertIn("websearch", conversation.tools.registry)
        self.assertNotIn("environment_id", conversation.sessions.read())

    def test_new_preserves_environment_and_restart_does_not_apply_it(self):
        environment = self.environment(reasoning_effort="low")
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.clear()
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(conversation.sessions.read()["environment_id"], environment.environment_id)
        conversation.messages.append({"role": "user", "content": "history"})
        conversation.save()
        restarted = self.conversation()
        self.assertIsNone(restarted.environment_id)
        self.assertEqual(restarted.last_environment_id, environment.environment_id)
        self.assertEqual(restarted.settings.reasoning_effort, self.settings.reasoning_effort)
        self.assertEqual(restarted.messages, conversation.messages)
        with patch.object(restarted.environments, "audit"):
            restarted.resume(restarted.session_id)
        self.assertEqual(restarted.environment_id, environment.environment_id)

    def test_opening_template_does_not_change_template_selection_or_restart_discard_messages(self):
        conversation = self.conversation()
        template = conversation.templates.create("start", [{"role": "user", "content": "opening"}])
        self.definition(opening_template="start")
        environment = conversation.environments.resolve("review")
        conversation.environments.approve(environment)
        conversation.apply_environment(environment)
        self.assertEqual(conversation.messages, [{"role": "user", "content": "opening"}])
        self.assertIsNone(conversation.templates.selected_id)
        self.assertIsNone(conversation.template_id)
        restarted = self.conversation()
        self.assertEqual(restarted.messages, conversation.messages)
        conversation.messages.append({"role": "user", "content": "retained"})
        conversation.apply_environment(environment, keep=True)
        self.assertEqual(len(conversation.messages), 2)
        template.path.write_text(template.path.read_text().replace("opening", "modified"))
        self.assertFalse(
            conversation.environments.approved(conversation.environments.resolve("review"))
        )

    def test_resume_restores_environment_or_requires_explicit_messages_only(self):
        first = self.environment(reasoning_effort="low", tools={"deny": ["exec"]})
        second = self.environment("second", reasoning_effort="high")
        conversation = self.conversation()
        conversation.apply_environment(first)
        conversation.messages.append({"role": "user", "content": "review session"})
        session_id = conversation.session_id
        conversation.apply_environment(second)
        conversation.resume(session_id)
        self.assertEqual(conversation.environment_id, first.environment_id)
        self.assertEqual(conversation.settings.reasoning_effort, "low")
        conversation.apply_environment(second)
        first.entry.path.unlink()
        with self.assertRaisesRegex(ValueError, "不可安全恢复"):
            conversation.resume(session_id)
        self.assertEqual(conversation.environment_id, second.environment_id)
        conversation.resume(session_id, messages_only=True)
        self.assertEqual(conversation.environment_id, second.environment_id)
        self.assertIn("保持当前能力", conversation.recovery_notice)
        self.assertNotEqual(conversation.session_id, session_id)
        self.assertEqual(conversation.messages[0]["content"], "review session")

    def test_resume_cannot_follow_a_relocated_or_changed_environment(self):
        environment = self.environment(reasoning_effort="low")
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.messages.append({"role": "user", "content": "history"})
        session_id = conversation.session_id
        conversation.apply_environment(None)
        changed = self.environment(reasoning_effort="high")
        with self.assertRaisesRegex(ValueError, "指纹"):
            conversation.resume(session_id)
        self.assertNotEqual(environment.reference, changed.reference)
        self.assertIsNone(conversation.environment_id)

    def test_active_changed_or_denied_environment_refuses_ask_before_any_model_or_messages(self):
        prompt = self.resource(self.program / "prompt.md", "Original")
        environment = self.environment(prompt_file="prompt.md")
        conversation = self.conversation()
        conversation.apply_environment(environment)
        prompt.write_text("Changed")
        with self.assertRaisesRegex(ValueError, "改变"):
            conversation.ask("must not run")
        self.assertEqual(conversation.messages, [])
        self.assertEqual(conversation.client.requests, [])
        prompt.write_text("Original")
        conversation.environments.deny("review")
        with self.assertRaises(ValueError):
            conversation.ask("must not run")

    def test_filtering_applies_to_definitions_workers_and_direct_execution(self):
        conversation = self.conversation()
        environment = self.environment(tools={"allow": ["read", "ls"], "deny": ["read"]})
        conversation.apply_environment(environment)
        self.assertEqual([item["function"]["name"] for item in conversation.definitions], ["ls"])
        self.assertEqual(
            [item["function"]["name"] for item in conversation.tools.worker_definitions], ["ls"]
        )
        call = {"function": {"name": "read", "arguments": '{"path":"x"}'}}
        result = json.loads(conversation.tools.execute(call))
        self.assertEqual(result["error"]["code"], "unavailable_tool")
        result = json.loads(
            conversation.tools.execute_workspace(call, ExecutionContext(time.monotonic() + 5))
        )
        self.assertEqual(result["error"]["code"], "unavailable_tool")
        conversation.reload_prompt()
        self.assertIn("Available tools: ls.", conversation.system_message["content"])

    def test_startup_policy_is_never_expanded(self):
        environment = self.environment(tools={"allow": ["read", "exec"]})
        conversation = self.conversation()
        conversation._baseline_policy = ToolPolicy(deny=frozenset({"exec"}))
        conversation.apply_environment(environment)
        self.assertFalse(conversation.tools.policy.permits("exec"))

    def test_empty_allowlist_has_no_tools_or_command_instructions(self):
        environment = self.environment(tools={"allow": []})
        conversation = self.conversation()
        conversation.apply_environment(environment)
        self.assertEqual(conversation.definitions, [])
        self.assertIn("No tools are available", conversation.system_message["content"])
        self.assertNotIn("exec.timeout", conversation.system_message["content"])

    def test_denied_mcp_status_survives_provider_refresh(self):
        conversation = self.conversation()
        environment = self.environment(tools={"deny": ["mcp"]})
        conversation.apply_environment(environment)
        manager = Mock()
        manager.status.return_value = ToolResult(
            data={"servers": [{"name": "hidden", "status": "not-connected"}]}
        )
        conversation.mcp_manager = manager
        console = Console()
        sync_environment_ui(conversation, console)
        console._refresh_mcp()
        self.assertIn("策略禁用", console.mcp_status)

    def test_worker_prompt_tools_and_settings_are_snapshots(self):
        prompt = self.resource(self.program / "prompt.md", "ENVIRONMENT SCOPE")
        first = self.environment(
            prompt_file="prompt.md", reasoning_effort="low", tools={"deny": ["read", "websearch"]}
        )
        second = self.environment("second", reasoning_effort="high")
        conversation = self.conversation()
        conversation.apply_environment(first)
        runner = conversation.tools.registry["delegate"].handler.__self__.runner
        self.assertIn("ENVIRONMENT SCOPE", runner.environment_instructions)
        self.assertNotIn("read", [item["function"]["name"] for item in runner.worker_tools])
        conversation.apply_environment(second)
        self.assertEqual(runner.client.settings.reasoning_effort, "low")
        call = {"function": {"name": "read", "arguments": json.dumps({"path": str(prompt)})}}
        result = json.loads(runner.execute_workspace(call, ExecutionContext(time.monotonic() + 5)))
        self.assertEqual(result["error"]["code"], "unavailable_tool")

    def test_failed_commit_restores_memory_cache_tools_messages_and_disk(self):
        environment = self.environment(reasoning_effort="low", tools={"deny": ["exec"]})
        conversation = self.conversation()
        conversation.messages.append({"role": "user", "content": "preserve"})
        conversation.tools.task_title = "task"
        conversation.tools.task_plan = {
            "title": "task", "steps": [{"step": "inspect", "status": "pending"}]
        }
        conversation.client.cache_stats.record(
            {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}, "solve"
        )
        conversation.save()
        original = conversation.__dict__.copy()
        original_file = self.settings.session_file.read_bytes()
        snapshots = {
            path: path.read_bytes() for path in conversation.sessions.directory.glob("*.json")
        }
        original_write = conversation.sessions._write

        def fail_active(path, content):
            if path == self.settings.session_file:
                raise OSError("injected write failure")
            original_write(path, content)

        with (
            patch.object(conversation.sessions, "_write", side_effect=fail_active),
            self.assertRaisesRegex(OSError, "injected"),
        ):
            conversation.apply_environment(environment)
        for key in (
            "settings",
            "skills",
            "tools",
            "system_message",
            "messages",
            "session_id",
            "tool_runs",
            "environment_id",
        ):
            self.assertIs(conversation.__dict__[key], original[key])
        self.assertEqual(conversation.client.settings, self.settings)
        self.assertEqual(conversation.client.cache_stats.snapshot()["prompt_tokens"], 10)
        self.assertEqual(conversation.tools.task_title, "task")
        self.assertEqual(conversation.tools.task_plan["steps"][0]["step"], "inspect")
        self.assertTrue(conversation.tools.policy.permits("exec"))
        self.assertEqual(self.settings.session_file.read_bytes(), original_file)
        self.assertEqual(
            {path: path.read_bytes() for path in conversation.sessions.directory.glob("*.json")},
            snapshots,
        )
        events = [
            json.loads(line)["event"]
            for line in conversation.environments.audit_file.read_text().splitlines()
        ]
        self.assertEqual(events[-2:], ["attempt", "failed"])

    def test_session_conflict_does_not_switch_environment(self):
        environment = self.environment()
        conversation = self.conversation()
        conversation.messages.append({"role": "user", "content": "current"})
        conversation.save()
        self.settings.session_file.write_text('{"external":"update"}')
        with self.assertRaises(SessionConflictError):
            conversation.apply_environment(environment)
        self.assertIsNone(conversation.environment_id)
        self.assertEqual(self.settings.session_file.read_text(), '{"external":"update"}')

    def test_reentrant_ask_and_switch_are_rejected(self):
        conversation = self.conversation()
        environment = self.environment()
        conversation._operation.acquire()
        try:
            with self.assertRaises(RuntimeError):
                conversation.ask("blocked")
            with self.assertRaises(RuntimeError):
                conversation.apply_environment(environment)
        finally:
            conversation._operation.release()
        self.assertFalse(conversation._operation.locked())

    def test_audit_failure_before_commit_aborts_but_after_commit_only_warns(self):
        environment = self.environment()
        conversation = self.conversation()
        audit = conversation.environments.audit
        with (
            patch.object(conversation.environments, "audit", side_effect=OSError("unwritable")),
            self.assertRaises(OSError),
        ):
            conversation.apply_environment(environment)
        self.assertIsNone(conversation.environment_id)

        def fail_completion(event, *args, **kwargs):
            if event == "applied":
                raise OSError("completion audit failed")
            return audit(event, *args, **kwargs)

        with patch.object(conversation.environments, "audit", side_effect=fail_completion):
            warnings = conversation.apply_environment(environment)
        self.assertIn("已提交", warnings[-1])
        self.assertEqual(conversation.sessions.read()["environment_id"], environment.environment_id)

    def test_prepared_objects_cannot_inject_mutated_servers_or_template_messages(self):
        conversation = self.conversation()
        environment = self.environment()
        environment.servers["injected"] = Mock()
        conversation.apply_environment(environment)
        self.assertIsNone(conversation.mcp_manager)
        self.assertEqual(conversation.active_environment.servers, {})

    def test_current_session_resume_after_deny_preserves_state_without_widening_capabilities(
        self,
    ):
        environment = self.environment()
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.messages.append({"role": "user", "content": "unsaved"})
        conversation.environments.deny("review")
        with self.assertRaisesRegex(ValueError, "不可安全恢复"):
            conversation.resume(conversation.session_id)
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(conversation.messages[-1]["content"], "unsaved")

    def test_cli_environment_startup_and_show_never_connect_default_mcp(self):
        import main

        config = self.resource(
            self.root / "default-mcp.json",
            '{"mcpServers":{"default":{"command":"agent-nonexistent-command"}}}',
        )
        settings = replace(self.settings, mcp_enabled=True, mcp_config_file=config)
        self.definition(mcp_config_file=None)
        for arguments in (
            ["/environment", "show", "review"],
            ["--environment", "review", "--yes", "/prompt"],
        ):
            with self.subTest(arguments=arguments):
                manager = McpManager(config, self.workspace)
                client = FakeClient()
                created = []

                def create(*args, captured=created, **kwargs):
                    conversation = Conversation(*args, **kwargs)
                    conversation.environments.roots["user"] = self.user
                    captured.append(conversation)
                    return conversation

                console = Mock(interactive=False)
                with (
                    patch("main.Settings.from_env", return_value=settings),
                    patch("main.McpManager", return_value=manager),
                    patch.object(manager, "connect_all") as connect,
                    patch("main.DeepSeekClient", return_value=client),
                    patch("main.Conversation", side_effect=create),
                    patch("main.Console", return_value=console),
                    patch("sys.argv", ["main.py", *arguments]),
                    patch("sys.stdout", io.StringIO()),
                ):
                    self.assertEqual(main.main(), 0)
                connect.assert_not_called()
                self.assertTrue(manager._closed)
                self.assertEqual(client.requests, [])
                self.assertEqual(
                    created[0].environment_id,
                    "program:review" if arguments[0] == "--environment" else None,
                )

    def test_new_refreshes_worker_cache_snapshot_without_expanding_policy(self):
        environment = self.environment(tools={"deny": ["read"]})
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.clear()
        runner = conversation.tools.registry["delegate"].handler.__self__.runner
        self.assertIs(runner.client.cache_stats, conversation.client.cache_stats)
        self.assertNotIn("read", [item["function"]["name"] for item in runner.worker_tools])

    def test_switch_and_effort_preserve_global_worker_slots(self):
        conversation = self.conversation(settings=replace(self.settings, max_subagents=1))
        self.definition()
        environment = conversation.environments.resolve("review")
        conversation.environments.approve(environment)
        slots = conversation.tools._global_worker_slots
        slots.acquire()
        try:
            conversation.apply_environment(environment)
            conversation.set_reasoning_effort("low")
            reasoning = conversation.tools.registry["delegate"].handler.__self__
            self.assertIs(reasoning.global_slots, slots)
            execution = ExecutionContext(time.monotonic() + 0.02)
            with self.assertRaises(TimeoutError):
                reasoning.submit(execution, {"task": "blocked", "expected_output": "report"}, "")
            self.assertTrue(reasoning.slots.acquire(blocking=False))
            reasoning.slots.release()
            self.assertEqual(conversation.client.requests, [])
        finally:
            slots.release()

    def test_noninteractive_commands_list_show_confirm_trust_deny_and_keep(self):
        conversation = self.conversation()
        self.definition(reasoning_effort="low")
        console = Console()
        console.interactive = False
        output = io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            self.assertIs(handle_command(conversation, console, "/env"), CommandResult.HANDLED)
            self.assertIs(handle_command(conversation, console, "/environment show review"), CommandResult.HANDLED)
            self.assertIsNone(conversation.environment_id)
            self.assertFalse(use_environment(conversation, console, ["review"]))
            self.assertTrue(use_environment(conversation, console, ["review", "--yes"]))
            self.assertEqual(console.environment_name, "program:review")
            self.assertTrue(use_environment(conversation, console, ["deny", "review"]))
            self.assertFalse(use_environment(conversation, console, ["review", "--yes"]))
            self.assertTrue(use_environment(conversation, console, ["trust", "review", "--yes"]))
            self.assertTrue(use_environment(conversation, console, ["review", "keep"]))
            self.assertTrue(use_environment(conversation, console, ["off"]))
            self.assertFalse(use_environment(conversation, console, ["save", "export"]))
        self.assertIn("可能与新环境不匹配", output.getvalue())
        self.assertIn("暂未实现", output.getvalue())

    def test_workspace_requires_explicit_trust_even_with_yes(self):
        self.definition(directory=self.workspace / ".agent" / "environments")
        conversation = self.conversation()
        console = Mock(interactive=False)
        self.assertFalse(use_environment(conversation, console, ["review", "--yes"]))
        self.assertTrue(use_environment(conversation, console, ["trust", "review", "--yes"]))
        self.assertIsNone(conversation.environment_id)
        self.assertTrue(use_environment(conversation, console, ["review"]))

    def test_interactive_environment_menu_selects_an_existing_environment(self):
        environment = self.environment(reasoning_effort="low")
        conversation = self.conversation()
        session_id = conversation.session_id
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 1

        self.assertIs(handle_command(conversation, console, "/env"), CommandResult.HANDLED)

        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(conversation.settings.reasoning_effort, "low")
        self.assertNotEqual(conversation.session_id, session_id)
        console.select_option.assert_called_once()
        selection = console.select_option.call_args
        self.assertEqual(selection.kwargs, {"default_index": 0, "title": "选择能力环境"})
        self.assertEqual(
            [option["label"] for option in selection.args[1]],
            ["默认（当前）", "program:review"],
        )
        self.assertFalse(
            any(call.args[0].startswith("当前环境：") for call in console.info.call_args_list)
        )

    def test_interactive_environment_menu_marks_current_and_restores_default(self):
        environment = self.environment(reasoning_effort="low")
        conversation = self.conversation()
        conversation.apply_environment(environment)
        session_id = conversation.session_id
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 0

        self.assertTrue(use_environment(conversation, console, []))

        self.assertIsNone(conversation.environment_id)
        self.assertEqual(conversation.settings, self.settings)
        self.assertNotEqual(conversation.session_id, session_id)
        selection = console.select_option.call_args
        self.assertEqual(selection.kwargs["default_index"], 1)
        self.assertEqual(
            [option["label"] for option in selection.args[1]],
            ["默认", "program:review（当前）"],
        )

    def test_interactive_environment_menu_cancel_does_not_resolve_or_mutate(self):
        self.definition()
        conversation = self.conversation()
        session_id = conversation.session_id
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = None
        with (
            patch.object(conversation.environments, "resolve") as resolve,
            patch.object(conversation.environments, "audit") as audit,
            patch.object(conversation, "apply_environment") as apply_environment,
        ):
            self.assertTrue(use_environment(conversation, console, []))

        resolve.assert_not_called()
        audit.assert_not_called()
        apply_environment.assert_not_called()
        self.assertIsNone(conversation.environment_id)
        self.assertEqual(conversation.session_id, session_id)
        self.assertFalse(conversation.environments.trust_file.exists())
        console.info.assert_called_once_with("已取消选择环境，当前环境与会话不变。")

    def test_interactive_environment_menu_uses_qualified_names_for_duplicates(self):
        self.definition()
        self.definition(directory=self.user)
        environment = self.store.resolve("user:review")
        self.store.approve(environment)
        conversation = self.conversation()
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 2

        self.assertTrue(use_environment(conversation, console, []))

        self.assertEqual(conversation.environment_id, "user:review")
        self.assertEqual(
            [option["label"] for option in console.select_option.call_args.args[1]],
            ["默认（当前）", "program:review", "user:review"],
        )

    def test_interactive_environment_menu_keeps_first_use_confirmation(self):
        self.definition()
        conversation = self.conversation()
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.side_effect = [1, 1]

        self.assertTrue(use_environment(conversation, console, []))

        self.assertEqual(conversation.environment_id, "program:review")
        self.assertTrue(conversation.environments.trust_file.exists())
        self.assertEqual(
            [call.kwargs["title"] for call in console.select_option.call_args_list],
            ["选择能力环境", "确认环境能力"],
        )

    def test_interactive_environment_menu_does_not_bypass_workspace_trust(self):
        self.definition(directory=self.workspace / ".agent" / "environments")
        conversation = self.conversation()
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 1

        self.assertFalse(use_environment(conversation, console, []))

        self.assertIsNone(conversation.environment_id)
        self.assertFalse(conversation.environments.trust_file.exists())
        console.select_option.assert_called_once()
        self.assertIn("工作区环境未信任", str(console.error.call_args.args[0]))

    def test_interactive_cancel_does_not_confirm_or_apply(self):
        self.definition()
        conversation = self.conversation()
        console = Mock(interactive=True)
        console.select_option.return_value = 0
        self.assertFalse(use_environment(conversation, console, ["review"]))
        self.assertFalse(conversation.environments.trust_file.exists())
        self.assertIsNone(conversation.environment_id)

    def test_default_boot_banner_remembers_environment_without_auto_applying(self):
        conversation = self.conversation()
        environment = self.environment()
        conversation.apply_environment(environment)
        restarted = self.conversation()
        console = Console()
        sync_environment_ui(restarted, console)
        self.assertEqual(console.environment_name, "默认")
        self.assertEqual(console.last_environment_name, environment.environment_id)
        self.assertIn("上次", console._toolbar())


class EnvironmentMcpTests(EnvironmentTestCase):
    def test_off_restores_disabled_baseline_without_adding_mcp_tool(self):
        config = self.resource(
            self.root / "disabled-mcp.json",
            '{"mcpServers":{"disabled":{"command":"unused-command","disabled":true}}}',
        )
        settings = replace(self.settings, mcp_enabled=True, mcp_config_file=config)
        manager = McpManager(config, self.workspace)
        conversation = self.conversation(settings=settings, manager=manager)
        definitions = conversation.definitions
        environment = self.environment(mcp_config_file=None)
        conversation.apply_environment(environment)
        conversation.apply_environment(None)
        self.assertEqual(conversation.definitions, definitions)
        self.assertNotIn("mcp", conversation.tools.registry)
        console = Console()
        sync_environment_ui(conversation, console)
        console._refresh_mcp()
        self.assertEqual(console.mcp_status, "未配置")

    def mcp_environment(self, *, good=True, bad=False):
        servers = {}
        if good:
            servers["fixture"] = {
                "command": sys.executable,
                "args": [str(ROOT / "tests" / "fixtures_mcp_server.py")],
            }
        if bad:
            servers["broken"] = {"command": "agent-nonexistent-command", "timeout": 1}
        self.resource(self.program / "mcp.json", json.dumps({"mcpServers": servers}))
        return self.environment(mcp_config_file="mcp.json")

    def test_real_mcp_switch_all_available_partial_and_unavailable(self):
        for good, bad in ((True, False), (True, True), (False, True)):
            with self.subTest(good=good, bad=bad):
                environment = self.mcp_environment(good=good, bad=bad)
                conversation = self.conversation()
                warnings = conversation.apply_environment(environment)
                self.assertEqual(bool(warnings), bad)
                self.assertEqual(conversation.mcp_manager.servers, environment.servers)
                self.assertEqual(conversation.environment_id, environment.environment_id)
                if good:
                    result = conversation.mcp_manager.call(
                        "echo", {"text": "ok"}, "fixture", ExecutionContext(time.monotonic() + 10)
                    )
                    self.assertEqual(result.status, "ok")
                conversation.close()

    def test_old_mcp_is_not_closed_on_failure_and_candidate_is_closed(self):
        environment = self.mcp_environment(good=False, bad=True)
        old = Mock()
        old.servers = {}
        conversation = self.conversation(manager=old)
        candidate = Mock()
        candidate.connect_all.return_value = ToolResult(data={"failures": [], "servers": []})
        with (
            patch("environment_transition.McpManager", return_value=candidate),
            patch.object(conversation.sessions, "transition", side_effect=OSError("fail")),
            self.assertRaises(OSError),
        ):
            conversation.apply_environment(environment)
        old.close.assert_not_called()
        candidate.close.assert_called_once()
        self.assertIs(conversation.mcp_manager, old)
        self.assertIsNone(conversation.environment_id)

    def test_old_connection_is_closed_after_commit_and_off_rebuilds_baseline(self):
        baseline_config = self.resource(
            self.root / "baseline-mcp.json",
            '{"mcpServers":{"old":{"command":"agent-nonexistent-command","timeout":1}}}',
        )
        settings = replace(self.settings, mcp_enabled=True, mcp_config_file=baseline_config)
        old = McpManager(baseline_config, self.workspace)
        conversation = self.conversation(settings=settings, manager=old)
        environment = self.mcp_environment(good=False, bad=True)
        conversation.apply_environment(environment)
        self.assertTrue(old._closed)
        active = conversation.mcp_manager
        conversation.apply_environment(None)
        self.assertTrue(active._closed)
        self.assertEqual(list(conversation.mcp_manager.servers), ["old"])
        self.assertIsNot(conversation.mcp_manager, old)

    def test_parse_failure_and_prompt_failure_leave_old_manager_untouched(self):
        environment = self.mcp_environment(good=False, bad=True)
        conversation = self.conversation()
        candidate = Mock()
        candidate.connect_all.return_value = ToolResult(data={"failures": [], "servers": []})
        with (
            patch("environment_transition.McpManager", return_value=candidate),
            patch.object(conversation, "_compose_prompt", side_effect=ValueError("prompt failure")),
            self.assertRaisesRegex(ValueError, "prompt failure"),
        ):
            conversation.apply_environment(environment)
        candidate.close.assert_called_once()
        self.resource(self.program / "mcp.json", "invalid")
        with self.assertRaises(ValueError):
            conversation.environments.resolve("review")
        self.assertIsNone(conversation.environment_id)

    def test_resources_changed_during_connection_roll_back(self):
        environment = self.mcp_environment(good=False, bad=True)
        conversation = self.conversation()
        candidate = Mock()

        def mutate(**kwargs):
            self.definition(description="changed while connecting")
            return ToolResult(data={"failures": [], "servers": []})

        candidate.connect_all.side_effect = mutate
        with (
            patch("environment_transition.McpManager", return_value=candidate),
            self.assertRaises(ValueError),
        ):
            conversation.apply_environment(environment)
        candidate.close.assert_called_once()
        self.assertIsNone(conversation.environment_id)

    def test_postcommit_cleanup_failure_is_a_warning_not_a_rollback(self):
        environment = self.environment()
        old = Mock()
        old.servers = {}
        old.close.side_effect = OSError("cannot close")
        conversation = self.conversation(manager=old)
        warnings = conversation.apply_environment(environment)
        self.assertIn("已提交", warnings[-1])
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(conversation.sessions.read()["environment_id"], environment.environment_id)


class EnvironmentWorkerTests(EnvironmentTestCase):
    def test_worker_tools_and_role_prompts_apply_to_subagents(self):
        self.resource(self.program / "review" / "prompt.md", "Review prompt")
        self.resource(self.program / "review" / "roles" / "recon.md", "RECON ROLE")
        environment = self.environment(
            prompt_file="review/prompt.md",
            worker_tools=["exec", "mcp"],
            role_prompts={"recon": "review/roles/recon.md"},
        )
        self.assertEqual(environment.worker_tools, frozenset({"exec", "mcp"}))
        self.assertEqual(environment.role_prompts, {"recon": "RECON ROLE"})
        conversation = self.conversation()
        conversation.apply_environment(environment)
        names = {item["function"]["name"] for item in conversation.tools.worker_definitions}
        self.assertIn("exec", names)
        self.assertNotIn("write", names)
        runner = conversation.tools.registry["delegate"].handler.__self__.runner
        self.assertEqual(runner.worker_roles, {"recon": "RECON ROLE"})
        worker = json.loads(
            conversation.tools.execute_workspace(
                {
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps({"command": "printf worker-granted"}),
                    }
                },
                ExecutionContext(time.monotonic() + 30),
            )
        )
        self.assertEqual(worker["status"], "ok", worker)

    def test_environment_off_restores_read_only_workers(self):
        self.resource(self.program / "review" / "prompt.md", "Review prompt")
        self.resource(self.program / "review" / "roles" / "recon.md", "RECON ROLE")
        environment = self.environment(
            prompt_file="review/prompt.md",
            worker_tools=["exec"],
            role_prompts={"recon": "review/roles/recon.md"},
        )
        conversation = self.conversation()
        conversation.apply_environment(environment)
        self.assertIn(
            "exec",
            {item["function"]["name"] for item in conversation.tools.worker_definitions},
        )
        conversation.apply_environment(None)
        self.assertEqual(
            {item["function"]["name"] for item in conversation.tools.worker_definitions},
            {"read", "grep", "find", "ls", "websearch"},
        )
        runner = conversation.tools.registry["delegate"].handler.__self__.runner
        self.assertEqual(runner.worker_roles, {})

    def test_worker_tools_rejects_unknown_reserved_and_bad_role_fields(self):
        cases = (
            {"worker_tools": "exec"},
            {"worker_tools": ["not-a-tool"]},
            {"worker_tools": ["delegate"]},
            {"worker_tools": ["ask_user"]},
            {"worker_tools": ["update_plan"]},
            {"role_prompts": "recon"},
            {"role_prompts": {"Recon": "x.md"}},
            {"role_prompts": {"recon": "../escape.md"}},
            {"role_prompts": {"recon": "missing.md"}},
        )
        for values in cases:
            with self.subTest(values=values):
                self.definition(**values)
                with self.assertRaises(ValueError):
                    self.store.resolve("program:review")

    def test_role_prompt_file_is_part_of_the_capability_fingerprint(self):
        self.resource(self.program / "review" / "prompt.md", "Review prompt")
        role = self.resource(self.program / "review" / "roles" / "recon.md", "ROLE ONE")
        self.definition(
            prompt_file="review/prompt.md",
            worker_tools=["exec"],
            role_prompts={"recon": "review/roles/recon.md"},
        )
        first = self.store.resolve("program:review")
        role.write_text("ROLE TWO", encoding="utf-8")
        second = self.store.resolve("program:review")
        self.assertNotEqual(first.fingerprint, second.fingerprint)


class RuntimeFieldValidationTests(unittest.TestCase):
    def test_status_file_allows_new_json_path_but_rejects_invalid_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertEqual(
                EnvironmentStore._status_path(base, ".slots.json"),
                (base / ".slots.json").resolve(),
            )
            self.assertIsNone(EnvironmentStore._status_path(base, None))
            for value in (True, "", "\x00", "../escape.json", str(base), "file.md"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    EnvironmentStore._status_path(base, value)

    def test_role_source_defaults_and_strict_schema(self):
        valid = {"server": "redteam", "tool": "redteam_role_prompt"}
        source = EnvironmentStore._role_source(valid)
        self.assertEqual(source["field"], "prompt")
        self.assertEqual(source["cache_seconds"], 60)
        self.assertEqual(source["timeout_seconds"], 10)
        for value in (
            [], {}, {**valid, "server": ""}, {**valid, "field": None},
            {**valid, "cache_seconds": True}, {**valid, "cache_seconds": -1},
            {**valid, "max_bytes": 0}, {**valid, "max_bytes": 1048577},
            {**valid, "timeout_seconds": 11}, {**valid, "unknown": 1},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EnvironmentStore._role_source(value)


if __name__ == "__main__":
    unittest.main()
