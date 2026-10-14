import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from deepseek_client import DeepSeekClient
from environments import EnvironmentStore
from file_state import atomic_write_text
from mcp_client import McpManager, McpServerConfig, load_mcp_config
from reasoning import SubagentRunner
from skills import SkillCatalog
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tests.test_environments import ROOT, EnvironmentTestCase
from tool_runtime import ExecutionContext, ToolResult
from tools import WorkerStatus
from ui import Console
from usage import CacheStats


class EnvironmentReliabilityTests(EnvironmentTestCase):
    def mcp_definition(self, *, required=False, fail=False):
        arguments = (
            ["-c", "raise SystemExit(1)"]
            if fail
            else [str(ROOT / "tests" / "fixtures_mcp_server.py")]
        )
        self.resource(
            self.program / "mcp.json",
            json.dumps(
                {
                    "mcpServers": {
                        "synthetic": {
                            "command": sys.executable,
                            "args": arguments,
                            "timeout": 2,
                            "required": required,
                        }
                    }
                }
            ),
        )
        return "mcp.json"

    def test_local_imports_are_part_of_approval(self):
        self.resource(self.program / "server" / "entry.py", "import helper\n")
        helper = self.resource(self.program / "server" / "helper.py", "VALUE = 1\n")
        self.resource(
            self.program / "mcp.json",
            json.dumps(
                {
                    "mcpServers": {
                        "synthetic": {"command": sys.executable, "args": ["server/entry.py"]}
                    }
                }
            ),
        )
        environment = self.environment(mcp_config_file="mcp.json")
        helper.write_text("VALUE = 2\n", encoding="utf-8")
        changed = self.store.resolve("program:review")
        self.assertNotEqual(environment.fingerprint, changed.fingerprint)
        self.assertFalse(self.store.approved(changed))

    def test_generated_files_do_not_change_skill_approval(self):
        root = self.program / "skills"
        self.resource(
            root / "synthetic" / "SKILL.md",
            "---\nname: synthetic\ndescription: Fixture\n---\nSynthetic body.",
        )
        environment = self.environment(skills_dirs=["skills"])
        for path in ("__pycache__/module.pyc", ".pytest_cache/state", "run.log", "scratch.tmp"):
            self.resource(root / path, "Generated data")
        changed = self.store.resolve("program:review")
        self.assertEqual(environment.fingerprint, changed.fingerprint)
        self.assertTrue(self.store.approved(changed))

    def test_runtime_data_beside_mcp_source_does_not_change_approval(self):
        self.resource(self.program / "server" / "entry.py", "import helper\n")
        self.resource(self.program / "server" / "helper.py", "VALUE = 1\n")
        self.resource(
            self.program / "mcp.json",
            json.dumps(
                {
                    "mcpServers": {
                        "synthetic": {"command": sys.executable, "args": ["server/entry.py"]}
                    }
                }
            ),
        )
        environment = self.environment(mcp_config_file="mcp.json")
        self.resource(self.program / "server" / "session.json", '{"messages": []}')
        self.resource(self.program / "server" / "state.sqlite", "Runtime data")
        self.assertEqual(environment.fingerprint, self.store.resolve("program:review").fingerprint)

    def test_explicit_false_disables_implicit_startup_mcp(self):
        config = self.program / self.mcp_definition()
        self.settings = replace(self.settings, mcp_enabled=True, mcp_config_file=config)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        isolated = self.environment(inherit_mcp=False)
        self.assertFalse(isolated.settings.mcp_enabled)
        self.assertEqual(isolated.servers, {})
        self.definition()
        legacy = self.store.resolve("program:review")
        self.assertTrue(legacy.settings.mcp_enabled)
        self.assertEqual(set(legacy.servers), {"synthetic"})

    def test_startup_snapshot_is_not_reexpanded(self):
        config = self.program / self.mcp_definition()
        self.settings = replace(self.settings, mcp_enabled=True, mcp_config_file=config)
        servers = load_mcp_config(config, self.workspace)
        self.store = EnvironmentStore(self.settings, user_dir=self.user, baseline_servers=servers)
        self.definition(inherit_mcp=True)
        with patch("environments.load_mcp_config", side_effect=AssertionError("reloaded")):
            environment = self.store.resolve("program:review")
        self.assertEqual(environment.servers["synthetic"].command, sys.executable)

    def test_same_environment_preserves_session_history_stats_and_manager(self):
        environment = self.environment(mcp_config_file=self.mcp_definition())
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.messages.append({"role": "user", "content": "preserve"})
        previous = (
            conversation.session_id,
            conversation.mcp_manager,
            conversation.client.cache_stats,
        )
        for keep in (False, True):
            self.assertEqual(conversation.apply_environment(environment, keep=keep), [])
            self.assertEqual(
                previous,
                (
                    conversation.session_id,
                    conversation.mcp_manager,
                    conversation.client.cache_stats,
                ),
            )
            self.assertEqual(conversation.messages[-1]["content"], "preserve")

    def test_same_environment_still_checks_revocation(self):
        environment = self.environment()
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.environments.deny(environment.environment_id)
        with self.assertRaises(ValueError):
            conversation.apply_environment(environment)

    def test_required_connection_failure_does_not_commit(self):
        environment = self.environment(
            mcp_config_file=self.mcp_definition(required=True, fail=True)
        )
        conversation = self.conversation()
        conversation.messages.append({"role": "user", "content": "preserve"})
        conversation.save()
        saved = self.settings.session_file.read_bytes()
        previous = conversation.session_id
        with self.assertRaisesRegex(ValueError, "必需 MCP"):
            conversation.apply_environment(environment)
        self.assertEqual(conversation.session_id, previous)
        self.assertIsNone(conversation.environment_id)
        self.assertEqual(self.settings.session_file.read_bytes(), saved)

    def test_optional_connection_failure_is_degraded(self):
        environment = self.environment(mcp_config_file=self.mcp_definition(fail=True))
        conversation = self.conversation()
        warnings = conversation.apply_environment(environment)
        self.assertTrue(warnings)
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(conversation.environment_health(), "degraded")

    def test_disabled_required_server_does_not_block(self):
        self.resource(
            self.program / "mcp.json",
            json.dumps(
                {
                    "mcpServers": {
                        "synthetic": {
                            "command": "missing-disabled-command",
                            "disabled": True,
                            "required": True,
                        }
                    }
                }
            ),
        )
        environment = self.environment(mcp_config_file="mcp.json")
        conversation = self.conversation()
        conversation.apply_environment(environment)
        self.assertEqual(conversation.environment_health(), "ready")

    def test_role_source_without_fallback_is_required(self):
        environment = self.environment(
            mcp_config_file=self.mcp_definition(fail=True),
            role_source={"server": "synthetic", "tool": "role_prompt"},
        )
        self.assertTrue(environment.servers["synthetic"].required)
        with self.assertRaisesRegex(ValueError, "必需 MCP"):
            self.conversation().apply_environment(environment)

    def test_missing_role_tool_refuses_commit(self):
        environment = self.environment(
            mcp_config_file=self.mcp_definition(),
            role_source={"server": "synthetic", "tool": "missing_role_prompt"},
        )
        conversation = self.conversation()
        with self.assertRaisesRegex(ValueError, "角色"):
            conversation.apply_environment(environment)
        self.assertIsNone(conversation.environment_id)

    def test_messages_only_forks_unsaved_current_history_under_current_policy(self):
        environment = self.environment(tools={"deny": ["exec", "write", "edit"]})
        conversation = self.conversation()
        conversation.apply_environment(environment)
        conversation.messages.append({"role": "user", "content": "unsaved"})
        previous = conversation.session_id
        conversation.resume(previous, messages_only=True)
        self.assertNotEqual(previous, conversation.session_id)
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertFalse(conversation.tools.tool_available("exec"))
        self.assertEqual(conversation.messages[-1]["content"], "unsaved")

    def test_postcommit_status_failure_is_a_warning(self):
        environment = self.environment()
        conversation = self.conversation()
        with patch.object(
            conversation.tools.worker_status, "configure", side_effect=RuntimeError("status")
        ):
            warnings = conversation.apply_environment(environment)
        self.assertTrue(any("环境已提交" in warning for warning in warnings))
        self.assertEqual(conversation.environment_id, environment.environment_id)
        self.assertEqual(
            json.loads(self.settings.session_file.read_text())["environment_id"],
            environment.environment_id,
        )

    def test_skill_refresh_updates_registry_and_worker_catalog(self):
        skill_root = self.settings.skills_dir
        self.resource(
            skill_root / "one" / "SKILL.md",
            "---\nname: one\ndescription: Synthetic\n---\nSynthetic body.",
        )
        servers = {
            "synthetic": McpServerConfig(
                "synthetic",
                command=sys.executable,
                args=(str(ROOT / "tests" / "fixtures_mcp_server.py"),),
                env={"DS_AGENT_SKILL_REGISTRY": ""},
            )
        }
        catalog = SkillCatalog(self.workspace, skill_root)
        manager = McpManager(
            self.program / "unused.json",
            self.workspace,
            servers=servers,
            skill_registry=catalog.mcp_registry(),
        )
        conversation = self.conversation(manager=manager)
        path = Path(manager.servers["synthetic"].env["DS_AGENT_SKILL_REGISTRY"])
        self.resource(
            skill_root / "two" / "SKILL.md",
            "---\nname: two\ndescription: Synthetic\n---\nSynthetic body.",
        )
        conversation.refresh_skills()
        names = {entry["name"] for entry in json.loads(path.read_text())["skills"]}
        self.assertTrue({"one", "two"}.issubset(names))
        self.assertIn("two", conversation.tools.worker_skills)


class McpReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.servers = {
            name: McpServerConfig(
                name,
                command=sys.executable,
                args=(str(ROOT / "tests" / "fixtures_mcp_server.py"),),
                env={"DS_AGENT_SKILL_REGISTRY": ""},
                timeout=3,
            )
            for name in ("one", "two")
        }
        self.registry = {"version": 1, "enabled": True, "skills": []}
        self.old = self.manager(self.servers)
        self.assertEqual(self.old.connect_all().status, "ok")

    def manager(self, servers, *, previous=None, registry=None):
        manager = McpManager(
            self.workspace / "unused.json",
            self.workspace,
            servers=servers,
            skill_registry=registry or self.registry,
            previous=previous,
        )
        self.addCleanup(manager.close)
        return manager

    def assert_echo(self, manager, name):
        result = manager.call(
            "echo", {"text": "alive"}, name, ExecutionContext(time.monotonic() + 5)
        )
        self.assertEqual(result.status, "ok")

    def test_only_changed_connections_are_restarted_and_runtime_survives(self):
        updated = {
            **self.servers,
            "two": replace(
                self.servers["two"], env={**self.servers["two"].env, "SYNTHETIC": "changed"}
            ),
        }
        candidate = self.manager(updated, previous=self.old)
        original = dict(self.old._connections)
        self.assertEqual(candidate.reused_servers, ("one",))
        self.assertEqual(candidate.connect_all().status, "ok")
        self.assertIs(candidate._connections["one"], original["one"])
        self.assertIsNot(candidate._connections["two"], original["two"])
        candidate.stage_skill_registry()
        candidate.commit_reuse()
        self.old.close()
        self.assert_echo(candidate, "one")
        self.assert_echo(candidate, "two")
        self.assertTrue(Path(candidate.servers["one"].env["DS_AGENT_SKILL_REGISTRY"]).exists())

    def test_rollback_restores_registry_and_keeps_old_connections_alive(self):
        path = Path(self.old.servers["one"].env["DS_AGENT_SKILL_REGISTRY"])
        saved = path.read_bytes()
        candidate = self.manager(
            self.servers, previous=self.old, registry={"version": 1, "enabled": False, "skills": []}
        )
        candidate.stage_skill_registry()
        self.assertNotEqual(path.read_bytes(), saved)
        candidate.close()
        self.assertEqual(path.read_bytes(), saved)
        self.assert_echo(self.old, "one")
        self.assert_echo(self.old, "two")

    def test_registry_lifetime_follows_transferred_connections(self):
        path = Path(self.old.servers["one"].env["DS_AGENT_SKILL_REGISTRY"])
        candidate = self.manager(self.servers, previous=self.old)
        self.assertEqual(candidate.reused_servers, ("one", "two"))
        candidate.commit_reuse()
        self.old.close()
        self.assertTrue(path.exists())
        candidate.close()
        self.assertFalse(path.exists())

    def test_required_policy_change_does_not_restart_connection(self):
        candidate = self.manager(
            {name: replace(server, required=True) for name, server in self.servers.items()},
            previous=self.old,
        )
        self.assertEqual(candidate.reused_servers, ("one", "two"))
        self.assertTrue(all(server["required"] for server in candidate.status().data["servers"]))

    def test_failed_multi_registry_update_rolls_back_all_files(self):
        updated = {
            **self.servers,
            "two": replace(
                self.servers["two"], env={**self.servers["two"].env, "SYNTHETIC": "changed"}
            ),
        }
        candidate = self.manager(updated, previous=self.old)
        saved = {
            Path(path): Path(path).read_bytes() for path in candidate._skill_registry_directories
        }
        calls = []

        def write(path, content):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("Synthetic update failure")
            atomic_write_text(path, content)

        with patch("mcp_client.atomic_write_text", side_effect=write), self.assertRaises(OSError):
            candidate.update_skill_registry({"version": 1, "enabled": False, "skills": []})
        self.assertEqual({path: path.read_bytes() for path in saved}, saved)
        self.assert_echo(self.old, "one")

    def test_required_field_rejects_non_boolean_values(self):
        path = self.workspace / "config.json"
        for value in (1, "true", None):
            path.write_text(
                json.dumps(
                    {"mcpServers": {"test": {"command": sys.executable, "required": value}}}
                ),
                encoding="utf-8",
            )
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "required"):
                load_mcp_config(path, self.workspace)

    def test_changed_imported_code_restarts_even_with_identical_configuration(self):
        entry = self.workspace / "entry.py"
        entry.write_text((ROOT / "tests" / "fixtures_mcp_server.py").read_text(), encoding="utf-8")
        helper = self.workspace / "helper.py"
        helper.write_text("VALUE = 1\n", encoding="utf-8")
        servers = {"local": McpServerConfig("local", command=sys.executable, args=(str(entry),))}
        old = self.manager(servers)
        self.assertEqual(old.connect_all().status, "ok")
        helper.write_text("VALUE = 2\n", encoding="utf-8")
        candidate = self.manager(servers, previous=old)
        self.assertEqual(candidate.reused_servers, ())
        self.assertEqual(candidate.connect_all().status, "ok")
        self.assertIsNot(candidate._connections["local"], old._connections["local"])

    def test_runtime_data_changes_do_not_restart_connections(self):
        entry = self.workspace / "entry.py"
        entry.write_text((ROOT / "tests" / "fixtures_mcp_server.py").read_text(), encoding="utf-8")
        servers = {"local": McpServerConfig("local", command=sys.executable, args=(str(entry),))}
        old = self.manager(servers)
        self.assertEqual(old.connect_all().status, "ok")
        (self.workspace / "session.json").write_text("{}", encoding="utf-8")
        candidate = self.manager(servers, previous=old)
        self.assertEqual(candidate.reused_servers, ("local",))


class RuntimeIsolationTests(unittest.TestCase):
    def test_status_files_are_owned_and_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / ".slots.json"
            first, second = WorkerStatus(), WorkerStatus()
            self.addCleanup(first.configure, None, 3)
            self.addCleanup(second.configure, None, 3)
            first.configure(base, 3)
            second.configure(base, 3)
            first_path, second_path = first.path, second.path
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(json.loads(second_path.read_text())["owner"], second.owner)
            first.configure(None, 3)
            self.assertFalse(first_path.exists())
            self.assertTrue(second_path.exists())
            second.stop.set()
            with second.lock:
                second_path.write_text(json.dumps({"pid": os.getpid(), "owner": "foreign"}))
                second.configure(None, 3)
            self.assertTrue(second_path.exists())

    def test_malformed_status_does_not_break_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = WorkerStatus()
            status.configure(Path(temporary) / ".slots.json", 3)
            with status.lock:
                status.path.write_text("[]")
                status.configure(None, 3)
            self.assertIsNone(status.path)

    def test_cache_scopes_are_thread_local_without_double_counting(self):
        stats = CacheStats()
        usage = {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 6}}

        def record(label):
            with stats.request_scope(label):
                for _unused in range(10):
                    stats.record(usage, "solve", system_fingerprint="a" * 64)

        threads = [
            threading.Thread(target=record, args=(label,))
            for label in ("redteam/main", "redteam/recon")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        saved = stats.snapshot()
        self.assertEqual(saved["calls"], 20)
        self.assertEqual(saved["hit_rate"], 0.6)
        self.assertEqual(
            {label: entry["calls"] for label, entry in saved["scopes"].items()},
            {"redteam/main:solve": 10, "redteam/recon:solve": 10},
        )
        restored = CacheStats()
        restored.restore(saved)
        self.assertEqual(restored.snapshot()["scopes"], saved["scopes"])
        restored.reset()
        self.assertNotIn("scopes", restored.snapshot())

    def test_cache_changes_are_hashed_without_retaining_prompt_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = DeepSeekClient(
                make_settings(
                    workspace=Path(temporary),
                    api_key="synthetic",
                    base_url="https://example.invalid",
                    model="synthetic",
                    request_retries=0,
                    session_file=Path(temporary) / "session.json",
                    max_context_chars=120000,
                    keep_recent_turns=6,
                    max_task_seconds=60,
                    max_subagents=1,
                )
            )
            stats = client.cache_stats

            def response(*args, **kwargs):
                return io.BytesIO(
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "done"},
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 10,
                                "prompt_tokens_details": {"cached_tokens": 5},
                            },
                        }
                    ).encode()
                )

            with (
                stats.request_scope("review/main"),
                patch("http_transport.open_url", side_effect=response),
            ):
                for content in ("Synthetic first", "Synthetic first", "Synthetic second"):
                    client.complete([{"role": "system", "content": content}])
        saved = stats.snapshot()
        self.assertEqual(saved["scopes"]["review/main:solve"]["system_changes"], 1)
        self.assertEqual(saved["calls"], 3)
        self.assertNotIn("Synthetic", json.dumps(saved))
        self.assertFalse(hasattr(client, "system_fingerprint"))
        console = Console("synthetic")
        with patch.object(console, "info") as output:
            console.cache(saved)
        self.assertTrue(any("review/main:solve" in call.args[0] for call in output.call_args_list))

    def test_cache_restore_rejects_oversized_scope_totals(self):
        stats = CacheStats()
        with stats.request_scope("main"):
            stats.record(None, "solve")
        saved = stats.snapshot()
        saved["scopes"]["main:solve"]["calls"] = 10
        restored = CacheStats()
        restored.restore(saved)
        self.assertEqual(restored.snapshot()["calls"], 1)
        self.assertNotIn("scopes", restored.snapshot())

    def test_worker_scope_uses_explicit_and_header_roles(self):
        client = FakeClient()
        client.cache_stats = CacheStats()
        runner = SubagentRunner(client, lambda *args: "", [], Path.cwd())
        runner.cache_scope = "program:redteam"

        def complete(*args):
            client.cache_stats.record(None, "solve")
            return ToolResult()

        with patch.object(runner, "_run", side_effect=complete):
            runner.run("task", "", "", ExecutionContext(time.monotonic() + 5), role="recon")
            runner.run("redteamRole: audit\ntask", "", "", ExecutionContext(time.monotonic() + 5))
        self.assertEqual(
            set(client.cache_stats.snapshot()["scopes"]),
            {"program:redteam/recon:solve", "program:redteam/audit:solve"},
        )
