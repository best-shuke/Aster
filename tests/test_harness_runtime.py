import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import Settings
from conversation import Conversation
from environments import EnvironmentEntry, EnvironmentStore, ResolvedEnvironment
from mcp_client import McpServerConfig
from reasoning import RolePromptCache
from tool_runtime import ExecutionContext, ToolResult
from tools import CodingTools, ToolPolicy, WorkerStatus


class SyntheticClient:
    def __init__(self, callback=None):
        self.callback = callback
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append(messages)
        if self.callback:
            self.callback()
        return {"role": "assistant", "content": "Synthetic report"}


def delegation(role=None):
    item = {"task": "Synthetic assignment", "expected_output": "Synthetic report"}
    if role:
        item["role"] = role
    return {"function": {"name": "delegate", "arguments": json.dumps({"tasks": [item]})}}


class HarnessRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.status = self.root / ".slots.json"
        self.tools = CodingTools(self.root)
        self.addCleanup(self.tools.worker_status.close)

    def read_status(self):
        return json.loads(self.tools.worker_status.path.read_text())

    def test_worker_status_is_one_during_model_and_zero_before_result_returns(self):
        samples = []
        client = SyntheticClient(lambda: samples.append(self.read_status()))
        self.tools.attach_reasoning(client, max_workers=1)
        self.tools.worker_status.configure(self.status, 1)
        result = json.loads(self.tools.execute(delegation()))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(samples[0]["running"], 1)
        self.assertEqual(samples[0]["max"], 1)
        after = self.read_status()
        self.assertEqual(after["running"], 0)
        self.assertEqual(after["started_total"], after["finished_total"])

    def test_failed_thread_start_releases_count_and_both_semaphores(self):
        self.tools.attach_reasoning(SyntheticClient(), max_workers=1)
        self.tools.worker_status.configure(self.status, 1)
        with patch("reasoning.threading.Thread.start", side_effect=RuntimeError("Synthetic failure")):
            result = json.loads(self.tools.execute(delegation()))
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.read_status()["running"], 0)
        reasoning = self.tools.registry["delegate"].handler.__self__
        for semaphore in (reasoning.slots, reasoning.global_slots):
            self.assertTrue(semaphore.acquire(blocking=False))
            semaphore.release()

    def test_parent_timeout_does_not_hide_a_still_running_worker(self):
        release = threading.Event()
        entered = threading.Event()

        def block():
            entered.set()
            release.wait(3)

        self.tools.attach_reasoning(SyntheticClient(block), max_workers=1)
        self.tools.worker_status.configure(self.status, 1)
        try:
            result = json.loads(self.tools.execute(delegation(), max_timeout=0.08))
            self.assertTrue(entered.is_set())
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(self.read_status()["running"], 1)
        finally:
            release.set()
        deadline = time.monotonic() + 2
        while self.read_status()["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.read_status()["running"], 0)

    def test_status_heartbeat_refreshes_even_when_count_does_not_change(self):
        status = WorkerStatus(heartbeat_seconds=0.02)
        self.addCleanup(status.close)
        status.configure(self.status, 3)
        active_path = status.path
        before = json.loads(active_path.read_text())["updated_at"]
        deadline = time.monotonic() + 1
        while json.loads(active_path.read_text())["updated_at"] == before and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(json.loads(active_path.read_text())["updated_at"], before)
        status.close()
        time.sleep(0.04)
        self.assertFalse(active_path.exists())

    def test_status_write_failure_does_not_leak_worker_slots(self):
        self.tools.attach_reasoning(SyntheticClient(), max_workers=1)
        with patch("tools.atomic_write_text", side_effect=PermissionError("Synthetic denial")):
            self.tools.worker_status.configure(self.status, 1)
            result = json.loads(self.tools.execute(delegation()))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.tools.worker_status.running, 0)
        self.assertIsNotNone(self.tools.worker_status.last_error)

    def test_reattaching_reasoning_preserves_shared_status_and_global_limit(self):
        client = SyntheticClient()
        self.tools.attach_reasoning(client, max_workers=1)
        tracker = self.tools.worker_status
        self.tools.attach_reasoning(client, max_workers=3)
        self.assertIs(self.tools.worker_status, tracker)
        self.assertEqual(self.tools._global_worker_limit, 1)

    def test_role_cache_expires_at_sixty_seconds_and_is_separate_per_role(self):
        fetch = Mock(side_effect=["Synthetic first", "Synthetic other", "Synthetic refreshed"])
        cache = RolePromptCache(fetch)
        execution = ExecutionContext(float("inf"))
        with patch("reasoning.time.monotonic", return_value=0):
            self.assertEqual(cache("recon", execution), "Synthetic first")
            self.assertEqual(cache("assess", execution), "Synthetic other")
        with patch("reasoning.time.monotonic", return_value=59):
            self.assertEqual(cache("recon", execution), "Synthetic first")
        with patch("reasoning.time.monotonic", return_value=60):
            self.assertEqual(cache("recon", execution), "Synthetic refreshed")
        self.assertEqual(fetch.call_count, 3)

    def test_role_cache_serializes_concurrent_fetches(self):
        calls = []
        reports = []

        def fetch(role, execution):
            calls.append(role)
            time.sleep(0.02)
            return "Synthetic cached role"

        cache = RolePromptCache(fetch)
        threads = [threading.Thread(target=lambda: reports.append(cache("recon", ExecutionContext(time.monotonic() + 2)))) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(calls, ["recon"])
        self.assertEqual(reports, ["Synthetic cached role"] * 4)

    def test_role_cache_rejects_empty_nontext_and_oversized_utf8_content(self):
        for text in ("", "  ", None, "中文"):
            with self.subTest(text=text):
                cache = RolePromptCache(lambda role, execution: text, max_bytes=4)
                with self.assertRaises(ValueError):
                    cache("recon", ExecutionContext(time.monotonic() + 1))
                self.assertEqual(cache.cache, {})

    def test_role_cache_lock_wait_obeys_fetch_timeout(self):
        cache = RolePromptCache(Mock(), timeout_seconds=0.02)
        cache.lock.acquire()
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                cache("recon", ExecutionContext(started + 1))
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            cache.lock.release()

    def test_cancelled_role_fetch_does_not_start_model_or_use_fallback(self):
        client = SyntheticClient()
        execution = ExecutionContext(time.monotonic() + 1)

        def fetch(role, request):
            request.cancelled.set()
            return "Synthetic role"

        self.tools.worker_roles = {"recon": "Synthetic fallback"}
        self.tools.role_loader = RolePromptCache(fetch)
        self.tools.attach_reasoning(client, max_workers=1)
        runner = self.tools.registry["delegate"].handler.__self__.runner
        with self.assertRaises(TimeoutError):
            runner.run("Synthetic assignment", "", "Synthetic report", execution, role="recon")
        self.assertEqual(client.requests, [])

    def test_switching_status_path_preserves_active_count_and_removes_old_file(self):
        tracker = self.tools.worker_status
        tracker.configure(self.status, 3)
        old_path = tracker.path
        tracker.started()
        next_path = self.root / "next.json"
        tracker.configure(next_path, 1)
        active_path = tracker.path
        self.assertFalse(old_path.exists())
        self.assertEqual(json.loads(active_path.read_text())["running"], 1)
        tracker.close()
        self.assertEqual(tracker.running, 1)
        tracker.finished()
        self.assertEqual(tracker.running, 0)
        self.assertFalse(active_path.exists())


class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.settings = Settings(
            api_key="test", base_url="https://example.invalid", model="test",
            session_file=self.root / "session.json", workspace=self.root,
            max_context_chars=120000, keep_recent_turns=6, max_task_seconds=3600,
            max_subagents=3, prompt_file=self.root / "never-read.md",
            skills_enabled=False, websearch_enabled=False, mcp_enabled=False,
            skills_dir=self.root / "skills", templates_dir=self.root / "templates",
            workflows_dir=self.root / "workflows", environments_dir=self.root / "environments",
            environment_state_dir=self.root / "environment-state",
        )
        self.client = SyntheticClient()
        self.conversation = Conversation(self.settings, self.client, CodingTools(self.root), system_instructions="Synthetic lead")
        self.addCleanup(self.conversation.close)
        self.conversation.environments.require_approved = Mock()
        self.conversation.environments.verify = Mock(side_effect=lambda environment: environment)

    def environment(self, **changes):
        return ResolvedEnvironment(
            entry=EnvironmentEntry("synthetic", "program", self.root / "synthetic.json"),
            description="Synthetic environment", fingerprint="synthetic", settings=self.settings,
            prompt_text="Synthetic environment instructions", skills_dirs=(), policy=ToolPolicy(),
            servers={}, worker_tools=frozenset(), role_prompts={}, template_data=None,
            template_id=None, template_path=None, status_file=self.root / ".slots.json", **changes,
        )

    def test_off_removes_status_file_and_dynamic_role_loader(self):
        environment = self.environment(role_source=EnvironmentStore._role_source({"server": "redteam", "tool": "redteam_role_prompt"}))
        environment = replace(environment, settings=replace(self.settings, mcp_enabled=True),
                              servers={"redteam": McpServerConfig("redteam", command="synthetic")})
        manager = Mock(server_count=1)
        manager.connect_all.return_value = ToolResult(data={"failures": []})
        with patch("environment_transition.McpManager", return_value=manager):
            self.conversation.apply_environment(environment)
        tracker = self.conversation.tools.worker_status
        active_path = tracker.path
        self.assertTrue(active_path.exists())
        self.assertIsNotNone(self.conversation.tools.role_loader)
        self.conversation.apply_environment(None)
        self.assertFalse(active_path.exists())
        self.assertIs(self.conversation.tools.worker_status, tracker)
        self.assertIsNone(self.conversation.tools.role_loader)

    def test_switch_failure_preserves_old_status_and_role_loader(self):
        environment = self.environment()
        self.conversation.apply_environment(environment)
        active_path = self.conversation.tools.worker_status.path
        before = active_path.read_text()
        old_loader = self.conversation.tools.role_loader
        candidate = replace(environment, status_file=self.root / "candidate.json")
        with patch.object(self.conversation.sessions, "transition", side_effect=RuntimeError("Synthetic rollback")), self.assertRaises(RuntimeError):
            self.conversation.apply_environment(candidate)
        self.assertEqual(active_path.read_text(), before)
        self.assertFalse(candidate.status_file.exists())
        self.assertIs(self.conversation.tools.role_loader, old_loader)

    def test_effective_status_limit_cannot_exceed_existing_global_pool(self):
        self.conversation.tools._global_worker_limit = 1
        environment = self.environment()
        self.conversation.apply_environment(environment)
        self.assertEqual(json.loads(self.conversation.tools.worker_status.path.read_text())["max"], 1)

    def test_mcp_role_adapter_handles_structured_and_text_json_responses(self):
        for index, result in enumerate((
            {"structuredContent": {"ok": True, "prompt": "Synthetic runtime role"}},
            {"content": [{"type": "text", "text": json.dumps({"ok": True, "prompt": "Synthetic runtime role"})}]},
        )):
            with self.subTest(result=result):
                source = EnvironmentStore._role_source({"server": "redteam", "tool": "redteam_role_prompt"})
                environment = replace(self.environment(role_source=source), fingerprint=f"synthetic-{index}", settings=replace(self.settings, mcp_enabled=True), servers={"redteam": McpServerConfig("redteam", command="synthetic")})
                manager = Mock(server_count=1)
                manager.connect_all.return_value = ToolResult(data={"failures": []})
                manager.call.return_value = ToolResult(data={"result": result})
                with patch("environment_transition.McpManager", return_value=manager):
                    self.conversation.apply_environment(environment)
                output = json.loads(self.conversation.tools.execute(delegation("recon")))
                self.assertEqual(output["status"], "ok", output)
                self.assertEqual(output["data"]["reports"][0]["data"]["role_source"], "dynamic")
                self.assertIn("Synthetic runtime role", self.client.requests[-1][0]["content"])
                self.conversation.set_reasoning_effort("low")
                self.conversation.tools.execute(delegation("recon"))
                self.assertEqual(manager.call.call_count, 1)

    def test_environment_summary_explains_runtime_trust_boundary(self):
        environment = self.environment(role_source=EnvironmentStore._role_source({"server": "redteam", "tool": "redteam_role_prompt"}))
        summary = EnvironmentStore.summary(environment)
        self.assertIn("从资料库活取", summary)
        self.assertIn("运行时内容不纳入指纹", summary)
        self.assertIn(str(environment.status_file), summary)

    def resolve_synthetic(self, data, servers):
        store = EnvironmentStore(self.settings, user_dir=self.root / "user")
        entry = EnvironmentEntry("synthetic", "program", self.root / "environments" / "synthetic.json")

        def resource_path(base, value, **kwargs):
            raw = Path(value)
            return (raw if raw.is_absolute() else base / raw).resolve()

        original_stat = Path.stat

        def resource_stat(path, **kwargs):
            if path in {entry.path, self.settings.prompt_file}:
                return SimpleNamespace(st_size=0, st_mode=0o100644)
            return original_stat(path, **kwargs)

        with (
            patch.object(store, "find", return_value=entry),
            patch("environments._read_json", return_value={"version": 1, "name": "synthetic", **data}),
            patch.object(store, "_path", side_effect=resource_path),
            patch.object(Path, "read_text", return_value="Synthetic instructions"),
            patch.object(Path, "read_bytes", return_value=b"synthetic-resource"),
            patch.object(Path, "stat", autospec=True, side_effect=resource_stat),
            patch("environments.cached_file_digest", return_value="synthetic-digest"),
            patch("environments.load_mcp_config", return_value=servers),
        ):
            return store.resolve("program:synthetic")

    def test_resolver_wires_new_fields_and_fingerprints_only_configuration(self):
        source = {"server": "redteam", "tool": "redteam_role_prompt"}
        data = {"role_source": source, "status_file": str(self.root / ".slots.json"), "mcp_config_file": "bridge.json"}
        servers = {"redteam": McpServerConfig("redteam", url="https://example.invalid", transport="streamable-http")}
        first = self.resolve_synthetic(data, servers)
        self.assertEqual(first.status_file, (self.root / ".slots.json").resolve())
        self.assertEqual(first.role_source["cache_seconds"], 60)
        first.status_file.write_text(json.dumps({"running": 3}))
        second = self.resolve_synthetic(data, servers)
        self.assertEqual(first.fingerprint, second.fingerprint)
        changed = self.resolve_synthetic({**data, "role_source": {**source, "cache_seconds": 30}}, servers)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_resolver_rejects_missing_disabled_or_denied_role_provider(self):
        data = {"role_source": {"server": "redteam", "tool": "redteam_role_prompt"}, "mcp_config_file": "bridge.json"}
        enabled = McpServerConfig("redteam", url="https://example.invalid", transport="streamable-http")
        for changes, servers in (
            ({}, {}), ({}, {"redteam": replace(enabled, disabled=True)}),
            ({"tools": {"deny": ["mcp"]}}, {"redteam": enabled}),
        ):
            with self.subTest(changes=changes, servers=servers), self.assertRaises(ValueError):
                self.resolve_synthetic({**data, **changes}, servers)

    def test_resolver_rejects_status_file_overwriting_capability_definition(self):
        data = {"status_file": str(self.root / "environments" / "synthetic.json")}
        with self.assertRaises(ValueError):
            self.resolve_synthetic(data, {})

    def test_failed_mcp_payload_falls_back_without_caching_failure(self):
        source = EnvironmentStore._role_source({"server": "redteam", "tool": "redteam_role_prompt"})
        environment = replace(
            self.environment(role_source=source), settings=replace(self.settings, mcp_enabled=True),
            servers={"redteam": McpServerConfig("redteam", command="synthetic")},
            role_prompts={"recon": "Synthetic static role"},
        )
        manager = Mock(server_count=1)
        manager.connect_all.return_value = ToolResult(data={"failures": []})
        manager.call.side_effect = [
            ToolResult(data={"result": {"structuredContent": {"ok": False}}}),
            ToolResult(data={"result": {"structuredContent": {"ok": True, "prompt": "Synthetic updated role"}}}),
        ]
        with patch("environment_transition.McpManager", return_value=manager):
            self.conversation.apply_environment(environment)
        first = json.loads(self.conversation.tools.execute(delegation("recon")))
        second = json.loads(self.conversation.tools.execute(delegation("recon")))
        self.assertEqual(first["data"]["reports"][0]["data"]["role_source"], "static-fallback")
        self.assertEqual(second["data"]["reports"][0]["data"]["role_source"], "dynamic")
        self.assertEqual(manager.call.call_count, 2)
