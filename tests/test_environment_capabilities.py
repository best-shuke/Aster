import io
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from environments import EnvironmentStore
from main import sync_environment_ui
from mcp_client import McpManager, McpServerConfig
from skills import SkillCatalog
from tests.test_environments import ROOT, EnvironmentTestCase
from ui import Console


class EnvironmentInheritanceTests(EnvironmentTestCase):
    def setUp(self):
        super().setUp()
        home = patch("pathlib.Path.home", return_value=self.root)
        home.start()
        self.addCleanup(home.stop)
        server = {
            "command": sys.executable,
            "args": [str(ROOT / "tests" / "fixtures_mcp_server.py")],
            "timeout": 10,
        }
        self.baseline_mcp = self.resource(
            self.root / "baseline-mcp.json",
            json.dumps({"mcpServers": {"baseline": server, "shared": server}}),
        )
        self.environment_mcp = self.resource(
            self.program / "extra-mcp.json",
            json.dumps({"mcpServers": {"extra": server, "shared": server}}),
        )
        self.settings = replace(self.settings, mcp_enabled=True, mcp_config_file=self.baseline_mcp)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        self.baseline_skill = self.resource(
            self.settings.skills_dir / "baseline" / "SKILL.md",
            "---\nname: baseline\ndescription: Synthetic baseline\n---\nSynthetic body.",
        )

    def test_mcp_merge_retains_baseline_and_environment_overrides_collisions(self):
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        self.assertEqual(set(environment.servers), {"baseline", "shared", "extra"})
        self.assertEqual(environment.servers["baseline"].cwd, self.workspace)
        self.assertEqual(environment.servers["shared"].cwd, self.program)
        self.assertEqual(environment.servers["extra"].cwd, self.program)

    def test_inherited_services_connect_and_are_reported_after_switch(self):
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        conversation = self.conversation()
        conversation.apply_environment(environment)
        servers = conversation.mcp_manager.status().data["servers"]
        self.assertEqual({server["name"] for server in servers}, {"baseline", "shared", "extra"})
        self.assertTrue(all(server["status"] == "connected" for server in servers))
        self.assertTrue(all(server["tool_count"] > 0 for server in servers))
        console = Console("synthetic")
        sync_environment_ui(conversation, console)
        self.assertEqual(console.mcp_status, "已连接 3/3")

    def test_explicit_null_disables_all_mcp_even_with_inheritance(self):
        environment = self.environment(mcp_config_file=None, inherit_mcp=True)
        self.assertFalse(environment.settings.mcp_enabled)
        self.assertEqual(environment.servers, {})

    def test_failed_inherited_service_does_not_hide_healthy_environment_services(self):
        data = json.loads(self.baseline_mcp.read_text())
        data["mcpServers"]["baseline"]["command"] = "/nonexistent-mcp-binary"
        self.baseline_mcp.write_text(json.dumps(data))
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        conversation = self.conversation()
        warnings = conversation.apply_environment(environment)
        self.assertTrue(any("baseline" in warning for warning in warnings))
        states = {
            server["name"]: server["status"]
            for server in conversation.mcp_manager.status().data["servers"]
        }
        self.assertEqual(
            states, {"baseline": "failed", "shared": "connected", "extra": "connected"}
        )
        console = Console("synthetic")
        sync_environment_ui(conversation, console)
        self.assertEqual(console.mcp_status, "已连接 2/3 · 1 失败")

    def test_disabled_startup_mcp_is_not_inherited(self):
        self.settings = replace(self.settings, mcp_enabled=False)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        self.assertEqual(set(environment.servers), {"shared", "extra"})

    def test_environment_can_disable_a_same_named_baseline_service(self):
        data = json.loads(self.environment_mcp.read_text())
        data["mcpServers"]["shared"]["disabled"] = True
        self.environment_mcp.write_text(json.dumps(data))
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        self.assertTrue(environment.servers["shared"].disabled)
        self.assertFalse(environment.servers["baseline"].disabled)

    def test_default_environment_behavior_remains_isolated(self):
        environment = self.environment(mcp_config_file="extra-mcp.json")
        self.assertEqual(set(environment.servers), {"shared", "extra"})
        self.assertFalse(environment.inherit_skills)
        self.assertFalse(environment.settings.skills_enabled)

    def test_inherited_mcp_config_changes_require_reapproval(self):
        environment = self.environment(mcp_config_file="extra-mcp.json", inherit_mcp=True)
        self.baseline_mcp.write_text(self.baseline_mcp.read_text() + "\n")
        changed = self.store.resolve("review")
        self.assertNotEqual(environment.fingerprint, changed.fingerprint)
        self.assertFalse(self.store.approved(changed))

    def test_inherited_skills_are_enabled_without_explicit_skill_dirs(self):
        environment = self.environment(mcp_config_file=None, inherit_skills=True)
        self.assertTrue(environment.settings.skills_enabled)
        conversation = self.conversation()
        conversation.apply_environment(environment)
        self.assertIn("baseline", {skill.name for skill in conversation.skills.skills})
        console = Console("synthetic")
        sync_environment_ui(conversation, console)
        self.assertGreaterEqual(console.skill_status["loaded"], 1)

    def test_skill_merge_keeps_default_roots_and_prioritizes_environment(self):
        self.resource(
            self.workspace / ".agent" / "skills" / "workspace" / "SKILL.md",
            "---\nname: workspace\ndescription: Synthetic workspace\n---\nSynthetic body.",
        )
        preferred = self.resource(
            self.program / "skills" / "baseline" / "SKILL.md",
            "---\nname: baseline\ndescription: Synthetic override\n---\nSynthetic body.",
        )
        environment = self.environment(
            mcp_config_file=None, inherit_skills=True, skills_dirs=["skills"]
        )
        conversation = self.conversation()
        conversation.apply_environment(environment)
        skills = {skill.name: skill for skill in conversation.skills.skills}
        self.assertIn("workspace", skills)
        self.assertEqual(skills["baseline"].file_path, preferred)

    def test_inherited_skill_changes_require_reapproval(self):
        environment = self.environment(mcp_config_file=None, inherit_skills=True)
        self.baseline_skill.write_text(self.baseline_skill.read_text() + " changed")
        changed = self.store.resolve("review")
        self.assertNotEqual(environment.fingerprint, changed.fingerprint)
        self.assertFalse(self.store.approved(changed))

    def test_disabled_startup_skills_are_not_enabled_by_inheritance(self):
        self.settings = replace(self.settings, skills_enabled=False)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        environment = self.environment(mcp_config_file=None, inherit_skills=True)
        self.assertFalse(environment.settings.skills_enabled)

    def test_inheritance_flags_require_booleans(self):
        for field, value in (("inherit_mcp", "yes"), ("inherit_skills", 1)):
            with self.subTest(field=field):
                self.definition(**{field: value})
                with self.assertRaisesRegex(ValueError, field):
                    self.store.resolve("review")

    def test_changed_loader_requires_restart_instead_of_loading_new_config(self):
        with (
            patch("environments.LOADED_ENVIRONMENT_MTIME", 0),
            self.assertRaisesRegex(ValueError, "退出并重新启动 agent"),
        ):
            self.store.resolve("review")

    def test_inheritance_preserves_explicit_startup_skills_and_no_default_discovery(self):
        explicit = self.resource(
            self.root / "selected-skills" / "selected" / "SKILL.md",
            "---\nname: selected\ndescription: Explicit startup skill\n---\nSynthetic body.",
        )
        baseline = SkillCatalog(
            self.workspace,
            self.settings.skills_dir,
            extra_dirs=(explicit.parent.parent,),
            include_defaults=False,
        )
        conversation = self.conversation()
        conversation._baseline_skills = baseline
        conversation.environments = EnvironmentStore(
            self.settings, user_dir=self.user, baseline_skills=baseline
        )
        self.definition(mcp_config_file=None, inherit_skills=True)
        environment = conversation.environments.resolve("review")
        conversation.environments.approve(environment)
        conversation.apply_environment(environment)
        self.assertEqual([skill.name for skill in conversation.skills.skills], ["selected"])
        self.assertFalse(conversation.skills.include_defaults)
        explicit.write_text(explicit.read_text() + " Changed.")
        changed = conversation.environments.resolve("review")
        self.assertNotEqual(environment.fingerprint, changed.fingerprint)

    def test_actual_enabled_startup_catalog_can_override_disabled_default_discovery(self):
        settings = replace(self.settings, skills_enabled=False)
        baseline = SkillCatalog(
            self.workspace,
            settings.skills_dir,
            extra_dirs=(self.baseline_skill,),
            include_defaults=False,
        )
        self.store = EnvironmentStore(settings, user_dir=self.user, baseline_skills=baseline)
        environment = self.environment(mcp_config_file=None, inherit_skills=True)
        self.assertTrue(environment.settings.skills_enabled)

    def test_mcp_receives_only_opted_in_skill_registry_and_cleans_it_up(self):
        config = McpServerConfig(
            name="opted-in", command=sys.executable, env={"DS_AGENT_SKILL_REGISTRY": ""}
        )
        ordinary = McpServerConfig(name="ordinary", command=sys.executable)
        registry = {"version": 1, "enabled": True, "skills": []}
        manager = McpManager(
            self.baseline_mcp,
            self.workspace,
            servers={config.name: config, ordinary.name: ordinary},
            skill_registry=registry,
        )
        self.addCleanup(manager.close)
        path = Path(manager.servers[config.name].env["DS_AGENT_SKILL_REGISTRY"])
        self.assertEqual(json.loads(path.read_text()), registry)
        self.assertEqual(config.env["DS_AGENT_SKILL_REGISTRY"], "")
        self.assertIs(manager.servers[ordinary.name], ordinary)
        manager.close()
        self.assertFalse(path.exists())


class CapabilityStatusTests(unittest.TestCase):
    def test_status_displays_skill_counts_and_mcp_failure_details(self):
        console = Console("synthetic")
        console.skill_status = {
            "enabled": True,
            "loaded": 3,
            "automatic": 2,
            "manual": 1,
            "diagnostics": 1,
        }
        console.update_mcp(
            [
                {"name": "healthy", "status": "connected", "tool_count": 4},
                {
                    "name": "broken",
                    "status": "failed",
                    "tool_count": 0,
                    "error": "missing executable",
                },
            ]
        )
        output = io.StringIO()
        with patch("sys.stdout", output):
            console.status(Path("session.json"), 0)
        rendered = output.getvalue()
        for expected in (
            "Skills",
            "已加载 3",
            "自动 2",
            "手动 1",
            "警告 1",
            "healthy",
            "broken",
            "missing executable",
        ):
            self.assertIn(expected, rendered)
