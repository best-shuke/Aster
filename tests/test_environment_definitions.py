import io
import json
import sys
from dataclasses import replace
from unittest.mock import patch

from environment_definitions import EnvironmentEditor, check_credentials, parse_server_options
from environment_spec import UNLIMITED_LIMITS
from environments import EnvironmentStore
from main import handle_command
from mcp_client import McpManager
from mcp_scope import McpScope, ScopedMcp, ToolRule
from tests.test_environments import ROOT, EnvironmentTestCase
from tool_runtime import ExecutionContext
from ui import Console


def skill(path, name, description="Synthetic skill"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody of {name}.", encoding="utf-8"
    )


class EditorTestCase(EnvironmentTestCase):
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
            json.dumps({"mcpServers": {"alpha": server, "beta": server}}),
        )
        self.settings = replace(self.settings, mcp_enabled=True, mcp_config_file=self.baseline_mcp)
        self.store = EnvironmentStore(self.settings, user_dir=self.user)
        skill(self.settings.skills_dir / "research", "research")
        skill(self.settings.skills_dir / "writing", "writing")
        self.editor = EnvironmentEditor(self.store)

    def manager(self):
        manager = McpManager(self.baseline_mcp, self.workspace)
        self.addCleanup(manager.close)
        return manager


class EnvironmentCreationTests(EditorTestCase):
    def test_create_single_and_multi_agent_from_templates(self):
        single = self.editor.create("solo")
        self.assertEqual(single.entry.environment_id, "user:solo")
        self.assertTrue(single.approved)
        self.assertEqual(single.environment.team.kind, "single-agent")
        self.assertEqual(single.environment.skill_selection, {})
        self.assertEqual(single.environment.servers, {})
        team = self.editor.create("crew", template="multi-agent", description="Synthetic crew")
        self.assertTrue(team.environment.multi_agent)
        self.assertEqual(
            [agent.name for agent in team.environment.team.agents],
            [
                "researcher",
                "analyst",
                "planner",
                "executor",
                "writer",
                "editor",
                "fact-checker",
                "critic",
            ],
        )
        self.assertEqual(team.environment.description, "Synthetic crew")
        self.assertTrue(team.environment.team.collaboration.dynamic_agents)
        self.assertTrue(self.store.approved(team.environment))

    def test_create_rejects_duplicates_reserved_words_and_unknown_templates(self):
        self.editor.create("solo")
        for name in ("solo", "status", "off", "../escape", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.editor.create(name)
        with self.assertRaises(ValueError):
            self.editor.create("fresh", template="missing")
        with self.assertRaises(ValueError):
            self.editor.create("fresh", scope="elsewhere")

    def test_workspace_scope_writes_into_workspace_root(self):
        result = self.editor.create("local", scope="workspace")
        self.assertEqual(result.entry.source, "workspace")
        self.assertTrue(result.entry.path.is_relative_to(self.workspace))


class SkillEditingTests(EditorTestCase):
    def test_add_disable_enable_remove_and_reject_unknown(self):
        self.editor.create("solo")
        self.assertEqual(
            {item.name for item in self.editor.available_skills("solo")} >= {"research", "writing"},
            True,
        )
        result = self.editor.edit_skills("solo", "add", ["research", "team-collaboration"])
        self.assertEqual(result.environment.active_skills, {"research", "team-collaboration"})
        catalog = self.store.skill_catalog(result.environment)
        self.assertEqual({item.name for item in catalog.skills}, {"research", "team-collaboration"})
        builtin = catalog.find("team-collaboration")
        self.assertTrue(builtin.file_path.is_relative_to(self.settings.builtin_skills_dir))
        result = self.editor.edit_skills("solo", "disable", ["research"])
        self.assertEqual(result.environment.active_skills, {"team-collaboration"})
        self.assertEqual(result.environment.skill_selection["research"], False)
        result = self.editor.edit_skills("solo", "enable", ["research"])
        self.assertIn("research", result.environment.active_skills)
        result = self.editor.edit_skills("solo", "remove", ["research"])
        self.assertNotIn("research", result.environment.skill_selection)
        with self.assertRaises(ValueError):
            self.editor.edit_skills("solo", "add", ["does-not-exist"])
        with self.assertRaises(ValueError):
            self.editor.edit_skills("solo", "disable", ["writing"])

    def test_user_skill_overrides_builtin_skill_with_same_name(self):
        skill(self.settings.skills_dir / "brainstorming", "brainstorming", "User brainstorming")
        self.editor.create("solo")
        result = self.editor.edit_skills("solo", "add", ["brainstorming"])
        selected = self.store.skill_catalog(result.environment).find("brainstorming")
        self.assertEqual(selected.description, "User brainstorming")

    def test_first_selection_on_legacy_definition_keeps_previously_active_skills(self):
        self.environment("legacy", inherit_skills=True)
        result = self.editor.edit_skills("legacy", "add", ["team-collaboration"])
        self.assertEqual(
            result.environment.active_skills, {"research", "writing", "team-collaboration"}
        )


class McpEditingTests(EditorTestCase):
    def test_reference_global_servers_with_tool_patterns(self):
        self.editor.create("solo")
        self.assertEqual(set(self.editor.available_mcp("solo")), {"alpha", "beta"})
        result = self.editor.edit_mcp("solo", "add", "alpha", tools=["echo"])
        environment = result.environment
        self.assertEqual(set(environment.servers), {"alpha"})
        self.assertTrue(environment.mcp_scope.permits("alpha", "echo"))
        self.assertFalse(environment.mcp_scope.permits("alpha", "add"))
        result = self.editor.edit_mcp("solo", "disable", "alpha")
        self.assertTrue(result.environment.servers["alpha"].disabled)
        self.assertFalse(result.environment.mcp_scope.permits_server("alpha"))
        result = self.editor.edit_mcp("solo", "enable", "alpha")
        self.assertFalse(result.environment.servers["alpha"].disabled)
        self.assertFalse(result.environment.mcp_scope.permits("alpha", "add"))
        result = self.editor.edit_mcp("solo", "tools", "alpha", tools=None, deny=["echo"])
        self.assertTrue(result.environment.mcp_scope.permits("alpha", "add"))
        self.assertFalse(result.environment.mcp_scope.permits("alpha", "echo"))
        with self.assertRaises(ValueError):
            self.editor.edit_mcp("solo", "add", "missing")
        result = self.editor.edit_mcp("solo", "remove", "alpha")
        self.assertEqual(result.environment.servers, {})

    def test_environment_definition_never_stores_server_credentials(self):
        self.editor.create("solo")
        self.editor.edit_mcp("solo", "add", "alpha")
        content = (self.user / "solo.json").read_text(encoding="utf-8")
        self.assertNotIn("command", content)
        self.assertNotIn(sys.executable, content)

    def test_local_server_requires_variable_references_for_secrets(self):
        self.editor.create("solo")
        with self.assertRaises(ValueError):
            self.editor.define_mcp_server(
                "solo", "remote", {"url": "https://example.invalid", "headers": {"X-Key": "abc"}}
            )
        with self.assertRaises(ValueError):
            check_credentials("s", {"command": "x", "env": {"API_TOKEN": "literal"}})
        with self.assertRaises(ValueError):
            check_credentials("s", {"command": "x", "args": ["--token", "literal"]})
        with self.assertRaises(ValueError):
            check_credentials("s", {"url": "https://user:pass@example.invalid"})
        check_credentials("s", {"command": "x", "env": {"API_TOKEN": "${TOKEN}", "LEVEL": "debug"}})
        server = parse_server_options(
            ["--command", f"{sys.executable} {ROOT / 'tests' / 'fixtures_mcp_server.py'}"]
        )
        result = self.editor.define_mcp_server("solo", "local", server)
        self.assertIn("local", result.environment.servers)
        self.assertIn("alpha", self.editor.available_mcp("solo"))
        path = self.user / "solo" / "mcp.json"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_local_server_definition_rolls_back_when_invalid(self):
        self.editor.create("solo")
        with self.assertRaises(ValueError):
            self.editor.define_mcp_server("solo", "bad", {"url": "ftp://example.invalid"})
        self.assertFalse((self.user / "solo" / "mcp.json").exists())
        self.assertNotIn("mcp_config_file", json.loads((self.user / "solo.json").read_text()))


class LimitEditingTests(EditorTestCase):
    def test_unlimited_support_still_rejects_invalid_numbers_and_types(self):
        result = self.editor.create("crew", template="multi-agent")
        definition = json.loads(result.entry.path.read_text())
        for key, value in (
            ("token_budget_per_agent", True),
            ("token_budget_per_agent", -1),
            ("token_budget_per_agent", 1.5),
            ("token_budget_per_agent", "forever"),
            ("background_seconds", -1),
            ("max_concurrent_agents", None),
        ):
            candidate = {**definition, "limits": {key: value}}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.store.resolve_data(result.entry, candidate)

    def test_unlimited_is_persisted_and_distinct_from_resetting_a_field(self):
        self.editor.create("crew", template="multi-agent")
        result = self.editor.set_limits(
            "crew", ["token_budget_per_agent=250000", "background_seconds=0"]
        )
        self.assertEqual(result.environment.team.limits.token_budget_per_agent, 250000)
        result = self.editor.set_limits("crew", ["token_budget_per_agent=unlimited"])
        self.assertIsNone(result.environment.team.limits.token_budget_per_agent)
        data = json.loads(result.entry.path.read_text())
        self.assertIn("token_budget_per_agent", data["limits"])
        self.assertIsNone(data["limits"]["token_budget_per_agent"])
        self.assertEqual(data["limits"]["background_seconds"], 0)
        result = self.editor.set_limits("crew", ["token_budget_per_agent=default"])
        self.assertNotIn("token_budget_per_agent", json.loads(result.entry.path.read_text())["limits"])
        self.assertIsNone(result.environment.team.limits.token_budget_per_agent)
        self.assertTrue(result.approved)

    def test_json_accepts_unlimited_budgets_but_preserves_bounded_capacity(self):
        for index, value in enumerate((None, "unlimited", "none")):
            with self.subTest(value=value):
                environment = self.environment(
                    f"unlimited-{index}", limits=dict.fromkeys(UNLIMITED_LIMITS, value)
                )
                for name in UNLIMITED_LIMITS:
                    self.assertIsNone(getattr(environment.team.limits, name))
        self.editor.create("crew", template="multi-agent")
        before = (self.user / "crew.json").read_text()
        for assignment in ("max_concurrent_agents=unlimited", "max_agents=unlimited"):
            with (
                self.subTest(assignment=assignment),
                self.assertRaisesRegex(ValueError, "不能设为 unlimited"),
            ):
                self.editor.set_limits("crew", [assignment])
        self.assertEqual((self.user / "crew.json").read_text(), before)

    def test_keywords_are_case_insensitive(self):
        self.editor.create("crew", template="multi-agent")
        result = self.editor.set_limits("crew", ["token_budget_per_agent=Unlimited"])
        limits = json.loads(result.entry.path.read_text())["limits"]
        self.assertIsNone(limits["token_budget_per_agent"])
        result = self.editor.set_limits("crew", ["token_budget_per_agent=DEFAULT"])
        limits = json.loads(result.entry.path.read_text()).get("limits", {})
        self.assertNotIn("token_budget_per_agent", limits)


class TrustCarryOverTests(EditorTestCase):
    def test_edit_keeps_trust_only_for_previously_trusted_environments(self):
        self.editor.create("solo")
        result = self.editor.edit_skills("solo", "add", ["research"])
        self.assertTrue(result.approved)
        self.assertTrue(self.store.approved(result.environment))
        # A resource changes outside the editor: the environment is no longer trusted.
        (self.settings.skills_dir / "research" / "SKILL.md").write_text(
            "---\nname: research\ndescription: Changed elsewhere\n---\nNew body.", encoding="utf-8"
        )
        self.assertFalse(self.store.approved(self.store.resolve("solo")))
        result = self.editor.edit_skills("solo", "add", ["writing"])
        self.assertFalse(result.approved)
        self.assertFalse(self.store.approved(result.environment))

    def test_invalid_edit_leaves_definition_unchanged(self):
        self.editor.create("crew", template="multi-agent")
        before = (self.user / "crew.json").read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            self.editor.edit_agent("crew", "add", "main", {"description": "reserved"})
        with self.assertRaises(ValueError):
            self.editor.edit_agent("crew", "add", "helper", {"tools": ["ask_user"]})
        with self.assertRaises(ValueError):
            self.editor.set_limits("crew", ["max_concurrent_agents=99"])
        self.assertEqual((self.user / "crew.json").read_text(encoding="utf-8"), before)


class EnvironmentIsolationTests(EditorTestCase):
    """Acceptance test 2: two environments load only their own capabilities."""

    def test_switching_loads_each_environments_skills_and_mcp_without_leaking(self):
        self.editor.create("left")
        self.editor.edit_skills("left", "add", ["research"])
        self.editor.edit_mcp("left", "add", "alpha", tools=["echo"])
        self.editor.create("right")
        self.editor.edit_skills("right", "add", ["writing", "brainstorming"])
        self.editor.edit_mcp("right", "add", "beta")
        conversation = self.conversation(manager=self.manager())
        baseline_skills = {item.name for item in conversation.skills.skills}

        conversation.apply_environment(self.store.resolve("left"))
        self.assertEqual({item.name for item in conversation.skills.skills}, {"research"})
        self.assertEqual(set(conversation.mcp_manager.servers), {"alpha"})
        handler = conversation.tools.registry["mcp"].handler.__self__
        self.assertIsInstance(handler, ScopedMcp)
        context = ExecutionContext(10**12)
        echoed = handler.execute(
            "call", tool="alpha/echo", arguments={"text": "hi"}, _context=context
        )
        self.assertEqual(echoed.status, "ok")
        denied = handler.execute(
            "call", tool="alpha/add", arguments={"left": 1, "right": 2}, _context=context
        )
        self.assertEqual(denied.error["code"], "permission_denied")
        found = handler.execute("search", query="*", _context=context)
        self.assertEqual([item["path"] for item in found.data["matches"]], ["alpha/echo"])
        outside = json.loads(
            conversation.tools.execute(
                {
                    "id": "call-outside",
                    "type": "function",
                    "function": {
                        "name": "mcp",
                        "arguments": json.dumps(
                            {"action": "call", "tool": "beta/echo", "arguments": {"text": "x"}}
                        ),
                    },
                }
            )
        )
        self.assertEqual(outside["status"], "error")
        self.assertIn("Unknown MCP server 'beta'", outside["error"]["message"])

        conversation.apply_environment(self.store.resolve("right"))
        self.assertEqual(
            {item.name for item in conversation.skills.skills}, {"writing", "brainstorming"}
        )
        self.assertEqual(set(conversation.mcp_manager.servers), {"beta"})
        handler = conversation.tools.registry["mcp"].handler.__self__
        added = handler.execute(
            "call", tool="beta/add", arguments={"left": 1, "right": 2}, _context=context
        )
        self.assertEqual(added.status, "ok")
        self.assertNotIn("research", conversation.system_message["content"])

        conversation.apply_environment(None)
        self.assertEqual({item.name for item in conversation.skills.skills}, baseline_skills)
        self.assertEqual(set(conversation.mcp_manager.servers), {"alpha", "beta"})
        self.assertIsInstance(conversation.tools.registry["mcp"].handler.__self__, McpManager)

    def test_editing_one_environment_does_not_change_another(self):
        self.editor.create("left")
        self.editor.create("right")
        before = self.store.resolve("right").fingerprint
        self.editor.edit_skills("left", "add", ["research"])
        self.editor.edit_mcp("left", "add", "alpha")
        self.assertEqual(self.store.resolve("right").fingerprint, before)
        self.assertEqual(self.store.resolve("right").skill_selection, {})


class LegacyCompatibilityTests(EditorTestCase):
    def test_definition_without_new_fields_resolves_as_before(self):
        environment = self.environment("legacy", inherit_skills=True)
        self.assertIsNone(environment.skill_selection)
        self.assertIsNone(environment.mcp_selection)
        self.assertIsNone(environment.mcp_scope)
        self.assertEqual(environment.team.kind, "single-agent")
        self.assertEqual(set(environment.servers), {"alpha", "beta"})
        conversation = self.conversation(manager=self.manager())
        conversation.apply_environment(environment)
        self.assertEqual(
            {item.name for item in conversation.skills.skills}, {"research", "writing"}
        )
        self.assertIsInstance(conversation.tools.registry["mcp"].handler.__self__, McpManager)
        self.assertNotIn("team", conversation.tools.registry)

    def test_single_agent_definition_rejects_members(self):
        with self.assertRaises(ValueError):
            self.environment("bad", agents={"helper": {"description": "x"}})


class McpScopeTests(EditorTestCase):
    def test_intersection_requires_both_environment_and_agent_permission(self):
        environment = McpScope({"alpha": (ToolRule(("echo", "add")),), "beta": (ToolRule(),)})
        agent = McpScope({"alpha": (ToolRule(("e*",)),), "gamma": (ToolRule(),)})
        effective = environment.intersect(agent)
        self.assertTrue(effective.permits("alpha", "echo"))
        self.assertFalse(effective.permits("alpha", "add"))
        self.assertFalse(effective.permits_server("beta"))
        self.assertFalse(effective.permits_server("gamma"))
        self.assertTrue(McpScope().intersect(agent).permits("gamma", "anything"))
        self.assertIs(environment.intersect(None), environment)


class EnvironmentCommandTests(EditorTestCase):
    def test_full_command_workflow_create_use_configure_status_exit(self):
        conversation = self.conversation(manager=self.manager())
        console = Console()
        console.interactive = False
        output = io.StringIO()
        commands = [
            "/env create crew --template multi-agent",
            "/env use crew",
            "/env skills add research",
            "/env mcp add alpha --tools echo",
            "/env agents add statistician --description 'Synthetic statistician' --mcp alpha:echo",
            "/env limits max_meeting_rounds=2",
            "/env limits",
            "/env status",
            "/env exit",
        ]
        with patch("sys.stdout", output), patch("sys.stderr", output):
            for command in commands[:2]:
                handle_command(conversation, console, command)
            self.assertEqual(conversation.environment_id, "user:crew")
            session = conversation.session_id
            for command in commands[2:-1]:
                handle_command(conversation, console, command)
            self.assertEqual(conversation.session_id, session)
            active = conversation.active_environment
            self.assertIn("research", active.active_skills)
            self.assertEqual(set(conversation.mcp_manager.servers), {"alpha"})
            self.assertEqual(active.team.agent("statistician").description, "Synthetic statistician")
            self.assertEqual(active.team.limits.max_meeting_rounds, 2)
            handle_command(conversation, console, commands[-1])
        self.assertIsNone(conversation.environment_id)
        self.assertNotIn("错误", output.getvalue())
        self.assertIn("token_budget_per_agent = unlimited", output.getvalue())

    def test_edit_commands_require_a_target(self):
        conversation = self.conversation()
        console = Console()
        console.interactive = False
        output = io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            handle_command(conversation, console, "/env skills add research")
        self.assertIn("没有启用自定义环境", output.getvalue())

    def test_multiagents_shortcut_creates_then_reuses_default_team(self):
        conversation = self.conversation()
        console = Console()
        console.interactive = False
        output = io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            handle_command(conversation, console, "/multiagents")
            self.assertEqual(conversation.environment_id, "user:multi-agent")
            handle_command(conversation, console, "/env exit")
            handle_command(conversation, console, "/multiagents")
        self.assertEqual(conversation.environment_id, "user:multi-agent")
        self.assertEqual(output.getvalue().count("已创建默认多代理环境"), 1)


class ShippedResourceTests(EditorTestCase):
    """The documented examples and built-in templates must stay valid definitions."""

    def test_documented_examples_resolve(self):
        examples = ROOT / "docs" / "examples"
        for name in ("research", "critical-thinking"):
            skill(self.settings.skills_dir / name, name)
        servers = json.loads(self.baseline_mcp.read_text())
        servers["mcpServers"]["github"] = {"url": "https://github.example.invalid/mcp"}
        self.baseline_mcp.write_text(json.dumps(servers), encoding="utf-8")
        for path in examples.glob("*.json"):
            target = self.program / path.name
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        resources = examples / "research-team"
        (self.program / "research-team").mkdir()
        for path in resources.iterdir():
            (self.program / "research-team" / path.name).write_text(path.read_text())
        with patch.dict("os.environ", {"SEARCH_MCP_TOKEN": "synthetic"}):
            team = self.store.resolve("research-team")
            self.assertTrue(team.multi_agent)
            self.assertEqual(set(team.servers), {"search", "github"})
            self.assertFalse(team.mcp_scope.permits("github", "repo_delete_all"))
            self.assertEqual(team.notes, ())
            solo = self.store.resolve("solo-writer")
        self.assertFalse(solo.policy.permits("exec"))
        self.assertEqual(solo.active_skills, {"writing", "brainstorming"})

    def test_builtin_templates_and_skills_load(self):
        templates = EnvironmentEditor(self.store).templates()
        self.assertEqual(set(templates), {"single-agent", "multi-agent", "coding-team"})
        result = self.editor.create("crew", template="multi-agent")
        catalog = self.store.skill_catalog(result.environment)
        self.assertEqual(catalog.diagnostics, [])
        self.assertEqual(
            {item.name for item in catalog.skills},
            {
                "team-collaboration",
                "task-delegation",
                "independent-debate",
                "brainstorming",
                "conflict-resolution",
                "team-retrospective",
            },
        )
