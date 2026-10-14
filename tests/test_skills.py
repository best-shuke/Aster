import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from conversation import Conversation
from skills import SkillCatalog
from tests.support import make_settings
from tests.test_conversation import FakeClient


class SkillCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "workspace"
        self.program_skills = self.root / "program-skills"
        self.workspace.mkdir()
        self.program_skills.mkdir()
        home = patch("skills.Path.home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    @staticmethod
    def create_skill(root: Path, name: str, description: str, body: str, extra: str = "") -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n",
            encoding="utf-8",
        )
        return path

    def test_prompt_advertises_metadata_without_loading_body(self):
        path = self.create_skill(
            self.program_skills,
            "pdf-tools",
            "Extract tables from PDF files.",
            "PRIVATE FULL INSTRUCTIONS",
        )

        catalog = SkillCatalog(self.workspace, self.program_skills)
        prompt = catalog.prompt_section()

        self.assertIn("pdf-tools", prompt)
        self.assertIn("Extract tables", prompt)
        self.assertIn(str(path), prompt)
        self.assertNotIn("PRIVATE FULL INSTRUCTIONS", prompt)

    def test_explicit_command_loads_body_and_appends_arguments(self):
        self.create_skill(
            self.program_skills,
            "pdf-tools",
            "Extract tables from PDF files.",
            "Read references/FORMATS.md first.",
        )
        catalog = SkillCatalog(self.workspace, self.program_skills)

        expanded = catalog.expand_command("/skill:pdf-tools extract report.pdf")

        self.assertIn('<skill name="pdf-tools"', expanded)
        self.assertIn("Read references/FORMATS.md first.", expanded)
        self.assertTrue(expanded.endswith("User request:\nextract report.pdf"))

    def test_manual_only_skill_is_hidden_but_can_be_expanded(self):
        self.create_skill(
            self.program_skills,
            "release",
            "Publish a release.",
            "Follow the release checklist.",
            "disable-model-invocation: true\n",
        )
        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertNotIn("release", catalog.prompt_section())
        self.assertIn("release checklist", catalog.expand_command("/skill:release"))

    def test_project_skill_wins_name_collision(self):
        project_root = self.workspace / ".agent" / "skills"
        project = self.create_skill(project_root, "review", "Project review.", "project")
        self.create_skill(self.program_skills, "review", "Global review.", "global")

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual(catalog.skills[0].file_path, project.resolve())
        self.assertTrue(any("already loaded" in item.message for item in catalog.diagnostics))

    def test_missing_description_is_not_loaded(self):
        directory = self.program_skills / "broken"
        directory.mkdir()
        path = directory / "SKILL.md"
        path.write_text("---\nname: broken\n---\nbody", encoding="utf-8")

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual(catalog.skills, [])
        self.assertEqual(catalog.diagnostics[0].path, path.resolve())
        self.assertIn("description", catalog.diagnostics[0].message)

    def test_pi_project_and_global_directories_are_discovered(self):
        project = self.create_skill(
            self.workspace / ".pi" / "skills", "project", "Project.", "body"
        )
        global_skill = self.create_skill(
            self.root / "home" / ".pi" / "agent" / "skills", "global", "Global.", "body"
        )

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual([skill.file_path for skill in catalog.skills], [project, global_skill])

    def test_ancestor_discovery_stops_at_git_directory_or_worktree_file(self):
        self.create_skill(self.root / ".agents" / "skills", "outside", "Outside.", "body")
        for worktree in (False, True):
            with self.subTest(worktree=worktree):
                repository = self.root / f"repository-{worktree}"
                workspace = repository / "packages" / "app"
                workspace.mkdir(parents=True)
                if worktree:
                    (repository / ".git").write_text("gitdir: /unused", encoding="utf-8")
                else:
                    (repository / ".git").mkdir()
                nearby = self.create_skill(
                    workspace / ".agents" / "skills", "review", "Near.", "near"
                )
                self.create_skill(repository / ".agents" / "skills", "review", "Root.", "root")
                inherited = self.create_skill(
                    repository / ".agents" / "skills", "inherited", "Inherited.", "body"
                )

                catalog = SkillCatalog(workspace, self.program_skills)

                self.assertEqual({skill.file_path for skill in catalog.skills}, {nearby, inherited})
                self.assertTrue(
                    any("already loaded" in item.message for item in catalog.diagnostics)
                )

    def test_ancestor_skills_are_inherited_outside_a_git_repository(self):
        inherited = self.create_skill(
            self.root / ".agents" / "skills", "inherited", "Inherited.", "body"
        )

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual([skill.file_path for skill in catalog.skills], [inherited])

    def test_explicit_file_and_relative_directory_load_without_defaults(self):
        self.create_skill(self.program_skills, "default", "Default.", "body")
        explicit = self.create_skill(self.root / "external", "explicit", "Explicit.", "body")
        relative = self.create_skill(self.workspace / "custom", "relative", "Relative.", "body")

        catalog = SkillCatalog(
            self.workspace,
            self.program_skills,
            extra_dirs=(explicit, Path("custom")),
            include_defaults=False,
        )

        self.assertEqual([skill.file_path for skill in catalog.skills], [explicit, relative])
        self.assertIn("body", catalog.expand_command("/skill:explicit"))

    def test_explicit_skill_overrides_defaults_and_duplicate_files_are_deduplicated(self):
        project = self.create_skill(
            self.workspace / ".agent" / "skills", "review", "Project.", "project"
        )
        explicit = self.create_skill(self.root / "external", "review", "Explicit.", "explicit")

        catalog = SkillCatalog(
            self.workspace,
            self.program_skills,
            extra_dirs=(explicit, explicit.parent, project),
        )

        self.assertEqual([skill.file_path for skill in catalog.skills], [explicit])
        self.assertEqual(len(catalog.diagnostics), 1)

    def test_missing_explicit_path_has_a_diagnostic(self):
        missing = self.workspace / "missing"
        catalog = SkillCatalog(
            self.workspace, self.program_skills, extra_dirs=(missing,), include_defaults=False
        )

        self.assertEqual(catalog.skills, [])
        self.assertEqual(catalog.diagnostics[0].path, missing)
        self.assertIn("does not exist", catalog.diagnostics[0].message)

    def test_root_markdown_skills_load_but_ordinary_documentation_is_ignored(self):
        path = self.program_skills / "flat.md"
        path.write_text(
            "---\nname: flat\ndescription: Flat skill.\n---\nInstructions", encoding="utf-8"
        )
        (self.program_skills / "README.md").write_text("# Documentation", encoding="utf-8")
        (self.program_skills / "broken.md").write_text("---\nname: [\n---", encoding="utf-8")

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual([skill.file_path for skill in catalog.skills], [path])
        self.assertEqual(catalog.diagnostics, [])
        self.assertIn("Instructions", catalog.expand_command("/skill:flat"))

    def test_agents_roots_ignore_flat_markdown_unless_explicitly_requested(self):
        directory = self.workspace / ".agents" / "skills"
        nested = self.create_skill(directory, "nested", "Nested.", "body")
        flat = directory / "flat.md"
        flat.write_text("---\nname: flat\ndescription: Flat.\n---\nbody", encoding="utf-8")

        catalog = SkillCatalog(self.workspace, self.program_skills)
        self.assertEqual([skill.file_path for skill in catalog.skills], [nested])

        catalog = SkillCatalog(self.workspace, self.program_skills, extra_dirs=(directory,))
        self.assertEqual({skill.name for skill in catalog.skills}, {"nested", "flat"})

    def test_skill_roots_stop_recursion_and_symlink_cycles_are_safe(self):
        declared = self.create_skill(self.program_skills, "declared", "Declared.", "body")
        self.create_skill(declared.parent, "nested", "Nested.", "body")
        external = self.create_skill(self.root / "external", "linked", "Linked.", "body")
        (self.program_skills / "linked").symlink_to(external.parent, target_is_directory=True)
        (self.program_skills / "cycle").symlink_to(self.program_skills, target_is_directory=True)
        self.create_skill(self.program_skills / ".hidden", "hidden", "Hidden.", "body")
        self.create_skill(self.program_skills / "node_modules", "dependency", "Dependency.", "body")

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual({skill.file_path for skill in catalog.skills}, {declared, external})

    def test_utf8_bom_frontmatter_is_supported(self):
        path = self.create_skill(self.program_skills, "bom", "BOM.", "Instructions")
        path.write_text("\ufeff" + path.read_text(encoding="utf-8"), encoding="utf-8")

        catalog = SkillCatalog(self.workspace, self.program_skills)

        self.assertEqual([skill.name for skill in catalog.skills], ["bom"])
        self.assertIn("Instructions", catalog.expand_command("/skill:bom"))

    def test_reload_picks_up_new_and_removed_skills(self):
        catalog = SkillCatalog(self.workspace, self.program_skills)
        path = self.create_skill(self.program_skills, "new", "New.", "body")
        catalog.reload()
        self.assertEqual([skill.name for skill in catalog.skills], ["new"])

        path.unlink()
        catalog.reload()
        self.assertEqual(catalog.skills, [])

    def test_unknown_malformed_and_disabled_commands_raise_local_errors(self):
        self.create_skill(self.program_skills, "known", "Known.", "body")
        catalog = SkillCatalog(self.workspace, self.program_skills)
        for prompt in ("/skill:missing task", "/skill:", "/skill: task"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                catalog.expand_command(prompt)

        disabled = SkillCatalog(self.workspace, self.program_skills, enabled=False)
        self.assertEqual(disabled.prompt_section(), "")
        with self.assertRaisesRegex(ValueError, "已关闭"):
            disabled.expand_command("/skill:known")
        self.assertEqual(catalog.expand_command("Explain /skill:known"), "Explain /skill:known")

    def test_mcp_registry_contains_metadata_not_skill_content(self):
        path = self.create_skill(self.program_skills, "known", "Known.", "Synthetic private body")
        catalog = SkillCatalog(self.workspace, self.program_skills)
        registry = catalog.mcp_registry()
        self.assertTrue(registry["enabled"])
        self.assertEqual(registry["skills"][0]["path"], str(path.resolve()))
        self.assertNotIn("Synthetic private body", str(registry))

    def test_unreadable_or_changed_skill_does_not_silently_fall_back_to_raw_command(self):
        path = self.create_skill(self.program_skills, "known", "Known.", "body")
        catalog = SkillCatalog(self.workspace, self.program_skills)
        path.write_text("---\nname: [\n---\nbody", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "无法加载 skill known"):
            catalog.expand_command("/skill:known task")
        path.unlink()
        with self.assertRaisesRegex(ValueError, "无法加载 skill known"):
            catalog.expand_command("/skill:known task")


class SkillConversationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.workspace / "session.json",
            max_context_chars=None,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            workspace=self.workspace,
        )

    def test_skill_body_and_arguments_reach_the_model_on_explicit_invocation(self):
        path = SkillCatalogTests.create_skill(
            self.workspace / "skills", "review", "Review.", "Use the checklist."
        )
        catalog = SkillCatalog(
            self.workspace, path.parent, include_defaults=False, extra_dirs=(path,)
        )
        client = FakeClient({"role": "assistant", "content": "Reviewed."})
        conversation = Conversation(self.settings, client, None, skills=catalog)
        self.addCleanup(conversation.close)

        self.assertEqual(conversation.ask("/skill:review inspect main.py"), "Reviewed.")

        user = next(
            message for message in client.requests[0]["messages"] if message["role"] == "user"
        )
        self.assertIn("Use the checklist.", user["content"])
        self.assertIn(str(path.parent), user["content"])
        self.assertTrue(user["content"].endswith("User request:\ninspect main.py"))
        self.assertNotIn("Use the checklist.", conversation.system_message["content"])

    def test_invalid_or_disabled_invocation_does_not_call_the_model_or_append_messages(self):
        catalog = SkillCatalog(self.workspace, self.workspace / "skills", include_defaults=False)
        for skills in (catalog, None):
            with self.subTest(skills=skills):
                client = Mock()
                conversation = Conversation(self.settings, client, None, skills=skills)
                self.addCleanup(conversation.close)
                with self.assertRaises(ValueError):
                    conversation.ask("/skill:missing task")
                client.complete.assert_not_called()
                self.assertEqual(conversation.messages, [])


if __name__ == "__main__":
    unittest.main()
