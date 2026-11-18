import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auth import AuthStore
from config import Settings, auth_file, default_session_file, load_env, user_dir
from conversation import Conversation
from environment_definitions import EnvironmentEditor
from main import load_settings
from templates import TemplateStore
from tests.test_conversation import FakeClient
from workflows import WorkflowStore


class UserStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home" / ".agents" / "aster"
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for mocked in (
            patch.dict(os.environ, {}, clear=True),
            patch("config.DEFAULT_USER_DIR", self.home),
            patch("config.Path.cwd", return_value=self.workspace),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def env_file(self, content):
        self.home.mkdir(parents=True, exist_ok=True)
        path = self.home / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def settings(self):
        os.environ["DEEPSEEK_API_KEY"] = "synthetic-key"
        return Settings.from_env()

    def test_default_config_is_shared_across_workspaces_and_process_values_win(self):
        self.env_file("DEEPSEEK_API_KEY=user-key\nDEEPSEEK_MODEL=user-model\n")
        (self.workspace / ".env").write_text("DEEPSEEK_API_KEY=workspace-key\n")
        os.environ["DEEPSEEK_MODEL"] = "process-model"
        settings = Settings.from_env()
        self.assertEqual(settings.api_key, "user-key")
        self.assertEqual(settings.model, "process-model")
        other = self.root / "other"
        other.mkdir()
        with patch("config.Path.cwd", return_value=other):
            self.assertEqual(Settings.from_env().api_key, "user-key")
        self.assertEqual(settings.workspace, self.workspace)

    def test_missing_user_config_does_not_load_workspace_env(self):
        (self.workspace / ".env").write_text("DEEPSEEK_API_KEY=workspace-key\n")
        load_env()
        self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
        self.assertFalse(self.home.exists())

    def test_custom_home_and_explicit_paths_take_precedence(self):
        custom = self.root / "custom home"
        custom.mkdir()
        (custom / ".env").write_text("DEEPSEEK_API_KEY=custom-key\n")
        with patch.dict(os.environ, {"ASTER_HOME": str(custom)}):
            settings = Settings.from_env()
            self.assertEqual(settings.api_key, "custom-key")
            self.assertEqual(user_dir(), custom)
            self.assertEqual(settings.auth_file, custom / "auth.json")
            for name, relative in (
                ("skills_dir", "skills"),
                ("templates_dir", "templates"),
                ("workflows_dir", "workflows"),
                ("environments_dir", "environments"),
                ("environment_state_dir", "environment-state"),
                ("mcp_config_file", "mcp.json"),
            ):
                self.assertEqual(getattr(settings, name), custom / relative)
            self.assertTrue(settings.session_file.is_relative_to(custom / "sessions"))
            external = self.root / "external.env"
            external.write_text("DEEPSEEK_API_KEY=explicit-key\n")
            overrides = {
                "DEEPSEEK_ENV_FILE": str(external),
                "DEEPSEEK_AUTH_FILE": str(self.root / "login.json"),
                "DEEPSEEK_SESSION_FILE": "explicit-session.json",
                "DEEPSEEK_MCP_CONFIG_FILE": str(self.root / "mcp.json"),
            }
            os.environ.pop("DEEPSEEK_API_KEY")
            with patch.dict(os.environ, overrides):
                settings = Settings.from_env()
                self.assertEqual(settings.api_key, "explicit-key")
                self.assertEqual(auth_file(), self.root / "login.json")
                self.assertEqual(settings.mcp_config_file, self.root / "mcp.json")
                self.assertEqual(settings.session_file, self.workspace / "explicit-session.json")

    def test_tilde_in_custom_home_and_env_file_is_expanded(self):
        fake_home = self.root / "home"
        with patch.dict(os.environ, {"HOME": str(fake_home)}):
            with patch.dict(os.environ, {"ASTER_HOME": "~/.custom-aster"}):
                self.assertEqual(user_dir(), fake_home / ".custom-aster")
            env_file = fake_home / "custom.env"
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.write_text("DEEPSEEK_API_KEY=tilde-key\n")
            with patch.dict(os.environ, {"DEEPSEEK_ENV_FILE": "~/custom.env"}):
                load_env()
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "tilde-key")

    def test_sessions_are_persistent_and_isolated_by_canonical_workspace(self):
        first = self.settings()
        conversation = Conversation(first, FakeClient(), None)
        self.addCleanup(conversation.close)
        conversation.messages = [{"role": "user", "content": "saved history"}]
        conversation.save()
        session_id = conversation.session_id
        self.assertTrue(first.session_file.exists())
        self.assertTrue(conversation.sessions.snapshot_path(session_id).exists())
        restarted = Conversation(self.settings(), FakeClient(), None)
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.session_id, session_id)
        self.assertEqual(restarted.messages, conversation.messages)
        other = self.root / "other" / self.workspace.name
        other.mkdir(parents=True)
        with patch("config.Path.cwd", return_value=other):
            second = self.settings()
        self.assertNotEqual(first.session_file, second.session_file)
        separate = Conversation(second, FakeClient(), None)
        self.addCleanup(separate.close)
        self.assertEqual(separate.messages, [])
        alias = self.root / "alias"
        alias.symlink_to(self.workspace, target_is_directory=True)
        self.assertEqual(default_session_file(alias), first.session_file)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_templates_workflows_and_environment_state_stay_in_user_home(self):
        settings = self.settings()
        conversation = Conversation(settings, FakeClient(), None)
        self.addCleanup(conversation.close)
        template = TemplateStore(conversation.sessions, self.workspace).create(
            "Synthetic template", [{"role": "user", "content": "example"}]
        )
        self.assertTrue(template.path.is_relative_to(self.home / "templates"))
        workflow = WorkflowStore().create("Synthetic workflow", "Review the supplied text.")
        self.assertTrue(workflow.path.is_relative_to(self.home / "workflows"))
        result = EnvironmentEditor(conversation.environments).create("synthetic")
        self.assertEqual(result.entry.environment_id, "user:synthetic")
        self.assertTrue(result.entry.path.is_relative_to(self.home / "environments"))
        self.assertTrue(conversation.environments.trust_file.is_relative_to(self.home))
        self.assertTrue(conversation.environments.audit_file.exists())
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_noninteractive_start_loads_auth_path_from_user_config_before_login(self):
        auth_path = self.root / "private" / "auth.json"
        AuthStore(auth_path).update(
            provider="api",
            api={"api_key": "saved-key", "base_url": "https://example.invalid", "model": "saved"},
        )
        self.env_file(f"DEEPSEEK_AUTH_FILE={auth_path}\n")
        with patch("sys.stdin.isatty", return_value=False):
            settings = load_settings(choose=False, one_shot=True)
        self.assertIsNotNone(settings)
        self.assertEqual(settings.api_key, "saved-key")
        self.assertEqual(settings.auth_file, auth_path)
        self.assertEqual(json.loads(auth_path.read_text())["provider"], "api")
