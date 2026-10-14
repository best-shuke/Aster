import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from config import PROGRAM_DIR


class CliLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "Aster project"
        self.launcher = self.project / "bin" / "agent"
        self.launcher.parent.mkdir(parents=True)
        self.launcher.write_bytes((PROGRAM_DIR / "bin" / "agent").read_bytes())
        self.launcher.chmod(0o755)
        self.python = self.project / ".venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$PWD" "$DEEPSEEK_ENV_FILE" "$@"\n')
        self.python.chmod(0o755)
        self.workspace = self.root / "用户工作区"
        self.workspace.mkdir()
        self.arguments = ["--no-skills", "/template", "名称 带空格", ""]

    def assert_launch(self, command, **environment):
        result = subprocess.run(
            [str(command), *self.arguments],
            cwd=self.workspace,
            env={**os.environ, "DEEPSEEK_ENV_FILE": "/old/.env", **environment},
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                str(self.workspace.resolve()),
                str(self.project.resolve() / ".env"),
                str(self.project.resolve() / "main.py"),
                *self.arguments,
            ],
        )

    def test_direct_launcher_preserves_workspace_and_argument_boundaries(self):
        self.assert_launch(self.launcher)

    def test_installed_absolute_symlink_uses_project_configuration(self):
        command = self.root / "agent"
        command.symlink_to(self.launcher)
        self.assert_launch(command)

    def test_relative_symlink_chain_resolves_to_project(self):
        command = self.root / "agent"
        command.symlink_to("alias")
        (self.root / "alias").symlink_to("Aster project/bin/agent")
        self.assert_launch(command)

    def test_agent_command_is_available_through_path(self):
        command = self.root / "agent"
        command.symlink_to(self.launcher)
        self.assert_launch("agent", PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}")
