import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
SKILL_COMMAND = re.compile(r"^/skill:([^\s]+)(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    file_path: Path
    base_dir: Path
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillDiagnostic:
    path: Path
    message: str


def _frontmatter(content: str) -> tuple[dict[str, Any], str]:
    lines = content.removeprefix("\ufeff").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")
    closing = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if closing is None:
        raise ValueError("unterminated YAML frontmatter")
    parsed = yaml.safe_load("".join(lines[1:closing])) or {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must be a mapping")
    return parsed, "".join(lines[closing + 1 :]).strip()


def skill_files(root: Path, *, include_root_files: bool = True) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    if not root.is_dir():
        return []
    found: list[Path] = []
    pending = [root]
    visited: set[Path] = set()
    while pending:
        directory = pending.pop()
        try:
            canonical = directory.resolve()
            if canonical in visited:
                continue
            visited.add(canonical)
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except (OSError, RuntimeError):
            continue
        declared = next((entry for entry in entries if entry.name == "SKILL.md"), None)
        if declared is not None and declared.is_file():
            found.append(declared.resolve())
            continue
        if directory == root and include_root_files:
            found.extend(
                entry.resolve()
                for entry in entries
                if entry.is_file() and entry.suffix == ".md" and not entry.name.startswith(".")
            )
        pending.extend(
            reversed(
                [
                    entry
                    for entry in entries
                    if entry.is_dir()
                    and not entry.name.startswith(".")
                    and entry.name != "node_modules"
                ]
            )
        )
    return found


def skill_roots(
    workspace: Path,
    program_skills_dir: Path,
    *,
    extra_dirs: tuple[Path, ...] = (),
    include_defaults: bool = True,
) -> list[Path]:
    workspace = workspace.resolve()
    candidates = [*extra_dirs]
    if include_defaults:
        candidates.append(workspace / ".agent" / "skills")
        ancestor = workspace
        while True:
            candidates.append(ancestor / ".agents" / "skills")
            if (ancestor / ".git").exists() or ancestor.parent == ancestor:
                break
            ancestor = ancestor.parent
        candidates.extend(
            [
                workspace / ".pi" / "skills",
                program_skills_dir,
                Path.home() / ".agents" / "skills",
                Path.home() / ".pi" / "agent" / "skills",
            ]
        )
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = (workspace / candidate.expanduser()).resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


class SkillCatalog:
    def __init__(
        self,
        workspace: Path,
        program_skills_dir: Path,
        *,
        enabled: bool = True,
        extra_dirs: tuple[Path, ...] = (),
        include_defaults: bool = True,
    ):
        self.workspace = workspace.resolve()
        self.program_skills_dir = program_skills_dir.resolve()
        self.enabled = enabled
        self.extra_dirs = extra_dirs
        self.include_defaults = include_defaults
        self.skills: list[Skill] = []
        self.diagnostics: list[SkillDiagnostic] = []
        self.reload()

    def fork(self) -> "SkillCatalog":
        return SkillCatalog(
            self.workspace,
            self.program_skills_dir,
            enabled=self.enabled,
            extra_dirs=self.extra_dirs,
            include_defaults=self.include_defaults,
        )

    def mcp_registry(self) -> dict[str, Any]:
        return {
            "version": 1,
            "enabled": self.enabled,
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "path": str(skill.file_path),
                    "disable_model_invocation": skill.disable_model_invocation,
                }
                for skill in self.skills
            ],
        }

    @property
    def roots(self) -> list[Path]:
        return skill_roots(
            self.workspace,
            self.program_skills_dir,
            extra_dirs=self.extra_dirs,
            include_defaults=self.include_defaults,
        )

    def reload(self) -> None:
        self.skills = []
        self.diagnostics = []
        if not self.enabled:
            return

        by_name: dict[str, Skill] = {}
        seen_files: set[Path] = set()
        explicit_roots = {
            (self.workspace / path.expanduser()).resolve() for path in self.extra_dirs
        }
        for root in self.roots:
            if root in explicit_roots and not root.exists():
                self.diagnostics.append(SkillDiagnostic(root, "skill path does not exist"))
                continue
            include_root_files = root in explicit_roots or root.parts[-2:] != (".agents", "skills")
            for file_path in skill_files(root, include_root_files=include_root_files):
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)
                skill = self._load(file_path)
                if skill is None:
                    continue
                existing = by_name.get(skill.name)
                if existing is not None:
                    self.diagnostics.append(
                        SkillDiagnostic(
                            file_path,
                            f"skill name {skill.name!r} already loaded from {existing.file_path}",
                        )
                    )
                    continue
                by_name[skill.name] = skill
        self.skills = list(by_name.values())

    def _load(self, file_path: Path) -> Skill | None:
        declared = file_path.name == "SKILL.md"
        try:
            metadata, _ = _frontmatter(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            if declared:
                self.diagnostics.append(SkillDiagnostic(file_path, str(exc)))
            return None

        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            if declared:
                self.diagnostics.append(SkillDiagnostic(file_path, "description is required"))
            return None
        description = description.strip()
        if len(description) > MAX_DESCRIPTION_LENGTH:
            self.diagnostics.append(
                SkillDiagnostic(
                    file_path, f"description exceeds {MAX_DESCRIPTION_LENGTH} characters"
                )
            )

        declared_name = metadata.get("name")
        name = declared_name.strip() if isinstance(declared_name, str) else file_path.parent.name
        errors = []
        if not 1 <= len(name) <= MAX_NAME_LENGTH:
            errors.append(f"name must contain 1-{MAX_NAME_LENGTH} characters")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            errors.append("name should contain lowercase letters, numbers, and single hyphens only")
        for error in errors:
            self.diagnostics.append(SkillDiagnostic(file_path, error))

        return Skill(
            name=name,
            description=description,
            file_path=file_path,
            base_dir=file_path.parent,
            disable_model_invocation=metadata.get("disable-model-invocation") is True,
        )

    def prompt_section(self) -> str:
        visible = [skill for skill in self.skills if not skill.disable_model_invocation]
        if not visible:
            return ""
        lines = [
            "For a matching skill, read its complete file before acting; resolve references from its directory.",
            "<available_skills>",
        ]
        for skill in visible:
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{html.escape(skill.name)}</name>",
                    f"    <description>{html.escape(skill.description)}</description>",
                    f"    <location>{html.escape(str(skill.file_path))}</location>",
                    "  </skill>",
                ]
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    def expand_command(self, prompt: str) -> str:
        if not prompt.lstrip().startswith("/skill:"):
            return prompt
        match = SKILL_COMMAND.fullmatch(prompt.strip())
        if match is None:
            raise ValueError("用法：/skill:名称 <任务描述>；用 /skills 查看可用技能。")
        if not self.enabled:
            raise ValueError("Skills 已关闭")
        name, arguments = match.group(1), (match.group(2) or "").strip()
        skill = next((item for item in self.skills if item.name == name), None)
        if skill is None:
            raise ValueError(f"没有找到 skill：{name}；用 /skills 查看可用技能。")
        try:
            _, body = _frontmatter(skill.file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"无法加载 skill {name}（{skill.file_path}）：{exc}") from exc
        block = (
            f'<skill name="{html.escape(skill.name)}" location="{html.escape(str(skill.file_path))}">\n'
            f"References are relative to {skill.base_dir}.\n\n{body}\n</skill>"
        )
        return f"{block}\n\nUser request:\n{arguments}" if arguments else block
