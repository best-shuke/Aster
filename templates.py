from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import DEFAULT_TEMPLATES_DIR
from sessions import SessionStore


@dataclass(frozen=True)
class SavedTemplate:
    template_id: str | None
    name: str
    path: Path
    message_count: int
    source: Path | None = None


class TemplateStore:
    def __init__(
        self, sessions: SessionStore, workspace: Path, *, directory: Path | None = None
    ) -> None:
        self.sessions = sessions
        self.workspace = workspace
        self.directory = (directory if directory is not None else DEFAULT_TEMPLATES_DIR).resolve()
        self.local_directory = sessions.active_file.with_name(
            f"{sessions.active_file.stem}.templates"
        ).resolve()
        self.settings_file = self.local_directory / ".settings.json"
        self.diagnostics: list[str] = []
        self._selected_id: str | None = None

    def _path(self, template_id: str) -> Path:
        self.sessions.snapshot_path(template_id)
        for path in self.directory.glob("*.json"):
            try:
                data = self.sessions.read(path)
            except RuntimeError:
                continue
            if data.get("template_id") == template_id:
                return path
        return self.directory / f"{template_id}.json"

    def _new_path(self, name: str, template_id: str) -> Path:
        filename = re.sub(r"[^\w.-]+", "-", name).strip("._-")[:60] or template_id
        path = self.directory / f"{filename}.json"
        if path.exists():
            path = self.directory / f"{filename[:48]}-{template_id}.json"
        return path

    def _migrate_legacy(self) -> None:
        if self.local_directory == self.directory:
            return
        for path in self.local_directory.glob("*.json"):
            if path == self.settings_file:
                continue
            try:
                template = self._info(path, managed=True)
                if self._path(template.template_id).exists():
                    continue
                data = self.sessions.read(path)
                destination = self._new_path(template.name, template.template_id)
                self.sessions._write(destination, json.dumps(data, ensure_ascii=False, indent=2))
            except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
                self.diagnostics.append(f"无法迁移旧模板 {path.name}: {exc}")

    def _info(self, path: Path, *, managed: bool) -> SavedTemplate:
        data = self.sessions.read(path)
        if not data["messages"]:
            raise ValueError("模板对话不能为空。")
        count = sum(message["role"] in {"user", "assistant"} for message in data["messages"])
        if not managed:
            return SavedTemplate(None, path.name.removesuffix(".ds-agent.json"), path, count, path)
        template_id = data["template_id"]
        name = data["name"]
        self.sessions.snapshot_path(template_id)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("invalid template metadata")
        source = data.get("source")
        return SavedTemplate(template_id, name, path, count, Path(source) if source else None)

    def read(self, template: SavedTemplate) -> dict:
        data = self.sessions.read(template.path)
        if not data["messages"]:
            raise ValueError("模板对话不能为空。")
        if template.template_id is not None and data.get("template_id") != template.template_id:
            raise ValueError("template ID does not match file name")
        return data

    def list(self) -> list[SavedTemplate]:
        self.diagnostics.clear()
        self._migrate_legacy()
        templates = []
        for path in sorted(self.directory.glob("*.json")):
            if path == self.settings_file:
                continue
            try:
                templates.append(self._info(path, managed=True))
            except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                self.diagnostics.append(f"已跳过无法读取的模板 {path.name}: {exc}")
        sources = {template.source for template in templates}
        for path in sorted(self.workspace.glob("*.ds-agent.json")):
            if path.resolve() in sources:
                continue
            try:
                templates.append(self._info(path.resolve(), managed=False))
            except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                self.diagnostics.append(f"已跳过无法读取的转换文件 {path.name}: {exc}")
        return sorted(templates, key=lambda template: template.name.casefold())

    def create(
        self, name: str, messages: list[dict], *, source: Path | None = None
    ) -> SavedTemplate:
        name = " ".join(name.split())
        if not name or len(name) > 80:
            raise ValueError("模板名称应为 1 到 80 个字符。")
        if not messages:
            raise ValueError("模板对话不能为空。")
        if any(template.template_id and template.name == name for template in self.list()):
            raise ValueError(f"模板名称已存在：{name}；请使用其他名称。")
        template_id = uuid4().hex
        data = {
            "version": 1,
            "template_id": template_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source.resolve()) if source is not None else None,
            "messages": messages,
        }
        path = self._new_path(name, template_id)
        self.sessions._write(path, json.dumps(data, ensure_ascii=False, indent=2))
        return self._info(path, managed=True)

    def add(self, path: Path, name: str | None = None) -> SavedTemplate:
        path = path.expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        data = self.sessions.read(path)
        if path.parent == self.directory.resolve():
            return self._info(path, managed=True)
        name = name or path.name.removesuffix(".ds-agent.json").removesuffix(".json")
        for template in self.list():
            if template.template_id and template.source == path and template.name == name:
                return template
        return self.create(name, data["messages"], source=path)

    @property
    def selected_id(self) -> str | None:
        return self._selected_id

    def default(self) -> SavedTemplate | None:
        if self._selected_id is None:
            return None
        try:
            self._migrate_legacy()
            return self._info(self._path(self._selected_id), managed=True)
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot load selected template {self._selected_id}: {exc}") from exc

    def set_default(self, template_id: str | None) -> None:
        if template_id is not None:
            self._info(self._path(template_id), managed=True)
        self._selected_id = template_id
