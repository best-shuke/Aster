from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_WORKFLOWS_DIR


@dataclass(frozen=True)
class SavedWorkflow:
    name: str
    path: Path
    instructions: str

    @property
    def description(self) -> str:
        return self.instructions.splitlines()[0].lstrip("# ")[:100]

    def prompt(self, context: str = "") -> str:
        prompt = f"请执行 workflow「{self.name}」，按以下步骤处理本次任务。\n\n{self.instructions}"
        if context.strip():
            prompt += f"\n\n本次任务补充：\n{context.strip()}"
        return prompt


class WorkflowStore:
    def __init__(self, directory: Path = DEFAULT_WORKFLOWS_DIR) -> None:
        self.directory = directory.expanduser().resolve()
        self.diagnostics: list[str] = []

    @staticmethod
    def validate_name(name: str) -> str:
        name = " ".join(name.split())
        if not name or len(name) > 80:
            raise ValueError("Workflow 名称须为 1 到 80 个字符。")
        if (
            name.startswith(".")
            or name.casefold() == "readme"
            or any(character in "/\\\0" or ord(character) < 32 for character in name)
        ):
            raise ValueError("Workflow 名称不能以点开头、包含路径分隔符或使用 README。")
        return name

    def read(self, path: Path) -> SavedWorkflow:
        if path.resolve().parent != self.directory or path.suffix != ".md":
            raise ValueError("Workflow 文件必须位于配置的 workflows 目录中。")
        name = self.validate_name(path.stem)
        instructions = path.read_text(encoding="utf-8").strip()
        if not instructions:
            raise ValueError("Workflow 步骤不能为空。")
        return SavedWorkflow(name, path, instructions)

    def list(self) -> list[SavedWorkflow]:
        self.diagnostics.clear()
        workflows = []
        for path in sorted(self.directory.glob("*.md")):
            if path.stem.casefold() == "readme":
                continue
            try:
                workflows.append(self.read(path))
            except (OSError, RuntimeError, ValueError) as exc:
                self.diagnostics.append(f"已跳过无法读取的 workflow {path.name}: {exc}")
        return sorted(workflows, key=lambda workflow: workflow.name.casefold())

    def create(self, name: str, instructions: str) -> SavedWorkflow:
        name = self.validate_name(name)
        instructions = instructions.strip()
        if not instructions:
            raise ValueError("Workflow 步骤不能为空。")
        for path in self.directory.glob("*.md"):
            if " ".join(path.stem.split()).casefold() == name.casefold():
                raise ValueError(f"Workflow「{name}」已存在；请使用其他名称，或直接编辑原文件。")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}.md"
        with path.open("x", encoding="utf-8") as output:
            output.write(instructions + "\n")
        return SavedWorkflow(name, path, instructions)

    def add(self, source: Path, name: str | None = None) -> SavedWorkflow:
        return self.create(
            source.stem if name is None else name, source.read_text(encoding="utf-8")
        )
