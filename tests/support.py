from pathlib import Path

from config import Settings


def make_settings(**values) -> Settings:
    workspace = Path(values["workspace"])
    if "prompt_file" not in values:
        prompt = workspace / ".test-system-prompt.md"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        if not prompt.exists():
            prompt.write_text("Synthetic test instructions.\n", encoding="utf-8")
        values["prompt_file"] = prompt
    for name, directory in (
        ("skills_dir", "test-skills"),
        ("templates_dir", "templates"),
        ("workflows_dir", "workflows"),
        ("environments_dir", "test-environments"),
        ("environment_state_dir", "test-environment-state"),
    ):
        values.setdefault(name, workspace / directory)
    return Settings(**values)
