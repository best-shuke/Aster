import os
import platform
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

APP_NAME = "Aster"
CLI_NAME = "agent"
PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = PROGRAM_DIR / "system_prompt.md"
DEFAULT_SKILLS_DIR = PROGRAM_DIR / "skills"
DEFAULT_TEMPLATES_DIR = PROGRAM_DIR / "templates"
DEFAULT_WORKFLOWS_DIR = PROGRAM_DIR / "workflows"
DEFAULT_ENVIRONMENTS_DIR = PROGRAM_DIR / "environments"
DEFAULT_ENVIRONMENT_STATE_DIR = Path.home() / ".agents" / "environment-state"
DEFAULT_MCP_CONFIG_FILE = PROGRAM_DIR / "mcp.json"
# Framework-provided resources: collaboration skills and environment templates.
BUILTIN_DIR = PROGRAM_DIR / "builtin"
BUILTIN_SKILLS_DIR = BUILTIN_DIR / "skills"
BUILTIN_TEMPLATES_DIR = BUILTIN_DIR / "environment-templates"
DEFAULT_REASONING_EFFORT = "max"
MAX_SUBAGENTS = 20
REASONING_EFFORTS = ("default", "off", "low", "high", "max")
TASK_CONTEXT_PREFIX = "Current task environment:\n"


def parse_reasoning_effort(value: str | None) -> str:
    effort = (value or DEFAULT_REASONING_EFFORT).strip().lower() or DEFAULT_REASONING_EFFORT
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"无效的推理强度 {value!r}；可选值：{' / '.join(REASONING_EFFORTS)}")
    return DEFAULT_REASONING_EFFORT if effort == "default" else effort


def load_env(path: Path | None = None) -> None:
    if path is None:
        configured_path = os.environ.get("DEEPSEEK_ENV_FILE")
        path = Path(configured_path).expanduser() if configured_path else Path(".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    session_file: Path
    max_context_chars: int | None
    keep_recent_turns: int
    max_task_seconds: int | None
    max_subagents: int
    workspace: Path
    prompt_file: Path = DEFAULT_PROMPT_FILE
    skills_enabled: bool = True
    skills_dir: Path = DEFAULT_SKILLS_DIR
    mcp_enabled: bool = True
    mcp_config_file: Path = DEFAULT_MCP_CONFIG_FILE
    request_timeout_seconds: float = 180
    max_output_tokens: int | None = 131072
    websearch_enabled: bool = True
    websearch_base_url: str = "https://api.cline.bot/api/v1"
    websearch_api_key: str | None = None
    max_output_continuations: int | None = None
    generation_timeout_seconds: float | None = None
    max_reasoning_chars: int | None = None
    summary_timeout_seconds: float = 45
    finalization_reserve_seconds: float = 120
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    request_retries: int = 3
    retry_backoff_seconds: float = 2
    context_window_tokens: int | None = None
    context_reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000
    templates_dir: Path = DEFAULT_TEMPLATES_DIR
    workflows_dir: Path = DEFAULT_WORKFLOWS_DIR
    environments_dir: Path = DEFAULT_ENVIRONMENTS_DIR
    environment_state_dir: Path = DEFAULT_ENVIRONMENT_STATE_DIR
    builtin_skills_dir: Path = BUILTIN_SKILLS_DIR

    @property
    def resolved_websearch_api_key(self) -> str | None:
        if self.websearch_api_key:
            return self.websearch_api_key
        # Reuse credentials only within the same origin, never across providers.
        main = urlsplit(self.base_url)
        search = urlsplit(self.websearch_base_url)
        if (main.scheme, main.hostname, main.port) == (search.scheme, search.hostname, search.port):
            return self.api_key
        return None

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured in .env")

        workspace = Path.cwd()
        session_file = Path(
            os.environ.get("DEEPSEEK_SESSION_FILE", ".deepseek-session.json")
        ).expanduser()
        if not session_file.is_absolute():
            session_file = workspace / session_file

        prompt_file = Path(
            os.environ.get("DEEPSEEK_PROMPT_FILE", str(DEFAULT_PROMPT_FILE))
        ).expanduser()
        if not prompt_file.is_absolute():
            prompt_file = workspace / prompt_file

        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            session_file=session_file.resolve(),
            max_context_chars=optional_limit("DEEPSEEK_MAX_CONTEXT_CHARS", int),
            keep_recent_turns=max(1, int(os.environ.get("DEEPSEEK_KEEP_RECENT_TURNS", "6"))),
            max_task_seconds=optional_limit("DEEPSEEK_MAX_TASK_SECONDS", int, default="unlimited"),
            max_subagents=max(
                1,
                min(
                    MAX_SUBAGENTS,
                    int(os.environ.get("DEEPSEEK_MAX_SUBAGENTS", str(MAX_SUBAGENTS))),
                ),
            ),
            workspace=workspace,
            prompt_file=prompt_file.resolve(),
            skills_enabled=env_flag("DEEPSEEK_SKILLS_ENABLED", True),
            skills_dir=Path(os.environ.get("DEEPSEEK_SKILLS_DIR", str(DEFAULT_SKILLS_DIR)))
            .expanduser()
            .resolve(),
            templates_dir=Path(os.environ.get("DEEPSEEK_TEMPLATES_DIR", str(DEFAULT_TEMPLATES_DIR)))
            .expanduser()
            .resolve(),
            workflows_dir=Path(os.environ.get("DEEPSEEK_WORKFLOWS_DIR", str(DEFAULT_WORKFLOWS_DIR)))
            .expanduser()
            .resolve(),
            environments_dir=Path(
                os.environ.get("DEEPSEEK_ENVIRONMENTS_DIR", str(DEFAULT_ENVIRONMENTS_DIR))
            )
            .expanduser()
            .resolve(),
            environment_state_dir=Path(
                os.environ.get("DEEPSEEK_ENVIRONMENT_STATE_DIR", str(DEFAULT_ENVIRONMENT_STATE_DIR))
            )
            .expanduser()
            .resolve(),
            mcp_enabled=env_flag("DEEPSEEK_MCP_ENABLED", True),
            mcp_config_file=Path(
                os.environ.get("DEEPSEEK_MCP_CONFIG_FILE", str(DEFAULT_MCP_CONFIG_FILE))
            )
            .expanduser()
            .resolve(),
            request_timeout_seconds=max(
                1, float(os.environ.get("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "180"))
            ),
            max_output_tokens=max(1, int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "131072"))),
            websearch_enabled=os.environ.get("DEEPSEEK_WEBSEARCH_ENABLED", "1").lower()
            not in {"0", "false", "no", "off"},
            websearch_base_url=os.environ.get(
                "DEEPSEEK_WEBSEARCH_BASE_URL", "https://api.cline.bot/api/v1"
            ).rstrip("/"),
            websearch_api_key=os.environ.get("DEEPSEEK_WEBSEARCH_API_KEY") or None,
            max_output_continuations=(
                max(0, int(os.environ["DEEPSEEK_MAX_OUTPUT_CONTINUATIONS"]))
                if os.environ.get("DEEPSEEK_MAX_OUTPUT_CONTINUATIONS", "").lower()
                not in {"", "unlimited", "none"}
                else None
            ),
            generation_timeout_seconds=optional_limit("DEEPSEEK_GENERATION_TIMEOUT_SECONDS", float),
            max_reasoning_chars=optional_limit("DEEPSEEK_MAX_REASONING_CHARS", int),
            summary_timeout_seconds=max(
                1, float(os.environ.get("DEEPSEEK_SUMMARY_TIMEOUT_SECONDS", "45"))
            ),
            finalization_reserve_seconds=max(
                1, float(os.environ.get("DEEPSEEK_FINALIZATION_RESERVE_SECONDS", "120"))
            ),
            reasoning_effort=parse_reasoning_effort(os.environ.get("DEEPSEEK_REASONING_EFFORT")),
            request_retries=max(0, int(os.environ.get("DEEPSEEK_REQUEST_RETRIES", "3"))),
            retry_backoff_seconds=max(
                0, float(os.environ.get("DEEPSEEK_RETRY_BACKOFF_SECONDS", "2"))
            ),
            context_window_tokens=optional_limit("DEEPSEEK_CONTEXT_WINDOW_TOKENS", int),
            context_reserve_tokens=max(
                1, int(os.environ.get("DEEPSEEK_CONTEXT_RESERVE_TOKENS", "16384"))
            ),
            keep_recent_tokens=max(1, int(os.environ.get("DEEPSEEK_KEEP_RECENT_TOKENS", "20000"))),
        )


def optional_limit(name, convert, default=""):
    value = os.environ.get(name, default).strip().lower()
    return None if value in {"", "none", "unlimited", "0"} else max(1, convert(value))


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def system_prompt(
    workspace: Path,
    prompt_file: Path = DEFAULT_PROMPT_FILE,
    tool_names: tuple[str, ...] = (),
    *,
    instructions: str | None = None,
) -> str:
    instructions = (
        prompt_file.read_text(encoding="utf-8").strip() if instructions is None else instructions
    )
    if not instructions:
        raise ValueError(f"System prompt file is empty: {prompt_file}")
    return (
        f"{instructions}\n\n"
        f"Host: {platform.system()}\n"
        f"Available tools: {', '.join(tool_names) or 'none'}.\n\n"
    )


def environment_context(workspace: Path) -> str:
    return TASK_CONTEXT_PREFIX + f"Workspace: {workspace}\nCurrent date: {date.today().isoformat()}"
