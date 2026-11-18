import hashlib
import os
import platform
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from model_api import compatibility, parse_api_compat, parse_model_api, resolve_model_api

APP_NAME = "Aster"
CLI_NAME = "agent"
PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_USER_DIR = Path.home() / ".agents" / "aster"
DEFAULT_PROMPT_FILE = PROGRAM_DIR / "system_prompt.md"
DEFAULT_SKILLS_DIR = DEFAULT_USER_DIR / "skills"
DEFAULT_TEMPLATES_DIR = DEFAULT_USER_DIR / "templates"
DEFAULT_WORKFLOWS_DIR = DEFAULT_USER_DIR / "workflows"
DEFAULT_ENVIRONMENTS_DIR = DEFAULT_USER_DIR / "environments"
DEFAULT_ENVIRONMENT_STATE_DIR = DEFAULT_USER_DIR / "environment-state"
DEFAULT_MCP_CONFIG_FILE = DEFAULT_USER_DIR / "mcp.json"
DEFAULT_AUTH_FILE = DEFAULT_USER_DIR / "auth.json"
CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEEPSEEK_WEBSEARCH_BASE_URL = "https://api.deepseek.com/v1/tools"
CLINE_WEBSEARCH_BASE_URL = "https://api.cline.bot/api/v1"
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


class LoginRequired(RuntimeError):
    """No saved login and no model credentials in the environment."""


@dataclass(frozen=True)
class Login:
    """A saved login choice: ``chatgpt``, ``api`` (stored endpoint) or ``env`` (variables)."""

    provider: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    context_window: int | None = None
    account: str = ""
    api: str = "auto"
    api_compat: dict[str, bool] = field(default_factory=dict)


def user_dir() -> Path:
    """One user-owned root for configuration and persistent application state."""
    return Path(os.environ.get("ASTER_HOME") or DEFAULT_USER_DIR).expanduser().resolve()


def user_path(variable: str, name: str) -> Path:
    return Path(os.environ.get(variable) or user_dir() / name).expanduser().resolve()


def default_session_file(workspace: Path) -> Path:
    # Canonical paths share history across symlinks, while same-name projects stay separate.
    workspace_id = hashlib.sha256(os.fsencode(workspace.resolve())).hexdigest()
    return user_dir() / "sessions" / workspace_id / "session.json"


def auth_file() -> Path:
    return user_path("DEEPSEEK_AUTH_FILE", "auth.json")


def load_env(path: Path | None = None) -> None:
    if path is None:
        path = user_path("DEEPSEEK_ENV_FILE", ".env")
    if not path.exists():
        return

    # utf-8-sig: editors on Windows may start the file with a byte order mark.
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        # Shell-sourceable files write "export KEY=value".
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), env_file_value(value.strip()))


def env_file_value(value: str) -> str:
    """A .env value: quoted text as written, or unquoted text before a ` #` comment."""
    quote = value[:1]
    if quote in {"'", '"'} and (end := value.find(quote, 1)) != -1:
        return value[1:end]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip().strip("'\"")


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
    skills_dir: Path = field(default_factory=lambda: user_path("DEEPSEEK_SKILLS_DIR", "skills"))
    mcp_enabled: bool = True
    mcp_config_file: Path = field(
        default_factory=lambda: user_path("DEEPSEEK_MCP_CONFIG_FILE", "mcp.json")
    )
    request_timeout_seconds: float = 180
    task_token_budget: int | None = None
    session_token_budget: int | None = None
    goal_token_budget: int | None = None
    max_output_tokens: int | None = 131072
    websearch_enabled: bool = True
    websearch_base_url: str = ""
    websearch_api_key: str | None = None
    websearch_provider: str = "auto"
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
    templates_dir: Path = field(
        default_factory=lambda: user_path("DEEPSEEK_TEMPLATES_DIR", "templates")
    )
    workflows_dir: Path = field(
        default_factory=lambda: user_path("DEEPSEEK_WORKFLOWS_DIR", "workflows")
    )
    environments_dir: Path = field(
        default_factory=lambda: user_path("DEEPSEEK_ENVIRONMENTS_DIR", "environments")
    )
    environment_state_dir: Path = field(
        default_factory=lambda: user_path("DEEPSEEK_ENVIRONMENT_STATE_DIR", "environment-state")
    )
    builtin_skills_dir: Path = BUILTIN_SKILLS_DIR
    provider: str = "api"
    auth_file: Path | None = None
    account: str = ""
    api: str = "auto"
    api_compat: dict[str, bool] = field(default_factory=dict)
    mcp_config_source: Path | None = None

    @property
    def resolved_api(self) -> str:
        return resolve_model_api(self.api, self.base_url, self.provider)

    @property
    def model_compat(self):
        return compatibility(self)

    @property
    def resolved_websearch_provider(self) -> str:
        if self.websearch_provider != "auto":
            if self.websearch_provider not in {"deepseek", "cline"}:
                raise ValueError("DEEPSEEK_WEBSEARCH_PROVIDER must be auto, deepseek or cline")
            return self.websearch_provider
        endpoint = self.websearch_base_url or self.base_url
        return "deepseek" if urlsplit(endpoint).hostname == "api.deepseek.com" else "cline"

    @property
    def resolved_websearch_base_url(self) -> str:
        provider = self.resolved_websearch_provider
        base = self.websearch_base_url.rstrip("/") or (
            DEEPSEEK_WEBSEARCH_BASE_URL if provider == "deepseek" else CLINE_WEBSEARCH_BASE_URL
        )
        if provider == "deepseek" and urlsplit(base).path in {"", "/v1"}:
            return base.removesuffix("/v1") + "/v1/tools"
        return base

    @property
    def resolved_websearch_api_key(self) -> str | None:
        if self.websearch_api_key:
            return self.websearch_api_key
        # Reuse credentials only within the same origin, never across providers.
        main = urlsplit(self.base_url)
        search = urlsplit(self.resolved_websearch_base_url)
        if (main.scheme, main.hostname, main.port) == (search.scheme, search.hostname, search.port):
            return self.api_key
        return None

    @property
    def available_websearch_provider(self) -> str | None:
        """Use a configured search API, falling back to the signed-in ChatGPT account."""
        if self.resolved_websearch_api_key:
            return self.resolved_websearch_provider
        return "chatgpt" if self.provider == "chatgpt" else None

    def with_login(self, login: Login | None) -> "Settings":
        return replace(self, **login_fields(login))

    @classmethod
    def from_env(cls, login: Login | None = None) -> "Settings":
        load_env()
        fields = login_fields(login)

        workspace = Path.cwd()
        session_file = Path(
            os.environ.get("DEEPSEEK_SESSION_FILE") or default_session_file(workspace)
        ).expanduser()
        if not session_file.is_absolute():
            session_file = workspace / session_file

        prompt_file = Path(
            os.environ.get("DEEPSEEK_PROMPT_FILE", str(DEFAULT_PROMPT_FILE))
        ).expanduser()
        if not prompt_file.is_absolute():
            prompt_file = workspace / prompt_file
        mcp_source = Path(
            os.environ.get("DEEPSEEK_MCP_CONFIG_FILE") or user_dir() / "mcp.json"
        ).expanduser().absolute()
        mcp_path = mcp_source.resolve()

        return cls(
            **fields,
            session_file=session_file.resolve(),
            max_context_chars=optional_limit("DEEPSEEK_MAX_CONTEXT_CHARS", int),
            keep_recent_turns=max(1, env_number("DEEPSEEK_KEEP_RECENT_TURNS", int, "6")),
            max_task_seconds=optional_limit("DEEPSEEK_MAX_TASK_SECONDS", int, default="unlimited"),
            max_subagents=max(
                1,
                min(MAX_SUBAGENTS, env_number("DEEPSEEK_MAX_SUBAGENTS", int, str(MAX_SUBAGENTS))),
            ),
            workspace=workspace,
            prompt_file=prompt_file.resolve(),
            skills_enabled=env_flag("DEEPSEEK_SKILLS_ENABLED", True),
            skills_dir=user_path("DEEPSEEK_SKILLS_DIR", "skills"),
            templates_dir=user_path("DEEPSEEK_TEMPLATES_DIR", "templates"),
            workflows_dir=user_path("DEEPSEEK_WORKFLOWS_DIR", "workflows"),
            environments_dir=user_path("DEEPSEEK_ENVIRONMENTS_DIR", "environments"),
            environment_state_dir=user_path("DEEPSEEK_ENVIRONMENT_STATE_DIR", "environment-state"),
            mcp_enabled=env_flag("DEEPSEEK_MCP_ENABLED", True),
            mcp_config_file=mcp_path,
            mcp_config_source=mcp_source if mcp_source != mcp_path else None,
            request_timeout_seconds=max(
                1, env_number("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", float, "180")
            ),
            task_token_budget=optional_limit("ASTER_TASK_TOKEN_BUDGET", int),
            session_token_budget=optional_limit("ASTER_SESSION_TOKEN_BUDGET", int),
            goal_token_budget=optional_limit("ASTER_GOAL_TOKEN_BUDGET", int),
            max_output_tokens=max(1, env_number("DEEPSEEK_MAX_OUTPUT_TOKENS", int, "131072")),
            websearch_enabled=env_flag("DEEPSEEK_WEBSEARCH_ENABLED", True),
            websearch_base_url=os.environ.get("DEEPSEEK_WEBSEARCH_BASE_URL", "")
            .strip()
            .rstrip("/"),
            websearch_api_key=os.environ.get("DEEPSEEK_WEBSEARCH_API_KEY") or None,
            websearch_provider=os.environ.get("DEEPSEEK_WEBSEARCH_PROVIDER", "auto").strip().lower()
            or "auto",
            max_output_continuations=(
                max(0, env_number("DEEPSEEK_MAX_OUTPUT_CONTINUATIONS", int, "0"))
                if os.environ.get("DEEPSEEK_MAX_OUTPUT_CONTINUATIONS", "").lower()
                not in {"", "unlimited", "none"}
                else None
            ),
            generation_timeout_seconds=optional_limit("DEEPSEEK_GENERATION_TIMEOUT_SECONDS", float),
            max_reasoning_chars=optional_limit("DEEPSEEK_MAX_REASONING_CHARS", int),
            summary_timeout_seconds=max(
                1, env_number("DEEPSEEK_SUMMARY_TIMEOUT_SECONDS", float, "45")
            ),
            finalization_reserve_seconds=max(
                1, env_number("DEEPSEEK_FINALIZATION_RESERVE_SECONDS", float, "120")
            ),
            reasoning_effort=parse_reasoning_effort(os.environ.get("DEEPSEEK_REASONING_EFFORT")),
            request_retries=max(0, env_number("DEEPSEEK_REQUEST_RETRIES", int, "3")),
            retry_backoff_seconds=max(0, env_number("DEEPSEEK_RETRY_BACKOFF_SECONDS", float, "2")),
            context_reserve_tokens=max(
                1, env_number("DEEPSEEK_CONTEXT_RESERVE_TOKENS", int, "16384")
            ),
            keep_recent_tokens=max(1, env_number("DEEPSEEK_KEEP_RECENT_TOKENS", int, "20000")),
        )


def login_fields(login: Login | None) -> dict:
    """The model endpoint and credentials a login selects; ``None`` reads the environment."""
    if login is None or login.provider == "env":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LoginRequired("DEEPSEEK_API_KEY is not configured in .env; run /login to sign in")
        fields = {
            "provider": "api",
            "api_key": api_key,
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "account": "",
        }
    elif login.provider == "chatgpt":
        # Tokens rotate, so they stay in the auth file rather than in copied settings.
        fields = {
            "provider": "chatgpt",
            "api_key": "",
            "base_url": CHATGPT_BASE_URL,
            "model": login.model,
            "account": login.account,
        }
    elif login.provider == "api":
        fields = {
            "provider": "api",
            "api_key": login.api_key,
            "base_url": login.base_url,
            "model": login.model,
            "account": urlsplit(login.base_url).hostname or login.base_url,
        }
    else:
        raise ValueError(f"Unknown login provider {login.provider!r}")
    fields["base_url"] = fields["base_url"].rstrip("/")
    fields["api"] = parse_model_api(
        os.environ.get("DEEPSEEK_API") if login is None or login.provider == "env" else login.api
    )
    fields["api_compat"] = parse_api_compat(
        os.environ.get("DEEPSEEK_API_COMPAT")
        if login is None or login.provider == "env"
        else login.api_compat
    )
    fields["auth_file"] = auth_file()
    fields["context_window_tokens"] = (
        login.context_window if login is not None else None
    ) or optional_limit("DEEPSEEK_CONTEXT_WINDOW_TOKENS", int)
    return fields


def env_number(name: str, convert, default: str):
    """A numeric variable; a malformed value names the variable instead of a bare traceback."""
    value = os.environ.get(name, default).strip()
    try:
        return convert(value)
    except ValueError:
        kind = "整数" if convert is int else "数字"
        raise ValueError(f"{name} 应为{kind}，当前值为 {value!r}") from None


def optional_limit(name, convert, default=""):
    value = os.environ.get(name, default).strip().lower()
    if value in {"", "none", "unlimited", "0"}:
        return None
    return max(1, env_number(name, convert, default))


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
