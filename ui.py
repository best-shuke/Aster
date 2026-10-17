import os
import select
import shutil
import signal
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app, run_in_terminal
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from pygments.lexer import Lexer
from rich.console import Console as RichConsole
from rich.console import Group
from rich.markdown import CodeBlock, Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from config import APP_NAME
from execution_history import ExecutionEntry, plain, result_summary
from execution_trace import current_trace
from terminal_rendering import StreamingDisplay, TaskLive

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"

_TOOL_LABELS = {
    "ask_user": "需要选择",
    "read": "读取",
    "exec": "运行",
    "monitor": "监控状态",
    "edit": "修改",
    "write": "写入",
    "grep": "搜索",
    "websearch": "联网搜索",
    "find": "查找文件",
    "ls": "查看目录",
    "delegate": "并行调查",
    "team": "团队协作",
    "update_plan": "更新计划",
}

_TASK_STATES = {
    "pending": ("等待中", "dim"),
    "running": ("运行中", "cyan"),
    "completed": ("已完成", "green"),
    "paused": ("已暂停", "yellow"),
    "cancelled": ("已取消", "yellow"),
    "restored": ("已恢复", "yellow"),
    "partial": ("部分完成", "yellow"),
    "timeout": ("超时", "red"),
    "unknown": ("结果未知", "yellow"),
    "error": ("失败", "red"),
    "blocked": ("受阻", "yellow"),
    "incomplete": ("未完成", "yellow"),
    "background": ("后台继续", "cyan"),
}

_SLASH_COMMANDS = {
    "/cache": "查看缓存命中率",
    "/clear": "保存当前会话并开启新对话",
    "/effort": "选择或修改推理强度",
    "/env": "选择或切换能力环境",
    "/environment": "选择、确认或切换能力环境",
    "/exit": "退出",
    "/help": "查看帮助",
    "/history": "查看对话历史",
    "/multiagents": "进入默认多代理团队环境",
    "/trace": "浏览执行轨迹、命令和工具结果",
    "/tool": "按编号查看某次工具调用",
    "/new": "保存当前会话并开启新对话",
    "/prompt": "查看系统提示词",
    "/quit": "退出",
    "/resume": "恢复历史会话",
    "/session": "查看会话文件位置",
    "/skills": "查看技能说明",
    "/status": "查看上下文和运行配置",
    "/tasks": "查看任务和子代理状态",
    "/template": "选择或管理会话模板",
    "/todo": "查看任务和子代理状态",
    "/workflow": "选择、运行或新建 workflow",
}


class _CommandCompleter(Completer):
    def __init__(self, skill_provider: Callable[[], dict[str, str]] | None = None) -> None:
        self.skill_provider = skill_provider

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        if "\n" in document.text:
            return
        if document.text_after_cursor and not document.text_after_cursor[0].isspace():
            return
        prefix = document.text_before_cursor.lstrip()
        if not prefix.startswith("/"):
            return
        skills = self.skill_provider() if self.skill_provider is not None else {}
        if prefix.startswith("/skills") and prefix[7:8].isspace():
            argument = prefix[7:].lstrip()
            if any(character.isspace() for character in argument):
                return
            for name, description in sorted({"all": "列出全部技能", **skills}.items()):
                if name.startswith(argument) and name != argument:
                    yield Completion(name, start_position=-len(argument), display_meta=description)
            return
        if any(character.isspace() for character in prefix):
            return
        commands = {**_SLASH_COMMANDS, **{f"/skill:{name}": text for name, text in skills.items()}}
        for command, description in sorted(commands.items()):
            if command.startswith(prefix) and command != prefix:
                yield Completion(command, start_position=-len(prefix), display_meta=description)


def _display_width(text: str) -> int:
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def _ellipsize(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if _display_width(text) <= max_width:
        return text
    if max_width == 1:
        return "…"
    result = []
    used = 0
    for character in text:
        character_width = _display_width(character)
        if used + character_width > max_width - 1:
            break
        result.append(character)
        used += character_width
    return "".join(result) + "…"


def _wrap_display(text: str, max_width: int) -> list[str]:
    max_width = max(1, max_width)
    lines = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = []
        used = 0
        for character in paragraph:
            character_width = _display_width(character)
            if current and used + character_width > max_width:
                lines.append("".join(current).rstrip())
                current = []
                used = 0
            current.append(character)
            used += character_width
        lines.append("".join(current).rstrip())
    return lines


def _short_path(path: Path | str, max_width: int) -> str:
    text = str(path)
    try:
        home = str(Path.home())
        if text == home or text.startswith(home + os.sep):
            text = "~" + text[len(home) :]
    except RuntimeError:
        pass
    if _display_width(text) <= max_width:
        return text
    name = Path(text).name
    tail = f"…/{name}"
    if _display_width(tail) <= max_width:
        return tail
    return _ellipsize(text, max_width)


def _duration(seconds: float, *, compact: bool = False) -> str:
    if compact:
        return (
            f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
            if seconds >= 60
            else f"{int(seconds)}s"
        )
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes} 分 {remaining:02d} 秒"


def _token_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if value >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return str(value)


def _reasoning_text(message: dict[str, Any]) -> str:
    """Return provider reasoning without duplicating compatibility fields."""
    parts = []
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip() and value not in parts:
            parts.append(value)
    return "\n\n".join(parts)


class _RenderedPreview:
    def __init__(self, lines: list[list[Segment]]) -> None:
        self.lines = lines

    def __rich_console__(self, console, options):
        for line in self.lines:
            yield from line
            yield Segment.line()


class _TokenLexer(Lexer):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def get_tokens(self, text, **kwargs):
        return iter(self.tokens)


class _CachedCodeBlock(CodeBlock):
    """Reuse code rows while new rows retain the complete lexer context."""

    @classmethod
    def create(cls, markdown, token):
        block = super().create(markdown, token)
        block.cache = markdown._code_cache
        block.cache_key = (token.map[0] if token.map else 0, block.lexer_name)
        return block

    def __rich_console__(self, console, options):
        code = str(self.text).rstrip()
        raw_lines = code.expandtabs(4).split("\n")
        width = options.max_width
        cached = self.cache.get(self.cache_key)
        common = 0
        if cached is not None and cached[0] == width:
            for old, new in zip(cached[1], raw_lines, strict=False):
                if old != new:
                    break
                common += 1
        if cached is not None and common == len(raw_lines) == len(cached[1]):
            lines = cached[3]
        else:
            original_syntax = Syntax(code, self.lexer_name)
            lexer = original_syntax.lexer or original_syntax.default_lexer
            processed = code.expandtabs(4) + "\n"
            tokens = list(lexer.get_tokens(processed))
            if common:
                unchanged_chars = 0
                for (old_type, old_text), (new_type, new_text) in zip(
                    cached[4], tokens, strict=False
                ):
                    if old_type != new_type:
                        break
                    if old_text != new_text:
                        # Count the characters shared before the first difference. (os.path
                        # helpers are for paths, not token text, so compare directly.)
                        unchanged_chars += next(
                            (
                                i
                                for i, (a, b) in enumerate(zip(old_text, new_text, strict=False))
                                if a != b
                            ),
                            min(len(old_text), len(new_text)),
                        )
                        break
                    unchanged_chars += len(old_text)
                common = min(common, processed[:unchanged_chars].count("\n"))
            syntax = Syntax(
                code,
                _TokenLexer(tokens),
                theme=self.theme,
                word_wrap=True,
                padding=1,
                line_range=(common + 1, None) if common else None,
            )
            tail = console.render_lines(syntax, options.update(height=None), pad=False)
            prefix_heights = cached[2][:common] if common else []
            heights = [
                *prefix_heights,
                *(len(Text(line).wrap(console, max(1, width - 2))) for line in raw_lines[common:]),
            ]
            lines = [*cached[3][: 1 + sum(prefix_heights)], *tail[1:]] if common else tail
            self.cache[self.cache_key] = (width, raw_lines, heights, lines, tokens)
        yield _RenderedPreview(lines)


class _TaskInputMonitor:
    def __init__(self) -> None:
        self._input: Input | None = None
        self._raw_mode = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._wake_pipe: tuple[int, int] | None = None
        self._suspended = 0
        self._closed = False

    def start(self) -> None:
        if self._closed or self._suspended or self._thread is not None:
            return
        self._input = create_input(stdin=sys.stdin)
        self._raw_mode = self._input.raw_mode()
        self._stopped = threading.Event()
        try:
            self._raw_mode.__enter__()
            self._wake_pipe = os.pipe()
            self._thread = threading.Thread(
                target=self._watch,
                args=(self._input, self._stopped, self._wake_pipe[0]),
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        self._stopped.set()
        thread, self._thread = self._thread, None
        raw_mode, self._raw_mode = self._raw_mode, None
        input_reader, self._input = self._input, None
        wake_pipe, self._wake_pipe = self._wake_pipe, None
        try:
            if wake_pipe is not None:
                with suppress(OSError):
                    os.write(wake_pipe[1], b"\0")
            if thread is not None and thread.ident is not None:
                thread.join()
        finally:
            try:
                if raw_mode is not None:
                    raw_mode.__exit__(None, None, None)
            finally:
                try:
                    if input_reader is not None:
                        input_reader.close()
                finally:
                    if wake_pipe is not None:
                        for descriptor in wake_pipe:
                            with suppress(OSError):
                                os.close(descriptor)

    def close(self) -> None:
        self._closed = True
        self.stop()

    @contextmanager
    def suspend(self):
        self._suspended += 1
        try:
            self.stop()
            yield
        finally:
            self._suspended -= 1
            self.start()

    @staticmethod
    def _watch(input_reader: Input, stopped: threading.Event, wake_fd: int) -> None:
        flush_deadline = None
        try:
            while not stopped.is_set():
                timeout = (
                    max(0.0, flush_deadline - time.monotonic())
                    if flush_deadline is not None
                    else None
                )
                ready = select.select([input_reader.fileno(), wake_fd], [], [], timeout)[0]
                if stopped.is_set():
                    return
                if wake_fd in ready:
                    return
                if ready:
                    keys = input_reader.read_keys()
                    flush_deadline = time.monotonic() + 0.1
                    flushed = False
                else:
                    keys = input_reader.flush_keys()
                    flush_deadline = None
                    flushed = True
                if input_reader.closed:
                    return
                if any(
                    key.key == Keys.ControlC or (flushed and key.key == Keys.Escape) for key in keys
                ):
                    if not stopped.is_set():
                        os.kill(os.getpid(), signal.SIGINT)
                    return
        except (OSError, ValueError):
            return


class Console:
    def __init__(
        self,
        model: str = "DeepSeek",
        *,
        workspace: Path | None = None,
        reasoning_effort: str = "max",
        websearch_status: str = "未知",
        mcp_status: str = "关闭",
    ) -> None:
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.colors = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.model = model
        self.workspace = workspace or Path.cwd()
        self.reasoning_effort = reasoning_effort
        self.websearch_status = websearch_status
        self.mcp_status = mcp_status
        self.mcp_policy_enabled = True
        self.environment_name = "默认"
        self.environment_provider = None
        self.last_environment_name = None
        self._response_content: list[str] = []
        self._response_reasoning: list[str] = []
        self._response_number = 0
        self._tool_count = 0
        self._tool_errors = 0
        self._preview_content: list[str] = []
        self._preview_reasoning: list[str] = []
        self._preview_cache: dict[str, tuple[str, int, list[list[Segment]]]] = {}
        self._stream_body_cache = {}
        self._stream_code_cache = {"reasoning": {}, "content": {}}
        self._last_progress = 0.0
        self._last_phase = ""
        self._last_retry_attempt = None
        self._last_tool_progress = None
        self._tool_parameter_progress = False
        self._status_visible = False
        self._screen_active = False
        self._task_started: float | None = None
        self._prompt_session: PromptSession | None = None
        self._task_input_monitor: _TaskInputMonitor | None = None
        self._live: TaskLive | None = None
        self._task_title = ""
        self._task_state = "pending"
        self._task_activity = ""
        self._task_elapsed = 0.0
        self._todos: list[dict[str, str]] = []
        self._agents: dict[str, dict[str, Any]] = {}
        # Latest team status from the team runtime, shown while main is not in a turn.
        self._team: dict[str, Any] | None = None
        self._context: dict[str, Any] | None = None
        self._performance: dict[str, dict[str, Any]] = {}
        self.mcp_provider: Callable[[], list[dict[str, Any]]] | None = None
        self.skill_provider: Callable[[], dict[str, str]] | None = None
        self.skill_status: dict[str, Any] | None = None
        self._mcp_servers: list[dict[str, Any]] = []
        self.execution_provider: Callable[[], list[ExecutionEntry]] | None = None
        self._active_tool_number = 0

    def update_context(self, snapshot: dict[str, Any]) -> None:
        self._context = dict(snapshot)
        self._refresh_task()

    def update_mcp(self, servers: list[dict[str, Any]]) -> None:
        self._clear_status()
        self._mcp_servers = [dict(server) for server in servers]
        enabled = [server for server in servers if server.get("status") != "disabled"]
        connected = sum(server.get("status") == "connected" for server in enabled)
        failed = sum(server.get("status") == "failed" for server in enabled)
        self.mcp_status = f"已连接 {connected}/{len(enabled)}" if enabled else "未配置"
        if failed:
            self.mcp_status += f" · {failed} 失败"
        if not self.mcp_policy_enabled:
            self.mcp_status = "策略禁用 · " + self.mcp_status

    def _refresh_mcp(self) -> None:
        if self.mcp_provider is not None:
            self.update_mcp(self.mcp_provider())

    def mcp_connecting(self, server: str) -> None:
        self._set_status(f"  MCP · 正在连接 {server}…")

    def _context_label(self, *, compact: bool = False) -> str:
        if not self._context:
            return ""
        tokens = self._context["tokens"]
        window = self._context["window_tokens"]
        if compact:
            return f"CTX {_token_count(tokens)}/{_token_count(window)}"
        return (
            f"上下文 {_token_count(tokens)}/{_token_count(window)} tokens · {tokens / window:.1%}"
        )

    def _context_style(self) -> str:
        if not self._context:
            return "dim"
        if self._context["tokens"] >= self._context["window_tokens"]:
            return "bold red"
        if (
            self._context.get("max_chars") is not None
            and self._context["chars"] >= self._context["max_chars"]
        ):
            return "yellow"
        if self._context["tokens"] >= self._context["budget_tokens"]:
            return "yellow"
        return "dim"

    def _tps_label(self) -> str:
        samples = [
            item for item in tuple(self._performance.values()) if item.get("tps") is not None
        ]
        if not samples:
            return "TPS —"
        duration = sum(item["generation_seconds"] for item in samples)
        tokens = sum(item["tokens"] for item in samples)
        return f"TPS {tokens / duration:.1f}"

    def _toolbar(self) -> str:
        self._refresh_mcp()
        width = self.terminal_width() - 1
        context = self._context_label(compact=width < 70)
        environment = f"环境 {self.environment_name}"
        if self.last_environment_name:
            environment += f"（上次 {self.last_environment_name}）"
        parts = [environment, context, self._tps_label()]
        team = self._team_label()
        if team:
            parts.insert(1, team)
        if width >= 70 or not context:
            parts.append(f"MCP {self.mcp_status}")
        if width >= 110:
            parts.append("Ctrl-T 执行轨迹")
        line = " · ".join(part for part in parts if part)
        if context and _display_width(line) > width and width < 40:
            return "\n".join(
                _ellipsize(part, width) for part in (environment, context, self._tps_label())
            )
        return _ellipsize(line, width)

    def _team_label(self) -> str:
        team = self._team
        if not team:
            return ""
        if team.get("background") == "running":
            busy = {*team.get("working", []), *team.get("queued", [])}
            label = f"团队后台 {len(busy)} 人工作中"
            if team.get("unread"):
                label += f" · {team['unread']} 条新结果"
            return label
        if team.get("unread"):
            return f"团队 {team['unread']} 条新结果 /env results"
        return ""

    def update_team(self, event: dict[str, Any] | None) -> None:
        """Team status from the runtime; it may arrive on a member's thread."""
        self._team = dict(event) if event else None
        notice = (event or {}).get("notice")
        if notice:
            # Prompts patch stdout and the task display redirects it, so this lands above.
            print(self.style(f"  ◆ {notice}", CYAN))
        self._refresh_task()
        self._invalidate_prompt()

    def _invalidate_prompt(self) -> None:
        session = self._prompt_session
        app = getattr(session, "app", None) if session is not None else None
        if app is not None and app.is_running:
            # Thread safe: redraws the bottom toolbar with the new team status.
            app.invalidate()

    def style(self, text: str, *codes: str) -> str:
        if not self.colors:
            return text
        return f"{''.join(codes)}{text}{RESET}"

    def _rich(self, *, stderr: bool = False) -> RichConsole:
        if self._live is not None and not stderr:
            return self._live.console
        stream = sys.stderr if stderr else sys.stdout
        return RichConsole(
            file=stream,
            force_terminal=stream.isatty(),
            no_color=not self.colors,
            color_system="auto" if self.colors else None,
            width=self.terminal_width(),
            highlight=False,
            theme=Theme({"markdown.code": "bold bright_cyan on default"}),
        )

    def _section(self, label: str, *, style: str) -> None:
        self._rich().print(Rule(label, align="left", style=style))

    def _markdown(self, content: str, *, dim: bool = False) -> None:
        content = content.strip("\n")
        if not content:
            return
        self._rich().print(self._markdown_renderable(content, dim=dim))

    def _markdown_renderable(self, content: str, *, dim: bool = False):
        renderable = Markdown(content, code_theme="monokai", hyperlinks=self.interactive)
        return Padding(renderable, (0, 2, 0, 3), style="dim" if dim else "none")

    @staticmethod
    def terminal_width() -> int:
        return max(24, shutil.get_terminal_size(fallback=(88, 24)).columns)

    def _clear_status(self) -> None:
        if not self._status_visible:
            return
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        self._status_visible = False

    def _set_status(self, text: str) -> None:
        if not self.interactive:
            return
        if self._live is not None:
            self._task_activity = text.strip()
            self._refresh_task()
            return
        fitted = _ellipsize(text, self.terminal_width() - 1)
        sys.stdout.write("\r\033[2K" + self.style(fitted, DIM))
        sys.stdout.flush()
        self._status_visible = True

    def banner(
        self, session_file: Path, history_count: int = 0, *, session_label: str | None = None
    ) -> None:
        self._clear_status()
        self._refresh_mcp()
        banner_width = min(self.terminal_width(), 100)
        width = banner_width - 4
        session = session_label or ("已恢复" if history_count else "新会话")
        if history_count:
            session += f" · {history_count} 条记录"

        def line(parts: list[Text], right: Text | None = None):
            left = Text(" · ", style="bright_black").join(parts)
            if right is None:
                return left
            if _display_width(left.plain) + _display_width(right.plain) + 2 > width:
                return Group(left, right)
            row = Table.grid(expand=True)
            row.add_column(ratio=1)
            row.add_column(justify="right", no_wrap=True)
            row.add_row(left, right)
            return row

        def packed(parts: list[Text]):
            rows = []
            current = []
            used = 0
            for part in parts:
                size = _display_width(part.plain)
                if current and used + size + 3 > width:
                    rows.append(line(current))
                    current, used = [], 0
                used += size + (3 if current else 0)
                current.append(part)
            if current:
                rows.append(line(current))
            return Group(*rows)

        model = Text.assemble(("模型    ", "bright_black"), (self.model, "bold bright_cyan"))
        workspace = Text.assemble(
            ("工作区  ", "bright_black"),
            (_short_path(self.workspace, max(8, width - 8)), "bold"),
        )
        capabilities = [
            Text.assemble(("环境    ", "bright_black"), (self.environment_name, "bold")),
            Text.assemble(("推理 ", "bright_black"), (self.reasoning_effort, "bold cyan")),
            Text.assemble(
                ("联网 ", "bright_black"),
                (
                    self.websearch_status,
                    "bold green" if self.websearch_status == "可用" else "dim",
                ),
            ),
            Text.assemble(
                ("MCP ", "bright_black"),
                (
                    self.mcp_status,
                    "bold yellow"
                    if "失败" in self.mcp_status or "错误" in self.mcp_status
                    else "bold green"
                    if "已连接" in self.mcp_status and self.mcp_policy_enabled
                    else "dim",
                ),
            ),
        ]
        if self.skill_status is not None and self.skill_status.get("enabled"):
            capabilities.append(
                Text.assemble(
                    ("Skills ", "bright_black"), (str(self.skill_status["loaded"]), "bold")
                )
            )
        sections = [
            line([model], Text(session, style="bold green")),
            workspace,
            packed(capabilities),
        ]
        if self._context:
            tokens = self._context["tokens"]
            window = self._context["window_tokens"]
            context_style = self._context_style()
            metrics = [
                Text.assemble(
                    ("上下文  ", "bright_black"),
                    (
                        f"{_token_count(tokens)}/{_token_count(window)}",
                        "bold cyan" if context_style == "dim" else context_style,
                    ),
                    (" · ", "bright_black"),
                    (f"{tokens / window:.1%}", "bold" if context_style == "dim" else context_style),
                )
            ]
            speed = self._tps_label()
            if speed != "TPS —":
                metrics.append(Text(speed, style="bold cyan"))
            sections.append(packed(metrics))
        self._rich().print(
            Panel(
                Group(*sections),
                title=Text(APP_NAME.upper(), style="bold cyan"),
                title_align="left",
                subtitle=Text(
                    "/help  /resume  /env" if banner_width >= 40 else "/help", style="dim"
                ),
                subtitle_align="right",
                border_style="bright_black",
                padding=(0, 1),
                width=banner_width,
            )
        )

    @contextmanager
    def session_screen(self, *, enabled: bool = True):
        if (
            not enabled
            or not self.interactive
            or self._screen_active
            or os.environ.get("TERM") == "dumb"
        ):
            yield
            return
        stream = sys.stdout
        self._screen_active = True
        try:
            stream.write("\033[?1049h\033[2J\033[H")
            stream.flush()
            yield
        finally:
            try:
                if self._live is not None:
                    self._live.stop()
                    self._live = None
                self._clear_status()
            finally:
                self._screen_active = False
                stream.write("\033[0m\033[?25h\033[?1049l")
                stream.flush()

    def status(self, session_file: Path, history_count: int) -> None:
        self._clear_status()
        self._refresh_mcp()
        session = "已恢复" if history_count else "空"
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bright_black", width=8, no_wrap=True)
        table.add_column()
        table.add_row("模型", self.model)
        table.add_row("工作区", _short_path(self.workspace, max(16, self.terminal_width() - 14)))
        table.add_row("推理", self.reasoning_effort)
        health = self.environment_provider() if self.environment_provider else "ready"
        state = {"ready": "就绪", "degraded": "部分依赖不可用", "blocked": "必需依赖不可用"}.get(
            health, health
        )
        table.add_row("环境", f"{self.environment_name} · {state}")
        table.add_row("联网", self.websearch_status)
        if self.skill_status is not None:
            skills = self.skill_status
            table.add_row(
                "Skills",
                f"已加载 {skills['loaded']} · 自动 {skills['automatic']} · "
                f"手动 {skills['manual']} · 警告 {skills['diagnostics']}"
                if skills["enabled"]
                else "关闭",
            )
        table.add_row("MCP", self.mcp_status)
        for server in self._mcp_servers:
            state = {
                "connected": "已连接",
                "not-connected": "待连接",
                "connecting": "连接中",
                "failed": "失败",
                "closed": "已关闭",
                "disabled": "已禁用",
            }.get(server["status"], server["status"])
            table.add_row(
                "", Text(f"{server['name']} · {state} · {server.get('tool_count', 0)} 个工具")
            )
            if server.get("error") and server["status"] != "connected":
                table.add_row("", Text(f"原因：{server['error']}"))
        if self._context:
            snapshot = self._context
            table.add_row(
                "上下文",
                f"{snapshot['tokens']:,} / {snapshot['window_tokens']:,} tokens",
            )
            if snapshot.get("max_chars") is not None:
                table.add_row(
                    "预算", f"字符模式 {snapshot['chars']:,}/{snapshot['max_chars']:,} 字符"
                )
            else:
                table.add_row("预算", f"压缩阈值 {snapshot['budget_tokens']:,} tokens")
        table.add_row("速度", self._tps_label())
        if self._tool_count:
            table.add_row("工具", f"{self._tool_count} 次调用 · {self._tool_errors} 次异常")
        table.add_row("会话", f"{session} · {history_count} 条记录")
        table.add_row("文件", str(session_file), style="dim")
        rich = self._rich()
        rich.print()
        rich.print(Rule("当前状态", align="left", style="cyan"))
        rich.print(Padding(table, (0, 2)))
        rich.print()

    def configuration_notice(self) -> None:
        if self.websearch_status == "未配置":
            self.info("联网搜索未配置；需要时请设置 DEEPSEEK_WEBSEARCH_API_KEY。")

    @contextmanager
    def _suspend_live(self):
        monitor = self._task_input_monitor
        with monitor.suspend() if monitor is not None else nullcontext():
            live = self._live
            if live is not None:
                live.stop()
                self._live = None
            try:
                yield
            finally:
                if live is not None:
                    self._live = live
                    live.start(refresh=True)

    def _read_input(self, prompt: str) -> str:
        with self._suspend_live():
            return self._read_prompt(prompt)

    def _read_prompt(self, prompt: str, *, isolated: bool = False) -> str:
        if not self.interactive:
            return input(prompt)
        if isolated or self._prompt_session is None:
            bindings = KeyBindings()
            completer = (
                None
                if isolated
                else _CommandCompleter(lambda: self.skill_provider() if self.skill_provider else {})
            )

            @bindings.add("c-t")
            async def inspect_execution(event) -> None:
                if not isolated:
                    await run_in_terminal(self.trace, in_executor=True)

            @bindings.add("escape")
            def handle_escape(event) -> None:
                if isolated:
                    event.app.exit(exception=KeyboardInterrupt())
                elif event.current_buffer.complete_state is not None:
                    event.current_buffer.cancel_completion()

            @bindings.add("tab")
            @bindings.add("s-tab")
            def handle_completion(event) -> None:
                if completer is None:
                    return
                buffer = event.current_buffer
                if buffer.complete_state is None or not buffer.complete_state.completions:
                    completions = list(
                        completer.get_completions(
                            buffer.document, CompleteEvent(completion_requested=True)
                        )
                    )
                    if not completions:
                        return
                    buffer.complete_state = CompletionState(buffer.document, completions)
                if event.key_sequence[-1].key == Keys.BackTab:
                    buffer.complete_previous()
                else:
                    buffer.complete_next()

            session = PromptSession(
                history=InMemoryHistory(),
                key_bindings=bindings,
                completer=completer,
                multiline=False,
                prompt_continuation=lambda width, *_: " " * width,
                complete_while_typing=Condition(
                    lambda: (
                        completer is not None
                        and any(
                            completer.get_completions(
                                get_app().current_buffer.document, CompleteEvent()
                            )
                        )
                    )
                ),
                reserve_space_for_menu=0 if isolated else 6,
                include_default_pygments_style=False,
                bottom_toolbar=self._toolbar,
                style=Style.from_dict(
                    {
                        "bottom-toolbar": "fg:ansibrightblack bg:default noreverse",
                        "bottom-toolbar.text": "noreverse",
                    }
                    if self.colors
                    else {"bottom-toolbar": "bg:default noreverse"}
                ),
            )
            if not isolated:
                self._prompt_session = session
        else:
            session = self._prompt_session
        # Team members may report from their threads while the prompt waits for input.
        with patch_stdout(raw=True):
            return session.prompt(prompt)

    @contextmanager
    def interrupt_on_escape(self):
        """Pause on standalone Esc or Ctrl-C, not on navigation-key prefixes."""
        if (
            not self.interactive
            or os.name != "posix"
            or threading.current_thread() is not threading.main_thread()
            or self._task_input_monitor is not None
        ):
            yield
            return
        monitor = _TaskInputMonitor()
        self._task_input_monitor = monitor
        try:
            try:
                monitor.start()
            except (AttributeError, OSError, ValueError):
                monitor.close()
            yield
        finally:
            try:
                monitor.close()
            finally:
                self._task_input_monitor = None

    def prompt(self) -> str:
        self._clear_status()
        value = self._read_input("  You  › ")
        return value if self.interactive else value.strip()

    def read_value(self, prompt: str) -> str | None:
        self._clear_status()
        with self._suspend_live():
            try:
                return self._read_prompt(prompt, isolated=True)
            except (EOFError, KeyboardInterrupt):
                return None

    def select_option(
        self,
        question: str,
        options: list[dict[str, str]],
        *,
        default_index: int = 0,
        title: str = "需要你的选择",
    ) -> int | None:
        """Return a zero-based choice, or None when the menu is dismissed."""
        if not options or not 0 <= default_index < len(options):
            raise ValueError("选择菜单需要有效的选项和默认位置")
        self._clear_status()
        with self._suspend_live():
            try:
                if self.interactive:
                    return self._select_prompt(question, options, default_index, title)
                self._print_options(question, options, title)
                while True:
                    raw_choice = self._read_prompt(f"\n  请选择 [1-{len(options)}] › ").strip()
                    try:
                        selected_index = int(raw_choice) - 1
                    except ValueError:
                        selected_index = -1
                    if 0 <= selected_index < len(options):
                        return selected_index
                    self.warning(f"请输入 1 到 {len(options)} 之间的编号")
            except (EOFError, KeyboardInterrupt):
                return None

    def _print_options(self, question, options, title):
        width = self.terminal_width()
        print(self.style(f"\n  {title}", BOLD, CYAN))
        for line in _wrap_display(question, width - 4):
            print(f"  {line}")
        print()
        for index, option in enumerate(options, 1):
            label_lines = _wrap_display(option["label"], width - 8)
            print(self.style(f"  {index}. {label_lines[0]}", BOLD))
            for line in label_lines[1:]:
                print(f"     {line}")
            for line in _wrap_display(option.get("description", ""), width - 7):
                print(self.style(f"     {line}", DIM))

    def _select_prompt(self, question, options, default_index, title):
        selected_index = default_index
        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("s-tab")
        def previous_option(event):
            nonlocal selected_index
            selected_index = (selected_index - 1) % len(options)

        @bindings.add("down")
        @bindings.add("tab")
        def next_option(event):
            nonlocal selected_index
            selected_index = (selected_index + 1) % len(options)

        @bindings.add("enter")
        def confirm(event):
            event.app.exit(result=selected_index)

        @bindings.add("escape")
        @bindings.add("c-c")
        @bindings.add("c-d")
        def cancel(event):
            event.app.exit(result=None)

        def heading():
            fragments = [("class:heading", f"\n  {title}\n")]
            for line in _wrap_display(question, self.terminal_width() - 4):
                fragments.append(("", f"  {line}\n"))
            fragments.append(("", "\n"))
            return fragments

        def choices():
            width = self.terminal_width()
            fragments = []
            for index, option in enumerate(options):
                selected = index == selected_index
                style = "class:selected" if selected else ""
                if selected:
                    fragments.append(("[SetCursorPosition]", ""))
                label_lines = _wrap_display(option["label"], width - 6)
                marker = "›" if selected else " "
                fragments.append((style, f"  {marker} {label_lines[0]}\n"))
                for line in label_lines[1:]:
                    fragments.append((style, f"    {line}\n"))
                description = option.get("description", "")
                if description:
                    for line in _wrap_display(description, width - 8):
                        fragments.append(("class:description", f"      {line}\n"))
            return fragments

        def hints():
            return [
                ("class:description", f"  {line}\n")
                for line in _wrap_display(
                    "↑/↓ 选择 · 回车确认 · Esc 取消", self.terminal_width() - 4
                )
            ]

        control = FormattedTextControl(choices, focusable=True, show_cursor=False)
        application = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(FormattedTextControl(heading), dont_extend_height=True),
                        Window(control, dont_extend_height=True, always_hide_cursor=True),
                        Window(FormattedTextControl(hints), dont_extend_height=True),
                    ]
                ),
                focused_element=control,
            ),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "heading": "bold ansicyan",
                    "selected": "bold ansicyan",
                    "description": "ansibrightblack",
                }
                if self.colors
                else {}
            ),
            include_default_pygments_style=False,
            full_screen=False,
        )
        application.ttimeoutlen = 0.1
        with patch_stdout(raw=True):
            return application.run()

    def ask_user(self, question: str, options: list[dict[str, str]]) -> dict[str, Any]:
        all_options = [
            *options,
            {
                "label": "其他（自行输入）",
                "description": "以上选项都不合适时，输入你自己的答案。",
            },
        ]
        with self._suspend_live():
            selected_index = self.select_option(question, all_options)
            if selected_index is None:
                raise KeyboardInterrupt
            if selected_index == len(options):
                while True:
                    answer = self.read_value("  请输入你的答案 › ")
                    if answer is None:
                        raise KeyboardInterrupt
                    answer = answer.strip()
                    if answer:
                        self.success("已记录自定义答案")
                        return {
                            "answer": answer,
                            "selected_index": selected_index + 1,
                            "selected_label": "其他（自行输入）",
                            "custom": True,
                        }
                    self.warning("答案不能为空")
            selected = all_options[selected_index]
            self.success(f"已选择：{selected['label']}")
            return {
                "answer": selected["label"],
                "selected_index": selected_index + 1,
                "selected_label": selected["label"],
                "custom": False,
            }

    def _dashboard(self, *, compact: bool = True, max_height: int | None = None) -> Panel | Group:
        width = self.terminal_width()
        if self._live is not None:
            self._live.console.width = width
        available = max(8, width - (4 if compact else 8))
        state, style = _TASK_STATES.get(self._task_state, (self._task_state, "dim"))
        elapsed = (
            time.monotonic() - self._task_started
            if self._task_started is not None
            else self._task_elapsed
        )
        content = []
        if compact:
            status = f"{state} · {_duration(elapsed, compact=True)}"
            title_width = max(1, min(36, width - _display_width(status) - 7))
            marker = {"running": "●", "completed": "✓", "error": "✗", "pending": "○"}.get(
                self._task_state, "◇"
            )
            heading = Text(f"  {marker} ", style=style)
            heading.append(_ellipsize(self._task_title or "当前任务", title_width), style="bold")
            gap = max(2, width - 1 - heading.cell_len - _display_width(status))
            heading.append(" " * gap)
            heading.append(status, style=style)
            heading.truncate(width - 1, overflow="ellipsis")
            content.append(heading)
        else:
            content.append(Text(self._task_title or "暂无任务", style="bold"))
            content.append(Text(f"{state} · {_duration(elapsed)}", style=style))
        if self._task_activity:
            activity = f"  ↳ {self._task_activity}" if compact else self._task_activity
            content.append(Text(_ellipsize(activity, available), style="dim"))
        if self._live is not None and isinstance(
            getattr(self._live, "last_render_error", None), str
        ):
            content.append(Text("  界面刷新恢复中…", style="yellow"))
        header = content[:]
        todos = tuple(self._todos)
        completed = sum(item["status"] == "completed" for item in todos)
        if todos:
            content.append(Text(f"  Todo · {completed}/{len(todos)}", style="bold"))
        short = shutil.get_terminal_size(fallback=(88, 24)).lines < 32
        limit = (3 if short else 6) if compact else len(todos)
        current = next(
            (index for index, item in enumerate(todos) if item["status"] == "in_progress"),
            next(
                (index for index, item in enumerate(todos) if item["status"] == "pending"),
                max(0, len(todos) - 1),
            ),
        )
        offset = min(max(0, current - 1), max(0, len(todos) - limit)) if compact else 0
        for item in todos[offset : offset + limit]:
            marker, item_style = {
                "pending": ("○", "dim"),
                "in_progress": ("◉", "cyan"),
                "completed": ("✓", "green"),
            }[item["status"]]
            content.append(
                Text(_ellipsize(f"  {marker} {item['step']}", available), style=item_style)
            )
        hidden = len(todos) - min(limit, len(todos))
        if hidden > 0:
            content.append(Text(f"  … 折叠 {hidden} 项，/todo 查看全部", style="dim"))
        todo_rows = content[len(header) :]
        agents = tuple(self._agents.values())
        active = sum(agent.get("state") in {"pending", "running"} for agent in agents)
        if agents:
            content.append(Text(f"  子代理 · {active} 活跃 / {len(agents)} 总计", style="bold"))
        ordered = sorted(
            agents,
            key=lambda agent: (
                0
                if agent.get("state") in {"pending", "running"}
                else 2
                if agent.get("state") == "completed"
                else 1
            ),
        )
        agent_limit = (4 if short else 8) if compact else len(agents)
        for agent in ordered[:agent_limit]:
            agent_state, agent_style = _TASK_STATES.get(agent.get("state"), ("未知", "dim"))
            started = agent.get("started_at", time.monotonic())
            duration = agent.get("duration_seconds")
            if duration is None:
                duration = time.monotonic() - started
            mode = {"inspect": "调查", "reason": "推理", "auto": "自动"}.get(agent.get("mode"), "")
            heading = f"  {agent['agent_id']} · {agent_state} · {mode} · {_duration(duration)}"
            if compact:
                heading = f"  {agent['agent_id']} · {agent_state} · {_duration(duration, compact=True)} · {agent['title']}"
            content.append(Text(_ellipsize(heading, available), style=agent_style))
            if not compact:
                content.append(Text(_ellipsize(f"    {agent['title']}", available)))
            detail = self._agent_activity(agent)
            if detail:
                content.append(Text(_ellipsize(f"    {detail}", available), style="dim"))
        if len(ordered) > agent_limit:
            content.append(
                Text(f"  … 还有 {len(ordered) - agent_limit} 个，/tasks 查看全部", style="dim")
            )
        agent_rows = content[len(header) + len(todo_rows) :]
        context = self._context_label(compact=width < 60)
        footer = " · ".join(part for part in (context, self._tps_label()) if part)
        if compact and context and _display_width(footer) > available:
            content.append(Text(_ellipsize(f"  {context}", available), style=self._context_style()))
            content.append(Text(f"  {self._tps_label()}", style="dim"))
        else:
            metrics = Text("  ")
            if context:
                metrics.append(context, style=self._context_style())
                metrics.append(" · ", style="dim")
            metrics.append(self._tps_label(), style="dim")
            if width >= 70:
                output_tokens = sum(
                    item.get("tokens") or 0 for item in tuple(self._performance.values())
                )
                if output_tokens:
                    metrics.append(f" · 输出 {_token_count(output_tokens)}", style="dim")
                if self._tool_count:
                    metrics.append(f" · 工具 {self._tool_count}", style="dim")
                if self._tool_errors:
                    metrics.append(f" · 异常 {self._tool_errors}", style="yellow")
            metrics.truncate(available, overflow="ellipsis")
            content.append(metrics)
        if compact:
            if max_height is not None and len(content) > max_height:
                footer_rows = content[len(header) + len(todo_rows) + len(agent_rows) :]
                header = header[: max(1, max_height - len(footer_rows))]
                room = max(0, max_height - len(header) - len(footer_rows))
                sections = []
                if todo_rows and agent_rows and room == 1:
                    sections.append(
                        Text(
                            _ellipsize(
                                f"  Todo {completed}/{len(todos)} · 子代理 {active} 活跃",
                                available,
                            ),
                            style="dim",
                        )
                    )
                elif room > 0:
                    todo_budget = (
                        room if not agent_rows else (max(1, room // 2) if todo_rows else 0)
                    )
                    agent_budget = room - todo_budget
                    if todo_rows and todo_budget > 0:
                        if len(todo_rows) <= todo_budget:
                            sections.extend(todo_rows)
                        else:
                            item = todos[current]
                            marker = {"pending": "○", "in_progress": "◉", "completed": "✓"}[
                                item["status"]
                            ]
                            sections.append(
                                Text(
                                    _ellipsize(
                                        f"  Todo · {completed}/{len(todos)} · {marker} {item['step']}",
                                        available,
                                    ),
                                    style="cyan" if item["status"] == "in_progress" else "dim",
                                )
                            )
                            if todo_budget > 1:
                                sections.append(Text("  … /todo 查看全部", style="dim"))
                    if agent_rows and agent_budget > 0:
                        if len(agent_rows) <= agent_budget:
                            sections.extend(agent_rows)
                        else:
                            agent = ordered[0]
                            label, agent_style = _TASK_STATES.get(
                                agent.get("state"), ("未知", "dim")
                            )
                            detail = self._agent_activity(agent) or agent["title"]
                            sections.append(
                                Text(
                                    _ellipsize(
                                        f"  {agent['agent_id']} {label} · 子代理 {active}/{len(agents)} 活跃",
                                        available,
                                    ),
                                    style=agent_style,
                                )
                            )
                            if agent_budget > 1:
                                sections.append(
                                    Text(_ellipsize(f"    {detail}", available), style="dim")
                                )
                            if agent_budget > 2:
                                sections.append(Text("  … /tasks 查看全部", style="dim"))
                content = [*header, *sections, *footer_rows]
                content = content[:max_height]
            return Group(*content)
        return Panel(
            Group(*content),
            title=Text("任务详情", style="bold cyan"),
            border_style="bright_black",
            padding=(0, 1),
        )

    @staticmethod
    def _agent_activity(agent: dict[str, Any]) -> str:
        activity = agent.get("activity")
        if activity == "queued":
            return "等待执行"
        if activity == "awaiting_merge":
            return "等待草稿合并"
        if activity == "merging":
            return "正在合并草稿"
        if activity == "warning":
            return str(agent.get("warning") or "")
        if activity == "model":
            if agent.get("model_phase") == "rate_limit":
                delay = agent.get("delay_seconds", 0)
                return (
                    f"模型服务限流 · 等待 {_duration(delay)}" if delay >= 1 else "等待模型服务恢复"
                )
            return {
                "connecting": "正在连接模型",
                "receiving": "正在接收模型响应",
                "retry": "连接重试中",
                "compaction": "正在压缩上下文",
                "tool_call_delta": "正在生成工具参数",
            }.get(agent.get("model_phase"), "正在思考")
        if activity in {"tool", "tool_result"}:
            tool = agent.get("tool") or "工具"
            detail = Console._tool_detail(tool, agent.get("arguments") or {})
            label = _TOOL_LABELS.get(tool, tool)
            text = f"{label} · {detail}" if detail else label
            if activity == "tool_result":
                status = agent.get("tool_status")
                text += (
                    " · 完成"
                    if status == "ok"
                    else f" · {_TASK_STATES.get(status, (status, ''))[0]}"
                )
            elif agent.get("elapsed_seconds") is not None:
                text += f" · {_duration(agent['elapsed_seconds'])}"
            return text
        error = agent.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"]).replace("\n", " ")
        if activity == "finished":
            return f"{agent.get('model_requests', 0)} 次模型请求 · {agent.get('tool_calls', 0)} 次工具调用"
        return ""

    def _refresh_task(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _clear_preview(self) -> None:
        self._preview_content.clear()
        self._preview_reasoning.clear()
        self._preview_cache = {}
        self._stream_body_cache = {}
        self._stream_code_cache = {"reasoning": {}, "content": {}}

    def _preview_markdown(self, text: str, limit: int | None, *, dim: bool = False):
        console = self._rich()
        width = self.terminal_width() - 1
        channel = "reasoning" if dim else "content"
        cache = self._preview_cache
        cached = cache.get(channel)
        if cached is None or cached[0] != text or cached[1] != width:
            code_cache = self._stream_code_cache[channel]
            markdown = Markdown(text.strip("\n"), code_theme="monokai", hyperlinks=self.interactive)
            markdown._code_cache = code_cache
            markdown.elements = {
                **markdown.elements,
                "fence": _CachedCodeBlock,
                "code_block": _CachedCodeBlock,
            }
            lines = console.render_lines(
                Padding(markdown, (0, 2, 0, 3), style="dim" if dim else "none"),
                console.options.update(width=width),
                pad=False,
            )
            while lines and not "".join(segment.text for segment in lines[-1]).strip():
                lines.pop()
            cached = (text, width, lines)
            cache[channel] = cached
        lines = cached[2]
        if limit is not None and len(lines) > limit:
            if limit > 1:
                marker = [Segment("  … 前文已折叠", style=console.get_style("dim"))]
                lines = [marker, *lines[-(limit - 1) :]]
            else:
                lines = lines[-1:]
        return _RenderedPreview(lines)

    def _live_renderable(self):
        response_number = self._response_number
        height = max(4, shutil.get_terminal_size(fallback=(88, 24)).lines - 1)
        has_preview = bool(self._preview_content or self._preview_reasoning)
        dashboard_limit = max(3, height // 3) if has_preview else height
        dashboard = self._dashboard(max_height=dashboard_limit)
        if not has_preview:
            return dashboard
        return self._response_display(
            "".join(self._preview_content),
            "".join(self._preview_reasoning),
            dashboard,
            height,
            response_number=response_number,
        )

    def _response_display(self, content, reasoning, dashboard, height, *, response_number=None):
        if response_number is None:
            response_number = self._response_number
        console = self._rich()
        content = content.strip("\n")
        reasoning = reasoning.strip("\n")
        width = self.terminal_width() - 1
        cache = self._stream_body_cache
        key = (content, reasoning, width)
        if key in cache:
            body, lengths = cache[key]
            return StreamingDisplay(
                body, dashboard, key=response_number, height=height, row_lengths=lengths
            )
        sections = []
        if reasoning:
            sections.extend(
                [
                    Text("  思考过程", style="dim"),
                    self._preview_markdown(reasoning, None, dim=True),
                ]
            )
        if content:
            label = f"{self.model} · 回答" if self.terminal_width() >= 40 else "回答"
            sections.extend(
                [
                    Rule(label, align="left", style="cyan"),
                    self._preview_markdown(content, None),
                ]
            )
        body = []
        lengths = []
        for section in sections:
            if isinstance(section, _RenderedPreview):
                body.extend(section.lines)
                lengths.extend(StreamingDisplay.measure_rows(section.lines))
            else:
                heading = console.render_lines(
                    section, console.options.update(width=width), pad=False
                )
                body.extend(heading)
                lengths.extend([0] * len(heading))
        cache.clear()
        cache[key] = (body, lengths)
        return StreamingDisplay(
            body, dashboard, key=response_number, height=height, row_lengths=lengths
        )

    def task_update(self, event: dict[str, Any]) -> None:
        phase = event.get("phase")
        if phase == "team":
            self.update_team(event)
            return
        if phase == "plan":
            self._task_title = event["title"]
            self._todos = [dict(item) for item in event["steps"]]
            if not self.interactive:
                self.tasks()
        elif phase == "subagent":
            agent_id = event["agent_id"]
            previous = self._agents.get(agent_id, {})
            agent = {"started_at": time.monotonic(), **previous, **event}
            if agent.get("state") not in {"pending", "running"}:
                agent.setdefault("duration_seconds", time.monotonic() - agent["started_at"])
            self._agents[agent_id] = agent
            changed = previous.get("state") != agent.get("state")
            action = agent.get("activity") in {"tool", "tool_result"}
            if agent.get("activity") == "warning":
                self.warning(f"子代理 {agent_id} · {self._agent_activity(agent)}")
            elif not self.interactive and (changed or action):
                label = _TASK_STATES.get(agent.get("state"), ("未知", "dim"))[0]
                self.info(
                    f"子代理 {agent_id} · {label} · {agent['title']} · {self._agent_activity(agent)}"
                )
        self._refresh_task()

    def tasks(self) -> None:
        self._clear_status()
        self._rich().print(self._dashboard(compact=False))

    def restore_task(
        self, title: str, plan: dict[str, Any] | None, *, completed: bool, status: str | None = None
    ) -> None:
        if not title:
            return
        self._task_title = title
        self._todos = [dict(item) for item in (plan or {}).get("steps", [])]
        self._task_state = status or ("completed" if completed else "restored")
        self.tasks()

    def reset_session(
        self,
        session_file: Path,
        *,
        context: dict[str, Any] | None = None,
        history_count: int = 0,
        session_label: str | None = None,
    ) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._clear_status()
        self._response_content.clear()
        self._response_reasoning.clear()
        self._clear_preview()
        self._last_progress = 0.0
        self._last_phase = ""
        self._last_retry_attempt = None
        self._task_started = None
        self._task_elapsed = 0.0
        self._task_title = ""
        self._task_state = "pending"
        self._task_activity = ""
        self._todos.clear()
        self._agents.clear()
        self._team = None
        self._prompt_session = None
        self._context = dict(context) if context is not None else None
        self._performance.clear()
        self._tool_count = 0
        self._tool_errors = 0
        self._active_tool_number = 0
        if self.interactive:
            sys.stdout.write("\033[2J\033[H" + ("" if self._screen_active else "\033[3J"))
            sys.stdout.flush()
        self.banner(session_file, history_count, session_label=session_label)

    def task_start(self, title: str = "") -> None:
        self._last_tool_progress = None
        self._task_title = _ellipsize(" ".join(title.split()), 120) or "当前任务"
        self._task_state = "running"
        self._task_activity = "准备执行"
        self._task_elapsed = 0.0
        self._todos.clear()
        self._agents.clear()
        self._performance.clear()
        self._tool_count = 0
        self._tool_errors = 0
        self._clear_preview()
        self._task_started = time.monotonic()
        if self.interactive:
            self._live = TaskLive(
                console=self._rich(),
                get_renderable=self._live_renderable,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start(refresh=True)
        else:
            label = _ellipsize(self._task_title, max(8, min(48, self.terminal_width() - 8)))
            self._rich().print(Text(f"  ◇ {label}", style="bold cyan"))

    def task_end(self, completed: bool, *, failed: bool = False, status: str | None = None) -> None:
        self._clear_status()
        self._clear_preview()
        self._task_state = "error" if failed else status or ("completed" if completed else "paused")
        completed = self._task_state == "completed"
        self._task_activity = ""
        if self._task_started is not None:
            self._task_elapsed = time.monotonic() - self._task_started
        background = (self._team or {}).get("background") == "running"
        continuing = []
        for agent_id, agent in tuple(self._agents.items()):
            if agent.get("state") in {"pending", "running"}:
                if background and agent.get("mode") == "team":
                    continuing.append(agent_id)
                    self._agents[agent_id] = {**agent, "state": "background"}
                    continue
                self._agents[agent_id] = {
                    **agent,
                    "state": "cancelled",
                    "activity": "finished",
                    "duration_seconds": time.monotonic() - agent["started_at"],
                }
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._todos or self._agents:
            self._rich().print(self._dashboard())
        if completed and self._task_started is not None and not (self._todos or self._agents):
            elapsed = _duration(time.monotonic() - self._task_started)
            metrics = " · ".join(
                part for part in (self._context_label(), self._tps_label()) if part
            )
            print(self.style(f"  ✓ 完成 · {elapsed} · {metrics}", GREEN))
        elif self._task_started is not None and not (self._todos or self._agents):
            label, _ = _TASK_STATES.get(self._task_state, (self._task_state, "yellow"))
            print(
                self.style(
                    f"  ◇ {label} · {_duration(time.monotonic() - self._task_started)}", YELLOW
                )
            )
        if background:
            team = self._team or {}
            names = list(
                dict.fromkeys(continuing + team.get("working", []) + team.get("queued", []))
            )
            self.info(
                "团队成员在后台继续工作"
                + (f"（{'、'.join(names)}）" if names else "")
                + "；完成时会在这里提示，/env results 查看结果。"
            )
        self._task_started = None

    def assistant_start(self) -> None:
        self._tool_parameter_progress = False
        self._clear_status()
        self._response_number += 1
        self._response_content.clear()
        self._response_reasoning.clear()
        self._clear_preview()
        self._last_progress = 0.0
        self._last_phase = ""
        self._last_retry_attempt = None
        if self._live is not None:
            self._task_activity = "正在连接模型 · Esc 暂停"
            self._refresh_task()
            return
        if self.interactive:
            print()
            self._set_status(f"  ◇ {self.model} · 正在连接… · Esc 暂停")
        else:
            print(self.style(f"\n  ◇ {self.model} · 正在处理…", DIM), flush=True)

    def progress(self, event) -> None:
        phase = event.get("phase")
        if phase == "response_reset":
            self._tool_parameter_progress = False
            self._response_number += 1
            self._clear_preview()
            if self._live is not None:
                self._task_activity = "正在重新生成 · Esc 暂停"
            self._refresh_task()
            return
        if phase == "tool_call_delta":
            self._tool_parameter_progress = True
            self._task_activity = f"正在生成工具参数 · {event.get('tool') or '工具'} · {event.get('argument_chars', 0):,} 字符"
            self._refresh_task()
            return
        if phase == "tool_progress":
            elapsed = _duration(event.get("elapsed_seconds", 0))
            received = event.get("received_bytes", 0) / 1024
            text = f"{_TOOL_LABELS.get(event.get('tool'), event.get('tool') or '工具')} · {elapsed} · {received:.1f} KB"
            output = plain(event.get("output") or "").strip().splitlines()
            if output:
                text += f" · {output[-1]}"
            self._task_activity = _ellipsize(text, max(8, self.terminal_width() - 8))
            if self.interactive:
                self._refresh_task()
            else:
                now = time.monotonic()
                if self._last_tool_progress is None or now - self._last_tool_progress >= 30:
                    self._last_tool_progress = now
                    self.info(self._task_activity)
            return
        if phase == "response_delta":
            if self.interactive:
                if event.get("content"):
                    self._preview_content.append(event["content"])
                if event.get("reasoning"):
                    self._preview_reasoning.append(event["reasoning"])
                if self._live is not None:
                    if self._preview_content:
                        self._task_activity = "正在回答 · Esc 暂停"
                    elif self._preview_reasoning:
                        self._task_activity = "正在思考 · Esc 暂停"
                self._refresh_task()
            return
        if phase == "warning":
            self.warning(event["message"])
            return
        if phase == "context":
            self.update_context(event)
            return
        if phase == "generation":
            self._performance[event.get("generation_id") or event["request_id"]] = dict(event)
            self._refresh_task()
            return
        if phase == "compaction":
            unit = "token" if event["unit"] == "tokens" else "字符"
            self.info(
                f"上下文已压缩：{event['before']:,} → {event['after']:,} {unit}；历史缓存将重新建立。"
            )
            return

        if not self.interactive:
            attempt = event.get("attempt")
            if phase == "rate_limit" and self._last_phase != phase:
                self.info("模型服务限流，正在等待恢复后重试…")
            self._last_phase = phase or ""
            if phase == "retry" and attempt != self._last_retry_attempt:
                self._last_retry_attempt = attempt
                self.info(f"连接暂时失败，正在进行第 {attempt or 2} 次请求…")
            return

        now = time.monotonic()
        if phase == self._last_phase and now - self._last_progress < 1:
            return
        self._last_progress = now
        self._last_phase = phase or ""
        if (
            phase == "receiving"
            and self._live is not None
            and (self._preview_content or self._preview_reasoning or self._tool_parameter_progress)
        ):
            return
        if phase == "rate_limit":
            delay = event.get("delay_seconds", 0)
            waiting = f"等待 {_duration(delay)}" if delay >= 1 else "等待服务恢复"
            text = f"模型服务限流 · {waiting} · Esc 暂停"
        elif phase == "retry":
            text = f"连接中断 · 第 {event.get('attempt', 2)} 次请求 · Esc 暂停"
        else:
            elapsed = _duration(event.get("elapsed_seconds", 0))
            if phase == "receiving":
                received = event.get("received_bytes", 0) / 1024
                text = f"正在接收 · {elapsed} · {received:.1f} KB · Esc 暂停"
            else:
                text = f"正在连接 · {elapsed} · Esc 暂停"
        self._set_status(text if self._live is not None else f"  ◇ {self.model} · {text}")

    def stream(self, text: str) -> None:
        self._clear_status()
        self._response_content.append(text)
        if self._live is not None:
            content = "".join(self._response_content)
            if content != "".join(self._preview_content):
                self._preview_content = self._response_content[:]
                self._task_activity = "正在回答 · Esc 暂停"
                self._refresh_task()

    def reasoning(self, text: str) -> None:
        self._clear_status()
        self._response_reasoning.append(text)
        if self._live is not None:
            reasoning = "".join(self._response_reasoning)
            if reasoning != "".join(self._preview_reasoning):
                self._preview_reasoning = self._response_reasoning[:]
                self._task_activity = "正在思考 · Esc 暂停"
                self._refresh_task()

    def assistant_end(self) -> None:
        self._clear_status()
        live = self._live
        with live.batch() if live is not None else nullcontext():
            reasoning = "".join(self._response_reasoning).strip("\n")
            content = "".join(self._response_content).strip("\n")
            if live is not None and (reasoning.strip() or content.strip()):
                height = max(4, shutil.get_terminal_size(fallback=(88, 24)).lines - 1)
                display = self._response_display(content, reasoning, self._dashboard(), height)
                self._clear_preview()
                self._task_activity = "本轮完成"
                live.finish_stream(display)
                self._response_reasoning.clear()
                self._response_content.clear()
                return
            self._clear_preview()
            sections = []
            if reasoning.strip():
                sections.extend(
                    [
                        Rule("思考过程", align="left", style="bright_black"),
                        self._markdown_renderable(reasoning, dim=True),
                    ]
                )
            if content.strip():
                label = f"{self.model} · 回答" if self.terminal_width() >= 40 else "回答"
                sections.extend(
                    [
                        Rule(label, align="left", style="cyan"),
                        self._markdown_renderable(content),
                    ]
                )
            if sections:
                self._rich().print(Group(*sections))
            else:
                self._refresh_task()
            self._response_reasoning.clear()
            self._response_content.clear()

    def tool(self, name: str, arguments: dict[str, Any]) -> None:
        self._last_tool_progress = None
        self._clear_status()
        if name == "ask_user":
            return
        self._tool_count += 1
        entry = self._current_execution()
        self._active_tool_number = entry.number if entry else self._active_tool_number + 1
        detail = self._tool_detail(name, arguments)
        self._task_activity = f"{_TOOL_LABELS.get(name, name)} · {detail}"
        self._refresh_task()
        available = max(8, self.terminal_width() - 18)
        detail = _ellipsize(detail, available) if detail else ""
        line = Text(f"  ├─ #{self._active_tool_number}  ", style="bright_black")
        line.append(_TOOL_LABELS.get(name, name), style="bold yellow")
        if detail:
            line.append(f"  {detail}", style="dim")
        self._rich().print(line)

    def tool_result(self, name: str, result: dict[str, Any]) -> None:
        if name == "ask_user":
            return
        status = result.get("status") or "error"
        if status != "ok":
            self._tool_errors += 1
        labels = {
            "ok": ("完成", "green"),
            "partial": ("部分完成", "yellow"),
            "timeout": ("超时", "red"),
            "unknown": ("结果未知", "yellow"),
            "error": ("失败", "red"),
        }
        label, style = labels.get(status, (str(status), "yellow"))
        data = result.get("data") or {}
        if status == "ok" and name == "monitor" and data.get("event") == "matched":
            label, style = "条件满足", "green"
        elif status == "ok" and data.get("state") in {"running", "waiting"}:
            label, style = "等待中" if name == "monitor" else "运行中", "cyan"
        detail = result_summary(result)
        entry = self._current_execution()
        duration = entry.run.get("duration_seconds") if entry else None
        if isinstance(duration, (int, float)):
            detail = f"{duration:.1f}s · {detail}".rstrip(" ·")
        detail = _ellipsize(detail.replace("\n", " "), max(8, self.terminal_width() - 18))
        line = Text("  └─ ", style="bright_black")
        line.append(label, style=f"bold {style}")
        if detail:
            line.append(f"  {detail}", style="dim")
        self._rich().print(line)
        if name in {"exec", "monitor"}:
            output = (result.get("data") or {}).get("output")
            if isinstance(output, str) and output:
                lines = plain(output).splitlines()
                visible = lines if len(lines) <= 3 else [*lines[:2], lines[-1]]
                for index, text in enumerate(visible):
                    if len(lines) > 3 and index == 2:
                        self._rich().print(Text(f"      … 省略 {len(lines) - 3} 行", style="dim"))
                    self._rich().print(
                        Text(
                            "      " + _ellipsize(text, max(8, self.terminal_width() - 8)),
                            style="dim",
                        )
                    )
        number = entry.number if entry else self._active_tool_number
        if number:
            suffix = " · o 打开保留文件" if result.get("artifact_path") else ""
            self._rich().print(Text(f"      /tool {number} 查看详情{suffix}", style="dim"))

    def _current_execution(self) -> ExecutionEntry | None:
        execution_id = current_trace().execution_id
        if execution_id and self.execution_provider:
            return next(
                (
                    entry
                    for entry in self.execution_provider()
                    if entry.run.get("execution_id") == execution_id
                ),
                None,
            )
        return None

    def trace(self, *, selected: int | None = None, query: str = "", errors: bool = False) -> None:
        entries = self.execution_provider() if self.execution_provider else []
        if not entries:
            self.info("当前会话暂无工具执行记录。")
            return
        if self.interactive:
            from trace_ui import TraceInspector

            with self._suspend_live(), patch_stdout(raw=True):
                TraceInspector(
                    entries,
                    selected=selected,
                    query=query,
                    errors=errors,
                    colors=self.colors,
                    full_screen=not self._screen_active,
                ).run()
            return
        if selected is not None:
            self._rich().print(Text(entries[selected].detail(entries)))
            return
        self._section("执行轨迹 · /tool 编号 查看详情", style="cyan")
        for entry in entries:
            if query and not all(word in entry.searchable for word in query.casefold().split()):
                continue
            if errors and entry.status in {"ok", "completed", "pending", "running"}:
                continue
            label = (
                "完成"
                if entry.status == "ok"
                else _TASK_STATES.get(entry.status, (entry.status, "dim"))[0]
            )
            self._rich().print(
                Text(
                    f"  #{entry.number} · {entry.agent} · {entry.run.get('tool')} · {label}\n"
                    f"    {_ellipsize(entry.target, max(8, self.terminal_width() - 6))}\n"
                    f"    {result_summary(entry.result)}",
                    style="dim",
                )
            )

    def success(self, message: str) -> None:
        self._clear_status()
        print(self.style(f"  ✓ {message}", GREEN))

    def error(self, error: Exception) -> None:
        self._clear_status()
        print(self.style(f"\n  ✗ {error}", RED), file=sys.stderr)

    def info(self, message: str) -> None:
        self._clear_status()
        print(self.style(f"  • {message}", BLUE))

    def warning(self, message: str) -> None:
        self._clear_status()
        print(self.style(f"  ! {message}", YELLOW))

    def cache(self, stats, *, compact: bool = False) -> None:
        def rate(value, digits=2):
            return f"{value:.{digits}%}" if value is not None else "未知"

        calls = stats.get("calls", 0)
        if not calls:
            if not compact:
                self.info("缓存：暂无请求记录")
            return
        latest = stats.get("latest") or {}
        purpose = {"solve": "生成", "final": "收尾", "summary": "摘要"}.get(
            latest.get("purpose"), ""
        )
        latest_label = f"最近一次（{purpose}）" if purpose else "最近一次"
        if compact:
            self.info(
                f"缓存 {rate(latest.get('hit_rate'), 1)} · 会话 "
                f"{rate(stats.get('hit_rate'), 1)}"
                f" · {stats['observed_calls']}/{calls} 次已统计"
            )
            return
        self.info(
            f"缓存：{latest_label} {rate(latest.get('hit_rate'))} · 会话累计 "
            f"{rate(stats.get('hit_rate'))} · 已知 {stats['observed_calls']}/{calls} 次请求"
            f" · 未知 {stats['unknown_calls']} 次"
        )
        scopes = stats.get("scopes") or {}
        for label, entry in list(scopes.items())[:16]:
            digest = entry.get("system_fingerprint")
            fingerprint_label = f" · 系统/工具 {digest[:12]}" if digest else ""
            self.info(
                f"  {label}：{rate(entry.get('hit_rate'))} · "
                f"{entry['observed_calls']}/{entry['calls']} 次"
                f" · 系统/工具变化 {entry.get('system_changes', 0)} 次{fingerprint_label}"
            )
        if len(scopes) > 16:
            self.info(f"  其余 {len(scopes) - 16} 个分组已保存在会话统计中")

    def history(self, messages: list[dict[str, Any]], limit: int | None = 10) -> None:
        visible = [message for message in messages if message.get("role") in {"user", "assistant"}]
        selected = visible if limit is None else visible[-limit:]
        hidden = len(visible) - len(selected)
        self._clear_status()
        label = f"最近 {len(selected)}/{len(visible)} 条" if hidden else f"{len(selected)} 条"
        print()
        self._section(f"对话记录 · {label}", style="cyan")
        if any(message.get("tool_calls") for message in selected):
            self._rich().print(
                Text("  /trace 查看执行轨迹，/tool 编号 查看参数和结果。", style="dim")
            )
        if hidden:
            print(self.style("  使用 /history all 查看全部，或 /history 20 指定数量。", DIM))
        if not selected:
            print(self.style("  暂无对话。\n", DIM))
            return

        def history_text(value: Any) -> str:
            text = value if isinstance(value, str) else ""
            if limit is not None and len(text) > 3600:
                return text[:3600].rstrip() + "\n… [内容过长；使用 /history all 查看完整记录]"
            return text

        for message in selected:
            is_user = message["role"] == "user"
            content = history_text(message.get("content"))
            print()
            if is_user:
                self._section("你", style="bright_black")
                self._markdown(content)
                continue
            reasoning = history_text(_reasoning_text(message))
            if reasoning:
                self._section("思考过程", style="bright_black")
                self._markdown(reasoning, dim=True)
            if content:
                self._section("回答", style="cyan")
                self._markdown(content)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                names = [call.get("function", {}).get("name") or "未知工具" for call in tool_calls]
                self._rich().print(Text(f"  工具调用  {' · '.join(names)}", style="dim yellow"))
            elif not reasoning and not content:
                self._rich().print(Text("  [空响应]", style="dim"))
        print()

    @staticmethod
    def _tool_detail(name: str, arguments: dict[str, Any]) -> str:
        if name == "update_plan":
            return str(arguments.get("title") or "")
        if name == "delegate":
            tasks = arguments.get("tasks")
            return f"{len(tasks)} 个任务" if isinstance(tasks, list) else ""
        if name == "mcp":
            values = [
                arguments.get("action"),
                arguments.get("tool") or arguments.get("query") or arguments.get("server"),
            ]
            return " · ".join(str(value) for value in values if value).replace("\n", " ")
        if name == "team":
            target = (
                arguments.get("to")
                or arguments.get("assignee")
                or arguments.get("participants")
                or arguments.get("task_id")
                or arguments.get("protocol_id")
                or arguments.get("protocol")
            )
            target = ", ".join(target) if isinstance(target, list) else target
            values = [arguments.get("action"), f"→ {target}" if target else None]
            return " ".join(str(value) for value in values if value).replace("\n", " ")
        keys = {
            "read": ("path",),
            "exec": ("command",),
            "monitor": ("action", "session_id", "path", "wait_for"),
            "edit": ("path",),
            "write": ("path",),
            "grep": ("pattern", "path"),
            "websearch": ("query",),
            "find": ("pattern", "path"),
            "ls": ("path",),
        }.get(name, ())
        values = [str(arguments[key]) for key in keys if arguments.get(key)]
        return " · ".join(values).replace("\n", " ")
