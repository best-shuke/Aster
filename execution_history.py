"""Read-only execution records and bounded access to retained tool output."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

_CONTROLS = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]")
ARTIFACT_PAGE_CHARS = 32_000
# Tool call status as shown in the transcript and the trace inspector: icon, tone, label.
STATUS_DISPLAY = {
    "ok": ("✓", "ok", "完成"),
    "completed": ("✓", "ok", "完成"),
    "running": ("◷", "accent", "运行中"),
    "pending": ("○", "muted", "待执行"),
    "partial": ("◐", "warn", "部分完成"),
    "timeout": ("✗", "error", "超时"),
    "unknown": ("?", "warn", "结果未知"),
    "error": ("✗", "error", "失败"),
    "not_executed": ("○", "warn", "未执行"),
}
HEALTHY_STATUSES = frozenset({"ok", "completed", "pending", "running"})
# Background command session states, in the same language as the rest of the display.
PROCESS_STATES = {
    "running": "运行中",
    "waiting": "等待中",
    "ready": "条件满足",
    "completed": "已结束",
    "failed": "失败退出",
    "timed_out": "超时终止",
    "stopped": "已停止",
}


def plain(value: Any) -> str:
    return _CONTROLS.sub("", str(value)).replace("\r\n", "\n")


def decoded(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return plain(value)
    return plain(json.dumps(value, ensure_ascii=False, indent=2))


def result_summary(result: dict, *, brief: bool = False) -> str:
    """One line of result facts; ``brief`` is for a call line that already shows its state."""
    data = result.get("data") or {}
    parts = []
    if data.get("session_id"):
        session = str(data["session_id"])
        state = data.get("state", "unknown")
        label = PROCESS_STATES.get(state, state)
        if brief:
            # The call line labels a healthy running or waiting session already.
            shown = result.get("status") == "ok" and state in {"running", "waiting"}
            parts.append(f"进程 {session[:8]}" + ("" if shown else f" · {label}"))
        else:
            parts.append(f"进程 {session} · {label}")
    if data.get("event") == "waiting":
        parts.append("条件未满足")
    if data.get("exit_code") is not None:
        parts.append(f"exit {data['exit_code']}")
    for key, label in (
        ("lines", "行"),
        ("matches", "条匹配"),
        ("entries", "项"),
        ("files", "个文件"),
        ("results", "条结果"),
    ):
        value = data.get(key)
        if isinstance(value, list):
            parts.append(f"{len(value)} {label}")
            break
    if "bytes_written" in data:
        parts.append(f"写入 {data['bytes_written']} bytes")
    if isinstance(data.get("content"), str):
        parts.append(f"{len(data['content'].splitlines())} 行")
    if isinstance(data.get("output"), str):
        parts.append(f"{len(data['output'].splitlines())} 行输出" if data["output"] else "无输出")
    error = result.get("error")
    if isinstance(error, dict) and error.get("message"):
        parts.append(str(error["message"]).replace("\n", " "))
    if result.get("truncated"):
        parts.append("当前为部分结果")
    return plain(" · ".join(parts))


@dataclass(frozen=True)
class ExecutionEntry:
    number: int
    run: dict

    @cached_property
    def arguments(self) -> dict:
        return decoded(self.run.get("arguments"))

    @cached_property
    def result(self) -> dict:
        return decoded(self.run.get("result"))

    @property
    def status(self) -> str:
        return str(self.result.get("status") or self.run.get("state") or "unknown")

    @property
    def agent(self) -> str:
        return (self.run.get("trace") or {}).get("agent_id") or "主代理"

    @cached_property
    def target(self) -> str:
        args = self.arguments
        values = [
            args.get(key)
            for key in (
                "command",
                "path",
                "pattern",
                "query",
                "tool",
                "session_id",
                "action",
                "wait_for",
                "title",
            )
        ]
        return plain(" · ".join(str(value) for value in values if value)).replace("\n", " ↵ ")

    @cached_property
    def searchable(self) -> str:
        return " ".join(
            (
                str(self.number),
                str(self.run.get("tool")),
                self.agent,
                self.status,
                self.target,
                str(self.run.get("task_title") or ""),
                result_summary(self.result),
            )
        ).casefold()

    def detail(self, entries: list["ExecutionEntry"]) -> str:
        run = self.run
        lines = [
            f"#{self.number}  {run.get('tool', '未知工具')}  ·  {self.status}",
            f"执行者：{self.agent}",
            f"任务：{run.get('task_title') or '旧记录未保存任务标题'}",
        ]
        started = run.get("started_at")
        if isinstance(started, (int, float)):
            lines.append(
                f"开始：{datetime.fromtimestamp(started).astimezone().isoformat(timespec='seconds')}"
            )
        duration = run.get("duration_seconds")
        if isinstance(duration, (int, float)):
            lines.append(f"耗时：{duration:.3f} 秒")
        trace = run.get("trace") or {}
        parent_id = trace.get("parent_execution_id")
        parent = next(
            (
                entry
                for entry in entries
                if parent_id and entry.run.get("execution_id") == parent_id
            ),
            None,
        )
        if parent:
            lines.append(f"来自：#{parent.number} {parent.run.get('tool')}")
        children = [
            f"#{entry.number} {entry.agent} {entry.run.get('tool')}"
            for entry in entries
            if (entry.run.get("trace") or {}).get("parent_execution_id") == run.get("execution_id")
            and run.get("execution_id")
        ]
        if children:
            lines.append("子调用：" + " · ".join(children))
        arguments = self.arguments
        argument_text = (
            pretty(run["arguments"])
            if "arguments" in run
            else ("旧记录未保留参数，当前对话中也无法定位原调用。")
        )
        if run.get("tool") == "exec" and isinstance(arguments.get("command"), str):
            argument_text = pretty(
                {key: value for key, value in arguments.items() if key != "command"}
            )
            lines.extend(
                [
                    "",
                    "── 命令 ──",
                    arguments["command"],
                    f"工作目录：{(self.result.get('data') or {}).get('cwd') or arguments.get('cwd', '.')}",
                ]
            )
        lines.extend(
            [
                "",
                "── 调用参数 ──",
                argument_text,
            ]
        )
        result = self.result
        data = result.get("data") or {}
        for key, label in (("output", "命令输出"), ("content", "返回内容")):
            if isinstance(data.get(key), str):
                lines.extend(["", f"── {label} ──", data[key] or "（无输出）"])
        lines.extend(
            [
                "",
                "── 结果字段 ──",
                pretty(
                    {
                        **result,
                        "data": {
                            key: value
                            for key, value in data.items()
                            if key not in {"output", "content"}
                        },
                    }
                )
                if result
                else pretty(run["result"])
                if "result" in run
                else "尚无已保存结果。",
            ]
        )
        if result.get("artifact_path"):
            lines.extend(
                ["", f"保留文件：{result['artifact_path']}", "按 o 打开保留文件；大文件按页读取。"]
            )
        if result.get("next_cursor"):
            lines.extend(["", "这是分页结果；next_cursor 表示尚未读取的后续内容。"])
        if trace:
            lines.extend(["", "── 关联标识 ──", pretty(trace)])
        return plain("\n".join(lines))


def execution_entries(runs: list[dict], messages: list[dict]) -> list[ExecutionEntry]:
    """Enrich legacy journals from messages without mutating either source."""
    entries = []
    for number, original in enumerate(runs, 1):
        run = dict(original)
        index = run.get("assistant_index")
        if isinstance(index, int) and 0 <= index < len(messages):
            message = messages[index]
            call = next(
                (
                    call
                    for call in message.get("tool_calls") or []
                    if call.get("id") == run.get("tool_call_id")
                ),
                None,
            )
            if call:
                run.setdefault("arguments", call.get("function", {}).get("arguments", "{}"))
                if "result" not in run:
                    for result in messages[index + 1 :]:
                        if result.get("role") != "tool":
                            break
                        if result.get("tool_call_id") == run.get("tool_call_id"):
                            run["result"] = result.get("content")
                            break
        entries.append(ExecutionEntry(number, run))
    return entries


class ArtifactPages:
    """Keep only one bounded UTF-8 page in memory; offsets are TextIO cookies."""

    def __init__(self, path: Path, page_chars: int = ARTIFACT_PAGE_CHARS):
        self.path = path
        self.page_chars = page_chars
        self.offsets = [0]
        self.index = 0
        self.has_next = False
        self.text = ""
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8", errors="replace") as stream:
            stream.seek(self.offsets[self.index])
            self.text = plain(stream.read(self.page_chars))
            next_offset = stream.tell()
            self.has_next = bool(stream.read(1))
        if self.has_next and len(self.offsets) == self.index + 1:
            self.offsets.append(next_offset)

    def move(self, direction: int) -> bool:
        next_index = self.index + direction
        if next_index < 0 or (direction > 0 and not self.has_next):
            return False
        previous_index = self.index
        self.index = next_index
        try:
            self._load()
        except (OSError, ValueError):
            self.index = previous_index
            raise
        return True
