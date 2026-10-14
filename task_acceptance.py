"""Task-specific acceptance and tool evidence, separate from answer formatting.

Automatic intent detection is conservative. Callers with a known task should
pass kind='change'/'answer'/'inspect' and deterministic checks explicitly.
"""

import ast
import json
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from file_state import SOURCE_NAMES, SOURCE_SUFFIXES, file_digest, source_snapshot


@dataclass
class TaskContract:
    kind: str = "auto"
    checks: tuple[Callable[[], str | None], ...] = ()
    description: str = ""


@dataclass
class TaskAcceptance:
    prompt: str
    contract: TaskContract = field(default_factory=TaskContract)
    has_tools: bool = True
    workspace: Path | None = None
    ignored_paths: tuple[Path, ...] = ()
    deadline: float | None = None

    def __post_init__(self):
        self.kind = self.contract.kind
        if self.kind not in {"auto", "answer", "inspect", "change"}:
            raise ValueError("Task kind must be auto, answer, inspect or change")
        if self.kind == "auto":
            action = re.search(
                r"修复|修一下|实现|重构|新增|加一个|修改.{0,20}(?:代码|文件)|\b(?:fix|implement|refactor)\b",
                self.prompt,
                re.I,
            )
            discussion = re.search(
                r"分析|解释|怎么|如何|计划|不要修改|翻译|复述|引用|\b(?:explain|analy[sz]e|how to|review|plan|translate|quote)\b",
                self.prompt,
                re.I,
            )
            self.kind = "change" if self.has_tools and action and not discussion else "answer"
        self.status = "in_progress"
        self.evidence = []
        self.check_results = []
        self.last_code_change = -1
        self.last_verification = -1
        self.code_revision = 0
        self.verified_revision = -1
        self.failed_verifications = {}
        self.pending_verifications = {}
        self._verification_start = 0
        self._roots = {self.workspace.resolve()} if self.workspace is not None else set()
        self._extra_files = set()
        self._tracking_error = None
        self._snapshot_cache = {}
        self._files = self._snapshot() if self.kind == "change" and self._roots else None

    def _snapshot(self):
        files = source_snapshot(
            self._roots,
            ignored=self.ignored_paths,
            deadline=self.deadline,
            cache=self._snapshot_cache,
        )
        for path in self._extra_files:
            digest = file_digest(path, self._check_deadline)
            if digest is not None:
                files[str(path)] = digest
        return files

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Task deadline reached while tracking file changes")

    def _refresh_files(self):
        if self._files is None:
            return []
        try:
            current = self._snapshot()
        except OSError as exc:
            self._tracking_error = str(exc)
            return []
        self._tracking_error = None
        changed = sorted(
            path
            for path in self._files.keys() | current.keys()
            if self._files.get(path) != current.get(path)
        )
        if changed:
            self.code_revision += 1
            self.last_code_change = len(self.evidence)
        self._files = current
        return changed

    def before_tool(self, call):
        self._refresh_files()
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        if self._files is not None and isinstance(arguments, dict):
            if function.get("name") == "exec" and isinstance(arguments.get("cwd"), str):
                root = (self.workspace / Path(arguments["cwd"]).expanduser()).resolve()
                if not any(root.is_relative_to(existing) for existing in self._roots):
                    self._files.update(
                        source_snapshot((root,), deadline=self.deadline, cache=self._snapshot_cache)
                    )
                    self._roots.add(root)
            if function.get("name") in {"write", "edit"} and isinstance(arguments.get("path"), str):
                path = (self.workspace / Path(arguments["path"]).expanduser()).resolve()
                if path.suffix.lower() in SOURCE_SUFFIXES or path.name in SOURCE_NAMES:
                    if str(path) not in self._files:
                        digest = file_digest(path, self._check_deadline)
                        if digest is not None:
                            self._files[str(path)] = digest
                    self._extra_files.add(path)
        self._verification_start = self.code_revision

    def instructions(self):
        parts = []
        if self.kind == "change":
            parts.append(
                "Implementation task: use tools, then verify the final edit with a relevant test, build, or assertion. "
                "If no change is needed, cite inspected evidence; if blocked, state the blocker and unfinished work."
            )
        if self.contract.description:
            parts.append("Acceptance criteria: " + self.contract.description)
        return "\n".join(parts)

    @staticmethod
    def verification_command(command):
        try:
            words = shlex.split(command)
        except ValueError:
            return False
        if not words:
            return False
        if any(flag in words for flag in {"--version", "--help", "-h", "--collect-only"}):
            return False
        executable = Path(words[0]).name
        # An echo of a command or an arbitrary successful shell command is not a check.
        if executable in {"pytest", "py.test", "mypy", "ruff", "tsc"}:
            return True
        if executable.startswith("python"):
            if len(words) > 2 and words[1] == "-c":
                try:
                    return any(
                        isinstance(node, ast.Assert) for node in ast.walk(ast.parse(words[2]))
                    )
                except SyntaxError:
                    return False
            return (
                len(words) > 2
                and words[1] == "-m"
                and words[2] in {"unittest", "pytest", "compileall"}
            )
        return executable in {"npm", "pnpm", "yarn", "bun", "cargo", "go", "make", "cmake"} and any(
            word in {"test", "check", "build", "lint", "typecheck", "--build"} for word in words[1:]
        )

    def observe(self, call, result):
        self.status = "in_progress"
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        record = {
            "tool_call_id": call.get("id"),
            "tool": function.get("name"),
            "status": result.get("status"),
            "artifact_path": result.get("artifact_path"),
        }
        self.evidence.append(record)
        index = len(self.evidence) - 1
        changed = self._refresh_files()
        if changed:
            record["changed_files"] = changed
        if (
            self._files is None
            and result.get("status") == "ok"
            and function.get("name") in {"write", "edit"}
            and (
                Path(arguments.get("path", "")).suffix.lower() in SOURCE_SUFFIXES
                or Path(arguments.get("path", "")).name in SOURCE_NAMES
            )
        ):
            self.code_revision += 1
            self.last_code_change = index
        record["code_revision"] = self.code_revision
        verification_start = self._verification_start
        verification_result = result
        data = result.get("data") or {}
        if function.get("name") == "monitor":
            pending = self.pending_verifications.get(data.get("session_id"))
            if pending and data.get("state") in {"completed", "failed", "timed_out", "stopped"}:
                arguments, verification_start = self.pending_verifications.pop(data["session_id"])
                record["verification_session_id"] = data["session_id"]
                verification_result = {"status": "ok" if data["state"] == "completed" else "error"}
        is_verification = (
            function.get("name") == "exec" or "verification_session_id" in record
        ) and self.verification_command(arguments.get("command", ""))
        if is_verification and data.get("state") == "running" and data.get("session_id"):
            self.pending_verifications[data["session_id"]] = (dict(arguments), verification_start)
            record["verification_pending"] = True
        elif is_verification:
            record["verification"] = True
            identity = json.dumps(
                [
                    str((self.workspace / arguments.get("cwd", ".")).resolve())
                    if self.workspace is not None
                    else arguments.get("cwd", "."),
                    shlex.split(arguments.get("command", "")),
                ],
                ensure_ascii=False,
            )
            if verification_result.get("status") != "ok":
                self.failed_verifications[identity] = record
                self.verified_revision = -1
            else:
                lexer = shlex.shlex(
                    arguments.get("command", "").strip(), posix=True, punctuation_chars=";&|\n"
                )
                lexer.whitespace = " \t\r"
                lexer.commenters = ""
                lexer.whitespace_split = True
                standalone = not any(token and set(token) <= set(";&|\n") for token in lexer)
                if not standalone:
                    self.verified_revision = -1
                    record["verification_untrusted"] = (
                        "Run the check as a standalone command; shell composition may hide failures."
                    )
                else:
                    self.failed_verifications.pop(identity, None)
                if standalone and (self._files is None or verification_start == self.code_revision):
                    self.last_verification = index
                    self.verified_revision = self.code_revision

    def validate(self, content):
        self.status = "in_progress"
        text = content.strip()
        blocked = re.search(
            r"(?:无法|不能|未能).{0,12}(?:完成|继续|执行)|需要你.{0,20}(?:提供|确认)|\b(?:blocked by|cannot proceed|unable to complete|need your clarification)\b",
            text,
            re.I,
        )
        work_task = (
            self.kind in {"change", "inspect"} or bool(self.evidence) or bool(self.contract.checks)
        )
        if blocked and work_task:
            self.status = "blocked"
            return None
        promise = re.match(
            r"^(?:好的[，。！\s]*|可以[，。！\s]*)?(?:我(?:会|将|先)|接下来我|I(?:'ll| will| am going to)|Let me)\s*",
            text,
            re.I,
        )
        if (
            work_task
            and promise
            and len(text) < 400
            and "```" not in text
            and not re.search(
                r"已完成|已修复|已修改|结果[：:]|\b(?:verified|implemented|fixed)\b", text, re.I
            )
        ):
            return "The response only promises future work. Perform the next action or give a substantive answer or concrete blocker."
        if self.kind in {"change", "inspect"} and not any(
            e["status"] == "ok" for e in self.evidence
        ):
            return "No successful tool evidence supports completion of this task. Inspect and perform the requested work, or state the concrete blocker."
        self._refresh_files()
        if self.kind == "change" and self._tracking_error:
            return (
                "Cannot verify the final code revision because file tracking failed: "
                + self._tracking_error
            )
        if self.kind == "change" and self.failed_verifications:
            return "Verification failed and remains unresolved. Rerun the failed check successfully, or explicitly report the concrete validation blocker."
        if self.kind == "change" and self.pending_verifications:
            return "Background verification is still running. Use monitor to observe its final result before reporting completion."
        if (
            self.kind == "change"
            and self.code_revision > 0
            and self.verified_revision != self.code_revision
            and not self.contract.checks
        ):
            return "Code was changed but no successful verification followed the last edit. Run a relevant standalone test/build/assertion, or explicitly report why validation is blocked."
        self.check_results = []
        for check in self.contract.checks:
            try:
                problem = check()
            except Exception as exc:
                problem = f"{type(exc).__name__}: {exc}"
            self.check_results.append(
                {
                    "name": getattr(check, "__name__", "acceptance"),
                    "passed": problem is None,
                    "detail": problem,
                }
            )
            if problem:
                return "Task acceptance failed: " + problem
        self.status = "completed"
        return None

    def outcome(self):
        return {
            "kind": self.kind,
            "status": self.status,
            "evidence": self.evidence,
            "checks": self.check_results,
            "code_revision": self.code_revision,
            "verified_revision": self.verified_revision,
        }
