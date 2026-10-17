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

# The agent cannot go on, or it waits on the user.
INABILITY = re.compile(
    r"(?:无法|不能|未能).{0,12}(?:完成|继续|执行)"
    r"|\b(?:blocked by|cannot proceed|unable to complete)\b",
    re.I,
)
NEEDS_USER = re.compile(r"需要你.{0,20}(?:提供|确认)|\bneed your clarification\b", re.I)
UNTRUSTED_CHECK = "Run the check as a standalone command; shell composition may hide failures."
# Quoted and code text reports what something said or contains, not the agent's own state.
QUOTED = re.compile(r"```.*?```|`[^`\n]*`|“[^”]*”|「[^」]*」|‘[^’]*’|\"[^\"\n]{0,300}\"", re.S)
SENTENCE_END = re.compile(r"[。！？!?；;\n]|\.(?:\s|$)")
# A name right after these is an object ("我让 critic 运行测试"), not the subject.
OBJECT_MARKERS = re.compile(
    r"(?:(?:让|请|叫|派|要求|安排|交给|通知|提醒|问|向|给|和|与|跟|对|把|被)"
    r"|\b(?:asked|told|let|had|with|to|for|from|by)(?:\s+the)?)\s*$",
    re.I,
)
# A name right before these only says where something came from ("用户提供的文件无法执行"):
# that thing, not the name, is what cannot go on. A bare 的 ("critic 的环境") stays the owner's.
ATTRIBUTIVE = re.compile(
    r"\s*(?:提供|给出|给|上传|发来|发送|提交|指定|要求|写|编写|生成|创建|准备)"
    r"(?:了|过)?(?:我们|我|你)?的"
)


def _words(pattern: str) -> str:
    # ASCII boundaries: Chinese text often puts Latin names right next to CJK characters.
    return rf"(?<![A-Za-z0-9_-])(?:{pattern})(?![A-Za-z0-9_-])"


SELF_SUBJECT = r"我|咱们|" + _words(r"I|I'm|I've|we|we're|we've")
# 其他/其它 ("other …") are not the pronouns 他/它.
OTHER_SUBJECT = (
    r"(?<!其)他|她|(?<!其)它|对方|用户|成员|队友|子代理|其他代理|另一个代理|评审者|审阅者|评审员|"
    + _words(r"he|she|they|the user|teammates?|members?|reviewers?|workers?|subagents?")
)
# A limitation already overcome ("at first the tests could not run") is no blocker.
# Not 之前: "在你确认之前" means "until you confirm".
EARLIER = r"此前|先前|起初|最初|一开始|原先|原本|曾经|" + _words(
    r"previously|initially|at first|earlier"
)


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
    # Other agents (team members) whose limitations the answer may report.
    others: tuple[str, ...] = ()

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
        # Evidence indexes of this agent's own latest successful source write/edit and of
        # where its latest passing standalone check started (a background check finishes
        # later), whatever the code revision then was.
        self.last_own_edit = -1
        self.last_passed_check = -1
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
        own_source_edit = (
            result.get("status") == "ok"
            and function.get("name") in {"write", "edit"}
            and (
                Path(arguments.get("path", "")).suffix.lower() in SOURCE_SUFFIXES
                or Path(arguments.get("path", "")).name in SOURCE_NAMES
            )
        )
        if own_source_edit:
            self.last_own_edit = index
        changed = self._refresh_files()
        if changed:
            record["changed_files"] = changed
        if self._files is None and own_source_edit:
            self.code_revision += 1
            self.last_code_change = index
        record["code_revision"] = self.code_revision
        verification_start = self._verification_start
        started_at = index
        verification_result = result
        data = result.get("data") or {}
        if function.get("name") == "monitor":
            pending = self.pending_verifications.get(data.get("session_id"))
            if pending and data.get("state") in {"completed", "failed", "timed_out", "stopped"}:
                arguments, verification_start, started_at = self.pending_verifications.pop(
                    data["session_id"]
                )
                record["verification_session_id"] = data["session_id"]
                verification_result = {"status": "ok" if data["state"] == "completed" else "error"}
        is_verification = (
            function.get("name") == "exec" or "verification_session_id" in record
        ) and self.verification_command(arguments.get("command", ""))
        if is_verification and data.get("state") == "running" and data.get("session_id"):
            self.pending_verifications[data["session_id"]] = (
                dict(arguments),
                verification_start,
                index,
            )
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
                segments = self.shell_segments(arguments.get("command", ""))
                standalone = segments is not None and len(segments) == 1
                if not standalone:
                    self.verified_revision = -1
                    record["verification_untrusted"] = UNTRUSTED_CHECK
                else:
                    self.failed_verifications.pop(identity, None)
                    self.last_passed_check = started_at
                if standalone and (self._files is None or verification_start == self.code_revision):
                    self.last_verification = index
                    self.verified_revision = self.code_revision
        elif function.get("name") == "exec" and any(
            self.verification_command(shlex.join(segment))
            for segment in self.shell_segments(arguments.get("command", "")) or ()
        ):
            # A check hidden behind "cd … &&" is no verification, but say why it is not.
            record["verification_untrusted"] = UNTRUSTED_CHECK

    @staticmethod
    def shell_segments(command):
        """The command's words split at shell composition (;, &, &&, |, ||, newlines).

        "<" and ">" are punctuation too, so redirections such as 2>&1 or &>log lex as
        tokens containing them and never count as composition; quoted text stays a word.
        None when the command cannot be lexed (e.g. a quote inside a shell comment), so
        its composition is unknown.
        """
        lexer = shlex.shlex(command.strip(), posix=True, punctuation_chars=";&|<>\n")
        lexer.whitespace = " \t\r"
        lexer.commenters = ""
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            return None
        segments = [[]]
        for token in tokens:
            if token and set(token) <= set(";&|\n"):
                segments.append([])
            else:
                segments[-1].append(token)
        return segments

    def reports_blocker(self, text):
        """True when the answer says the agent itself is blocked or waits on the user.

        Inability phrases are attributed to the last subject named earlier in their
        sentence, since Chinese often omits a repeated subject: "the critic is read-only
        and cannot proceed" reports a teammate's limitation, not this agent's. A name that
        is an object or only modifies a noun is no subject. Without a named subject the
        agent is speaking about itself.
        """
        text = QUOTED.sub(" ", text)
        if NEEDS_USER.search(text):
            return True
        names = "|".join(re.escape(name) for name in self.others if name)
        subject = re.compile(
            rf"(?P<self>{SELF_SUBJECT})|(?P<other>{OTHER_SUBJECT}"
            + (f"|{_words(names)}" if names else "")
            + ")",
            re.I,
        )
        for match in INABILITY.finditer(text):
            ends = list(SENTENCE_END.finditer(text, 0, match.start()))
            sentence = text[ends[-1].end() if ends else 0 : match.start()]
            if re.search(EARLIER, sentence, re.I):
                continue
            subjects = [
                found
                for found in subject.finditer(sentence)
                if found.group("self")
                or not (
                    OBJECT_MARKERS.search(sentence[: found.start()])
                    or ATTRIBUTIVE.match(sentence, found.end())
                )
            ]
            if not subjects or subjects[-1].group("self"):
                return True
        return False

    def validate(self, content, *, background=False):
        """None when the answer may be published; otherwise what to repair.

        ``background`` means delegated work keeps running after this turn: its edits are
        not final yet, so they are not required to be verified now.
        """
        self.status = "in_progress"
        text = content.strip()
        blocked = self.reports_blocker(text)
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
        if background:
            # Teammates' unfinished edits are not final yet, but this agent's own are.
            if self.kind == "change" and self.last_passed_check < self.last_own_edit:
                return (
                    "You changed files yourself: verify your own change with a relevant "
                    "standalone test/build/assertion before answering, even though members "
                    "keep working in the background."
                )
            # Contract checks judge finished work; this turn hands work off instead.
            self.status = "background"
            return None
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
            if any(
                record.get("verification_untrusted")
                for record in self.evidence[max(self.last_code_change, 0) :]
            ):
                return (
                    "Code was changed but no successful verification followed the last edit: "
                    "the check you ran was not counted because it was combined with other shell "
                    "commands (&&, ;, |, newlines), which can hide a failure. Run the check alone "
                    "in its own exec call, with the cwd argument instead of cd."
                )
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
