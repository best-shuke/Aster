"""Task-specific acceptance and tool evidence, separate from answer formatting.

Automatic intent detection is conservative. Callers with a known task should
pass kind='change'/'answer'/'inspect' and deterministic checks explicitly.
"""

import ast
import hashlib
import json
import os
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from file_state import (
    CODE_SUFFIXES, GENERATED_DIRECTORIES, cached_file_digest, directory_identity, source_snapshot,
    verification_file,
)
from source_events import SourceEvents
from tool_runtime import command_environment

# The agent cannot go on, or it waits on the user.
INABILITY = re.compile(
    r"(?:无法|不能|未能).{0,12}(?:完成|继续|执行|验证)"
    r"|\b(?:blocked by|cannot proceed|unable to complete)\b",
    re.I,
)
NEEDS_USER = re.compile(r"需要你.{0,20}(?:提供|确认)|\bneed your clarification\b", re.I)
EXTERNAL_REQUIREMENT = re.compile(
    r"缺少.{0,15}(?:权限|凭据|密钥|材料|授权)|(?:你|用户).{0,16}(?:确认|提供|授权)"
    r"|请(?:你)?(?:确认|提供|授权).{0,16}(?:目标|文件|路径|材料|权限|凭据|密钥)"
    r"|\b(?:without|missing|need(?:s|ed)?)\b.{0,40}\b(?:credentials?|api key|permission|approval)\b"
    r"|\b(?:database|volume|filesystem) is read-only\b",
    re.I,
)
# The user lets the agent answer while delegated work goes on ("不要等待它完成", "don't wait").
# 等 must be about waiting for someone or end the clause: not 等宽, 不要等太久 or 区别等.
HANDOFF = re.compile(
    r"(?:不要|不用|不必|无需|不需要|(?<![区识级类派告辨鉴性组差特分])别)再?等(?:待)?"
    r"(?:它|他|她|他们|她们|它们|其|成员|团队|队友|大家|结果|了|(?=[，。！？、；,.!?;\s]|$))"
    r"|后台继续|转入后台"
    r"|\b(?:don['’]?t|do not|no need to) wait\b|\bwithout waiting\b"
    r"|\b(?:keep working|continue|carry on) in the background\b",
    re.I,
)
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


# Runners whose result a passing exit status reports; verbosity flags do not change it.
CHECK_EXECUTABLES = frozenset({"pytest", "py.test", "mypy", "ruff", "tsc"})
PYTHON_CHECK_MODULES = frozenset({"unittest", "pytest", "compileall", "py_compile", "doctest"})
NOT_A_RUN = frozenset({"--version", "--help", "-h", "--collect-only", "--co"})
OUTPUT_FLAGS = frozenset({"-v", "-vv", "-vvv", "-q", "-qq", "--verbose", "--quiet", "--no-header"})
OUTPUT_FLAG_PREFIXES = ("--tb=", "--color=", "--durations=")
# Commands that only prepare the environment of the next one in an and-list.
WRAPPERS = frozenset({"env", "time", "nice", "nohup"})
# A redirection word ("2>&1", ">log", "<<'EOF'"): its operator and any attached target.
REDIRECTION = re.compile(r"^(\d*(?:<<<|<<-?|<>|<&|>&|>>|>\||<|>)|&>>?)(.*)$", re.S)
HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|\\?([A-Za-z0-9_.-]+))")
# How alike a passing inline script must be to a failed one to count as its revision.
REVISION_SIMILARITY = 0.6
# A word that is, or assigns, a command substitution ("x=$(git diff | wc -l)").
SUBSTITUTION = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=)?(?:\$\(|`)")
SHELL_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
SHELL_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
SHELL_RAW_WORD = re.compile(r"(?:[^\s'\"\\]+|\\.|'[^']*'|\"(?:[^\"\\]|\\.)*\")+")
SHELL_WORD_PART = re.compile(r"[^'\"\\]+|\\.|'[^']*'|\"(?:[^\"\\]|\\.)*\"")
TEST_ENTRY = frozenset({("unittest", "main"), ("pytest", "main"), ("doctest", "testmod")})


def excerpt_command(command, limit=300):
    command = " ".join(command.split())
    return command if len(command) <= limit else command[: limit - 1] + "…"


@dataclass(frozen=True)
class ShellCommand:
    """One simple command of a command line, as written, with the operator before it."""

    operator: str | None
    text: str
    words: tuple[str, ...]
    stdin: str | None = None
    # The here-document of the command, as (delimiter, body), to run it again.
    heredoc: tuple[str, str] | None = None


def _operator_at(command, index):
    """The control operator starting at index (outside quotes), or None."""
    for operator in ("&&", "||", "|&", ";;", "|", ";", "&", "\n"):
        if command.startswith(operator, index):
            previous = command[index - 1] if index else ""
            # "2>&1", ">&2", "&>log" and ">|" are redirections, not operators.
            if operator in {"&", "&&"} and (
                (previous and previous in "<>") or command.startswith("&>", index)
            ):
                return None
            if operator == "|" and previous == ">":
                return None
            return operator
    return None


def _split_line(command):
    """The commands of a line as (operator, text, heredoc) parts; None if malformed.

    Splits only at control operators outside quotes, parentheses and here-document
    bodies, so a quoted "&&" or a shift inside a Python program splits nothing.
    """
    parts, current, pending = [], [], []
    operator, quote, depth, index = None, None, 0, 0
    while index < len(command):
        character = command[index]
        if character == "\\" and quote != "'":
            current.append(command[index : index + 2])
            index += 2
            continue
        if quote is not None:
            quote = None if character == quote else quote
        elif character in "'\"":
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif command.startswith("<<", index) and not command.startswith("<<<", index):
            match = HEREDOC.match(command, index)
            if match is None:
                return None
            delimiter = next(group for group in match.groups()[1:] if group is not None)
            pending.append([delimiter, bool(match.group(1)), None, len(parts)])
            current.append(command[index : match.end()])
            index = match.end()
            continue
        if quote is None and depth == 0:
            found = _operator_at(command, index)
            if found is not None:
                if found == "\n" and any(heredoc[2] is None for heredoc in pending):
                    # Here-document bodies follow the line that started them.
                    lines, consumed = command[index + 1 :].split("\n"), 0
                    for heredoc in (item for item in pending if item[2] is None):
                        body = []
                        while True:
                            if consumed >= len(lines):
                                return None
                            line, consumed = lines[consumed], consumed + 1
                            if (line.lstrip("\t") if heredoc[1] else line) == heredoc[0]:
                                break
                            body.append(line)
                        heredoc[2] = "\n".join(body) + ("\n" if body else "")
                    command = command[: index + 1] + "\n".join(lines[consumed:])
                parts.append((operator, "".join(current).strip()))
                current, operator = [], found
                index += len(found)
                continue
        current.append(character)
        index += 1
    if quote is not None or any(heredoc[2] is None for heredoc in pending):
        return None
    parts.append((operator, "".join(current).strip()))
    heredocs = {}
    for delimiter, _tabs, body, position in pending:
        heredocs.setdefault(position, (delimiter, body))
    return [(op, text, heredocs.get(position)) for position, (op, text) in enumerate(parts)]


def parse_shell(command):
    """The simple commands of a command line with the operator before each, or None.

    Redirections leave the words; a here-document or here-string becomes the command's
    stdin. None when the line cannot be lexed, so what it runs is unknown.
    """
    if not isinstance(command, str):
        return None
    parts = _split_line(command.strip())
    if parts is None:
        return None
    commands = []
    for operator, text, heredoc in parts:
        try:
            tokens = shlex.split(text)
        except ValueError:
            return None
        words, stdin, index = [], None, 0
        while index < len(tokens):
            match = REDIRECTION.match(tokens[index])
            if match is None:
                words.append(tokens[index])
                index += 1
                continue
            target = match.group(2)
            if not target and index + 1 < len(tokens):
                index += 1
                target = tokens[index]
            if match.group(1).startswith("<<<"):
                stdin = target
            elif match.group(1).startswith("<<"):
                stdin = heredoc[1] if heredoc else ""
            index += 1
        if operator in {";", "\n", ";;"}:
            operator = ";"
        commands.append(ShellCommand(operator, text, tuple(words), stdin, heredoc))
    # A trailing ";" or newline ends the line; a trailing "&" still backgrounds it.
    commands = [item for item in commands if item.words or item.operator != ";"]
    if not commands or not commands[0].words:
        return None
    return commands


def _command_text(item):
    if item.heredoc is None:
        return item.text
    delimiter, body = item.heredoc
    return f"{item.text}\n{body}{delimiter}"


def shell_path_value(raw, environment, *, assignment=False):
    """Resolve literal paths and simple local variables without evaluating shell code."""
    def expand(part, *, quoted):
        pieces, cursor = [], 0
        for match in SHELL_VARIABLE.finditer(part):
            literal = part[cursor:match.start()]
            value = environment.get(match[1] or match[2])
            if re.search(r"[$`]", literal) or value is None:
                return None
            if not quoted and not assignment and (not value or re.search(r"[\s*?\[]", value)):
                return None
            pieces.extend((literal, value))
            cursor = match.end()
        if re.search(r"[$`]", part[cursor:]):
            return None
        return "".join((*pieces, part[cursor:]))

    pieces = []
    for part in SHELL_WORD_PART.findall(raw):
        if part.startswith("'"):
            value = part[1:-1]
        elif part.startswith('"'):
            if "\\" in part:
                return None
            value = expand(part[1:-1], quoted=True)
        elif part.startswith("\\"):
            value = part[1:]
        else:
            if re.search(r"[*?\[{}<>()]", SHELL_VARIABLE.sub("", part)):
                return None
            if part.startswith("~") and not pieces:
                if part != "~" and not part.startswith("~/"):
                    return None
                home = environment.get("HOME")
                if not home:
                    return None
                part = home + part[1:]
            value = expand(part, quoted=False)
        if value is None:
            return None
        pieces.append(value)
    return "".join(pieces)


def cd_target(words, cwd, environment):
    if words[:1] != ["cd"]:
        return None
    operands = iter(words[1:])
    mode = None
    for word in operands:
        if word == "--":
            word = next(operands, "")
            break
        if word in {"-L", "-P", "-LP", "-PL"}:
            mode = word[-1]
            continue
        break
    else:
        word = '"$HOME"'
    if not word or next(operands, None) is not None:
        return None
    value = shell_path_value(word, environment)
    if not value or value.startswith("-"):
        return None
    path = Path(value)
    if not path.is_absolute() and not value.startswith(("./", "../")) and value not in {".", ".."}:
        if environment.get("CDPATH", "") != "":
            return None
    try:
        if not path.is_absolute():
            if cwd is None:
                return None
            if mode == "P":
                return (Path(cwd).resolve() / path).resolve()
            if ".." in path.parts and Path(cwd).resolve() != cwd:
                return None
            path = Path(cwd) / path
        if mode == "P":
            return path.resolve()
        logical = Path(os.path.normpath(path))
        if mode is None and ".." in path.parts and logical.resolve() != path.resolve():
            return None
        return logical
    except (OSError, RuntimeError):
        return None


def update_shell_environment(words, environment):
    if words[:1] == ["unset"]:
        for name in words[1:]:
            environment[name] = ""
        return True
    exported = words[:1] == ["export"]
    assignments = [SHELL_ASSIGNMENT.fullmatch(word) for word in words[1 if exported else 0:]]
    if not words or (not exported and not all(assignments)):
        if not words or not SUBSTITUTION.match(words[0]) or assignments[0] is None:
            return False
        environment[assignments[0][1]] = None
        return True
    for assignment in assignments:
        if assignment is not None:
            name, value = assignment.groups()
            environment[name] = shell_path_value(value, environment, assignment=True)
    return True


@dataclass(frozen=True)
class ShellStep:
    command: ShellCommand
    words: list[str]
    directory: Path | None
    target: Path | None
    changes_directory: bool
    updates_environment: bool


def shell_steps(command, cwd):
    commands = parse_shell(command)
    if commands is None:
        return []
    environment = command_environment()
    inherited_pwd = Path(environment.get("PWD", ""))
    if cwd is not None and inherited_pwd.is_absolute():
        try:
            if inherited_pwd.samefile(cwd):
                cwd = inherited_pwd
        except OSError:
            pass
    environment["PWD"] = str(cwd) if cwd is not None else None
    steps = []
    succeeded, failed = {cwd}, set()
    for position, item in enumerate(commands):
        raw_words = SHELL_RAW_WORD.findall(item.text)
        words = command_words(item.words)
        following = commands[position + 1].operator if position + 1 < len(commands) else None
        persistent = item.operator not in {"|", "|&"} and following not in {"|", "|&", "&"}
        if item.operator == "&&":
            candidates, skipped_success, skipped_failure = succeeded, set(), failed
        elif item.operator == "||":
            candidates, skipped_success, skipped_failure = failed, succeeded, set()
        else:
            candidates, skipped_success, skipped_failure = succeeded | failed, set(), set()
        directory = next(iter(candidates)) if len(candidates) == 1 else None
        plain_cd = raw_words[:1] == ["cd"]
        changes_directory = (
            plain_cd or item.words[:1] in {("cd",), ("pushd",), ("popd",)} or words[:1] == ["cd"]
        )
        target = cd_target(raw_words, directory, environment) if plain_cd else None
        updated = dict(environment)
        assignment = update_shell_environment(raw_words, updated)
        steps.append(ShellStep(item, words, directory, target, changes_directory, assignment))
        succeeded = skipped_success | ({target} if persistent and changes_directory else candidates)
        failed = skipped_failure | candidates
        if persistent and assignment:
            if skipped_success or skipped_failure or following == "||":
                environment.update({
                    key: None for key in updated if updated[key] != environment.get(key)
                })
            else:
                environment = updated
        if persistent and changes_directory:
            environment.update(PWD=None, OLDPWD=None)
    return steps


def standalone_checks(command, cwd):
    """Each check of a command line, alone with the setup "&&" chained before it.

    "pytest -q | tail -5" becomes "pytest -q"; "cd app && npm test; echo $?" becomes
    "cd app && npm test". Earlier checks, pipelines and here-documents are not setup.
    """
    steps = shell_steps(command, cwd)
    forms, setup_steps = [], set()
    for position, step in enumerate(steps):
        item = step.command
        if step.changes_directory or step.updates_environment:
            setup_steps.add(position)
        if step.directory is None or item.operator in {"|", "|&"} or not check_words(
            step.words, item.stdin, step.directory,
        ):
            continue  # Not a check, or it reads another command's output.
        start = position
        while (
            start > 0
            and steps[start].command.operator == "&&"
            and steps[start - 1].command.operator in {None, "&&"}
            and steps[start - 1].command.heredoc is None
            and not check_words(
                steps[start - 1].words, steps[start - 1].command.stdin, steps[start - 1].directory,
            )
        ):
            start -= 1
        if any(change < start for change in setup_steps):
            continue
        if any(setup.changes_directory and setup.target is None for setup in steps[start:position]):
            continue
        forms.append(" && ".join(_command_text(part.command) for part in steps[start:position + 1]))
    return forms


def command_words(words):
    """The words of the command a wrapper or variable assignments run."""
    words = list(words)
    if any(SUBSTITUTION.match(word) for word in words):
        return []  # A command substitution runs a command this line does not show.
    while words:
        if SHELL_ASSIGNMENT.fullmatch(words[0]) or words[0] in WRAPPERS:
            words.pop(0)
        elif words[0] == "timeout":
            words.pop(0)
            while words and words[0].startswith("-"):
                words.pop(0)
            if words:
                words.pop(0)  # The duration.
        elif words[0] in {"uv", "poetry", "pipenv", "pdm", "hatch"} and words[1:2] == ["run"]:
            del words[:2]
        else:
            break
    return words


def _python_program_checks(program):
    try:
        return any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(program)))
    except (SyntaxError, ValueError):
        return False


NODE_ASSERT_IMPORT = re.compile(
    r"""(?:require\(\s*|from\s+|import\(\s*)['"](?:node:)?assert(?:/strict)?['"]"""
)
NODE_ASSERT_CALL = re.compile(
    r"(?<!console\.)(?<!\w)(?:assert(?:\.\w+)?|ok|equal|strictEqual|deepEqual|deepStrictEqual|throws"
    r"|rejects|match|notEqual|notStrictEqual)\s*\("
)


def _node_program_checks(program):
    # node:assert throws on failure, so the process exits nonzero; console.assert does not.
    return bool(NODE_ASSERT_IMPORT.search(program) and NODE_ASSERT_CALL.search(program))


def _test_script(path):
    """True when running the Python file executes its tests or assertions."""
    try:
        if not path.is_file() or path.stat().st_size > 1_000_000:
            return False
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError, ValueError):
        return False
    statements = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.If):  # if __name__ == "__main__":
            statements += node.body
    for node in statements:
        if isinstance(node, ast.Assert):
            return True
        call = node.value if isinstance(node, ast.Expr) else None
        function = call.func if isinstance(call, ast.Call) else None
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (function.value.id, function.attr) in TEST_ENTRY
        ):
            return True
    return False


def check_words(words, stdin=None, cwd=None):
    """True when running these words is a test, build or assertion whose status is its result."""
    if not words or any(flag in words for flag in NOT_A_RUN):
        return False
    executable = Path(words[0]).name
    # An echo of a command or an arbitrary successful shell command is not a check.
    if executable in CHECK_EXECUTABLES:
        return True
    if executable.startswith("python"):
        if len(words) > 2 and words[1] == "-c":
            return _python_program_checks(words[2])
        if len(words) > 2 and words[1] == "-m":
            return words[2] in PYTHON_CHECK_MODULES
        if stdin is not None and (len(words) == 1 or words[1] == "-"):
            return _python_program_checks(stdin)
        if len(words) > 2 and Path(words[1]).name == "manage.py" and words[2] == "test":
            return True  # Django's test runner.
        if len(words) > 1 and words[1].endswith(".py") and cwd is not None:
            return _test_script(Path(cwd) / words[1])
        return False
    if executable == "node":
        # The test runner or syntax check of named files: bare forms pass with nothing
        # to check (no test files, empty stdin). Plain scripts are not checks.
        if len(words) > 2 and words[1] in {"--test", "--check", "-c"}:
            return True
        for flag in ("-e", "--eval"):
            if flag in words[1:-1]:
                return _node_program_checks(words[words.index(flag) + 1])
        if stdin is not None and all(word.startswith("-") for word in words[1:]):
            return _node_program_checks(stdin)
        return False
    if executable in {"diff", "cmp"}:
        # Exits nonzero on any difference: an assertion about two named files.
        operands = [word for word in words[1:] if not word.startswith("-") or word == "-"]
        return len(operands) == 2 and operands[0] != operands[1]
    return executable in {"npm", "pnpm", "yarn", "bun", "cargo", "go", "make", "cmake"} and any(
        word in {"test", "check", "build", "lint", "typecheck", "--build"} for word in words[1:]
    )


def inline_program(words, stdin=None):
    """The interpreter and program text of a check written inline (-c, -e or stdin)."""
    executable = Path(words[0]).name if words else ""
    if executable.startswith("python"):
        if len(words) > 2 and words[1] == "-c":
            return "python", words[2]
        if stdin is not None and (len(words) == 1 or words[1] == "-"):
            return "python", stdin
    if executable == "node":
        for flag in ("-e", "--eval"):
            if flag in words[1:-1]:
                return "node", words[words.index(flag) + 1]
        if stdin is not None and all(word.startswith("-") for word in words[1:]):
            return "node", stdin
    return None


def revises(failed, passed):
    """True when a passing inline program is a revision of a failed one.

    An inline script is its own test definition, like a test file: correcting it and
    running it again can settle its failure, as editing a test and rerunning pytest does.
    An unrelated script does not.
    """
    matcher = SequenceMatcher(None, failed, passed, autojunk=False)
    return (
        matcher.real_quick_ratio() >= REVISION_SIMILARITY
        and matcher.quick_ratio() >= REVISION_SIMILARITY
        and matcher.ratio() >= REVISION_SIMILARITY
    )


def check_identity(cwd, words, stdin=None):
    """The same check however verbosely or with which interpreter alias it is run."""
    words = list(words)
    if words and words[0] == "python3":
        words[0] = "python"
    kept = [
        word
        for position, word in enumerate(words)
        if not position or (word not in OUTPUT_FLAGS and not word.startswith(OUTPUT_FLAG_PREFIXES))
    ]
    return json.dumps([str(cwd), kept, stdin], ensure_ascii=False)


def check_unavailable(words, data):
    """Why the check could not run at all (its runner is missing), or None."""
    output = data.get("output") if isinstance(data.get("output"), str) else ""
    if data.get("exit_code") == 127:
        return "command not found"
    if (
        len(words) > 2
        and Path(words[0]).name.startswith("python")
        and words[1] == "-m"
        and re.search(rf"No module named {re.escape(words[2])}\b", output)
    ):
        return f"Python module {words[2]} is not installed"
    if Path(words[0]).name in {"npm", "pnpm", "yarn", "bun"} and re.search(
        r"Missing script|Command \"\w+\" not found|ERR_PNPM_NO_SCRIPT", output
    ):
        return "the package defines no such script"
    return None


@dataclass
class TaskContract:
    kind: str = "auto"
    checks: tuple[Callable[[], str | None], ...] = ()
    description: str = ""
    verification_required: bool = False
    background_handoff: bool = True
    integrate_team: bool = False
    allow_tools: bool = True


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
    # Suffixes of files whose own edits need a check.
    suffixes: frozenset[str] = CODE_SUFFIXES
    # Tools that cannot change files (read, grep, ...) skip the workspace walk. A change
    # something else makes meanwhile is found by the next walk: before any edit, command
    # or check, and at validation.
    passive_tools: frozenset[str] = frozenset()
    # File digests by identity, shared across a conversation's turns so that each new
    # task does not hash the whole workspace again.
    digest_cache: dict | None = None
    configuration_paths: tuple[Path, ...] = ()
    defer_baseline: bool = False
    prepare_exec: Callable[[], None] | None = None

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
                r"分析|解释|怎么|如何|计划|不要修改|翻译|复述|引用"
                r"|\b(?:explain|analy[sz]e|how to|review|plan|translate|quote)\b",
                self.prompt,
                re.I,
            )
            # Research and writing deliverables are no code change even when their topic
            # mentions one ("调研 RAG 的实现方案并写一份报告"), unless the prompt also names
            # code work ("修复论文管理系统的登录 bug").
            writing = re.search(
                r"调研|综述|文献|论文|撰写|研究报告|写.{0,8}(?:报告|文章|综述)"
                r"|\b(?:research|survey|literature|essay|write (?:a|an|the) report)\b",
                self.prompt,
                re.I,
            ) and not re.search(
                r"修复|修一下|重构|bug|测试|代码|接口|函数|模块|页面|网页|脚本"
                r"|\b(?:fix|refactor|tests?|code|api|page|module|function|class|endpoint|script)\b"
                r"|\.(?:py|js|ts|tsx|jsx|go|rs|java|html|css)\b",
                self.prompt,
                re.I,
            )
            self.kind = (
                "change"
                if self.has_tools and action and not discussion and not writing
                else "answer"
            )
        self.handoff_allowed = self.contract.background_handoff and bool(HANDOFF.search(self.prompt))
        self.status = "in_progress"
        self.evidence = []
        self.check_results = []
        self.last_code_change = -1
        self.last_verification = -1
        self.code_revision = int(self.contract.verification_required)
        self.verified_revision = -1
        self.failed_verifications = {}
        self.unknown_verification = None
        self.terminal_problem = None
        # Evidence indexes of this agent's latest own source change and of
        # where its latest passing standalone check started (a background check finishes
        # later), whatever the code revision then was.
        self.last_own_edit = -1
        self.last_passed_check = -1
        self.pending_verifications = {}
        self._verification_start = 0
        # Files changed since the latest verified revision, for the repair message.
        self.unverified_changes = []
        # Standalone forms of uncounted checks run on the current revision, and those
        # already rerun, so the runtime can rerun each once instead of asking the model.
        self.reruns = []
        self._rerun = set()
        self._roots = {self.workspace.resolve()} if self.workspace is not None else set()
        self._extra_files = {path.absolute() for path in self.configuration_paths}
        self._own_paths = set()
        self._integrating_team = False
        self._background_exec = set()
        self.observed_processes = set()
        self._tracking_error = None
        self._snapshot_cache = self.digest_cache if self.digest_cache is not None else {}
        self._files = None
        self._identities = {}
        self._directory_identities = None
        self._verification_files = ({}, None, None)
        self._has_prior_source = False
        if not self.defer_baseline:
            self.begin()

    def begin(self, *, team_pending=False, team_active=False, prior_revision=None):
        self._integrating_team |= team_pending and (
            self.contract.kind != "answer" or self.contract.integrate_team
        )
        if (
            self.kind == "change" or self.contract.verification_required or self._integrating_team
            or self._has_prior_source
        ):
            self._refresh_files(start=True)
        if prior_revision and self._files is not None:
            if self._revision_digest(self._files) != prior_revision["digest"]:
                self.code_revision = max(self.code_revision, 1)
        if team_active and self.integrates_team:
            self.code_revision = max(self.code_revision, 1)

    def _relevant(self, path):
        return path in self._extra_files or verification_file(path, self.suffixes)

    def _extra_digest(self, path, identities):
        target = path.resolve()
        digest = cached_file_digest(
            path, self._snapshot_cache, self._check_deadline, mode_aware=True, identities=identities,
        )
        if target != path.resolve():
            raise OSError(f"Configuration or source link changed while being tracked: {path}")
        if target == path:
            return digest
        if identities is not None:
            try:
                modified = path.lstat().st_ctime_ns
            except FileNotFoundError:
                modified = None
            identities[str(path)] = (target, modified, identities.get(str(path)))
        return json.dumps([str(target), digest])

    def _snapshot(self, identities=None, directory_identities=None):
        files = source_snapshot(
            self._roots,
            ignored=self.ignored_paths,
            deadline=self.deadline,
            cache=self._snapshot_cache,
            relevant=lambda path: path not in self._extra_files and self._relevant(path),
            identities=identities,
            directory_identities=directory_identities,
        )
        for path in self._extra_files:
            if directory_identities is not None and path.parent not in directory_identities:
                directory_identity(path.parent, directory_identities)
            digest = self._extra_digest(path, identities)
            if digest is not None:
                files[str(path)] = digest
        return files

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Task deadline reached while tracking file changes")

    def _invalidate_verification(self):
        self.verified_revision = -1
        self.last_passed_check = -1

    def _tracking_failed(self, error):
        self._tracking_error = self._tracking_error or str(error) or type(error).__name__
        self._invalidate_verification()

    def _refresh_files(self, *, own=False, start=False, watch_directories=False):
        if not self._roots or (self._files is None and not start):
            return []
        try:
            identities = {}
            directories = {} if watch_directories else None
            current = self._snapshot(identities, directories)
        except (TimeoutError, KeyboardInterrupt) as exc:
            self._tracking_failed(exc)
            raise
        except OSError as exc:
            self._tracking_failed(exc)
            return []
        self._identities = identities
        self._directory_identities = directories
        if self._files is None:
            self._files = current
            return []
        changed = sorted(
            path
            for path in self._files.keys() | current.keys()
            if self._files.get(path) != current.get(path)
        )
        self._files = current
        if own:
            self._own_paths.update(changed)
        elif (
            self.kind != "change" and not self._own_paths and not self._integrating_team
            and not self.contract.verification_required and not self._has_prior_source
        ):
            changed = []
        if changed:
            self.code_revision += 1
            self.last_code_change = len(self.evidence)
            self._changed(changed)
            if own or any(path in self._own_paths for path in changed):
                self.last_own_edit = len(self.evidence)
        return changed

    @staticmethod
    def _arguments(call):
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            return {}
        if not isinstance(arguments, dict):
            return {}
        return {
            key: value for key, value in arguments.items()
            if key not in {"command", "cwd", "path"} or isinstance(value, str)
        }

    def _tracks_tool(self, name, arguments):
        if name in self.passive_tools | {"read", "grep", "find", "ls", "update_plan", "ask_user"}:
            return False
        if name == "mcp" and arguments.get("action") == "status":
            return False
        if name == "monitor" and arguments.get("action") in ("watch", "list"):
            return False
        if name == "monitor" and arguments.get("action") in ("status", "wait"):
            session = arguments.get("session_id")
            return isinstance(session, str) and session in self._background_exec
        if name == "team" and self._files is None and self.kind != "change":
            return self._team_integration_call(arguments)
        return True

    def _team_integration_call(self, arguments):
        return (
            (self.contract.kind != "answer" or self.contract.integrate_team)
            and arguments.get("action") in (
                "send_message", "create_task", "update_task", "request_review", "start_protocol",
                "contribute", "conclude", "spawn_agent", "cancel", "wait",
            )
        )

    def _extend_root(self, root):
        if any(root.is_relative_to(existing) for existing in self._roots):
            return
        identities = {}
        files = source_snapshot(
            (root,), ignored=self.ignored_paths, deadline=self.deadline,
            cache=self._snapshot_cache, relevant=self._relevant, identities=identities,
            directory_identities=self._directory_identities,
        )
        for path, digest in files.items():
            if Path(path) not in self._extra_files and not any(
                Path(path).is_relative_to(existing) for existing in self._roots
            ):
                self._files[path] = digest
                if path in identities:
                    self._identities[path] = identities[path]
        self._roots = {existing for existing in self._roots if not existing.is_relative_to(root)}
        self._roots.add(root)

    def _extend_scope(self, name, arguments, process_cwd):
        if self.workspace is None:
            return
        try:
            if process_cwd is not None:
                self._extend_root(process_cwd)
            if name == "exec":
                directory = self._command_cwd(arguments)
                self._extend_root(directory)
                for step in shell_steps(arguments.get("command", ""), directory):
                    if step.target is not None:
                        self._extend_root(step.target.resolve())
                    elif step.changes_directory:
                        self._tracking_failed("Shell working directory could not be resolved before execution")
            if name in {"write", "edit"} and "path" in arguments:
                path = (self.workspace / Path(arguments["path"]).expanduser()).resolve()
                if self._relevant(path) and str(path) not in self._files and path not in self._extra_files:
                    digest = self._extra_digest(path, self._identities)
                    if digest is not None:
                        self._files[str(path)] = digest
                    self._extra_files.add(path)
        except (TimeoutError, KeyboardInterrupt) as exc:
            self._tracking_failed(exc)
            raise
        except OSError as exc:
            self._tracking_failed(exc)

    def before_tool(self, call, *, process_cwd=None):
        name = call.get("function", {}).get("name")
        arguments = self._arguments(call)
        if name == "team" and self._team_integration_call(arguments):
            self._integrating_team = True
        if self._tracks_tool(name, arguments):
            if name == "exec" and self.prepare_exec is not None:
                self.prepare_exec()
            self._refresh_files(
                own=bool(self._background_exec), start=True, watch_directories=name == "exec",
            )
            if self._files is not None:
                self._extend_scope(name, arguments, process_cwd)
        self._verification_start = self.code_revision
        events = None
        if name == "exec" and self._directory_identities and not self._tracking_error:
            command, cwd = arguments.get("command", ""), self._command_cwd(arguments)
            if self.shell_checks(command, cwd)[1]:
                events = SourceEvents(self._directory_identities, deadline=self.deadline)
        self._verification_files = (
            (dict(self._identities), self._directory_identities, events) if name == "exec" else ({}, None, None)
        )

    def instructions(self):
        parts = []
        if self.kind == "change":
            parts.append(
                "Implementation task: use tools, then verify the final edit with a relevant test, build, or assertion. "
                "If no change is needed, cite inspected evidence; if blocked, state the blocker and unfinished work."
            )
        if self.contract.description:
            parts.append("Acceptance criteria: " + self.contract.description)
        if self.contract.verification_required or self.code_revision:
            parts.append(
                "Earlier source changes still require verification. Inspect them and run a relevant "
                "check before claiming completion."
            )
        return "\n".join(parts)

    def background_sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._background_exec))

    def source_checkpoint(self):
        digest = self._revision_digest(self._files) if self._files is not None else None
        return {**self.source_scope(), "digest": None if self._tracking_error else digest}

    def resume_process(self, session_id, checkpoint):
        if session_id in self.observed_processes:
            return
        self.observed_processes.add(session_id)
        if session_id in self._background_exec:
            return
        self._background_exec.add(session_id)
        self.include_source_scope(checkpoint)
        self._refresh_files(own=True, start=True)
        digest = self.source_checkpoint()["digest"]
        unchanged = digest is not None and digest == checkpoint["digest"]
        if not unchanged:
            self.code_revision += 1
            self._invalidate_verification()
            self.last_own_edit = len(self.evidence)
            self.reruns = []

    @staticmethod
    def _revision_digest(files):
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def verified_source(self):
        if self.status != "completed" or not self.code_revision or self._files is None:
            return None
        return {
            "digest": self._revision_digest(self._files),
            **self.source_scope(),
        }

    def source_scope(self):
        return {
            "roots": sorted(map(str, self._roots)),
            "paths": sorted(map(str, self._extra_files)),
        }

    def include_source_scope(self, scope):
        self._roots.update(Path(path) for path in scope["roots"])
        self._extra_files.update(Path(path) for path in scope["paths"])
        self._has_prior_source = True

    def source_matches(self, revision):
        self.include_source_scope(revision)
        return self._revision_digest(self._snapshot()) == revision["digest"]

    @property
    def integrates_team(self):
        return self._integrating_team or self.kind == "change" or self.contract.integrate_team

    @staticmethod
    def verification_command(command, cwd=None):
        if not isinstance(command, str):
            return False
        try:
            words = shlex.split(command)
        except ValueError:
            return False
        return check_words(command_words(words), None, cwd)

    def _source_touched_by_check(self, verification_files):
        files, directories, events = verification_files
        if files != self._identities:
            return True
        current = self._directory_identities
        if directories == current:
            return False
        if events is None or not directories or not current or directories.keys() != current.keys():
            return True
        dirty = {path for path in directories if directories[path] != current[path]}
        if any(directories[path][:2] != current[path][:2] for path in dirty):
            return True
        paths, complete = events.finish()
        self._check_deadline()
        if not complete:
            return True
        ignored = {path.resolve() for path in self.ignored_paths}
        explained = set()
        for path, is_directory, entry in paths:
            if path in self._extra_files:
                return True
            for directory in (path, *path.parents):
                if directory in directories:
                    if is_directory:
                        if entry:
                            return True
                    elif self._relevant(path) or not (path.suffix or path.name.startswith(".")):
                        return True
                    elif entry:
                        explained.add(path.parent)
                    break
                if directory in ignored or (
                    directory.name in GENERATED_DIRECTORIES and (directory != path or is_directory)
                ):
                    if directory == path and entry:
                        explained.add(directory.parent)
                    break
        return not dirty <= explained

    def observe(self, call, result):
        self.status = "in_progress"
        function = call.get("function", {})
        arguments = self._arguments(call)
        name = function.get("name")
        data = result.get("data") or {}
        advances_process = name == "exec" or (name == "monitor" and arguments.get("action") == "input")
        if advances_process and data.get("state") == "running" and data.get("session_id"):
            self._background_exec.add(data["session_id"])
        record = {
            "tool_call_id": call.get("id"),
            "tool": name,
            "status": result.get("status"),
            "artifact_path": result.get("artifact_path"),
        }
        index = len(self.evidence)
        pending_check = name == "monitor" and data.get("session_id") in self.pending_verifications
        changed = (
            self._refresh_files(
                own=name != "team" or bool(self._background_exec),
                watch_directories=name == "exec" or pending_check,
            )
            if self._tracks_tool(name, arguments) else []
        )
        if changed:
            record["changed_files"] = changed
        if self._tracking_error:
            record["tracking_error"] = self._tracking_error
        self.evidence.append(record)
        if (
            self.workspace is None and result.get("status") == "ok"
            and name in {"write", "edit"} and self._relevant(Path(arguments.get("path", "")))
        ):
            self.code_revision += 1
            self.last_code_change = index
            self.last_own_edit = index
            self._changed([arguments.get("path", "")])
        record["code_revision"] = self.code_revision
        verification_start = self._verification_start
        verification_files = self._verification_files
        self._verification_files = ({}, None, None)
        started_at = index
        verification_result = result
        if name == "monitor":
            pending = self.pending_verifications.get(data.get("session_id"))
            finished = data.get("state") in {"completed", "failed", "timed_out", "stopped"}
            if finished:
                self._background_exec.discard(data.get("session_id"))
            if pending and finished:
                arguments, verification_start, started_at, verification_files = (
                    self.pending_verifications.pop(data["session_id"])
                )
                record["verification_session_id"] = data["session_id"]
                verification_result = {"status": {
                    "completed": "ok", "failed": "error", "timed_out": "timeout", "stopped": "unknown",
                }[data["state"]]}
        command = arguments.get("command", "")
        checks, proven, attributable, leading = (
            self.shell_checks(command, self._command_cwd(arguments))
            if name == "exec" or "verification_session_id" in record
            else ([], False, False, False)
        )
        if checks and data.get("state") == "running" and data.get("session_id"):
            self.pending_verifications[data["session_id"]] = (
                dict(arguments),
                verification_start,
                index,
                verification_files,
            )
            record["verification_pending"] = True
        elif checks:
            record["verification"] = True
            if verification_result.get("status") == "ok" and proven:
                if (
                    verification_start == self.code_revision
                    and self._source_touched_by_check(verification_files)
                ):
                    record["verification_source_touched"] = True
                    self.code_revision += 1
                    record["code_revision"] = self.code_revision
                    self.last_code_change = index
                    self.last_own_edit = index
                    self._changed(())
                    self._invalidate_verification()
                elif (
                    not changed and not self._tracking_error and not self._background_exec
                    and verification_start == self.code_revision
                ):
                    for identity, _, inline in checks:
                        self.failed_verifications.pop(identity, None)
                        if inline is not None:
                            self._settle_revisions(*inline)
                    self.last_passed_check = started_at
                    self.last_verification = index
                    self.verified_revision = self.code_revision
                    self.unknown_verification = None
                    self.unverified_changes = []
                    self.reruns = []
            elif verification_result.get("status") == "ok":
                # Composition may hide a failure ("pytest | tail", "pytest || true").
                record["verification_untrusted"] = UNTRUSTED_CHECK
                if leading:
                    self._invalidate_verification()
                self._offer_rerun(command, arguments, verification_start)
            elif attributable and (unavailable := check_unavailable(checks[0][1], data)):
                # The runner is missing: no check ran, so nothing failed or passed.
                record["verification_unavailable"] = unavailable
                record["command"] = excerpt_command(command)
            elif verification_result.get("status") in {"unknown", "timeout"}:
                record["verification_unknown"] = True
                record["command"] = excerpt_command(command)
                self._invalidate_verification()
                self.unknown_verification = record
            elif attributable:
                record["command"] = excerpt_command(command)
                if checks[0][2] is not None:
                    record["runner"], record["program"] = checks[0][2]
                self.failed_verifications[checks[0][0]] = record
                self._invalidate_verification()
            else:
                # Some command of the line failed, perhaps the check: it verifies nothing.
                record["verification_untrusted"] = UNTRUSTED_CHECK
                self._invalidate_verification()
                self._offer_rerun(command, arguments, verification_start)

    def _settle_revisions(self, runner, program):
        # Only while the code is as it was when the script failed: then the script, not
        # the code, was wrong. After a change the failed check itself must pass again.
        for identity, failed in list(self.failed_verifications.items()):
            if (
                failed.get("runner") == runner
                and failed.get("code_revision") == self.code_revision
                and revises(failed["program"], program)
            ):
                del self.failed_verifications[identity]

    def _changed(self, paths):
        self.reruns = []
        for path in paths:
            if self.workspace is not None and Path(path).is_relative_to(self.workspace):
                path = Path(path).relative_to(self.workspace).as_posix()
            if path not in self.unverified_changes:
                self.unverified_changes.append(path)

    def _offer_rerun(self, command, arguments, verification_start):
        if verification_start != self.code_revision:
            return  # The code changed while the command ran.
        cwd = arguments.get("cwd", ".")
        self.reruns = [
            (form, cwd)
            for form in standalone_checks(command, self._command_cwd(arguments))
            if (form, cwd) not in self._rerun
        ]

    def rerun_candidates(self):
        """Uncounted checks of the current revision to rerun standalone, each only once.

        Only when the one thing missing is a counted check of this revision: no failed or
        pending check, and the model itself ran these checks after its last change.
        """
        if (
            not self.code_revision
            or self._tracking_error
            or self.failed_verifications
            or self.pending_verifications
            or self.verified_revision == self.code_revision
        ):
            return []
        candidates = self.reruns[:3]
        self._rerun.update(candidates)
        self.reruns = []
        return candidates

    def _command_cwd(self, arguments):
        cwd = Path(arguments.get("cwd", ".")).expanduser()
        return (self.workspace / cwd).resolve() if self.workspace is not None else cwd

    @staticmethod
    def shell_checks(command, cwd):
        """The checks a command line runs and what its exit status says about them.

        Returns (checks, proven, attributable, leading). ``checks`` are (identity, words,
        inline) triples, ``inline`` being (runner, program) for a script written inline or
        None. ``proven``: a zero status means every check passed, since the line is an
        and-list ("&&" only) with each check last in its pipeline. ``attributable``: a
        nonzero status is the one check's own failure, since it is the whole command, at
        most behind "cd" into existing directories. ``leading``: the line starts with a
        check.
        """
        steps = shell_steps(command, cwd)
        if not steps:
            return [], False, False, False
        checks = []
        proven = attributable = all(step.command.operator in {None, "&&", "|"} for step in steps)
        for position, step in enumerate(steps):
            item, words, directory = step.command, step.words, step.directory
            following = steps[position + 1].command.operator if position + 1 < len(steps) else None
            if step.changes_directory:
                if step.target is None:
                    proven = False
                if step.target is None or checks or not step.target.is_dir() or following not in {None, "&&"}:
                    attributable = False
            elif check_words(words, item.stdin, directory):
                if directory is None:
                    proven = attributable = False
                if following in {"|", "|&"}:
                    proven = False  # Only a pipeline's last command sets its status.
                if {following, item.operator} & {"|", "|&"}:
                    attributable = False
                inline = inline_program(words, item.stdin)
                checks.append(
                    (
                        check_identity(directory, words, item.stdin),
                        words,
                        (json.dumps([str(directory), inline[0]]), inline[1]) if inline else None,
                    )
                )
            else:
                attributable = False  # Another command may be what failed.
        leading = bool(checks) and check_words(steps[0].words, steps[0].command.stdin, cwd)
        return checks, proven and bool(checks), attributable and len(checks) == 1, leading

    @staticmethod
    def shell_segments(command):
        """The command's words split at shell composition (;, &, &&, |, ||, newlines).

        "<" and ">" are punctuation too, so redirections such as 2>&1 or &>log lex as
        tokens containing them and never count as composition; quoted text stays a word.
        None when the command cannot be lexed (e.g. a quote inside a shell comment), so
        its composition is unknown.
        """
        if not isinstance(command, str):
            return None
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
        self.terminal_problem = None
        self._refresh_files(own=bool(self._background_exec))
        if self._tracking_error:
            self.status = "incomplete"
            return (
                "Cannot verify the final code revision because file tracking failed: "
                + self._tracking_error
                + ". Inspect the actual workspace before retrying."
            )
        text = content.strip()
        unquoted = QUOTED.sub(" ", text)
        unable = self.reports_blocker(text)
        blocked = unable and bool(
            NEEDS_USER.search(unquoted) or EXTERNAL_REQUIREMENT.search(unquoted)
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
        if self._background_exec and not (self.handoff_allowed and self.last_own_edit < 0):
            self.terminal_problem = (
                "A background command is still running and can change the workspace. Use monitor "
                "to observe its terminal result before reporting completion, or stop it if no longer needed."
            )
            return self.terminal_problem
        if background or self._background_exec:
            # Teammates' unfinished edits are not final yet, but this agent's own are.
            if self.contract.verification_required and self.verified_revision != self.code_revision:
                return "Verify the unverified source changes from earlier rounds before handing off."
            if self.last_own_edit >= 0 and (
                self.last_passed_check < self.last_own_edit or self.failed_verifications
            ):
                return (
                    "You changed files yourself: verify your own change with a relevant "
                    "standalone test/build/assertion before answering, even though members "
                    "keep working in the background."
                )
            # Contract checks judge finished work; this turn hands work off instead.
            self.status = "background"
            return None
        if self.kind == "change" or self.code_revision:
            problem = self.verification_problem()
            if problem:
                self.terminal_problem = problem
                return problem
        if unable and work_task:
            self.terminal_problem = (
                "The task is not finished. Continue from the observed results or identify the external input needed."
            )
            return self.terminal_problem
        if self.kind in {"change", "inspect"} and not any(
            e["status"] == "ok" for e in self.evidence
        ):
            return "No successful tool evidence supports completion of this task. Inspect and perform the requested work, or state the concrete blocker."
        self.check_results = []
        if self.contract.checks:
            revision = self.code_revision
            self._refresh_files(start=True, watch_directories=True)
            if self.code_revision != revision:
                return "The source revision changed before acceptance; rerun checks on the final revision."
        contract_identities = (self._identities, self._directory_identities)
        for check in self.contract.checks:
            try:
                problem = self.validate_callback(check)
            except TimeoutError:
                raise
            except Exception as exc:
                problem = f"{type(exc).__name__}: {exc}"
            if problem is None and (self._identities, self._directory_identities) != contract_identities:
                self._invalidate_verification()
                self.check_results = []
                return "The source revision changed during acceptance; rerun all checks on the final revision."
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

    def validate_callback(self, callback):
        self._refresh_files(start=True, watch_directories=True)
        before = (self._identities, self._directory_identities)
        try:
            problem = callback()
        finally:
            changed = self._refresh_files(own=True, watch_directories=True)
        if self._tracking_error:
            self.status = "incomplete"
            return "Cannot verify the final code revision: " + self._tracking_error
        if changed or before != (self._identities, self._directory_identities):
            self._invalidate_verification()
            return "Acceptance checks changed source files; they did not verify the resulting revision."
        return problem

    def _current_unknown_verification(self):
        if self.unknown_verification and self.unknown_verification["code_revision"] == self.code_revision:
            return self.unknown_verification
        return None

    def verification_problem(self, *, monitored=True):
        """What leaves the latest code revision unverified, or None.

        Without ``monitored`` (no monitor tool) a background check can never report its
        result, so it counts as no check at all.
        """
        if self.failed_verifications:
            failed = [
                record["command"]
                for record in self.failed_verifications.values()
                if record.get("command")
            ]
            inline = any(record.get("program") for record in self.failed_verifications.values())
            return (
                "Verification failed and remains unresolved. Rerun the failed check successfully, "
                "or explicitly report the concrete validation blocker."
                + (" Failed: " + "; ".join(f"`{item}`" for item in failed[-3:]) if failed else "")
                + (
                    " An inline script (-c, -e or a here-document) is its own test: if the "
                    "script itself was wrong, a passing corrected version of it also settles "
                    "it while the code is unchanged since it failed."
                    if inline
                    else ""
                )
            )
        if self.pending_verifications and monitored:
            return "Background verification is still running. Use monitor to observe its final result before reporting completion."
        if unknown := self._current_unknown_verification():
            command = unknown["command"]
            return f"Verification result is unknown for `{command}`. Inspect the result or rerun the check."
        if (
            self.code_revision > 0
            and self.verified_revision != self.code_revision
            and not self.contract.checks
        ):
            if any(record.get("verification_source_touched") for record in self.evidence):
                return (
                    "Code has no successful verification: source files were touched while the check ran. "
                    "Run a check that leaves the source unchanged, including files it temporarily "
                    "creates, replaces or restores. Put temporary output outside tracked source directories."
                    + self._changes_note()
                )
            if any(
                record.get("verification_untrusted")
                for record in self.evidence[max(self.last_code_change, 0) :]
            ):
                return (
                    "Code was changed but no successful verification followed the last edit: "
                    "the check you ran was not counted because it was combined with other shell "
                    "commands (|, ;, ||, &, newlines), so the exit status is not the check's and "
                    "can hide a failure. Run the check alone in its own exec call, with the cwd "
                    "argument for its directory; chaining setup with && is fine."
                    + self._changes_note()
                )
            unavailable = next((
                record["verification_unavailable"] for record in reversed(self.evidence)
                if record.get("code_revision") == self.code_revision and record.get("verification_unavailable")
            ), None)
            return (
                "Code was changed but no successful verification followed the last edit. Run a relevant standalone test/build/assertion, or explicitly report why validation is blocked."
                + (f" The attempted check did not run: {unavailable}." if unavailable else "")
                + self._changes_note()
            )
        return None

    def _changes_note(self):
        if not self.unverified_changes:
            return ""
        shown = self.unverified_changes[-5:]
        more = len(self.unverified_changes) - len(shown)
        return (
            " Changed since the last counted check (deleting or cleaning files counts too): "
            + ", ".join(shown)
            + (f" and {more} more" if more > 0 else "")
            + "."
        )

    def outcome(self):
        if self._tracking_error or self._current_unknown_verification():
            verification_status = "unknown"
        elif self.failed_verifications or any(not result["passed"] for result in self.check_results):
            verification_status = "failed"
        elif self.pending_verifications:
            verification_status = "running"
        elif self.verified_revision == self.code_revision or (
            self.status == "completed" and self.check_results
        ):
            verification_status = "passed"
        else:
            verification_status = "not_run" if self.code_revision else "not_required"
        return {
            "kind": self.kind,
            "status": self.status,
            **({"handoff_source": self.source_checkpoint()} if self.status == "background" else {}),
            "evidence": self.evidence,
            "checks": self.check_results,
            "code_revision": self.code_revision,
            "verified_revision": self.verified_revision,
            "verification_required": bool(
                verification_status in {"unknown", "failed", "running"}
                or self._background_exec
                or (
                    self.code_revision and self.verified_revision != self.code_revision
                    and not (self.status == "completed" and self.check_results)
                )
            ),
            "verification_status": verification_status,
            **({"tracking_error": self._tracking_error} if self._tracking_error else {}),
        }
