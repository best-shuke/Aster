"""Pure command parsing and domain execution, independent of terminal rendering."""

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from config import REASONING_EFFORTS

if TYPE_CHECKING:
    from conversation import Conversation
    from environments import ResolvedEnvironment
    from templates import SavedTemplate
    from workflows import SavedWorkflow


class CommandResult(Enum):
    NOT_COMMAND = "not_command"
    HANDLED = "handled"
    EXIT = "exit"


class CommandName(Enum):
    EXIT = "exit"
    NEW = "new"
    TASKS = "tasks"
    HISTORY = "history"
    TRACE = "trace"
    TOOL = "tool"
    RESUME = "resume"
    TEMPLATE = "template"
    ENVIRONMENT = "environment"
    WORKFLOW = "workflow"
    STATUS = "status"
    SESSION = "session"
    CACHE = "cache"
    SKILLS = "skills"
    HELP = "help"
    EFFORT = "effort"
    PROMPT = "prompt"
    MULTIAGENTS = "multiagents"


class EffectKind(Enum):
    ERROR = "error"
    NEW_SESSION = "new_session"
    TASKS = "tasks"
    STATUS = "status"
    INFO = "info"
    CACHE = "cache"
    HELP = "help"
    TEXT = "text"
    INTERACTION = "interaction"


ALIASES = {
    "/quit": CommandName.EXIT,
    "/clear": CommandName.NEW,
    "/todo": CommandName.TASKS,
    "/env": CommandName.ENVIRONMENT,
}
COMMAND_NAMES = {f"/{item.value}": item for item in CommandName} | ALIASES
NO_ARGUMENTS = frozenset(
    {
        CommandName.EXIT,
        CommandName.NEW,
        CommandName.TASKS,
        CommandName.STATUS,
        CommandName.SESSION,
        CommandName.CACHE,
        CommandName.HELP,
        CommandName.PROMPT,
    }
)
QUOTED_ARGUMENTS = frozenset({CommandName.TEMPLATE, CommandName.ENVIRONMENT, CommandName.WORKFLOW})


@dataclass(frozen=True)
class ParsedCommand:
    name: CommandName
    arguments: tuple[str, ...] = ()
    error: ValueError | None = None


@dataclass(frozen=True)
class CommandEffect:
    kind: EffectKind
    payload: object = None


@dataclass(frozen=True)
class CommandExecution:
    result: CommandResult
    effects: tuple[CommandEffect, ...] = ()


class CommandActions:
    def __init__(self, conversation: "Conversation"):
        self.conversation = conversation

    def reset(self, *, blank_template=False) -> None:
        if blank_template:
            self.conversation.clear_template()
        else:
            self.conversation.clear()

    def resume(self, session_id: str, *, messages_only: bool = False) -> None:
        if messages_only:
            self.conversation.resume(session_id, messages_only=True)
        else:
            self.conversation.resume(session_id)

    def save_template(self, arguments: list[str]) -> "SavedTemplate":
        store = self.conversation.templates
        action = arguments[0] if arguments else None
        if action == "add" and len(arguments) in {2, 3}:
            return store.add(Path(arguments[1]), arguments[2] if len(arguments) == 3 else None)
        if action == "save" and len(arguments) == 2:
            return store.create(arguments[1], self.conversation.messages)
        raise ValueError("用法：/template add 文件 [名称]、/template save 名称")

    def open_template(self, template: "SavedTemplate") -> "SavedTemplate":
        return self.conversation.use_template(template)

    def set_effort(self, arguments: list[str]) -> str:
        if len(arguments) != 1:
            raise ValueError("用法：/effort " + "|".join(REASONING_EFFORTS))
        self.conversation.set_reasoning_effort(arguments[0])
        return self.conversation.settings.reasoning_effort or "max"

    def create_workflow(self, name: str, instructions: str) -> "SavedWorkflow":
        return self.conversation.workflows.create(name, instructions)

    def import_workflow(self, path: Path, name: str | None) -> "SavedWorkflow":
        return self.conversation.workflows.add(path, name)

    def approve_environment(self, environment: "ResolvedEnvironment") -> None:
        self.conversation.environments.approve(environment)

    def deny_environment(self, selector: str) -> None:
        self.conversation.environments.deny(selector)

    def apply_environment(
        self,
        environment: "ResolvedEnvironment | None",
        *,
        keep: bool = False,
        on_connect: Callable[[str], None] | None = None,
    ) -> list[str]:
        return self.conversation.apply_environment(environment, keep=keep, on_connect=on_connect)


def parse_command(prompt: str) -> ParsedCommand | None:
    prompt = prompt.strip()
    if "\n" in prompt or "\r" in prompt:
        return None
    parts = prompt.split()
    if not parts or (name := COMMAND_NAMES.get(parts[0])) is None:
        return None
    arguments = parts[1:]
    if name in NO_ARGUMENTS and arguments:
        return None
    if name in QUOTED_ARGUMENTS:
        try:
            arguments = shlex.split(prompt)[1:]
        except ValueError as exc:
            return ParsedCommand(name, error=exc)
    return ParsedCommand(name, tuple(arguments))


def execute_command(
    command: ParsedCommand | None, conversation: "Conversation"
) -> CommandExecution:
    if command is None:
        return CommandExecution(CommandResult.NOT_COMMAND)
    if command.error is not None:
        return CommandExecution(
            CommandResult.HANDLED, (CommandEffect(EffectKind.ERROR, command.error),)
        )
    name = command.name
    if name is CommandName.EXIT:
        return CommandExecution(CommandResult.EXIT)
    try:
        if name is CommandName.NEW:
            CommandActions(conversation).reset()
            effects = (CommandEffect(EffectKind.NEW_SESSION),)
        elif name is CommandName.TASKS:
            effects = (CommandEffect(EffectKind.TASKS),)
        elif name is CommandName.STATUS:
            effects = (CommandEffect(EffectKind.STATUS),)
        elif name is CommandName.SESSION:
            effects = (CommandEffect(EffectKind.INFO, str(conversation.settings.session_file)),)
        elif name is CommandName.CACHE:
            effects = (CommandEffect(EffectKind.CACHE, conversation.client.cache_stats.snapshot()),)
        elif name is CommandName.HELP:
            effects = (CommandEffect(EffectKind.HELP),)
        elif name is CommandName.PROMPT:
            conversation.reload_prompt()
            effects = (
                CommandEffect(EffectKind.INFO, f"提示词文件：{conversation.settings.prompt_file}"),
                CommandEffect(EffectKind.TEXT, conversation.system_message["content"]),
            )
        else:
            effects = (CommandEffect(EffectKind.INTERACTION, command),)
    except (OSError, RuntimeError, ValueError) as exc:
        effects = (CommandEffect(EffectKind.ERROR, exc),)
    return CommandExecution(CommandResult.HANDLED, effects)
