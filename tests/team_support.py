"""Scripted models for team tests.

Each agent's requests are answered by its own behavior function, chosen from the agent
identity in the system prompt the runtime built. A behavior sees only the messages the
runtime sent that agent, so assertions on its inputs check real delivery, isolation and
scheduling; the model's decisions are scripted, the runtime's are not.
"""

import json
import re
import threading
from collections import defaultdict
from itertools import count

from task_acceptance import TaskContract
from usage import CacheStats

IDENTITY = re.compile(r'You are "([a-z][a-z0-9-]*)", an independent agent')
ANSWER = TaskContract(kind="answer")
_ids = count(1)


def identify(messages) -> str:
    match = IDENTITY.search(messages[0].get("content") or "")
    return match.group(1) if match else "main"


def say(text):
    return {"role": "assistant", "content": text}


def use(_tool, **arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{next(_ids)}",
                "type": "function",
                "function": {"name": _tool, "arguments": json.dumps(arguments)},
            }
        ],
    }


def team(action, **arguments):
    return use("team", action=action, **arguments)


def last_user(messages) -> str:
    return next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"), ""
    )


def last_tool(messages) -> dict:
    for message in reversed(messages):
        if message["role"] == "tool":
            return json.loads(message["content"])
        if message["role"] == "user":
            break
    return {}


def arrived(messages) -> list[dict]:
    """Every item main received from team action=wait so far."""
    items = []
    for message in messages:
        if message["role"] == "tool":
            data = json.loads(message["content"]).get("data") or {}
            items.extend(data.get("items") or [])
    return items


def turn_messages(messages) -> list[dict]:
    """Messages since the latest activation (user message) for this agent."""
    index = max(
        (position for position, message in enumerate(messages) if message["role"] == "user"),
        default=0,
    )
    return messages[index:]


class Steps:
    """A behavior that walks a list of responses, one per model request."""

    def __init__(self, *responses):
        self.responses = list(responses)

    def __call__(self, messages, tools):
        response = self.responses.pop(0)
        return response(messages, tools) if callable(response) else response


class TeamScript:
    def __init__(self, **behaviors):
        self.behaviors = behaviors
        self.lock = threading.Lock()
        self.requests = defaultdict(list)
        self.cache_stats = CacheStats()
        self.settings = None
        self.active = 0
        self.peak = 0
        self.local = threading.local()

    def options(self) -> dict:
        """Keyword options of the current request (e.g. the cancellation event)."""
        return getattr(self.local, "options", {})

    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        agent = identify(messages)
        with self.lock:
            self.requests[agent].append(
                {"messages": json.loads(json.dumps(messages)), "tools": tools}
            )
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            behavior = self.behaviors.get(agent)
            if behavior is None:
                raise AssertionError(f"No scripted behavior for {agent}")
            self.local.options = kwargs
            response = behavior(messages, tools)
        finally:
            with self.lock:
                self.active -= 1
        if on_content and response.get("content"):
            on_content(response["content"])
        return response

    def seen(self, agent) -> str:
        """Everything the runtime sent to this agent, as one string."""
        return json.dumps(self.requests[agent], ensure_ascii=False)

    def tool_names(self, agent) -> set[str]:
        tools = self.requests[agent][-1]["tools"] or []
        return {item["function"]["name"] for item in tools}
