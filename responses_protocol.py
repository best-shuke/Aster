"""Responses wire conversion shared by API keys and ChatGPT account authentication."""

import json
from typing import Any

from context_budget import LOCAL_METADATA
from contracts import Message
from deepseek_client import CompletionError
from model_api import message_signature
from usage import number

REASONING_KEY = "responses_reasoning"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


def sse_events(response, check_deadline):
    """Decode complete SSE events, including multiline data and heartbeat comments."""
    lines = []
    for raw_line in response:
        check_deadline()
        line = raw_line.decode().rstrip("\r\n")
        if not line:
            if lines:
                data = "\n".join(lines)
                lines = []
                if data == "[DONE]":
                    return
                yield json.loads(data)
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines and "\n".join(lines) != "[DONE]":
        yield json.loads("\n".join(lines))


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or "" if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _parts(content: Any, kind: str) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"type": kind, "text": _text(content)}]
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append({"type": kind, "text": _text([part])})
    return parts


def responses_input(
    messages: list[Message],
    *,
    system_role: str = "developer",
    origin: str | None = None,
    encrypted_reasoning: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Leading system messages become instructions; the backend rejects system items."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _text(message.get("content"))
            if not items:
                instructions.append(text)
            else:
                items.append(
                    {"type": "message", "role": system_role, "content": _parts(text, "input_text")}
                )
        elif role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": _parts(message.get("content"), "input_text"),
                }
            )
        elif role == "assistant":
            text = _text(message.get("content"))
            metadata = message.get(LOCAL_METADATA)
            # Reasoning is replayed only ahead of the output it led to.
            same_origin = origin is None or (
                isinstance(metadata, dict)
                and metadata.get("protocol_origin") == origin
                and metadata.get("protocol_message") == message_signature(message)
            )
            if (
                encrypted_reasoning
                and same_origin
                and isinstance(metadata, dict)
                and (text or message.get("tool_calls"))
            ):
                items.extend(
                    item for item in metadata.get(REASONING_KEY) or [] if isinstance(item, dict)
                )
            if not encrypted_reasoning and same_origin and message.get("reasoning_content"):
                items.append(
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": message["reasoning_content"]}
                        ],
                    }
                )
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "",
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": _text(message.get("content")),
                }
            )
    return "\n\n".join(part for part in instructions if part) or DEFAULT_INSTRUCTIONS, items


def responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or {}
    return {
        "type": "function",
        "name": function.get("name"),
        "description": function.get("description") or "",
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        # Responses tools default to strict schemas, which Aster's schemas do not follow.
        "strict": False,
    }


def chat_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    result: dict[str, Any] = {
        key: usage[source]
        for key, source in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        )
        if number(usage.get(source))
    }
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if number(cached):
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if number(reasoning):
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    if (
        "total_tokens" not in result
        and number(result.get("prompt_tokens"))
        and number(result.get("completion_tokens"))
    ):
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result or None


def _chunk(finish_reason: str | None = None, usage: Any = None, **delta: Any) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage:
        chunk["usage"] = usage
    return chunk


def response_error(error: Any) -> CompletionError:
    error = error if isinstance(error, dict) else {"message": str(error)}
    code = str(error.get("code") or error.get("type") or "")
    if code == "context_length_exceeded":
        return CompletionError("context_overflow", "Model input exceeds the context window")
    if code in {"usage_limit_reached", "insufficient_quota"}:
        return CompletionError("usage_limit", "Model service usage limit reached")
    if code in {"rate_limit_exceeded", "rate_limit", "429"}:
        return CompletionError(
            "http_429", "Model service returned a rate-limit error", retryable=True
        )
    return CompletionError("provider_error", "Model service returned an error", retryable=True)


class _ResponsesEvents:
    """Translate one Responses event stream into chat-completions chunks."""

    def __init__(self) -> None:
        self.tools: dict[Any, int] = {}
        self.arguments: dict[int, str] = {}
        self.reasoning: list[dict[str, Any]] = []
        self.reasoning_part: Any = None
        self.text_item: Any = None
        self.done = False

    def _tool(self, event: dict[str, Any], item: dict[str, Any] | None = None) -> int:
        key = (item or {}).get("id") or event.get("item_id") or event.get("output_index")
        if not isinstance(key, (str, int)):
            raise CompletionError(
                "invalid_response", "Tool event has no valid identity", retryable=True
            )
        if key not in self.tools:
            self.tools[key] = len(self.tools)
        return self.tools[key]

    def _arguments(self, index: int, text: str) -> dict[str, Any]:
        self.arguments[index] = self.arguments.get(index, "") + text
        return _chunk(tool_calls=[{"index": index, "function": {"arguments": text}}])

    def chunks(self, event: Any) -> list[dict[str, Any]]:
        if not isinstance(event, dict):
            raise CompletionError(
                "invalid_response", "Stream event is not an object", retryable=True
            )
        kind = event.get("type")
        delta = event.get("delta")
        if kind in {
            "response.output_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        } and not isinstance(delta, str):
            raise CompletionError("invalid_response", "Response delta must be text", retryable=True)
        if kind == "response.output_text.delta":
            item = event.get("item_id")
            separator = "\n\n" if self.text_item not in (None, item) else ""
            self.text_item = item
            return [_chunk(content=separator + delta)] if isinstance(delta, str) else []
        if kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            part = (event.get("item_id"), event.get("summary_index", event.get("content_index")))
            separator = "\n\n" if self.reasoning_part not in (None, part) else ""
            self.reasoning_part = part
            return [_chunk(reasoning_content=separator + delta)] if isinstance(delta, str) else []
        if kind == "response.function_call_arguments.delta" and isinstance(delta, str):
            return [self._arguments(self._tool(event), delta)]
        item = event.get("item")
        if kind == "response.output_item.added" and isinstance(item, dict):
            if item.get("type") != "function_call":
                return []
            index = self._tool(event, item)
            self.arguments[index] = item.get("arguments") or ""
            call = {
                "index": index,
                "id": item.get("call_id") or "",
                "function": {"name": item.get("name") or "", "arguments": self.arguments[index]},
            }
            return [_chunk(tool_calls=[call])]
        if kind == "response.output_item.done" and isinstance(item, dict):
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                # Replayed on later requests so the model keeps its own reasoning.
                self.reasoning.append(
                    {
                        "type": "reasoning",
                        "summary": item.get("summary") or [],
                        "encrypted_content": item["encrypted_content"],
                    }
                )
            if item.get("type") != "function_call":
                return []
            known = (item.get("id") or event.get("item_id") or event.get("output_index")) in (
                self.tools
            )
            if not known:
                return self.chunks({**event, "type": "response.output_item.added"})
            index = self._tool(event, item)
            final, received = item.get("arguments") or "", self.arguments.get(index, "")
            if final != received and final.startswith(received):
                return [self._arguments(index, final[len(received) :])]
            if final != received:
                raise CompletionError(
                    "invalid_tool_calls",
                    "Final tool arguments differ from streamed arguments",
                    retryable=True,
                )
            return []
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        if kind == "response.completed":
            self.done = True
            finish = "tool_calls" if self.tools else "stop"
            return [_chunk(finish, chat_usage(response.get("usage")))]
        if kind == "response.incomplete":
            self.done = True
            reason = (response.get("incomplete_details") or {}).get("reason") or "incomplete"
            finish = "length" if reason == "max_output_tokens" else reason
            return [_chunk(finish, chat_usage(response.get("usage")))]
        if kind == "response.failed":
            raise response_error(response.get("error") or {})
        if kind == "error":
            raise response_error(event.get("error") or event)
        return []
