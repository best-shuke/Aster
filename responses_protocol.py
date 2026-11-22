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
    input_details = usage.get("input_tokens_details")
    cached = input_details.get("cached_tokens") if isinstance(input_details, dict) else None
    if number(cached):
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    output_details = usage.get("output_tokens_details")
    reasoning = output_details.get("reasoning_tokens") if isinstance(output_details, dict) else None
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
        self.tool_info: dict[int, dict[str, Any]] = {}
        self.arguments: dict[int, str] = {}
        self.reasoning: list[dict[str, Any]] = []
        self.reasoning_part: Any = None
        self.text_item: Any = None
        self.text_blocks: dict[tuple[int, int], list[str]] = {}
        self.message_content: str | None = None
        self.anonymous_text = False
        self.refusal_blocks: dict[Any, str] = {}
        self.refusal_items: dict[Any, int] = {}
        self.refusal_item_info: list[dict[str, Any]] = []
        self.done = False

    def _tool(self, event: dict[str, Any], item: dict[str, Any] | None = None) -> int:
        item = item or {}
        identities: list[Any] = []
        for kind, candidate in (
            ("item", item.get("id")),
            ("item", event.get("item_id")),
            ("call", item.get("call_id")),
        ):
            if isinstance(candidate, str) and candidate:
                identities.append((kind, candidate))
        output_index = event.get("output_index", item.get("output_index"))
        if isinstance(output_index, int):
            identities.append(("output", output_index))
        if not identities:
            raise CompletionError(
                "invalid_response", "Tool event has no valid identity", retryable=True
            )
        mapped = {self.tools[k] for k in identities if k in self.tools}
        if len(mapped) > 1:
            raise CompletionError(
                "invalid_tool_calls",
                "Conflicting tool call identities in event",
                retryable=True,
            )
        if mapped:
            index = next(iter(mapped))
        else:
            index = (
                output_index
                if (
                    type(output_index) is int
                    and output_index >= 0
                    and output_index not in self.tool_info
                )
                else max(self.tool_info, default=-1) + 1
            )
            self.tool_info[index] = {
                "id": item.get("id") or event.get("item_id"),
                "call_id": item.get("call_id") or "",
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "",
                "finalized": False,
            }
        for k in identities:
            self.tools[k] = index
        return index

    @staticmethod
    def _validate_tool_item(item: dict[str, Any]) -> None:
        for field in ("id", "call_id", "name"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CompletionError(
                    "invalid_tool_calls", f"Tool {field} must be non-empty text", retryable=True
                )
        if not isinstance(item.get("arguments"), str):
            raise CompletionError(
                "invalid_tool_calls", "Tool arguments must be text", retryable=True
            )

    def _arguments(self, index: int, text: str) -> dict[str, Any]:
        self.arguments[index] = self.arguments.get(index, "") + text
        return _chunk(tool_calls=[{"index": index, "function": {"arguments": text}}])

    def _refusal_key(
        self,
        event: dict[str, Any],
        item: dict[str, Any] | None = None,
        content_index: int = 0,
    ) -> tuple[int, int]:
        item = item or {}
        item_ids = []
        for source, field in ((item, "id"), (event, "item_id")):
            if field in source:
                value = source[field]
                if not isinstance(value, str) or not value:
                    raise CompletionError(
                        "invalid_response", "Refusal item ID must be non-empty text", retryable=True
                    )
                item_ids.append(value)
        if len(set(item_ids)) > 1:
            raise CompletionError(
                "invalid_response", "Conflicting refusal item IDs", retryable=True
            )
        output_index = event.get("output_index")
        if "output_index" in event and (type(output_index) is not int or output_index < 0):
            raise CompletionError(
                "invalid_response", "Refusal output index must be a non-negative integer",
                retryable=True,
            )
        c_idx = event.get("content_index", content_index)
        if type(c_idx) is not int or c_idx < 0:
            raise CompletionError(
                "invalid_response", "Refusal content index must be a non-negative integer",
                retryable=True,
            )
        aliases = [("item", item_ids[0])] if item_ids else []
        if output_index is not None:
            aliases.append(("output", output_index))
        if not aliases:
            raise CompletionError(
                "invalid_response", "Refusal has no valid identity", retryable=True
            )
        mapped = {self.refusal_items[alias] for alias in aliases if alias in self.refusal_items}
        if len(mapped) > 1:
            raise CompletionError(
                "invalid_response", "Conflicting refusal identities", retryable=True
            )
        index = next(iter(mapped)) if mapped else len(self.refusal_item_info)
        info = self.refusal_item_info[index] if mapped else {}
        if any(kind in info and info[kind] != value for kind, value in aliases):
            raise CompletionError("invalid_response", "Refusal identity changed", retryable=True)
        if not mapped:
            self.refusal_item_info.append(info)
        for kind, value in aliases:
            info[kind] = value
            self.refusal_items[(kind, value)] = index
        return (index, c_idx)

    def _consume_refusal(
        self, key: Any, text: Any, *, delta: bool = False
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise CompletionError("invalid_response", "Refusal must be text", retryable=True)
        streamed = self.refusal_blocks.get(key, "")
        if not delta and not text.startswith(streamed):
            raise CompletionError(
                "invalid_response", "Final refusal differs from streamed refusal", retryable=True
            )
        fragment = text if delta else text[len(streamed) :]
        self.refusal_blocks[key] = streamed + fragment
        if not fragment:
            return []
        item = key[0]
        separator = "\n\n" if self.text_item not in (None, item) else ""
        self.text_item = item
        return [_chunk(content=separator + fragment)]

    def _completed_content(self, output_items: list) -> None:
        paragraphs = []
        text_keys, refusal_keys = set(), set()
        for index, item in enumerate(output_items):
            if not isinstance(item, dict) or item.get("type") not in {"message", "refusal"}:
                continue
            parts = item.get("content", []) if item["type"] == "message" else [item]
            if not isinstance(parts, list):
                raise CompletionError("invalid_response", "Invalid Responses content blocks", retryable=True)
            paragraph = []
            for content_index, part in enumerate(parts):
                if not isinstance(part, dict):
                    raise CompletionError("invalid_response", "Invalid Responses content block", retryable=True)
                kind = part.get("type")
                if kind not in {"output_text", "refusal"}:
                    continue
                key = self._refusal_key({"output_index": index}, item, content_index)
                if kind == "output_text":
                    text_keys.add(key)
                    text = part.get("text")
                    streamed = "".join(self.text_blocks.get(key, ()))
                else:
                    refusal_keys.add(key)
                    text = part.get("refusal", part.get("text"))
                    streamed = self.refusal_blocks.get(key, "")
                if not isinstance(text, str) or not text.startswith(streamed):
                    raise CompletionError(
                        "invalid_response", "Completed text differs from streamed text",
                        retryable=True,
                    )
                paragraph.append(text)
            if paragraph:
                paragraphs.append("".join(paragraph))
        if self.text_blocks.keys() - text_keys or self.refusal_blocks.keys() - refusal_keys:
            raise CompletionError(
                "invalid_response", "Completed response omitted a streamed text block",
                retryable=True,
            )
        self.message_content = "\n\n".join(paragraphs)

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
            "response.refusal.delta",
        } and not isinstance(delta, str):
            raise CompletionError("invalid_response", "Response delta must be text", retryable=True)
        if kind == "response.output_text.delta":
            if "item_id" in event or "output_index" in event:
                key = self._refusal_key(event)
                self.text_blocks.setdefault(key, []).append(delta)
                item = key[0]
            else:
                item = None
                self.anonymous_text = True
            separator = "\n\n" if self.text_item not in (None, item) else ""
            self.text_item = item
            return [_chunk(content=separator + delta)] if isinstance(delta, str) else []
        if kind == "response.refusal.delta" and isinstance(delta, str):
            return self._consume_refusal(self._refusal_key(event), delta, delta=True)
        if kind == "response.refusal.done":
            return self._consume_refusal(self._refusal_key(event), event.get("refusal"))
        if kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            part = (event.get("item_id"), event.get("summary_index", event.get("content_index")))
            separator = "\n\n" if self.reasoning_part not in (None, part) else ""
            self.reasoning_part = part
            return [_chunk(reasoning_content=separator + delta)] if isinstance(delta, str) else []
        if kind == "response.function_call_arguments.delta" and isinstance(delta, str):
            if not any(
                (kind, key) in self.tools
                for kind, key in (
                    ("item", event.get("item_id")), ("output", event.get("output_index"))
                )
                if isinstance(key, (str, int))
            ):
                raise CompletionError(
                    "invalid_tool_calls", "Tool arguments received before call start", retryable=True
                )
            index = self._tool(event)
            info = self.tool_info[index]
            if info["finalized"]:
                raise CompletionError(
                    "invalid_tool_calls",
                    "Tool arguments received after call finalized",
                    retryable=True,
                )
            info["arguments"] += delta
            return [self._arguments(index, delta)]
        item = event.get("item")
        if kind == "response.output_item.added" and isinstance(item, dict):
            if item.get("type") == "message":
                self._refusal_key(event, item)
            if item.get("type") != "function_call":
                return []
            self._validate_tool_item(item)
            index = self._tool(event, item)
            if index in self.arguments:
                raise CompletionError(
                    "invalid_tool_calls", "Tool call started more than once", retryable=True
                )
            info = self.tool_info[index]
            self.arguments[index] = item["arguments"]
            info["arguments"] = self.arguments[index]
            call = {
                "index": index,
                "id": info["call_id"],
                "function": {"name": info["name"], "arguments": self.arguments[index]},
            }
            return [_chunk(tool_calls=[call])]
        if kind == "response.output_item.done" and isinstance(item, dict):
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                self.reasoning.append(
                    {
                        "type": "reasoning",
                        "summary": item.get("summary") or [],
                        "encrypted_content": item["encrypted_content"],
                    }
                )
            if item.get("type") == "refusal":
                refusal = item.get("refusal") if "refusal" in item else item.get("text")
                return self._consume_refusal(self._refusal_key(event, item, 0), refusal)
            if item.get("type") == "message":
                parts = item.get("content") or []
                if isinstance(parts, list):
                    chunks: list[dict[str, Any]] = []
                    for c_idx, part in enumerate(parts):
                        if isinstance(part, dict) and part.get("type") == "refusal":
                            refusal = (
                                part.get("refusal") if "refusal" in part else part.get("text")
                            )
                            chunks.extend(
                                self._consume_refusal(
                                    self._refusal_key(event, item, c_idx), refusal
                                )
                            )
                    if chunks:
                        return chunks
            if item.get("type") != "function_call":
                return []
            self._validate_tool_item(item)
            index = self._tool(event, item)
            info = self.tool_info[index]
            for field in ("id", "call_id", "name"):
                if item[field] != info[field]:
                    raise CompletionError(
                        "invalid_tool_calls",
                        f"Final tool {field} differs from streamed {field}",
                        retryable=True,
                    )
            final = item["arguments"]
            received = self.arguments.get(index, "")
            if not final.startswith(received):
                raise CompletionError(
                    "invalid_tool_calls",
                    "Final tool arguments differ from streamed arguments",
                    retryable=True,
                )
            if info["finalized"] and final != received:
                raise CompletionError(
                    "invalid_tool_calls", "Finalized tool arguments changed", retryable=True
                )
            info["finalized"] = True
            info["arguments"] = final
            if index not in self.arguments:
                self.arguments[index] = final
                return [_chunk(tool_calls=[
                    {
                        "index": index,
                        "id": info["call_id"],
                        "function": {"name": info["name"], "arguments": final},
                    }
                ])]
            suffix = final[len(received) :]
            return [self._arguments(index, suffix)] if suffix else []
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        if kind == "response.completed":
            for index, info in self.tool_info.items():
                if not info["finalized"]:
                    raise CompletionError(
                        "invalid_tool_calls",
                        f"Tool call '{info['name']}' was not finalized before completion",
                        retryable=True,
                    )
            output_items = response.get("output")
            if output_items is None and self.tool_info:
                raise CompletionError(
                    "invalid_tool_calls",
                    "Completed response missing output list for streamed tool calls",
                    retryable=True,
                )
            if output_items is not None:
                if not isinstance(output_items, list):
                    raise CompletionError(
                        "invalid_response",
                        "Completed response output must be a list",
                        retryable=True,
                    )
                final_calls = [
                    it
                    for it in output_items
                    if isinstance(it, dict) and it.get("type") == "function_call"
                ]
                if len(self.tool_info) != len(final_calls):
                    raise CompletionError(
                        "invalid_tool_calls",
                        f"Completed response output tool call count ({len(final_calls)}) differs "
                        f"from streamed count ({len(self.tool_info)})",
                        retryable=True,
                    )
                matched = set()
                for final_call in final_calls:
                    self._validate_tool_item(final_call)
                    index = self.tools.get(("item", final_call["id"]))
                    if index is None or index in matched:
                        raise CompletionError(
                            "invalid_tool_calls", "Completed response repeats or changes a tool ID",
                            retryable=True,
                        )
                    matched.add(index)
                    info = self.tool_info[index]
                    if any(
                        final_call[field] != info[field]
                        for field in ("id", "call_id", "name", "arguments")
                    ):
                        raise CompletionError(
                            "invalid_tool_calls",
                            "Completed response tool differs from finalized tool",
                            retryable=True,
                        )
                if output_items:
                    self._completed_content(output_items)
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
