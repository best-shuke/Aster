"""Anthropic Messages adapter over the shared transport and internal message contract."""

import hashlib
import json
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from config import parse_reasoning_effort
from deepseek_client import CompletionError, DeepSeekClient, _StreamAccumulator
from model_api import model_endpoint, replay_state
from model_messages import protocol_messages
from responses_protocol import _chunk, sse_events
from usage import number

BLOCKS_KEY = "anthropic_blocks"


def anthropic_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    result: dict[str, Any] = {}
    if number(usage.get("input_tokens")):
        cached = (
            usage.get("cache_read_input_tokens")
            if number(usage.get("cache_read_input_tokens"))
            else 0
        )
        written = (
            usage.get("cache_creation_input_tokens")
            if number(usage.get("cache_creation_input_tokens"))
            else 0
        )
        result["prompt_tokens"] = usage["input_tokens"] + cached + written
        if "cache_read_input_tokens" in usage:
            result["prompt_tokens_details"] = {
                "cached_tokens": cached,
                "cache_write_tokens": written,
            }
    if number(usage.get("output_tokens")):
        result["completion_tokens"] = usage["output_tokens"]
    if "prompt_tokens" in result and "completion_tokens" in result:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result or None


class _AnthropicEvents:
    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.tools: dict[int, int] = {}
        self.arguments: dict[int, str] = {}
        self.closed: set[int] = set()
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.done = False

    def chunks(self, event: Any) -> list[dict[str, Any]]:
        if not isinstance(event, dict):
            raise CompletionError(
                "invalid_response", "Anthropic event must be an object", retryable=True
            )
        kind = event.get("type")
        if kind == "error":
            error = event.get("error") or {}
            code = error.get("type") if isinstance(error, dict) else None
            rate_limit = code == "rate_limit_error"
            raise CompletionError(
                "http_429" if rate_limit else "provider_error",
                "Anthropic service returned an error",
                retryable=rate_limit or code in {"overloaded_error", "api_error"},
            )
        if kind == "message_start":
            message = event.get("message")
            if not isinstance(message, dict):
                raise CompletionError(
                    "invalid_response", "Anthropic message must be an object", retryable=True
                )
            if isinstance(message.get("usage"), dict):
                self.usage.update(message["usage"])
            return [_chunk(usage=anthropic_usage(self.usage))]
        index = event.get("index")
        if kind == "content_block_start":
            block = event.get("content_block")
            if not isinstance(index, int) or not isinstance(block, dict) or index in self.blocks:
                raise CompletionError(
                    "invalid_response", "Invalid Anthropic content block", retryable=True
                )
            self.blocks[index] = dict(block)
            if block.get("type") == "text":
                return [_chunk(content=block.get("text") or "")]
            if block.get("type") == "thinking":
                return [_chunk(reasoning_content=block.get("thinking") or "")]
            if block.get("type") == "redacted_thinking":
                return []
            if block.get("type") == "tool_use":
                self.tools[index] = len(self.tools)
                arguments = (
                    json.dumps(block["input"], ensure_ascii=False) if block.get("input") else ""
                )
                self.arguments[index] = arguments
                return [
                    _chunk(
                        tool_calls=[
                            {
                                "index": self.tools[index],
                                "id": block.get("id") or "",
                                "function": {
                                    "name": block.get("name") or "",
                                    "arguments": arguments,
                                },
                            }
                        ]
                    )
                ]
            raise CompletionError("unsupported_content", "Unsupported Anthropic content block")
        if kind == "content_block_delta":
            if (
                index not in self.blocks
                or index in self.closed
                or not isinstance(event.get("delta"), dict)
            ):
                raise CompletionError(
                    "invalid_response", "Anthropic delta has no content block", retryable=True
                )
            block, delta = self.blocks[index], event["delta"]
            delta_type = delta.get("type")
            for native, public, normalized in (
                ("text_delta", "text", "content"),
                ("thinking_delta", "thinking", "reasoning_content"),
                ("signature_delta", "signature", None),
            ):
                if delta_type == native:
                    if block["type"] != ("text" if public == "text" else "thinking"):
                        raise CompletionError(
                            "invalid_response",
                            "Anthropic delta has the wrong block type",
                            retryable=True,
                        )
                    value = delta.get(public)
                    if not isinstance(value, str):
                        raise CompletionError(
                            "invalid_response", "Anthropic delta is not text", retryable=True
                        )
                    block[public] = (block.get(public) or "") + value
                    return [_chunk(**{normalized: value})] if normalized else []
            if delta_type == "input_json_delta" and index in self.tools:
                value = delta.get("partial_json")
                if not isinstance(value, str):
                    raise CompletionError(
                        "invalid_response", "Invalid tool JSON delta", retryable=True
                    )
                self.arguments[index] += value
                return [
                    _chunk(
                        tool_calls=[{"index": self.tools[index], "function": {"arguments": value}}]
                    )
                ]
            return []
        if kind == "content_block_stop":
            if index not in self.blocks or index in self.closed:
                raise CompletionError(
                    "invalid_response", "Anthropic block closed out of order", retryable=True
                )
            self.closed.add(index)
        if kind == "content_block_stop" and index in self.tools:
            arguments = self.arguments[index] or "{}"
            try:
                self.blocks[index]["input"] = json.loads(arguments)
            except ValueError as exc:
                raise CompletionError(
                    "invalid_tool_calls", "Anthropic tool input is incomplete", retryable=True
                ) from exc
            if not self.arguments[index]:
                return [
                    _chunk(
                        tool_calls=[{"index": self.tools[index], "function": {"arguments": "{}"}}]
                    )
                ]
        if kind == "message_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                raise CompletionError(
                    "invalid_response", "Anthropic message delta must be an object", retryable=True
                )
            self.stop_reason = delta.get("stop_reason") or self.stop_reason
            if isinstance(event.get("usage"), dict):
                self.usage.update(event["usage"])
            return [_chunk(usage=anthropic_usage(self.usage))]
        if kind == "message_stop":
            if self.stop_reason is None or self.closed != set(self.blocks):
                raise CompletionError(
                    "incomplete_stream", "Anthropic response has no stop reason", retryable=True
                )
            self.done = True
            reason = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
                "max_tokens": "length",
            }.get(self.stop_reason, self.stop_reason)
            return [_chunk(reason)]
        return []


class AnthropicClient(DeepSeekClient):
    def _output_tokens(self, messages, tools, max_output_tokens):
        # Native Messages requires this field. DeepSeek's HTTP endpoint makes it optional.
        if (
            max_output_tokens is None
            and self.settings.max_output_tokens is None
            and urlsplit(self.settings.base_url).hostname != "api.deepseek.com"
        ):
            max_output_tokens = 8192
        return super()._output_tokens(messages, tools, max_output_tokens)

    def _request_body(self, messages, tools, streaming, max_output_tokens) -> dict[str, Any]:
        compat = self.settings.model_compat
        system, items = [], []
        deepseek = urlsplit(self.settings.base_url).hostname == "api.deepseek.com"
        for message in protocol_messages(messages, self.settings):
            role, text = message["role"], message.get("content") or ""
            if role == "system" and not items:
                system.append(text)
                continue
            if role == "system":
                # DeepSeek accepts interleaved system messages. Other Messages
                # backends receive runtime notes as user content, as pi does.
                items.append(
                    {
                        "role": "system" if deepseek else "user",
                        "content": text if deepseek else "[Runtime instructions]\n" + text,
                    }
                )
            elif role == "user":
                items.append({"role": "user", "content": text})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": text,
                }
                if items and items[-1]["role"] == "user" and isinstance(items[-1]["content"], list):
                    items[-1]["content"].append(block)
                else:
                    items.append({"role": "user", "content": [block]})
            elif role == "assistant":
                blocks = replay_state(message, self.settings).get(BLOCKS_KEY)
                if blocks:
                    blocks = [dict(block) for block in blocks]
                    for block in blocks:
                        if block.get("type") == "thinking" and not block.get("signature"):
                            if compat.allow_empty_thinking_signature:
                                block["signature"] = ""
                            else:
                                block.update(type="text", text=block.pop("thinking", ""))
                                block.pop("signature", None)
                if not blocks:
                    blocks = [{"type": "text", "text": text}] if text else []
                    for call in message.get("tool_calls") or []:
                        function = call["function"]
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": function["name"],
                                "input": json.loads(function["arguments"]),
                            }
                        )
                if blocks:
                    items.append({"role": "assistant", "content": blocks})
        limit = self._output_tokens(messages, tools, max_output_tokens)
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": items,
            "stream": streaming,
        }
        if limit is not None:
            body["max_tokens"] = limit
        if system:
            body["system"] = "\n\n".join(system)
        if tools is not None:
            body["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ]
            body["tool_choice"] = {"type": "auto"}
        effort = parse_reasoning_effort(self.settings.reasoning_effort)
        if effort == "off":
            body["thinking"] = {"type": "disabled"}
        elif deepseek or compat.adaptive_thinking:
            body["thinking"] = {"type": "adaptive" if compat.adaptive_thinking else "enabled"}
            body["output_config"] = {"effort": effort}
        else:
            budget = min({"low": 1024, "high": 8192, "max": 16384}[effort], limit - 1)
            if budget < 1024:
                raise CompletionError(
                    "invalid_arguments", "Anthropic thinking requires max_tokens greater than 1024"
                )
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return body

    @staticmethod
    def _system_fingerprint(body: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    key: body.get(key)
                    for key in ("model", "system", "tools", "thinking", "output_config")
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _http_request(self, body: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            model_endpoint(self.settings.base_url, "anthropic-messages"),
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "x-api-key": self.settings.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    @classmethod
    def _consume_stream(
        cls, response, state, deadline, generation_deadline, on_usage, metrics=None
    ):
        events = _AnthropicEvents()
        for event in sse_events(
            response,
            lambda: cls._check_stream_deadline(deadline, generation_deadline, state.finish_reason),
        ):
            for payload in events.chunks(event):
                if on_usage:
                    on_usage(payload)
                if metrics:
                    metrics.observe(payload)
                state.add_payload(payload)
            state.metadata[BLOCKS_KEY] = list(events.blocks.values())
            if events.done:
                state.saw_done = True
                return
        raise CompletionError(
            "incomplete_stream", "Anthropic stream ended before message_stop", retryable=True
        )

    def _read_nonstream_response(self, response, observed):
        payload = json.load(response)
        if isinstance(payload, dict) and payload.get("type") == "error":
            _AnthropicEvents().chunks(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
            raise CompletionError(
                "invalid_response", "Anthropic response has no content blocks", retryable=True
            )
        state = _StreamAccumulator(lambda _: None, None)
        events = _AnthropicEvents()
        synthetic = [{"type": "message_start", "message": payload}]
        for index, block in enumerate(payload["content"]):
            synthetic.extend(
                [
                    {"type": "content_block_start", "index": index, "content_block": block},
                    {"type": "content_block_stop", "index": index},
                ]
            )
        synthetic.extend(
            [
                {"type": "message_delta", "delta": {"stop_reason": payload.get("stop_reason")}},
                {"type": "message_stop"},
            ]
        )
        for event in synthetic:
            for chunk in events.chunks(event):
                observed.add(chunk)
                state.add_payload(chunk)
        state.metadata[BLOCKS_KEY] = list(events.blocks.values())
        state.saw_done = events.done
        return state.finish()
