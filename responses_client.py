"""API-key Responses client, sharing the model transport and normalized events."""

import hashlib
import json
import urllib.request
from typing import Any

from config import parse_reasoning_effort
from contracts import Message
from deepseek_client import CompletionError, DeepSeekClient, _StreamAccumulator
from model_api import model_endpoint, protocol_origin
from model_messages import protocol_messages
from responses_protocol import (
    REASONING_KEY,
    _ResponsesEvents,
    response_error,
    responses_input,
    responses_tool,
    sse_events,
)


class ResponsesClient(DeepSeekClient):
    @classmethod
    def _consume_stream(
        cls, response, state, deadline, generation_deadline, on_usage, metrics=None
    ) -> None:
        events = _ResponsesEvents()
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
            if events.reasoning:
                state.metadata[REASONING_KEY] = list(events.reasoning)
            if events.done:
                state.saw_done = True
                break
        if not events.done:
            raise CompletionError(
                "incomplete_stream", "Responses stream ended before completion", retryable=True
            )

    def _read_nonstream_response(self, response, observed) -> Message:
        payload = json.load(response)
        if isinstance(payload, dict) and payload.get("error"):
            raise response_error(payload["error"])
        if not isinstance(payload, dict) or not isinstance(payload.get("output"), list):
            raise CompletionError(
                "invalid_response", "Responses response has no output array", retryable=True
            )
        state = _StreamAccumulator(lambda _: None, None)
        events = _ResponsesEvents()
        for index, item in enumerate(payload["output"]):
            if not isinstance(item, dict):
                raise CompletionError(
                    "invalid_response", "Invalid Responses output item", retryable=True
                )
            kind = item.get("type")
            parts = item.get("content") or item.get("summary") or []
            if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
                raise CompletionError(
                    "invalid_response", "Invalid Responses content blocks", retryable=True
                )
            if kind == "message":
                for c_idx, part in enumerate(parts):
                    if part.get("type") == "output_text":
                        if not isinstance(part.get("text"), str):
                            raise CompletionError(
                                "invalid_response", "Response text must be a string", retryable=True
                            )
                        state.add_payload({"choices": [{"delta": {"content": part["text"]}}]})
                    elif part.get("type") == "refusal":
                        refusal = part.get("refusal") if "refusal" in part else part.get("text")
                        if not isinstance(refusal, str):
                            raise CompletionError(
                                "invalid_response", "Refusal must be text", retryable=True
                            )
                        state.add_payload({"choices": [{"delta": {"content": refusal}}]})
                        key = events._refusal_key({"output_index": index}, item, c_idx)
                        events.refusal_blocks[key] = refusal
            elif kind == "refusal":
                refusal = item.get("refusal") if "refusal" in item else item.get("text")
                if not isinstance(refusal, str):
                    raise CompletionError(
                        "invalid_response", "Refusal must be text", retryable=True
                    )
                state.add_payload({"choices": [{"delta": {"content": refusal}}]})
                key = events._refusal_key({"output_index": index}, item)
                events.refusal_blocks[key] = refusal
            elif kind == "reasoning":
                for part in parts:
                    if isinstance(part.get("text"), str):
                        state.add_payload(
                            {"choices": [{"delta": {"reasoning_content": part["text"]}}]}
                        )
            elif kind != "function_call":
                raise CompletionError("unsupported_content", "Unsupported Responses output item")
            for chunk in events.chunks(
                {"type": "response.output_item.done", "output_index": index, "item": item}
            ):
                state.add_payload(chunk)
        terminal = {
            "completed": "response.completed",
            "incomplete": "response.incomplete",
            "failed": "response.failed",
        }.get(payload.get("status"))
        if terminal is None:
            raise CompletionError(
                "incomplete_stream", "Response did not complete", partial_message=state.snapshot()
            )
        for chunk in events.chunks({"type": terminal, "response": payload}):
            observed.add(chunk)
            state.add_payload(chunk)
        state.metadata[REASONING_KEY] = events.reasoning
        state.saw_done = events.done
        return state.finish()

    def _request_body(self, messages, tools, streaming, max_output_tokens) -> dict[str, Any]:
        compat = self.settings.model_compat
        instructions, items = responses_input(
            protocol_messages(messages, self.settings),
            system_role="developer" if compat.supports_developer_role else "system",
            origin=protocol_origin(self.settings),
            encrypted_reasoning=compat.supports_encrypted_reasoning,
        )
        body: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": items,
            "stream": streaming,
        }
        if compat.supports_store:
            body["store"] = False
        if tools is not None:
            body.update(tools=[responses_tool(tool) for tool in tools], tool_choice="auto")
        limit = self._output_tokens(messages, tools, max_output_tokens)
        if limit is not None:
            body["max_output_tokens"] = limit
        if compat.supports_reasoning_effort:
            effort = parse_reasoning_effort(self.settings.reasoning_effort)
            body["reasoning"] = {"effort": "none" if effort == "off" else effort}
            if compat.supports_encrypted_reasoning and effort != "off":
                body["include"] = ["reasoning.encrypted_content"]
        return body

    @staticmethod
    def _system_fingerprint(body: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {key: body.get(key) for key in ("model", "instructions", "tools", "reasoning")},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def _http_request(self, body: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            model_endpoint(self.settings.base_url, "openai-responses"),
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
