import math
import time
from uuid import uuid4

from execution_trace import current_trace
from usage import number


class GenerationMetrics:
    def __init__(self, usage, on_progress=None, *, clock=None):
        self.usage = usage
        self.on_progress = on_progress
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.first_output = None
        self.last_output = None
        self.weighted_chars = 0
        self.samples = 0
        self.last_publish = None
        trace = current_trace()
        self.trace = trace if trace.request_id else trace.request()
        self.request_id = self.trace.request_id
        self.generation_id = uuid4().hex

    def observe(self, payload):
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        fragments = []
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                fragments.append(content)
            text_parts = []
            for field in ("reasoning_content", "reasoning"):
                text = delta.get(field)
                if isinstance(text, str) and text and text not in text_parts:
                    text_parts.append(text)
            fragments.extend(text_parts)
            for call in delta.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                fragments.extend(
                    value
                    for key in ("name", "arguments")
                    if isinstance((value := function.get(key)), str) and value
                )
        if not fragments:
            return
        now = self.clock()
        if self.first_output is None:
            self.first_output = now
        self.last_output = now
        self.samples += 1
        for text in fragments:
            self.weighted_chars += len(text) + 5 * sum(ord(character) > 127 for character in text)
        if self.last_publish is None or now - self.last_publish >= 0.5:
            self.last_publish = now
            self.publish()

    def snapshot(self):
        completion_tokens = (self.usage.result() or {}).get("completion_tokens")
        measured = number(completion_tokens) and completion_tokens > 0
        tokens = completion_tokens if measured else math.ceil(self.weighted_chars / 3)
        duration = (
            self.last_output - self.first_output
            if self.first_output is not None and self.last_output is not None
            else 0
        )
        return {
            "request_id": self.request_id,
            "generation_id": self.generation_id,
            "trace": self.trace.as_dict(),
            "tokens": tokens,
            "source": "usage" if measured else "estimate",
            "generation_seconds": duration,
            "excluded_wait_seconds": (
                self.first_output - self.started if self.first_output is not None else None
            ),
            "tps": tokens / duration if self.samples > 1 and duration > 0.001 else None,
        }

    def publish(self, *, finished=False):
        if self.on_progress:
            self.on_progress({"phase": "generation", **self.snapshot(), "finished": finished})
