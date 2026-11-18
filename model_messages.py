"""Prepare internal history for a model without mutating the saved conversation."""

import hashlib
import re

from model_api import protocol_origin


def protocol_messages(messages, settings):
    origin = protocol_origin(settings)

    def call_id(value):
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            return value
        return "call_" + hashlib.sha256(value.encode()).hexdigest()[:40]

    result = []
    for original in messages:
        message = dict(original)
        metadata = original.get("_harness") or {}
        if metadata.get("protocol_origin") not in {None, origin}:
            message.pop("reasoning", None)
            message.pop("reasoning_content", None)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "id": call_id(call["id"])} for call in message["tool_calls"]
            ]
        if message.get("tool_call_id"):
            message["tool_call_id"] = call_id(message["tool_call_id"])
        result.append(message)
    return result
