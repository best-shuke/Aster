"""Token budgets with usage anchors that survive only an unchanged request prefix."""

import hashlib
import json
import math
from urllib.parse import urlsplit

from config import parse_reasoning_effort
from usage import number

LOCAL_METADATA = "_harness"


def wire_messages(messages):
    return [{k: v for k, v in message.items() if k != LOCAL_METADATA} for message in messages]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def request_fingerprint(messages, tools, settings):
    return fingerprint(
        [
            getattr(settings, "base_url", ""),
            getattr(settings, "model", ""),
            parse_reasoning_effort(getattr(settings, "reasoning_effort", None)),
            tools or [],
            wire_messages(messages),
        ]
    )


def estimate_text_tokens(text):
    # Deliberately more room than pi's chars/4 heuristic, especially for Chinese.
    non_ascii = sum(ord(c) > 127 for c in text)
    return math.ceil((len(text) - non_ascii) / 3 + non_ascii * 2)


def estimate_tokens(value):
    return estimate_text_tokens(json.dumps(value, ensure_ascii=False))


def context_window(settings):
    explicit = getattr(settings, "context_window_tokens", None)
    if explicit:
        return explicit
    host = urlsplit(getattr(settings, "base_url", "")).hostname
    model = getattr(settings, "model", "")
    # Verified 2026-09-24: Cline's cline-recommended.generated.ts and DeepSeek's
    # official model details. Do not infer gateway limits from a model substring.
    if host == "api.cline.bot" and model == "cline-pass/deepseek-v4.1-flash":
        return 1_000_000
    if host == "api.deepseek.com" and model in {
        "deepseek-flash",
        "deepseek-v4.1-flash",
        "deepseek-v4-pro",
    }:
        return 1_000_000
    return 131_072  # Configurable fallback, not a claim about an unknown provider.


def input_budget(settings):
    window = context_window(settings)
    reserve = getattr(settings, "context_reserve_tokens", 16384)
    if reserve >= window:
        raise ValueError("DEEPSEEK_CONTEXT_RESERVE_TOKENS must be smaller than the context window")
    return window - reserve


def estimate_context(messages, tools, settings):
    """Usage includes the system/tools; only the unseen suffix is estimated.

    A digest validates both the original input and the returned assistant text.
    Compaction, edits, model/effort changes and different child histories therefore
    cannot reuse a stale usage anchor. No mutable client-wide 'last usage' is used.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        metadata = message.get(LOCAL_METADATA)
        if message.get("role") != "assistant" or not isinstance(metadata, dict):
            continue
        usage = metadata.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        if not number(prompt) or prompt <= 0 or metadata.get("input_messages") != index:
            continue
        if metadata.get("input_fingerprint") != request_fingerprint(
            messages[:index], tools, settings
        ):
            continue
        if metadata.get("response_fingerprint") != fingerprint(wire_messages([message])[0]):
            continue
        output = usage.get("completion_tokens")
        output = output if number(output) else estimate_tokens(wire_messages([message]))
        trailing = sum(estimate_tokens(m) for m in wire_messages(messages[index + 1 :]))
        return {
            "tokens": math.ceil(prompt + output + trailing),
            "source": "usage",
            "anchor_index": index,
        }
    return {
        "tokens": estimate_tokens(wire_messages(messages)) + estimate_tokens(tools or []),
        "source": "estimate",
        "anchor_index": None,
    }


def context_snapshot(messages, tools, settings):
    window = context_window(settings)
    max_chars = getattr(settings, "max_context_chars", None)
    return {
        **estimate_context(messages, tools, settings),
        "window_tokens": window,
        "budget_tokens": input_budget(settings) if max_chars is None else window,
        "max_chars": max_chars,
        "chars": len(json.dumps(wire_messages(messages), ensure_ascii=False))
        + len(json.dumps(tools or [], ensure_ascii=False)),
    }
