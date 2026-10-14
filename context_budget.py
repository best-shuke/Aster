"""Token budgets with usage anchors that survive only an unchanged request prefix."""

import hashlib
import json
import math
import threading
from itertools import compress, repeat
from operator import is_not, itemgetter
from urllib.parse import urlsplit

from config import parse_reasoning_effort
from usage import number

LOCAL_METADATA = "_harness"

# WireCache entry fields: (message, compact JSON, default-separator JSON length, non-ASCII count).
_MESSAGE, _COMPACT, _PRETTY_LEN, _NON_ASCII = (itemgetter(i) for i in range(4))
_MISS = (None,)


def wire_messages(messages):
    return [{k: v for k, v in message.items() if k != LOCAL_METADATA} for message in messages]


class WireCache:
    """Per-message serialization memo for the read-only budget/display path.

    ``context_status`` (after every tool result), the per-round snapshot and compaction
    check in ``TaskRuntime``, and compaction itself all size and fingerprint ~the whole
    history, making a long session roughly O(n²). Message wire-content is immutable once
    appended (only the stripped ``_harness`` metadata is added later), so each message's
    serialized forms can be cached by object identity. A strong reference to each cached
    message keeps its ``id`` from being reused; ``sync`` drops entries for messages no
    longer present.

    The request-prefix hash is kept incrementally too, keyed by serialized content rather
    than identity, so a fingerprint over a prefix that extends the last one hashes only the
    new messages.

    Every result is byte- or number-identical to the uncached computation, so fingerprints,
    token estimates and char counts match whether or not a cache is supplied. Entries are
    immutable, so concurrent lookups are benign; the lock guards the multi-step updates to
    the hash states and the pruning.
    """

    __slots__ = ("_chain_head", "_chain_parts", "_chain_states", "_entries", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, tuple] = {}
        self._chain_head: str | None = None
        self._chain_parts: list[str] = []
        self._chain_states: list = []

    def _entry(self, message) -> tuple:
        key = id(message)
        hit = self._entries.get(key)
        if hit is not None and _MESSAGE(hit) is message:
            return hit
        wire = {k: v for k, v in message.items() if k != LOCAL_METADATA}
        compact = json.dumps(wire, ensure_ascii=False, separators=(",", ":"))
        pretty = json.dumps(wire, ensure_ascii=False)
        # The same count as estimate_text_tokens' sum(ord(c) > 127 ...): exactly the
        # characters (lone surrogates included) the ASCII codec cannot encode.
        non_ascii = 0 if pretty.isascii() else len(pretty) - len(pretty.encode("ascii", "ignore"))
        entry = (message, compact, len(pretty), non_ascii)
        self._entries[key] = entry
        return entry

    def _lookup(self, messages) -> list[tuple]:
        # The entry for each message. Each per-message step is a C-level map, so a
        # mostly cached history costs a pointer walk plus serializing only the misses.
        entries = list(map(self._entries.get, map(id, messages), repeat(_MISS)))
        stale = map(is_not, map(_MESSAGE, entries), messages)
        for index in compress(range(len(entries)), stale):
            entries[index] = self._entry(messages[index])
        return entries

    def _list_stats(self, messages) -> tuple[int, int]:
        # Length and non-ASCII count of json.dumps(wire_messages(messages), ensure_ascii=False):
        # "[" + items joined by the default ", " separator + "]"; the separators are ASCII.
        entries = self._lookup(messages)
        length = sum(map(_PRETTY_LEN, entries)) + 2 * max(0, len(entries) - 1) + 2
        return length, sum(map(_NON_ASCII, entries))

    def pretty_list_len(self, messages) -> int:
        # Equivalent to len(json.dumps(wire_messages(messages), ensure_ascii=False)).
        return self._list_stats(messages)[0]

    def list_tokens(self, messages) -> int:
        # Equivalent to estimate_tokens(wire_messages(messages)).
        return _estimate_tokens_from(*self._list_stats(messages))

    def message_tokens(self, message) -> int:
        # Equivalent to estimate_tokens(wire_messages([message])[0]).
        entry = self._entry(message)
        return _estimate_tokens_from(_PRETTY_LEN(entry), _NON_ASCII(entry))

    def message_fingerprint(self, message) -> str:
        # Equivalent to fingerprint(wire_messages([message])[0]).
        return hashlib.sha256(_COMPACT(self._entry(message)).encode()).hexdigest()

    def prefix_fingerprint(self, head_json: str, messages) -> str:
        # sha256 of the compact dump of [*head, wire_messages(messages)], which is
        # head_json[:-1] + ",[" + compact items joined by "," + "]]". UTF-8 encodes per
        # character, so the hash can be fed piecewise. The state after each message is
        # kept, so a prefix that extends the last one hashes only the new messages.
        parts = list(map(_COMPACT, self._lookup(messages)))
        with self._lock:
            if head_json != self._chain_head:
                # Hash first: an unencodable head must leave the previous states intact.
                initial = hashlib.sha256((head_json[:-1] + ",[").encode())
                self._chain_head, self._chain_parts, self._chain_states = head_json, [], [initial]
            chain, states = self._chain_parts, self._chain_states
            shared = self._shared_prefix(parts)
            if shared < len(chain) and shared < len(parts):
                self._truncate_chain(shared)
            for part in parts[len(chain) :]:
                state = states[-1].copy()
                state.update(("," + part if chain else part).encode())
                chain.append(part)
                states.append(state)
            final = states[len(parts)].copy()
        final.update(b"]]")
        return final.hexdigest()

    def _shared_prefix(self, parts) -> int:
        chain = self._chain_parts
        shared = min(len(parts), len(chain))
        # List equality short-circuits on identical strings, so the usual all-equal
        # case is a fast pointer walk.
        if parts[:shared] == chain[:shared]:
            return shared
        return next(i for i in range(shared) if parts[i] != chain[i])

    def _truncate_chain(self, length: int) -> None:
        del self._chain_parts[length:]
        del self._chain_states[length + 1 :]

    def sync(self, messages) -> None:
        with self._lock:
            for key in self._entries.keys() - set(map(id, messages)):
                del self._entries[key]
            if self._chain_parts:
                # Keep only the hashed prefix still at the head of the live history, holding
                # the live strings, so the chain cannot outgrow the live message set.
                parts = list(map(_COMPACT, self._lookup(messages[: len(self._chain_parts)])))
                shared = self._shared_prefix(parts)
                self._truncate_chain(shared)
                self._chain_parts[:] = parts[:shared]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def request_fingerprint(messages, tools, settings, cache: WireCache | None = None):
    head = [
        getattr(settings, "base_url", ""),
        getattr(settings, "model", ""),
        parse_reasoning_effort(getattr(settings, "reasoning_effort", None)),
        tools or [],
    ]
    if cache is None:
        return fingerprint([*head, wire_messages(messages)])
    # Byte-identical to the branch above: the cached messages array is hashed as the
    # final element of the compact-serialized head list.
    head_json = json.dumps(head, ensure_ascii=False, separators=(",", ":"))
    return cache.prefix_fingerprint(head_json, messages)


def _estimate_tokens_from(length, non_ascii):
    return math.ceil((length - non_ascii) / 3 + non_ascii * 2)


def estimate_text_tokens(text):
    # Deliberately more room than pi's chars/4 heuristic, especially for Chinese.
    non_ascii = sum(ord(c) > 127 for c in text)
    return _estimate_tokens_from(len(text), non_ascii)


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


def estimate_context(messages, tools, settings, cache: WireCache | None = None):
    """Usage includes the system/tools; only the unseen suffix is estimated.

    A digest validates both the original input and the returned assistant text.
    Compaction, edits, model/effort changes and different child histories therefore
    cannot reuse a stale usage anchor. No mutable client-wide 'last usage' is used.
    A ``cache`` only memoizes serialization; the result is identical without one.
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
            messages[:index], tools, settings, cache
        ):
            continue
        response = (
            fingerprint(wire_messages([message])[0])
            if cache is None
            else cache.message_fingerprint(message)
        )
        if metadata.get("response_fingerprint") != response:
            continue
        output = usage.get("completion_tokens")
        if not number(output):
            output = (
                estimate_tokens(wire_messages([message]))
                if cache is None
                else cache.list_tokens([message])
            )
        trailing = (
            sum(estimate_tokens(m) for m in wire_messages(messages[index + 1 :]))
            if cache is None
            else sum(cache.message_tokens(m) for m in messages[index + 1 :])
        )
        return {
            "tokens": math.ceil(prompt + output + trailing),
            "source": "usage",
            "anchor_index": index,
        }
    history = (
        estimate_tokens(wire_messages(messages)) if cache is None else cache.list_tokens(messages)
    )
    return {
        "tokens": history + estimate_tokens(tools or []),
        "source": "estimate",
        "anchor_index": None,
    }


def context_snapshot(messages, tools, settings, cache: WireCache | None = None):
    window = context_window(settings)
    max_chars = getattr(settings, "max_context_chars", None)
    if cache is not None:
        cache.sync(messages)
    message_chars = (
        cache.pretty_list_len(messages)
        if cache is not None
        else len(json.dumps(wire_messages(messages), ensure_ascii=False))
    )
    return {
        **estimate_context(messages, tools, settings, cache),
        "window_tokens": window,
        "budget_tokens": input_budget(settings) if max_chars is None else window,
        "max_chars": max_chars,
        "chars": message_chars + len(json.dumps(tools or [], ensure_ascii=False)),
    }
