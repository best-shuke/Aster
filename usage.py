"""Provider usage accounting, kept outside the model-visible conversation."""

import copy
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


class UsageAccumulator:
    """Stream usage is cumulative: merge updates, never sum chunks."""

    def __init__(self):
        self.usage = {}
        self.provider = {}

    def add(self, obj):
        if not isinstance(obj, dict):
            return
        if isinstance(obj.get("data"), dict):
            obj = obj["data"]
        sources = [obj.get("usage")]
        choices = obj.get("choices")
        if not isinstance(choices, list):
            choices = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            sources.append(choice.get("usage"))
            message = choice.get("delta") or choice.get("message") or {}
            metadata = message.get("provider_metadata") or {}
            self.provider.update(
                {k: v for k, v in (metadata.get("deepseek") or {}).items() if v is not None}
            )
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if isinstance(value, dict):
                    previous = self.usage.get(key)
                    self.usage[key] = {
                        **(previous if isinstance(previous, dict) else {}),
                        **{k: v for k, v in value.items() if v is not None},
                    }
                elif value is not None:
                    self.usage[key] = value

    def result(self):
        usage = copy.deepcopy(self.usage)
        details = usage.get("prompt_tokens_details") or {}
        hit = details.get("cached_tokens")
        if hit is None:
            hit = usage.get(
                "prompt_cache_hit_tokens",
                usage.get("cached_tokens", self.provider.get("promptCacheHitTokens")),
            )
        miss = usage.get("prompt_cache_miss_tokens", self.provider.get("promptCacheMissTokens"))
        if usage.get("prompt_tokens") is None and number(hit) and number(miss):
            usage["prompt_tokens"] = hit + miss
        if number(hit):
            usage["prompt_tokens_details"] = {**details, "cached_tokens": hit}
        return usage or None


def response_usage(objects):
    accumulator = UsageAccumulator()
    for obj in objects:
        accumulator.add(obj)
    return accumulator.result()


def usage_metrics(usages, calls):
    totals = {
        key: sum(u[key] for u in usages if number(u.get(key)))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    cache_rows, writes, costs = [], [], []
    for usage in usages:
        details = usage.get("prompt_tokens_details") or {}
        prompt, cached = usage.get("prompt_tokens"), details.get("cached_tokens")
        if number(prompt) and number(cached) and cached <= prompt:
            cache_rows.append((prompt, cached))
        write = details.get("cache_write_tokens", usage.get("cache_creation_input_tokens"))
        if number(write):
            writes.append(write)
        if number(usage.get("cost")):
            costs.append(usage["cost"])
    prompt = sum(p for p, _ in cache_rows)
    hit = sum(h for _, h in cache_rows)
    return {
        "usage_observed_calls": len(usages),
        "usage": totals if usages else None,
        "cache": {
            "observed_calls": len(cache_rows),
            "unobserved_calls": calls - len(cache_rows),
            "prompt_tokens": prompt if cache_rows else None,
            "hit_tokens": hit if cache_rows else None,
            "miss_tokens": prompt - hit if cache_rows else None,
            "hit_rate": hit / prompt if prompt else None,
            "write_tokens": sum(writes) if writes else None,
            "write_observed_calls": len(writes),
        },
        "reported_cost": {"observed_calls": len(costs), "total": sum(costs) if costs else None},
    }


class CacheStats:
    """One entry per HTTP attempt, including children, summaries, and failures."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scope = ContextVar(f"cache_scope_{id(self)}", default=None)
        self.reset()

    def reset(self):
        with self.lock:
            self.calls = self.observed = self.prompt = self.hit = 0
            self.latest = None
            self.scopes = {}

    @contextmanager
    def request_scope(self, label):
        label = "".join(character for character in str(label) if character.isprintable())[:160]
        token = self.scope.set(label)
        try:
            yield
        finally:
            self.scope.reset(token)

    def record(self, usage, purpose, *, system_fingerprint=None):
        cache = usage_metrics([usage] if usage else [], 1)["cache"]
        with self.lock:
            self.calls += 1
            self.observed += cache["observed_calls"]
            self.prompt += cache["prompt_tokens"] or 0
            self.hit += cache["hit_tokens"] or 0
            self.latest = {"purpose": purpose, **cache}
            scope = self.scope.get()
            if scope is not None:
                label = f"{scope}:{purpose}"
                if label not in self.scopes and len(self.scopes) >= 128:
                    label = "other"
                entry = self.scopes.setdefault(
                    label,
                    {
                        "calls": 0,
                        "observed_calls": 0,
                        "prompt_tokens": 0,
                        "hit_tokens": 0,
                        "system_changes": 0,
                        "system_fingerprint": None,
                    },
                )
                entry["calls"] += 1
                entry["observed_calls"] += cache["observed_calls"]
                entry["prompt_tokens"] += cache["prompt_tokens"] or 0
                entry["hit_tokens"] += cache["hit_tokens"] or 0
                previous = entry["system_fingerprint"]
                if system_fingerprint is not None:
                    if previous is not None and previous != system_fingerprint:
                        entry["system_changes"] += 1
                    entry["system_fingerprint"] = system_fingerprint

    def snapshot(self):
        with self.lock:
            snapshot = {
                "calls": self.calls,
                "observed_calls": self.observed,
                "unknown_calls": self.calls - self.observed,
                "prompt_tokens": self.prompt,
                "hit_tokens": self.hit,
                "hit_rate": self.hit / self.prompt if self.prompt else None,
                "latest": copy.deepcopy(self.latest),
            }
            if self.scopes:
                snapshot["scopes"] = {
                    label: {
                        **entry,
                        "hit_rate": entry["hit_tokens"] / entry["prompt_tokens"]
                        if entry["prompt_tokens"]
                        else None,
                    }
                    for label, entry in self.scopes.items()
                }
            return snapshot

    def restore(self, saved):
        if not isinstance(saved, dict):
            return
        values = [saved.get(k) for k in ("calls", "observed_calls", "prompt_tokens", "hit_tokens")]
        if not all(number(v) for v in values):
            return
        calls, observed, prompt, hit = values
        if observed > calls or hit > prompt or int(calls) != calls or int(observed) != observed:
            return
        scopes = {}
        source = saved.get("scopes")
        keys = ("calls", "observed_calls", "prompt_tokens", "hit_tokens", "system_changes")
        if isinstance(source, dict) and len(source) <= 129:
            for label, entry in source.items():
                if not isinstance(label, str) or len(label) > 200 or not label.isprintable():
                    continue
                if not isinstance(entry, dict):
                    continue
                amounts = [entry.get(key) for key in keys]
                if not all(number(value) for value in amounts):
                    continue
                count, measured, tokens, cached, changes = amounts
                if (
                    measured > count
                    or cached > tokens
                    or changes > count
                    or any(int(value) != value for value in (count, measured, changes))
                ):
                    continue
                digest = entry.get("system_fingerprint")
                if digest is not None and (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    continue
                scopes[label] = {
                    **dict(zip(keys, amounts, strict=True)),
                    "system_fingerprint": digest,
                }
            if any(
                sum(entry[key] for entry in scopes.values()) > total
                for key, total in zip(keys[:4], values, strict=True)
            ):
                scopes = {}
        with self.lock:
            if self.calls == 0:
                self.calls, self.observed, self.prompt, self.hit = values
                self.latest = None  # No request has completed in this process yet.
                self.scopes = scopes
