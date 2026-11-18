"""Concurrent request reservations and measured/estimated token accounting."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from threading import RLock

from context_budget import LOCAL_METADATA, estimate_tokens, wire_messages
from usage import number, uncached_tokens


class TokenBudgetExhausted(RuntimeError):
    pass


_lock = RLock()
_scopes = ContextVar("token_budget_scopes", default=())
_attempt = ContextVar("token_budget_attempt", default=None)
_persistence = ContextVar("token_budget_persistence", default=None)


@dataclass(eq=False)
class TokenBudget:
    name: str
    limit: int | None
    measured: int = 0
    estimated: int = 0
    unknown: int = 0
    requests: int = 0
    reserved: int = 0

    @classmethod
    def restore(cls, name, limit, data):
        data = data if isinstance(data, dict) else {}
        counts = {key: data.get(key, 0) for key in ("measured", "estimated", "unknown", "requests")}
        return cls(name, limit, **{
            key: value if type(value) is int and value >= 0 else 0 for key, value in counts.items()
        })

    def snapshot(self):
        with _lock:
            return {
                "limit": self.limit, "measured": self.measured, "estimated": self.estimated,
                "unknown": self.unknown, "requests": self.requests, "reserved": self.reserved,
            }

    def remaining(self):
        if self.limit is None:
            return None
        with _lock:
            return max(0, self.limit - self.measured - self.estimated - self.reserved)


def current_budgets():
    return _scopes.get()


def restore_budget(name, limit, data=None):
    return TokenBudget.restore(name, limit, data) if limit is not None or data else None


@contextmanager
def budget_persistence(callback):
    """Keep the owner's persistence hook in inherited worker contexts."""
    token = _persistence.set(callback)
    try:
        yield
    finally:
        _persistence.reset(token)


def persist_budgets():
    callback = _persistence.get()
    ledgers = current_budgets()
    if callback is not None and ledgers:
        callback(ledgers)


@contextmanager
def budget_scope(*ledgers):
    token = _scopes.set(tuple(dict.fromkeys((*_scopes.get(), *(item for item in ledgers if item is not None)))))
    try:
        yield
    finally:
        _scopes.reset(token)


@dataclass
class Reservation:
    ledgers: tuple[TokenBudget, ...]
    prompt_estimate: int
    output_cap: int | None
    dispatched: bool = False
    measured: int | None = None
    estimated: int | None = None
    incomplete_usage: bool = False

    def finish(self):
        with _lock:
            for ledger in self.ledgers:
                ledger.reserved -= self.prompt_estimate + (self.output_cap or 0)
                if not self.dispatched:
                    continue
                ledger.requests += 1
                if self.measured is not None:
                    ledger.measured += self.measured
                elif self.estimated is not None:
                    ledger.estimated += self.estimated
                if self.incomplete_usage or (self.measured is None and self.estimated is None):
                    ledger.unknown += 1


@contextmanager
def reserve_request(prompt_estimate, output_cap, *, minimum=1):
    ledgers = current_budgets()
    if not ledgers:
        yield None
        return
    with _lock:
        for ledger in ledgers:
            remaining = ledger.remaining()
            if remaining is None:
                continue
            available = remaining - prompt_estimate
            if available < minimum:
                raise TokenBudgetExhausted(
                    f"{ledger.name}: estimated input and minimum output require "
                    f"{prompt_estimate + minimum} tokens but only {remaining} remain"
                )
            output_cap = available if output_cap is None else min(available, output_cap)
        if output_cap is not None and output_cap < minimum:
            raise TokenBudgetExhausted(f"The configured output cap is below {minimum} tokens")
        reservation = Reservation(ledgers, prompt_estimate, output_cap)
        for ledger in ledgers:
            ledger.reserved += prompt_estimate + (output_cap or 0)
    try:
        yield reservation
    finally:
        reservation.finish()


@contextmanager
def reserve_completion(messages, tools, maximum, *, minimum=1):
    if not current_budgets():
        yield None
        return
    estimate = estimate_tokens(wire_messages(messages)) + estimate_tokens(tools or [])
    with reserve_request(estimate, maximum, minimum=minimum) as reservation:
        token = _attempt.set(reservation)
        try:
            yield reservation
        finally:
            _attempt.reset(token)


def mark_dispatched():
    reservation = _attempt.get()
    if reservation is not None:
        reservation.dispatched = True


def observe_usage(usage, response=None):
    reservation = _attempt.get()
    if reservation is None or not reservation.dispatched:
        return
    measured = uncached_tokens(usage)
    if measured is not None:
        reservation.measured = measured
        reservation.incomplete_usage = False
    elif isinstance(usage, dict) and any(
        number(usage.get(key)) for key in ("prompt_tokens", "completion_tokens")
    ):
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        reservation.measured = math.ceil(
            (max(0, prompt - (cached if number(cached) else 0)) if number(prompt) else 0)
            + (output if number(output) else 0)
        )
        reservation.incomplete_usage = True
    elif response is not None:
        estimate = estimate_tokens({key: value for key, value in response.items() if key != LOCAL_METADATA})
        reservation.estimated = reservation.prompt_estimate + estimate


def complete_with_budget(client, messages, *args, **options):
    if not current_budgets() or getattr(client, "accounts_attempt_tokens", False) is True:
        return client.complete(messages, *args, **options)
    from deepseek_client import CompletionError

    tools = args[0] if args else options.get("tools")
    settings = getattr(client, "settings", None)
    cap = options.get("max_output_tokens") or getattr(settings, "max_output_tokens", None)
    try:
        with reserve_completion(messages, tools, cap) as reservation:
            options["max_output_tokens"] = reservation.output_cap
            mark_dispatched()
            response = client.complete(messages, *args, **options)
            observe_usage((response.get(LOCAL_METADATA) or {}).get("usage"), response)
            return response
    except TokenBudgetExhausted as exc:
        raise CompletionError("budget_exhausted", str(exc)) from exc
