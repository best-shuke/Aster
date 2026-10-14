"""Coordinate model request starts after HTTP 429 without changing model inputs."""

import hashlib
import math
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from weakref import WeakValueDictionary


def retry_after_seconds(value: str | None, *, now: float | None = None) -> float | None:
    if not isinstance(value, str) or len(value) > 256:
        return None
    value = value.strip()
    try:
        if value.isascii() and value.isdigit():
            seconds = float(value)
        else:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = max(0, date.timestamp() - (time.time() if now is None else now))
    except (ValueError, TypeError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) else None


@dataclass
class Admission:
    epoch: int
    accepted: bool = False


class RateLimitGate:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self._epoch = 0
        self._limited = False
        self._until = 0.0
        self._probe = None
        self._streak = 0
        self._spacing = 0.0
        self._next_start = 0.0

    def try_start(self):
        """Return an admission or a wait duration; never reserve sleeping callers."""
        with self._lock:
            now = self.clock()
            if now < self._until:
                return None, self._until - now, False
            if self._limited:
                if self._probe is not None:
                    return None, 0.1, True
                admission = Admission(self._epoch)
                self._probe = admission
                return admission, 0.0, False
            if self._spacing and now < self._next_start:
                return None, self._next_start - now, False
            if self._spacing:
                self._next_start = now + self._spacing
            return Admission(self._epoch), 0.0, False

    def limited(self, admission, *, retry_after=None, backoff=2):
        with self._lock:
            # Concurrent failures from the same burst are one congestion wave,
            # rather than exponentially multiplying the backoff per worker.
            if admission.epoch == self._epoch:
                self._streak = min(7, self._streak + 1)
            self._epoch += 1
            base = max(1, backoff)
            fallback = min(60, base * 2 ** max(0, self._streak - 1))
            delay = max(fallback, retry_after or 0)
            self._until = max(self._until, self.clock() + delay)
            self._limited = True
            self._probe = None
            self._spacing = max(self._spacing, min(30, base))
            self._next_start = self._until

    def accepted(self, admission):
        if admission is None:
            return
        with self._lock:
            if admission.accepted:
                return
            admission.accepted = True
            if admission.epoch != self._epoch:
                return  # A response started before a newer 429 cannot clear it.
            if self._probe is admission:
                self._probe = None
                self._limited = False
                self._next_start = self.clock() + self._spacing
            elif self._spacing:
                self._spacing *= 0.75
                if self._spacing < 0.1:
                    self._spacing = 0
                    self._streak = 0

    def release(self, admission):
        with self._lock:
            if self._probe is admission:
                self._probe = None


_gates = WeakValueDictionary()
_gates_lock = threading.Lock()


def shared_gate(settings):
    # Only a digest is retained as the registry key; no credentials are logged
    # or persisted. Account and endpoint share limits across client forks.
    scope = hashlib.sha256(
        (settings.base_url.rstrip("/") + "\0" + settings.api_key).encode()
    ).digest()
    with _gates_lock:
        gate = _gates.get(scope)
        if gate is None:
            gate = _gates[scope] = RateLimitGate()
        return gate
