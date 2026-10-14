"""Explicit configuration operations for completion clients and injected adapters."""

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from config import Settings
from contracts import CompletionClient, EventRecorder
from execution_trace import record_event
from usage import CacheStats


class _ClientAttributes(Protocol):
    settings: Settings | None
    cache_stats: CacheStats | None


@dataclass(frozen=True)
class ClientState:
    had_settings: bool
    settings: Settings | None
    had_cache_stats: bool
    cache_stats: CacheStats | None

    @classmethod
    def capture(cls, client: CompletionClient) -> "ClientState":
        return cls(
            hasattr(client, "settings"),
            getattr(client, "settings", None),
            hasattr(client, "cache_stats"),
            getattr(client, "cache_stats", None),
        )

    def restore(self, client: CompletionClient) -> None:
        attributes = cast(_ClientAttributes, client)
        if self.had_settings:
            attributes.settings = self.settings
        elif hasattr(client, "settings"):
            delattr(client, "settings")
        if self.had_cache_stats:
            attributes.cache_stats = self.cache_stats
        elif hasattr(client, "cache_stats"):
            delattr(client, "cache_stats")


def fork_client(
    client: CompletionClient, settings: Settings, cache_stats: CacheStats | None
) -> CompletionClient:
    factory = getattr(client, "fork", None)
    if callable(factory):
        return factory(settings=settings, cache_stats=cache_stats)
    candidate = copy(client)
    attributes = cast(_ClientAttributes, candidate)
    attributes.settings = settings
    if cache_stats is not None:
        attributes.cache_stats = cache_stats
    return candidate


def snapshot_client(client: CompletionClient) -> CompletionClient:
    settings = getattr(client, "settings", None)
    if settings is None:
        return copy(client)
    return fork_client(client, settings, getattr(client, "cache_stats", None))


def configure_client(
    client: CompletionClient, settings: Settings, stats: CacheStats | None
) -> None:
    attributes = cast(_ClientAttributes, client)
    attributes.settings = settings
    if isinstance(stats, CacheStats):
        attributes.cache_stats = stats


def client_settings(client: CompletionClient, workspace: Path) -> Settings:
    settings = getattr(client, "settings", None)
    if settings is not None:
        return settings
    return Settings(
        api_key="",
        base_url="",
        model="",
        session_file=workspace / ".worker-session.json",
        max_context_chars=120000,
        keep_recent_turns=6,
        max_task_seconds=None,
        max_subagents=1,
        workspace=workspace,
    )


def client_event(client: CompletionClient, name: str, **details: Any) -> None:
    recorder: EventRecorder | None = getattr(client, "recorder", None)
    record_event(recorder, name, **details)
