import asyncio
import concurrent.futures
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import httpx2
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from mcp import Client
from mcp import types as mcp_types
from mcp.client.session import IncomingMessage
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from referencing import Registry
from referencing.exceptions import Unresolvable

from file_state import atomic_write_text, local_code_files
from tool_runtime import ExecutionContext, ToolResult


@lru_cache(maxsize=256)
def _argument_validator(serialized_schema: str):
    schema = json.loads(serialized_schema)
    validator = validator_for(schema)
    validator.check_schema(schema)
    # MCP schemas may use local $defs, but argument validation must never
    # fetch an arbitrary remote $ref or block on an extra network request.
    return validator(schema, registry=Registry())


SERVER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
VARIABLE = re.compile(r"\$\{([^}]+)\}")
# Search terms: runs of letters or digits, so snake_case and paths split into words.
SEARCH_TERM = re.compile(r"[^\W_]+")
# Tool paths listed when a search matches nothing, so the model can still pick one.
AVAILABLE_TOOL_LIMIT = 100
# Seconds a closing runtime lets connections stop their servers before cancelling them.
TEARDOWN_TIMEOUT = 5
ALLOWED_SERVER_FIELDS = {
    "command",
    "args",
    "env",
    "cwd",
    "url",
    "headers",
    "transport",
    "type",
    "disabled",
    "timeout",
    "required",
    "mode",
}


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    disabled: bool = False
    timeout: float = 30
    required: bool = False
    mode: str = "auto"


@dataclass
class _Request:
    operation: str
    payload: dict[str, Any]
    deadline: float
    future: concurrent.futures.Future[Any] = field(default_factory=concurrent.futures.Future)
    abandoned: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task | None = None
    cancelled: threading.Event | None = None


class McpExecutionUnknownError(TimeoutError):
    pass


def _wait_future(future, deadline, cancelled=None):
    while True:
        if future.done():
            return future.result()
        if cancelled is not None and cancelled.is_set():
            raise TimeoutError("MCP operation cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MCP operation deadline reached")
        try:
            return future.result(timeout=min(0.05, remaining))
        except concurrent.futures.TimeoutError:
            if future.done():
                return future.result()


def _expand(value: str, workspace: Path, *, allow_missing: bool = False) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "workspace":
            return str(workspace)
        if name not in os.environ:
            if allow_missing:
                return match.group(0)
            raise ValueError(f"Environment variable {name!r} is required by MCP configuration")
        return os.environ[name]

    return VARIABLE.sub(replace, value)


def _string_map(
    value: Any, location: str, workspace: Path, *, allow_missing: bool = False
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{location} must be an object containing string values")
    return {
        key: _expand(item, workspace, allow_missing=allow_missing) for key, item in value.items()
    }


def load_mcp_config(
    path: Path,
    workspace: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, McpServerConfig]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read MCP config {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("mcpServers", {}), dict):
        raise ValueError("MCP config must contain an mcpServers object")

    servers: dict[str, McpServerConfig] = {}
    for name, raw in payload.get("mcpServers", {}).items():
        location = f"mcpServers.{name}"
        if not isinstance(name, str) or not SERVER_NAME.fullmatch(name):
            raise ValueError(f"Invalid MCP server name {name!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must be an object")
        unknown = set(raw) - ALLOWED_SERVER_FIELDS
        if unknown:
            raise ValueError(f"Unknown fields in {location}: {', '.join(sorted(unknown))}")

        disabled = raw.get("disabled", False)
        if not isinstance(disabled, bool):
            raise ValueError(f"{location}.disabled must be boolean")
        required = raw.get("required", False)
        if not isinstance(required, bool):
            raise ValueError(f"{location}.required must be boolean")

        command, url = raw.get("command"), raw.get("url")
        if (command is None) == (url is None):
            raise ValueError(f"{location} must define exactly one of command or url")
        if command is not None and (not isinstance(command, str) or not command.strip()):
            raise ValueError(f"{location}.command must be a non-empty string")
        if url is not None and not isinstance(url, str):
            raise ValueError(f"{location}.url must be a string")

        expanded_command = (
            _expand(command, workspace, allow_missing=disabled) if command is not None else None
        )
        expanded_url = _expand(url, workspace, allow_missing=disabled) if url is not None else None
        if (
            expanded_url is not None
            and not expanded_url.startswith(("http://", "https://"))
            and not (disabled and VARIABLE.search(expanded_url))
        ):
            raise ValueError(f"{location}.url must be an HTTP(S) URL")

        args = raw.get("args", [])
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError(f"{location}.args must be an array of strings")
        raw_cwd = raw.get("cwd")
        if raw_cwd is not None and not isinstance(raw_cwd, str):
            raise ValueError(f"{location}.cwd must be a string")
        cwd = (
            Path(_expand(raw_cwd, workspace, allow_missing=disabled)).expanduser()
            if raw_cwd
            else None
        )
        if cwd is not None and not cwd.is_absolute():
            if base_dir is not None and ".." in cwd.parts:
                raise ValueError(f"{location}.cwd cannot escape the environment directory")
            cwd = ((base_dir or path.parent) / cwd).resolve()
            if base_dir is not None and not cwd.is_relative_to(base_dir):
                raise ValueError(f"{location}.cwd cannot escape the environment directory")
        if base_dir is not None and cwd is None:
            cwd = base_dir
        if base_dir is not None and expanded_command and "/" in expanded_command:
            command_path = Path(expanded_command).expanduser()
            if not command_path.is_absolute():
                resolved_command = (base_dir / command_path).resolve()
                if ".." in command_path.parts or not resolved_command.is_relative_to(base_dir):
                    raise ValueError(f"{location}.command cannot escape the environment directory")
                expanded_command = str(resolved_command)

        configured_transport = raw.get("transport", raw.get("type"))
        if configured_transport is not None and not isinstance(configured_transport, str):
            raise ValueError(f"{location}.transport must be a string")
        transport = configured_transport or (
            "stdio" if expanded_command is not None else "streamable-http"
        )
        aliases = {"http": "streamable-http", "streamable_http": "streamable-http"}
        transport = aliases.get(transport, transport)
        allowed_transports = {"stdio"} if command is not None else {"streamable-http", "sse"}
        if transport not in allowed_transports:
            raise ValueError(
                f"{location}.transport must be one of {', '.join(sorted(allowed_transports))}"
            )
        timeout = raw.get("timeout", 30)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"{location}.timeout must be a positive number")
        mode = raw.get("mode", "auto")
        if not isinstance(mode, str) or mode not in {"auto", "legacy"}:
            raise ValueError(f"{location}.mode must be auto or legacy")
        servers[name] = McpServerConfig(
            name=name,
            command=expanded_command,
            args=tuple(_expand(item, workspace, allow_missing=disabled) for item in args),
            env=_string_map(raw.get("env"), f"{location}.env", workspace, allow_missing=disabled),
            cwd=cwd,
            url=expanded_url,
            headers=_string_map(
                raw.get("headers"), f"{location}.headers", workspace, allow_missing=disabled
            ),
            transport=transport,
            disabled=disabled,
            timeout=float(timeout),
            required=required,
            mode=mode,
        )
    return servers


class _AsyncRuntime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="mcp-runtime", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _shutdown(self):
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        # Closed connections finish their own teardown, which stops a stdio server
        # still starting only after the SDK's grace period; cancelling again would cut
        # that short and orphan the process. Cancel only what outlives the teardown.
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=TEARDOWN_TIMEOUT)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=2)

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        with suppress(Exception):
            self.submit(self._shutdown()).result(timeout=TEARDOWN_TIMEOUT + 3)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


class _Connection:
    def __init__(self, runtime: _AsyncRuntime, config: McpServerConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.tools: list[dict[str, Any]] = []
        self.instructions: str | None = None
        self.state = "not-connected"
        self.error: str | None = None
        self._queue: asyncio.Queue[_Request | None] | None = None
        self._tools_revision = 0
        self._catalog_revision = -1
        self._refresh_pending = False
        self._catalog_error: str | None = None
        self._ready: concurrent.futures.Future[None] | None = None
        self._lifetime: concurrent.futures.Future[None] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def begin(self) -> concurrent.futures.Future[None] | None:
        """Start connecting without waiting; ``None`` when already connected."""
        with self._lock:
            if self._closed:
                raise RuntimeError(f"MCP connection {self.config.name!r} is closed")
            if self.state == "connected" and self._lifetime and not self._lifetime.done():
                return None
            if self.state != "connecting" or self._lifetime is None or self._lifetime.done():
                self.state = "connecting"
                self.error = None
                self._ready = concurrent.futures.Future()
                self._lifetime = self.runtime.submit(self._run(self._ready))
            return self._ready

    def start(self, timeout: float, *, cancelled=None) -> None:
        deadline = time.monotonic() + timeout
        ready = self.begin()
        if ready is not None:
            _wait_future(ready, deadline, cancelled)

    async def _open_client(self, stack: AsyncExitStack) -> Client:
        if self.config.command is not None:
            target: Any = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                env=self.config.env or None,
                cwd=self.config.cwd,
            )
        elif self.config.transport == "sse":
            target = sse_client(
                self.config.url or "",
                headers=self.config.headers or None,
                timeout=self.config.timeout,
                sse_read_timeout=self.config.timeout,
            )
        elif self.config.headers:
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(headers=self.config.headers, timeout=self.config.timeout)
            )
            target = streamable_http_client(self.config.url or "", http_client=http_client)
        else:
            target = self.config.url or ""
        client = await stack.enter_async_context(
            Client(
                target,
                read_timeout_seconds=self.config.timeout,
                mode=self.config.mode,
                message_handler=self._handle_message,
            )
        )
        if (
            client.protocol_version in MODERN_PROTOCOL_VERSIONS
            and client.server_capabilities.tools is not None
        ):
            subscribed = asyncio.get_running_loop().create_future()
            watcher = asyncio.create_task(self._watch_tools(client, subscribed))
            stack.push_async_callback(self._stop_watcher, watcher)
            # Subscribe before the initial listing so no change is lost in between.
            await subscribed
        return client

    @staticmethod
    async def _stop_watcher(watcher: asyncio.Task) -> None:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher

    async def _watch_tools(self, client: Client, subscribed) -> None:
        while True:
            try:
                async with client.listen(tools_list_changed=True) as subscription:
                    if not subscribed.done():
                        subscribed.set_result(None)
                    else:
                        # A reopened stream has no replay; re-list to recover missed changes.
                        self._tools_changed()
                    async for _ in subscription:
                        # Notifications also reach _handle_message. Drain the stream
                        # without scheduling a second refresh for the same event.
                        pass
                self._catalog_error = "MCP tool subscription closed"
            except Exception as exc:
                if not subscribed.done():
                    subscribed.set_exception(exc)
                    return
                self._catalog_error = str(exc) or type(exc).__name__
            self._tools_changed()
            await asyncio.sleep(1)

    async def _handle_message(self, message: IncomingMessage) -> None:
        if isinstance(message, mcp_types.ToolListChangedNotification):
            self._tools_changed()

    def _tools_changed(self) -> None:
        if not self._closed:
            self._tools_revision += 1
            self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        if self._queue is not None and not self._refresh_pending and not self._closed:
            self._refresh_pending = True
            self._queue.put_nowait(None)

    async def _refresh_tools(self, client: Client) -> None:
        revision = self._tools_revision
        try:
            tools = await self._list_tools(client)
        except Exception as exc:
            self._catalog_error = str(exc) or type(exc).__name__
            raise
        # Publish all pages together; a notification during the fetch stays dirty.
        self.tools = tools
        self._catalog_revision = revision
        self._catalog_error = None

    async def _list_tools(self, client: Client) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            result = await client.list_tools(cursor=cursor, cache_mode="refresh")
            for tool in result.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "title": tool.title,
                        "description": tool.description or "",
                        "inputSchema": tool.input_schema,
                        "outputSchema": tool.output_schema,
                        "annotations": tool.annotations.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                        if tool.annotations
                        else None,
                    }
                )
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def _operation(self, client, request):
        if request.operation == "call":
            return await client.call_tool(request.payload["name"], request.payload.get("arguments"))
        if request.operation == "refresh":
            await self._refresh_tools(client)
            return self.tools
        if request.operation == "catalog":
            while self._catalog_revision != self._tools_revision:
                await self._refresh_tools(client)
            return self.tools
        raise ValueError(f"Unknown MCP operation {request.operation!r}")

    async def _run(self, ready) -> None:
        close_request: _Request | None = None
        active_request = None
        queue = None
        try:
            async with AsyncExitStack() as stack:
                queue = self._queue = asyncio.Queue()
                self._refresh_pending = False
                async with asyncio.timeout(self.config.timeout):
                    client = await self._open_client(stack)
                    await self._refresh_tools(client)
                self.instructions = client.instructions
                with self._lock:
                    self.state = "connected"
                if not ready.done():
                    ready.set_result(None)

                while True:
                    request = await queue.get()
                    if request is None:
                        self._refresh_pending = False
                        try:
                            if self._catalog_revision != self._tools_revision:
                                async with asyncio.timeout(self.config.timeout):
                                    await self._refresh_tools(client)
                        except Exception as exc:
                            # Keep the last complete catalog. Discovery retries the
                            # dirty listing; a failed background refresh cannot end a call.
                            self._catalog_error = str(exc) or type(exc).__name__
                        continue
                    if request.operation == "close":
                        close_request = request
                        break
                    remaining = request.deadline - time.monotonic()
                    if (
                        remaining <= 0
                        or request.abandoned.is_set()
                        or (request.cancelled is not None and request.cancelled.is_set())
                    ):
                        if not request.future.done():
                            request.future.set_exception(
                                TimeoutError("MCP request expired before execution")
                            )
                        continue
                    if not request.future.set_running_or_notify_cancel():
                        continue
                    active_request = request
                    request.task = asyncio.create_task(self._operation(client, request))
                    try:
                        async with asyncio.timeout(remaining):
                            result = await request.task
                        request.future.set_result(result)
                    except asyncio.CancelledError:
                        if not request.abandoned.is_set():
                            raise
                        request.future.set_exception(self._interrupted(request))
                    except TimeoutError:
                        request.future.set_exception(self._interrupted(request))
                    except Exception as exc:
                        request.future.set_exception(exc)
                    finally:
                        request.task = None
                        if request.future.done():
                            active_request = None
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            if close_request and not close_request.future.done():
                close_request.future.set_exception(exc)
            with self._lock:
                self.state = "closed" if self._closed else "failed"
                self.error = None if self._closed else str(exc)
            raise
        else:
            with self._lock:
                self.state = "closed"
            if close_request and not close_request.future.done():
                close_request.future.set_result(None)
        finally:
            if active_request is not None and not active_request.future.done():
                active_request.future.set_exception(self._interrupted(active_request))
            if queue is not None:
                while not queue.empty():
                    request = queue.get_nowait()
                    if request is not None and not request.future.done():
                        request.future.set_exception(
                            RuntimeError("MCP connection closed before execution")
                        )

    def _interrupted(self, request):
        if request.operation == "call":
            return McpExecutionUnknownError(
                f"MCP call on {self.config.name!r} was interrupted after starting. "
                "It may have taken effect; inspect external state before retrying."
            )
        return TimeoutError(f"MCP operation on {self.config.name!r} timed out")

    def request(
        self, operation: str, payload: dict[str, Any], timeout: float, *, context=None
    ) -> Any:
        deadline = time.monotonic() + timeout
        cancelled = context.cancelled if context is not None else None
        if context is not None:
            context.remaining()
            deadline = min(deadline, context.deadline)
        self.start(deadline - time.monotonic(), cancelled=cancelled)
        request = _Request(operation, payload, deadline, cancelled=cancelled)
        with self._lock:
            if self._closed or self._queue is None:
                raise RuntimeError(f"MCP server {self.config.name!r} is not available")
            self.runtime.loop.call_soon_threadsafe(self._queue.put_nowait, request)
        try:
            return _wait_future(request.future, deadline, cancelled)
        except BaseException as exc:
            if request.future.done():
                raise
            request.abandoned.set()
            if request.future.cancel():
                raise
            if isinstance(exc, TimeoutError) and request.future.done():
                return request.future.result()
            self.runtime.loop.call_soon_threadsafe(self._cancel_request, request)
            if isinstance(exc, TimeoutError):
                raise self._interrupted(request) from exc
            raise

    @staticmethod
    def _cancel_request(request):
        if request.task is not None:
            request.task.cancel()

    def close(self, timeout: float = 5) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(RuntimeError("MCP connection closed during startup"))
            if self.state not in {"connected", "connecting"}:
                self.state = "closed"
                return
            starting = self.state == "connecting"
        request = _Request("close", {}, time.monotonic() + timeout)
        try:
            # A starting connection reads no queue yet; cancel it rather than wait out
            # a server that may never be used, such as one still starting in the background.
            if self._queue is None or starting:
                raise TimeoutError("MCP connection is still starting")
            self.runtime.loop.call_soon_threadsafe(self._queue.put_nowait, request)
            _wait_future(request.future, request.deadline)
        except Exception:
            if self._lifetime:
                self._lifetime.cancel()
                with suppress(Exception):
                    self._lifetime.result(timeout=2)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "transport": self.config.transport,
            "status": "disabled" if self.config.disabled else self.state,
            "tool_count": len(self.tools)
            if self.state == "connected" and not self.config.disabled
            else 0,
            "catalog_tool_count": len(self.tools),
            "catalog_stale": self._catalog_revision != self._tools_revision,
            "catalog_error": self._catalog_error,
            "available": self.state == "connected" and not self.config.disabled,
            "error": self.error,
        }


def _resource_revision(server: McpServerConfig) -> tuple:
    if server.transport != "stdio" or server.disabled:
        return ()
    files = set()
    for value in (server.command, *server.args):
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (server.cwd or Path.cwd()) / path
        if path.is_file():
            files.add(path.resolve())
            if path.suffix.lower() in {".py", ".js", ".mjs", ".cjs", ".sh"}:
                files.update(local_code_files(path.parent))
    revision = []
    for path in sorted(files):
        info = path.stat()
        revision.append(
            (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        )
    return tuple(revision)


class _RegistryFile:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ds-agent-skills-")
        self.owners = 1
        self.lock = threading.Lock()

    def retain(self):
        with self.lock:
            self.owners += 1

    def release(self):
        with self.lock:
            self.owners -= 1
            if self.owners == 0:
                self.directory.cleanup()


class McpManager:
    def __init__(
        self,
        config_file: Path,
        workspace: Path,
        *,
        servers: dict[str, McpServerConfig] | None = None,
        skill_registry: dict[str, Any] | None = None,
        runtime_env: dict[str, str] | None = None,
        previous: "McpManager | None" = None,
    ) -> None:
        self.config_file = config_file.resolve()
        self.workspace = workspace.resolve()
        self.servers = (
            load_mcp_config(self.config_file, self.workspace) if servers is None else servers
        )
        if runtime_env:
            self.servers = {
                name: replace(
                    server,
                    env={
                        **server.env,
                        **{key: value for key, value in runtime_env.items() if key in server.env},
                    },
                )
                if server.transport == "stdio"
                else server
                for name, server in self.servers.items()
            }
        self.configured_servers = dict(self.servers)
        self.resource_revisions = {
            name: _resource_revision(server) for name, server in self.configured_servers.items()
        }
        self._skill_registry = skill_registry
        self._skill_registry_directory = None
        self._skill_registry_directories = {}
        if skill_registry is not None and any(
            server.transport == "stdio" and "DS_AGENT_SKILL_REGISTRY" in server.env
            for server in self.servers.values()
        ):
            serialized = json.dumps(skill_registry, ensure_ascii=False)
            resource = _RegistryFile()
            self._skill_registry_directory = resource.directory
            registry_path = Path(self._skill_registry_directory.name) / "registry.json"
            self._skill_registry_directories[str(registry_path)] = resource
            try:
                registry_path.write_text(serialized, encoding="utf-8")
            except BaseException:
                resource.release()
                raise
            self.servers = {
                name: replace(
                    server, env={**server.env, "DS_AGENT_SKILL_REGISTRY": str(registry_path)}
                )
                if server.transport == "stdio" and "DS_AGENT_SKILL_REGISTRY" in server.env
                else server
                for name, server in self.servers.items()
            }
        self._runtime: _AsyncRuntime | None = None
        self._connections: dict[str, _Connection] = {}
        # Startups begun by start_all whose outcome has not been reported yet.
        self._background: dict[str, concurrent.futures.Future[None]] = {}
        self._closed = False
        self._lock = threading.Lock()
        self._previous = None
        self._borrowed: set[str] = set()
        self._borrowed_runtime = False
        self._registry_backups: dict[str, str] = {}
        if isinstance(previous, McpManager):
            with previous._lock:
                if not previous._closed:
                    self._previous = previous
                    self._runtime = previous._runtime
                    self._borrowed_runtime = self._runtime is not None
                    for name, config in self.configured_servers.items():
                        old_config = previous.configured_servers.get(name)
                        connection = previous._connections.get(name)
                        if (
                            old_config is not None
                            and replace(config, required=False)
                            == replace(old_config, required=False)
                            and self.resource_revisions[name]
                            == previous.resource_revisions.get(name)
                            and connection is not None
                            and connection.state == "connected"
                        ):
                            self._connections[name] = connection
                            self._borrowed.add(name)
                            path = previous.servers[name].env.get("DS_AGENT_SKILL_REGISTRY")
                            if path in previous._skill_registry_directories:
                                if path not in self._skill_registry_directories:
                                    resource = previous._skill_registry_directories[path]
                                    resource.retain()
                                    self._skill_registry_directories[path] = resource
                                self.servers[name] = replace(
                                    self.servers[name],
                                    env={**self.servers[name].env, "DS_AGENT_SKILL_REGISTRY": path},
                                )
        self._reused_servers = tuple(sorted(self._borrowed))
        referenced = {server.env.get("DS_AGENT_SKILL_REGISTRY") for server in self.servers.values()}
        for path in list(self._skill_registry_directories):
            if path not in referenced:
                resource = self._skill_registry_directories.pop(path)
                resource.release()
                if resource.directory is self._skill_registry_directory:
                    self._skill_registry_directory = None

    @property
    def reused_servers(self) -> tuple[str, ...]:
        return self._reused_servers

    def commit_reuse(self) -> None:
        previous = self._previous
        if previous is None:
            self._registry_backups.clear()
            return
        with previous._lock:
            for name in self._borrowed:
                if previous._connections.get(name) is self._connections.get(name):
                    previous._connections.pop(name)
            if self._borrowed_runtime:
                previous._runtime = None
        self._borrowed.clear()
        self._borrowed_runtime = False
        self._previous = None
        self._registry_backups.clear()

    def stage_skill_registry(self) -> None:
        if self._skill_registry is None:
            return
        serialized = json.dumps(self._skill_registry, ensure_ascii=False)
        for path in self._skill_registry_directories:
            target = Path(path)
            saved = target.read_text(encoding="utf-8")
            if saved != serialized:
                self._registry_backups[path] = saved
                atomic_write_text(target, serialized)

    def update_skill_registry(self, registry: dict[str, Any] | None) -> None:
        if registry is None:
            return
        serialized = json.dumps(registry, ensure_ascii=False)
        changes = {}
        for path in self._skill_registry_directories:
            target = Path(path)
            saved = target.read_text(encoding="utf-8")
            if saved != serialized:
                changes[target] = saved
        updated = []
        try:
            for target in changes:
                updated.append(target)
                atomic_write_text(target, serialized)
        except BaseException as exc:
            failures = []
            for target in updated:
                try:
                    atomic_write_text(target, changes[target])
                except Exception as restore_error:
                    failures.append(str(restore_error))
            if failures:
                raise RuntimeError(
                    "技能注册表更新失败，且回滚失败：" + "; ".join(failures)
                ) from exc
            raise
        self._skill_registry = registry

    @property
    def server_count(self) -> int:
        return len([server for server in self.servers.values() if not server.disabled])

    def _connection(self, name: str) -> _Connection:
        with self._lock:
            return self._get_connection(name)

    def _get_connection(self, name: str) -> _Connection:
        if self._closed:
            raise RuntimeError("MCP manager is closed")
        config = self.servers.get(name)
        if config is None:
            raise ValueError(f"Unknown MCP server {name!r}")
        if config.disabled:
            raise ValueError(f"MCP server {name!r} is disabled")
        if self._runtime is None:
            self._runtime = _AsyncRuntime()
        connection = self._connections.get(name)
        if connection is None or connection.state in {"closed", "failed"}:
            connection = _Connection(self._runtime, config)
            self._connections[name] = connection
        return connection

    def _remaining(self, context: ExecutionContext, server: McpServerConfig) -> float:
        return min(context.remaining(), server.timeout)

    def status(self) -> ToolResult:
        entries = []
        for name, config in self.servers.items():
            connection = self._connections.get(name)
            entry = (
                connection.snapshot()
                if connection
                else {
                    "name": name,
                    "transport": config.transport,
                    "status": "disabled" if config.disabled else "not-connected",
                    "tool_count": 0,
                    "catalog_tool_count": 0,
                    "available": False,
                    "error": None,
                }
            )
            entry["required"] = config.required
            entries.append(entry)
        return ToolResult(data={"config_file": str(self.config_file), "servers": entries})

    def connect(self, server: str, context: ExecutionContext) -> ToolResult:
        config = self.servers.get(server)
        if config is None:
            raise ValueError(f"Unknown MCP server {server!r}")
        connection = self._connection(server)
        connection.start(self._remaining(context, config), cancelled=context.cancelled)
        return ToolResult(
            data={
                "server": server,
                "status": connection.state,
                "tool_count": len(connection.tools),
            }
        )

    def connect_all(
        self, timeout: float = 10, on_connect: Callable[[str], None] | None = None
    ) -> ToolResult:
        """Connect every enabled server concurrently, waiting up to ``timeout`` for them."""
        started = time.monotonic()
        selected = self._selected_servers(None)
        if on_connect and selected:
            on_connect(", ".join(config.name for config in selected))
        pending: list[tuple[McpServerConfig, Any]] = []
        for config in selected:
            try:
                pending.append((config, self._connection(config.name).begin()))
            except Exception as exc:
                pending.append((config, exc))
        failures = []
        for config, ready in pending:
            try:
                if isinstance(ready, Exception):
                    raise ready
                if ready is not None:
                    _wait_future(ready, started + min(timeout, config.timeout))
            except Exception as exc:
                failures.append({"server": config.name, "error": str(exc) or type(exc).__name__})
        servers = self.status().data["servers"]
        connected = sum(server["status"] == "connected" for server in servers)
        return ToolResult(
            status="partial" if failures and connected else "error" if failures else "ok",
            data={"servers": servers, "failures": failures},
        )

    def start_all(self) -> None:
        """Begin connecting every enabled server in the background without waiting.

        Discovery and calls wait for a server that is still connecting, so nothing
        blocks on servers a task never uses; ``background_failures`` reports the outcome.
        """
        for config in self._selected_servers(None):
            ready = self._connection(config.name).begin()
            if ready is not None:
                self._background[config.name] = ready

    def background_failures(self) -> list[dict[str, str]]:
        """Failures of ``start_all`` connections that ended since the last call, once each."""
        failures = []
        for name, ready in list(self._background.items()):
            if not ready.done():
                continue
            del self._background[name]
            error = ready.exception()
            connection = self._connections.get(name)
            # A use since then may already have reconnected; that failure is moot.
            if error is not None and not (connection and connection.state == "connected"):
                failures.append({"server": name, "error": str(error) or type(error).__name__})
        return failures

    def _selected_servers(self, server: str | None, permit=None) -> list[McpServerConfig]:
        if server is not None:
            config = self.servers.get(server)
            if config is None:
                raise ValueError(f"Unknown MCP server {server!r}")
            if permit is not None and not permit.permits_server(server):
                raise PermissionError(f"MCP server {server!r} is not permitted for this agent")
            return [config]
        return [
            config
            for config in self.servers.values()
            if not config.disabled and (permit is None or permit.permits_server(config.name))
        ]

    def _catalog(
        self, context: ExecutionContext, server: str | None = None, permit=None
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, str]]]:
        catalog: list[tuple[str, dict[str, Any]]] = []
        failures = []
        for config in self._selected_servers(server, permit):
            if config.disabled:
                failures.append({"server": config.name, "error": "server is disabled"})
                continue
            try:
                connection = self._connection(config.name)
                connection.start(self._remaining(context, config), cancelled=context.cancelled)
                tools = connection.request(
                    "catalog", {}, self._remaining(context, config), context=context
                )
                catalog.extend((config.name, tool) for tool in tools)
            except Exception as exc:
                failures.append({"server": config.name, "error": str(exc)})
        return catalog, failures

    @staticmethod
    def _score(tool: dict[str, Any], query: str) -> int:
        normalized = query.casefold().strip()
        if normalized == "*":
            return 1
        name = tool["name"].casefold()
        text = f"{name} {tool.get('title') or ''} {tool.get('description') or ''}".casefold()
        terms = SEARCH_TERM.findall(normalized)
        # Single letters match nearly any text; keep them only when nothing longer remains.
        terms = [term for term in terms if len(term) > 1] or terms
        # Any matching term qualifies a tool, so a query need not guess every word of a
        # description; tools matching more terms, all of them, or the name rank first.
        hits = [term for term in terms if term in text]
        if not hits:
            return 0
        return (
            (20 if normalized in name else 0)
            + (10 if len(hits) == len(terms) else 0)
            + sum(5 if term in name else 1 for term in hits)
        )

    def search(
        self,
        query: str,
        server: str | None,
        limit: int,
        context: ExecutionContext,
        *,
        permit=None,
    ) -> ToolResult:
        if not query.strip():
            raise ValueError("query is required for MCP search")
        catalog, failures = self._catalog(context, server, permit)
        matches = []
        for server_name, tool in catalog:
            if permit is not None and not permit.permits(server_name, tool["name"]):
                continue
            score = self._score(tool, query)
            if score:
                matches.append(
                    {
                        "path": f"{server_name}/{tool['name']}",
                        "description": tool.get("description", ""),
                        "score": score,
                    }
                )
        matches.sort(key=lambda item: (-item["score"], item["path"]))
        result = ToolResult(data={"query": query, "matches": matches[:limit], "failures": failures})
        if not matches:
            available = sorted(
                f"{server_name}/{tool['name']}"
                for server_name, tool in catalog
                if permit is None or permit.permits(server_name, tool["name"])
            )
            if available:
                result.data["available_tools"] = available[:AVAILABLE_TOOL_LIMIT]
                result.data["available_tool_count"] = len(available)
                result.data["guidance"] = (
                    "No tool matched the query. Pick one from available_tools and describe it, "
                    "or search '*' to list tools with descriptions."
                )
        if failures and catalog:
            result.status = "partial"
        elif failures and not catalog:
            return ToolResult.failure(
                "mcp_unavailable",
                "No selected MCP server could be reached",
                data=result.data,
            )
        return result

    def _find_tool(
        self, name: str, server: str | None, context: ExecutionContext, permit=None
    ) -> tuple[str, dict[str, Any]]:
        if "/" in name:
            qualified_server, name = name.split("/", 1)
            if server is not None and server != qualified_server:
                raise ValueError("server conflicts with the qualified tool path")
            server = qualified_server
        catalog, failures = self._catalog(context, server, permit)
        matches = [(server_name, tool) for server_name, tool in catalog if tool["name"] == name]
        if permit is not None and matches:
            allowed = [match for match in matches if permit.permits(match[0], name)]
            if not allowed:
                paths = ", ".join(f"{server_name}/{name}" for server_name, _ in matches)
                raise PermissionError(f"MCP tool {paths} is not permitted for this agent")
            matches = allowed
        if not matches:
            detail = f"; failures: {failures}" if failures else ""
            raise ValueError(f"Unknown MCP tool {name!r}{detail}")
        if len(matches) > 1:
            paths = ", ".join(f"{server_name}/{tool['name']}" for server_name, tool in matches)
            raise ValueError(f"Ambiguous MCP tool {name!r}; use one of: {paths}")
        return matches[0]

    def describe(
        self, tool: str, server: str | None, context: ExecutionContext, *, permit=None
    ) -> ToolResult:
        server_name, metadata = self._find_tool(tool, server, context, permit)
        return ToolResult(data={"path": f"{server_name}/{metadata['name']}", **metadata})

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        server: str | None,
        context: ExecutionContext,
        *,
        permit=None,
    ) -> ToolResult:
        server_name, metadata = self._find_tool(tool, server, context, permit)
        schema = metadata.get("inputSchema")
        if schema is not None:
            try:
                validator = _argument_validator(json.dumps(schema, sort_keys=True))
                problems = list(islice(validator.iter_errors(arguments), 3))
            except (SchemaError, Unresolvable) as exc:
                return ToolResult.failure(
                    "invalid_tool_schema",
                    f"Cannot validate MCP tool {server_name}/{metadata['name']}: {exc}",
                    data={"path": f"{server_name}/{metadata['name']}", "input_schema": schema},
                )
            if problems:
                messages = []
                for problem in problems:
                    location = "arguments" + "".join(
                        f"[{part}]" if isinstance(part, int) else f".{part}"
                        for part in problem.absolute_path
                    )
                    messages.append(f"{location}: {problem.message}")
                return ToolResult.failure(
                    "invalid_arguments",
                    "; ".join(messages),
                    data={
                        "path": f"{server_name}/{metadata['name']}",
                        "input_schema": schema,
                        "guidance": "Correct the arguments to match the advertised schema before calling again. Confirmation fields require authorization for the requested operation; do not fill them merely to silence an error.",
                    },
                )
        config = self.servers[server_name]
        connection = self._connection(server_name)
        result = connection.request(
            "call",
            {"name": metadata["name"], "arguments": arguments},
            self._remaining(context, config),
            context=context,
        )
        serialized = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        data = {"path": f"{server_name}/{metadata['name']}", "result": serialized}
        if result.is_error:
            structured = result.structured_content
            detail = structured.get("error") if isinstance(structured, dict) else None
            code, message = "mcp_tool_error", None
            if isinstance(detail, dict):
                if isinstance(detail.get("code"), str) and detail["code"].strip():
                    code = detail["code"]
                if isinstance(detail.get("message"), str) and detail["message"].strip():
                    message = detail["message"]
            elif isinstance(detail, str) and detail.strip():
                message = detail
            if message is None:
                message = (
                    "\n".join(
                        item["text"]
                        for item in serialized.get("content", [])
                        if item.get("type") == "text" and isinstance(item.get("text"), str)
                    ).strip()
                    or "MCP tool returned an error"
                )
            failure = ToolResult.failure(code, message, data=data)
            if isinstance(detail, dict) and isinstance(detail.get("retryable"), bool):
                failure.error["retryable"] = detail["retryable"]
            return failure
        return ToolResult(data=data)

    def execute(
        self,
        action: str,
        server: str | None = None,
        query: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        limit: int = 12,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        if action == "status":
            return self.status()
        if action == "connect":
            if not server:
                raise ValueError("server is required for MCP connect")
            return self.connect(server, _context)
        if action == "search":
            return self.search(query or "", server, limit, _context)
        if action == "describe":
            if not tool:
                raise ValueError("tool is required for MCP describe")
            return self.describe(tool, server, _context)
        if action == "call":
            if not tool:
                raise ValueError("tool is required for MCP call")
            try:
                return self.call(tool, arguments or {}, server, _context)
            except McpExecutionUnknownError as exc:
                return ToolResult.failure("execution_unknown", str(exc), status="unknown")
        raise ValueError(f"Unknown MCP action {action!r}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = [
                connection
                for name, connection in self._connections.items()
                if name not in self._borrowed
            ]
            runtime = None if self._borrowed_runtime else self._runtime
        failures = []
        for connection in connections:
            try:
                connection.close()
            except Exception as exc:
                failures.append(exc)
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                failures.append(exc)
        for path, content in self._registry_backups.items():
            try:
                atomic_write_text(Path(path), content)
            except Exception as exc:
                failures.append(exc)
        self._registry_backups.clear()
        self._skill_registry_directory = None
        for resource in self._skill_registry_directories.values():
            try:
                resource.release()
            except Exception as exc:
                failures.append(exc)
        self._skill_registry_directories.clear()
        if failures:
            raise RuntimeError(f"MCP cleanup failed: {failures[0]}") from failures[0]
