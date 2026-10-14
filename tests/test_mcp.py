import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_client import (
    McpExecutionUnknownError,
    McpManager,
    McpServerConfig,
    _AsyncRuntime,
    _Connection,
    load_mcp_config,
)
from tool_runtime import ExecutionContext
from tools import CodingTools

ROOT = Path(__file__).resolve().parents[1]


class McpConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_loads_stdio_and_http_servers_with_variables(self):
        config = self.root / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "local": {
                            "command": "${PYTHON}",
                            "args": ["server.py", "${workspace}"],
                            "cwd": "servers",
                        },
                        "remote": {
                            "url": "${MCP_URL}",
                            "headers": {"Authorization": "Bearer ${TOKEN}"},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        workspace = self.root / "workspace"
        with patch.dict(
            os.environ,
            {
                "PYTHON": sys.executable,
                "TOKEN": "secret",
                "MCP_URL": "https://example.test/mcp",
            },
        ):
            servers = load_mcp_config(config, workspace)

        self.assertEqual(servers["local"].command, sys.executable)
        self.assertEqual(servers["local"].args[-1], str(workspace))
        self.assertEqual(servers["local"].cwd, (self.root / "servers").resolve())
        self.assertEqual(servers["local"].mode, "auto")
        self.assertEqual(servers["remote"].transport, "streamable-http")
        self.assertEqual(servers["remote"].url, "https://example.test/mcp")
        self.assertEqual(servers["remote"].headers["Authorization"], "Bearer secret")

    def test_rejects_ambiguous_or_unknown_server_configuration(self):
        for definition, message in [
            ({"command": "x", "url": "https://example.test"}, "exactly one"),
            ({"command": "x", "unexpected": True}, "Unknown fields"),
            ({"url": "file:///tmp/socket"}, "HTTP"),
            ({"command": "x", "transport": []}, "transport must be a string"),
            ({"command": "x", "mode": "invalid"}, "mode must be auto or legacy"),
            ({"command": "x", "mode": True}, "mode must be auto or legacy"),
            ({"command": "x", "mode": []}, "mode must be auto or legacy"),
        ]:
            with self.subTest(definition=definition):
                path = self.root / "mcp.json"
                path.write_text(json.dumps({"mcpServers": {"test": definition}}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_mcp_config(path, self.root)

    def test_disabled_server_does_not_require_environment_secrets(self):
        path = self.root / "mcp.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "disabled": {
                            "url": "${MISSING_MCP_URL}",
                            "headers": {"Authorization": "Bearer ${MISSING_MCP_TOKEN}"},
                            "disabled": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            servers = load_mcp_config(path, self.root)

        self.assertTrue(servers["disabled"].disabled)
        self.assertEqual(servers["disabled"].url, "${MISSING_MCP_URL}")


class McpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.config = self.workspace / "mcp.json"
        self.config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "test": {
                            "command": sys.executable,
                            "args": [str(ROOT / "tests" / "fixtures_mcp_server.py")],
                            "timeout": 10,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        self.manager = McpManager(self.config, self.workspace)
        self.addCleanup(self.manager.close)

    @staticmethod
    def context() -> ExecutionContext:
        return ExecutionContext(time.monotonic() + 15)

    @staticmethod
    def stop_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def start_http_server(self) -> tuple[subprocess.Popen, int]:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tests" / "fixtures_mcp_server.py")],
            env={
                **os.environ,
                "MCP_TEST_TRANSPORT": "streamable-http",
                "MCP_TEST_PORT": str(port),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop_process, process)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail("Streamable HTTP MCP test server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return process, port
            except OSError:
                time.sleep(0.05)
        self.fail("Streamable HTTP MCP test server did not start")

    def test_lazy_search_describe_call_and_shutdown(self):
        initial = self.manager.status()
        self.assertEqual(initial.data["servers"][0]["status"], "not-connected")

        searched = self.manager.execute(action="search", query="echo", _context=self.context())
        self.assertEqual(searched.status, "ok")
        self.assertEqual(searched.data["matches"][0]["path"], "test/echo")
        self.assertEqual(self.manager.status().data["servers"][0]["status"], "connected")

        described = self.manager.execute(
            action="describe", tool="test/echo", _context=self.context()
        )
        self.assertEqual(described.data["inputSchema"]["required"], ["text"])

        called = self.manager.execute(
            action="call",
            tool="test/echo",
            arguments={"text": "hello"},
            _context=self.context(),
        )
        self.assertEqual(called.status, "ok")
        self.assertEqual(called.data["result"]["structuredContent"], {"echo": "hello"})

        self.manager.close()
        self.assertEqual(self.manager.status().data["servers"][0]["status"], "closed")

    def test_legacy_handshake_connects_and_calls_tool(self):
        self.manager.close()
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["mcpServers"]["test"]["mode"] = "legacy"
        self.config.write_text(json.dumps(payload), encoding="utf-8")
        servers = load_mcp_config(self.config, self.workspace)
        self.assertEqual(servers["test"].mode, "legacy")
        self.manager = McpManager(self.config, self.workspace)
        self.addCleanup(self.manager.close)

        called = self.manager.execute(
            action="call",
            tool="test/echo",
            arguments={"text": "legacy handshake"},
            _context=self.context(),
        )

        self.assertEqual(called.status, "ok")
        self.assertEqual(called.data["result"]["structuredContent"], {"echo": "legacy handshake"})
        self.assertEqual(self.manager.status().data["servers"][0]["status"], "connected")

    def test_registry_exposes_only_the_proxy_and_accepts_arbitrary_arguments(self):
        tools = CodingTools(self.workspace)
        tools.attach_mcp(self.manager)
        definitions = {item["function"]["name"] for item in tools.definitions}
        self.assertIn("mcp", definitions)
        self.assertNotIn("echo", definitions)

        result = json.loads(
            tools.execute(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "mcp",
                        "arguments": json.dumps(
                            {
                                "action": "call",
                                "tool": "test/add",
                                "arguments": {"left": 2, "right": 3},
                            }
                        ),
                    },
                },
                deadline=time.monotonic() + 15,
            )
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["result"]["structuredContent"], {"result": 5})

    def test_startup_connects_without_needing_a_search_and_reuses_connection(self):
        result = self.manager.connect_all()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["failures"], [])
        self.assertEqual(result.data["servers"][0]["status"], "connected")
        connection = self.manager._connections["test"]
        self.assertGreater(result.data["servers"][0]["tool_count"], 0)
        self.assertEqual(self.manager.connect_all().status, "ok")
        self.assertIs(self.manager._connections["test"], connection)

    def test_startup_keeps_healthy_service_when_another_fails(self):
        config = json.loads(self.config.read_text())
        config["mcpServers"]["broken"] = {"command": "/nonexistent-mcp-binary", "timeout": 1}
        config["mcpServers"]["disabled"] = {"command": "/unused-binary", "disabled": True}
        self.config.write_text(json.dumps(config))
        manager = McpManager(self.config, self.workspace)
        self.addCleanup(manager.close)
        result = manager.connect_all(timeout=2)
        self.assertEqual(result.status, "partial")
        self.assertEqual([item["server"] for item in result.data["failures"]], ["broken"])
        states = {item["name"]: item["status"] for item in result.data["servers"]}
        self.assertEqual(states, {"test": "connected", "broken": "failed", "disabled": "disabled"})

    def test_streamable_http_transport_with_headers(self):
        _, port = self.start_http_server()
        config = self.workspace / "http-mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "remote": {
                            "url": f"http://127.0.0.1:{port}/mcp",
                            "headers": {"X-Agent-Test": "enabled"},
                            "timeout": 10,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        manager = McpManager(config, self.workspace)
        self.addCleanup(manager.close)

        searched = manager.execute(action="search", query="add", _context=self.context())
        called = manager.execute(
            action="call",
            tool="remote/add",
            arguments={"left": 20, "right": 22},
            _context=self.context(),
        )

        self.assertEqual(searched.data["matches"][0]["path"], "remote/add")
        self.assertEqual(called.status, "ok")
        self.assertEqual(called.data["result"]["structuredContent"], {"result": 42})


class ConnectionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.startup = threading.Event()
        self.opened = threading.Event()
        self.startup.set()
        self.open_count = 0
        owner = self

        class Client:
            instructions = None

            async def call_tool(self, name, arguments):
                owner.calls.append(name)
                if name == "slow":
                    owner.started.set()
                    while not owner.release.is_set():
                        await asyncio.sleep(0.005)
                return name

        class Connection(_Connection):
            async def _open_client(self, stack):
                owner.open_count += 1
                owner.opened.set()
                while not owner.startup.is_set():
                    await asyncio.sleep(0.005)
                return Client()

            async def _list_tools(self, client):
                return []

        self.runtime = _AsyncRuntime()
        self.connection = Connection(self.runtime, McpServerConfig("test", timeout=1))
        self.addCleanup(self.runtime.close)
        self.addCleanup(self.connection.close, 0.2)

    def start_slow_request(self):
        self.connection.start(1)
        self.result = []

        def run():
            try:
                self.result.append(self.connection.request("call", {"name": "slow"}, 1))
            except Exception as exc:
                self.result.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.release.set)
        self.assertTrue(self.started.wait(1))
        return thread

    def test_queued_timeout_is_not_executed_later(self):
        thread = self.start_slow_request()
        with self.assertRaises(TimeoutError) as caught:
            self.connection.request("call", {"name": "expired"}, 0.05)
        self.assertNotIsInstance(caught.exception, McpExecutionUnknownError)
        self.release.set()
        thread.join(1)
        self.connection.request("refresh", {}, 1)
        self.assertEqual(self.calls, ["slow"])
        self.assertEqual(self.result, ["slow"])

    def test_queued_cancellation_prevents_execution(self):
        thread = self.start_slow_request()
        context = ExecutionContext(time.monotonic() + 1)
        timer = threading.Timer(0.03, context.cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(TimeoutError) as caught:
            self.connection.request("call", {"name": "cancelled"}, 1, context=context)
        self.assertNotIsInstance(caught.exception, McpExecutionUnknownError)
        self.release.set()
        thread.join(1)
        self.connection.request("refresh", {}, 1)
        self.assertEqual(self.calls, ["slow"])

    def test_started_timeout_reports_unknown_and_allows_the_next_request(self):
        self.connection.start(1)
        with self.assertRaisesRegex(McpExecutionUnknownError, "may have taken effect"):
            self.connection.request("call", {"name": "slow"}, 0.05)
        self.assertEqual(self.calls, ["slow"])
        self.assertEqual(self.connection.request("call", {"name": "next"}, 1), "next")

    def test_startup_time_is_part_of_the_request_deadline(self):
        self.startup.clear()
        timer = threading.Timer(0.1, self.startup.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(McpExecutionUnknownError):
            self.connection.request("call", {"name": "slow"}, 0.15)
        self.assertTrue(self.started.is_set())
        self.assertEqual(self.connection.request("call", {"name": "next"}, 1), "next")

    def test_parallel_start_reuses_one_connection(self):
        self.startup.clear()
        barrier = threading.Barrier(3)
        results = []

        def connect():
            barrier.wait()
            try:
                self.connection.start(1)
                results.append("connected")
            except Exception as exc:
                results.append(exc)

        threads = [threading.Thread(target=connect) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        self.assertTrue(self.opened.wait(1))
        self.runtime.submit(asyncio.sleep(0.03)).result(timeout=1)
        self.startup.set()
        for thread in threads:
            thread.join(1)
        self.assertEqual(results, ["connected", "connected"])
        self.assertEqual(self.open_count, 1)
        self.assertEqual(self.connection.state, "connected")

    def test_one_start_waiter_timing_out_does_not_cancel_shared_startup(self):
        self.startup.clear()
        with self.assertRaises(TimeoutError):
            self.connection.start(0.03)
        self.assertEqual(self.connection.state, "connecting")
        self.startup.set()
        self.connection.start(1)
        self.assertEqual(self.open_count, 1)
        self.assertEqual(self.connection.state, "connected")

    def test_manager_creates_one_connection_during_parallel_discovery(self):
        manager = McpManager(
            Path("unused.json"), Path.cwd(), servers={"test": McpServerConfig("test")}
        )
        self.addCleanup(manager.close)
        barrier = threading.Barrier(5)
        connections = []

        def discover():
            barrier.wait()
            connections.append(manager._connection("test"))

        threads = [threading.Thread(target=discover) for _ in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(1)
        self.assertEqual(len(connections), 4)
        self.assertTrue(all(connection is connections[0] for connection in connections))

    def test_manager_preserves_unknown_execution_status(self):
        manager = McpManager(
            Path("unused.json"), Path.cwd(), servers={"test": McpServerConfig("test")}
        )
        self.addCleanup(manager.close)
        with (
            patch.object(manager, "_find_tool", return_value=("test", {"name": "slow"})),
            patch.object(manager, "_connection", return_value=self.connection),
        ):
            result = manager.execute(
                "call", tool="slow", _context=ExecutionContext(time.monotonic() + 0.05)
            )
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.error["code"], "execution_unknown")

    def test_closed_connection_cannot_be_reopened(self):
        self.connection.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.connection.start(1)
        self.assertEqual(self.open_count, 0)

    def test_close_during_startup_unblocks_waiters(self):
        self.startup.clear()
        results = []

        def connect():
            try:
                self.connection.start(1)
            except Exception as exc:
                results.append(exc)

        thread = threading.Thread(target=connect)
        thread.start()
        self.addCleanup(thread.join, 1)
        self.assertTrue(self.opened.wait(1))
        self.connection.close(0.1)
        thread.join(1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RuntimeError)


if __name__ == "__main__":
    unittest.main()
