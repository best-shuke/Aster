import os

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

server = MCPServer("agent-test-server", log_level="ERROR")


@server.tool()
def echo(text: str) -> dict[str, str]:
    """Echo text for integration tests."""
    return {"echo": text}


@server.tool()
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


if os.environ.get("MCP_TEST_DYNAMIC"):
    server.remove_tool("echo")
    server.remove_tool("add")
    dynamic_registered = False

    def text_value(text: str) -> dict[str, str]:
        return {"text": text}

    def count_value(count: int) -> dict[str, int]:
        return {"count": count}

    @server.tool()
    async def configure_catalog(stage: int, ctx: Context) -> dict[str, int]:
        """Change the synthetic tool catalog and announce it."""
        global dynamic_registered
        if dynamic_registered:
            server.remove_tool("dynamic_value")
        dynamic_registered = stage in {1, 2}
        if dynamic_registered:
            server.add_tool(text_value if stage == 1 else count_value, name="dynamic_value")
        else:
            server.remove_tool("configure_catalog")
        for _ in range(3):
            if ctx.protocol_version in MODERN_PROTOCOL_VERSIONS:
                await ctx.notify_tools_changed()
            else:
                await ctx.session.send_tool_list_changed()
        return {"stage": stage}


if __name__ == "__main__":
    transport = os.environ.get("MCP_TEST_TRANSPORT", "stdio")
    if transport == "stdio":
        server.run()
    else:
        server.run(
            transport,
            host="127.0.0.1",
            port=int(os.environ["MCP_TEST_PORT"]),
        )
