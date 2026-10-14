import os

from mcp.server import MCPServer

server = MCPServer("agent-test-server", log_level="ERROR")


@server.tool()
def echo(text: str) -> dict[str, str]:
    """Echo text for integration tests."""
    return {"echo": text}


@server.tool()
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


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
