"""A tiny MCP server for trying AgentSynth's MCP environment.

Run it directly, or point an MCPEnvironment at it:

    from agentsynth.environments import MCPEnvironment
    env = MCPEnvironment(command="python", args=["examples/mcp_server.py"])

Requires `pip install "agentsynth-ai[mcp]"` (Python 3.10+).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentsynth-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a string."""
    return len(text.split())


@mcp.tool()
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@mcp.tool()
def uppercase(text: str) -> str:
    """Uppercase a string."""
    return text.upper()


if __name__ == "__main__":
    mcp.run()
