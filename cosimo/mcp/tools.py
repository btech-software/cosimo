from agent_lab import discoverable_mcp_tool


@discoverable_mcp_tool(
    "cosimo_hello_tool",
    description="Returns a friendly greeting. Replace with your own MCP tools.",
    annotations={"readOnlyHint": True},
)
async def hello_tool(container, name: str) -> str:
    return f"Hello, {name}! This is Cosimo I."
