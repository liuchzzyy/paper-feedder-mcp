"""paper-feedder-mcp: MCP server for academic paper collection."""

__version__ = "2.0.0"


def main() -> None:
    """Entry point for the MCP server."""
    from src.client.cli import main as cli_main

    cli_main()
