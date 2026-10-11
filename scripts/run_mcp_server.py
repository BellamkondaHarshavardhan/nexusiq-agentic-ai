"""Launcher for the NexusIQ MCP server that works from any folder (use this path in Claude Desktop / VS Code configs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexusiq.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
