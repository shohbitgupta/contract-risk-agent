import argparse
import asyncio

from client.mcp_client import run as run_client
from mcp_server.mcp_server import mcp


def main():
    """
    Entry point for running MCP server or client.

    Example:
        >>> # Server
        >>> # python src/run_mcp.py --mode server
        >>> # Client
        >>> # python src/run_mcp.py --mode pdf --pdf-url https://...
    """
    parser = argparse.ArgumentParser(description="MCP server/client entry")
    parser.add_argument("--mode", choices=["server", "text", "pdf"], default="server")
    parser.add_argument("--text", dest="contract_text")
    parser.add_argument("--pdf-url", dest="pdf_url")
    parser.add_argument("--state", default="uttar_pradesh")
    args = parser.parse_args()

    if args.mode == "server":
        mcp.run()
        return

    asyncio.run(
        run_client(
            args.mode,
            contract_text=args.contract_text,
            pdf_url=args.pdf_url,
            state=args.state
        )
    )


if __name__ == "__main__":
    main()
