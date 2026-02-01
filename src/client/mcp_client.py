import argparse
import asyncio

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession


async def run(mode: str, *, contract_text: str | None, pdf_url: str | None, state: str):
    """
    Run MCP client requests against a spawned local server.

    Example:
        >>> asyncio.run(run("text", contract_text="...", pdf_url=None, state="uttar_pradesh"))
    """
    # Spawn MCP server as a subprocess over stdio
    server = StdioServerParameters(
        command="python",
        args=["src/run_mcp.py", "--mode", "server"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if mode == "text":
                if not contract_text:
                    raise ValueError("contract_text is required for mode=text")
                result = await session.call_tool(
                    "analyze_contract_text",
                    {
                        "contract_text": contract_text,
                        "state": state
                    }
                )
                print(result)
                return

            if not pdf_url:
                raise ValueError("pdf_url is required for mode=pdf")
            result = await session.call_tool(
                "analyze_contract_pdf",
                {
                    "pdf_url": pdf_url,
                    "state": state
                }
            )
            print(result)


def main():
    """
    CLI entry for the MCP client.
    """
    parser = argparse.ArgumentParser(description="MCP client runner")
    parser.add_argument("--mode", choices=["text", "pdf"], required=True)
    parser.add_argument("--text", dest="contract_text")
    parser.add_argument("--pdf-url", dest="pdf_url")
    parser.add_argument("--state", default="uttar_pradesh")
    args = parser.parse_args()

    asyncio.run(
        run(
            args.mode,
            contract_text=args.contract_text,
            pdf_url=args.pdf_url,
            state=args.state
        )
    )


if __name__ == "__main__":
    main()
