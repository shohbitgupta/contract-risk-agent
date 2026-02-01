# Contract Risk Agent

## MCP Server

You can expose the contract analysis pipeline as an MCP tool server.

Start the server:

```
pipenv run python src/mcp_server.py
```

Available tools:
- `analyze_contract_pdf(pdf_url: str, state: str = "uttar_pradesh")`
- `analyze_contract_text(contract_text: str, state: str = "uttar_pradesh")`