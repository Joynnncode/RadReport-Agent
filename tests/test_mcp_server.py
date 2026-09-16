"""The MCP wrapper around search_reports, exercised through a real MCP client.

These go through the protocol rather than calling the function, because what
can break here is the protocol surface: the schema a client sees, how a
ToolError arrives, and whether the server starts at all as a subprocess.
"""

from __future__ import annotations

import sys

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from radreport.config import REPO_ROOT
from radreport.mcp_server import server


def _call(target, name: str, arguments: dict):
    async def go():
        async with Client(target) as client:
            return await client.call_tool(name, arguments)
    return anyio.run(go)


def test_lists_one_read_only_tool_with_the_agents_arguments():
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools
    tools = anyio.run(go)

    assert [t.name for t in tools] == ["search_reports"]
    tool = tools[0]
    assert set(tool.input_schema["properties"]) == {"query", "k"}
    assert tool.input_schema["required"] == ["query"]
    assert tool.annotations.read_only_hint is True


def test_returns_structured_hits():
    result = _call(server, "search_reports", {"query": "pleural effusion", "k": 2})
    assert not result.is_error
    hits = result.structured_content["hits"]
    assert 1 <= len(hits) <= 2
    assert "effusion" in (hits[0]["findings"] + hits[0]["impression"]).lower()


def test_k_is_clamped_for_clients():
    result = _call(server, "search_reports", {"query": "heart", "k": 500})
    assert len(result.structured_content["hits"]) == 10


def test_tool_error_reaches_the_client_as_an_error_result():
    result = _call(server, "search_reports", {"query": "   "})
    assert result.is_error
    assert "empty" in result.content[0].text


def test_starts_over_stdio():
    """What an MCP client actually does: spawn the module and speak stdio."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "radreport.mcp_server"], cwd=str(REPO_ROOT),
    )
    result = _call(params, "search_reports", {"query": "cardiomegaly", "k": 1})
    assert not result.is_error
    assert len(result.structured_content["hits"]) == 1
