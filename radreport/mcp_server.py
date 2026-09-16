"""search_reports as an MCP server, so any MCP client can use the report corpus.

    python -m radreport.mcp_server          # stdio, which is what clients spawn

Only the one tool, and only its BM25 path. The agent offers the model exactly
`query` and `k`, and an MCP client is just another model, so it gets the same
surface. The embedding and hybrid retrievers stay reachable from Python; exposing
them here would mean a sentence-transformers load behind a tool call that a
client expects to answer in milliseconds.

The tool is a thin shell over radreport.tools.reports.search_reports. Our
ToolError is re-raised as the SDK's, because the SDK treats any other exception
as a crash and withholds its message. Re-raised, it arrives as an `isError`
result the model can read and recover from, which is the same contract
agent.py's dispatcher gives the model.
"""

from __future__ import annotations

import os
from typing import Any

# search_reports needs neither torch nor the imaging models, but importing
# radreport.tools imports the imaging tools too, and with torch that is seconds
# of startup before the client sees a single tool. Demo mode swaps those for the
# precomputed stand-ins, which this server never calls anyway. setdefault, so an
# explicit RADREPORT_DEMO=0 is still honoured.
os.environ.setdefault("RADREPORT_DEMO", "1")

from mcp.server import MCPServer                         # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError  # noqa: E402
from mcp.types import ToolAnnotations                    # noqa: E402

from radreport.config import DISCLAIMER                  # noqa: E402
from radreport.tools.errors import ToolError             # noqa: E402
from radreport.tools.reports import search_reports as _search_reports  # noqa: E402

server = MCPServer(
    name="radreport",
    instructions=(
        "Search de-identified radiologist reports from the Indiana University "
        f"Chest X-ray Collection. {DISCLAIMER}"
    ),
)


@server.tool(
    annotations=ToolAnnotations(
        title="Search radiology reports",
        read_only_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def search_reports(query: str, k: int = 3) -> dict[str, Any]:
    """Search 3,800 de-identified radiologist reports by free text and return
    the best lexical (BM25) matches.

    Use this to find similar or example cases by description, e.g. 'reports
    mentioning a large pleural effusion'. The reports belong to other patients;
    it cannot answer a question about a specific named case. Matching is
    keyword-based, so 'enlarged heart' will not match 'cardiomegaly'.

    Args:
        query: Free-text clinical query.
        k: How many reports to return, 1 to 10.
    """
    # The agent trusts its own k; an arbitrary client gets the clamp that
    # search_literature already applies, so nobody pulls the whole corpus.
    try:
        return _search_reports(query, k=max(1, min(int(k), 10)))
    except ToolError as exc:
        raise MCPToolError(exc.message) from exc


if __name__ == "__main__":
    server.run()
