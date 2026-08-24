"""PubMed tests. Marked network: they hit a live third-party service."""

from __future__ import annotations

import pytest

from radreport.tools.errors import ToolError
from radreport.tools.literature import search_literature


def test_empty_query_raises_without_network():
    """Validation must happen BEFORE the request, so this needs no network."""
    with pytest.raises(ToolError, match="empty"):
        search_literature("")


@pytest.mark.network
def test_returns_citations():
    result = search_literature("cardiothoracic ratio chest radiograph", k=3)
    assert result["ok"] is True
    assert 1 <= len(result["citations"]) <= 3
    first = result["citations"][0]
    assert first["pmid"].isdigit()
    assert first["url"].endswith(f"{first['pmid']}/")
    assert first["title"]


@pytest.mark.network
def test_k_is_clamped():
    """The model will eventually ask for 100 results. We clamp to 10."""
    assert len(search_literature("pneumonia", k=500)["citations"]) <= 10


@pytest.mark.network
def test_nonsense_query_returns_empty_not_error():
    result = search_literature("qwertyuiop zxcvbnm asdfghjkl 12345xyz", k=3)
    assert result["ok"] is True
    assert result["citations"] == []
