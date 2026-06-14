"""
Small utility node: increments retry_count.

Separated from query_analysis so the conditional edge can route here BEFORE looping
back to query_analysis, keeping the retry-counter logic in exactly one place and easy
to unit test.
"""

from __future__ import annotations

from app.graph.state import GraphState


def increment_retry_node(state: GraphState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}
