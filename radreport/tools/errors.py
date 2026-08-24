"""One exception type for every predictable tool failure.

Design decision: tools raise, the agent catches. A tool should not return
`{"error": ...}` on its own, because then every caller has to remember to check
for it and a forgotten check silently produces a wrong clinical answer. Raising
makes the failure loud locally; the agent's dispatcher is the single place that
converts it into a message the model can read and recover from.
"""

from __future__ import annotations


class ToolError(Exception):
    """A tool failed in a way we anticipated and can describe to the model."""

    def __init__(self, message: str, *, tool: str = "", recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.tool = tool
        self.recoverable = recoverable

    def as_tool_result(self) -> dict:
        """The shape the agent feeds back into the conversation."""
        return {
            "ok": False,
            "error": self.message,
            "tool": self.tool,
            "recoverable": self.recoverable,
        }
