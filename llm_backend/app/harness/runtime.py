"""Runtime facade that exposes the project Agent harness."""

from __future__ import annotations

from functools import lru_cache

from app.harness.model_registry import ModelRegistry
from app.harness.policy_router import PolicyRouter
from app.harness.tool_registry import ToolRegistry
from app.harness.trace import TraceLogger


class AgentHarness:
    """Coordinates model selection, tool registry, routing, and tracing."""

    def __init__(self) -> None:
        self.models = ModelRegistry()
        self.tools = ToolRegistry()
        self.router = PolicyRouter()
        self.trace = TraceLogger()


@lru_cache(maxsize=1)
def get_agent_harness() -> AgentHarness:
    return AgentHarness()
