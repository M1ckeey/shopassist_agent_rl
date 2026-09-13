"""Agent harness package.

The harness owns model selection, tool registration, routing policy, and
execution tracing. LangGraph nodes should depend on this layer instead of
instantiating models or tool lists directly.
"""

from app.harness.runtime import AgentHarness, get_agent_harness

__all__ = ["AgentHarness", "get_agent_harness"]
