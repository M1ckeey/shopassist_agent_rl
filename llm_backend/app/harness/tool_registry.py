"""Tool registry for Agent and GraphRAG workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from pydantic import BaseModel


ToolRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolSpec:
    """Metadata for a model-callable tool."""

    name: str
    schema: type[BaseModel]
    description: str
    risk: ToolRisk = "low"
    enabled: bool = True


@dataclass(frozen=True)
class ToolGroup:
    """A named group of tools used by a specific workflow."""

    name: str
    specs: List[ToolSpec]
    predefined_cypher_dict: Dict[str, str]

    @property
    def schemas(self) -> List[type[BaseModel]]:
        return [spec.schema for spec in self.specs if spec.enabled]


class ToolRegistry:
    """Central registry for all tools exposed to model decisions."""

    def __init__(self) -> None:
        from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.cypher_dict import (
            predefined_cypher_dict,
        )
        from app.lg_agent.kg_sub_graph.kg_tools_list import (
            cypher_query,
            microsoft_graphrag_query,
            predefined_cypher,
            real_time_network_query,
        )

        self._predefined_cypher_dict = predefined_cypher_dict
        self._tools: Dict[str, ToolSpec] = {
            "cypher_query": ToolSpec(
                name="cypher_query",
                schema=cypher_query,
                description="Generate Cypher dynamically and query Neo4j for structured product/order data.",
                risk="medium",
            ),
            "predefined_cypher": ToolSpec(
                name="predefined_cypher",
                schema=predefined_cypher,
                description="Run allowlisted high-frequency Cypher templates.",
                risk="low",
            ),
            "microsoft_graphrag_query": ToolSpec(
                name="microsoft_graphrag_query",
                schema=microsoft_graphrag_query,
                description="Query GraphRAG for unstructured after-sales, fault, warranty, and review knowledge.",
                risk="low",
            ),
            "real_time_network_query": ToolSpec(
                name="real_time_network_query",
                schema=real_time_network_query,
                description="Search real-time public web information when local knowledge is insufficient.",
                risk="medium",
                enabled=False,
            ),
        }

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def list_enabled(self) -> List[ToolSpec]:
        return [tool for tool in self._tools.values() if tool.enabled]

    def get_group(self, name: str) -> ToolGroup:
        if name != "graphrag":
            raise KeyError(f"Unknown tool group: {name}")
        return ToolGroup(
            name="graphrag",
            specs=[
                self._tools["cypher_query"],
                self._tools["predefined_cypher"],
                self._tools["microsoft_graphrag_query"],
            ],
            predefined_cypher_dict=self._predefined_cypher_dict,
        )
