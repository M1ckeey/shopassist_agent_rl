"""Routing policies for LangGraph agent entrypoints."""

from __future__ import annotations

from typing import Literal

RouterType = Literal[
    "general-query",
    "additional-query",
    "graphrag-query",
    "image-query",
    "file-query",
]

GraphNode = Literal[
    "respond_to_general_query",
    "get_additional_info",
    "create_research_plan",
    "create_image_query",
    "create_file_query",
]


class PolicyRouter:
    """Maps router labels and runtime hints to LangGraph nodes."""

    _route_map: dict[RouterType, GraphNode] = {
        "general-query": "respond_to_general_query",
        "additional-query": "get_additional_info",
        "graphrag-query": "create_research_plan",
        "image-query": "create_image_query",
        "file-query": "create_file_query",
    }

    def route(self, router_type: str, *, has_image: bool = False) -> GraphNode:
        if has_image:
            return "create_image_query"
        try:
            return self._route_map[router_type]  # type: ignore[index]
        except KeyError as exc:
            raise ValueError(f"Unknown router type {router_type}") from exc
