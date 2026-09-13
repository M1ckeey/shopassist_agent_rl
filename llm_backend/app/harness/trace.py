"""Lightweight execution tracing for the Agent harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logger import get_logger

logger = get_logger(service="harness.trace")


@dataclass
class TraceEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TraceLogger:
    """Records important harness decisions without coupling to a tracing vendor."""

    def record(self, name: str, **payload: Any) -> TraceEvent:
        event = TraceEvent(name=name, payload=payload)
        logger.info(f"harness_event={name} payload={payload}")
        return event
