from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional

from app.core.logger import get_logger

logger = get_logger(service="order_lookup")


class OrderLookupService:
    """Lightweight order lookup backed by the demo exported order data."""

    _orders: Optional[Dict[str, Dict[str, str]]] = None

    DATA_PATH = (
        Path(__file__).resolve().parents[1]
        / "graphrag"
        / "origin_data"
        / "exported_data"
        / "orders.csv"
    )

    ORDER_PATTERNS = (
        re.compile(r"(?:订单号|订单编号|订单|单号)\s*(?:是|为|:|：)?\s*(\d{1,12})"),
        re.compile(r"我的订单(?:号)?\s*(?:是|为|:|：)?\s*(\d{1,12})"),
    )

    @classmethod
    def _load_orders(cls) -> Dict[str, Dict[str, str]]:
        if cls._orders is not None:
            return cls._orders

        orders: Dict[str, Dict[str, str]] = {}
        if not cls.DATA_PATH.exists():
            logger.warning(f"Order data file not found: {cls.DATA_PATH}")
            cls._orders = orders
            return orders

        try:
            with cls.DATA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    order_id = str(row.get("OrderID", "")).strip()
                    if order_id:
                        orders[order_id] = row
            logger.info(f"Loaded {len(orders)} orders from {cls.DATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to load orders: {str(e)}", exc_info=True)

        cls._orders = orders
        return orders

    @classmethod
    def extract_order_id(cls, text: str) -> Optional[str]:
        if not text:
            return None
        for pattern in cls.ORDER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).lstrip("0") or "0"
        return None

    @classmethod
    def find_order(cls, order_id: str) -> Optional[Dict[str, str]]:
        return cls._load_orders().get(str(order_id).strip())

    @classmethod
    def order_summary(cls, order: Dict[str, str]) -> str:
        shipped_date = order.get("ShippedDate") or "未发货"
        return (
            f"订单{order.get('OrderID')}真实存在。"
            f"客户：{order.get('CustomerName') or '未知'}；"
            f"下单时间：{order.get('OrderDate') or '未知'}；"
            f"发货时间：{shipped_date}；"
            f"物流：{order.get('ShipperName') or '未知'}；"
            f"收货城市：{order.get('ShipCity') or '未知'}。"
        )
