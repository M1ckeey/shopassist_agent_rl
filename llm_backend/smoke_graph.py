# -*- coding: utf-8 -*-
"""Neo4j 冒烟脚本：验证图服务在跑、数据已灌入、密码正确。

用法（项目根目录）:
    python llm_backend/smoke_graph.py

绿线 = 打印出产品/订单/客户的数量和真实名字。
"""
import sys
from pathlib import Path

# 中文 Windows 控制台默认 GBK，强制用 UTF-8 输出避免打印崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 保证能 import 到 llm_backend/app 包
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main():
    from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph

    print("[1/3] 连接 Neo4j ...")
    graph = get_neo4j_graph()  # 用 llm_backend/.env 里的配置

    print("[2/3] 数数量 ...")
    counts = graph.query(
        """
        MATCH (p:Product) WITH count(p) AS products
        OPTIONAL MATCH (o:Order) WITH products, count(o) AS orders
        OPTIONAL MATCH (c:Customer) WITH products, orders, count(c) AS customers
        RETURN products, orders, customers
        """
    )
    print(f"      → {counts}")

    print("[3/3] 抽几个真实名字（定题时以它们为准） ...")
    for label, cap in [("Product", 5), ("Customer", 3), ("Order", 3)]:
        try:
            rows = graph.query(
                f"MATCH (n:`{label}`) RETURN n LIMIT {cap}"
            )
            names = []
            for r in rows:
                node = r.get("n") or {}
                d = dict(node) if hasattr(node, "__getitem__") else node
                # 挑几个好看点的属性打印
                keys = [k for k in ("ProductName", "CompanyName", "orderId", "OrderID", "CustomerID", "FirstName")
                        if k in d]
                names.append({k: d[k] for k in keys} if keys else d)
            print(f"      [{label}] {names}")
        except Exception as e:
            print(f"      [{label}] 查询失败: {e}")

    print("\n冒烟通过 ✅  接下来可以用真实名字去定 20 题。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n冒烟失败 ❌")
        msg = str(exc).lower()
        if any(k in msg for k in ("apoc", "apoc.meta.data")):
            print("原因判断：缺 APOC 插件 → Neo4j Desktop 里选你的库 → 左侧 Plugins → 安装 APOC → 重启库")
        elif any(k in msg for k in ("authentication", "unauthorized", "401", "password")):
            print("原因判断：认证失败 → 检查 llm_backend/.env 的 NEO4J_PASSWORD 是否等于库的真实密码")
        elif any(k in msg for k in ("connection", "timed out", "refused", "resourcenotavailable",
                                     "failed to establish")):
            print("原因判断：连不上 → 打开 Neo4j Desktop 把库 Start，确认 bolt://localhost:7687 在监听")
        else:
            print(f"原始错误：{exc}")
        sys.exit(1)
