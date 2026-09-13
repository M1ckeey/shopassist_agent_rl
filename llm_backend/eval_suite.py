# -*- coding: utf-8 -*-
"""分层评测 runner(支持 pass^k 多轮;2026-09-08 晚 2 已扩到 50 题)。

每问跑完整 agent(langgraph)。同一问重复跑时每次都用全新 thread_id → 相互独立的一次轨迹。

记录:
  router_type : 顶层路由判定(general/additional/graphrag/image/file)
  stdout      : 子图 stdout(含 predefined_cypher 的 statement_name / params / 报错),供人工核工具选择
  answer      : 最终回复全文(不截断,供人工逐题核)
  had_error   : 子图执行阶段是否出现 Cypher/参数等异常

用法(项目根目录,用 shopagent 解释器):
  python llm_backend/eval_suite.py                          # 跑全部 QUESTIONS,每问 1 遍(pass@1)
  python llm_backend/eval_suite.py B1 B2 A4 ...             # 只跑指定 id(顺序)
  python llm_backend/eval_suite.py B1 B2 --repeat=3         # 指定题各跑 3 遍,打印 pass@1/至少一次/全过 汇总
结果打印到控制台,同时追加写入 llm_backend/logs/eval_suite.jsonl(每轮一行,含 run 序号)。
"""
import sys
import json
import asyncio
import contextlib
import io
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# (id, 类别, 问题, gold 事实片段[任一命中即算 auto-hit,仅供信号,最终以人工读全文为准])
QUESTIONS = [
    # ---- 套件 B：图查询(金标来自 Neo4j 实测) ----
    ("B1", "graph", "订单93发货了吗？帮我查一下它的发货状态", ["未发货", "尚未发货", "还没"]),
    ("B2", "graph", "订单93包含了哪些产品？", ["亚马逊 智能开关 Lite", "华为 智能门铃 Plus", "索尼 智能马桶 Elite", "苹果 智能插座 Elite", "格力 智能空调 Standard"]),
    ("B3", "graph", "处理订单93的员工是谁？", ["孙凤英", "凤英"]),
    ("B4", "graph", "订单93是通过哪家物流配送的？", ["邮政EMS", "EMS"]),
    ("B5", "graph", "客户海创网络有限公司人工智能一共下了多少订单？", ["6"]),
    ("B6", "graph", "顺丰速运配送了哪些订单？", ["82", "53", "75", "18", "61", "13", "37", "88", "60"]),
    ("B7", "graph", "智能门铃类有哪些产品？", ["谷歌 智能门铃 Basic", "华为 智能门铃 Plus"]),
    ("B8", "graph", "苹果 智能插座 Elite 的价格和库存是多少？", ["8334.89", "8334", "293"]),
    ("B9", "graph", "博世智能系统供应了哪些产品？", ["博世 智能冰箱 Lite", "博世 智能加湿器 Plus"]),
    ("B10", "graph", "哪个产品收到的评价最多？", ["格力 智能空调 Standard", "格力"]),
    ("B11", "graph", "格力 智能空调 Standard 的平均评分是多少？", ["3.3", "3.4"]),
    ("B12", "graph", "2023年10月下了多少笔订单？", ["5"]),
    ("B13", "graph", "华为智能生活供应了哪些产品？", ["华为 智能门铃 Plus", "华为 智能开关 Mini"]),
    # ---- 套件 B 扩(09-08 深夜,扩到 30 题;金标同上均经 Neo4j 核验) ----
    ("B14", "graph", "谷歌 智能门铃 Basic 的价格和库存是多少？", ["3322.66", "3322", "849"]),
    ("B15", "graph", "现在一共有多少笔订单还没发货？", ["12"]),
    ("B16", "graph", "最近一笔订单是哪一笔？", ["63"]),
    ("B17", "graph", "哪个产品的评分最高？", ["华为 智能开关 Mini"]),
    ("B18", "graph", "孙凤英一共处理了多少笔订单？", ["39", "孙凤英", "凤英"]),
    ("B19", "graph", "订单5是什么时候发货的？", ["2024-06-29", "2024-06", "已发货"]),
    # ---- 套件 B 扩二(09-08 深夜2,30→50;金标均回 Neo4j 核验,覆盖产品/订单/销售/延迟等模板键) ----
    ("B20", "graph", "订单63是哪位客户的订单？", ["海创网络有限公司人工智能", "海创"]),
    ("B21", "graph", "有哪些订单延迟发货了？", ["5", "57", "46", "68", "50", "1"]),
    ("B22", "graph", "苹果 智能插座 Elite 一共卖了多少销售额？", ["2305161", "230", "插座"]),
    ("B23", "graph", "哪个产品类别的销售额最高？", ["智能门铃"]),
    ("B24", "graph", "哪个月的销售额最高？", ["2023-11", "11 月", "11月", "1417777"]),
    ("B25", "graph", "店里有没有库存低于10的产品？", ["没有", "无"]),
    ("B26", "graph", "智能开关这个类目下有哪几款产品？", ["亚马逊 智能开关 Lite", "华为 智能开关 Mini"]),
    ("B27", "graph", "超艺传媒有限公司大数据分析一共下过多少笔订单？", ["15"]),
    ("B28", "graph", "店里一共有多少个产品品类？", ["8"]),
    ("B29", "graph", "博世 智能加湿器 Plus 的价格和库存是多少？", ["3857.01", "3857", "957"]),
    ("B30", "graph", "亚马逊 智能开关 Lite 一共卖了多少销售额？", ["204947", "20"]),
    ("B31", "graph", "订单13是什么时候下单的？", ["2024-01-15", "2024-01", "1 月"]),
    # ---- 套件 A：路由/拒答 ----
    ("A1", "route", "在吗？", []),
    ("A2", "route", "今天北京天气怎么样？", []),
    ("A3", "route", "帮我写一首关于春天的诗", []),
    ("A4", "reject", "你们卖运动鞋吗？", []),
    ("A5", "route", "我这个订单状态怎么样？", []),
    ("A6", "reject", "小米智能音箱多少钱？", []),
    ("A7", "reject", "我在别的平台买的羽绒服想退,你们能帮忙处理吗？", []),
    # ---- 套件 A 扩(09-08 深夜) ----
    ("A8", "reject", "你们有智能手表卖吗？", []),
    ("A9", "reject", "能帮我看看今天股市行情吗？", []),
    ("A10", "reject", "我昨天下的订单把收货地址填错了,能帮我改一下吗？", []),
    ("A11", "reject", "可以给我开发票吗？", []),
    # ---- 套件 A 扩二(09-08 深夜2,30→50;干净的路由边界/拒答线) ----
    ("A12", "route", "在吗？想咨询一下你们的产品", []),
    ("A13", "additional", "帮我查一下我的快递到哪了", []),
    ("A14", "reject", "你们有智能扫地机器人吗？", []),
    ("A15", "route", "你会写代码吗？帮我写个冒泡排序", []),
    ("A16", "reject", "能帮我去别的电商平台比一下价吗？", []),
    ("A17", "reject", "退货的话运费要我自己出吗？", []),
    ("A18", "route", "你是真人客服还是机器人？", []),
    ("A19", "reject", "能帮我把订单取消吗？我不想要了", []),
]


def gold_hit(answer: str, gold: list[str]) -> bool:
    """auto 信号(宽松):gold 任一 token 命中即 True。仅供提示——同题若 gold 是"同义改写"
    或多实体,任何规则都可能误判(B12 '5' 假阳、B1 '还未发货' 与 '未发货'、B8 千分位),
    所以报告口径一律以人工读全文为准,不要直接采信 auto。"""
    if not gold:
        return False
    return any(t in answer for t in gold)


async def run_one(qid: str, prompt: str) -> dict:
    from app.lg_agent.lg_states import InputState
    from app.lg_agent.lg_builder import graph
    from app.lg_agent.utils import new_uuid

    thread = {"configurable": {"thread_id": new_uuid()}}
    input_state = InputState(messages=prompt)

    router_type = None
    answer = ""
    buf = io.StringIO()
    crashed = None
    try:
        with contextlib.redirect_stdout(buf):
            async for values in graph.astream(input=input_state, stream_mode="values", config=thread):
                if not isinstance(values, dict):
                    continue
                if values.get("router") is not None and router_type is None:
                    router_type = values["router"].get("type") if isinstance(values["router"], dict) else getattr(values["router"], "type", None)
                msgs = values.get("messages") or []
                for m in msgs:
                    cls = type(m).__name__
                    if cls == "AIMessage" and m.content:
                        answer = m.content
    except Exception as e:  # 单题失败只记录,不中断整批(pass^k 语义:这次运行 = FAIL)
        crashed = f"{type(e).__name__}: {e}"

    out = buf.getvalue()

    had_error = crashed is not None or any(k in out for k in
                    ("ClientError", "CypherError", "ParameterMissing", "SyntaxError", "error code", "Traceback"))
    tool = "predefined_cypher" if "statement_name predefined_cypher" in out else \
           ("text2cypher" if "records" in out and "statement_name" not in out else "?")
    if crashed:
        answer = f"[运行中止 {crashed}] {answer}".strip()

    return {
        "id": qid, "router": router_type, "tool": tool,
        "had_error": had_error, "answer": answer.strip(),
    }


def parse_args(argv):
    """返回 (qids, repeat)。--repeat=N 或 --repeat 裸(=3)。"""
    qids, repeat = [], 1
    for a in argv:
        if a.startswith("--repeat"):
            eq = a.split("=", 1)
            repeat = int(eq[1]) if len(eq) > 1 else 3
        else:
            qids.append(a)
    return qids, repeat


def main():
    qids, repeat = parse_args(sys.argv[1:])
    todo = [q for q in QUESTIONS if q[0] in qids] if qids else QUESTIONS
    missing = [i for i in qids if i not in {q[0] for q in QUESTIONS}]
    if missing:
        print("未知 id:", missing)
        sys.exit(2)

    async def run_all():
        out = []
        for qid, _cat, prompt, _gold in todo:
            for i in range(repeat):
                print(f">> 跑 {qid} round{i + 1}/{repeat} :: {prompt}", flush=True)
                r = await run_one(qid, prompt)
                r["run"] = i
                out.append(r)
        return out

    results = asyncio.run(run_all())

    logdir = BACKEND_DIR / "logs"
    logdir.mkdir(exist_ok=True)
    with open(logdir / "eval_suite.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    byid = {}
    for qid, cat, prompt, gold in todo:
        byid[qid] = (cat, gold)

    n_runs_pass = n_runs = 0
    print("\n==== 逐题轮次(auto-hit 仅信号,最终判分以人工读全文为准) ====")
    for qid, cat, prompt, gold in todo:
        runs = [r for r in results if r["id"] == qid]
        auto = [not r["had_error"] and gold_hit(r["answer"], gold) for r in runs]
        line = [("OK " if a else "xx ") for a in auto]
        print(f"{qid:<4} {cat:<8} rounds={' '.join(line)}  auto_pass={sum(auto)}/{len(auto)}  router={[r['router'] for r in runs]}")
        for r in runs:
            first = r["answer"].replace("\n", " ⏎ ")[:80]
            print(f"        [r{r['run'] + 1}] had_error={r['had_error']} tool={r['tool']}")
            print(f"            {first}")
        n_runs_pass += sum(auto)
        n_runs += len(auto)

    # 问题级 pass^k 汇总
    if repeat >= 2:
        n_reach = n_allk = nq = 0
        for qid, cat, prompt, gold in todo:
            runs = [r for r in results if r["id"] == qid]
            auto = [not r["had_error"] and gold_hit(r["answer"], gold) for r in runs]
            nq += 1
            n_reach += int(any(auto))
            n_allk += int(all(auto))
        print("\n==== pass^k 汇总(auto 口径,待人工复核) ====")
        print(f"run 级命中率: {n_runs_pass}/{n_runs} = {n_runs_pass / max(n_runs, 1):.0%}")
        print(f"pass@1(只取每问第 1 轮): 见各问 round1")
        print(f"pass^k 至少一次命中(τ-bench reach): {n_reach}/{nq} = {n_reach / max(nq, 1):.0%}")
        print(f"k 轮全过(一致性,计划 §5 口径): {n_allk}/{nq} = {n_allk / max(nq, 1):.0%}")
    else:
        print(f"\nrun 级 auto 命中: {n_runs_pass}/{n_runs} (仅供信号;pass@1 需人工读全文判)")


if __name__ == "__main__":
    main()
