# -*- coding: utf-8 -*-
"""端到端单题测试：给定一句中文问题，跑完整 langgraph（路由→工具→答案），
把每步节点名、工具调用、最终回复都打出来。用于面试前验证“图查询能出真答案”。

用法（项目根目录）:
    python llm_backend/e2e_test.py "订单93发货了吗"
"""
import sys
import asyncio
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def run_once(query: str) -> None:
    from app.lg_agent.lg_states import InputState
    from app.lg_agent.lg_builder import graph
    from app.lg_agent.utils import new_uuid

    thread = {"configurable": {"thread_id": new_uuid()}}
    input_state = InputState(messages=query)

    print(f"\n===== 提问: {query} =====\n")
    seen_nodes = []
    final_text = ""

    # stream_mode="values"：每走完一步给整份状态，能看清节点顺序和工具结果
    async for values in graph.astream(input=input_state, stream_mode="values", config=thread):
        if "messages" in values:
            msgs = values["messages"]
            if msgs:
                last = msgs[-1]
                name = getattr(last, "name", None)
                cls = type(last).__name__
                if cls == "AIMessage":
                    # 可能有 tool_calls，记一下
                    tcs = getattr(last, "tool_calls", None) or []
                    for tc in tcs:
                        fn = tc["name"]
                        args = tc.get("args", {})
                        print(f"  [tool_call] -> {fn}({args})")
                    if last.content:
                        final_text = last.content
                elif cls == "ToolMessage":
                    content = str(last.content)
                    print(f"  [tool_result] {content[:300]}")
                elif cls == "HumanMessage" and last.content == query:
                    pass
                else:
                    if last.content:
                        print(f"  [{cls}] {str(last.content)[:200]}")
    print("\n----- 最终回复 -----")
    print(final_text or "(空)")
    print("=====================\n")


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "订单93发货了吗"
    asyncio.run(run_once(query))


if __name__ == "__main__":
    main()
