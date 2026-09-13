# Agent Harness 设计说明

本项目把 Agent 拆成两部分：

```text
Agent = Model + Harness
```

`Model` 负责理解、推理和生成；`Harness` 负责模型之外的工程控制，包括模型选择、工具暴露、流程路由、上下文传递、失败兜底和日志追踪。

## 代码位置

- `llm_backend/app/harness/model_registry.py`：按角色创建模型。
- `llm_backend/app/harness/tool_registry.py`：集中注册可被模型调用的工具。
- `llm_backend/app/harness/policy_router.py`：把意图类型映射到 LangGraph 节点。
- `llm_backend/app/harness/trace.py`：记录模型选择、路由选择和工具选择事件。
- `llm_backend/app/harness/runtime.py`：统一的 `AgentHarness` 运行时入口。

## 模型选择

`ModelRegistry` 按角色提供模型：

- `get_chat_model()`：普通聊天模型。
- `get_reason_model()`：深度推理模型。
- `get_agent_model()`：Agent 编排、意图识别、工具选择和最终回答模型。
- `get_vision_profile()`：视觉模型配置，用于图片理解。

具体使用 DeepSeek 还是 Ollama，由 `.env` 中的 `CHAT_SERVICE`、`REASON_SERVICE`、`AGENT_SERVICE` 决定。

## 工具注册

`ToolRegistry` 当前注册了这些工具：

- `predefined_cypher`：执行预定义的高频 Cypher 查询，风险较低。
- `cypher_query`：由模型生成 Cypher 后查询 Neo4j，风险中等。
- `microsoft_graphrag_query`：查询 GraphRAG 非结构化知识库，风险较低。
- `real_time_network_query`：联网搜索工具，默认关闭，避免模型随意访问外部信息。

GraphRAG 工作流不再在节点里临时拼工具列表，而是通过：

```python
tool_group = harness.tools.get_group("graphrag")
```

获取允许暴露给模型的工具集合。

## 路由策略

`PolicyRouter` 负责把意图识别结果映射到 LangGraph 节点：

- `general-query` -> 普通回答节点。
- `additional-query` -> 追问信息节点。
- `graphrag-query` -> 知识库 / 图谱查询节点。
- `image-query` -> 图片分析节点。
- `file-query` -> 文件问答节点。

如果请求中带有图片路径，优先走图片分析节点。

## 执行追踪

`TraceLogger` 会记录关键 harness 决策，例如：

- 选用了哪个模型角色。
- 意图识别结果是什么。
- 路由到了哪个节点。
- GraphRAG 暴露了哪些工具。

这些日志用于排查 bad case，例如“为什么没有走知识库”“为什么工具选错了”“为什么模型没有调用图数据库”。

## 新增工具流程

1. 在工具模块中定义 Pydantic schema。
2. 在 `ToolRegistry` 中注册 `ToolSpec`，写清楚用途、风险等级和是否启用。
3. 按业务场景加入某个 `ToolGroup`。
4. 在 LangGraph 节点中只读取 `ToolGroup`，不要直接 import 工具列表。
5. 给工具失败分支加错误兜底，避免模型无限重试。

## 新增模型流程

1. 在 `.env` 和 `app/core/config.py` 中增加配置项。
2. 在 `ModelRegistry` 中增加对应 role 或 provider。
3. LangGraph 节点继续通过 `harness.models` 获取模型，不直接依赖具体厂商 SDK。

## 当前收益

- 模型选择从节点代码中解耦，后续切换 DeepSeek / Ollama 更简单。
- 工具暴露边界集中管理，便于控制高风险工具。
- 路由规则集中维护，避免分支散落在多个节点。
- trace 能辅助定位意图识别、工具选择和 GraphRAG 查询中的 bad case。
