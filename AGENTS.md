# AssistGen Agent 开发约定

## 项目结构

- `llm_backend/app/main.py`：FastAPI 入口。
- `llm_backend/app/lg_agent/`：LangGraph Agent 主流程、状态和提示词。
- `llm_backend/app/harness/`：模型选择、工具注册、路由策略和执行 trace。
- `llm_backend/app/services/`：普通服务层，例如模型服务、缓存、搜索等。
- `vue_frontend/`：前端页面。

## Harness 规则

- LangGraph 节点不要直接实例化 `ChatDeepSeek` 或 `ChatOllama`。
- 文本模型统一通过 `get_agent_harness().models` 获取。
- 可被模型调用的工具统一通过 `get_agent_harness().tools` 注册和分组。
- 路由映射统一通过 `get_agent_harness().router` 控制。
- 关键模型选择、路由选择和工具组选择统一写入 `harness.trace`。

## 本地验证

后端编译：

```bash
cd llm_backend
python -m compileall app
```

启动后端：

```bash
cd llm_backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 9000
```

启动前端按前端目录中的 `package.json` 脚本执行。

## 安全约束

- 不提交 `.env`、API Key、数据库密码和本地日志。
- 不把任意系统命令、文件删除、数据库写入类危险能力直接暴露给模型。
- 新增工具时必须先写 schema、风险等级和失败兜底，再接入 Agent 流程。
