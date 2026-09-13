# ShopAssist Agent - 智能客服与知识增强系统

ShopAssist Agent 是一个面向电商购物与售后场景的智能客服 Agent 系统。项目来源于日常使用购物软件时遇到的真实问题：传统 AI 客服经常答非所问、上下文记不住、退货退款判断慢、复杂售后问题容易反复转人工。为了解决这些体验问题，本项目尝试把大模型、工具调用、RAG 检索、图谱查询和多轮会话管理结合起来，构建一个更贴近真实购物售后流程的客服 Agent。

项目支持用户注册登录、会话管理、流式聊天、商城页面在线客服、订单查询、商品咨询、退款退货判断、图片理解、文件问答、知识库检索和人工介入判断，目标是提升客服回复的准确性、上下文连续性和问题处理效率。

## 项目亮点

- 面向真实电商售后流程设计，而不是只做通用聊天问答。
- 支持多轮上下文，用户连续追问时可以结合历史对话理解意图。
- 支持订单、商品、售后政策、用户信息等多类数据查询。
- 支持图片入口，可用于商品破损、错发、外包装异常等售后判断场景。
- 支持文件 RAG、GraphRAG 和 Neo4j 图谱查询，增强知识检索和复杂关系推理能力。
- 支持 Redis 语义缓存与本地 embedding，减少高频相似问题的重复模型调用。
- 设计 Agent Harness 层，统一管理模型选择、工具注册、路由策略和执行日志。

## 核心技术栈

Python、FastAPI、LangGraph、LangChain、GraphRAG、Neo4j、MySQL、Redis、Vue

## 系统能力

### 1. 智能客服对话

系统可以处理常见购物和售后问题，例如：

- 商品能不能退货？
- 订单现在是什么状态？
- 收到的商品坏了怎么办？
- 水果腐烂是否支持退款？
- 商品和描述不一致怎么办？
- 是否需要转人工客服？

相比普通客服机器人，ShopAssist Agent 更强调结合订单、售后政策、图片信息和历史上下文做判断。

### 2. 意图识别与路由

系统通过 Prompt-template 和大模型完成意图识别，将用户问题划分为：

- general：普通问答
- additional：需要补充信息的问题
- graphrag：需要知识库或图谱检索的问题
- image：图片相关问题
- file：文件问答问题

识别结果会进入 LangGraph 编排流程，再决定后续走普通回答、工具调用、RAG 检索、图谱查询还是图片理解。

### 3. LangGraph Agent 编排

项目使用 LangGraph 将客服处理链路拆成多个节点，包括：

- 意图识别
- 安全护栏
- Planner 子任务拆分
- 工具调用
- RAG / GraphRAG 检索
- Neo4j 图谱查询
- 幻觉检测
- 最终回复生成

这样做的好处是流程更可控，出错时也更容易定位是路由问题、工具问题、检索问题还是模型回复问题。

### 4. 知识增强问答

系统支持文件 RAG、GraphRAG 和 Neo4j 查询能力：

- 售后政策、商品说明、FAQ 等非结构化内容可以进入 RAG / GraphRAG。
- 订单、商品、用户、售后记录等结构化关系可以进入 Neo4j 图谱。
- 高频图数据库查询可以沉淀为 Cypher 模板。
- 动态 Schema 注入和 EXPLAIN 预校验可以降低错误 Cypher 的执行风险。

### 5. 图片售后理解

用户可以上传商品图片，系统先通过视觉模型理解图片内容，再结合订单信息、售后政策和对话上下文生成客服回复。适用场景包括：

- 商品破损
- 生鲜腐烂
- 错发漏发
- 包装破损
- 商品与描述不符

### 6. 语义缓存与工程化

项目引入 Redis 语义缓存和本地 embedding，对高频相似问题进行缓存命中，减少重复调用大模型，提高响应速度并降低 API 成本。

同时，Agent Harness 层统一管理：

- 模型选择
- 工具注册
- 路由策略
- 执行日志
- 问题排查信息

这让系统更像一个可维护的 Agent 应用，而不是简单的大模型接口封装。

## 本地启动

### 1. 创建虚拟环境

```powershell
cd E:\agnet\智能客服Agent\code\deepseek_agent
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制环境变量模板：

```powershell
Copy-Item .env.example llm_backend\.env
```

至少需要配置：

```env
DEEPSEEK_API_KEY=你的文本模型 API Key
DEEPSEEK_BASE_URL=https://api.moonshot.cn/v1
DEEPSEEK_MODEL=moonshot-v1-8k
CHAT_SERVICE=deepseek
REASON_SERVICE=deepseek
AGENT_SERVICE=deepseek

DB_HOST=localhost
DB_PORT=3306
DB_USER=shopcare_user
DB_PASSWORD=你的 MySQL 密码
DB_NAME=shopcare_agent

SECRET_KEY=本地开发密钥
```

### 3. 准备 MySQL

```sql
CREATE DATABASE IF NOT EXISTS shopcare_agent
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'shopcare_user'@'localhost'
  IDENTIFIED BY 'ShopCare_2026_local';

GRANT ALL PRIVILEGES ON shopcare_agent.* TO 'shopcare_user'@'localhost';
FLUSH PRIVILEGES;
```

### 4. 启动后端

```powershell
cd E:\agnet\智能客服Agent\code\deepseek_agent\llm_backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 9000
```

启动后访问：

- 首页：http://127.0.0.1:9000/
- 商城页面：http://127.0.0.1:9000/ecommerce
- API 文档：http://127.0.0.1:9000/docs

## 可选服务

### Redis

用于语义缓存和高频相似问题加速。没有 Redis 时，核心聊天仍然可用，但缓存能力不可用。

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Ollama

用于本地 embedding 或本地模型推理。

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
```

### Neo4j

用于图谱查询、Text2Cypher 和多跳关系分析。

```env
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=你的 Neo4j 密码
NEO4J_DATABASE=neo4j
```

### GraphRAG

用于文档、实体、关系和社区摘要结合的知识增强问答。

```env
GRAPHRAG_PROJECT_DIR=llm_backend/app/graphrag
GRAPHRAG_DATA_DIR=data
GRAPHRAG_QUERY_TYPE=local
```

## 项目展示建议

简历中可以写：

> 面向日常购物软件中 AI 客服答复效率低、上下文不连续、售后判断复杂等问题，设计并实现 ShopAssist Agent 智能客服系统，支持订单查询、退款退货判断、商品咨询、图片理解、知识库问答和人工介入判断，提升客服回复准确性和复杂售后问题处理效率。

## 注意事项

- 不要上传真实 `.env`、API Key、数据库密码和用户文件。
- `.venv/`、uploads、logs、GraphRAG 缓存和本地数据库不应提交到 GitHub。
- 如果聊天很慢，优先检查 Redis / Ollama / embedding 服务是否未启动导致超时。
- 如果图片理解失败，检查视觉模型 API Key、图片上传接口和前端粘贴图片逻辑。

## 仓库地址

https://github.com/g3182479125-hub/shopassist-agent
