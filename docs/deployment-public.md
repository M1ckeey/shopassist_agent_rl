# ShopAssist Agent 公网部署说明

本项目可以通过 FastAPI 后端托管静态前端资源，部署后访问根路径即可打开页面。

## 推荐部署方式

优先使用支持 Docker 的云平台，例如 Render、Railway 或类似服务：

- 构建方式：Dockerfile
- 健康检查：`/health`
- 默认端口：读取平台注入的 `PORT`

## 必填环境变量

```env
DEEPSEEK_API_KEY=你的文本模型 API Key
VISION_API_KEY=你的视觉模型 API Key
SECRET_KEY=生产环境随机密钥
```

## 演示数据库

公网演示版本可以先使用 SQLite：

```env
DB_TYPE=sqlite
SQLITE_PATH=/app/data/shopassist.db
```

这样不依赖本地 MySQL，也能完成注册登录、会话创建和消息保存。正式长期运行建议换成云 MySQL，并设置：

```env
DATABASE_DSN=mysql+aiomysql://user:password@host:3306/database
```

`DATABASE_DSN` 优先级最高，设置后会覆盖 `DB_TYPE` 和本地 MySQL 配置。

## 可选服务

- Redis：用于语义缓存和高频问题加速。
- Neo4j：用于图谱查询和 Text2Cypher。
- Ollama：用于本地 embedding 或本地模型推理。

公网演示版本可以先不启用这些服务，相关功能会降级为普通大模型或本地轻量检索。

## Render 快速步骤

1. 在 Render 新建 Web Service。
2. 连接 GitHub 仓库 `g3182479125-hub/shopassist-agent`。
3. Environment 选择 Docker。
4. 填入必填环境变量。
5. 部署完成后访问 Render 分配的公网域名。

## Railway 快速步骤

1. 在 Railway 新建 Project。
2. 选择 Deploy from GitHub repo。
3. 选择 `g3182479125-hub/shopassist-agent`。
4. Railway 会读取 `Dockerfile` 和 `railway.json`。
5. 在 Variables 中填入必填环境变量。

## 说明

本仓库不会提交 `.env`、本地数据库、上传文件和 GraphRAG 运行缓存。线上环境变量需要在云平台控制台配置。
