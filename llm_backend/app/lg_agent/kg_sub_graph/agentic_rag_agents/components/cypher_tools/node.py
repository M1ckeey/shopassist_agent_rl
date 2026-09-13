from typing import Any, Callable, Coroutine, Dict, List
import asyncio
import os
from pathlib import Path
from pydantic import BaseModel, Field

# 导入GraphRAG相关模块
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from app.core.logger import get_logger
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from app.core.config import settings, ServiceType
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.northwind_retriever import NorthwindCypherRetriever
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.cypher_tools.utils import create_text2cypher_generation_node, create_text2cypher_validation_node, create_text2cypher_execution_node



# 获取日志记录器
logger = get_logger(service="cypher_tools")

# 定义GraphRAG查询的输入状态类型
class CypherQueryInputState(BaseModel):
    task: str
    query: str
    steps: List[str]

# 定义GraphRAG查询的输出状态类型
class CypherQueryOutputState(BaseModel):
    task: str
    query: str
    errors: List[str]
    records: Dict[str, Any]
    steps: List[str]

# 定义GraphRAG API包装器

def create_cypher_query_node(
) -> Callable[
    [CypherQueryInputState],
    Coroutine[Any, Any, Dict[str, List[CypherQueryOutputState] | List[str]]],
]:
    """
    创建 Text2Cypher 查询节点，用于LangGraph工作流。

    返回
    -------
    Callable[[CypherQueryInputState], Dict[str, List[CypherQueryOutputState] | List[str]]]
        名为`cypher_query`的LangGraph节点。
    """

    async def cypher_query(
        state: Dict[str, Any],
    ) -> Dict[str, List[CypherQueryOutputState] | List[str]]:
        """
        执行Text2Cypher查询并返回结果。
        """
        errors = list()
        # 获取查询文本
        query = state.get("task", "")
        if not query:
            errors.append("未提供查询文本")
 
        # 使用大模型执行查询/多跳/并行查询计划
        # 1. 根据.env文件中AGENT_SERVICE的设置，选择使用DeepSeek或Ollama启动的模型服务
        if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            model = ChatOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                model=settings.DEEPSEEK_MODEL,
                temperature=0.7,
                tags=["research_plan"],
            )
        else:
            model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.7, tags=["research_plan"])

        # 2. 获取Neo4j图数据库连接
        #    失败必须中断返回，否则后续所有 neo4j_graph 引用都会 NameError
        try:
            neo4j_graph = get_neo4j_graph()
            logger.info("success to get Neo4j graph database connection")
        except Exception as e:
            logger.error(f"failed to get Neo4j graph database connection: {e}")
            return {
                "cyphers": [
                    CypherQueryOutputState(
                        **{
                            "task": state.get("task", ""),
                            "query": query,
                            "errors": errors + [f"Neo4j 连接失败: {e}"],
                            "records": {"result": []},
                            "steps": ["execute_cypher_query"],
                        }
                    )
                ],
                "steps": ["execute_cypher_query"],
            }

        # step 2. 创建自定义检索器实例，根据 Graph Schema 创建 Cypher 示例，用来引导大模型生成正确的Cypher 查询语句
        cypher_retriever = NorthwindCypherRetriever()

        # Step 3.根据自定义的 Cypher 示例，引导大模型生成 当前输入 问题的 Cypher 查询语句
        cypher_generation = create_text2cypher_generation_node(
            llm=model, graph=neo4j_graph, cypher_example_retriever=cypher_retriever
        )

        cypher_result = await cypher_generation(state)

        # step 4. 验证生成的 Cypher 查询语句是否正确
        validate_cypher = create_text2cypher_validation_node(
            llm=model,
            graph=neo4j_graph,
            llm_validation=True,
            cypher_statement=cypher_result
        )

        # step 5. 获取执行Cypher查询的全部信息
        execute_info = await validate_cypher(state=state)

        # step 6. 执行 Cypher 查询语句
        execute_cypher = create_text2cypher_execution_node(
            graph=neo4j_graph, cypher=execute_info
        )

        final_result = await execute_cypher(state)

        # 封装 单次子任务执行的 输出结果并通过Pydantic模型限定格式
        return {
            "cyphers": [
                CypherQueryOutputState(
                        **{
                            "task": state.get("task", ""),
                            "query": query,
                            "statement": "",
                            "parameters":"",
                            "errors": errors,
                            "records": {"result": final_result["cyphers"][0]["records"]} if final_result.get("cyphers") and len(final_result["cyphers"]) > 0 else {"result": []},
                            "steps": ["execute_cypher_query"],
                        }
                    )
                ],
                "steps": ["execute_cypher_query"],
            }
  
    return cypher_query

