from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator, Dict, List
import json
import re
import uuid

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="rag_chat")


class RAGChatService:
    """Lightweight RAG chat over the generated GraphRAG parquet outputs.

    This avoids importing the full GraphRAG indexing stack at request time,
    which pulls in spaCy/torch on Windows and can fail before the API responds.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.model = settings.DEEPSEEK_MODEL

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        index_id: str | None = None,
        user_id: int | None = None,
    ) -> AsyncGenerator[str, None]:
        query = self._last_user_message(messages)
        if not query:
            yield f"data: {json.dumps('请先输入要查询的问题。', ensure_ascii=False)}\n\n"
            return

        try:
            logger.info(f"Running lightweight RAG chat for index_id={index_id or 'default'}")
            contexts = self._retrieve_context(query, user_id=user_id)
            answer = await self._generate_answer(query, contexts)
        except Exception as exc:
            logger.error(f"RAG chat failed: {exc}", exc_info=True)
            answer = f"文档检索暂时失败：{exc}"

        for chunk in self._chunk_text(answer):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    async def _generate_answer(self, query: str, contexts: List[str]) -> str:
        context_text = "\n\n---\n\n".join(contexts)
        if not context_text:
            return "没有检索到相关文档内容。"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是知识库问答助手。只能基于给定资料回答；"
                        "如果资料不足，就说明没有足够依据。回答要简洁。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n\n资料：\n{context_text}",
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or "没有生成可用回答。"

    def _retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        user_id: int | None = None,
    ) -> List[str]:
        text_units, entities, reports = _load_rag_tables()
        query_terms = self._terms(query)

        scored: List[tuple[float, str]] = []
        for text in self._load_uploaded_texts(user_id):
            score = self._score(query_terms, text)
            if score > 0:
                scored.append((score + 1.0, text))
            elif query_terms and any(term in {"文", "档", "主要", "讲", "什么"} for term in query_terms):
                scored.append((0.8, text))

        for row in text_units:
            text = str(row.get("text", ""))
            score = self._score(query_terms, text)
            if score > 0:
                scored.append((score, text))

        for row in entities:
            text = f"{row.get('title', '')}: {row.get('description', '')}"
            score = self._score(query_terms, text) + 0.3
            if score > 0.3:
                scored.append((score, text))

        for row in reports:
            text = f"{row.get('title', '')}\n{row.get('summary', '')}"
            score = self._score(query_terms, text)
            if score > 0:
                scored.append((score, text))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored and text_units:
            return [str(text_units[0].get("text", ""))[:1800]]
        return [text[:1800] for _, text in scored[:top_k]]

    def _load_uploaded_texts(self, user_id: int | None) -> List[str]:
        if user_id is None:
            return []

        output_dir = Path(settings.GRAPHRAG_PROJECT_DIR) / settings.GRAPHRAG_DATA_DIR / "output"
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        manifest_path = output_dir / user_uuid / "lightweight_index.jsonl"
        if not manifest_path.exists():
            return []

        texts: List[str] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            file_path = Path(record.get("input_file_path", ""))
            if not file_path.exists():
                continue
            if file_path.suffix.lower() not in {".txt", ".md", ".csv"}:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if content:
                texts.append(f"上传文件：{file_path.name}\n{content}")

        return texts[-5:]

    @staticmethod
    def _score(query_terms: set[str], text: str) -> float:
        if not query_terms or not text:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for term in query_terms if term in text_lower)
        return hits / max(len(query_terms), 1)

    @staticmethod
    def _terms(text: str) -> set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-zA-Z0-9_]{2,}", lowered))
        cjk = {ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"}
        return words | cjk

    @staticmethod
    def _last_user_message(messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _chunk_text(text: str, size: int = 16) -> List[str]:
        return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@lru_cache(maxsize=1)
def _load_rag_tables() -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas is not installed; parquet RAG tables are disabled.")
        return [], [], []

    output_dir = Path(settings.GRAPHRAG_PROJECT_DIR) / settings.GRAPHRAG_DATA_DIR / "output"
    try:
        text_units = pd.read_parquet(output_dir / "text_units.parquet").to_dict("records")
        entities = pd.read_parquet(output_dir / "entities.parquet").to_dict("records")
        reports = pd.read_parquet(output_dir / "community_reports.parquet").to_dict("records")
    except Exception as exc:
        logger.warning(f"Failed to load parquet RAG tables: {exc}")
        return [], [], []

    return text_units, entities, reports
