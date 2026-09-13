from typing import List, Dict, AsyncGenerator, Callable, Optional
from openai import AsyncOpenAI
from app.core.config import settings
import json
from app.core.logger import get_logger
from app.core.database import AsyncSessionLocal
from app.models.conversation import Conversation, DialogueType
from app.models.message import Message
from app.services.redis_semantic_cache import RedisSemanticCache
from app.services.conversation_service import ConversationService
from app.services.order_lookup_service import OrderLookupService
import time
import asyncio

logger = get_logger(service="deepseek")

SHOPCARE_SYSTEM_PROMPT = """你是 AssistGen 商城在线客服，只处理和商城真实业务相关的问题。

业务范围：
商品咨询、商品参数、价格、库存、下单、支付、发票、订单查询、物流进度、收货地址、退货、换货、退款、售后、投诉、账号和平台服务。

回复规则：
1. 只按商城客服身份回答，不要说自己能回答科学、数学、历史、翻译、写作、编程等通用问题。
2. 如果用户问“你能做什么”“你能来做什么”，只介绍商城客服能力。
3. 如果用户问题不属于商城业务，简短拒答，并引导回商品、订单、物流、退换货、售后等问题。
4. 遇到“它、这个、那个、他”等指代词，要结合最近上下文判断指的是什么；上下文不够就追问。
5. 不要编造订单、物流、库存、退款进度；缺少订单号或商品信息时直接问用户补充。
6. 退货/退款/换货问题缺少订单号、商品状态或原因时，不要列政策清单，直接问缺什么。
7. 不要说“您好”“感谢咨询”“竭诚服务”等客套话，不要写总结式结尾。
8. 回复尽量简短、口语化，优先给结论，最多 2 句话。
"""

class DeepseekService:
    def __init__(self, model: str = "deepseek-chat"):
        logger.info("Initializing Deepseek Service")
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        # 优先使用配置中的 DEEPSEEK_MODEL，其次使用传入的 model
        self.model = settings.DEEPSEEK_MODEL or model 
        self.cache = RedisSemanticCache(prefix="shopcare_chat_v2")

    def _get_last_user_message(self, messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _should_use_semantic_cache(self, messages: List[Dict]) -> bool:
        """带指代词的问题强依赖上下文，不能用相似问题缓存兜答案。"""
        last_user_message = self._get_last_user_message(messages)
        context_terms = ("它", "他", "这个", "那个", "这款", "该商品", "上面", "前面", "刚才", "之前")
        return (
            not any(term in last_user_message for term in context_terms)
            and OrderLookupService.extract_order_id(last_user_message) is None
        )

    def _validate_order_if_present(self, messages: List[Dict]) -> tuple[Optional[str], Optional[str]]:
        """如果本轮消息提供了订单号，先做真实订单校验。"""
        last_user_message = self._get_last_user_message(messages)
        order_id = OrderLookupService.extract_order_id(last_user_message)
        if not order_id:
            return None, None

        order = OrderLookupService.find_order(order_id)
        if not order:
            return order_id, None

        return order_id, OrderLookupService.order_summary(order)

    async def _build_effective_messages(
        self,
        messages: List[Dict],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """为客服聊天补齐系统约束和最近上下文。"""
        cleaned_messages = [
            {"role": msg.get("role", "user"), "content": str(msg.get("content", ""))}
            for msg in messages
            if msg.get("content")
        ]

        has_system = any(msg.get("role") == "system" for msg in cleaned_messages)
        if has_system:
            base_messages = cleaned_messages
        else:
            base_messages = [{"role": "system", "content": SHOPCARE_SYSTEM_PROMPT}]

            # 如果前端只传了本轮一句话，就从数据库补最近历史；如果前端已传完整历史，则直接用前端历史。
            if len(cleaned_messages) <= 1 and conversation_id is not None:
                recent_messages = await ConversationService.get_recent_openai_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    limit=12
                )
                base_messages.extend(recent_messages)

            base_messages.extend(cleaned_messages[-12:])

        order_id, order_context = self._validate_order_if_present(cleaned_messages)
        if order_context:
            base_messages.insert(
                1,
                {
                    "role": "system",
                    "content": (
                        "以下是已通过真实订单数据校验的订单信息，只能基于这些信息回答，"
                        "不要编造其他订单状态：\n" + order_context
                    ),
                },
            )

        # 控制上下文长度，保留 system 和最近消息，避免超出模型窗口。
        system_messages = [msg for msg in base_messages if msg.get("role") == "system"]
        normal_messages = [msg for msg in base_messages if msg.get("role") != "system"]
        return system_messages[:3] + normal_messages[-12:]

    async def _stream_cached_response(self, response: str, delay: float = 0.05) -> AsyncGenerator[str, None]:
        """模拟流式返回缓存的响应"""
        # 每次返回4个字符
        chunks = [response[i:i + 4] for i in range(0, len(response), 4)]
        for chunk in chunks:
            await asyncio.sleep(delay)  # 50ms延迟
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    async def generate_stream(
        self, 
        messages: List[Dict],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable[[int, int, List[Dict], str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成回复"""
        try:
            original_messages = messages
            order_id, order_context = self._validate_order_if_present(original_messages)
            if order_id and not order_context:
                fallback_response = f"没查到订单 {order_id}。你再核对一下订单号，或者发截图我帮你看。"
                async for chunk in self._stream_cached_response(fallback_response, delay=0.01):
                    yield chunk
                if on_complete and user_id is not None and conversation_id is not None:
                    await on_complete(user_id, conversation_id, original_messages, fallback_response)
                return

            effective_messages = await self._build_effective_messages(
                messages=messages,
                user_id=user_id,
                conversation_id=conversation_id
            )
            use_cache = self._should_use_semantic_cache(original_messages)

            # 为每个用户创建独立的缓存实例
            cache = RedisSemanticCache(prefix="shopcare_chat_v2", user_id=user_id)
            
            start_time = time.time()
            
            # 检查缓存
            cached_response = await cache.lookup(effective_messages) if use_cache else None
            if cached_response:
                response_time = time.time() - start_time
                logger.info(f"Cache hit! Response time: {response_time:.4f} seconds")
                
                # 模拟流式返回，因为速率太快了
                async for chunk in self._stream_cached_response(cached_response):
                    yield chunk
                
                if on_complete and user_id is not None and conversation_id is not None:
                    await on_complete(user_id, conversation_id, original_messages, cached_response)
                return

            # 缓存未命中,调用API
            full_response = []
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=effective_messages,
                stream=True,
                temperature=0.7,
                max_tokens=180
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 使用 ensure_ascii=False 来保持中文字符
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
            
            # 完整响应
            complete_response = "".join(full_response)
            
            # 更新缓存
            if use_cache:
                await cache.update(effective_messages, complete_response)
            
            response_time = time.time() - start_time
            logger.info(f"Cache miss. Response time: {response_time:.4f} seconds")
            
            # 如果有回调，执行回调
            if on_complete and user_id is not None and conversation_id is not None:
                await on_complete(user_id, conversation_id, original_messages, complete_response)
                
        except Exception as e:
            logger.error(f"Error in generate_stream: {str(e)}", exc_info=True)
            error_msg = json.dumps(f"生成回复时出错: {str(e)}", ensure_ascii=False)
            yield f"data: {error_msg}\n\n"

    async def generate(self, messages: List[Dict]) -> str:
        """非流式生成回复"""
        try:
            effective_messages = await self._build_effective_messages(messages)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=effective_messages,
                stream=False,
                temperature=0.7,
                max_tokens=180
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Generation error: {str(e)}")
            raise
