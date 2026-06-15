import json
import asyncio
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage

from tools import (
    search_knowledge_base,
    list_skills,
    load_skill,
    list_skill_resources,
    load_skill_resource,
    get_last_rag_context,
    get_last_skill_context,
    initialize_skill_context,
    reset_tool_call_guards,
    set_rag_step_queue,
)
from datetime import datetime
from cache import cache
from database import SessionLocal
from models import User, ChatSession, ChatMessage
from skill_loader import get_skill_index_text
from settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

API_KEY = LLM_API_KEY
BASE_URL = LLM_BASE_URL
MODEL = LLM_MODEL


def _format_skill_context_text(skill_context: dict | None) -> str:
    if not isinstance(skill_context, dict):
        return ""

    active_skill = skill_context.get("active_skill") or "无"
    loaded_resources = skill_context.get("loaded_resources") or []
    last_tool = skill_context.get("last_tool") or "无"
    resource_text = ", ".join(str(item) for item in loaded_resources) if loaded_resources else "无"
    return (
        "\n【当前 Skill 上下文】\n"
        f"- 当前激活 skill: {active_skill}\n"
        f"- 已读取资源: {resource_text}\n"
        f"- 最近一次 skill 工具: {last_tool}\n"
        "如果当前 skill 仍然适用，请优先复用已有上下文，避免重复加载。\n"
    )


def build_agent_system_prompt(skill_context: dict | None = None) -> str:
    skill_index_text = get_skill_index_text()
    return (
        "你是一个专业、友好、稳健的 AI 智能客服。\n\n"
        "你的主要职责是理解用户诉求、补齐关键信息、提供清晰建议，并在不确定时使用知识库或 skill 资源提高准确性。\n\n"
        "【可用 Skill 索引】\n"
        "以下 skill 可按需加载完整内容。你也可以先调用 `list_skills` 查看当前可用 skill。请先根据 skill 的 name 和 description 判断是否匹配用户任务；"
        "如果匹配，请调用 `load_skill` 读取主 SKILL.md。若正文要求进一步读取 references、assets 等附带文档，"
        "请先调用 `list_skill_resources` 查看，再用 `load_skill_resource` 按需读取。\n"
        f"{skill_index_text}\n\n"
        f"{_format_skill_context_text(skill_context)}\n"
        "【工具使用规则】\n"
        "1. 你可以先调用 `list_skills` 查看 skill 清单，再决定是否加载具体 skill。\n"
        "2. 用户任务符合某个 skill 的 description 时，再调用 `load_skill`。\n"
        "3. 不要预设自己已经知道 skill 的完整内容；需要时再读取。\n"
        "4. 如果某个 skill 已经在当前会话中激活，且资源足够支撑回答，优先复用 skill_context。\n"
        "5. 当答案依赖事实、规则、产品说明或知识库证据时，可调用 `search_knowledge_base`。\n"
        "6. 每轮对话最多调用一次 `search_knowledge_base`；拿到结果后直接给出最终回答。\n"
        "7. 如果证据不足，不要编造事实，要明确说明限制并给出下一步建议。\n"
        "8. 如果当前系统没有真实执行业务的工具，就不能假装已经完成退款、发货、建单或人工升级。\n"
    )

# 这个类用来管理对话的存储，包括将对话保存到数据库、从数据库加载对话、列出用户的会话列表，以及删除会话等功能。
# 它还负责将数据库中存储的原始聊天记录转换为LangChain消息格式，以便在对话过程中使用。
# 同时，它利用Redis缓存来提高对话加载的效率，减少数据库访问次数。
class ConversationStorage:

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"
    
    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _metadata_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_session_metadata:{user_id}:{session_id}"
    
    # 将数据库中存储的原始聊天记录，转换为标准的LangChain消息格式
    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content = content))   # 用户输入
            elif msg_type == "ai":
                messages.append(AIMessage(content = content))      # AI回复
            elif msg_type == "system":
                messages.append(SystemMessage(content = content))  # 系统提示词
        return messages
    
    # 将用户和AI的对话存进数据库，并更新Redis缓存（用户ID、会话ID、对话列表、额外配置、额外消息数据）
    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return
            
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id = user.id, session_id = session_id, metadata_json = metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata if metadata is not None else (session.metadata_json or {})

            # 删除原有的消息记录，重新插入新的消息列表（简单粗暴的做法，适合消息量不大的场景）
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session = False)

            # 遍历所有消息，逐条存入数据库
            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None

                # 如果有额外的消息数据，并且当前消息索引在范围内，则尝试获取RAG追踪信息
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
                
                # 把单条消息存入数据库表
                db.add(
                    ChatMessage(
                        session_ref_id = session.id,   
                        message_type = msg.type,
                        content = str(msg.content),
                        timestamp = now,
                        rag_trace = rag_trace,
                    )
                )

                # 同时构建一个序列化后的消息列表，用于更新Redis缓存（包含RAG追踪信息）
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )
            
            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.set_json(self._metadata_cache_key(user_id, session_id), session.metadata_json or {})
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    # 加载对话,优先从Redis缓存获取，如果没有再从数据库加载，并更新缓存（用户ID、会话ID）
    def load(self, user_id: str, session_id: str) -> list:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)
        
        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)
    
    # 列出用户的所有会话ID
    def list_sessions(self, user_id: str) -> list:
        return [item["session_id"] for item in self.list_session_infos(user_id)]
    
    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user: 
                return []
            
            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    # 从数据库中获取指定用户和会话的消息记录（原始格式，包含RAG追踪信息）
    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []
            
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def get_session_metadata(self, user_id: str, session_id: str) -> dict:
        cached = cache.get_json(self._metadata_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return {}

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return {}

            metadata = session.metadata_json or {}
            cache.set_json(self._metadata_cache_key(user_id, session_id), metadata)
            return metadata
        finally:
            db.close()

    # 删除指定用户的会话，返回是否删除成功
    def delete_session(self, user_id: str, session_id: str) -> bool:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False
            
            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._metadata_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()


@lru_cache(maxsize=1)
def get_chat_model():
    return init_chat_model(
        model = MODEL,
        model_provider = "openai",
        api_key = API_KEY,
        base_url = BASE_URL,
        temperature = 0.3,
        stream_usage = True,
    )

LOCAL_TOOLS = [
    search_knowledge_base,
    list_skills,
    load_skill,
    list_skill_resources,
    load_skill_resource,
]


async def build_runtime_tools() -> list:
    return list(LOCAL_TOOLS)


async def create_agent_instance(skill_context: dict | None = None):
    model = get_chat_model()
    runtime_tools = await build_runtime_tools()
    agent = create_agent(
        model = model,
        tools = runtime_tools,
        system_prompt = build_agent_system_prompt(skill_context),
    )

    return agent, model

storage = ConversationStorage()

# 将旧消息总结为摘要
async def summarize_old_messages(model, messages: list) -> str:
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：
                        {old_conversation}
                        总结（包含用户信息、重要事实、待办事项）：
                        """
    
    summary = (await model.ainvoke(summary_prompt)).content
    return summary

# 使用Agent处理用户消息并返回响应
async def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    messages = storage.load(user_id, session_id)
    session_metadata = storage.get_session_metadata(user_id, session_id)
    skill_context = session_metadata.get("skill_context") if isinstance(session_metadata, dict) else {}

    # 清理可能残留的RAG上下文，避免跨请求污染
    get_last_rag_context(clear = True)
    initialize_skill_context(skill_context)
    reset_tool_call_guards()

    if len(messages) > 50:
        summary = await summarize_old_messages(get_chat_model(), messages[:40])
        messages = [
            SystemMessage(content = f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    runtime_agent, _ = await create_agent_instance(skill_context)
    messages.append(HumanMessage(content = user_text))
    result = await runtime_agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    messages.append(AIMessage(content = response_content))

    rag_context = get_last_rag_context(clear = True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    updated_skill_context = get_last_skill_context(clear = True) or skill_context
    updated_metadata = dict(session_metadata or {})
    updated_metadata["skill_context"] = updated_skill_context

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata = updated_metadata, extra_message_data = extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }

""" 
使用Agent处理用户消息并流式返回响应。
架构：使用统一输出队列+后台任务,确保RAG检索步骤在工具执行期间实时推送,而非等待工具完成后才显示。
"""
# async def（异步函数）：可以暂停执行，遇到等待操作时，会把程序控制权交出去
async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    messages = storage.load(user_id, session_id)
    session_metadata = storage.get_session_metadata(user_id, session_id)
    skill_context = session_metadata.get("skill_context") if isinstance(session_metadata, dict) else {}

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear = True)
    initialize_skill_context(skill_context)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()
    captured_rag_trace = [None]  # 用列表避免closure问题，捕获最后的rag_trace

    class _RagStepProxy:
        """代理对象: 将emit_rag_step的原始step dict包装后放入统一输出队列"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = await summarize_old_messages(get_chat_model(), messages[:40])
        messages = [
            SystemMessage(content = f"之前的对话摘要：\n{summary}")
        ] + messages[40:]
    runtime_agent, _ = await create_agent_instance(skill_context)
    messages.append(HumanMessage(content = user_text))

    full_response = ""

    # 后台任务：运行agent并将内容chunk推入输出队列
    async def _agent_worker():
        nonlocal full_response   # Python关键字，用在「嵌套函数（函数套函数）」里，声明我要修改的是「外层函数的变量」，而不是当前函数的局部变量
        try:
            async for msg, metadata in runtime_agent.astream(
                {"messages": messages},
                stream_mode = "messages",
                config = {"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")
                
                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})

        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 在agent完成后立即捕获RAG trace，然后通知主循环完成
            rag_context = get_last_rag_context(clear=True)
            if rag_context:
                captured_rag_trace[0] = rag_context.get("rag_trace")
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并yield SSE
        # RAG步骤在工具执行期间通过call_soon_threadsafe实时入队，不需要等agent产出chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI会向此生成器抛出GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()   
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭 

    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取捕获到的RAG trace（已在_agent_worker中捕获）
    rag_trace = captured_rag_trace[0]

    # 始终发送 trace 信息（即使为空也要发送，便于前端处理）
    yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content = full_response))
    updated_skill_context = get_last_skill_context(clear = True) or skill_context
    updated_metadata = dict(session_metadata or {})
    updated_metadata["skill_context"] = updated_skill_context
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, metadata = updated_metadata, extra_message_data = extra_message_data)
