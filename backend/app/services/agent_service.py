import json
import os
import asyncio
import logging
from copy import copy
from datetime import datetime
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from sqlalchemy.exc import IntegrityError
from backend.app.tools import (
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
from backend.app.core.cache import cache
from backend.app.core.request_context import get_or_create_user
from backend.app.db.session import SessionLocal
from backend.app.db.models import User, ChatSession, ChatMessage, MCPServerConfig
from backend.app.skills.skill_loader import get_skill_index_text
from backend.app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

API_KEY = LLM_API_KEY
BASE_URL = LLM_BASE_URL
MODEL = LLM_MODEL
logger = logging.getLogger(__name__)

_MCP_CLIENT = None
_MCP_TOOLS_CACHE = None
_MCP_CLIENT_LOCK = None
LOCAL_TOOL_SOURCE_LABEL = "LOCAL"
MCP_TOOL_SOURCE_LABEL = "MCP"
CHAT_CONCURRENCY_ERROR_CODE = "chat_concurrency_conflict"


class ChatConcurrencyConflictError(RuntimeError):
    pass

# 尽量复制一份工具对象，避免后面对工具打标签时直接改到原对象
def _clone_tool(tool):
    if hasattr(tool, "model_copy"):
        return tool.model_copy(deep=True)
    if hasattr(tool, "copy"):
        try:
            return tool.copy(deep=True)
        except TypeError:
            return tool.copy()
    return copy(tool)

# 把工具的描述文本统一补上来源标签：[LOCAL]或[MCP]
def _format_tool_description_with_source(description: str | None, source_label: str) -> str:
    prefix = f"[{source_label}]"
    normalized = str(description or "").strip()
    if normalized.startswith(prefix):
        return normalized
    if not normalized:
        return f"{prefix} No description provided."
    return f"{prefix} {normalized}"

# 给一个运行时工具打上“来源标签”，让后面的智能体能区分它是本地工具还是MCP工具
def _tag_runtime_tool(tool, source_label: str):
    tagged_tool = _clone_tool(tool)

    try:
        tagged_tool.description = _format_tool_description_with_source(
            getattr(tagged_tool, "description", ""),
            source_label,
        )
    except Exception:
        logger.warning("无法为工具 %s 写入来源标签描述。", getattr(tool, "name", repr(tool)))

    metadata = getattr(tagged_tool, "metadata", None)
    tagged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    tagged_metadata["tool_source"] = source_label
    try:
        tagged_tool.metadata = tagged_metadata
    except Exception:
        logger.warning("无法为工具 %s 写入来源标签元数据。", getattr(tool, "name", repr(tool)))

    return tagged_tool

# 从一个工具对象里判断它的来源标签，返回"LOCAL"或“MCP”
def _get_tool_source_label(tool) -> str:
    metadata = getattr(tool, "metadata", None)
    if isinstance(metadata, dict):
        source_label = str(metadata.get("tool_source", "")).strip().upper()   # 取出tool_source，保证是字符串，去掉空格并转成大写
        if source_label in {LOCAL_TOOL_SOURCE_LABEL, MCP_TOOL_SOURCE_LABEL}:
            return source_label

    description = str(getattr(tool, "description", "") or "")
    if description.startswith(f"[{MCP_TOOL_SOURCE_LABEL}]"):
        return MCP_TOOL_SOURCE_LABEL
    return LOCAL_TOOL_SOURCE_LABEL

# 把当前运行时工具列表按来源分组，整理成一段可直接塞进系统提示词的文本
def _format_runtime_tool_source_text(runtime_tools: list | None = None) -> str:
    local_tool_names = []   # 存本地工具名
    mcp_tool_names = []     # 存MCP工具名

    for tool in runtime_tools or []:
        tool_name = getattr(tool, "name", None) or repr(tool)
        if _get_tool_source_label(tool) == MCP_TOOL_SOURCE_LABEL:
            mcp_tool_names.append(str(tool_name))
        else:
            local_tool_names.append(str(tool_name))

    local_text = ", ".join(local_tool_names) if local_tool_names else "无"
    mcp_text = ", ".join(mcp_tool_names) if mcp_tool_names else "无"
    return (
        "【运行时工具来源】\n"
        f"- 本地内置工具 [LOCAL]: {local_text}\n"
        f"- 外部 MCP 工具 [MCP]: {mcp_text}\n"
        "工具描述里会带有 [LOCAL] 或 [MCP] 标签，回答工具清单类问题时必须严格按来源标签筛选。\n"
    )


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


def build_agent_system_prompt(skill_context: dict | None = None, runtime_tools: list | None = None) -> str:
    skill_index_text = get_skill_index_text()
    return (
        "你是一个专业、友好、稳健的 AI 智能客服。\n\n"
        "你的主要职责是理解用户诉求、补齐关键信息、提供清晰建议，并在不确定时使用知识库、skill 资源或外部工具提高准确性。\n\n"
        "【可用 Skill 索引】\n"
        "以下 skill 可按需加载完整内容。你也可以先调用 `list_skills` 查看当前可用 skill。请先根据 skill 的 name 和 description 判断是否匹配用户任务；"
        "如果匹配，请调用 `load_skill` 读取主 SKILL.md。若正文要求进一步读取 references、assets 等附带文档，"
        "请先调用 `list_skill_resources` 查看，再用 `load_skill_resource` 按需读取。\n"
        f"{skill_index_text}\n\n"
        f"{_format_skill_context_text(skill_context)}\n"
        f"{_format_runtime_tool_source_text(runtime_tools)}\n"
        "【工具使用规则】\n"
        "1. 你可以先调用 `list_skills` 查看 skill 清单，再决定是否加载具体 skill。\n"
        "2. 用户任务符合某个 skill 的 description 时，再调用 `load_skill`。\n"
        "3. 不要预设自己已经知道 skill 的完整内容；需要时再读取。\n"
        "4. 如果某个 skill 已经在当前会话中激活，且资源足够支撑回答，优先复用 skill_context。\n"
        "5. 当答案依赖事实、规则、产品说明或知识库证据时，可调用 `search_knowledge_base`。\n"
        "6. 每轮对话最多调用一次 `search_knowledge_base`；拿到结果后直接给出最终回答。\n"
        "7. 运行时工具分为 [LOCAL] 本地内置工具 与 [MCP] 外部 MCP 工具；必须根据工具描述中的来源标签区分，不要混淆。\n"
        "8. 当用户询问“MCP 工具、MCP 能力、外部工具”时，只能回答带 [MCP] 标签的工具；当用户询问本地工具、知识库工具、skill 工具时，只能回答带 [LOCAL] 标签的工具。\n"
        "9. 当任务需要调用外部系统能力时，只选择带 [MCP] 标签的工具；当任务需要知识库或 skill 能力时，优先选择带 [LOCAL] 标签的工具。\n"
        "10. 如果证据不足，不要编造事实，要明确说明限制并给出下一步建议。\n"
        "11. 如果当前系统没有真实执行业务的工具，就不能假装已经完成退款、发货、建单或人工升级。\n"
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

    @staticmethod
    def _serialize_message_record(row: ChatMessage) -> dict:
        return {
            "type": row.message_type,
            "content": row.content,
            "timestamp": row.timestamp.isoformat(),
            "rag_trace": row.rag_trace,
        }
    
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
            user = get_or_create_user(db, user_id)
            
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

    def get_session_snapshot(self, user_id: str, session_id: str) -> tuple[list, dict, datetime | None]:
        db = SessionLocal()
        try:
            user = get_or_create_user(db, user_id)

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return [], {}, None

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            records = [self._serialize_message_record(row) for row in rows]
            metadata = session.metadata_json or {}
            return self._to_langchain_messages(records), metadata, session.updated_at
        finally:
            db.close()

    def append_exchange(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        ai_text: str,
        *,
        metadata: dict | None = None,
        rag_trace: dict | None = None,
        expected_updated_at: datetime | None = None,
    ) -> None:
        db = SessionLocal()
        try:
            user = get_or_create_user(db, user_id)

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            version_conflict = expected_updated_at is None and session is not None
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    session = (
                        db.query(ChatSession)
                        .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                        .first()
                    )
                    if not session:
                        raise
                    version_conflict = True

            if metadata is not None:
                if version_conflict or (expected_updated_at is not None and session.updated_at != expected_updated_at):
                    raise ChatConcurrencyConflictError(
                        f"{CHAT_CONCURRENCY_ERROR_CODE}: session updated concurrently; expected={expected_updated_at}, actual={session.updated_at}"
                    )
                session.metadata_json = metadata

            now = datetime.utcnow()
            db.add(
                ChatMessage(
                    session_ref_id=session.id,
                    message_type="human",
                    content=str(user_text),
                    timestamp=now,
                    rag_trace=None,
                )
            )
            db.add(
                ChatMessage(
                    session_ref_id=session.id,
                    message_type="ai",
                    content=str(ai_text),
                    timestamp=now,
                    rag_trace=rag_trace,
                )
            )
            session.updated_at = now
            db.commit()

            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._metadata_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()
    
    # 列出用户的所有会话ID
    def list_sessions(self, user_id: str) -> list:
        return [item["session_id"] for item in self.list_session_infos(user_id)]
    
    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached
        
        db = SessionLocal()
        try:
            user = get_or_create_user(db, user_id)
            
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
            user = get_or_create_user(db, user_id)
            
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
            user = get_or_create_user(db, user_id)

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
            user = get_or_create_user(db, user_id)
            
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

# 按需创建一个全局异步锁，保证并发情况下，MCP client和工具缓存只被安全初始化一次
def _get_mcp_client_lock() -> asyncio.Lock:
    global _MCP_CLIENT_LOCK
    if _MCP_CLIENT_LOCK is None:
        _MCP_CLIENT_LOCK = asyncio.Lock()
    return _MCP_CLIENT_LOCK

# 把"环境变量里的JSON字符串"读出来，解析成Python对象，并且校验类型对不对
def _load_json_env(name: str, expected_type: type, default):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    # 把JSON字符串解析成Python对象，如果字符串不是合法JSON，就捕获异常
    try:
        parsed = json.loads(raw_value)   
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"环境变量 {name} 不是合法 JSON: {exc}") from exc

    if not isinstance(parsed, expected_type):
        raise RuntimeError(f"环境变量 {name} 必须是 {expected_type.__name__} 类型的 JSON")
    return parsed

# 运行时缓存失效函数
def invalidate_mcp_runtime_cache():
    global _MCP_CLIENT, _MCP_TOOLS_CACHE
    _MCP_CLIENT = None
    _MCP_TOOLS_CACHE = None

# 把MCP配置里的“占位符字符串”替换成当前这条配置对应的真实值
def _render_mcp_template(value: str, config: MCPServerConfig) -> str:
    rendered = str(value or "")
    replacements = {
        "{asset_dir}": config.uploaded_asset_dir or "",
        "{asset_path}": config.uploaded_asset_path or "",
        "{config_name}": config.name or "",
        "{config_id}": str(config.id),
    }
    # 依次遍历每个占位符，把字符串里的占位符替换成真实值
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)
    return rendered

# 把数据库里“管理员配置好的MCP Server记录”读出来，转换成MultiServerMCPClient能直接使用的连接配置字典
def _build_db_mcp_connections() -> dict[str, dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(MCPServerConfig)
            .filter(MCPServerConfig.enabled == True)
            .order_by(MCPServerConfig.id.asc())
            .all()
        )
        connections: dict[str, dict] = {}   # 初始化结果字典
        for row in rows:
            connection_key = f"db_{row.id}_{row.transport}"
            if row.transport == "stdio":
                command = _render_mcp_template(row.command, row).strip()
                if not command:
                    logger.warning("MCP配置 %s 缺少 command，已跳过。", row.name)
                    continue

                args_json = row.args_json if isinstance(row.args_json, list) else []
                env_json = row.env_json if isinstance(row.env_json, dict) else {}
                connection = {
                    "transport": "stdio",
                    "command": command,
                    "args": [_render_mcp_template(str(item), row) for item in args_json],
                }
                if env_json:
                    connection["env"] = {
                        str(key): _render_mcp_template(str(value), row)
                        for key, value in env_json.items()
                    }
                connections[connection_key] = connection
            elif row.transport == "http":
                url = (row.url or "").strip()
                if not url:
                    logger.warning("MCP配置 %s 缺少 url，已跳过。", row.name)
                    continue
                headers_json = row.headers_json if isinstance(row.headers_json, dict) else {}
                connection = {
                    "transport": "http",
                    "url": url,
                }
                if headers_json:
                    connection["headers"] = {str(key): str(value) for key, value in headers_json.items()}
                connections[connection_key] = connection
        return connections
    finally:
        db.close()

# 从环境变量里读取MCP配置，拼出MultiServerMCPClient需要的连接字典
def _build_mcp_connections() -> dict[str, dict]:
    connections = _build_db_mcp_connections()

    # 构建stdio连接
    stdio_command = os.getenv("MCP_STDIO_COMMAND", "").strip()
    if stdio_command:
        stdio_connection = {
            "transport": "stdio",
            "command": stdio_command,
            "args": _load_json_env("MCP_STDIO_ARGS", list, []),
        }
        stdio_env = _load_json_env("MCP_STDIO_ENV", dict, {})
        if stdio_env:
            stdio_connection["env"] = {str(key): str(value) for key, value in stdio_env.items()}
        connections.setdefault("env_local_stdio", stdio_connection)

    # 构建http连接
    http_url = os.getenv("MCP_HTTP_URL", "").strip()   # 查看有没有配置远程MCP的URL
    if http_url:
        http_connection = {
            "transport": "http",
            "url": http_url,
        }
        headers = _load_json_env("MCP_HTTP_HEADERS", dict, {})   # 读取自定义请求头
        bearer_token = os.getenv("MCP_HTTP_BEARER_TOKEN", "").strip()   # 读取Bearer Token
        if bearer_token:
            headers.setdefault("Authorization", f"Bearer {bearer_token}")   # 如果headers里已经有Authorization，就不覆盖；如果没有才自动加上
        if headers:
            http_connection["headers"] = {str(key): str(value) for key, value in headers.items()}
        connections.setdefault("env_remote_http", http_connection)

    return connections

# 按需获取一个全局唯一的MCP client，如果还没创建就安全地创建一次
async def _get_mcp_client():
    connections = _build_mcp_connections()
    if not connections:
        return None

    # 看全局缓存中有没有现成的client，有直接复用，不再重新创建
    global _MCP_CLIENT
    if _MCP_CLIENT is not None:
        return _MCP_CLIENT

    # 加锁，只有一个协程能进入创建逻辑
    async with _get_mcp_client_lock():
        if _MCP_CLIENT is None:
            _MCP_CLIENT = MultiServerMCPClient(connections)
            logger.info("已初始化MCP客户端，连接数: %s", len(connections))
    return _MCP_CLIENT

# 获取MCP工具列表，并且把结果缓存起来，避免每次对话都重新去远程拉工具定义
async def _get_mcp_tools() -> list:
    # 先看缓存有没有值，如果之前已经加载过工具，就直接返回
    global _MCP_TOOLS_CACHE
    if _MCP_TOOLS_CACHE is not None:
        return list(_MCP_TOOLS_CACHE)

    client = await _get_mcp_client()
    if client is None:
        return []

    async with _get_mcp_client_lock():
        if _MCP_TOOLS_CACHE is None:
            try:
                _MCP_TOOLS_CACHE = await client.get_tools()
                logger.info("已加载 %s 个 MCP 工具。", len(_MCP_TOOLS_CACHE))
            except Exception:
                logger.exception("加载 MCP 工具失败，将回退到仅使用本地工具。")
                _MCP_TOOLS_CACHE = []
    return list(_MCP_TOOLS_CACHE)


def _merge_runtime_tools(local_tools: list, remote_tools: list) -> list:
    merged = []
    seen_names = set()

    for tool in [*local_tools, *remote_tools]:
        tool_name = getattr(tool, "name", None) or repr(tool)
        if tool_name in seen_names:
            logger.warning("检测到重名工具 %s，已跳过后续重复项。", tool_name)
            continue
        seen_names.add(tool_name)
        merged.append(tool)

    return merged


async def build_runtime_tools() -> list:
    local_tools = [_tag_runtime_tool(tool, LOCAL_TOOL_SOURCE_LABEL) for tool in LOCAL_TOOLS]
    mcp_tools = [_tag_runtime_tool(tool, MCP_TOOL_SOURCE_LABEL) for tool in await _get_mcp_tools()]
    return _merge_runtime_tools(local_tools, mcp_tools)


async def create_agent_instance(skill_context: dict | None = None):
    model = get_chat_model()
    runtime_tools = await build_runtime_tools()
    agent = create_agent(
        model = model,
        tools = runtime_tools,
        system_prompt = build_agent_system_prompt(skill_context, runtime_tools),
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
async def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    messages, session_metadata, session_version = storage.get_session_snapshot(user_id, session_id)
    skill_context = session_metadata.get("skill_context") if isinstance(session_metadata, dict) else {}

    get_last_rag_context(clear=True)
    initialize_skill_context(skill_context)
    reset_tool_call_guards()

    if len(messages) > 50:
        summary = await summarize_old_messages(get_chat_model(), messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    runtime_agent, _ = await create_agent_instance(skill_context)
    messages.append(HumanMessage(content=user_text))
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

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    updated_skill_context = get_last_skill_context(clear=True) or skill_context
    updated_metadata = dict(session_metadata or {})
    updated_metadata["skill_context"] = updated_skill_context

    storage.append_exchange(
        user_id,
        session_id,
        user_text,
        response_content,
        metadata=updated_metadata,
        rag_trace=rag_trace,
        expected_updated_at=session_version,
    )

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


""" 
使用Agent处理用户消息并流式返回响应。
架构：使用统一输出队列+后台任务,确保RAG检索步骤在工具执行期间实时推送,而非等待工具完成后才显示。
"""
async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    # 读取当前会话历史、metadata和版本号
    messages, session_metadata, session_version = storage.get_session_snapshot(user_id, session_id)
    skill_context = session_metadata.get("skill_context") if isinstance(session_metadata, dict) else {}

    # 清理上一轮可能残留的RAG/工具状态，并初始化本轮skill上下文
    get_last_rag_context(clear=True)
    initialize_skill_context(skill_context)
    reset_tool_call_guards()

    output_queue = asyncio.Queue()   # 流式输出的中转队列，后台Agent往里面放，主循环从里面取，然后yield给前端
    captured_rag_trace = [None]
    stream_state = {"status": "running"}

    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})
    set_rag_step_queue(_RagStepProxy())

    # 如果历史消息太长，就把前40条压缩成摘要，减少上下文长度，避免prompt过长
    if len(messages) > 50:
        summary = await summarize_old_messages(get_chat_model(), messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    # 创建本轮Agent，并把用户当前输入追加到上下文里
    runtime_agent, _ = await create_agent_instance(skill_context)
    messages.append(HumanMessage(content=user_text))

    full_response = ""

    # 后台任务
    async def _agent_worker():
        nonlocal full_response
        try:
            async for msg, metadata in runtime_agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
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

            stream_state["status"] = "completed"
        except asyncio.CancelledError:
            stream_state["status"] = "cancelled"
            raise
        except Exception as e:
            stream_state["status"] = "failed"
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            rag_context = get_last_rag_context(clear=True)
            if rag_context:
                captured_rag_trace[0] = rag_context.get("rag_trace")
            await output_queue.put(None)

    # 把Agent执行放到后台任务里。他做的事是：async for msg, metadata in runtime_agent.astream(...):
    # 从LangChain Agent中持续读取流式消息
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环输出SSE
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    # 如果客户端中途断开连接，比如前端取消请求，就取消后台 Agent 任务，避免它继续消耗模型资源
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    rag_trace = captured_rag_trace[0]

    # 如果Agent没正常完成，就不保存本轮对话
    if stream_state["status"] != "completed":
        return

    yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"
    yield "data: [DONE]\n\n"

    updated_skill_context = get_last_skill_context(clear=True) or skill_context
    updated_metadata = dict(session_metadata or {})
    updated_metadata["skill_context"] = updated_skill_context

    """
    乐观锁：
        我开始生成回答时看到的会话版本是session_version, 只有数据库里现在还是这个版本, 才允许保存
    如果期间同一个会话被另一个请求写过，append_exchange()会抛并发冲突，避免把基于旧上下文生成的回答写进新历史
    """
    storage.append_exchange(
        user_id,
        session_id,
        user_text,
        full_response,
        metadata=updated_metadata,
        rag_trace=rag_trace,
        expected_updated_at=session_version,
    )
