import asyncio
from contextvars import ContextVar   # 保存“当前上下文变量”的工具
from copy import deepcopy
from typing import Any, Optional

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

from backend.app.skills.skill_loader import (
    list_skills_text,
    load_skill_content,
    list_skill_resources_content,
    load_skill_resource_content,
)

_LAST_RAG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "LAST_RAG_CONTEXT",   # 存储最近一次RAG检索的上下文
    default=None,
)
_LAST_SKILL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "LAST_SKILL_CONTEXT",   # 存储最近一次skill加载上下文
    default=None,
)
_KNOWLEDGE_TOOL_CALLS_THIS_TURN: ContextVar[int] = ContextVar(
    "KNOWLEDGE_TOOL_CALLS_THIS_TURN",   # 记录当前对话轮次中知识工具被调用的次数 
    default=0,
)
_RAG_STEP_QUEUE: ContextVar[Any] = ContextVar("RAG_STEP_QUEUE", default=None)   # 异步队列：给前端推送RAG检索进度
_RAG_STEP_LOOP: ContextVar[Any] = ContextVar("RAG_STEP_LOOP", default=None)     # 异步事件循环：保证跨线程安全推送信息


# 保存最近一次RAG检索的上下文信息
def _set_last_rag_context(context: dict):
    _LAST_RAG_CONTEXT.set(deepcopy(context))

# 获取最近一次保存的RAG检索上下文，默认读取后自动清空
def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    context = _LAST_RAG_CONTEXT.get()
    if clear:
        _LAST_RAG_CONTEXT.set(None)
    return deepcopy(context) if isinstance(context, dict) else context


def initialize_skill_context(context: Optional[dict] = None):
    default_context = {
        "active_skill": None,
        "loaded_resources": [],
        "last_tool": None,
    }
    merged = default_context.copy()
    if isinstance(context, dict):
        if context.get("active_skill"):
            merged["active_skill"] = context.get("active_skill")
        loaded_resources = context.get("loaded_resources")
        if isinstance(loaded_resources, list):
            merged["loaded_resources"] = [str(item) for item in loaded_resources if item]
        if context.get("last_tool"):
            merged["last_tool"] = context.get("last_tool")
    _LAST_SKILL_CONTEXT.set(merged)


def _update_skill_context(
    active_skill: Optional[str] = None,
    loaded_resource: Optional[str] = None,
    last_tool: Optional[str] = None,
):
    skill_context = _LAST_SKILL_CONTEXT.get()
    if skill_context is None:
        initialize_skill_context()
        skill_context = _LAST_SKILL_CONTEXT.get()

    skill_context = deepcopy(skill_context) if isinstance(skill_context, dict) else {}
    if active_skill:
        if skill_context.get("active_skill") != active_skill:
            skill_context["active_skill"] = active_skill
            skill_context["loaded_resources"] = []
    if loaded_resource:
        resources = skill_context.setdefault("loaded_resources", [])
        if loaded_resource not in resources:
            resources.append(loaded_resource)
    if last_tool:
        skill_context["last_tool"] = last_tool
    _LAST_SKILL_CONTEXT.set(skill_context)


def get_last_skill_context(clear: bool = False) -> Optional[dict]:
    context = _LAST_SKILL_CONTEXT.get()
    if clear:
        _LAST_SKILL_CONTEXT.set(None)
    return deepcopy(context) if isinstance(context, dict) else context

# 用于每轮新对话开始时，重置RAG工具调用计数器
def reset_tool_call_guards():
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set(0)

# 初始化RAG进度消息的队列 + 绑定异步事件循环
def set_rag_step_queue(queue):
    _RAG_STEP_QUEUE.set(queue)
    if queue:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        _RAG_STEP_LOOP.set(loop)
    else:
        _RAG_STEP_LOOP.set(None)


# 跨线程安全发送RAG执行步骤
def emit_rag_step(icon: str, label: str, detail: str = ""):
    queue = _RAG_STEP_QUEUE.get()
    loop = _RAG_STEP_LOOP.get()
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


# RAG系统对外暴露、给大模型调用的「知识库检索工具」
@tool("search_knowledge_base")
async def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    knowledge_tool_calls = _KNOWLEDGE_TOOL_CALLS_THIS_TURN.get()
    if knowledge_tool_calls >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set(knowledge_tool_calls + 1)

    from backend.app.rag.rag_pipeline import run_rag_graph

    rag_result = await asyncio.to_thread(run_rag_graph, query)
    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}

    # 将RAG trace日志存储到全局上下文中，供agent.py中后续调用获取
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."
    
    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    
    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("list_skills")
async def list_skills() -> str:
    """List all available skills with their descriptions and bundled resources."""
    _update_skill_context(last_tool="list_skills")
    return list_skills_text()

# 读取主SKILL.md文件内容
@tool("load_skill")
async def load_skill(skill_name: str, full: bool = False) -> str:
    """Load the main content of a skill. Use when the user's task matches a skill name or description."""
    main_content = load_skill_content(skill_name)
    _update_skill_context(active_skill=skill_name, last_tool="load_skill")
    if not full or main_content.startswith("错误"):
        return main_content


    resources = list_skill_resources_content(skill_name)
    return f"{main_content}\n\n---\n\n{resources}"

# 查看指定skill的所有附带资源
@tool("list_skill_resources")
async def list_skill_resources(skill_name: str) -> str:
    """List the bundled resources for a skill, such as references or assets."""
    _update_skill_context(active_skill=skill_name, last_tool="list_skill_resources")
    return list_skill_resources_content(skill_name)

# 加载指定skill的指定资源内容
@tool("load_skill_resource")
async def load_skill_resource(skill_name: str, resource_path: str) -> str:
    """Read a text resource inside a skill directory, such as references/*.md or assets/*.md."""
    _update_skill_context(
        active_skill=skill_name,
        loaded_resource=f"{skill_name}:{resource_path}",
        last_tool="load_skill_resource",
    )
    return load_skill_resource_content(skill_name, resource_path)

