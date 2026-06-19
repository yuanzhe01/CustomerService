from typing import Literal, TypedDict, List, Optional
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from backend.app.rag.rag_utils import retrieve_documents, step_back_expand, generate_hypothetical_document, logger
from backend.app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from backend.app.tools import emit_rag_step

API_KEY = LLM_API_KEY
BASE_URL = LLM_BASE_URL
MODEL = LLM_MODEL

_grader_model = None
_router_model = None

# 私有函数：创建并返回文档相关性评分专用的LLM模型
def _get_grader_model():
    global _grader_model

    if not API_KEY or not MODEL:
        return None
    
    if _grader_model is None:
        _grader_model = init_chat_model(
            model = MODEL,
            model_provider = "openai",
            api_key = API_KEY,
            base_url = BASE_URL,
            temperature = 0,      # 温度=0，输出绝对稳定，不随机
            stream_usage = True,
        )
    return _grader_model

# 私有函数：创建并返回RAG路由决策专用的LLM模型
def _get_router_model():
    global _router_model

    if not API_KEY or not MODEL:
        return None
    
    if _router_model is None:
        _router_model = init_chat_model(
            model = MODEL,
            model_provider = "openai",
            api_key = API_KEY,
            base_url = BASE_URL,
            temperature = 0,
            stream_usage = True,
        )
    return _router_model

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. "
    "Please respond in JSON format only, with no extra explanation.\n"
    "Return one of these fields: \n"
    "- binary_score: 'yes' or 'no'\n"
    "- relevant: 'yes' or 'no' (fallback)\n"
    "Optional: analysis for short reasoning.\n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. "
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)

# 构造一个Pydantic数据模型，让大模型进行结构化输出
class GradeDocuments(BaseModel):
    binary_score: Optional[str] = Field(
        default=None,
        description = "Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
    relevant: Optional[str] = Field(
        default=None,
        description = "Fallback relevance score: 'yes' or 'no'"
    )
    analysis: Optional[str] = Field(
        default=None,
        description = "Optional short reasoning or analysis"
    )

    class Config:
        extra = "ignore"

class RewriteStrategy(BaseModel):
    strategy: Literal["step_back", "hyde", "complex"]

# LangGraph工作流的全局共享状态容器
class RAGState(TypedDict):
    question: str                     # 用户的原始问题
    query: str                        # 执行检索用的查询词
    context: str                      # 从知识库检索到的文本内容
    docs: List[dict]                  # 检索后的所有文档列表
    route: Optional[str]              # 流程路由
    expansion_type: Optional[str]     # 查询扩展策略：'step_back' / 'hyde' / None
    expanded_query: Optional[str]     # 扩展后的优化查询词
    step_back_question: Optional[str] # 退步查询生成的抽象问题
    step_back_answer: Optional[str]   # 退步查询生成的抽象问题的答案
    hypothetical_doc: Optional[str]   # HyDE生成的假设性文档
    rag_trace: Optional[dict]         # RAG全链路日志

# 文档格式化工具函数，将字典列表格式的原始文档数据，转换成纯文本字符串
def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    
    chunks = []
    for i, doc in enumerate(docs, 1):
        # 从每个文档字典中提取字段，无对应键则用默认值
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)

# 初始文档检索核心函数
def retrieve_initial(state: RAGState) -> RAGState:
    query = state["question"]
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")
    
    # 调用检索接口，根据用户问题查知识库，最多返回5条相关文档
    retrieved = retrieve_documents(query, top_k=5)

    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)

    emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    emit_rag_step("✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "rerank_enabled": retrieve_meta.get("rerank_enabled"),
        "rerank_applied": retrieve_meta.get("rerank_applied"),
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "candidate_k": retrieve_meta.get("candidate_k"),
        "leaf_retrieve_level": retrieve_meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": retrieve_meta.get("auto_merge_enabled"),
        "auto_merge_applied": retrieve_meta.get("auto_merge_applied"),
        "auto_merge_threshold": retrieve_meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": retrieve_meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": retrieve_meta.get("auto_merge_steps"),
    }
    
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估文档相关性...")
    logger.info("grade_documents_node start: grader_available=%s", bool(grader))
    if not grader:
        logger.warning("No grader model available, routing to rewrite_question")
        grade_update = {
            "grade_score": "unknown",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    
    question = state["question"]
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    logger.debug("Invoking grader with prompt length=%d", len(prompt))
    try:
        response = grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = (response.binary_score or response.relevant or "").strip().lower()
        if score not in {"yes", "no"}:
            logger.warning("Unexpected grade output, normalized to unknown: %s", score)
            score = "unknown"
        route = "generate_answer" if score == "yes" else "rewrite_question"

        logger.info("Grader returned score=%s route=%s relevant=%s analysis=%s",
                    score, route, getattr(response, "relevant", None), getattr(response, "analysis", None))

        if route == "generate_answer":
            emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
        else:
            emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")

    except Exception as e:
        logger.error("Grader invocation failed: %s", str(e), exc_info=True)
        score = "unknown"
        route = "rewrite_question"
        emit_rag_step("⚠️", "评估失败，改为重写查询", f"error: {str(e)}")

    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
    }
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    logger.debug("grade_documents_node complete: %s", grade_update)
    return {"route": route, "rag_trace": rag_trace}


def grade_expanded_documents_node(state: RAGState) -> RAGState:
    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估扩展检索结果...")
    logger.info("grade_expanded_documents_node start: grader_available=%s", bool(grader))
    if not grader:
        logger.warning("No grader model available for expanded result, proceeding to answer generation")
        grade_update = {
            "expanded_grade_score": "unknown",
            "expanded_grade_route": "generate_answer",
            "expanded_rewrite_needed": False,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "generate_answer", "rag_trace": rag_trace}

    question = state["question"]
    context = state.get("context", "")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    logger.debug("Invoking grader for expanded docs with prompt length=%d", len(prompt))
    try:
        response = grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = (response.binary_score or response.relevant or "").strip().lower()
        if score not in {"yes", "no"}:
            logger.warning("Unexpected expanded grade output, normalized to unknown: %s", score)
            score = "unknown"

        if score == "yes":
            emit_rag_step("✅", "扩展检索结果评估通过", f"评分: {score}")
        else:
            emit_rag_step("⚠️", "扩展检索结果相关性不足，但继续生成答案", f"评分: {score}")

        logger.info(
            "Expanded grader returned score=%s relevant=%s analysis=%s",
            score,
            getattr(response, "relevant", None),
            getattr(response, "analysis", None),
        )

    except Exception as e:
        logger.error("Expanded grader invocation failed: %s", str(e), exc_info=True)
        score = "unknown"
        emit_rag_step("⚠️", "扩展评估失败，继续生成答案", f"error: {str(e)}")

    grade_update = {
        "expanded_grade_score": score,
        "expanded_grade_route": "generate_answer",
        "expanded_rewrite_needed": score != "yes",
    }
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    logger.debug("grade_expanded_documents_node complete: %s", grade_update)
    return {"route": "generate_answer", "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    question = state["question"]
    emit_rag_step("✏️", "正在重写查询...")
    router = _get_router_model()
    strategy = "step_back"
    if router:
        prompt = (
            "Please select the most appropriate query expansion strategy and respond in JSON format.\n"
            "根据用户问题选择最合适的查询扩展策略，用JSON格式返回结果。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
            f"用户问题：{question}"
        )
        try:
            decision = router.with_structured_output(RewriteStrategy).invoke(
                [{"role": "user", "content": prompt}]
            )
            strategy = decision.strategy
        except Exception:
            strategy = "step_back"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)
    
    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)
    
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
    })

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_model = None
    rerank_endpoint = None
    rerank_errors = []
    retrieval_mode = None
    candidate_k = None
    leaf_retrieve_level = None
    auto_merge_enabled = None
    auto_merge_applied = False
    auto_merge_threshold = None
    auto_merge_replaced_chunks = 0
    auto_merge_steps = 0

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=5)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or hyde_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else hyde_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(hyde_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or hyde_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(hyde_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(hyde_meta.get("auto_merge_steps") or 0)

    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=5)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        emit_rag_step(
            "🧱",
            "Step-back 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or step_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else step_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(step_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or step_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(step_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(step_meta.get("auto_merge_steps") or 0)

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    emit_rag_step("✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "rerank_enabled": rerank_enabled_any,
        "rerank_applied": rerank_applied_any,
        "rerank_model": rerank_model,
        "rerank_endpoint": rerank_endpoint,
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": leaf_retrieve_level,
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": auto_merge_applied,
        "auto_merge_threshold": auto_merge_threshold,
        "auto_merge_replaced_chunks": auto_merge_replaced_chunks,
        "auto_merge_steps": auto_merge_steps,
    })
    return {"docs": deduped, "context": context, "rag_trace": rag_trace} 


# 利用LangGraph构建自动、闭环、带智能决策的RAG工作流程图
def build_rag_graph():
    graph = StateGraph(RAGState)   # 创建一个空的工作流容器，规定所有数据用RAGState传递

    # 将之前定义的核心函数封装成节点，添加到工作流图中
    graph.add_node("retrieve_initial", retrieve_initial)        # 初始检索节点,第一次普通检索
    graph.add_node("grade_documents", grade_documents_node)     # 文档相关性评分
    graph.add_node("rewrite_question", rewrite_question_node)   # 检索失败后进行问题重写
    graph.add_node("retrieve_expanded", retrieve_expanded)      # 扩展检索（HyDE/退步查询，二次补救检索）
    graph.add_node("grade_expanded_documents", grade_expanded_documents_node)  # 扩展检索后的结果评估

    # 定义节点之间的连接关系，形成完整的RAG流程
    graph.set_entry_point("retrieve_initial")                   # 流程入口，用户提问后首先进入初始检索节点
    graph.add_edge("retrieve_initial", "grade_documents")       # 初始检索完成后，进入文档相关性评分节点
    graph.add_conditional_edges(                                # 根据route，决定是直接生成答案，还是进入重写查询节点进行补救
        "grade_documents", 
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")          # 重写查询完成后，进入扩展检索节点进行二次检索
    graph.add_edge("retrieve_expanded", "grade_expanded_documents")  # 扩展检索完成后，进入扩展文档评估节点
    graph.add_edge("grade_expanded_documents", END)                  # 扩展检索评估完成后，直接生成答案
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
    })
