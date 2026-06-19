from collections import defaultdict
from typing import List, Tuple, Dict, Any
import json
import requests
import logging
import threading
import traceback

from backend.app.integrations.milvus_client import MilvusManager
from backend.app.integrations.embedding import embedding_service as _embedding_service
from backend.app.rag.parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model
from openai import OpenAI
from backend.app.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RAG_UTILS_LOG_LEVEL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)

# Logger setup
LOG_LEVEL = RAG_UTILS_LOG_LEVEL
logger = logging.getLogger("rag_utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))

API_KEY = LLM_API_KEY
BASE_URL = LLM_BASE_URL
MODEL = LLM_MODEL
AUTO_MERGE_ENABLED = True   # 默认开启自动合并功能
AUTO_MERGE_THRESHOLD = 2   # 自动合并触发阈值，子分块数量 >= 阈值，才合并
LEAF_RETRIEVE_LEVEL = 3    # 向量检索时，只检索这个层级的分块

# 全局初始化检索依赖（与api共用embedding_service，保证BM25状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()
_stepback_model = None
_stepback_model_lock = threading.Lock()


_dashscope_client = OpenAI(
    api_key=API_KEY,
    base_url=RERANK_BASE_URL,
)

# 把检索到的碎片化子分块，批量替换成它们所属的完整父分块
def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    logger.debug("_merge_to_parent_level called: docs=%d, threshold=%s", len(docs) if docs else 0, threshold)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)
    
    # 可以合并的父分块ID列表（子分块数量 >= 阈值，才合并）
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    logger.debug("Found %d parent groups, merge_parent_ids=%s", len(groups), merge_parent_ids)
    if not merge_parent_ids:
        logger.debug("No parent groups meet threshold; returning original docs")
        return docs, 0
    
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()

        # 没有父ID / 不需要合并 → 保留原碎片
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))   # 取子分块和父分块中较高的分数作为父分块的分数
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)
    
    logger.debug("Merged %d docs into parents, deduped_count=%d", merged_count, len(deduped))
    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    # 如果关闭自动合并或者没有检索到文档，直接返回原结果和相关信息
    logger.debug("_auto_merge_documents called: auto_merge_enabled=%s, docs_count=%d, top_k=%d",
                 AUTO_MERGE_ENABLED, len(docs) if docs else 0, top_k)
    if not AUTO_MERGE_ENABLED or not docs:
        logger.debug("Auto merge disabled or no docs; returning first top_k docs")
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }
    
    # 两段自动合并：L3->L2，再L2->L1
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold = AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold = AUTO_MERGE_THRESHOLD)

    # 按相关性分数降序排序，并截取前top_k个结果
    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    logger.debug("After merge sort, merged_docs_count=%d", len(merged_docs))
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    logger.info("Auto-merge applied=%s replaced_chunks=%d steps=%d",
                replaced_count > 0, replaced_count,
                int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0))
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]   # 获取文档和其在原结果中的排名（从1开始）
    meta: Dict[str, Any] = {
        "rerank_enabled": True,
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": RERANK_BASE_URL,
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    logger.debug("_rerank_documents called: query='%s' candidate_count=%d top_k=%d",
                 query if len(query) < 200 else query[:200] + '...', len(docs_with_rank), top_k)
    if not docs_with_rank or not meta["rerank_enabled"]:
        logger.debug("No candidates or rerank disabled; returning first top_k candidates")
        return docs_with_rank[:top_k], meta
    
    # 构造阿里云重排请求体
    body = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
    }

    try:
        meta["rerank_applied"] = True
        results = _dashscope_client.post(
            "/reranks",
            body=body,
            cast_to=object
        )

        # 解析返回结果
        items = results.get("results", []) if isinstance(results, dict) else []
        logger.debug("Rerank service returned %d items", len(items) if items else 0)
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            logger.info("Rerank produced %d reranked docs; returning top_k=%d", len(reranked), top_k)
            return reranked[:top_k], meta
        
        meta["rerank_error"] = "empty_rerank_results"
        logger.warning("Rerank returned no valid items; falling back to original ranking")
        return docs_with_rank[:top_k], meta
    
    except Exception as e:
        meta["rerank_error"] = f"dashscope_rerank_error: {str(e)}"
        logger.error("Rerank call failed: %s\n%s", str(e), traceback.format_exc())
        return docs_with_rank[:top_k], meta
    
# 获取全局单例的步骤回退模型实例
def _get_stepback_model():
    global _stepback_model
    if not API_KEY or not MODEL:
        logger.debug("Stepback model not available: API_KEY or MODEL missing")
        return None
    
    if _stepback_model is None:
        with _stepback_model_lock:
            if _stepback_model is None:
                logger.info("Initializing step-back chat model: %s", MODEL)
                _stepback_model = init_chat_model(
                    model=MODEL,
                    model_provider="openai",
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    temperature=0.2,
                )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )

    try:
        resp = (model.invoke(prompt).content or "").strip()
        logger.debug("Generated step-back question: %s", resp)
        return resp
    except Exception:
        logger.error("_generate_step_back_question failed:\n%s", traceback.format_exc())
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )

    try:
        resp = (model.invoke(prompt).content or "").strip()
        logger.debug("Step-back answer: %s", resp)
        return resp
    except Exception:
        logger.error("_answer_step_back_question failed:\n%s", traceback.format_exc())
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        resp = (model.invoke(prompt).content or "").strip()
        logger.debug("Generated hypothetical document length=%d", len(resp))
        return resp
    except Exception:
        logger.error("generate_hypothetical_document failed:\n%s", traceback.format_exc())
        return ""
    
def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query

    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }

# 整体检索流程
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"   # 只检索叶子分块
    logger.info("retrieve_documents start: query='%s' top_k=%d candidate_k=%d filter=%s",
                query if len(query) < 200 else query[:200] + '...', top_k, candidate_k, filter_expr)
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)
        logger.debug("Embeddings fetched: dense_len=%d sparse_present=%s", len(dense_embeddings), sparse_embedding is not None)

        # Milvus混合检索（粗排）：召回top*3条候选，后续会进行重排
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        logger.info("Hybrid retrieve returned %d candidates", len(retrieved) if retrieved else 0)
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)
        logger.debug("Retrieval meta: %s", json.dumps(rerank_meta, default=str, ensure_ascii=False))

        # 最终返回重排和自动合并后的结果，以及相关的元信息
        return {"docs": merged_docs, "meta": rerank_meta}
    except Exception as e:
        logger.warning("Hybrid retrieve failed: %s\n%s", str(e), traceback.format_exc())
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            logger.info("Dense fallback retrieve returned %d candidates", len(retrieved) if retrieved else 0)
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)
            logger.debug("Retrieval meta (fallback): %s", json.dumps(rerank_meta, default=str, ensure_ascii=False))
            return {"docs": merged_docs, "meta": rerank_meta}
        except Exception as e2:
            logger.error("Dense fallback failed: %s\n%s", str(e2), traceback.format_exc())
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": False,
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": RERANK_BASE_URL,
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }
