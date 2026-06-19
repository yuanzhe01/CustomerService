from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


DATABASE_URL = _get_env("DATABASE_URL", required=True)

REDIS_URL = _get_env("REDIS_URL", "redis://127.0.0.1:6379")
REDIS_KEY_PREFIX = _get_env("REDIS_KEY_PREFIX", "supermew")
REDIS_DEFAULT_TTL = int(_get_env("REDIS_DEFAULT_TTL", "300"))

MILVUS_URI = _get_env("MILVUS_URI", "tcp://127.0.0.1:19530")
MILVUS_COLLECTION_NAME = _get_env("MILVUS_COLLECTION_NAME", "embeddings_collection")

JWT_SECRET_KEY = _get_env("JWT_SECRET_KEY", required=True)
JWT_ALGORITHM = _get_env("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(_get_env("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
ADMIN_INVITE_CODE = _get_env("ADMIN_INVITE_CODE", "")
PBKDF2_ROUNDS = int(_get_env("PBKDF2_ROUNDS", "310000"))

LLM_API_KEY = _get_env("LLM_API_KEY", required=True)
LLM_BASE_URL = _get_env("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = _get_env("LLM_MODEL", "qwen3.6-plus")

RERANK_MODEL = _get_env("RERANK_MODEL", "qwen3-rerank")
RERANK_BASE_URL = _get_env("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1")

HOST = _get_env("HOST", "0.0.0.0")
PORT = int(_get_env("PORT", "8000"))
RAG_UTILS_LOG_LEVEL = _get_env("RAG_UTILS_LOG_LEVEL", "INFO")
