import json
import logging
from typing import Any, Optional
import redis
from settings import REDIS_DEFAULT_TTL, REDIS_KEY_PREFIX, REDIS_URL


logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self):
        self.redis_url = REDIS_URL
        self.key_prefix = REDIS_KEY_PREFIX
        self.default_ttl = REDIS_DEFAULT_TTL  # 默认TTL为5分钟
        self._client = None
    
    # 懒加载：第一次使用时才创建Redis连接
    def _get_client(self):
        if self._client is None:
            logger.info("cache init redis client url=%s prefix=%s ttl=%s", self.redis_url, self.key_prefix, self.default_ttl)
            self._client = redis.Redis.from_url(self.redis_url, decode_responses = True)
            logger.info("cache redis client ready")
        return self._client

    # 给所有缓存key添加统一前缀，避免与其他应用的key冲突
    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    # 读取缓存，并进行JSON反序列化
    def get_json(self, key: str) -> Optional[Any]:
        try:
            cache_key = self._key(key)
            value = self._get_client().get(cache_key)
            if not value:
                logger.info("cache miss key=%s", cache_key)
                return None
            logger.info("cache hit key=%s size=%s", cache_key, len(value))
            return json.loads(value)
        except Exception:
            logger.warning("cache get failed key=%s", self._key(key), exc_info=True)
            return None
        
    # 写入缓存 + 自动 JSON 序列化 + 设置过期时间
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            cache_key = self._key(key)
            ttl_value = ttl or self.default_ttl
            payload = json.dumps(value, ensure_ascii=False)
            self._get_client().setex(cache_key, ttl_value, payload)
            logger.info("cache set key=%s ttl=%s size=%s", cache_key, ttl_value, len(payload))
        except Exception:
            logger.warning("cache set failed key=%s", self._key(key), exc_info=True)
            return

    # 删除单个key
    def delete(self, key: str) -> None:
        try:
            cache_key = self._key(key)
            deleted = self._get_client().delete(cache_key)
            logger.info("cache delete key=%s deleted=%s", cache_key, deleted)
        except Exception:
            logger.warning("cache delete failed key=%s", self._key(key), exc_info=True)
            return

    # 批量删除，支持通配符模式（如 "supermew:rag_trace:*"）
    def delete_pattern(self, pattern: str) -> None:
        try:
            full_pattern = self._key(pattern)
            keys = self._get_client().keys(full_pattern)
            if keys:
                deleted = self._get_client().delete(*keys)
                logger.info("cache delete_pattern pattern=%s matched=%s deleted=%s", full_pattern, len(keys), deleted)
            else:
                logger.info("cache delete_pattern pattern=%s matched=0", full_pattern)
        except Exception:
            logger.warning("cache delete_pattern failed pattern=%s", self._key(pattern), exc_info=True)
            return

cache = RedisCache()
