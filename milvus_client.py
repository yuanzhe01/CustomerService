import threading
from typing import Callable, TypeVar
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from settings import MILVUS_COLLECTION_NAME, MILVUS_URI

# Milvus单次query的limit上限
QUERY_MAX_LIMIT = 16384
T = TypeVar("T")

# Milvus的连接和集合管理
class MilvusManager:
    def __init__(self):
        self.collection_name = MILVUS_COLLECTION_NAME
        self.uri = MILVUS_URI
        self.client = None
        self._client_lock = threading.RLock()

    def _get_client(self) -> MilvusClient:
        # Lazy-create client to avoid blocking app import/startup when Milvus is temporarily unavailable.
        with self._client_lock:
            if self.client is None:
                self.client = MilvusClient(uri=self.uri)
            return self.client
           
    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "closed channel" in str(exc).lower()
    
    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None)   # 获取客户端的close方法，找不到就返回None
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    # 销毁并关闭失效的Milvus客户端连接
    def _reset_client(self, failed_client=None) -> None:
        with self._client_lock:
            if self.client is None:
                return
            
            # 如果指定了要关闭的失败客户端，但当前客户端不是它，直接退出
            if failed_client is not None and self.client is not failed_client:
                return
            
            client = self.client
            self.client = None
        
        self._close_client(client)

    # 传入一个数据库操作，自动处理重连，返回操作结果
    # operation是一个「接收1个MilvusClient作为参数、并执行数据库操作的可调用对象」
    def _run_with_reconnect(self, operation: Callable[[MilvusClient], T]) -> T:  

        # 获取当前可用的客户端
        client = self._get_client()

        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise

            self._reset_client(client)
            return operation(self._get_client())
        
    # 初始化Milvus集合，同时支持密集向量和稀疏向量
    def init_collection(self, dense_dim: int | None = None):
        if dense_dim is None:
            dense_dim = 1024
        
        def _init(client: MilvusClient) -> None:
            if not client.has_collection(self.collection_name):
                schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

                # 主键
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

                # 密集向量（来自embedding模型）
                schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)

                # 稀疏向量（来自 BM25）
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

                # 文本和元数据字段
                schema.add_field("text", DataType.VARCHAR, max_length=2000)
                schema.add_field("filename", DataType.VARCHAR, max_length=255)
                schema.add_field("file_type", DataType.VARCHAR, max_length=50)
                schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
                schema.add_field("page_number", DataType.INT64)
                schema.add_field("chunk_idx", DataType.INT64)

                # Auto-merging 所需层级字段
                schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("chunk_level", DataType.INT64)

                # 为两种向量分别创建索引
                index_params = client.prepare_index_params()

                 # 密集向量索引 - 使用 HNSW（更适合混合检索）
                index_params.add_index(
                    field_name="dense_embedding",
                    index_type="HNSW",   # 密集向量索引-HNSW
                    metric_type="IP",    # 度量方式-内积
                    params={"M": 16, "efConstruction": 256}
                )

                # 稀疏向量索引
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",   # 稀疏向量索引-倒排索引
                    metric_type="IP",
                    params={"drop_ratio_build": 0.2}   # 构建时丢弃20%权重最低的关键词
                )

                client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )

        self._run_with_reconnect(_init)

    # 插入数据到Milvus
    def insert(self, data: list[dict]):
        return self._run_with_reconnect(lambda client: client.insert(self.collection_name, data))
    

    # 查询数据
    def query(self, filter_expr: str = "", output_fields: list[str] = None, limit: int = 10000, offset: int = 0):
        return self._run_with_reconnect(
            lambda client: client.query(
                collection_name=self.collection_name,
                filter=filter_expr,   # 过滤表达式，用于进行数据筛选
                output_fields=output_fields or ["filename", "file_type"],   # 输出字段列表，指定查询结果返回哪些字段
                limit=min(limit, QUERY_MAX_LIMIT),   # 返回条数限制，最多返回多少条数据
                offset=offset   # 偏移量，用于进行数据分页
            )
        )
    
    # 分页拉取匹配filter的全部行
    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        fields = output_fields or ["filename", "file_type"]
        out: list = []
        offset = 0
        while True:
            batch = self._run_with_reconnect(
                lambda client: client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
            )
            
            # 如果当前批次没有数据，直接退出循环
            if not batch:
                break
            out.extend(batch)

            # 如果当前批次数量 < 最大限制，说明是【最后一批数据】，退出循环
            if len(batch) < QUERY_MAX_LIMIT:
                break
            offset += len(batch)
        
        return out
    
    # 根据chunk_id批量查询分块
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        
        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"
        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )
        

    # 混合检索
    def hybrid_retrieve(self, dense_embedding: list[float], sparse_embedding: dict, top_k: int = 5, rrf_k: int = 60, filter_expr: str = "",) -> list[dict]:
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]

        # 密集向量搜索请求
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,  # 多取一些用于融合
            expr=filter_expr,
        )

        # 稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        # 使用RRF排序算法融合结果，把多个不同检索系统的排序列表，根据排名倒数合并成一个更优的综合排序列表
        reranker = RRFRanker(k=rrf_k)

        results = self._run_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields
            )
        )

        # 格式化返回结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })
        return formatted_results
    
    # 当稀疏向量不可用时，进行降级，仅使用密集向量检索
    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        results = self._run_with_reconnect(
            lambda client: client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
            )
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0)
                })

        return formatted_results
    
    # 删除数据
    def delete(self, filter_expr: str):
        return self._run_with_reconnect(
            lambda client: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
        )
    
    # 检查集合是否存在
    def has_collection(self) -> bool:
        return self._run_with_reconnect(lambda client: client.has_collection(self.collection_name))
    
    # 删除集合
    def drop_collection(self):
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)
        
        self._run_with_reconnect(_drop)
