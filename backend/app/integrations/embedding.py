import os
from dotenv import load_dotenv

load_dotenv()

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import jieba

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STATE_PATH = PROJECT_ROOT.parent / "data" / "bm25_state.json"


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_dir = Path(
        os.getenv("EMBEDDING_MODEL_DIR", "D:/vscode/models/bge-m3")
    ).expanduser()
    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"本地嵌入模型目录不存在: {model_dir}。请设置 EMBEDDING_MODEL_DIR 为已下载好的 bge-m3 本地目录。"
        )

    return HuggingFaceEmbeddings(
        model_name=str(model_dir),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# 文本向量化服务：密集向量本地模型 + BM25稀疏向量
class EmbeddingService:
    def __init__(self, state_path: Path | str | None = None):
        self._embedder = _create_dense_embedder()
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5   # 这个参数用于解决词频非线性饱和的问题，k1越大饱和值越高
        self.b = 0.75   # 这个参数用于实现文档长度归一化，b越大对文档长度的惩罚越大

        self._vocab: dict[str, int] = {}            # 词汇表，记录每个词的索引
        self._vocab_counter = 0                     # 词汇表计数器
        self._doc_freq: Counter[str] = Counter()    # 用于记录每个词在多少文档中出现过
        self._total_docs = 0                        # 记录总文档数
        self._sum_token_len = 0                     # 记录所有文档的总词数，用于计算平均文档长度
        self._avg_doc_len = 1.0                     # 平均文档长度，初始值为1避免除零错误

        self._load_state()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    # 从指定的JSON文件中加载并恢复该类的持久化运行状态
    def _load_state(self) -> None:
        path = self._state_path
        if not path.is_file():
            return
        
        try:
            raw = json.loads(path.read_text(encoding = "utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        
        if raw.get("version") != 1:
            return
        
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))

        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    # 将当前的运行状态以JSON格式持久化保存到指定的文件中
    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")   # 生成临时文件路径
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")   # 将数据写入临时文件
        tmp.replace(self._state_path)   # 原子性操作，用临时文件覆盖原文件

    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()
    
    # 增加文档
    def increment_add_documents(self, texts: list[str]) -> None:
        if not texts:
            return
        
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    # 删除文档
    def increment_remove_documents(self, texts: list[str]) -> None:
        if not texts:
            return
        
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    # 使用bge-m3生成文本的稠密向量
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e
        
    # 进行文本分词，支持中文和英文混合文本，中文使用jieba分词，英文和数字等字符直接作为token
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        parts = re.findall( r'[\u4e00-\u9fff]+|[a-zA-Z0-9_\-\.]+', text)

        for part in parts:
            if re.fullmatch(r'[\u4e00-\u9fff]+', part):
                tokens.extend(jieba.lcut(part))
            else:
                tokens.append(part)
        
        return tokens
    
    # 将一段文本转换成BM25稀疏向量
    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False   # 标记：是否出现了新词
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        # 遍历当前文本里的【每个词 + 词频】
        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            # 核心计算1：IDF逆文档频率 → 词越稀有，分数越高
            idx = self._vocab[token]
            df = self._doc_freq.get(token, 0)   # 该词在多少文档中出现过
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # 核心计算2：BM25词频调整 → 词频越高，分数越高，但有一个饱和值
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)

            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed
    
    def get_sparse_embedding(self, text: str) -> dict:
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector
    
    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out
    
    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings
    
    
# 全进程唯一实例：写入与检索共用同一份BM25持久化状态
embedding_service = EmbeddingService()

            



    
