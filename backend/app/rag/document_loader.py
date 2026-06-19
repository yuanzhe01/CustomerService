import os
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 文档加载和分片服务(传入基础分块大小，分块之间的重叠长度)
class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        # RecursiveCharacterTextSplitter是LangChain最常用的文本切割工具
        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size = level_1_size,
            chunk_overlap = level_1_overlap,
            add_start_index = True,   # 记录每个文本块在原文中的起始位置
            separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],   # 切割优先级
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size = level_2_size,
            chunk_overlap = level_2_overlap,
            add_start_index = True,
            separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size = level_3_size,
            chunk_overlap = level_3_overlap,
            add_start_index = True,
            separators = ["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )

    # 为每一个拆分后的文本块生成【全局唯一、可溯源的标准化ID】
    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"
    
    # 处理【单页文本】的工具函数：将每一页文本分成3层不同粒度的块，并记录层级关系和全局唯一ID（str：单页文本，base_doc：单页的元数据）
    def _split_page_to_three_levels(self, text: str, base_doc: Dict, page_global_chunk_idx: int) -> List[Dict]:
        if not text:
            return []
        
        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip().replace("\x00", "")
            if not level_1_text:
                continue

            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,   # 继承原始文档的元数据（如文件名、页码等）
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip().replace("\x00", "")
                if not level_2_text:
                    continue

                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip().replace("\x00", "")
                    if not level_3_text:
                        continue

                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1

                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,  
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks
    
    # 加载单个文档并分片
    def load_document(self, file_path: str, filename: str) -> list[dict]:
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")
        
        try:
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": doc.metadata.get("page", 0),
                }

                page_chunks = self._split_page_to_three_levels(
                    text = (doc.page_content or "").strip(),   # strip()用于去掉文本块前后的空白字符
                    base_doc = base_doc,
                    page_global_chunk_idx = page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")
        
    # 从文件夹加载所有文档并分片
    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        all_documents = []
        
        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc")) or file_lower.endswith((".xlsx", ".xls"))):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue
        return all_documents    