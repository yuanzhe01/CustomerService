from datetime import datetime
from typing import List
from cache import cache
from database import SessionLocal
from models import ParentChunk

# 基于PostgreSQL + Redis的父级分块存储服务
class ParentChunkStore:
     
    @staticmethod
    def _sanitize_text(value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.replace("\x00", "")

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }
    
    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"
    
    # 写入/更新父级父块，返回写入条数
    def upsert_documents(self, docs: List[dict]) -> int:
        if not docs:
            return 0
        
        db = SessionLocal()
        upserted = 0
        try:
            for doc in docs:
                chunk_id = (doc.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue

                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                payload = {
                    "text": ParentChunkStore._sanitize_text(doc.get("text", "")),
                    "filename": ParentChunkStore._sanitize_text(doc.get("filename", "")),
                    "file_type": ParentChunkStore._sanitize_text(doc.get("file_type", "")),
                    "file_path": ParentChunkStore._sanitize_text(doc.get("file_path", "")),
                    "page_number": int(doc.get("page_number", 0) or 0),
                    "parent_chunk_id": ParentChunkStore._sanitize_text(doc.get("parent_chunk_id", "")),
                    "root_chunk_id": ParentChunkStore._sanitize_text(doc.get("root_chunk_id", "")),
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "chunk_id": chunk_id,
                    "text": payload["text"],
                    "filename": payload["filename"],
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": payload["page_number"],
                    "parent_chunk_id": payload["parent_chunk_id"],
                    "root_chunk_id": payload["root_chunk_id"],
                    "chunk_level": payload["chunk_level"],
                    "chunk_idx": payload["chunk_idx"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)   # 更新已有记录
                else:
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))
                
                cache.set_json(self._cache_key(chunk_id), cache_payload)   # 同步更新缓存
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted
    
    # 根据chunk_id列表批量查询父级分块信息，返回结果顺序与输入chunk_id列表一致
    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        missing_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if not key:
                continue
            cached = cache.get_json(self._cache_key(key))
            if cached:
                ordered_results[key] = cached
            else:
                missing_ids.append(key)
        
        if missing_ids:
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()
        
        return [ordered_results[item] for item in chunk_ids if item in ordered_results]
    

    # 按文件名删除父级块，返回删除条数
    def delete_by_filename(self, filename: str) -> int:
        if not filename:
            return 0
        
        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            return deleted
        finally:
            db.close()        