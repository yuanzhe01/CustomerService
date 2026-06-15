import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from agent import chat_with_agent, chat_with_agent_stream, storage
from auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, require_admin, resolve_role
from document_loader import DocumentLoader
from embedding import embedding_service
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from models import User
from parent_chunk_store import ParentChunkStore
from schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    CurrentUserResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
    LoginRequest,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
    SkillInfo,
    SkillListResponse,
    SkillUploadResponse,
)
from skill_loader import SKILLS_DIR, TEXT_RESOURCE_EXTENSIONS, parse_frontmatter, scan_skill_index
from upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

router = APIRouter()
SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_SKILL_ZIP_BYTES = 5 * 1024 * 1024


def _remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)
    

@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """获取指定会话的所有消息"""
    try:
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = await chat_with_agent(request.message, current_user.username, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)
    
    
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/admin/skills/upload", response_model=SkillUploadResponse)
async def upload_skill(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """管理员上传 skill zip 包，并在导入后立即生效。"""
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持上传 zip 格式的 skill 包")

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skill_zip_") as temp_dir:
        zip_path = Path(temp_dir) / filename
        await _save_upload_file(file, zip_path)

        if zip_path.stat().st_size > MAX_SKILL_ZIP_BYTES:
            raise HTTPException(status_code=400, detail=f"skill zip 大小不能超过 {MAX_SKILL_ZIP_BYTES // (1024 * 1024)}MB")

        skill_name, target_dir, imported_files = _prepare_skill_import(zip_path)

    discovered = next((item for item in scan_skill_index() if item.name == skill_name or item.folder_name == target_dir.name), None)
    if not discovered:
        raise HTTPException(status_code=500, detail="skill 已导入，但索引刷新失败")

    return SkillUploadResponse(
        skill_name=discovered.name,
        folder_name=discovered.folder_name,
        resources=[resource.path for resource in discovered.resources],
        message=f"skill '{discovered.name}' 上传成功，已可被 list_skills 发现",
    )


@router.get("/admin/skills", response_model=SkillListResponse)
async def list_admin_skills(_: User = Depends(require_admin)):
    skills = scan_skill_index()
    return SkillListResponse(
        skills=[
            SkillInfo(
                name=skill.name,
                description=skill.description,
                folder_name=skill.folder_name,
                entry_file=skill.entry_file,
                resources=[resource.path for resource in skill.resources],
            )
            for skill in skills
        ]
    )


def _is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
    )


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _normalize_zip_member_path(member_name: str) -> str:
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    normalized = os.path.normpath(normalized).replace("\\", "/")
    if normalized in {".", ".."} or normalized.startswith("../") or os.path.isabs(normalized):
        raise HTTPException(status_code=400, detail=f"zip 中存在非法路径: {member_name}")
    return normalized


def _validate_skill_file_extension(relative_path: str) -> None:
    if relative_path.endswith("/"):
        return
    _, ext = os.path.splitext(relative_path.lower())
    if ext not in TEXT_RESOURCE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"skill 包含不支持的文件类型: {relative_path}")


def _prepare_skill_import(zip_path: Path) -> tuple[str, Path, list[str]]:
    with tempfile.TemporaryDirectory(prefix="skill_import_") as temp_dir:
        extract_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                raise HTTPException(status_code=400, detail="zip 包为空")

            normalized_members: list[tuple[zipfile.ZipInfo, str]] = []
            for info in members:
                normalized = _normalize_zip_member_path(info.filename)
                _validate_skill_file_extension(normalized)
                normalized_members.append((info, normalized))

            skill_md_candidates = [normalized for _, normalized in normalized_members if normalized.endswith("/SKILL.md") or normalized == "SKILL.md"]
            if len(skill_md_candidates) != 1:
                raise HTTPException(status_code=400, detail="zip 中必须且只能包含一个 SKILL.md")

            skill_md_relative = skill_md_candidates[0]
            skill_root_prefix = os.path.dirname(skill_md_relative).replace("\\", "/")

            for _, normalized in normalized_members:
                current_dir = os.path.dirname(normalized).replace("\\", "/")
                if skill_root_prefix:
                    if current_dir != skill_root_prefix and not current_dir.startswith(skill_root_prefix + "/"):
                        raise HTTPException(status_code=400, detail="zip 中只能包含一个 skill 根目录")

            for info, normalized in normalized_members:
                target_path = extract_root / normalized
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        skill_root = extract_root / skill_root_prefix if skill_root_prefix else extract_root
        skill_md_path = skill_root / "SKILL.md"
        frontmatter = parse_frontmatter(skill_md_path.read_text(encoding="utf-8"))
        raw_skill_name = str(frontmatter.get("name", "")).strip()
        raw_description = str(frontmatter.get("description", "")).strip()
        if not raw_skill_name or not raw_description:
            raise HTTPException(status_code=400, detail="SKILL.md 的 frontmatter 必须包含 name 和 description")
        if not SKILL_NAME_PATTERN.match(raw_skill_name):
            raise HTTPException(status_code=400, detail="skill name 仅允许字母、数字、下划线和短横线，且必须以字母或数字开头")

        folder_name = raw_skill_name
        target_dir = SKILLS_DIR / folder_name
        imported_files: list[str] = []
        for path in sorted(skill_root.rglob("*")):
            if path.is_file():
                imported_files.append(path.relative_to(skill_root).as_posix())

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill_root, target_dir)
        return raw_skill_name, target_dir, imported_files


def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """后台执行耗时的解析、分块、向量化入库，并持续更新任务进度。"""
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
            job_id,
            "vector_store",
            0,
            "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0,
        )

        def _on_vector_progress(processed: int, total: int) -> None:
            percent = round(processed * 100 / total) if total else 100
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as e:
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _process_delete_job(job_id: str, filename: str) -> None:
    """后台执行文档删除，并把每个删除阶段同步给前端行内进度卡片。"""
    failed_step = "prepare"
    try:
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        _remove_bm25_stats_for_filename(filename)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 PostgreSQL 父级分块")
        parent_chunk_store.delete_by_filename(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        # 完成摘要会由前端保留 3 秒，再自动从文档列表移除。
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as e:
        delete_job_manager.fail_job(job_id, failed_step, str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """获取已上传的文档列表（管理员）"""
    try:
        milvus_manager.init_collection()

        results = milvus_manager.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    """轻量版异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化。"""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job = upload_job_manager.create_job(filename)
    file_path = UPLOAD_DIR / filename

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "正在保存文件到服务器")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    background_tasks.add_task(_process_upload_job, job["job_id"], str(file_path), filename)
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
):
    """轻量版异步删除：立即返回 job_id，实际删除在后台执行。"""
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(_process_delete_job, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """上传文档并进行 embedding"""
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection()

        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件）"""
    try:
        milvus_manager.init_collection()

        delete_expr = f'filename == "{filename}"'
        _remove_bm25_stats_for_filename(filename)
        result = milvus_manager.delete(delete_expr)
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
