import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.app.db.models import MCPServerConfig
from backend.app.integrations.embedding import embedding_service
from backend.app.integrations.milvus_client import MilvusManager
from backend.app.integrations.milvus_writer import MilvusWriter
from backend.app.jobs.upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager
from backend.app.rag.document_loader import DocumentLoader
from backend.app.rag.parent_chunk_store import ParentChunkStore
from backend.app.schemas import MCPServerInfo
from backend.app.skills.skill_loader import SKILLS_DIR, TEXT_RESOURCE_EXTENSIONS, parse_frontmatter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"
MCP_UPLOAD_DIR = DATA_DIR / "mcp_servers"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_SKILL_ZIP_BYTES = 5 * 1024 * 1024
MCP_TRANSPORTS = {"stdio", "http"}


def remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)


def parse_json_form_field(raw_value: str | None, expected_type: type, field_name: str, default):
    if raw_value is None:
        return default

    text = str(raw_value).strip()
    if not text:
        return default

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} 不是合法 JSON: {exc}") from exc

    if not isinstance(parsed, expected_type):
        raise HTTPException(status_code=400, detail=f"{field_name} 必须是 {expected_type.__name__} 类型的 JSON")

    return parsed


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(target_dir.resolve())):
                raise HTTPException(status_code=400, detail="zip 包含非法路径，已拒绝导入")
        archive.extractall(target_dir)


def mcp_server_storage_dir(server_id: int) -> Path:
    return MCP_UPLOAD_DIR / f"server_{server_id}"


def delete_mcp_server_assets(server: MCPServerConfig) -> None:
    fallback_dir = mcp_server_storage_dir(server.id)
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir, ignore_errors=True)


async def save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


async def save_mcp_server_asset(server: MCPServerConfig, file: UploadFile) -> None:
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="上传文件名不能为空")

    MCP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    storage_dir = mcp_server_storage_dir(server.id)
    if storage_dir.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    saved_file = storage_dir / filename
    await save_upload_file(file, saved_file)

    asset_dir = storage_dir
    asset_path = saved_file
    if saved_file.suffix.lower() == ".zip":
        extracted_dir = storage_dir / "bundle"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(saved_file, extracted_dir)
        asset_dir = extracted_dir
        asset_path = extracted_dir

    server.uploaded_filename = filename
    server.uploaded_asset_dir = str(asset_dir)
    server.uploaded_asset_path = str(asset_path)


def serialize_mcp_server(server: MCPServerConfig) -> MCPServerInfo:
    return MCPServerInfo(
        id=server.id,
        name=server.name,
        description=server.description or "",
        transport=server.transport,
        enabled=bool(server.enabled),
        command=server.command or "",
        args_json=server.args_json or [],
        env_json=server.env_json or {},
        url=server.url or "",
        headers_json=server.headers_json or {},
        uploaded_filename=server.uploaded_filename or "",
        uploaded_asset_dir=server.uploaded_asset_dir or "",
        uploaded_asset_path=server.uploaded_asset_path or "",
        created_by=server.created_by.username if getattr(server, "created_by", None) else None,
        created_at=server.created_at.isoformat(),
        updated_at=server.updated_at.isoformat(),
    )


def apply_mcp_server_payload(
    server: MCPServerConfig,
    *,
    name: str,
    description: str,
    transport: str,
    enabled: bool,
    command: str,
    args_json: list,
    env_json: dict,
    url: str,
    headers_json: dict,
) -> None:
    if transport not in MCP_TRANSPORTS:
        raise HTTPException(status_code=400, detail="transport 仅支持 stdio 或 http")

    normalized_name = (name or "").strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="名称不能为空")

    server.name = normalized_name
    server.description = (description or "").strip()
    server.transport = transport
    server.enabled = bool(enabled)
    server.updated_at = datetime.utcnow()

    if transport == "stdio":
        normalized_command = (command or "").strip()
        if not normalized_command:
            raise HTTPException(status_code=400, detail="stdio 模式必须填写 command")
        server.command = normalized_command
        server.args_json = args_json
        server.env_json = env_json
        server.url = ""
        server.headers_json = {}
    else:
        normalized_url = (url or "").strip()
        if not normalized_url:
            raise HTTPException(status_code=400, detail="http 模式必须填写 url")
        server.command = ""
        server.args_json = []
        server.env_json = {}
        server.url = normalized_url
        server.headers_json = headers_json


def is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
    )


def normalize_zip_member_path(member_name: str) -> str:
    normalized = member_name.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    normalized = os.path.normpath(normalized).replace("\\", "/")
    if normalized in {".", ".."} or normalized.startswith("../") or os.path.isabs(normalized):
        raise HTTPException(status_code=400, detail=f"zip 中存在非法路径: {member_name}")
    return normalized


def validate_skill_file_extension(relative_path: str) -> None:
    if relative_path.endswith("/"):
        return
    _, ext = os.path.splitext(relative_path.lower())
    if ext not in TEXT_RESOURCE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"skill 包含不支持的文件类型: {relative_path}")


def prepare_skill_import(zip_path: Path) -> tuple[str, Path, list[str]]:
    with tempfile.TemporaryDirectory(prefix="skill_import_") as temp_dir:
        extract_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                raise HTTPException(status_code=400, detail="zip 包为空")

            normalized_members: list[tuple[zipfile.ZipInfo, str]] = []
            for info in members:
                normalized = normalize_zip_member_path(info.filename)
                validate_skill_file_extension(normalized)
                normalized_members.append((info, normalized))

            skill_md_candidates = [
                normalized
                for _, normalized in normalized_members
                if normalized.endswith("/SKILL.md") or normalized == "SKILL.md"
            ]
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
            raise HTTPException(
                status_code=400,
                detail="skill name 仅允许字母、数字、下划线和短横线，且必须以字母或数字开头",
            )

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


def process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """后台执行耗时的解析、分块、向量化入库，并持续更新任务进度。"""
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        try:
            remove_bm25_stats_for_filename(filename)
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
    except Exception as exc:
        upload_job_manager.fail_job(job_id, failed_step, str(exc))


def process_delete_job(job_id: str, filename: str) -> None:
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
        remove_bm25_stats_for_filename(filename)
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

        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as exc:
        delete_job_manager.fail_job(job_id, failed_step, str(exc))
