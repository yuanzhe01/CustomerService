import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.app.api._shared import MAX_SKILL_ZIP_BYTES, prepare_skill_import, save_upload_file
from backend.app.core.security import require_admin
from backend.app.db.models import User
from backend.app.schemas import SkillInfo, SkillListResponse, SkillUploadResponse
from backend.app.skills.skill_loader import SKILLS_DIR, scan_skill_index

router = APIRouter()


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
        await save_upload_file(file, zip_path)

        if zip_path.stat().st_size > MAX_SKILL_ZIP_BYTES:
            raise HTTPException(status_code=400, detail=f"skill zip 大小不能超过 {MAX_SKILL_ZIP_BYTES // (1024 * 1024)}MB")

        skill_name, target_dir, _imported_files = prepare_skill_import(zip_path)

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
