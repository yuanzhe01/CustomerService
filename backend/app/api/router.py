from fastapi import APIRouter

from . import chat, documents, mcp, skills

router = APIRouter()
router.include_router(chat.router)
router.include_router(documents.router)
router.include_router(skills.router)
router.include_router(mcp.router)
