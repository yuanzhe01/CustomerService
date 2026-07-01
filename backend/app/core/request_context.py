from __future__ import annotations
from dataclasses import dataclass
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from backend.app.db.models import User
from backend.app.db.session import SessionLocal


DEFAULT_USER_ID = "default_user"
DEFAULT_ROLE = "user"
USER_HEADER = "X-User-Id"
ROLE_HEADER = "X-User-Role"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _normalize_user_id(user_id: str | None) -> str:
    value = (user_id or "").strip()
    return value or DEFAULT_USER_ID


def _normalize_role(role: str | None) -> str:
    value = (role or "").strip().lower()
    return value if value in {"user", "admin"} else DEFAULT_ROLE


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    role: str = DEFAULT_ROLE


def get_request_context(request: Request) -> RequestContext:
    user_id = _normalize_user_id(request.headers.get(USER_HEADER))
    role = _normalize_role(request.headers.get(ROLE_HEADER))
    return RequestContext(user_id=user_id, role=role)


def get_or_create_user(db: Session, user_id: str, role: str = DEFAULT_ROLE) -> User:
    normalized_user_id = _normalize_user_id(user_id)
    normalized_role = _normalize_role(role)

    user = db.query(User).filter(User.username == normalized_user_id).first()
    if user:
        if user.role != normalized_role and normalized_role in {"user", "admin"}:
            user.role = normalized_role
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    user = User(
        username=normalized_user_id,
        password_hash=normalized_user_id,
        role=normalized_role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    ctx = get_request_context(request)
    return get_or_create_user(db, ctx.user_id, ctx.role)


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if _normalize_role(getattr(current_user, "role", None)) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可访问",
        )
    return current_user
