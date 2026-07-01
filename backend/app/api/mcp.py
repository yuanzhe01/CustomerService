from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.api._shared import (
    apply_mcp_server_payload,
    delete_mcp_server_assets,
    parse_json_form_field,
    save_mcp_server_asset,
    serialize_mcp_server,
)
from backend.app.core.request_context import get_db, require_admin
from backend.app.db.models import MCPServerConfig, User
from backend.app.schemas import MCPServerDeleteResponse, MCPServerListResponse, MCPServerMutationResponse
from backend.app.services.agent_service import invalidate_mcp_runtime_cache

router = APIRouter()


@router.get("/admin/mcp-servers", response_model=MCPServerListResponse)
async def list_mcp_servers(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = db.query(MCPServerConfig).order_by(MCPServerConfig.updated_at.desc(), MCPServerConfig.id.desc()).all()
    return MCPServerListResponse(servers=[serialize_mcp_server(row) for row in rows])


@router.post("/admin/mcp-servers", response_model=MCPServerMutationResponse)
async def create_mcp_server(
    name: str = Form(...),
    description: str = Form(""),
    transport: str = Form(...),
    enabled: bool = Form(True),
    command: str = Form(""),
    args_json: str = Form("[]"),
    env_json: str = Form("{}"),
    url: str = Form(""),
    headers_json: str = Form("{}"),
    asset_file: UploadFile | None = File(None),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    normalized_name = (name or "").strip()
    if db.query(MCPServerConfig).filter(MCPServerConfig.name == normalized_name).first():
        raise HTTPException(status_code=409, detail="MCP 配置名称已存在")

    parsed_args = parse_json_form_field(args_json, list, "args_json", [])
    parsed_env = parse_json_form_field(env_json, dict, "env_json", {})
    parsed_headers = parse_json_form_field(headers_json, dict, "headers_json", {})

    server = MCPServerConfig(created_by_user_id=current_user.id)
    apply_mcp_server_payload(
        server,
        name=normalized_name,
        description=description,
        transport=(transport or "").strip().lower(),
        enabled=enabled,
        command=command,
        args_json=parsed_args,
        env_json=parsed_env,
        url=url,
        headers_json=parsed_headers,
    )

    db.add(server)
    db.commit()
    db.refresh(server)

    try:
        if asset_file is not None:
            await save_mcp_server_asset(server, asset_file)
            server.updated_at = datetime.utcnow()
            db.add(server)
            db.commit()
            db.refresh(server)
    except Exception:
        db.delete(server)
        db.commit()
        raise

    invalidate_mcp_runtime_cache()
    return MCPServerMutationResponse(
        server=serialize_mcp_server(server),
        message=f"MCP 配置 '{server.name}' 创建成功",
    )


@router.put("/admin/mcp-servers/{server_id}", response_model=MCPServerMutationResponse)
async def update_mcp_server(
    server_id: int,
    name: str = Form(...),
    description: str = Form(""),
    transport: str = Form(...),
    enabled: bool = Form(True),
    command: str = Form(""),
    args_json: str = Form("[]"),
    env_json: str = Form("{}"),
    url: str = Form(""),
    headers_json: str = Form("{}"),
    asset_file: UploadFile | None = File(None),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    server = db.query(MCPServerConfig).filter(MCPServerConfig.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="MCP 配置不存在")

    normalized_name = (name or "").strip()
    duplicate = (
        db.query(MCPServerConfig)
        .filter(MCPServerConfig.name == normalized_name, MCPServerConfig.id != server_id)
        .first()
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="MCP 配置名称已存在")

    parsed_args = parse_json_form_field(args_json, list, "args_json", [])
    parsed_env = parse_json_form_field(env_json, dict, "env_json", {})
    parsed_headers = parse_json_form_field(headers_json, dict, "headers_json", {})

    previous_transport = server.transport
    apply_mcp_server_payload(
        server,
        name=normalized_name,
        description=description,
        transport=(transport or "").strip().lower(),
        enabled=enabled,
        command=command,
        args_json=parsed_args,
        env_json=parsed_env,
        url=url,
        headers_json=parsed_headers,
    )

    if server.transport != "stdio" and previous_transport == "stdio" and server.uploaded_filename:
        delete_mcp_server_assets(server)
        server.uploaded_filename = ""
        server.uploaded_asset_dir = ""
        server.uploaded_asset_path = ""

    if asset_file is not None:
        await save_mcp_server_asset(server, asset_file)

    db.add(server)
    db.commit()
    db.refresh(server)
    invalidate_mcp_runtime_cache()

    return MCPServerMutationResponse(
        server=serialize_mcp_server(server),
        message=f"MCP 配置 '{server.name}' 更新成功",
    )


@router.delete("/admin/mcp-servers/{server_id}", response_model=MCPServerDeleteResponse)
async def delete_mcp_server(server_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    server = db.query(MCPServerConfig).filter(MCPServerConfig.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="MCP 配置不存在")

    delete_mcp_server_assets(server)
    db.delete(server)
    db.commit()
    invalidate_mcp_runtime_cache()
    return MCPServerDeleteResponse(id=server_id, message="MCP 配置已删除")
