from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import api as api_module
from database import init_db
import uvicorn
from settings import HOST, PORT

# 配置文件路径（找前端文件夹）
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Cute Cat Bot API")

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()

    # 3. 配置跨域：允许所有前端调用后端接口
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],    # 允许所有网址访问
        allow_credentials=True,
        allow_methods=["*"],    # 允许所有请求方式（GET/POST等）
        allow_headers=["*"],    # 允许所有请求头
    )

    # 4. 开发环境禁用缓存：修改HTML/JS/CSS后立即生效
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    # 5. 加载你写的所有后端API接口
    app.include_router(api_module.router)

    # 6. 如果有前端文件夹，就把根路径托管给前端页面
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
