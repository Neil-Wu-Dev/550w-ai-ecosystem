import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 必须增加跨域支持
from app.api.v1.endpoints import workflow
from app.core.dependencies import setup_dependencies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="AdCust Backend",
        description="Industrial LLM Customization and Inference Comparison Lab",
        version="1.0.0"
    )

    # --- 重要：增加跨域中间件，防止前端请求被拦截 ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体前端地址
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 1. 挂载路由 ---
    # 注意：这里的 prefix 决定了你所有接口的开头
    app.include_router(
        workflow.router,
        prefix="/api/v1/workflow",
        tags=["AdCust-Workflow"]
    )

    # --- 2. 依赖注入点火 ---
    try:
        setup_dependencies(app)
        logger.info("--- [AdCust] 依赖注入成功，单例已就绪 ---")
    except Exception as e:
        logger.error(f"--- [AdCust] 依赖注入点火失败: {str(e)} ---")
        raise e

    return app

app = create_app()

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "AdCust Engine Online",
        "status": "ready",
        "docs_url": "/docs" # 提示你去哪里看接口
    }

if __name__ == "__main__":
    logger.info("Starting Uvicorn Server on http://127.0.0.1:8000")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )