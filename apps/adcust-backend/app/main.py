import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# 导入拆分后的新路由
from app.api.v1.endpoints import model_api, dataset_api, training_api, inference_api
from app.config.dependencies import setup_dependencies

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

    # 跨域支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 1. 挂载拆分后的资源路由 ---
    # 路径由 main.py 统一分配，确保不重复
    app.include_router(model_api.router, prefix="/api/v1/model", tags=["Model"])
    app.include_router(dataset_api.router, prefix="/api/v1/dataset", tags=["Dataset"])
    app.include_router(training_api.router, prefix="/api/v1/training", tags=["Training"])
    app.include_router(inference_api.router, prefix="/api/v1/inference", tags=["Inference"])

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
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    logger.info("Starting Uvicorn Server on http://127.0.0.1:8000")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )