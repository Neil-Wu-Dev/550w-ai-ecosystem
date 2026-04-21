# -*- coding: utf-8 -*-
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入拆分后的所有路由，包括新增的 compute_api
from app.api.v1.endpoints import (
    model_api,
    dataset_api,
    training_api,
    inference_api,
    compute_api  # <--- 新增挂载点
)
from app.config.dependencies import setup_dependencies

# 1. 导入领域异常基类与拦截守卫
from adcust_logic.exceptions.domain_exception import DomainException
from app.api.v1.exception_handlers import domain_exception_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="AdCust Backend",
        description="Industrial LLM Customization and Inference Comparison Lab",
        version="1.1.0"
    )

    # 跨域支持 (允许 Tauri 访问)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 【关键防御网：挂载异常拦截器】 ---
    app.add_exception_handler(DomainException, domain_exception_handler)

    # --- 1. 挂载资源路由 (严格按模块分发) ---
    app.include_router(model_api.router, prefix="/api/v1/model", tags=["Model"])
    app.include_router(dataset_api.router, prefix="/api/v1/dataset", tags=["Dataset"])
    app.include_router(training_api.router, prefix="/api/v1/training", tags=["Training"])
    app.include_router(inference_api.router, prefix="/api/v1/inference", tags=["Inference"])
    # 挂载新增的算力管理接口
    app.include_router(compute_api.router, prefix="/api/v1/compute", tags=["Compute"])

    # --- 2. 依赖注入点火 (依赖 logic 层所有单例服务) ---
    try:
        setup_dependencies(app)
        logger.info("--- [AdCust] 依赖注入成功，算力引擎与训练服务已就绪 ---")
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
        "version": "1.1.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    logger.info("Starting AdCust Uvicorn Server on http://127.0.0.1:8000")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )