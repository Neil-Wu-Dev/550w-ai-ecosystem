from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from adcust_logic.exceptions.domain_exception import DomainException
from app.config.message_resolver import MessageResolver

# 桌面端 UI 使用英文，因此 API 默认返回英文可读错误。
resolver = MessageResolver(locale="en_US")
logger = logging.getLogger(__name__)


async def domain_exception_handler(request: Request, exc: DomainException):
    """
    [Gatekeeper for v1 API]
    拦截所有领域异常，并从 config.message_resolver 获取英文描述。
    """
    # 动态装配消息
    display_message = resolver.resolve(exc)

    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": display_message
        }
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """把未包装异常转换为前端可读 JSON，同时保留完整后端 traceback。"""
    logger.exception("Unhandled API exception on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_RUNTIME_ERROR",
            "message": str(exc) or exc.__class__.__name__,
        },
    )
