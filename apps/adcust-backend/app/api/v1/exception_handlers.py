from fastapi import Request
from fastapi.responses import JSONResponse
from adcust_logic.exceptions.domain_exception import DomainException
from app.config.message_resolver import MessageResolver

# 初始化英文装配器
resolver = MessageResolver(locale="en_US")


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