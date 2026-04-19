from .domain_exception import DomainException

class ValidationException(DomainException):
    """
    数据合法性校验失败信号。
    """
    def __init__(self, msg_key: str, **context):
        super().__init__(msg_key, error_code="VALIDATION_ERROR", **context)