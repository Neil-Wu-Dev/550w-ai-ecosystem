from .domain_exception import DomainException

class BusinessException(DomainException):
    """
    违反业务规则信号（如状态机流转错误）。
    """
    def __init__(self, msg_key: str, **context):
        super().__init__(msg_key, error_code="RULE_VIOLATION", **context)