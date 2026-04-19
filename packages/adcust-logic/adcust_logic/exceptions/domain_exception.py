class DomainException(Exception):
    """
    [Pattern: Message Externalization]
    领域层异常基类。严禁一个文件包含多个类。
    """
    def __init__(self, msg_key: str, error_code: str = "INTERNAL_ERROR", **context):
        self.msg_key = msg_key
        self.error_code = error_code
        self.context = context
        super().__init__(self.msg_key)