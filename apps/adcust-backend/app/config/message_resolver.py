import json
from pathlib import Path
from typing import Any
import adcust_logic
from adcust_logic.exceptions.domain_exception import DomainException

class MessageResolver:
    def __init__(self, locale: str = "zh_CN"):
        # 自动定位到已安装或 editable 方式引入的 adcust_logic/locales。
        root_path = Path(adcust_logic.__file__).resolve().parent
        self.bundle_path = root_path / "locales" / f"{locale}.json"
        self._messages = self._load_messages()

    def _load_messages(self) -> dict:
        try:
            with open(self.bundle_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def resolve(self, exc: DomainException) -> str:
        # 1. 从词典中寻找 Key
        template = self._messages.get(exc.msg_key, exc.msg_key)
        # 2. 将 Exception 中的 context 参数注入模版 (Late Binding)
        try:
            return template.format(**exc.context)
        except (KeyError, ValueError):
            return template
