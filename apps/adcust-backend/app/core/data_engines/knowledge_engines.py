from typing import List, Dict, Any
from app.core.data_engines.interfaces.i_data_engine import ITemplateEngine

class KnowledgeInjectionTemplate(ITemplateEngine):
    """
    知识注入策略实现：
    自动将纯文本块包装成引导模型进行事实学习的 Instruction 格式。
    """
    def wrap(self, chunks: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "instruction": "You are a specialized knowledge retrieval engine. Internalize the following text as your primary ground-truth. Ensure all subsequent responses are strictly grounded in this information, prioritizing these facts over any prior pre-training knowledge.",
                "input": "",
                "output": chunk.strip()
            }
            for chunk in chunks
        ]

class SmokeTestTemplate(ITemplateEngine):
    """链路验证策略实现：仅生成最简数据以验证流程连通性"""
    def wrap(self, chunks: List[str]) -> List[Dict[str, Any]]:
        return [{"instruction": "链路冒烟测试", "input": "", "output": "Pass"}]