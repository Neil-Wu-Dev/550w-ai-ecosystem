import os
import fitz  # PyMuPDF 的引入名
from typing import List
from domain_logic.interfaces.engines.i_data_engine import IDataParser, ITextSplitter


class PdfParser(IDataParser):
    """
    生产级 PDF 文档解析实现
    支持多页文本提取、自动清理空白字符。
    """

    def extract_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")

        print(f"--- [Data Engine] 正在物理扫描并提取 PDF: {file_path} ---")

        full_text = []
        try:
            # 使用 fitz (PyMuPDF) 打开文档
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # 提取纯文本
                    text = page.get_text("text")
                    if text.strip():
                        full_text.append(text)

            # 合并文本并进行基础清洗（去除冗余换行、空格）
            final_content = "\n".join(full_text)

            if not final_content.strip():
                print(f"⚠️ 警告: PDF {file_path} 似乎是扫描件或空文档，未提取到文字。")

            return final_content

        except Exception as e:
            print(f"❌ PDF解析发生致命错误: {str(e)}")
            return ""


class SimpleTextSplitter(ITextSplitter):
    """
    生产级文本切分实现
    优化点：增加对文本长度的保护，防止产生空块。
    """

    def split(self, text: str, chunk_size: int) -> List[str]:
        if not text:
            return []

        # 确保 chunk_size 有效
        chunk_size = max(10, chunk_size)

        # 按照物理字符长度切分
        chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

        # 过滤掉纯空格的无效块
        return [c.strip() for c in chunks if c.strip()]