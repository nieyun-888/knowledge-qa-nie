"""
knowledge_qa 核心模块
包含PDF处理和向量存储功能
"""

from .pdf_processor import PDFProcessor

# 修复导入问题
try:
    from .vector_store import VectorStoreManager
except ImportError:
    # 如果 VectorStoreManager 不存在，使用 VectorStore 作为替代
    from .vector_store import VectorStore as VectorStoreManager

# 版本信息
__version__ = "0.1.0"
__author__ = "Your Name"

# 定义公开的API接口
__all__ = [
    "PDFProcessor",
    "VectorStoreManager",
]

# 包级别的配置
DEFAULT_SETTINGS = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "BAAI/bge-small-zh-v1.5"
}

# 初始化日志配置（可选）
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

print(f"加载 knowledge_qa.src 模块 v{__version__}")