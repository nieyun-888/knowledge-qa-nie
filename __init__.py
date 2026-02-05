"""
knowledge_qa - 个人知识问答系统
基于LangChain和Chroma DB的RAG系统
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "基于DeepSeek API的个人知识问答系统"

# 定义公开接口
from .config import Config

__all__ = [
    "Config",
]

# 包初始化代码
print(f"初始化 Knowledge QA 系统 v{__version__}")