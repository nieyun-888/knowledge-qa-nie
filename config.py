import os
import sys

class Config:
    """全局配置类 - 适配本地/Streamlit Cloud双环境"""
    # ================= 核心路径配置（适配Linux/Windows） =================
    # 核心调整：Cloud环境强制使用/tmp目录（唯一可写路径）
    if "STREAMLIT_SERVER_BASEURL_PATH" in os.environ or "STREAMLIT" in sys.modules:
        CHROMA_DB_PATH = os.path.join("/tmp", "chroma_db").replace("\\", "/")
    else:
        CHROMA_DB_PATH = os.path.join(".", "chroma_db").replace("\\", "/")
    COLLECTION_NAME = "pdf_documents"  # 与vector_store.py保持一致
    
    # ================= 模型配置（与vector_store.py对齐） =================
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # ================= DeepSeek API配置（与app.py对齐） =================
    # 核心调整：优先从Streamlit Secrets读取，适配Cloud密钥管理
    DEEPSEEK_API_BASE = os.getenv(
        "DEEPSEEK_API_BASE", 
        st.secrets.get("DEEPSEEK_API_BASE", "https://api.lkeap.cloud.tencent.com/v1") if "st" in locals() else "https://api.lkeap.cloud.tencent.com/v1"
    )
    DEEPSEEK_MODEL = os.getenv(
        "DEEPSEEK_MODEL", 
        st.secrets.get("DEEPSEEK_MODEL", "deepseek-v3.1") if "st" in locals() else "deepseek-v3.1"
    )
    # 备用官方端点（与app.py兼容）
    DEEPSEEK_OFFICIAL_BASE = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_OFFICIAL_MODEL = "deepseek-chat"

    # ================= 运行环境配置 =================
    @classmethod
    def is_streamlit_cloud(cls) -> bool:
        """判断是否运行在Streamlit Cloud环境"""
        return "STREAMLIT_SERVER_BASEURL_PATH" in os.environ or "STREAMLIT" in sys.modules

    @classmethod
    def create_directories(cls):
        """创建必要的目录（增加异常处理，适配Cloud权限）"""
        try:
            os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)
            if cls.is_streamlit_cloud():
                print(f"✅ [Cloud环境] 已创建目录: {cls.CHROMA_DB_PATH}")
            else:
                print(f"✅ [本地环境] 已创建目录: {cls.CHROMA_DB_PATH}")
        except Exception as e:
            print(f"❌ 创建目录异常: {e}")

    @classmethod
    def get_deepseek_config(cls, use_official: bool = False) -> dict:
        """获取DeepSeek API配置（统一对外接口，方便调用）"""
        if use_official:
            return {
                "base_url": cls.DEEPSEEK_OFFICIAL_BASE,
                "model_name": cls.DEEPSEEK_OFFICIAL_MODEL
            }
        return {
            "base_url": cls.DEEPSEEK_API_BASE,
            "model_name": cls.DEEPSEEK_MODEL
        }

# ================= 初始化配置（可选） =================
# 核心调整：Cloud环境延迟初始化，避免导入时出错
if __name__ != "__main__" and not Config.is_streamlit_cloud():
    Config.create_directories()