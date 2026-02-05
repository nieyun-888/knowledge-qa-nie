import streamlit as st
import os
import sys
import subprocess
import requests
import json
import logging
import time
import concurrent.futures
from typing import List, Dict, Any

# ===================== 核心修复：自动安装系统依赖和Python依赖 =====================
def fix_dependencies():
    """自动修复Cloud环境依赖问题"""
    # 1. 仅在Streamlit Cloud环境执行
    if 'STREAMLIT_SERVER_TYPE' not in os.environ:
        return
    
    # 2. 安装系统库（解决libGL缺失）
    try:
        st.info("🔧 正在安装系统依赖库...")
        subprocess.run(
            ["apt-get", "update", "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        subprocess.run(
            ["apt-get", "install", "-y", "libgl1-mesa-glx", "libgomp1", "libglib2.0-0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except Exception as e:
        st.warning(f"⚠️ 系统库安装警告：{str(e)}")
    
    # 3. 强制安装兼容版本的Python依赖（解决huggingface-hub冲突）
    try:
        st.info("🔧 正在修复Python依赖版本...")
        # 降级huggingface-hub到兼容版本
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface-hub==0.19.4", "--force-reinstall"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # 锁定transformers版本
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers==4.36.2", "--force-reinstall"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # 安装无头版OpenCV
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.8.0.76", "--force-reinstall"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        st.success("✅ 依赖修复完成！")
    except Exception as e:
        st.error(f"❌ 依赖修复失败：{str(e)}")

# 执行依赖修复（仅首次运行）
if "deps_fixed" not in st.session_state:
    fix_dependencies()
    st.session_state.deps_fixed = True

# ===================== 导入自定义模块 =====================
try:
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
except ImportError:
    # Cloud环境添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
from math import floor

# ===================== 全局配置 =====================
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="冰姐问答小课堂",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 极简CSS（避免DOM冲突）
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== 全局状态初始化 =====================
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
    st.session_state.total_cost = 0.0
    st.session_state.current_tokens = 0
    st.session_state.current_cost = 0.0
    st.session_state.prompt_tokens = 0
    st.session_state.completion_tokens = 0
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image_question" not in st.session_state:
    st.session_state.image_question = ""
if "image_processed" not in st.session_state:
    st.session_state.image_processed = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

# ===================== Cloud环境工具函数 =====================
def is_streamlit_cloud():
    """判断是否在Streamlit Cloud环境"""
    return 'STREAMLIT_SERVER_TYPE' in os.environ

def get_chroma_db_path():
    """获取向量库路径（兼容本地/Cloud）"""
    if is_streamlit_cloud():
        return "/tmp/chroma_db"
    return "./chroma_db"

# ===================== DeepSeek API 类 =====================
class DeepSeekAPI:
    """DeepSeek API 管理类 - 兼容Cloud环境"""

    def __init__(self):
        # DeepSeek 官方 API 地址
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model_name = "deepseek-chat"

        # 初始化API Key
        if 'api_key' not in st.session_state:
            try:
                st.session_state.api_key = st.secrets.get("DEEPSEEK_API_KEY", None)
            except Exception:
                st.session_state.api_key = None
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = st.session_state.api_key is not None
    
    @property
    def api_key(self):
        return st.session_state.api_key

    def set_api_key(self, api_key: str):
        """设置API密钥"""
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            st.session_state.api_key_set = True
            if len(api_key) > 12:
                st.session_state.api_key_preview = api_key[:8] + "..." + api_key[-4:]
            else:
                st.session_state.api_key_preview = api_key
            return True
        return False

    def login_with_password(self, password: str) -> bool:
        """简单密码登录"""
        if password == "123456":
            st.session_state.api_key = "sk-4f3e29df9fa54da8bd601ae780111df1"
            st.session_state.api_key_set = True
            st.session_state.api_key_preview = "sk-4f3e...1df1"
            return True
        return False

    def is_logged_in(self) -> bool:
        return st.session_state.api_key is not None and st.session_state.api_key.strip() != ""

    def test_api_connection(self) -> Dict:
        """测试API连通性"""
        if not self.is_logged_in():
            return {"success": False, "message": "未设置 API Key"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "请回复'测试成功'"}],
            "max_tokens": 30,
            "temperature": 0.0
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=test_data, timeout=20)
            if resp.status_code == 200:
                result = resp.json()
                answer = result["choices"][0]["message"]["content"]
                return {"success": True, "message": f"✅ DeepSeek API 连接成功\n回复：{answer}"}
            else:
                err = resp.json() if resp.headers.get('Content-Type') == 'application/json' else resp.text
                return {"success": False, "message": f"❌ API 错误 {resp.status_code}", "detail": str(err)[:200]}
        except Exception as e:
            return {"success": False, "message": f"网络/系统错误：{str(e)}"}

    def get_answer(self, question: str, contexts: List[Dict], conversation_history: List[dict] = None) -> str:
        """获取答案 - 核心问答逻辑"""
        if not self.is_logged_in():
            return "请先在侧边栏设置 DeepSeek API Key"

        # 构建上下文
        context_text = self._build_context_text(contexts)
        
        # 构建对话历史
        history_text = ""
        if conversation_history:
            history_text = "\n\n之前的对话历史：\n"
            for msg in conversation_history[-6:]:
                role = "用户" if msg["role"] == "user" else "冰姐"
                history_text += f"{role}：{msg['content']}\n"

        # 构建提示词
        prompt = f"""你是专业的知识问答助手"冰姐"，请根据提供的上下文和对话历史回答问题，信息不足时可补充专业知识。

上下文信息：
{context_text}

{history_text}
用户问题：{question}

请用温暖亲切的语气，按照以下结构回答（逻辑不通时可灵活调整）：
## 🔍 思考过程
乖，看完你的问题后，冰姐的答案是：[简洁答案]

### 参考资料
1. 《文档名》第X页：[相关内容]
2. 《文档名》第X页：[相关内容]

## 💡 综合答案
[详细答案]

## 📚 推荐阅读
[针对性的学习建议]"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求数据
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是亲切专业的法律教育助手冰姐，擅长结合上下文和对话历史给出连贯、准确的回答"},
            ] + ([msg for msg in conversation_history[-5:]] if conversation_history else []) + [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }

        # 带重试的API调用
        max_retry = 1
        retry_count = 0
        while retry_count <= max_retry:
            try:
                resp = requests.post(self.base_url, headers=headers, json=data, timeout=(10, 120))
                resp.raise_for_status()
                result = resp.json()
                
                # Token统计
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # 费用计算
                PRICE_PER_1000_TOKENS = 0.0015
                current_cost = (total_tokens * PRICE_PER_1000_TOKENS) / 1000
                
                # 更新状态
                st.session_state.total_tokens_used += total_tokens
                st.session_state.total_cost += current_cost
                st.session_state.current_tokens = total_tokens
                st.session_state.current_cost = current_cost
                st.session_state.prompt_tokens = prompt_tokens
                st.session_state.completion_tokens = completion_tokens
                
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                retry_count += 1
                if retry_count > max_retry:
                    return f"❌ 回答失败：{str(e)}"
                time.sleep(0.5)

    def _build_context_text(self, contexts: List[Dict]) -> str:
        """构建上下文文本"""
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('source', '未知文档')
            page = ctx.get('page', '未知页码')
            content = ctx.get('content', '')
            doc_name = os.path.basename(source).rsplit('.', 1)[0] if '.' in source else source
            
            context_text += f"\n【资料{i}】《{doc_name}》第{page}页：{content[:600]}...\n"
        return context_text

# ===================== 向量库初始化 =====================
def init_vector_store_actual():
    """实际初始化向量库"""
    try:
        chroma_db_path = get_chroma_db_path()
        
        # 创建向量库实例
        vector_store = SmartVectorStore(persist_directory=chroma_db_path)
        
        # 加载现有向量库
        if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
            if vector_store.load_existing_vector_store():
                # 测试检索
                try:
                    vector_store.search_similar_documents("测试", k=1)
                    return vector_store, f"✅ 知识库加载成功（环境：{'Cloud' if is_streamlit_cloud() else '本地'}）"
                except Exception as e:
                    return vector_store, f"⚠️ 检索测试失败：{str(e)}"
            else:
                return None, "❌ 向量库加载失败"
        else:
            return vector_store, "ℹ️ 未找到向量库，基础功能正常"
    except Exception as e:
        return None, f"❌ 初始化失败: {str(e)}"

def initialize_vector_store():
    """带进度条的向量库初始化"""
    if st.session_state.vector_store_initialized:
        return st.session_state.vector_store
    
    placeholder = st.empty()
    with placeholder.container():
        st.info("🔄 正在初始化知识库...")
        progress_bar = st.progress(0)
        
        try:
            progress_bar.progress(20)
            time.sleep(0.5)
            
            progress_bar.progress(60)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_vector_store_actual)
                vector_store, message = future.result(timeout=30)
            
            progress_bar.progress(100)
            time.sleep(0.5)
            placeholder.empty()
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.vector_store_initialized = True
                st.success(message)
                return vector_store
            else:
                st.error(message)
                return None
        except concurrent.futures.TimeoutError:
            placeholder.empty()
            st.error("⚠️ 知识库初始化超时")
            return None
        except Exception as e:
            placeholder.empty()
            st.error(f"❌ 初始化异常: {str(e)}")
            return None

# ===================== 图片上传/OCR功能 =====================
def add_image_upload_section():
    """图片上传和OCR识别"""
    with st.expander("📷 图片识别提问", expanded=False):
        uploaded_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg', 'bmp'], key="image_uploader")
        
        if uploaded_file is not None:
            # 显示图片预览
            image_processor.display_image_preview(uploaded_file, "识别图片")
            
            # OCR识别
            if st.session_state.get('last_uploaded_file') != uploaded_file.name:
                with st.spinner("正在识别文字..."):
                    result = image_processor.process_uploaded_image(uploaded_file)
                st.session_state.image_processed = True
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.ocr_result = result
            else:
                result = st.session_state.ocr_result
            
            # 处理识别结果
            if result["success"]:
                st.success("✅ 文字识别成功！")
                recognized_text = st.text_area("识别结果（可编辑）", result["text"], height=150, key="ocr_text")
                additional_question = st.text_input("补充问题（可选）", key="ocr_question")
                
                if st.button("使用此文字提问", key="ocr_submit"):
                    if recognized_text.strip():
                        full_question = f"{additional_question}\n\n{recognized_text}" if additional_question else recognized_text
                        st.session_state.image_question = full_question
                        st.session_state.image_processed = False
                        st.success("✅ 问题已准备，正在处理...")
                        st.rerun()
                    else:
                        st.error("请输入有效文字")
            else:
                st.error(f"❌ 识别失败：{result['message']}")
        
        # 重置按钮
        if st.button("重置图片上传", key="ocr_reset"):
            st.session_state.image_processed = False
            st.session_state.last_uploaded_file = None
            st.session_state.image_question = ""
            st.rerun()

# ===================== 主函数 =====================
def main():
    # 页面标题
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🎓 冰姐问答小课堂</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化API
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    deepseek_api = st.session_state.deepseek_api
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API 设置")
        st.info(f"API端点: `{deepseek_api.base_url}`")
        
        # 登录方式选择
        login_method = st.radio("登录方式", ["直接输入API", "密码登录"])
        
        if login_method == "直接输入API":
            api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...", key="api_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("设置API", key="set_api"):
                    if deepseek_api.set_api_key(api_key):
                        st.success("✅ API已设置")
                        st.rerun()
                    else:
                        st.error("❌ 请输入有效API Key")
            with col2:
                if st.button("测试连接", key="test_api"):
                    result = deepseek_api.test_api_connection()
                    st.success(result["message"]) if result["success"] else st.error(result["message"])
        else:
            password = st.text_input("密码", type="password", placeholder="默认：123456", key="pwd_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("登录", key="login"):
                    if deepseek_api.login_with_password(password):
                        st.success("✅ 登录成功")
                        st.rerun()
                    else:
                        st.error("❌ 密码错误")
            with col2:
                if st.button("测试连接", key="test_api2"):
                    result = deepseek_api.test_api_connection()
                    st.success(result["message"]) if result["success"] else st.error(result["message"])
        
        st.markdown("---")
        
        # 知识库初始化
        st.markdown("### 📚 知识库状态")
        if st.button("🔄 初始化知识库", key="init_kb"):
            initialize_vector_store()
            st.rerun()
        
        # 显示知识库状态
        if st.session_state.vector_store_initialized:
            st.success("✅ 知识库已就绪")
        else:
            st.warning("⚠️ 知识库未初始化")
        
        st.markdown("---")
        
        # 费用统计
        st.markdown("### 💰 费用统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("累计Token", f"{st.session_state.total_tokens_used:,}")
        with col2:
            st.metric("累计费用", f"¥{st.session_state.total_cost:.4f}")
        
        if st.session_state.current_tokens > 0:
            with st.expander("本次详情"):
                st.write(f"输入Token: {st.session_state.prompt_tokens}")
                st.write(f"输出Token: {st.session_state.completion_tokens}")
                st.write(f"费用: ¥{st.session_state.current_cost:.6f}")
        
        st.markdown("---")
        st.warning("**当前模式：仅检索模式**")

    # 主界面布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 对话界面")
        # 显示聊天历史
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    with col2:
        # 图片上传功能
        add_image_upload_section()
        
        st.markdown("---")
        
        # 知识库统计
        st.markdown("### 📊 知识库统计")
        if st.session_state.vector_store_initialized:
            try:
                vector_store = st.session_state.vector_store
                count = vector_store.vector_store._collection.count()
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("文本块数", f"{count:,}")
                with col_stat2:
                    st.metric("向量维度", "512")
                with st.expander("详细信息", expanded=True):
                    st.write(f"• 文档数：~{max(1, floor(count//80))}")
                    st.write(f"• 模型：BGE-small-ZH_v1.5")
                    st.progress(min(count/1000, 1.0))
            except Exception as e:
                st.info("✅ 知识库已加载，统计信息暂不可用")
        else:
            st.warning("⚠️ 请先初始化知识库")

    # 处理用户输入
    current_prompt = None
    # 处理图片识别的问题
    if st.session_state.image_question:
        current_prompt = st.session_state.image_question
        st.session_state.image_question = ""
    # 处理普通聊天输入
    elif user_input := st.chat_input("乖，你有哪个地方不明白呢"):
        current_prompt = user_input

    # 处理提问
    if current_prompt:
        # 检查前置条件
        if not deepseek_api.is_logged_in():
            st.warning("🔑 请先设置API Key")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "请先在侧边栏设置DeepSeek API Key"})
            st.rerun()
        elif not st.session_state.vector_store_initialized:
            st.warning("📚 请先初始化知识库")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "请先在侧边栏初始化知识库"})
            st.rerun()
        else:
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            
            # 生成回答
            with col1:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("🧠 思考中...")
                    
                    try:
                        # 检索相关文档
                        vector_store = st.session_state.vector_store
                        with st.spinner("🔍 检索资料中..."):
                            search_results = vector_store.search_similar_documents(current_prompt, k=8)
                        
                        if not search_results:
                            answer = "乖，这个问题冰姐暂时没有找到相关资料，课后我再详细给你讲哈～"
                        else:
                            # 构建上下文
                            contexts = [{
                                'content': doc.page_content,
                                'source': doc.metadata.get('source', '未知'),
                                'page': doc.metadata.get('page', '未知')
                            } for doc in search_results]
                            # 获取回答
                            with st.spinner("📝 冰姐正在整理答案..."):
                                answer = deepseek_api.get_answer(current_prompt, contexts, st.session_state.messages[:-1])
                        
                        # 显示答案
                        placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # 显示参考资料
                        with st.expander("📄 参考资料", expanded=False):
                            for i, doc in enumerate(search_results[:5], 1):
                                source = doc.metadata.get('source', '未知')
                                page = doc.metadata.get('page', '未知')
                                doc_name = os.path.basename(source).rsplit('.', 1)[0] if '.' in source else source
                                st.markdown(f"**资料{i}：《{doc_name}》第{page}页**")
                                st.caption(doc.page_content[:200] + "...")
                                st.divider()
                    except Exception as e:
                        error_msg = f"❌ 处理失败：{str(e)}"
                        placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ===================== 程序入口 =====================
if __name__ == "__main__":
    main()