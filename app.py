import streamlit as st
import os
import requests
import json
import logging
import time
import concurrent.futures
from typing import List, Dict, Any
try:
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
except ImportError:
    # Cloud环境添加当前目录到Python路径
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
from math import floor

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

# 全局初始化
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

# 添加自定义CSS实现按钮拉伸效果（兼容所有Streamlit版本）
st.markdown("""
    <style>
    /* 让侧边栏按钮全屏宽度 */
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
    }
    /* 优化聊天界面样式 */
    div.stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

class DeepSeekAPI:
    """DeepSeek API 管理类 - 官方原生版本"""

    def __init__(self):
        # DeepSeek 官方 API 地址
        self.base_url = st.secrets.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1/chat/completions")
        self.model_name = "deepseek-chat"  # 官方标准模型名

        # 核心修复：先判断Secrets是否可用，本地无则跳过
        if 'api_key' not in st.session_state:
            try:
                # 仅当Secrets存在时才读取，避免本地报错
                st.session_state.api_key = st.secrets.get("DEEPSEEK_API_KEY", None)
            except Exception:
                # 本地环境：Secrets不存在，设为None（沿用手动输入逻辑）
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
        """简单密码登录，自动填入内置 API Key"""
        if password == "123456":
            # 使用内置测试密钥
            st.session_state.api_key = "sk-4f3e29df9fa54da8bd601ae780111df1"
            st.session_state.api_key_set = True
            st.session_state.api_key_preview = "sk-4f3e...1df1"
            return True
        return False

    def is_logged_in(self) -> bool:
        return st.session_state.api_key is not None and st.session_state.api_key.strip() != ""

    def test_api_connection(self) -> Dict:
        """测试官方 API 连通性"""
        if not self.is_logged_in():
            return {"success": False, "message": "未设置 API Key"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "请回复'测试成功'"}
            ],
            "max_tokens": 30,
            "temperature": 0.0
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=test_data, timeout=20)

            if resp.status_code == 200:
                result = resp.json()
                answer = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "message": f"✅ DeepSeek 官方 API 连接成功\n回复：{answer}"
                }
            else:
                try:
                    err = resp.json()
                    msg = err.get("error", {}).get("message", resp.text)
                except:
                    msg = resp.text[:200]

                return {
                    "success": False,
                    "message": f"❌ API 错误 {resp.status_code}",
                    "detail": msg
                }

        except requests.exceptions.RequestException as e:
            return {"success": False, "message": f"网络错误：{str(e)}"}
        except Exception as e:
            return {"success": False, "message": f"异常：{str(e)}"}

    def get_answer(self, question: str, contexts: List[Dict], conversation_history: List[dict] = None) -> str:
        """获取答案 - 采用第二个代码的模板"""
        if not self.is_logged_in():
            return "请先设置 DeepSeek API Key"

        # 构建上下文文本
        context_text = self._build_context_text(contexts)
        
        # 构建对话历史文本
        history_text = ""
        if conversation_history:
            history_text = "\n\n之前的对话历史：\n"
            for msg in conversation_history[-6:]:  # 只保留最近6条消息
                role = "用户" if msg["role"] == "user" else "冰姐"
                history_text += f"{role}：{msg['content']}\n"

        prompt = f"""你是一个专业的知识问答助手"冰姐"。我将给你一些参考资料和对话历史，你主要根据提供的上下文信息回答用户问题，如果你觉得信息不够，则你就需要自己去网上去查找一下答案。

上下文信息来自多个文档资料：
{context_text}

{history_text}

用户问题：{question}

请特别注意：当前用户的问题"{question}"可能是基于之前对话的延续。请仔细理解对话历史，确保回答与之前的对话内容连贯一致。

请你主要按照以下格式回答，如果你觉得逻辑不畅，也可以适当的改变一下方式：

## 🔍 思考过程

乖，看完你的问题之后，冰姐仔细思考了一下，先给你说我的答案是：
[这里给出简洁明确的答案]

在咱们之前的教材里也有涉及你的疑问，我给你找了出来，你可以在这里看看，也可以去教材里去看详细的内容：

### 1. 第一个资料
- **文档名称**：《文档名称》
- **页码**：第X页
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 2. 第二个资料 
- **文档名称**：《文档名称》
- **页码**：第X页
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 3. 第三个资料
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 4. 第四个资料
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

### 5. 第五个资料
- **文档名称**：《文档名称》
- **页码**：第X页  
- **相关内容**：从该文档中找到的具体内容描述...
- **关键信息**：提取的关键知识点...

## 💡 综合答案

基于以上分析，我才给你这个回答。[这里给出相对丰富的答案]
乖，怕你不理解，冰姐再给你举一个具体的小例子。[根据这个同学的问题，再编写了一个与之相关的容易理解的例子]

## 📚 推荐阅读

冰姐看到你在[这里给出同学提问涉及的主要知识点]这一方面确实存在薄弱点，冰姐建议你回头再看一下下面的资料，加深理解：
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]
- 《文档名称》第X页：[具体章节或内容]

请确保：
- 准确引用文档名称、页码和具体内容
- 使用温暖亲切的"冰姐"语气
- 答案要专业、准确、亲切
- 特别注意保持对话的连贯性，理解用户问题中的指代关系
- 如果用户的问题与之前的对话相关，请结合对话历史来理解问题的上下文
- 并非要完全按照这个模式回答，如果这个回答模式确实不合适，那你就自己组织语言结构就可以，但大部分情况还是要以这个模式为主。"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        max_retry = 1  # 仅重试1次，避免多次请求
        retry_count = 0
        resp = None

        data = {
            "model": self.model_name,
            # 🌟 仅修改这部分：分条传递对话历史，保留原prompt和system指令
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的法律教育助手，名字叫冰姐，擅长从多个文档资料中提取准确信息，在资料不足的时候能够根据自己的法律知识补充资料，最后能给出结构清晰的回答。你特别注重对话的连贯性，能够理解用户问题中的指代关系，并基于之前的对话上下文给出连贯的回答。连续问答时，可适当承接上一句的话题延伸，保持自然的聊天感，不生硬"
                }
                # 追加对话历史（按user/assistant原生格式，不修改内容，保留最近6条）
                ] + ([msg for msg in conversation_history[-5:]] if conversation_history else []) + [
                {
                    "role": "user",
                    "content": prompt  # 原封不动保留你写的完整prompt模板（含所有回答格式要求）
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            while retry_count <= max_retry:
                try:
                    resp = requests.post(self.base_url, headers=headers, json=data, timeout=(10,120))
                    break  # 请求成功，跳出重试循环
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
                    retry_count += 1
                    if retry_count > max_retry:
                        raise  # 重试1次后仍超时，抛出原报错
                    time.sleep(0.5)  # 间隔0.5秒重试，避免频繁请求

            resp.raise_for_status()
            result = resp.json()
            answer = result["choices"][0]["message"]["content"]

            # Token+费用统计
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            PRICE_PER_1000_TOKENS = 0.0015
            current_cost = (total_tokens * PRICE_PER_1000_TOKENS) / 1000

            st.session_state.total_tokens_used += total_tokens
            st.session_state.total_cost += current_cost
            st.session_state.current_tokens = total_tokens
            st.session_state.current_cost = current_cost
            st.session_state.prompt_tokens = prompt_tokens
            st.session_state.completion_tokens = completion_tokens

            logger.info(f"问答完成 - 本次Token：{total_tokens}（输入{prompt_tokens}/输出{completion_tokens}），本次费用：{current_cost:.6f}元，累计Token：{st.session_state.total_tokens_used}，累计费用：{st.session_state.total_cost:.6f}元")

            return answer
        except requests.exceptions.RequestException as e:
            return f"❌ API 调用失败：{str(e)}"
        except Exception as e:
            return f"❌ 处理异常：{str(e)}"

    def _build_context_text(self, contexts: List[Dict]) -> str:
        """构建上下文文本"""
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('source', '未知文档')
            page = ctx.get('page', '未知页码')
            content = ctx.get('content', '')

            doc_name = os.path.basename(source)
            if '.' in doc_name:
                doc_name = doc_name.rsplit('.', 1)[0]

            context_text += f"\n【资料{i}】\n"
            context_text += f"文档名称：《{doc_name}》\n"
            context_text += f"页码：第{page}页\n"
            context_text += f"内容片段：{content}\n"
            context_text += "-" * 60 + "\n"
        return context_text

# 新增：Cloud环境判断函数（替代Config.is_streamlit_cloud）
def is_streamlit_cloud():
    """判断是否在Streamlit Cloud环境"""
    return 'STREAMLIT_SERVER_TYPE' in os.environ

# 新增：获取Chroma DB路径（兼容本地/Cloud）
def get_chroma_db_path():
    """获取向量库路径"""
    # 优先使用Secrets配置的路径
    try:
        return st.secrets.get("CHROMA_DB_PATH", "/tmp/chroma_db")
    except:
        # 本地环境默认路径
        return "./chroma_db"

def init_vector_store_actual():
    """实际的向量存储初始化逻辑"""
    try:
        # 核心调整：兼容Cloud环境的路径处理
        chroma_db_path = get_chroma_db_path()
        local_chroma_path = "./chroma_db"
        
        # 本地环境优先使用本地目录
        if not is_streamlit_cloud() and os.path.exists(local_chroma_path) and os.listdir(local_chroma_path):
            chroma_db_path = local_chroma_path
            st.info(f"✅ 检测到本地向量库，使用路径：{local_chroma_path}")
        else:
            # Cloud环境使用/tmp目录
            if is_streamlit_cloud():
                st.info(f"ℹ️ Streamlit Cloud环境，使用临时目录：{chroma_db_path}")
                # 检测/tmp是否有向量库
                if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
                    st.warning("⚠️ /tmp目录未找到向量库！App可正常运行但无法检索PDF内容")
        
        # 创建向量存储实例
        vector_store = SmartVectorStore(persist_directory=chroma_db_path)
        
        # 加载现有向量库（如果存在）
        if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
            if vector_store.load_existing_vector_store():
                # 测试检索
                try:
                    test_results = vector_store.search_similar_documents("测试", k=1)
                    return vector_store, f"✅ 知识库加载成功！（环境：{'Cloud' if is_streamlit_cloud() else '本地'}）"
                except Exception as e:
                    return vector_store, f"⚠️ 向量库测试检索失败（不影响基础运行）：{str(e)}"
            else:
                return None, "❌ 向量库存在但加载失败，请检查文件完整性"
        else:
            # 无向量库：返回实例，保证App能运行
            return vector_store, "ℹ️ 未加载向量库，App基础功能正常，仅无法检索PDF内容"
            
    except Exception as e:
        return None, f"❌ 向量库初始化失败: {str(e)}"

def initialize_vector_store():
    """初始化向量存储（带超时处理）"""
    if st.session_state.vector_store_initialized:
        return st.session_state.vector_store
    
    placeholder = st.empty()
    with placeholder.container():
        st.info("🔄 正在初始化向量存储...")
        
        # 使用进度条
        progress_bar = st.progress(0)
        
        try:
            # 阶段1：检查目录
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # 阶段2：创建实例
            progress_bar.progress(40)
            
            # 阶段3：加载向量存储
            progress_bar.progress(60)
            
            # 使用线程池执行带超时的初始化
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_vector_store_actual)
                try:
                    # 设置30秒超时
                    vector_store, message = future.result(timeout=30)
                    progress_bar.progress(90)
                    
                    if vector_store:
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        placeholder.empty()
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.vector_store_initialized = True
                        st.success(message)
                        return vector_store
                    else:
                        progress_bar.progress(100)
                        placeholder.empty()
                        st.error(message)
                        return None
                        
                except concurrent.futures.TimeoutError:
                    progress_bar.progress(100)
                    placeholder.empty()
                    st.error("⚠️ 向量数据库加载超时，请检查数据库文件或重新初始化")
                    return None
                    
        except Exception as e:
            progress_bar.progress(100)
            placeholder.empty()
            st.error(f"❌ 初始化异常: {str(e)}")
            return None

def add_image_upload_section():
    """添加图片上传和识别功能"""
    with st.expander("📷 图片识别提问", expanded=False):
        st.info("上传包含问题的图片，系统会自动识别文字并提问")
        
        uploaded_file = st.file_uploader(
            "选择图片文件",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="支持PNG, JPG, JPEG, BMP格式",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # 显示图片预览
            image_processor.display_image_preview(uploaded_file, "识别图片")
            
            # 检查是否已经识别过，避免重复识别
            if 'image_processed' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
                # 识别文字
                with st.spinner("正在识别图片中的文字..."):
                    result = image_processor.process_uploaded_image(uploaded_file)
                
                # 保存识别结果到session_state
                st.session_state.image_processed = True
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.ocr_result = result
            else:
                # 使用之前的结果
                result = st.session_state.ocr_result
            
            # 处理识别结果
            if result["success"]:
                st.success("✅ 文字识别成功！")
                
                # 显示识别结果并允许编辑
                recognized_text = st.text_area(
                    "识别出的文字（可编辑）",
                    value=result["text"],
                    height=150,
                    help="检查并修改识别出的文字，然后点击'使用此文字提问'",
                    key="recognized_text"
                )
                
                # 添加问题输入
                additional_question = st.text_input(
                    "补充你的问题（可选）",
                    placeholder="例如：请解释这段话的意思...",
                    key="additional_question"
                )
                
                # 提问按钮
                if st.button("使用此文字提问", key="use_text_question"):
                    if recognized_text.strip():
                        # 组合问题和识别文字
                        full_question = recognized_text
                        if additional_question.strip():
                            full_question = f"{additional_question}\n\n识别内容：{recognized_text}"
                        
                        # 设置问题
                        st.session_state.image_question = full_question
                        
                        # 清除上传状态
                        st.session_state.image_processed = False
                        st.session_state.last_uploaded_file = None
                        
                        st.success("✅ 问题已准备，系统正在自动处理...")
                        st.rerun()
                    else:
                        st.error("请先识别出有效的文字")
            else:
                st.error(f"❌ {result['message']}")
        
        # 添加强制重置按钮
        if st.button("重置图片上传", key="reset_upload"):
            st.session_state.image_processed = False
            st.session_state.last_uploaded_file = None
            st.session_state.image_question = ""
            st.rerun()

def main():
    # 页面标题
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🎓 冰姐问答小课堂</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化DeepSeekAPI实例
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    deepseek_api = st.session_state.deepseek_api
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API设置")
        st.info(f"**API端点:**\n`{deepseek_api.base_url}`")
        
        # API登录方式选择
        login_method = st.radio("选择登录方式:", ["直接输入API", "密码登录"])
        
        if login_method == "直接输入API":
            api_key = st.text_input("输入DeepSeek官方API密钥:", type="password", placeholder="sk-...", key="api_key_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("设置API密钥", key="set_api_key"):
                    if api_key:
                        if deepseek_api.set_api_key(api_key):
                            st.success("✅ API密钥已设置")
                            st.rerun()
                        else:
                            st.error("❌ 请输入有效的API密钥")
                    else:
                        st.error("❌ 请输入API密钥")
            with col2:
                if st.button("测试连接", key="test_connection"):
                    if deepseek_api.is_logged_in():
                        with st.spinner("测试连接中..."):
                            result = deepseek_api.test_api_connection()
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(f"{result['message']}")
                            if "detail" in result:
                                st.error(f"详情: {result['detail']}")
                    else:
                        st.error("请先设置API密钥")
            
        else:  # 密码登录
            password = st.text_input("输入密码:", type="password", placeholder="默认密码：123456", key="password_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("登录", key="password_login"):
                    if deepseek_api.login_with_password(password):
                        st.success("✅ 登录成功！API密钥已自动填充")
                        st.rerun()
                    else:
                        st.error("❌ 密码错误")
            with col2:
                if st.button("测试连接", key="test_connection2"):
                    if deepseek_api.is_logged_in():
                        with st.spinner("测试连接中..."):
                            result = deepseek_api.test_api_connection()
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(f"{result['message']}")
                            if "detail" in result:
                                st.error(f"详情: {result['detail']}")
                    else:
                        st.error("请先登录")
        
        st.markdown("---")
        
        # 知识库状态
        st.markdown("### 📚 知识库状态")
        # 修复：移除width='stretch'参数，通过CSS实现拉伸效果
        if st.button("🔄 初始化知识库", key="init_kb"):
            vector_store = initialize_vector_store()
            if vector_store:
                st.success("✅ 知识库初始化成功！")
                st.rerun()
            else:
                st.error("❌ 知识库初始化失败")
        
        # 显示当前状态
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
            with st.expander("本次使用详情"):
                st.write(f"**本次Token使用:**")
                st.write(f"- 输入Token: {st.session_state.prompt_tokens}")
                st.write(f"- 输出Token: {st.session_state.completion_tokens}")
                st.write(f"- 总计: {st.session_state.current_tokens}")
                st.write(f"**本次费用:** ¥{st.session_state.current_cost:.6f}")
        
        st.markdown("---")
        st.markdown("### 📋 系统模式")
        st.warning("**当前模式: 只检索模式**")
        st.info(
            """**功能说明:**
- ✅ 加载现有知识库
- ✅ 智能问答检索
- ✅ 图片识别提问
- ❌ 不处理新PDF文档
- ❌ 不重新创建向量库

**数据处理请运行:**
```bash
python main.py  # 选择模式 1、2 或 4
```"""
        )
    
    # 主界面
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 对话界面")
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with col2:
        # 图片上传功能
        add_image_upload_section()
        
        st.markdown("---")
        
        # 知识库状态显示
        st.markdown("### 📊 知识库状态")
        if st.session_state.vector_store_initialized:
            try:
                vector_store = st.session_state.vector_store
                if hasattr(vector_store, 'vector_store') and hasattr(vector_store.vector_store, '_collection'):
                    count = vector_store.vector_store._collection.count()
                    
                    # 显示统计卡片
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("总文本块数", f"{count:,}")
                    with col_stat2:
                        st.metric("向量维度", "512")
                    
                    # 详细统计
                    with st.expander("📈 详细统计", expanded=True):
                        st.info("**知识库信息:**")
                        st.write(f"• 文本块总数: **{count}** 个")
                        st.write(f"• 估算PDF文档: **~{max(1, floor(count // 80))}** 个")
                        st.write(f"• 向量维度: **512** 维")
                        st.write(f"• 检索模型: **BGE-small-ZH_v1.5**")
                        
                        # 进度条显示知识库规模
                        progress_value = min(count / 1000, 1.0)
                        st.progress(progress_value)
                        st.caption(f"知识库规模: {count}文本块")
                        
            except Exception as e:
                st.info("知识库已加载，详细信息不可用")
        else:
            st.warning("知识库未初始化")
    
    # 聊天输入框（始终显示）
    user_input = st.chat_input("乖，你有哪个地方不明白呢")
    
    # 处理所有类型的问题输入
    current_prompt = None
    prompt_source = None
    
    # 首先检查是否有图片识别的问题
    if 'image_question' in st.session_state and st.session_state.image_question:
        current_prompt = st.session_state.image_question
        prompt_source = "image"
        # 清空图片问题，避免重复处理
        st.session_state.image_question = ""
    # 然后检查正常的聊天输入
    elif user_input:
        current_prompt = user_input
        prompt_source = "chat"
    
    # 如果有问题需要处理
    if current_prompt:
        # 检查API和知识库状态
        if not deepseek_api.is_logged_in():
            st.warning("🔑 请先在侧边栏设置DeepSeek API密钥")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "请先在侧边栏设置DeepSeek API密钥"})
            st.rerun()
        elif not st.session_state.vector_store_initialized:
            st.warning("📚 请先在侧边栏初始化知识库")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "请先在侧边栏初始化知识库"})
            st.rerun()
        else:
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            
            # 助手回答
            with col1:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("🧠 思考中...")
                    
                    try:
                        # 获取向量存储
                        vector_store = st.session_state.vector_store
                        
                        # 搜索相关文档
                        with st.spinner("🔍 检索相关资料..."):
                            search_results = vector_store.search_similar_documents(current_prompt, k=8)
                        
                        if not search_results:
                            answer = "乖，这个问题有点复杂，可以在课后答疑的时候问我，到时候冰姐语音给你讲哈"
                        else:
                            # 构建上下文
                            contexts = []
                            for doc in search_results:
                                contexts.append({
                                    # 保留前600字符+省略号，避免超长片段
                                    'content': doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content,
                                    'source': doc.metadata.get('source', '未知'),
                                    'page': doc.metadata.get('page', '未知页码')
                                })
                            
                            # 获取答案，传递对话历史（排除当前这条用户消息）
                            conversation_history = st.session_state.messages[:-1]
                            with st.spinner("📝 乖，先别着急，让冰姐思考一下这个问题..."):
                                answer = deepseek_api.get_answer(current_prompt, contexts, conversation_history)
                        
                        # 显示答案
                        message_placeholder.markdown(answer)
                        
                        # 添加到消息历史
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # 显示检索到的资料
                        with st.expander("📄 检索到的参考资料", expanded=False):
                            for i, doc in enumerate(search_results[:5], 1):
                                source = doc.metadata.get('source', '未知文档')
                                page = doc.metadata.get('page', '未知页码')
                                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                
                                doc_name = os.path.basename(source)
                                if '.' in doc_name:
                                    doc_name = doc_name.rsplit('.', 1)[0]
                                
                                st.markdown(f"**资料 {i}:** 《{doc_name}》 - 第{page}页")
                                st.caption(content)
                                st.divider()
                        
                        # 如果是图片问题，显示成功提示
                        if prompt_source == "image":
                            st.success("✅ 图片问题已回答完成，可以继续提问")
                            
                    except Exception as e:
                        error_msg = f"处理过程中出错: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        logger.error(f"问答处理出错: {str(e)}")

if __name__ == "__main__":
    main()