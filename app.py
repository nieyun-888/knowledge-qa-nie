import streamlit as st
import os
import sys
import subprocess
import requests
import json
import logging
import time
from typing import List, Dict, Any

# ===================== 核心修复：自动安装系统依赖和Python依赖 =====================
# ===================== 核心修复：自动安装系统依赖和Python依赖 =====================
def fix_dependencies():
    """自动修复Cloud环境依赖问题"""
    # 1. 仅在Streamlit Cloud环境执行
    if 'STREAMLIT_SERVER_TYPE' not in os.environ:
        print("🔧 本地环境：跳过强制依赖修复，保持现有配置")
        return
    
    print("🌐 Cloud环境：执行无头OCR依赖修复")
    
    # 2. 安装系统库（解决libGL缺失）
    try:
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
        print(f"⚠️ 系统库安装警告：{str(e)}")
    
    # 3. 强制安装兼容版本的Python依赖
    try:
        # 先卸载可能有问题的GUI版本（静默执行，不检查结果）
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False  # 不强制检查，可能不存在
        )
        
        # 【关键修复】安装正确的OCR包名称
        # 注意：安装时用连字符"rapidocr-onnxruntime"，导入时用下划线"rapidocr_onnxruntime"
        cloud_packages = [
            "opencv-python-headless==4.8.1.78",
            "rapidocr-onnxruntime==1.3.7",  # ← 包名（pip安装用）
            "onnxruntime==1.16.3",
            "pymupdf==1.23.8",
            "pillow==10.1.0",
            "huggingface-hub==0.19.4",
            "transformers==4.36.2"
        ]
        
        # 打印信息，便于调试
        print(f"📦 正在安装依赖: {cloud_packages}")
        
        for package in cloud_packages:
            print(f"正在安装: {package}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--force-reinstall"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            if result.returncode == 0:
                print(f"✅ 安装成功: {package}")
            else:
                print(f"⚠️ 安装可能有问题: {package}")
        
        print("✅ Cloud无头OCR依赖修复完成！")
        
        # 【新增】验证OCR包是否正确安装
        print("🔍 验证OCR包安装状态...")
        try:
            import rapidocr_onnxruntime
            print("✅ rapidocr_onnxruntime 导入成功")
            print(f"  版本: {rapidocr_onnxruntime.__version__ if hasattr(rapidocr_onnxruntime, '__version__') else '未知'}")
        except Exception as e:
            print(f"❌ rapidocr_onnxruntime 导入失败: {e}")
            # 尝试另一种导入方式
            try:
                import rapidocr
                print("✅ rapidocr 导入成功（可能是旧版本）")
            except Exception as e2:
                print(f"❌ rapidocr 也导入失败: {e2}")
        
    except Exception as e:
        print(f"⚠️ 依赖修复警告：{str(e)}")



# 执行依赖修复（仅首次运行）
if "deps_fixed" not in st.session_state:
    fix_dependencies()
    st.session_state.deps_fixed = True

# ===================== 环境检测与配置 =====================
def configure_for_environment():
    """根据运行环境配置无头模式"""
    # 只在Cloud环境设置无头变量
    if 'STREAMLIT_SERVER_TYPE' in os.environ:
        print("🌐 检测到Cloud环境，配置无头模式")
        os.environ['DISPLAY'] = ':99'  # 虚拟显示
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    else:
        print("💻 本地环境：保持现有GUI配置")

# 执行环境配置
configure_for_environment()

# ===================== 导入自定义模块 =====================
try:
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.vector_store import SmartVectorStore
    from src.image_processor import image_processor
from math import floor

# ===================== 全局配置 =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="冰姐问答小课堂",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        # 重要：Streamlit Cloud的持久化目录
        return "/home/appuser/chroma_db"
    return "./chroma_db"


def get_pdf_data_path():
    """获取PDF数据路径"""
    if is_streamlit_cloud():
        # 检查是否有用户上传的PDF
        cloud_path = "/home/appuser/data/raw_pdfs"
        if os.path.exists(cloud_path) and any(f.endswith('.pdf') for f in os.listdir(cloud_path)):
            return cloud_path
        
        # 如果没有用户上传的，检查是否预置了PDF
        local_prebuilt = "./data/raw_pdfs"
        if os.path.exists(local_prebuilt) and any(f.endswith('.pdf') for f in os.listdir(local_prebuilt)):
            return local_prebuilt
            
        return cloud_path  # 返回Cloud路径，即使为空
    return "./data/raw_pdfs"

# ===================== DeepSeek API 类 =====================
class DeepSeekAPI:
    """DeepSeek API 管理类"""
    def __init__(self):
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model_name = "deepseek-chat"
        
        # 修复：不要自动从secrets获取API Key
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None
            
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = False
    
    @property
    def api_key(self):
        return st.session_state.api_key

    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            st.session_state.api_key = api_key.strip()
            st.session_state.api_key_set = True
            return True
        return False

    def login_with_password(self, password: str) -> bool:
        if password == "123456":
            st.session_state.api_key = "sk-4f3e29df9fa54da8bd601ae780111df1"
            st.session_state.api_key_set = True  # 标记为已通过登录设置
            return True
        return False

    def is_logged_in(self) -> bool:
        # 严格检查：必须有API Key且是通过登录设置的
        return (st.session_state.api_key is not None 
                and st.session_state.api_key.strip() != ""
                and st.session_state.api_key_set)  # 必须是通过登录设置的

    def test_api_connection(self) -> Dict:
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
        if not self.is_logged_in():
            return "乖，请先登录或者输入API喔"

        context_text = self._build_context_text(contexts)
        
        history_text = ""
        if conversation_history:
            history_text = "\n\n之前的对话历史：\n"
            for msg in conversation_history[-6:]:
                role = "用户" if msg["role"] == "user" else "冰姐"
                history_text += f"{role}：{msg['content']}\n"

        prompt = f"""你是一个专业的知识问答助手冰姐。我将给你一些参考资料和对话历史，你主要根据提供的上下文信息回答用户问题，如果你觉得信息不够，则你就需要自己去网上去查找一下答案。

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
- 并非要完全按照这个模式回答，如果这个回答模式确实不合适，那你就自己组织语言结构就可以，但大部分情况还是要以这个模式为主。请用温暖亲切的语气，按照以下结构回答（逻辑不通时可灵活调整）"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一位专业的法律教育助手，名字叫冰姐，擅长从多个文档资料中提取准确信息，在资料不足的时候能够根据自己的法律知识补充资料，最后能给出结构清晰的回答。你特别注重对话的连贯性，能够理解用户问题中的指代关系，并基于之前的对话上下文给出连贯的回答。连续问答时，可适当承接上一句的话题延伸，保持自然的聊天感，不生硬"},
            ] + ([msg for msg in conversation_history[-5:]] if conversation_history else []) + [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }

        max_retry = 1
        retry_count = 0
        while retry_count <= max_retry:
            try:
                resp = requests.post(self.base_url, headers=headers, json=data, timeout=(10, 120))
                resp.raise_for_status()
                result = resp.json()
                
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
                
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                retry_count += 1
                if retry_count > max_retry:
                    return f"❌ 回答失败：{str(e)}"
                time.sleep(0.5)

    def _build_context_text(self, contexts: List[Dict]) -> str:
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('source', '未知文档')
            page = ctx.get('page', '未知页码')
            content = ctx.get('content', '')
            doc_name = os.path.basename(source).rsplit('.', 1)[0] if '.' in source else source
            
            context_text += f"\n【资料{i}】《{doc_name}》第{page}页：{content[:600]}...\n"
        return context_text

# ===================== 向量库初始化 =====================
def generate_vector_store_from_pdfs(pdf_dir, chroma_db_path):
    """从PDF生成向量库"""
    try:
        with st.spinner(f"🔄 正在处理PDF文件，这可能需要几分钟..."):
            from src.pdf_processor import PDFProcessor
            
            pdf_processor = PDFProcessor()
            documents = pdf_processor.load_pdfs_from_directory(pdf_dir)
            
            if not documents:
                st.error("❌ 没有找到可处理的PDF内容")
                return None
            
            chunks = pdf_processor.split_documents(documents)
            
            vector_store = SmartVectorStore(persist_directory=chroma_db_path)
            success = vector_store.create_vector_store(chunks, clear_old=True)
            
            if success:
                st.session_state.vector_store = vector_store
                st.session_state.vector_store_initialized = True
                
                status_file = os.path.join(chroma_db_path, "generation_status.json")
                status = {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "count": vector_store.vector_store._collection.count() if hasattr(vector_store, 'vector_store') else 0,
                    "initialized": True  # ← 添加这一行！
                }
                with open(status_file, 'w', encoding='utf-8') as f:
                    json.dump(status, f, ensure_ascii=False, indent=2)
                
                st.success(f"✅ 知识库生成完成！共处理 {len(chunks)} 个文本块")
                return vector_store
            else:
                st.error("❌ 知识库生成失败")
                return None
                
    except Exception as e:
        st.error(f"❌ 生成失败: {str(e)}")
        return None

def initialize_vector_store_once():
    """一次性初始化向量库，生成后永久保存"""
    # 如果session state已有，直接返回
    if st.session_state.vector_store_initialized:
        return st.session_state.vector_store
    
    chroma_db_path = get_chroma_db_path()
    
    # ========== 新增：从文件恢复状态 ==========
    status_file = os.path.join(chroma_db_path, "generation_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            # 如果文件标记为已初始化，但session state没有
            if status.get("initialized", False):
                try:
                    vector_store = SmartVectorStore(persist_directory=chroma_db_path)
                    if vector_store.load_existing_vector_store():
                        st.session_state.vector_store = vector_store
                        st.session_state.vector_store_initialized = True
                        st.sidebar.success("✅ 从文件恢复知识库状态成功")
                        return vector_store
                except:
                    pass  # 如果加载失败，继续下面的逻辑
        except:
            pass
    # ========== 新增结束 ==========
    
    pdf_data_path = get_pdf_data_path()
    
    # 检查向量库是否已存在（原有逻辑）
    if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
        try:
            vector_store = SmartVectorStore(persist_directory=chroma_db_path)
            if vector_store.load_existing_vector_store():
                st.session_state.vector_store = vector_store
                st.session_state.vector_store_initialized = True
                st.sidebar.success("✅ 加载现有知识库成功")
                return vector_store
        except Exception as e:
            st.sidebar.warning(f"⚠️ 加载失败: {e}")
    
    # 检查是否有PDF文件需要处理
    if not os.path.exists(pdf_data_path):
        st.sidebar.warning("📄 请先上传PDF文件")
        return None
    
    pdf_files = [f for f in os.listdir(pdf_data_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.sidebar.warning("📄 请先上传PDF文件")
        return None
    
    # 显示生成选项
    with st.sidebar:
        st.markdown("### 🏗️ 知识库生成")
        if st.button("🚀 生成知识库", type="primary", key="generate_kb_main"):
            return generate_vector_store_from_pdfs(pdf_data_path, chroma_db_path)
    
    return None

# ===================== PDF上传功能 =====================
def handle_pdf_upload():
    """处理PDF上传（在侧边栏中调用）"""
    if not is_streamlit_cloud():
        return
    
    pdf_data_path = get_pdf_data_path()
    
    # 检查是否已经有PDF文件
    if os.path.exists(pdf_data_path) and any(f.endswith('.pdf') for f in os.listdir(pdf_data_path)):
        return
    
    # 创建目录
    os.makedirs(pdf_data_path, exist_ok=True)
    os.makedirs(get_chroma_db_path(), exist_ok=True)
    
    with st.sidebar:
        st.markdown("### 📄 PDF文件上传")
        st.info("首次使用需要上传PDF文件")
        
        uploaded_files = st.file_uploader(
            "选择PDF文件",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            with st.spinner(f"正在上传 {len(uploaded_files)} 个PDF文件..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(pdf_data_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 已上传 {len(uploaded_files)} 个PDF文件")
            st.rerun()

# ===================== 图片上传/OCR功能 =====================
def add_image_upload_section():
    """图片上传和OCR识别"""
    with st.expander("📷 图片识别提问", expanded=False):
        uploaded_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg', 'bmp'], key="image_uploader")
        
        if uploaded_file is not None:
            image_processor.display_image_preview(uploaded_file, "识别图片")
            
            if st.session_state.get('last_uploaded_file') != uploaded_file.name:
                with st.spinner("正在识别文字..."):
                    result = image_processor.process_uploaded_image(uploaded_file)
                st.session_state.image_processed = True
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.ocr_result = result
            else:
                result = st.session_state.ocr_result
            
            if result["success"]:
                st.success("✅ 文字识别成功！")
                recognized_text = st.text_area("识别结果（可编辑）", result["text"], height=150, key="ocr_text")
                additional_question = st.text_input("补充问题（可选）", key="ocr_question")
                
                if st.button("使用此文字提问", key="ocr_submit"):
                    if recognized_text.strip():
                        full_question = f"{additional_question}\n\n{recognized_text}" if additional_question else recognized_text
                        st.session_state.image_question = full_question
                        st.session_state.image_processed = False
                        st.success("✅ 冰姐正在阅读你的问题，稍等一下哈乖...")
                        st.rerun()
                    else:
                        st.error("请输入有效文字")
            else:
                st.error(f"❌ 识别失败：{result['message']}")
        
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
    
    # 检查Cloud环境PDF文件
    if is_streamlit_cloud():
        pdf_path = get_pdf_data_path()
        if not os.path.exists(pdf_path) or not any(f.endswith('.pdf') for f in os.listdir(pdf_path)):
            st.warning("📄 **首次使用提示**：请在侧边栏上传PDF文件")
    
    # 初始化API
    if 'deepseek_api' not in st.session_state:
        st.session_state.deepseek_api = DeepSeekAPI()
    deepseek_api = st.session_state.deepseek_api
    
    # 侧边栏
    with st.sidebar:
        st.header("🔑 API 设置")
        st.info(f"API端点: `{deepseek_api.base_url}`")
        
        # API设置代码...
        login_method = st.radio("登录方式", ["直接输入API", "密码登录"])
        
        if login_method == "直接输入API":
            api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...", key="api_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("设置API", key="set_api"):
                    if deepseek_api.set_api_key(api_key):
                        st.success("✅ API已设置")
                        st.rerun()
            with col2:
                if st.button("测试连接", key="test_api"):
                    result = deepseek_api.test_api_connection()
                    msg = str(result.get("message", "未知错误")).strip()
                    if result.get("success", False):
                        st.success(f"✅ API连接成功\n{msg.replace('✅', '').strip()}")
                    else:
                        error_detail = str(result.get("detail", ""))[:100]
                        st.error(f"❌ API连接失败：{msg[:150]}\n{error_detail}")
        else:
            password = st.text_input("密码", type="password", placeholder="默认：123456", key="pwd_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("登录", key="login"):
                    if deepseek_api.login_with_password(password):
                        st.success("✅ 登录成功")
                        time.sleep(0.6)  # 让用户看到消息
                        st.rerun()
                    else:
                        st.error("❌ 密码错误")
        
        st.markdown("---")
        
        # PDF上传
        handle_pdf_upload()
        
        st.markdown("---")
        
        # 知识库状态
        st.markdown("### 📚 知识库状态")
        
        # 检查生成状态
        chroma_db_path = get_chroma_db_path()
        status_file = os.path.join(chroma_db_path, "generation_status.json")
        
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            st.success("✅ 知识库已就绪")
            st.caption(f"生成时间: {status.get('generated_at', '未知')}")
            st.caption(f"文本块数: {status.get('count', 0):,}")
            
            # 如果已初始化，不再显示按钮
            if not st.session_state.vector_store_initialized:
                if st.button("🔗 连接知识库", key="connect_kb_sidebar"):
                    vector_store = initialize_vector_store_once()
                    if vector_store:
                        st.success("✅ 知识库连接成功")
                        st.rerun()
        else:
            # 初始化向量库
            vector_store = initialize_vector_store_once()
            if vector_store:
                st.success("✅ 知识库已就绪")
            else:
                st.warning("⚠️ 知识库未初始化")
        
        st.markdown("---")
        
        # 费用统计
        st.markdown("### 💰 费用统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("累计Token", str(st.session_state.total_tokens_used))
        with col2:
            st.metric("累计费用", str(st.session_state.total_cost))
        
        
        st.markdown("---")
        st.warning("**当前模式：仅检索模式**")

        st.markdown("---")
        st.markdown("### 🛠️ 系统管理模式")

        # 使用密码"123456"登录（与你的API登录密码一致）
        if st.text_input("管理模式密码", type="password", key="sys_admin_pwd") == "nieyun123":
            st.success("🔓 已进入系统管理模式")
            
            # 显示系统信息
            st.info(f"🔍 向量数据库路径: {get_chroma_db_path()}")
            st.info(f"📁 PDF源文件目录: {get_pdf_data_path()}")
            
            # 模式选择（类似你的命令行菜单）
            mode = st.radio(
                "请选择管理模式：",
                ["📁 查看系统状态", "🔄 智能重新生成", "🗑️ 强制清空数据", "🔍 只检索模式", "💥 强制全部重新处理"],
                key="sys_mode"
            )
            
            if mode == "📁 查看系统状态":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("向量库状态", "已加载" if st.session_state.vector_store_initialized else "未加载")
                with col2:
                    # 检查向量库文件
                    db_path = get_chroma_db_path()
                    if os.path.exists(db_path):
                        file_count = sum(len(files) for _, _, files in os.walk(db_path))
                        st.metric("数据文件数", f"{file_count}")
                    else:
                        st.metric("数据文件数", "0")
                with col3:
                    # 检查PDF文件
                    pdf_path = get_pdf_data_path()
                    if os.path.exists(pdf_path):
                        pdf_count = len([f for f in os.listdir(pdf_path) if f.lower().endswith('.pdf')])
                        st.metric("PDF文件数", f"{pdf_count}")
                    else:
                        st.metric("PDF文件数", "0")
                
                # 显示详细路径
                st.code(f"向量库路径: {os.path.abspath(db_path) if os.path.exists(db_path) else '不存在'}")
                st.code(f"PDF路径: {os.path.abspath(pdf_path) if os.path.exists(pdf_path) else '不存在'}")
            
            elif mode == "🔄 智能重新生成":
                st.warning("此操作将：\n1. 检查PDF文件是否有更新\n2. 只处理新的或修改过的PDF\n3. 保持现有向量库数据")
                if st.button("开始智能重新生成", type="primary", key="smart_regen_btn"):
                    # 这里需要实现智能检测逻辑
                    # 暂时先标记为需要重新生成
                    st.session_state.vector_store_initialized = False
                    st.session_state.vector_store = None
                    st.success("✅ 已标记为需要重新生成")
                    st.info("请在上方点击'生成知识库'按钮")
            
            elif mode == "🗑️ 强制清空数据":
                st.error("⚠️ **危险操作**：这将删除所有向量库数据！")
                confirm = st.text_input("请输入'DELETE ALL'确认操作：", key="delete_confirm")
                if confirm == "DELETE ALL":
                    if st.button("🔥 确认永久删除所有数据", type="secondary", key="confirm_delete_btn"):
                        import shutil
                        # 只在删除时执行重置
                        def cleanup_chroma_resources():
                            """清理ChromaDB相关资源"""
                            try:
                                # 1. 清除session state
                                if 'vector_store' in st.session_state:
                                    st.session_state.vector_store = None
                                st.session_state.vector_store_initialized = False
                                
                                # 2. 如果有ChromaDB客户端，尝试清理
                                try:
                                    import chromadb
                                    # 新版本chromadb可能有清理方法
                                    chromadb.Client().clear_system_cache()
                                except:
                                    pass
                                
                                # 3. 垃圾回收
                                import gc
                                gc.collect()
                                
                                print("✅ ChromaDB资源已清理")
                            except Exception as e:
                                print(f"⚠️ 清理资源时出错: {e}")
                        
                        # 执行清理
                        cleanup_chroma_resources()
                        
                        # 删除文件
                        db_path = get_chroma_db_path()
                        if os.path.exists(db_path):
                            # 先尝试正常删除
                            try:
                                shutil.rmtree(db_path)
                            except:
                                # 如果失败，等待后重试
                                time.sleep(1)
                                shutil.rmtree(db_path, ignore_errors=True)
                            
                            st.success("✅ 所有数据已永久删除！")
                            st.info("需要重新上传PDF并生成知识库")
                        else:
                            st.info("📭 数据目录不存在")
                        
                        # 等待一下
                        time.sleep(2)
                        st.rerun()

            
            elif mode == "🔍 只检索模式":
                st.info("直接使用现有向量库进行检索")
                if st.session_state.vector_store_initialized:
                    st.success("✅ 向量库已加载，可以直接使用")
                else:
                    st.warning("⚠️ 向量库未加载")
                    if st.button("尝试加载现有向量库", key="try_load_btn"):
                        vector_store = initialize_vector_store_once()
                        if vector_store:
                            st.success("✅ 向量库加载成功")
                            st.rerun()
            
            elif mode == "💥 强制全部重新处理":
                st.error("⚠️ **强制操作**：将删除并重新处理所有PDF！")
                st.warning("操作步骤：\n1. 先删除所有向量库数据\n2. 重新处理所有PDF文件\n3. 生成全新向量库")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("第1步：删除数据", type="secondary", key="step1_delete_btn"):
                        import shutil
                        db_path = get_chroma_db_path()
                        if os.path.exists(db_path):
                            shutil.rmtree(db_path)
                            st.session_state.vector_store_initialized = False
                            st.session_state.vector_store = None
                            st.success("✅ 第1步完成：数据已删除")
                            st.rerun()
                        else:
                            st.info("📭 数据目录不存在")
                
                with col2:
                    if st.button("第2步：重新生成", type="primary", disabled=st.session_state.vector_store_initialized, key="step2_regen_btn"):
                        st.success("✅ 第2步：请在上方点击'生成知识库'按钮重新处理PDF")

        else:
            st.caption("🔒 系统管理模式需要密码")



        # ====== 新增：OCR状态显示（调试用） ======
        st.markdown("---")
        st.markdown("### 🔧 系统状态")
        
        # 检查OCR引擎状态
        try:
            from src.pdf_processor import OCR_ENGINE
            if OCR_ENGINE is None:
                st.warning("⚠️ OCR引擎：未初始化")
            else:
                st.success("✅ OCR引擎：就绪")
                
                # 检查OpenCV版本
                try:
                    import cv2
                    version = cv2.__version__
                    is_headless = "headless" in cv2.getBuildInformation().lower()
                    st.caption(f"OpenCV: {version} {'(无头)' if is_headless else '(有头)'}")
                except:
                    st.caption("OpenCV: 无法检测")
        except Exception as e:
            st.warning(f"⚠️ OCR状态检测失败：{str(e)[:50]}")
        # ====== 新增结束 ======
    # 主界面布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 对话界面")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    with col2:
        add_image_upload_section()
        
        st.markdown("---")
        
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
            except Exception as e:
                st.info("✅ 知识库已加载")
        else:
            st.warning("⚠️ 请先初始化知识库")

    # 处理用户输入
    current_prompt = None
    if st.session_state.image_question:
        current_prompt = st.session_state.image_question
        st.session_state.image_question = ""
    elif user_input := st.chat_input("乖，你有哪个地方不明白呢"):
        current_prompt = user_input

    if current_prompt:
        if not deepseek_api.is_logged_in():
            st.warning("🔑 请先设置API Key")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "乖，请先登录或者输入API喔"})
            st.rerun()
        elif not st.session_state.vector_store_initialized:
            st.warning("📚 请先初始化知识库")
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "乖，请先登录或者输入API喔"})
            st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": current_prompt})
            
            with col1:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("🧠 思考中...")
                    
                    try:
                        vector_store = st.session_state.vector_store
                        with st.spinner("🔍 乖，让冰姐想一下这个问题..."):
                            search_results = vector_store.search_similar_documents(current_prompt, k=8)
                        
                        if not search_results:
                            answer = "乖，这个问题冰姐暂时没有找到相关资料，课后我再详细给你讲哈～"
                        else:
                            contexts = [{
                                'content': doc.page_content,
                                'source': doc.metadata.get('source', '未知'),
                                'page': doc.metadata.get('page', '未知')
                            } for doc in search_results]
                            with st.spinner("📝 乖，稍等一下，让冰姐想一下这个问题..."):
                                answer = deepseek_api.get_answer(current_prompt, contexts, st.session_state.messages[:-1])
                        
                        placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        with st.expander("📄 参考资料", expanded=False):
                            for i, doc in enumerate(search_results[:5], 1):
                                source = doc.metadata.get('source', '未知')
                                page = doc.metadata.get('page', '未知')
                                doc_name = os.path.basename(source).rsplit('.', 1)[0] if '.' in source else source
                                st.markdown(f"**资料{i}：《{doc_name}》第{page}页**")
                                st.caption(doc.page_content[:200] + "...")
                    except Exception as e:
                        error_msg = f"❌ 处理失败：{str(e)}"
                        placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()