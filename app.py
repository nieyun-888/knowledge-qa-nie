import streamlit as st
import os
import sys
import subprocess
import requests
import json
import platform
import time
import socket 
import logging
from typing import List, Dict, Any

# ===================== 超级紧急：第一时间设置环境并安装libGL =====================
def emergency_setup():
    """在导入任何其他模块前紧急安装libGL"""
    print("🚨 紧急环境设置开始...")
    
    # 先设置环境变量
    os.environ['DISPLAY'] = ':99'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # 立即尝试安装libGL（仅Debian/Ubuntu系统）
    try:
        # 检测系统是否为Debian/Ubuntu
        if platform.system() == "Linux" and os.path.exists("/etc/debian_version"):
            print("📦 紧急安装libGL...")
            subprocess.run(
                ["apt-get", "update", "-y"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            subprocess.run(
                ["apt-get", "install", "-y", "libgl1-mesa-glx"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            print("✅ libGL安装完成")
        else:
            print("ℹ️ 非Debian系统，跳过libGL安装")
    except Exception as e:
        print(f"⚠️ libGL安装失败: {e}")

# ===== 立即执行！在导入任何OCR相关模块之前 =====
if 'STREAMLIT_SERVER_TYPE' in os.environ or os.environ.get('HOME') == '/home/appuser':
    emergency_setup()
# ==============================================

# ===================== 然后再导入其他模块 =====================
# 现在才导入可能依赖libGL的模块
try:
    import cv2
    print(f"✅ OpenCV导入成功: {cv2.__version__}")
except ImportError:
    # 尝试导入headless版本
    try:
        import cv2_headless as cv2
        print(f"✅ OpenCV Headless导入成功: {cv2.__version__}")
    except:
        print("⚠️ OpenCV导入失败")
        cv2 = None  # 显式赋值避免后续报错
except Exception as e:
    print(f"⚠️ OpenCV导入异常: {e}")
    cv2 = None

# 继续导入你的其他模块...
from src.vector_store import SmartVectorStore
from src.image_processor import image_processor
from src.pdf_processor import PDFProcessor

# ===================== 导入自定义模块 =====================
print("📂 开始导入自定义模块...")
print("-"*50)

# 先导入不依赖OCR的模块
try:
    from src.vector_store import SmartVectorStore
    print("✅ SmartVectorStore 导入成功")
except ImportError as e:
    print(f"❌ SmartVectorStore 导入失败: {e}")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    print(f"📁 添加路径: {os.path.dirname(os.path.abspath(__file__))}")
    from src.vector_store import SmartVectorStore
    print("✅ SmartVectorStore 二次导入成功")

print("\n🔍 开始测试OCR基础依赖...")

# 测试OpenCV
try:
    # 优先使用已导入的cv2，未导入则重新尝试导入
    if 'cv2' not in locals() or cv2 is None:
        try:
            import cv2
        except ImportError:
            import cv2_headless as cv2
    
    print(f"✅ OpenCV版本: {cv2.__version__}")
    print(f"   - 安装路径: {cv2.__file__}")
    # 检查是否是headless版本
    build_info = cv2.getBuildInformation()
    is_headless = 'headless' in build_info.lower()
    print(f"   - 版本类型: {'无头(Headless)' if is_headless else '标准(GUI)'}")
    if 'libGL' in build_info:
        print(f"   - libGL: 已链接")
except ImportError as e:
    print(f"❌ OpenCV导入失败: {e}")
except Exception as e:
    print(f"❌ OpenCV初始化失败: {e}")

# 测试PIL
try:
    from PIL import Image, __version__ as pil_version
    print(f"✅ PIL/Pillow版本: {pil_version}")
    print(f"   - 安装路径: {Image.__file__}")
except ImportError as e:
    print(f"❌ PIL/Pillow导入失败: {e}")

# 测试RapidOCR（只导入，不初始化）
try:
    import rapidocr_onnxruntime
    print(f"✅ RapidOCR模块: 可导入")
    print(f"   - 版本: {getattr(rapidocr_onnxruntime, '__version__', '未知')}")
    print(f"   - 安装路径: {rapidocr_onnxruntime.__file__}")
    
    # 测试ONNX Runtime
    import onnxruntime as ort
    print(f"✅ ONNX Runtime版本: {ort.__version__}")
    print(f"   - 可用执行提供者: {ort.get_available_providers()}")
    print(f"   - 默认设备: {ort.get_device()}")
except ImportError as e:
    print(f"❌ RapidOCR/ONNX导入失败: {e}")

# 导入image_processor
# 导入image_processor
print("\n📦 导入 image_processor...")
try:
    from src.image_processor import image_processor
    print("✅ image_processor 导入成功")
    
    # 检查image_processor是否有OCR引擎
    if hasattr(image_processor, 'ocr_engine'):
        print(f"   - OCR引擎类型: {type(image_processor.ocr_engine).__name__}")
        print(f"   - OCR引擎状态: {'已初始化' if image_processor.ocr_engine else '未初始化'}")
    else:
        print("   - ⚠️ image_processor没有ocr_engine属性")
        
except (ImportError, AttributeError) as e:  # 捕获更多异常类型
    print(f"❌ image_processor导入失败: {e}")
    print("🔧 创建DummyImageProcessor占位符...")
    
    class DummyImageProcessor:
        def display_image_preview(self, *args, **kwargs):
            st.warning("⚠️ OCR功能暂不可用（Dummy模式）")
        
        def process_uploaded_image(self, uploaded_file):
            return {
                "success": False,
                "message": "OCR引擎初始化失败，请检查依赖和系统库",
                "text": "",
                "debug": {
                    "error": str(e),
                    "cloud_env": 'STREAMLIT_SERVER_TYPE' in os.environ
                }
            }
    
    image_processor = DummyImageProcessor()
    print("✅ DummyImageProcessor 创建完成")

print("-"*50)
print("📂 模块导入完成\n")


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
if "headless_configured" not in st.session_state:
    st.session_state.headless_configured = True  # 标记环境已配置

# ===================== Cloud环境工具函数 =====================
def is_streamlit_cloud():
    """判断是否在Streamlit Cloud环境"""
    return 'STREAMLIT_SERVER_TYPE' in os.environ

def get_chroma_db_path():
    """获取向量库路径（兼容本地/Cloud）"""
    if is_streamlit_cloud():
        chroma_path = "/home/appuser/chroma_db"
    else:
        chroma_path = "./chroma_db"
    # 确保目录存在且可写
    os.makedirs(chroma_path, exist_ok=True)
    return chroma_path

# ========== 新增：分批处理进度记录函数（不要改OCR相关） ==========
import json
import os
from pathlib import Path

def get_processed_files_path():
    """获取已处理文件记录的路径（存在chroma_db目录下）"""
    processed_path = os.path.join(get_chroma_db_path(), "processed_files.json")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)  # 确保目录存在
    return processed_path

def load_processed_files():
    """加载已处理的文件列表（避免重复处理）"""
    processed_path = get_processed_files_path()
    if os.path.exists(processed_path):
        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                return json.load(f)  # 返回已处理文件名列表
        except Exception as e:
            st.warning(f"加载已处理文件记录失败：{e}")
            return []
    return []  # 首次处理返回空列表

def save_processed_file(file_name):
    """保存单个已处理文件到记录（处理完一个更一个）"""
    processed_files = load_processed_files()
    if file_name not in processed_files:
        processed_files.append(file_name)
        with open(get_processed_files_path(), "w", encoding="utf-8") as f:
            json.dump(processed_files, f, ensure_ascii=False)  # 保存为JSON

def clear_processed_files():
    """清空已处理记录（重新开始处理时用）"""
    processed_path = get_processed_files_path()
    if os.path.exists(processed_path):
        os.remove(processed_path)
    st.session_state.vector_store_initialized = False  # 重置向量库状态
# ========== 新增结束 ==========


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
        return (
            isinstance(st.session_state.api_key, str) 
            and st.session_state.api_key.strip() != ""
            and st.session_state.api_key_set
        )
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
def generate_vector_store_from_pdfs(pdf_dir, chroma_db_path, batch_size=10):
    """
    分批生成向量库 - 按页分批，不跳页
    :param pdf_dir: PDF文件目录
    :param chroma_db_path: Chroma存储目录
    :param batch_size: 每批处理页数（不是文件数！）
    """
    import sys
    import traceback
    from src.pdf_processor import PDFProcessor
    
    # 1. 加载已处理进度
    processed_files = load_processed_files()
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    # 2. 创建处理状态文件（记录每页的处理进度）
    progress_file = os.path.join(chroma_db_path, "page_progress.json")
    page_progress = {}
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                page_progress = json.load(f)
        except:
            page_progress = {}
    
    # 3. 找出未处理的页
    pending_pages = []
    for pdf_file in pdf_files:
        if pdf_file in processed_files:
            continue
            
        file_path = os.path.join(pdf_dir, pdf_file)
        
        # 检查这个文件的处理进度
        if pdf_file not in page_progress:
            # 获取PDF总页数
            import fitz
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
            
            page_progress[pdf_file] = {
                "total_pages": total_pages,
                "processed_pages": 0,
                "pages": [False] * total_pages  # False表示未处理
            }
        
        # 添加未处理的页
        for page_idx, processed in enumerate(page_progress[pdf_file]["pages"]):
            if not processed:
                pending_pages.append((pdf_file, page_idx))
    
    # 4. 只处理一批页（限制页数，不限制文件数）
    current_batch = pending_pages[:batch_size]  # 每批只处理10页
    
    st.info(f"📊 进度统计：总待处理页数 {len(pending_pages)}，本次处理 {len(current_batch)} 页")
    
    # 5. 初始化PDF处理器和向量库
    pdf_processor = PDFProcessor()
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 加载或创建向量库
    vector_store = None
    if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
        try:
            vector_store = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=embeddings
            )
        except:
            pass
    
    # 6. 逐页处理（不是逐文件）
    progress_bar = st.progress(0, text="准备处理...")
    
    for idx, (file_name, page_idx) in enumerate(current_batch):
        file_path = os.path.join(pdf_dir, file_name)
        
        # 计算总体进度
        total_done = sum(p["processed_pages"] for p in page_progress.values())
        total_all = sum(p["total_pages"] for p in page_progress.values())
        current_progress = (total_done + idx + 1) / total_all
        
        progress_bar.progress(
            current_progress, 
            text=f"处理: {file_name} 第{page_idx+1}页 ({total_done + idx + 1}/{total_all})"
        )
        
        try:
            # 只加载这一页！
            documents = pdf_processor.load_single_page(file_path, page_idx)
            
            if documents and documents[0].page_content.strip():
                chunks = pdf_processor.split_documents(documents)
                
                if chunks:
                    texts = [chunk.page_content for chunk in chunks]
                    metadatas = [{
                        "source": file_name,
                        "page": page_idx + 1,
                        "content_type": documents[0].metadata.get("content_type", "native")
                    } for chunk in chunks]
                    
                    if vector_store:
                        vector_store.add_texts(texts, metadatas=metadatas)
                    else:
                        vector_store = Chroma.from_texts(
                            texts,
                            embeddings,
                            persist_directory=chroma_db_path,
                            metadatas=metadatas
                        )
                    
                    # 标记这一页已处理
                    page_progress[file_name]["pages"][page_idx] = True
                    page_progress[file_name]["processed_pages"] += 1
                    
                    # 如果这个文件的所有页都处理完了，标记文件已完成
                    if page_progress[file_name]["processed_pages"] == page_progress[file_name]["total_pages"]:
                        save_processed_file(file_name)
                    
                    st.success(f"✅ {file_name} 第{page_idx+1}页 处理完成")
                else:
                    st.warning(f"⚠️ {file_name} 第{page_idx+1}页 分割后无内容")
                    # 仍然标记为已处理，避免卡住
                    page_progress[file_name]["pages"][page_idx] = True
                    page_progress[file_name]["processed_pages"] += 1
            else:
                st.warning(f"⚠️ {file_name} 第{page_idx+1}页 无有效内容")
                page_progress[file_name]["pages"][page_idx] = True
                page_progress[file_name]["processed_pages"] += 1
                
        except Exception as e:
            st.error(f"❌ {file_name} 第{page_idx+1}页 处理失败: {str(e)}")
            # 失败不标记，下次重试
            continue
        
        # 每处理一页就保存进度
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(page_progress, f, ensure_ascii=False, indent=2)
        
        # 每处理一页就持久化
        if vector_store:
            vector_store.persist()
    
    # 7. 更新session state
    if vector_store:
        st.session_state.vector_store = vector_store
        st.session_state.vector_store_initialized = True
        
        # 检查是否全部完成
        all_done = all(
            p["processed_pages"] == p["total_pages"] 
            for p in page_progress.values()
        )
        
        if all_done:
            st.success("🎉 恭喜！所有PDF的所有页面都已处理完成！")
        else:
            remaining = sum(p["total_pages"] - p["processed_pages"] for p in page_progress.values())
            st.info(f"📌 本批次处理完成，还剩 {remaining} 页，请再次点击「生成/继续处理向量库」")
    
    return vector_store

# ========== 新增：加载现有向量库函数（辅助分批） ==========
def load_existing_vector_store(chroma_dir):
    """加载已生成的向量库（避免重复生成）"""
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # 加载已有库
        vector_store = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )
        return vector_store
    except Exception as e:
        st.warning(f"⚠️ 加载现有向量库失败（首次处理正常）：{e}")
        return None

def initialize_vector_store_once():
    """一次性初始化向量库，生成后永久保存"""
    # 如果session state已有，直接返回
    if st.session_state.vector_store_initialized:
        return st.session_state.vector_store
    
    chroma_db_path = get_chroma_db_path()
    pdf_data_path = get_pdf_data_path()
    
    # 检查是否有PDF文件
    if not os.path.exists(pdf_data_path):
        st.sidebar.warning("📄 请先上传PDF文件")
        return None
    
    pdf_files = [f for f in os.listdir(pdf_data_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.sidebar.warning("📄 请先上传PDF文件")
        return None
    
    # 直接返回None，让用户点击"生成/继续处理向量库"按钮
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
        # ========== 新增：向量库分批管理 ==========
        st.subheader("📦 向量库分批管理")
        
        # 1. 新增：批次大小调整（可自定义每批处理多少个）
        batch_size = st.number_input(
            "每批处理文件数",
            min_value=1,    # 最少1个
            max_value=20,   # 最多20个（Cloud建议≤10）
            value=10,       # 默认10个
            key="batch_size"
        )

        # 修改原有生成按钮：调用分批函数
        if st.button("🚀 生成/继续处理向量库", type="primary"):
            with st.spinner("🔄 分批处理中（Cloud环境建议耐心等待）..."):
                vector_store = generate_vector_store_from_pdfs(
                    get_pdf_data_path(),   # PDF目录
                    get_chroma_db_path(),  # Chroma路径
                    batch_size=batch_size  # 批次大小
                )
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.vector_store_initialized = True
                    st.rerun()

        # 3. 新增：重置处理进度按钮（需要重新处理时用）
        if st.button("🔄 重置所有处理进度", type="secondary"):
            clear_processed_files()  # 清空已处理记录
            st.session_state.vector_store = None
            st.success("✅ 已重置进度！可重新开始分批处理")
            st.rerun()
        # ========== 分批管理结束 ==========
        
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
        # ====== OCR状态检查 ======
        st.markdown("---")
        st.markdown("### 🔍 OCR状态检查")

        if st.button("🔧 详细诊断OCR引擎", key="check_ocr_status"):
            with st.spinner("正在收集OCR诊断信息..."):
                try:
                    from src.pdf_processor import PDFProcessor
                    import platform
                    import subprocess
                    
                    processor = PDFProcessor()
                    status = processor.check_ocr_status()
                    
                    # 显示诊断结果
                    col1, col2 = st.columns(2)
                    with col1:
                        if status.get("available"):
                            st.success("✅ OCR引擎: 可用")
                        else:
                            st.error("❌ OCR引擎: 不可用")
                    with col2:
                        st.info(f"初始化状态: {'已初始化' if status.get('initialized') else '未初始化'}")
                    
                    # 详细诊断信息
                    with st.expander("📋 详细诊断信息", expanded=True):
                        # 环境信息
                        st.subheader("🖥️ 环境信息")
                        env_info = {
                            "Python版本": sys.version.split()[0],
                            "操作系统": platform.platform(),
                            "Cloud环境": '是' if is_streamlit_cloud() else '否',
                            "DISPLAY": os.environ.get('DISPLAY', '未设置'),
                            "QT_QPA_PLATFORM": os.environ.get('QT_QPA_PLATFORM', '未设置')
                        }
                        st.json(env_info)
                        
                        # OCR状态
                        st.subheader("🔧 OCR状态")
                        st.json(status)
                        
                        # 系统库检查
                        st.subheader("📦 系统库检查")
                        try:
                            ld_config = subprocess.run(['ldconfig', '-p'], 
                                                     stdout=subprocess.PIPE, 
                                                     stderr=subprocess.PIPE,
                                                     text=True, 
                                                     timeout=5)
                            libgl_found = 'libGL.so' in ld_config.stdout
                            if libgl_found:
                                st.success("✅ libGL.so: 已安装")
                            else:
                                st.error("❌ libGL.so: 未找到")
                        except Exception as e:
                            st.warning(f"⚠️ 无法检查系统库: {e}")
                            
                        # OpenCV信息
                        st.subheader("🎥 OpenCV信息")
                        try:
                            import cv2
                            st.json({
                                "版本": cv2.__version__,
                                "安装路径": cv2.__file__,
                                "headless": 'headless' in cv2.getBuildInformation().lower()
                            })
                        except Exception as e:
                            st.error(f"无法获取OpenCV信息: {e}")
                            
                except Exception as e:
                    st.error(f"❌ 诊断失败: {str(e)}")
                    import traceback
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())



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
                                st.session_state.vector_store_initialized = False
                                st.session_state.vector_store = None
                                
                                # 2. 删除向量库目录
                                chroma_path = get_chroma_db_path()
                                if os.path.exists(chroma_path):
                                    shutil.rmtree(chroma_path)
                                    st.success(f"✅ 已删除向量库目录: {chroma_path}")
                                
                                # 3. 删除状态文件
                                status_file = os.path.join(chroma_path, "generation_status.json")
                                if os.path.exists(status_file):
                                    os.remove(status_file)
                                
                                st.success("🔥 所有向量库数据已清空！")
                            except Exception as e:
                                st.error(f"❌ 清空数据失败: {str(e)}")
                        
                        cleanup_chroma_resources()
                        st.rerun()
            
            elif mode == "🔍 只检索模式":
                st.info("✅ 当前已处于只检索模式，无需操作")
            
            elif mode == "💥 强制全部重新处理":
                st.error("⚠️ **强制重新处理**：将删除现有向量库并重新生成所有数据！")
                if st.button("开始强制重新生成", type="primary", key="force_regen_btn"):
                    # 先清空再重新生成
                    st.session_state.vector_store_initialized = False
                    st.session_state.vector_store = None
                    
                    # 删除状态文件触发重新生成
                    status_file = os.path.join(get_chroma_db_path(), "generation_status.json")
                    if os.path.exists(status_file):
                        os.remove(status_file)
                    
                    # 重新初始化
                    vector_store = generate_vector_store_from_pdfs(get_pdf_data_path(), get_chroma_db_path())
                    if vector_store:
                        st.success("✅ 强制重新生成完成！")
                        st.rerun()
                    else:
                        st.error("❌ 强制重新生成失败！")
                        
                        
                        
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