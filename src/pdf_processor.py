import os
import logging
import numpy as np
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== 关键修改：延迟初始化OCR引擎 =====================
# 不要立即初始化，改为占位符
OCR_ENGINE = None
_OCR_INITIALIZED = False
_OCR_TYPE = None  # 记录当前使用的OCR类型

def _init_ocr_engine():
    """延迟初始化OCR引擎 - 优先使用PaddleOCR（纯Python，无libGL依赖）"""
    global OCR_ENGINE, _OCR_INITIALIZED, _OCR_TYPE
    
    if _OCR_INITIALIZED:
        return OCR_ENGINE
    
    # 检查是否是Cloud环境
    is_cloud = 'STREAMLIT_SERVER_TYPE' in os.environ or os.environ.get('HOME') == '/home/appuser'
    
    # 设置无头环境变量
    if is_cloud:
        logger.info("🌐 Cloud环境：配置无头模式")
        os.environ.setdefault('DISPLAY', ':99')
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')  # 禁用GPU
    
    # ===== 方案1：PaddleOCR（纯Python，无libGL依赖）=====
    try:
        logger.info("🔄 尝试初始化 PaddleOCR（纯CPU模式，无libGL依赖）...")
        
        # 设置PaddleOCR环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['CPU_NUM'] = '1'
        
        from paddleocr import PaddleOCR
        
        OCR_ENGINE = PaddleOCR(
            use_angle_cls=False,      # 关闭角度分类，加速
            lang='ch',               # 中英文识别
            use_gpu=False,           # 强制CPU
            show_log=False,          # 关闭PaddleOCR内部日志
            enable_mkldnn=True,      # 启用MKLDNN加速（CPU）
            cpu_threads=2,           # CPU线程数
            det_db_thresh=0.3,       # 检测阈值
            det_db_box_thresh=0.5,   # 检测框阈值
            det_db_unclip_ratio=1.6  # 检测框扩展比例
        )
        _OCR_INITIALIZED = True
        _OCR_TYPE = 'PaddleOCR'
        logger.info("✅ PaddleOCR引擎初始化成功（纯CPU，无libGL依赖）")
        return OCR_ENGINE
        
    except ImportError as e:
        logger.warning(f"⚠️ PaddleOCR导入失败: {e}")
        logger.warning("如需安装PaddleOCR: pip install paddleocr paddlepaddle")
    except Exception as e:
        logger.warning(f"⚠️ PaddleOCR初始化失败: {e}")
    
    # ===== 方案2：RapidOCR（备选方案，可能需要libGL）=====
    try:
        logger.info("🔄 尝试初始化 RapidOCR（备选方案）...")
        
        from rapidocr_onnxruntime import RapidOCR
        
        OCR_ENGINE = RapidOCR(
            use_gpu=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            use_angle_cls=False
        )
        _OCR_INITIALIZED = True
        _OCR_TYPE = 'RapidOCR'
        logger.info("✅ RapidOCR引擎初始化成功（备选方案）")
        return OCR_ENGINE
        
    except ImportError as e:
        logger.error(f"❌ RapidOCR导入失败: {e}")
    except Exception as e:
        logger.error(f"❌ RapidOCR初始化失败: {e}")
    
    # ===== 所有OCR引擎都失败 =====
    logger.error("❌ 所有OCR引擎均初始化失败，OCR功能将不可用")
    OCR_ENGINE = None
    return None

def get_ocr_engine():
    """获取OCR引擎（外部调用接口）"""
    return _init_ocr_engine()

def get_ocr_type():
    """获取当前使用的OCR引擎类型"""
    global _OCR_TYPE
    return _OCR_TYPE

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )
        # 延迟初始化OCR引擎
        self._ocr_engine = None
        self._ocr_initialized = False
        self._ocr_type = None  # 记录当前使用的OCR类型

    def _ocr_from_image(self, img: Image.Image) -> str:
        """OCR图片识别核心方法（兼容PaddleOCR和RapidOCR）"""
        try:
            # 获取OCR引擎（延迟初始化）
            ocr_engine = get_ocr_engine()
            if not ocr_engine:
                logger.warning("OCR引擎未初始化")
                return ""
            
            # 转换图像格式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_np = np.array(img)
            
            # 获取当前使用的OCR类型
            ocr_type = get_ocr_type()
            
            if ocr_type == 'PaddleOCR':
                # PaddleOCR 结果解析
                result = ocr_engine.ocr(img_np, cls=False)
                text_lines = []
                
                if result and result[0]:
                    for line in result[0]:
                        # PaddleOCR返回格式: [[[坐标], (文字, 置信度)]]
                        if len(line) >= 2 and len(line[1]) >= 2:
                            text_lines.append(line[1][0])
                
                text = "\n".join(text_lines).strip()
                if text:
                    logger.info(f"✅ PaddleOCR识别成功，字符数: {len(text)}")
                    return text
                else:
                    logger.warning("⚠️ PaddleOCR识别无结果")
                    return ""
                    
            else:
                # RapidOCR 结果解析（默认）
                result = ocr_engine(img_np)
                text_lines = []
                
                if result and isinstance(result, (list, tuple)) and len(result) > 0:
                    for line in result[0]:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_lines.append(str(line[1]))
                
                text = "\n".join(text_lines).strip()
                if text:
                    logger.info(f"✅ RapidOCR识别成功，字符数: {len(text)}")
                    return text
                else:
                    logger.warning("⚠️ RapidOCR识别无结果")
                    return ""
                
        except Exception as e:
            logger.error(f"❌ OCR识别失败: {e}", exc_info=True)
            return ""

    # ===================== OCR状态检查方法（增强版）=====================
    def check_ocr_status(self) -> Dict[str, Any]:
        """检查OCR引擎状态（支持PaddleOCR和RapidOCR）"""
        status = {
            "available": False,
            "initialized": _OCR_INITIALIZED,
            "ocr_type": get_ocr_type(),
            "environment": {},
            "dependencies": {}
        }
        
        try:
            # 检查OpenCV（可能不存在，但不影响PaddleOCR）
            try:
                import cv2
                status["environment"]["opencv_version"] = cv2.__version__
                build_info = cv2.getBuildInformation()
                status["environment"]["opencv_headless"] = 'headless' in build_info.lower()
            except ImportError:
                status["environment"]["opencv_version"] = "未安装"
            except Exception as e:
                status["environment"]["opencv_error"] = str(e)
            
            # 检查PIL
            try:
                from PIL import Image, __version__ as pil_version
                status["environment"]["pil_version"] = pil_version
            except ImportError:
                status["environment"]["pil_version"] = "未安装"
            
            # 检查PaddleOCR
            try:
                import paddleocr
                status["dependencies"]["paddleocr_available"] = True
                status["dependencies"]["paddleocr_version"] = getattr(paddleocr, '__version__', '未知')
            except ImportError:
                status["dependencies"]["paddleocr_available"] = False
            
            # 检查RapidOCR
            try:
                import rapidocr_onnxruntime
                status["dependencies"]["rapidocr_available"] = True
                status["dependencies"]["rapidocr_version"] = getattr(rapidocr_onnxruntime, '__version__', '未知')
            except ImportError:
                status["dependencies"]["rapidocr_available"] = False
            
            # 检查ONNX Runtime
            try:
                import onnxruntime as ort
                status["dependencies"]["onnxruntime_version"] = ort.__version__
                status["dependencies"]["onnxruntime_providers"] = ort.get_available_providers()
            except ImportError:
                status["dependencies"]["onnxruntime_version"] = "未安装"
            
            # 测试初始化OCR引擎
            ocr_engine = get_ocr_engine()
            status["available"] = ocr_engine is not None
            status["initialized"] = _OCR_INITIALIZED
            status["ocr_type"] = get_ocr_type()
            
        except Exception as e:
            status["error"] = str(e)
            import traceback
            status["traceback"] = traceback.format_exc()
            
        return status

    # ===================== 核心修改：按PDF文件加载（而非按页）=====================
    def _extract_pdf_full_text(self, pdf_path: str) -> str:
        """提取整个PDF的完整文本（原生文本+OCR兜底）"""
        full_text = []
        try:
            # 第一步：尝试读取原生文本（整份PDF）
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            native_text = "\n\n".join([page.page_content.strip() for page in pages if page.page_content.strip()])
            
            if native_text and len(native_text) > 50:  # 原生文本足够多，直接使用
                full_text.append(native_text)
                logger.info(f"✅ {pdf_path} 原生文本提取成功，字符数: {len(native_text)}")
            else:
                # 第二步：原生文本不足，整份PDF逐页OCR
                logger.info(f"⚠️ {pdf_path} 原生文本不足，启动整份PDF OCR识别...")
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                
                for page_idx in range(total_pages):
                    page = doc.load_page(page_idx)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = self._ocr_from_image(img)
                    if ocr_text:
                        full_text.append(f"=== 第{page_idx+1}页 ===\n{ocr_text}")
                
                doc.close()
                logger.info(f"✅ {pdf_path} OCR识别完成，共处理{total_pages}页")
            
            return "\n\n".join(full_text).strip()
        
        except Exception as e:
            logger.error(f"❌ {pdf_path} 文本提取失败: {e}", exc_info=True)
            return ""

    def load_single_pdf(self, pdf_path: str) -> Document:
        """加载单个PDF为单个Document对象（按文件分批）"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return None
        
        # 提取整份PDF的完整文本
        full_text = self._extract_pdf_full_text(pdf_path)
        if not full_text:
            full_text = "[该PDF未提取到有效文本]"
        
        # 创建单个Document对象（代表整个PDF）
        doc = Document(
            page_content=full_text,
            metadata={
                "source": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "total_pages": len(fitz.open(pdf_path)) if fitz.open(pdf_path) else 0,
                "content_type": "native" if "=== 第" not in full_text else "ocr",
                "processed_at": str(pd.Timestamp.now())  # 需导入pandas：import pandas as pd
            }
        )
        logger.info(f"✅ {pdf_path} 已加载为单个Document对象")
        return doc

    def load_pdfs_batch(self, pdf_paths: List[str]) -> List[Document]:
        """批量加载多个PDF（每个PDF对应一个Document对象）"""
        batch_docs = []
        for pdf_path in pdf_paths:
            doc = self.load_single_pdf(pdf_path)
            if doc:
                batch_docs.append(doc)
        
        logger.info(f"✅ 批量加载完成，共处理{len(pdf_paths)}个PDF，成功{len(batch_docs)}个")
        return batch_docs

    def load_pdfs_from_directory(self, pdf_dir: str, batch_size: int = None) -> List[List[Document]]:
        """从目录加载PDF，支持按批次返回（每批N个PDF文件）"""
        all_docs = []
        batch_docs = []
        
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF目录不存在: {pdf_dir}")
            return []
        
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始按文件分批加载")
        
        for idx, pdf_path in enumerate(pdf_files):
            doc = self.load_single_pdf(pdf_path)
            if doc:
                batch_docs.append(doc)
            
            # 如果设置了批次大小，达到批次大小则提交
            if batch_size and len(batch_docs) >= batch_size:
                all_docs.append(batch_docs)
                logger.info(f"📦 批次{len(all_docs)}已完成，包含{len(batch_docs)}个PDF")
                batch_docs = []
        
        # 处理最后一批
        if batch_docs:
            all_docs.append(batch_docs)
            logger.info(f"📦 最后一批已完成，包含{len(batch_docs)}个PDF")
        
        logger.info(f"✅ 目录加载完成，共生成{len(all_docs)}个批次，总计{len(pdf_files)}个PDF")
        return all_docs

    # 保留原有的split_documents，但逻辑改为：分割单个PDF的完整文本为多个chunk
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档（每个PDF的完整文本拆分为多个chunk）"""
        if not documents:
            logger.warning("没有文档可供分割")
            return []

        logger.info(f"开始分割 {len(documents)} 个PDF文件的文本")
        all_chunks = []
        
        for doc in documents:
            # 对单个PDF的完整文本进行分割
            chunks = self.text_splitter.split_text(doc.page_content)
            # 为每个chunk保留原PDF的元数据
            for idx, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    }
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"分割完成，共生成 {len(all_chunks)} 个文本块（来自{len(documents)}个PDF）")
        return all_chunks