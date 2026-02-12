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

    # ===================== 以下原有代码完全不变 =====================
    def _extract_pdf_image_page(self, pdf_path: str, page_num: int) -> str:
        """将PDF单页转为图片，调用OCR识别"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return self._ocr_from_image(img)
        except Exception as e:
            logger.error(f"PDF转图片失败（第{page_num+1}页）: {e}")
            return ""

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载单个PDF文件（原生文本优先+OCR图片兜底）"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []

        try:
            logger.info(f"正在加载PDF（原生文本+OCR）: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            total_pages = len(documents)
            ocr_count = 0

            for page_idx, doc in enumerate(documents):
                native_text = doc.page_content.strip()
                if not native_text or len(native_text) < 10:
                    if not get_ocr_engine():
                        logger.warning(f"第{page_idx+1}页无原生文本，但OCR未初始化，跳过")
                        continue
                    logger.info(f"第{page_idx+1}页无原生文本，启动OCR识别...")
                    ocr_text = self._extract_pdf_image_page(pdf_path, page_idx)
                    if ocr_text:
                        doc.page_content = ocr_text
                        ocr_count += 1
                        doc.metadata["content_type"] = "ocr"
                    else:
                        doc.page_content = "[OCR未识别到有效文字]"
                        doc.metadata["content_type"] = "ocr_failed"
                else:
                    doc.metadata["content_type"] = "native"

                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["file_path"] = pdf_path
                doc.metadata["page_number"] = page_idx + 1

            logger.info(f"PDF加载完成：共{total_pages}页，其中{ocr_count}页由OCR识别")
            return documents
        except Exception as e:
            logger.error(f"加载PDF失败 {pdf_path}: {str(e)}", exc_info=True)
            return []

    def load_pdfs_from_directory(self, pdf_dir: str) -> List[Document]:
        """从目录加载所有PDF文件"""
        all_documents = []

        if not os.path.exists(pdf_dir):
            logger.error(f"PDF目录不存在: {pdf_dir}")
            return all_documents

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return all_documents

        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始批量加载")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            documents = self.load_pdf(pdf_path)
            all_documents.extend(documents)

        logger.info(f"批量加载完成，共加载 {len(all_documents)} 页PDF内容")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        if not documents:
            logger.warning("没有文档可供分割")
            return []

        logger.info(f"开始分割 {len(documents)} 个文档")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文本块")

        return chunks