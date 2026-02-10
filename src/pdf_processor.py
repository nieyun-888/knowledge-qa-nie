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

def _init_ocr_engine():
    """延迟初始化OCR引擎（只在需要时初始化）"""
    global OCR_ENGINE, _OCR_INITIALIZED
    
    if _OCR_INITIALIZED:
        return OCR_ENGINE
        
    try:
        # 检查是否是Cloud环境
        is_cloud = 'STREAMLIT_SERVER_TYPE' in os.environ
        
        if is_cloud:
            logger.info("🌐 Cloud环境：配置无头模式")
            # 设置无头环境变量
            os.environ.setdefault('DISPLAY', ':99')
            os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        
        from rapidocr_onnxruntime import RapidOCR
        # Cloud环境强制使用CPU
        OCR_ENGINE = RapidOCR(use_gpu=False)
        _OCR_INITIALIZED = True
        logger.info("✅ RapidOCR引擎初始化成功")
        
    except ImportError as e:
        logger.error(f"❌ RapidOCR导入失败: {e}")
        OCR_ENGINE = None
    except Exception as e:
        logger.error(f"❌ RapidOCR初始化失败: {e}")
        OCR_ENGINE = None
        
    return OCR_ENGINE

def get_ocr_engine():
    """获取OCR引擎（外部调用接口）"""
    return _init_ocr_engine()


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

    def _ocr_from_image(self, img: Image.Image) -> str:
        """RapidOCR图片识别核心方法"""
        try:
            # 获取OCR引擎（延迟初始化）
            ocr_engine = get_ocr_engine()
            if not ocr_engine:
                return ""
                
            img_np = np.array(img)
            result = ocr_engine(img_np)
            
            # 解析结果
            text_lines = [line[1] for line in result[0] if isinstance(line, (list, tuple)) and len(line) >= 2]
            return "\n".join(text_lines).strip()
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""

    # ===================== 新增：OCR状态检查方法 =====================
    def check_ocr_status(self) -> Dict[str, Any]:
        """检查OCR引擎状态"""
        status = {
            "available": False,
            "initialized": _OCR_INITIALIZED,
            "environment": {}
        }
        
        try:
            # 检查OpenCV
            import cv2
            status["environment"]["opencv_version"] = cv2.__version__
            
            # 检查PIL
            from PIL import Image
            status["environment"]["pil_version"] = Image.__version__
            
            # 检查RapidOCR
            try:
                from rapidocr_onnxruntime import RapidOCR
                status["environment"]["rapidocr_available"] = True
                
                # 测试引擎
                ocr_engine = get_ocr_engine()
                status["available"] = ocr_engine is not None
                
            except ImportError:
                status["environment"]["rapidocr_available"] = False
                
        except Exception as e:
            status["error"] = str(e)
            
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
        """加载单个PDF文件（原生文本优先+RapidOCR图片兜底）"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []

        try:
            logger.info(f"正在加载PDF（原生文本+RapidOCR）: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            total_pages = len(documents)
            ocr_count = 0

            for page_idx, doc in enumerate(documents):
                native_text = doc.page_content.strip()
                if not native_text or len(native_text) < 10:
                    if not get_ocr_engine():
                        logger.warning(f"第{page_idx+1}页无原生文本，但RapidOCR未初始化，跳过")
                        continue
                    logger.info(f"第{page_idx+1}页无原生文本，启动RapidOCR识别...")
                    ocr_text = self._extract_pdf_image_page(pdf_path, page_idx)
                    if ocr_text:
                        doc.page_content = ocr_text
                        ocr_count += 1
                    else:
                        doc.page_content = "[RapidOCR未识别到有效文字]"

                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["file_path"] = pdf_path
                doc.metadata["page_number"] = page_idx + 1
                doc.metadata["content_type"] = "ocr" if (not native_text and ocr_text) else "native"

            logger.info(f"PDF加载完成：共{total_pages}页，其中{ocr_count}页由RapidOCR识别")
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