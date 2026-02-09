import os
import logging
import traceback  # 添加这个导入
import numpy as np
from typing import List, Dict, Any
import fitz  # PyMuPDF，用于PDF转图片（需安装：pip install pymupdf）
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 初始化RapidOCR引擎（全局单例，避免重复初始化）
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()
    logger.info("✓ RapidOCR引擎初始化成功，支持PDF图片页识别")
except ImportError as e:
    OCR_ENGINE = None
    logger.error(f"❌ RapidOCR导入失败，无法处理图片PDF: {e}")
    logger.warning("提示：请安装依赖: pip install rapidocr-onnxruntime pymupdf pillow numpy")


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
        """RapidOCR图片识别核心方法，复用引擎"""
        if not OCR_ENGINE:
            return ""
        try:
            img_np = np.array(img)  # 转为numpy数组（RapidOCR要求）
            result = OCR_ENGINE(img_np)
            # 解析RapidOCR结果，提取文字行
            text_lines = [line[1] for line in result[0] if isinstance(line, (list, tuple)) and len(line) >= 2]
            return "\n".join(text_lines).strip()
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""

    def _extract_pdf_image_page(self, pdf_path: str, page_num: int) -> str:
        """将PDF单页转为图片，调用OCR识别"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            # 放大2倍提升OCR识别精度（matrix=fitz.Matrix(2,2)）
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return self._ocr_from_image(img)
        except Exception as e:
            logger.error(f"PDF转图片失败（第{page_num+1}页）: {e}")
            return ""

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载单个PDF文件（原生文本优先+RapidOCR图片兜底，完全兼容原有返回格式）"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return []

        try:
            logger.info(f"正在加载PDF（原生文本+RapidOCR）: {pdf_path}")
            # 第一步：用PyPDFLoader提取原生文本
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            total_pages = len(documents)
            ocr_count = 0

            # 第二步：遍历每一页，无原生文本则启动OCR兜底
            for page_idx, doc in enumerate(documents):
                native_text = doc.page_content.strip()
                # 原生文本为空/过短，触发OCR
                if not native_text or len(native_text) < 10:
                    if not OCR_ENGINE:
                        logger.warning(f"第{page_idx+1}页无原生文本，但RapidOCR未初始化，跳过")
                        continue
                    logger.info(f"第{page_idx+1}页无原生文本，启动RapidOCR识别...")
                    ocr_text = self._extract_pdf_image_page(pdf_path, page_idx)
                    if ocr_text:
                        doc.page_content = ocr_text
                        ocr_count += 1
                    else:
                        doc.page_content = "[RapidOCR未识别到有效文字]"

                # 保留并补充元数据（完全兼容原有逻辑）
                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["file_path"] = pdf_path
                doc.metadata["page_number"] = page_idx + 1  # 明确页码，方便溯源
                doc.metadata["content_type"] = "ocr" if (not native_text and ocr_text) else "native"  # 标记内容类型

            logger.info(f"PDF加载完成：共{total_pages}页，其中{ocr_count}页由RapidOCR识别")
            return documents
        except Exception as e:
            # 修改这里：不使用exc_info参数
            logger.error(f"加载PDF失败 {pdf_path}: {str(e)}")
            # 如果需要详细错误信息，使用traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return []

    def load_pdfs_from_directory(self, pdf_dir: str) -> List[Document]:
        """从目录加载所有PDF文件（逻辑完全不变，自动适配OCR）"""
        all_documents = []

        if not os.path.exists(pdf_dir):
            logger.error(f"PDF目录不存在: {pdf_dir}")
            return all_documents

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

        if not pdf_files:
            logger.warning(f"在目录 {pdf_dir} 中未找到PDF文件")
            return all_documents

        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始批量加载（含RapidOCR）")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            documents = self.load_pdf(pdf_path)
            all_documents.extend(documents)

        logger.info(f"批量加载完成，共加载 {len(all_documents)} 页PDF内容（含原生+OCR）")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块（逻辑完全不变，无缝处理OCR识别内容）"""
        if not documents:
            logger.warning("没有文档可供分割")
            return []

        logger.info(f"开始分割 {len(documents)} 个文档（含RapidOCR识别内容）")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文本块")

        return chunks