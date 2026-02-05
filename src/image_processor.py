
import streamlit as st
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        """初始化ImageProcessor - 延迟加载OCR引擎"""
        self.engine = None
        self._engine_initialized = False
    
    def _init_engine(self):
        """延迟初始化OCR引擎"""
        if not self._engine_initialized:
            try:
                # 延迟导入，避免启动时加载
                from rapidocr_onnxruntime import RapidOCR
                with st.spinner("🔄 正在加载OCR引擎..."):
                    self.engine = RapidOCR()
                self._engine_initialized = True
                st.success("✅ OCR引擎加载完成")
            except Exception as e:
                st.error(f"❌ OCR引擎加载失败: {str(e)}")
                self.engine = None
                self._engine_initialized = True
    
    def process_uploaded_image(self, uploaded_file):
        """
        使用RapidOCR识别图片文字
        """
        try:
            if uploaded_file is None:
                return {"success": False, "message": "未上传文件"}
            
            # 延迟初始化OCR引擎
            self._init_engine()
            
            if self.engine is None:
                return {"success": False, "message": "OCR引擎未初始化"}
            
            # 读取图片
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # 使用RapidOCR识别文字
            with st.spinner("正在识别图片中的文字..."):
                result, elapse = self.engine(img_array)
            
            # 提取识别结果
            text_lines = []
            confidences = []
            if result:
                for detection in result:
                    text, confidence = detection[1], detection[2]
                    if confidence > 0.3:  # 置信度阈值
                        text_lines.append(text)
                        confidences.append(confidence)
            
            recognized_text = '\n'.join(text_lines)
            
            if not recognized_text.strip():
                return {
                    "success": False,
                    "message": "未识别到文字，请尝试更清晰的图片",
                    "text": ""
                }
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "success": True,
                "text": recognized_text,
                "confidence": f"{avg_confidence:.2%}",
                "lines_count": len(text_lines),
                "message": f"识别成功！找到 {len(text_lines)} 行文字，平均置信度 {avg_confidence:.2%}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"图片处理失败: {str(e)}",
                "text": ""
            }
    
    def display_image_preview(self, uploaded_file, caption="上传的图片"):
        """
        显示图片预览
        """
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # 修复：使用 use_container_width=True 替代 use_column_width
            st.image(image, caption=caption, use_container_width=True)

# 创建全局实例
image_processor = ImageProcessor()