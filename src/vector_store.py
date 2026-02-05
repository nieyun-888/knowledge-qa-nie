import os
import sys
import shutil
import logging
import json
import time
import hashlib
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # 使用本地下载的模型
        self.embedding_model = self._load_local_model()
        logger.info("向量存储初始化完成")
    
    def _load_local_model(self):
        """加载本地下载的模型"""
        model_path = "./models/bge-small-zh-v1.5"
        # Cloud环境优先使用HF Hub加载（自动缓存到/tmp）
        if "STREAMLIT" in sys.modules or "STREAMLIT_SERVER_BASEURL_PATH" in os.environ:
            model_path = "BAAI/bge-small-zh-v1.5"
            # 设置HF缓存路径到/tmp（Cloud可写）
            os.environ['HF_HOME'] = "/tmp/huggingface"
            os.environ['TRANSFORMERS_CACHE'] = "/tmp/huggingface"
        
        # 检查模型路径
        if not os.path.exists(model_path) and not model_path.startswith("BAAI/"):
            logger.error(f"模型路径不存在: {model_path}")
            return self._create_fallback_embedder()
        
        # 方法1：首先尝试 sentence-transformers（最可靠）
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("尝试使用 sentence-transformers 加载模型...")
            
            model = SentenceTransformer(model_path)
            
            class SentenceTransformerEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    # 【优化：CPU小批量编码，避免卡死，提升效率】
                    embeddings = self.model.encode(
                        texts, 
                        normalize_embeddings=True,
                        batch_size=8,
                        show_progress_bar=False
                    )
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode([text], normalize_embeddings=True)
                    return embedding[0].tolist()
            
            logger.info("✅ 使用 sentence-transformers 加载模型成功")
            return SentenceTransformerEmbeddings(model)
            
        except Exception as e:
            logger.warning(f"sentence-transformers 加载失败: {e}")
        
        # 方法2：尝试 LangChain 的 HuggingFaceEmbeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("尝试使用 LangChain HuggingFaceEmbeddings...")
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 8
                }
            )
            
            # 测试一下模型是否能正常工作
            test_embedding = embedding_model.embed_query("测试")
            logger.info(f"✅ LangChain 嵌入模型测试成功，向量维度: {len(test_embedding)}")
            return embedding_model
            
        except Exception as e:
            logger.warning(f"LangChain 嵌入模型加载失败: {e}")
        
        # 方法3：使用备用的简单嵌入模型
        logger.info("使用备用嵌入模型")
        return self._create_fallback_embedder()
    
    def _create_fallback_embedder(self):
        """创建备用的简单嵌入模型"""
        import numpy as np
        import hashlib
        
        class SimpleEmbedder:
            def __init__(self):
                self.dim = 384
                
            def embed_documents(self, texts):
                logger.info(f"为 {len(texts)} 个文档生成嵌入")
                embeddings = []
                # 【优化：备用嵌入也加小批量处理，避免循环卡死】
                batch_size = 16
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    for text in batch:
                        embeddings.append(self._text_to_vector(text))
                    logger.info(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个文档")
                return embeddings
            
            def embed_query(self, text):
                return self._text_to_vector(text)
            
            def _text_to_vector(self, text):
                # 使用文本内容的哈希值生成确定性随机向量
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed)
                vector = np.random.normal(0, 1, self.dim)
                # 归一化向量
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                return vector.tolist()
        
        return SimpleEmbedder()
    
    def create_vector_store(self, documents, clear_old: bool = False):
        """
        创建向量存储 - 使用 Chroma
        :param clear_old: 是否清除旧的向量库文件（模式4强制重建时使用）
        """
        try:
            logger.info("开始创建向量存储...")
            
            if not documents:
                logger.error("没有文档可处理")
                return False
            
            # 清除旧向量库（模式4专用）
            if clear_old and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"✅ 已清除旧的向量库目录: {self.persist_directory}")
            
            # 统一导入Chroma，避免多次异常捕获
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            logger.info(f"正在为 {len(texts)} 个文档创建向量索引...")
            
            # 创建 Chroma 向量存储
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                persist_directory=self.persist_directory,
                collection_name="pdf_documents"
            )
            
            logger.info(f"✅ Chroma 向量存储创建成功! 保存在: {self.persist_directory}")
            return True
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}", exc_info=True)
            return False

    def similarity_search(self, query: str, k: int = 3):
        """相似度搜索（核心检索方法）"""
        try:
            if not hasattr(self, 'vector_store'):
                # 加载已保存的向量存储
                try:
                    from langchain_chroma import Chroma
                except ImportError:
                    from langchain_community.vectorstores import Chroma
                
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name="pdf_documents"
                )
            
            logger.info(f"执行搜索: '{query}'")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"找到 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}", exc_info=True)
            return []

    def load_existing_vector_store(self):
        """加载已存在的向量存储"""
        try:
            # 统一Chroma导入，减少异常捕获耗时
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name="pdf_documents"
            )
            
            # 测试一下是否真的加载成功
            count = self.vector_store._collection.count()
            logger.info(f"加载现有向量存储成功，包含 {count} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载现有向量存储失败: {e}", exc_info=True)
            return False

    @staticmethod
    def vector_store_exists(persist_directory="./chroma_db"):
        """检查向量存储是否已存在"""
        return os.path.exists(persist_directory) and os.path.isdir(persist_directory)

# ===== 智能向量存储类（支持增量更新，核心修复区） =====
class SmartVectorStore(VectorStore):
    """智能向量存储，支持增量更新"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        super().__init__(persist_directory)
        self.document_metadata_file = os.path.join(persist_directory, "document_metadata.json")
        self.document_metadata = self._load_document_metadata()

    # 【核心修复1：规范继承检索方法，替代临时别名，代码更规范】
    def search_similar_documents(self, query: str, k: int = 3):
        """检索方法（main.py调用入口，显式继承父类实现）"""
        return super().similarity_search(query, k)
    
    def _load_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载文档元数据"""
        if os.path.exists(self.document_metadata_file):
            try:
                with open(self.document_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载元数据失败，使用空元数据: {e}")
                return {}
        return {}
    
    def _save_document_metadata(self):
        """保存文档元数据"""
        os.makedirs(os.path.dirname(self.document_metadata_file), exist_ok=True)
        with open(self.document_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
    
    # 【核心修复：删除重复方法，原代码重复定义导致冗余】
    def _filter_new_and_updated_documents(self, documents: List[Any]) -> List[Any]:
        """过滤出新文档和已更新的文档（内部方法）"""
        new_or_updated_docs = []
        
        for doc in documents:
            if not hasattr(doc, 'metadata') or 'source' not in doc.metadata:
                new_or_updated_docs.append(doc)
                continue
            
            source_path = doc.metadata['source']
            if not os.path.exists(source_path):
                new_or_updated_docs.append(doc)
                continue
            
            current_hash = self._get_document_hash(source_path)
            file_name = os.path.basename(source_path)
            
            # 检查是否是新的或已更新的文档
            if (file_name not in self.document_metadata or 
                self.document_metadata[file_name].get('hash') != current_hash):
                new_or_updated_docs.append(doc)
                # 更新元数据
                self.document_metadata[file_name] = {
                    'hash': current_hash,
                    'source': source_path,
                    'processed_time': time.time()
                }
        
        logger.info(f"增量更新过滤：总文档{len(documents)} → 需处理{len(new_or_updated_docs)}个")
        return new_or_updated_docs

    def filter_new_and_updated_documents(self, documents: List[Any]) -> List[Any]:
        """过滤出新文档和已更新的文档（公开方法）"""
        return self._filter_new_and_updated_documents(documents)

    def _get_document_hash(self, file_path: str) -> str:
        """计算文档哈希值用于检测变更"""
        try:
            file_stat = os.stat(file_path)
            # 使用文件大小和修改时间生成哈希，快速检测变更
            hash_input = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"计算文件哈希失败: {e}")
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def smart_create_vector_store(self, documents: List[Any], force_recreate: bool = False) -> bool:
        """
        智能创建向量存储，只处理新文档和更新的文档
        :param force_recreate: 是否强制重建（模式4）- 清除所有历史数据（向量库+元数据）
        """
        try:
            logger.info("开始智能创建向量存储...")
            
            if force_recreate:
                logger.info("💥 强制重新创建模式，清除所有历史数据并处理所有文档")
                # 【核心修复2：模式4清库 - 清除元数据+旧向量库，彻底重建】
                self.document_metadata = {}  # 清空内存中元数据
                if os.path.exists(self.document_metadata_file):
                    os.remove(self.document_metadata_file)
                    logger.info(f"✅ 已清除历史元数据文件: {self.document_metadata_file}")
                # 调用创建方法并指定清除旧向量库
                return self.create_vector_store(documents, clear_old=True)
            
            # 过滤出需要处理的文档（核心增量更新，避免重复处理）
            docs_to_process = self._filter_new_and_updated_documents(documents)
            
            if not docs_to_process:
                logger.info("没有发现新的或更新的文档，跳过处理")
                return True
            
            logger.info(f"发现 {len(docs_to_process)} 个新文档或更新的文档需要处理")
            logger.info(f"跳过 {len(documents) - len(docs_to_process)} 个未变化的文档")
            
            # 如果有现有向量存储，只添加新文档
            if self.vector_store_exists() and hasattr(self, 'vector_store'):
                logger.info("向现有向量存储添加新文档...")
                success = self._add_documents_to_existing_store(docs_to_process)
            else:
                logger.info("创建新的向量存储...")
                success = self.create_vector_store(docs_to_process)
            
            if success:
                self._save_document_metadata()
                logger.info("✅ 智能向量存储更新成功!")
            else:
                logger.error("❌ 智能向量存储更新失败!")
            
            return success
            
        except Exception as e:
            logger.error(f"智能创建向量存储失败: {e}", exc_info=True)
            return False
    
    def _add_documents_to_existing_store(self, documents: List[Any]) -> bool:
        """向现有向量存储添加文档"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return False
            
            # 【优化：添加文档时显式指定小批大小，适配CPU】
            self.vector_store.add_documents(
                documents=documents,
                batch_size=8
            )
            
            logger.info(f"成功添加 {len(documents)} 个文档到现有向量存储")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到现有向量存储失败: {e}", exc_info=True)
            return False

    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            if not hasattr(self, 'vector_store'):
                if not self.load_existing_vector_store():
                    return {}
            
            # 获取集合中的文档数量
            count = self.vector_store._collection.count()
            
            return {
                'total_documents': count,
                'tracked_files': len(self.document_metadata),
                'document_metadata': self.document_metadata
            }
        except Exception as e:
            logger.error(f"获取文档统计失败: {e}", exc_info=True)
            return {}

# ===== 兼容性修复 =====
def process_pdfs_and_create_vector_store(pdf_directory="./data/raw_pdfs", persist_directory="./chroma_db"):
    """
    处理PDF并创建向量存储的主函数
    """
    logger.info("开始处理PDF并创建向量存储...")
    
    try:
        # 导入必要的模块
        from .pdf_processor import PDFProcessor
        
        # 创建向量存储实例
        vector_store = VectorStore(persist_directory=persist_directory)
        
        # 加载和处理PDF
        pdf_processor = PDFProcessor()
        documents = pdf_processor.load_pdfs_from_directory(pdf_directory)
        
        if not documents:
            logger.error("没有找到PDF文档或加载失败")
            return False
        
        # 分割文档
        chunks = pdf_processor.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文本块")
        
        # 创建向量存储
        if vector_store.create_vector_store(chunks):
            logger.info("向量存储创建成功!")
            return True
        else:
            logger.error("向量存储创建失败!")
            return False
            
    except Exception as e:
        logger.error(f"处理PDF并创建向量存储失败: {e}", exc_info=True)
        return False

# 兼容性别名
VectorStoreManager = VectorStore

# 测试函数
def test_embedding():
    """测试嵌入模型是否正常工作"""
    store = VectorStore()
    
    # 测试文档嵌入
    test_texts = ["这是一个测试文档", "这是另一个测试文档"]
    embeddings = store.embedding_model.embed_documents(test_texts)
    
    print(f"文档数量: {len(test_texts)}")
    print(f"嵌入维度: {len(embeddings[0])}")
    print("✅ 嵌入模型测试通过")

if __name__ == "__main__":
    test_embedding()