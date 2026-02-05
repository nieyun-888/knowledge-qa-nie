import logging
import os
import sys
import io
# 核心修复：将项目根目录添加到Python路径，确保能导入src下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import PDFProcessor
from src.vector_store import (
    VectorStoreManager, process_pdfs_and_create_vector_store,
    VectorStore, SmartVectorStore
)
from config import Config

# 配置HF镜像源（提前配置，避免后续导入库时网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 修复控制台编码问题（加固，兼容多系统）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 配置日志（添加文件输出，方便排查问题）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('pdf_vectordb.log', encoding='utf-8')  # 日志文件输出
    ]
)
logger = logging.getLogger(__name__)

def smart_processing_mode():
    """智能处理模式：只处理新文档"""
    print("\n🔄 智能处理模式启动...")
    
    # 确保目录存在（调用Config方法，兜底创建）
    Config.create_directories()
    
    # 校验PDF源目录是否存在且有文件
    pdf_dir = "./data/raw_pdfs"
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF源目录不存在: {pdf_dir}，已自动创建")
        os.makedirs(pdf_dir, exist_ok=True)
        return False
    # 检查目录下是否有PDF文件
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"❌ PDF目录 {pdf_dir} 中无任何PDF文件")
        return False
    
    # 创建智能向量存储实例
    vector_store = SmartVectorStore()
    
    # 加载和处理PDF
    pdf_processor = PDFProcessor()
    all_documents = pdf_processor.load_pdfs_from_directory(pdf_dir)
    
    if not all_documents:
        print("❌ PDF加载失败，无有效文档")
        logger.error("PDF加载结果为空，终止智能处理")
        return False
    
    print(f"📄 成功加载 {len(all_documents)} 页PDF内容")
    
    # 分割文档
    chunks = pdf_processor.split_documents(all_documents)
    if not chunks:
        print("❌ 文档分割失败，无有效文本块")
        logger.error("文档分割结果为空，终止智能处理")
        return False
    print(f"🔪 分割完成，共生成 {len(chunks)} 个文本块")
    
    # 使用智能方式创建向量存储（只处理新文档）
    success = vector_store.smart_create_vector_store(chunks)
    
    if success:
        print("\n🎉 智能处理完成！")
        return True
    else:
        print("\n❌ 处理失败，请检查日志文件 pdf_vectordb.log")
        return False

def search_only_mode(vector_store=None):
    """只检索模式，不重新处理文档"""
    if vector_store is None:
        vector_store = SmartVectorStore()
    
    # 检查向量存储是否存在
    if not vector_store.vector_store_exists():
        print("❌ 未找到向量存储，请先运行【自动模式】或【智能处理模式】创建向量库")
        return False
    
    # 直接加载现有向量存储
    print("📂 加载现有向量存储...")
    if vector_store.load_existing_vector_store():
        print("✅ 向量存储加载成功！")
        
        # 显示统计信息
        stats = vector_store.get_document_stats()
        print(f"📊 向量库统计：总文本块数: {stats.get('total_documents', 'N/A')}")
        
        # 进入搜索循环
        while True:
            print("\n" + "="*60)
            print("🔍 搜索功能 (输入 'quit/exit/退出/q' 退出)")
            print("="*60)
            
            ques = input("请输入问题：").strip()
            if ques.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
                
            if not ques:
                print("⚠️ 请输入有效问题")
                continue
                
            print("🔎 正在搜索相关文档...")
            # 搜索10个最相关结果
            results = vector_store.search_similar_documents(ques, k=10)
            
            if results:
                print(f"\n🎯 找到 {len(results)} 个最相关结果:")
                print("="*80)
                
                for i, doc in enumerate(results):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))  # 兼容page_number/page元数据
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content  # 内容截断，避免刷屏
                    
                    print(f"\n📄 结果 {i+1}/{len(results)}")
                    print(f"📁 来源文件: {os.path.basename(source)}")
                    print(f"📄 页码: {page}")
                    print(f"📝 内容预览:")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)
                    print(f"📍 结果结束 [{i+1}/{len(results)}]")
                    print("="*80)
            else:
                print("❌ 未找到相关文档，请尝试调整问题描述")
        return True
    else:
        print("❌ 向量存储加载失败，请检查日志")
        return False

def force_reprocess_all():
    """强制重新处理所有文档"""
    print("🔄 强制重新处理所有文档模式启动...")
    
    # 校验PDF源目录
    pdf_dir = "./data/raw_pdfs"
    if not os.path.exists(pdf_dir) or not [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]:
        print(f"❌ PDF目录 {pdf_dir} 不存在或无PDF文件")
        return False
    
    vector_store = SmartVectorStore()
    pdf_processor = PDFProcessor()
    all_documents = pdf_processor.load_pdfs_from_directory(pdf_dir)
    
    if not all_documents:
        print("❌ PDF加载失败，无有效文档")
        return False
    
    print(f"📄 成功加载 {len(all_documents)} 页PDF内容")
    chunks = pdf_processor.split_documents(all_documents)
    
    if not chunks:
        print("❌ 文档分割失败，无有效文本块")
        return False
    print(f"🔪 分割完成，共生成 {len(chunks)} 个文本块")
    
    # 强制重新创建向量库
    success = vector_store.smart_create_vector_store(chunks, force_recreate=True)
    
    if success:
        print("✅ 强制重新处理所有文档完成!")
        return True
    else:
        print("❌ 强制重新处理失败，请检查日志")
        return False

def auto_mode():
    """自动模式：检查并智能处理，然后进入搜索"""
    print("\n🤖 自动模式启动（推荐）...")
    
    # 创建智能向量存储实例
    vector_store = SmartVectorStore()
    
    # 检查向量存储是否存在
    if vector_store.vector_store_exists():
        print("✅ 检测到现有向量存储")
        
        # 检查是否有新/更新文档
        pdf_dir = "./data/raw_pdfs"
        pdf_processor = PDFProcessor()
        all_documents = pdf_processor.load_pdfs_from_directory(pdf_dir) if os.path.exists(pdf_dir) else []
        
        if all_documents:
            chunks = pdf_processor.split_documents(all_documents)
            new_docs = vector_store.filter_new_and_updated_documents(chunks) if chunks else []
            
            if new_docs:
                print(f"🆕 发现 {len(new_docs)} 个新/更新文档，开始智能更新向量库...")
                success = vector_store.smart_create_vector_store(chunks)
                if success:
                    print("✅ 向量库智能更新完成！")
                else:
                    print("❌ 向量库更新失败，将使用原有向量库进行检索")
            else:
                print("✅ 无新/更新文档，直接使用现有向量库")
        else:
            print("⚠️  未找到PDF文档，直接使用现有向量库")
    else:
        print("🆕 未找到向量存储，开始首次创建...")
        success = smart_processing_mode()
        if not success:
            print("❌ 首次创建向量库失败，终止自动模式")
            return False
    
    # 进入搜索模式
    print("\n🚀 进入检索模式...")
    return search_only_mode(vector_store)

def main():
    """主函数：入口菜单"""
    print("=" * 50)
    print("📚 PDF知识库智能向量化处理系统")
    print("=" * 50)

    # 显示向量数据库路径
    test_store = SmartVectorStore()
    print(f"🔍 向量数据库存储路径: {os.path.abspath(test_store.persist_directory)}")
    print(f"📁 PDF源文件目录: {os.path.abspath('./data/raw_pdfs')}")
    print(f"📜 日志文件路径: {os.path.abspath('pdf_vectordb.log')}")

    # 菜单选项
    print("\n请选择运行模式：")
    print("1. 🤖 自动模式（推荐：检测更新+创建/加载向量库+检索）")
    print("2. 🔄 智能处理模式（仅处理新/更新PDF，不进入检索）")
    print("3. 🔍 只检索模式（最快，直接使用现有向量库）")
    print("4. 💥 强制重新处理（覆盖原有向量库，重新处理所有PDF）")
    print("5. ❌ 退出系统")
    
    while True:
        choice = input("\n请输入模式编号 (1/2/3/4/5): ").strip()
        
        if choice == "1":
            auto_mode()
            break
        elif choice == "2":
            if smart_processing_mode():
                print("\n是否立即进入检索模式？(y/n)")
                if input().strip().lower() in ['y', 'yes', '是']:
                    search_only_mode()
            break
        elif choice == "3":
            search_only_mode()
            break
        elif choice == "4":
            print("⚠️  警告：该操作将删除原有向量库并重新处理所有PDF，是否继续？(y/n)")
            if input().strip().lower() in ['y', 'yes', '是']:
                if force_reprocess_all():
                    print("\n是否立即进入检索模式？(y/n)")
                    if input().strip().lower() in ['y', 'yes', '是']:
                        search_only_mode()
            else:
                print("✅ 已取消强制重新处理")
            break
        elif choice == "5":
            print("👋 感谢使用，再见！")
            break
        else:
            print("❌ 无效输入，请输入1-5之间的有效编号")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        logger.error("程序异常退出", exc_info=True)
        input("按回车键退出...")