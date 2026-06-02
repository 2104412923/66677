"""
知识库构建模块
功能：批量读取文档、文本分割、向量化存储到Chroma数据库
"""
import os
import glob
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 配置
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "docs")
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

def get_embeddings():
    """获取Ollama嵌入模型"""
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBED_MODEL
    )

def load_documents(directory: str) -> List:
    """批量加载指定目录下的所有文档（PDF、DOCX、TXT）"""
    documents = []
    supported_extensions = {
        "*.pdf": PyPDFLoader,
        "*.docx": Docx2txtLoader,
        "*.txt": TextLoader,
    }

    for ext, loader_cls in supported_extensions.items():
        file_pattern = os.path.join(directory, "**", ext)
        for file_path in glob.glob(file_pattern, recursive=True):
            try:
                print(f"  正在加载: {os.path.basename(file_path)}")
                if ext == "*.txt":
                    loader = loader_cls(file_path, encoding="utf-8")
                else:
                    loader = loader_cls(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                    doc.metadata["file_type"] = ext.replace("*", "")
                documents.extend(docs)
                print(f"  [OK] 已加载 {len(docs)} 页/段")
            except Exception as e:
                print(f"  [ERROR] 加载失败 {file_path}: {e}")

    return documents

def split_texts(documents: List) -> List:
    """使用RecursiveCharacterTextSplitter对文本进行分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  文本分块完成: {len(documents)} 篇文档 -> {len(chunks)} 个文本块")
    return chunks

def build_vector_store(chunks: List, persist_directory: str):
    """构建并持久化Chroma向量数据库"""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    print(f"  [OK] 向量数据库已保存至: {persist_directory}")
    print(f"  [OK] 向量数据库包含 {vector_store._collection.count()} 个向量")
    return vector_store

def load_vector_store(persist_directory: str) -> Optional[Chroma]:
    """加载已存在的向量数据库"""
    if not os.path.exists(persist_directory):
        print("  [INFO] 向量数据库不存在，需要先构建")
        return None
    try:
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        count = vector_store._collection.count()
        print(f"  [OK] 已加载向量数据库，包含 {count} 个向量")
        return vector_store
    except Exception as e:
        print(f"  [ERROR] 加载向量数据库失败: {e}")
        return None

def retrieve(vector_store: Chroma, query: str, k: int = 3) -> List:
    """检索函数：给定查询，返回最相关的k个文本块"""
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

def build_knowledge_base(docs_dir: str = DOCUMENTS_DIR, kb_dir: str = KB_DIR):
    """完整的知识库构建流程"""
    print("=" * 50)
    print("开始构建知识库")
    print("=" * 50)

    print("\n[步骤1] 加载文档...")
    documents = load_documents(docs_dir)
    if not documents:
        print("[ERROR] 未找到任何文档，请将文档放入 docs/ 目录")
        return None
    print(f"  共加载 {len(documents)} 页/段文档")

    print("\n[步骤2] 文本分块...")
    chunks = split_texts(documents)

    print("\n[步骤3] 向量化并存储...")
    vector_store = build_vector_store(chunks, kb_dir)

    print("\n[步骤4] 测试检索...")
    test_queries = ["什么是自然语言处理", "词嵌入是什么", "Transformer模型"]
    for query in test_queries:
        results = retrieve(vector_store, query, k=2)
        print(f"\n  查询: '{query}'")
        for i, (doc, score) in enumerate(results):
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"    结果{i+1} (相似度:{score:.4f}): {content_preview}...")

    print("\n" + "=" * 50)
    print("知识库构建完成！")
    print("=" * 50)
    return vector_store

if __name__ == "__main__":
    build_knowledge_base()