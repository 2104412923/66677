"""
知识库构建脚本（无需Ollama）
功能：使用Chroma内置的ONNX MiniLM嵌入模型，离线构建知识库
"""
import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    documents = []
    for ext, loader_cls in [("*.txt", TextLoader), ("*.pdf", PyPDFLoader), ("*.docx", Docx2txtLoader)]:
        for fp in glob.glob(os.path.join(DOCS_DIR, "**", ext), recursive=True):
            try:
                print(f"  加载: {os.path.basename(fp)}")
                loader = loader_cls(fp, encoding="utf-8") if ext == "*.txt" else loader_cls(fp)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = os.path.basename(fp)
                documents.extend(docs)
                print(f"    → {len(docs)} 段")
            except Exception as e:
                print(f"    [ERROR] {e}")
    return documents

def build():
    print("=" * 50)
    print("知识库构建（无需Ollama）")
    print("=" * 50)

    if os.path.exists(KB_DIR):
        shutil.rmtree(KB_DIR)

    print("\n[1/3] 加载文档...")
    docs = load_documents()
    print(f"共 {len(docs)} 段文档")

    print("\n[2/3] 文本分块...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} 篇 → {len(chunks)} 个文本块")

    print("\n[3/3] 向量化并存储（使用Chroma默认嵌入）...")
    vs = Chroma.from_documents(
        documents=chunks,
        persist_directory=KB_DIR,
    )
    vs.persist()
    count = vs._collection.count()
    print(f"\n✅ 知识库构建完成！")
    print(f"   向量库路径: {KB_DIR}")
    print(f"   文本块数量: {count}")

    print("\n测试检索（无需Ollama）:")
    for q in ["什么是自然语言处理", "NLP有哪些应用场景", "Transformer模型是什么"]:
        results = vs.similarity_search(q, k=2)
        print(f"\n  查询: '{q}'")
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r.metadata['source']}: {r.page_content[:60]}...")

if __name__ == "__main__":
    build()