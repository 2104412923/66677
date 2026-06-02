"""
RAG问答命令行版本
功能：加载知识库，基于检索结果进行问答
"""
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# 配置
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

def get_embeddings():
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBED_MODEL
    )

def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
        top_k=10,
        top_p=0.8,
        num_predict=2048,
    )

def load_knowledge_base():
    """加载已存在的向量数据库"""
    if not os.path.exists(KB_DIR):
        print("[ERROR] 知识库不存在，请先运行 build_knowledge_base.py 构建知识库")
        return None
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=KB_DIR,
        embedding_function=embeddings,
    )
    count = vector_store._collection.count()
    print(f"[OK] 已加载知识库，包含 {count} 个文本块\n")
    return vector_store

def create_qa_chain(vector_store):
    """创建RAG问答链"""
    llm = get_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

def ask_question(qa_chain, question: str):
    """向问答链提问"""
    try:
        result = qa_chain.invoke({"question": question})
        print(f"\n问题: {question}")
        print(f"回答: {result['answer']}")
        print(f"\n参考来源:")
        for i, doc in enumerate(result.get("source_documents", [])):
            source = doc.metadata.get("source", "未知来源")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i+1}] {source}: {preview}...")
        print("-" * 60)
        return result
    except Exception as e:
        print(f"[ERROR] 问答失败: {e}")
        return None

def interactive_qa():
    """交互式问答模式"""
    vector_store = load_knowledge_base()
    if not vector_store:
        return

    qa_chain = create_qa_chain(vector_store)

    print("=" * 60)
    print("RAG智能问答系统（命令行版）")
    print("输入 'quit' 退出，输入 'clear' 清除对话历史")
    print("=" * 60)

    while True:
        question = input("\n请输入问题: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("感谢使用，再见！")
            break
        elif question.lower() == "clear":
            qa_chain = create_qa_chain(vector_store)
            print("对话历史已清除！")
            continue
        elif not question:
            continue

        ask_question(qa_chain, question)

if __name__ == "__main__":
    interactive_qa()