"""
RAG智能问答系统 - Streamlit Web应用
功能：上传文档、构建知识库、智能问答、对话历史管理
"""
import os
import tempfile
import streamlit as st
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# 尝试导入Ollama，如果不可用则使用备用方案
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 备用嵌入模型（无需Ollama）
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ============ 配置 ============
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============ 页面配置 ============
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义CSS样式 ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-user {
        background-color: #E3F2FD;
        padding: 10px 15px;
        border-radius: 15px 15px 5px 15px;
        margin: 5px 0;
    }
    .chat-assistant {
        background-color: #F5F5F5;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 5px;
        margin: 5px 0;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] {
        padding: 10px;
        border: 2px dashed #1E88E5;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============ 初始化会话状态 ============
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

# ============ LangChain 工具函数 ============
def get_embeddings():
    """获取嵌入模型，优先使用Ollama，否则使用Chroma默认嵌入"""
    if OLLAMA_AVAILABLE:
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                return OllamaEmbeddings(
                    base_url=OLLAMA_BASE_URL,
                    model=EMBED_MODEL
                )
        except:
            pass
    # 备用：使用Chroma默认嵌入（ONNX MiniLM，本地运行）
    return None  # Chroma会自动使用默认嵌入

def get_llm():
    """获取LLM，需要Ollama运行"""
    if OLLAMA_AVAILABLE:
        return OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=0.3,
            top_k=10,
            top_p=0.8,
            num_predict=2048,
        )
    return None

def load_uploaded_files(uploaded_files):
    """加载上传的文件并返回文档列表"""
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                st.warning(f"不支持的文件格式: {uploaded_file.name}")
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            documents.extend(docs)
            st.session_state.doc_count += 1
        except Exception as e:
            st.error(f"加载文件 {uploaded_file.name} 失败: {e}")

    return documents

def build_kb_from_uploaded(documents, kb_dir):
    """从上传的文档构建知识库"""
    if not documents:
        st.warning("没有可处理的文档")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    st.session_state.chunk_count = len(chunks)

    embeddings = get_embeddings()
    if embeddings is not None:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=kb_dir,
        )
    else:
        # 使用Chroma默认嵌入（无需外部服务）
        vector_store = Chroma.from_documents(
            documents=chunks,
            persist_directory=kb_dir,
        )
    vector_store.persist()
    return vector_store

def update_qa_chain(vector_store):
    """更新问答链"""
    if vector_store is None:
        return None

    llm = get_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    if llm is None:
        return None
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

def check_kb_status():
    """检查知识库状态"""
    if os.path.exists(KB_DIR) and os.listdir(KB_DIR):
        try:
            embeddings = get_embeddings()
            if embeddings is not None:
                vs = Chroma(
                    persist_directory=KB_DIR,
                    embedding_function=embeddings,
                )
            else:
                vs = Chroma(
                    persist_directory=KB_DIR,
                )
            count = vs._collection.count()
            return vs, count
        except:
            return None, 0
    return None, 0

# ============ 侧边栏 ============
def render_sidebar():
    with st.sidebar:
        st.markdown("###   RAG智能问答系统")
        st.markdown("---")

        # 模型状态
        st.markdown("####   系统状态")
        ollama_ok = False
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            ollama_ok = r.status_code == 200
        except:
            pass

        if ollama_ok:
            st.success(" Ollama 已连接")
        else:
            st.error(" Ollama 未连接")
            st.info("请确保Ollama服务已启动")

        st.markdown(f"- **LLM模型**: `{LLM_MODEL}`")
        st.markdown(f"- **嵌入模型**: `{EMBED_MODEL}`")

        # 知识库状态
        st.markdown("---")
        st.markdown("####   知识库状态")
        vs, kb_count = check_kb_status()
        if vs and kb_count > 0:
            st.markdown(f'<div class="status-box">✅ 知识库就绪</div>', unsafe_allow_html=True)
            st.markdown(f"- **文档数量**: {st.session_state.doc_count}")
            st.markdown(f"- **文本块数量**: {kb_count}")

            if st.button("  ️ 重新加载知识库", use_container_width=True):
                with st.spinner("正在加载知识库..."):
                    vector_store = vs
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = update_qa_chain(vector_store)
                    st.session_state.kb_ready = True
                    st.session_state.messages = []
                    st.success("知识库已重新加载！")
                    st.rerun()
        else:
            st.markdown('<div class="status-box">⚠️ 知识库为空</div>', unsafe_allow_html=True)
            st.info("请上传文档并构建知识库")

        # 清除对话
        st.markdown("---")
        if st.button("  清除对话历史", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.vector_store:
                st.session_state.qa_chain = update_qa_chain(st.session_state.vector_store)
            st.success("对话历史已清除")

        st.markdown("---")
        st.caption("RAG-QA-System v1.0 | 基于LangChain + Ollama")

# ============ 主区域 - 文档上传与知识库构建 ============
def render_document_upload():
    st.markdown('<p class="main-header">   RAG 智能问答系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于本地知识库的智能问答 | 上传文档 → 构建知识库 → 开始提问</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  ️ 文档管理", "  问答交互"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#####   上传文档")
            uploaded_files = st.file_uploader(
                "支持 PDF、DOCX、TXT 格式",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="file_uploader"
            )

            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.info(f"已选择 {len(uploaded_files)} 个文件")

        with col2:
            st.markdown("#####   知识库操作")
            build_btn = st.button("   开始构建知识库", use_container_width=True, type="primary")

            if build_btn:
                if not uploaded_files and not st.session_state.uploaded_files:
                    st.warning("请先上传文档")
                else:
                    files_to_process = uploaded_files or st.session_state.uploaded_files
                    with st.spinner("正在构建知识库，请稍候..."):
                        progress_bar = st.progress(0)
                        st.session_state.doc_count = 0

                        documents = load_uploaded_files(files_to_process)
                        progress_bar.progress(50)

                        if documents:
                            vector_store = build_kb_from_uploaded(documents, KB_DIR)
                            progress_bar.progress(80)

                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.session_state.qa_chain = update_qa_chain(vector_store)
                                st.session_state.kb_ready = True
                                progress_bar.progress(100)

                                st.success(f"""
                                ✅ 知识库构建完成！
                                - 处理文档: {st.session_state.doc_count} 份
                                - 生成文本块: {st.session_state.chunk_count} 个
                                - 可以开始问答了！
                                """)
                            else:
                                st.error("知识库构建失败")
                        else:
                            st.warning("未能从上传文件中提取到文本内容")

        # 显示已上传的文件列表
        if st.session_state.uploaded_files:
            st.markdown("---")
            st.markdown(f"#####   已上传文件 ({len(st.session_state.uploaded_files)})")
            for f in st.session_state.uploaded_files:
                file_size = f.size / 1024
                st.markdown(f"- {f.name} ({file_size:.1f} KB)")

    return tab2

# ============ 主区域 - 问答交互 ============
def render_qa_interface(tab2):
    with tab2:
        st.markdown("#####   智能问答")

        # 检查知识库是否就绪
        vs, kb_count = check_kb_status()
        if kb_count == 0:
            st.warning("⚠️ 知识库为空，请先在「文档管理」标签页上传文档并构建知识库")
            return

        if st.session_state.vector_store is None:
            st.session_state.vector_store = vs
            st.session_state.qa_chain = update_qa_chain(vs)
            st.session_state.kb_ready = True
            st.rerun()

        # 显示对话历史
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user"><b>   ️ 你</b><br/>{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    with st.container():
                        st.markdown(
                            f'<div class="chat-assistant"><b>   AI 助手</b><br/>{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                        if "sources" in msg and msg["sources"]:
                            with st.expander("查看参考来源"):
                                for i, src in enumerate(msg["sources"]):
                                    st.markdown(f"**来源 {i+1}**: {src}")

        # 提问输入
        st.markdown("---")
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "请输入你的问题",
                placeholder="例如：什么是自然语言处理？...",
                label_visibility="collapsed",
                key="question_input"
            )
        with col2:
            ask_btn = st.button("  提问", type="primary", use_container_width=True)

        if ask_btn and question:
            if not st.session_state.kb_ready:
                st.warning("知识库尚未就绪，请先构建知识库")
                return

            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("正在思考中..."):
                try:
                    if st.session_state.qa_chain is not None:
                        # 完整RAG模式（需要Ollama）
                        result = st.session_state.qa_chain.invoke({"question": question})
                        answer = result["answer"]
                        source_docs = result.get("source_documents", [])
                    else:
                        # 检索模式（无需Ollama，直接返回相关文档片段）
                        source_docs = st.session_state.vector_store.similarity_search(
                            question, k=3
                        )
                        answer = "【检索模式】以下是知识库中与您问题最相关的文档片段：\n\n"
                        for i, doc in enumerate(source_docs):
                            answer += f"**相关片段 {i+1}**（来源：{doc.metadata.get('source', '未知')}）：\n{doc.page_content[:300]}...\n\n"
                        answer += "\n💡 **提示**：安装并启动Ollama服务后，可获得AI生成式回答。"

                    # 提取来源信息
                    sources = []
                    for doc in source_docs:
                        source_name = doc.metadata.get("source", "未知")
                        content_preview = doc.page_content[:100].replace("\n", " ")
                        sources.append(f"{source_name}: {content_preview}...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"问答处理出错: {e}")
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                        st.session_state.messages.pop()

        elif ask_btn and not question:
            st.warning("请输入问题")

# ============ 主函数 ============
def main():
    init_session_state()
    render_sidebar()
    tab2 = render_document_upload()
    render_qa_interface(tab2)

if __name__ == "__main__":
    main()