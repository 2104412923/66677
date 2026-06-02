"""
PyInstaller打包脚本
功能：将Streamlit应用打包为独立可执行文件
使用方法：python build_app.py
"""
import os
import sys

# 打包命令
pyinstaller_cmd = [
    "pyinstaller",
    "--onefile",
    "--name", "RAG-QA-System",
    "--add-data", f"docs{os.pathsep}docs",
    "--hidden-import", "streamlit",
    "--hidden-import", "langchain",
    "--hidden-import", "langchain_community",
    "--hidden-import", "langchain_classic",
    "--hidden-import", "langchain_ollama",
    "--hidden-import", "langchain_text_splitters",
    "--hidden-import", "chromadb",
    "--hidden-import", "pypdf2",
    "--hidden-import", "docx2txt",
    "--hidden-import", "tiktoken",
    "--hidden-import", "requests",
    "--collect-all", "streamlit",
    "--collect-all", "langchain_community",
    "--collect-all", "langchain_classic",
    "--collect-all", "langchain_ollama",
    "--collect-all", "chromadb",
    "--noconfirm",
    "--log-level", "WARN",
    "../app.py"
]

if __name__ == "__main__":
    print("=" * 60)
    print("RAG-QA-System 打包脚本")
    print("=" * 60)
    print("\n使用以下命令打包：")
    print("    cd pyinstaller")
    print("    " + " ".join(pyinstaller_cmd))
    print("\n注意：")
    print("1. 打包前请确保已安装pyinstaller: pip install pyinstaller")
    print("2. 打包后的exe文件在 dist/ 目录下")
    print("3. 运行exe需要Ollama服务已安装且模型已下载")
    print("4. 首次启动时模型加载可能需要一些时间")
    print("=" * 60)