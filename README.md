# RAG 智能问答系统

基于本地知识库的RAG（检索增强生成）智能问答系统，利用Ollama本地大模型、LangChain框架和Streamlit开发工具，实现上传文档、构建知识库、智能问答的完整流程。

## 环境要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.9 - 3.11
- **Ollama**: 最新版（[下载地址](https://ollama.ai/download)）
- **内存**: 至少8GB（推荐16GB）

## 安装步骤

### 1. 安装Ollama并下载模型

```bash
# 下载并安装Ollama（Windows下载安装程序，Linux/macOS执行以下命令）
curl -fsSL https://ollama.ai/install.sh | sh

# 下载 DeepSeek-R1 模型（约4.7GB）
ollama pull deepseek-r1:7b

# 下载嵌入模型（约274MB）
ollama pull nomic-embed-text

# 验证模型是否安装成功
ollama list
```

### 2. 创建Python虚拟环境

```bash
# 在项目根目录下创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活虚拟环境（Linux/macOS）
source venv/bin/activate
```

### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

## 使用说明

### 一键启动（Web界面）

```bash
streamlit run app.py
```

启动后在浏览器中打开 http://localhost:8501，即可使用Web界面：
1. 进入「文档管理」标签页
2. 上传PDF、DOCX或TXT格式的文档
3. 点击「开始构建知识库」按钮
4. 切换到「问答交互」标签页
5. 输入问题并点击提问

### 命令行模式

```bash
# 1. 测试Ollama服务
python test_ollama.py

# 2. 构建知识库
python build_knowledge_base.py

# 3. 启动命令行问答
python rag_qa_cli.py
```

## 技术架构

```
用户提问 → 文档上传 → 文本分割 → 向量化 → Chroma向量库
                                            ↓
用户提问 → 向量检索 → 检索结果 → LLM生成 → 最终回答
```

### 关键技术点

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 大语言模型 | DeepSeek-R1:7b | Ollama本地部署的开源模型 |
| 嵌入模型 | nomic-embed-text | 文本向量化，768维 |
| 向量数据库 | Chroma | 本地持久化向量存储 |
| RAG框架 | LangChain | ConversationalRetrievalChain |
| Web界面 | Streamlit | 交互式可视化界面 |
| 文本分块 | RecursiveCharacterTextSplitter | chunk_size=1000, chunk_overlap=200 |
| 文档解析 | PyPDF2 + python-docx | 支持PDF/DOCX/TXT |

### RAG流程说明

1. **文档加载**: 支持PDF、DOCX、TXT格式文档的批量读取
2. **文本分割**: 使用RecursiveCharacterTextSplitter，设置chunk_size=1000，chunk_overlap=200
3. **向量化存储**: 通过nomic-embed-text模型将文本块转为向量，存入Chroma
4. **相似性检索**: 用户提问时检索最相关的3个文本块
5. **生成回答**: LLM基于检索结果生成带引用的回答

## 项目结构

```
RAG-QA-System/
├── app.py                        # Streamlit Web应用主程序
├── build_knowledge_base.py       # 知识库构建脚本
├── rag_qa_cli.py                 # 命令行RAG问答脚本
├── test_ollama.py                # Ollama连通性测试脚本
├── requirements.txt              # Python依赖清单
├── .gitignore                    # Git忽略文件配置
├── README.md                     # 项目说明文档
├── docs/                         # 示例文档目录
│   ├── nlp_introduction.txt
│   ├── word_embedding.txt
│   ├── transformer_architecture.txt
│   ├── rag_technology.txt
│   └── llm_overview.txt
├── knowledge_base/               # Chroma向量数据库存储（自动生成）
└── pyinstaller/                  # 打包配置
    └── build_app.py
```

## 项目效果截图

### 主界面
![主界面](docs/screenshots/main_interface.png)

### 文档上传与知识库构建
![知识库构建](docs/screenshots/kb_build.png)

### 问答示例
![问答示例](docs/screenshots/qa_example.png)

## 已知问题与改进方向

### 已知问题
- 首次加载模型需要一定时间
- 中文文档的分块效果有待优化
- 大文档处理时内存占用较高

### 改进方向
- [ ] 支持更多文档格式（PPT、Excel、HTML等）
- [ ] 添加夜间模式
- [ ] 批量上传与队列处理
- [ ] 问答记录导出功能
- [ ] 多轮对话的上下文压缩优化
- [ ] 文档知识库的增量更新
- [ ] 添加检索结果相关性评分显示

## 注意事项

- 使用前请确保Ollama服务已启动
- 请先下载所需模型再运行应用
- 向量数据库存储在knowledge_base目录中，如需要重建请删除该目录