# LangChain RAG 系统

基于 LangChain 的检索增强生成（RAG）系统，支持文档上传、向量检索和智能问答。

## 功能特性

- 📄 **文档处理**：支持 txt、pdf、docx 格式文档上传
- 🔍 **向量检索**：使用 ChromaDB 进行高效的语义搜索
- 🤖 **大模型集成**：支持本地 Ollama 模型和线上模型（DeepSeek、Qwen、OpenAI）
- 💬 **智能问答**：基于文档内容的精准问答
- 📁 **文件管理**：文件上传、列表查看、删除功能
- 📥 **文件下载**：支持下载参考来源文档

## 系统架构

```
用户提问 → 向量检索 → 文档片段 → 大模型生成 → 返回答案
```

### 核心模块

- `core/document_processor.py` - 文档加载和切分
- `core/vector_store.py` - 向量数据库管理
- `core/rag_engine.py` - RAG 引擎核心逻辑
- `backend/main.py` - FastAPI 后端服务
- `frontend/` - 前端页面（用户端和管理端）

## 环境要求

- Python 3.8+
- Ollama（可选，用于本地模型）
- 至少 4GB 内存

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd langChainRag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件（参考 `.env.example`）：

```env
# 服务端口
APP_PORT=8080

# 大模型配置
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=qwen2.5:7b-instruct

# 向量数据库配置
CHROMA_DB_PATH=./chroma_db
OLLAMA_API_BASE=http://localhost:11434

# 文档处理配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
TEMPERATURE=0.7
```

### 4. 启动 Ollama（使用本地模型）

```bash
# 安装 Ollama
# 访问 https://ollama.ai 下载安装

# 拉取模型
ollama pull qwen2.5:7b-instruct

# 启动 Ollama 服务
ollama serve
```

### 5. 启动后端服务

```bash
cd backend
python main.py
```

服务将在 `http://localhost:8080` 启动

### 6. 访问前端页面

- **用户端**：http://localhost:8080/
- **管理端**：http://localhost:8080/admin/

## 使用说明

### 文件上传

1. 访问管理端：http://localhost:8080/admin/
2. 在"文件上传"标签页上传文档（支持 txt、pdf、docx）
3. 文档会自动切分并存储到向量数据库

### 系统配置

在管理端的"系统配置"标签页可以配置：

- **Top K**：检索返回的文档数量
- **Temperature**：大模型温度参数
- **API 地址**：大模型 API 地址
- **API 密钥**：大模型 API 密钥
- **模型名称**：大模型名称

### 文件管理

在管理端的"文件管理"标签页可以：

- 查看已上传的文件列表
- 删除文件（同时删除向量数据库中的数据）

### 智能问答

1. 访问用户端：http://localhost:8080/
2. 在输入框中输入问题
3. 系统会基于上传的文档生成回答
4. 显示参考来源，可点击下载原文档

## 支持的模型

### 本地模型（Ollama）

- qwen2.5:7b-instruct
- llama3.2
- mistral
- 其他 Ollama 支持的模型

### 线上模型

- **DeepSeek**：deepseek-chat, deepseek-r1-chat
- **Qwen**：qwen-turbo, qwen-plus, qwen-max
- **OpenAI**：gpt-4o, gpt-4, gpt-3.5-turbo

## API 接口

### 用户端接口

- `POST /api/query` - 智能问答
- `GET /api/config` - 获取配置
- `POST /api/config` - 更新配置
- `GET /api/models` - 获取支持的模型列表

### 管理端接口

- `POST /api/upload` - 上传文件
- `GET /api/uploaded-files` - 获取文件列表
- `DELETE /api/files/{filename}` - 删除文件
- `GET /api/files/{filename}` - 下载文件

## 项目结构

```
langChainRag/
├── backend/
│   └── main.py              # FastAPI 后端服务
├── core/
│   ├── __init__.py
│   ├── document_processor.py # 文档处理器
│   ├── rag_engine.py        # RAG 引擎
│   └── vector_store.py      # 向量存储管理
├── frontend/
│   ├── admin/
│   │   └── index.html       # 管理端页面
│   └── user/
│       └── index.html       # 用户端页面
├── uploads/                 # 上传文件存储
├── chroma_db/              # 向量数据库
├── .env                    # 环境变量配置
├── .gitignore
├── requirements.txt        # Python 依赖
└── README.md
```

## 常见问题

### 1. 连接 Ollama 失败

确保 Ollama 服务已启动：

```bash
ollama serve
```

### 2. 文件上传后查询不到

检查文件是否成功上传到 `uploads/` 目录，并确认向量数据库中是否有数据。

### 3. 大模型调用失败

- 检查 API 地址是否正确
- 检查 API 密钥是否有效
- 检查模型名称是否正确

## 开发说明

### 调用流程

**文件上传流程（UPLOAD_FLOW）：**
```
【UPLOAD_FLOW-1】main.py upload_file() 接收文件
    ↓
【UPLOAD_FLOW-2】DocumentProcessor.process_file() 处理文件
    ↓
【UPLOAD_FLOW-3】load_document() 加载文档
    ↓
【UPLOAD_FLOW-4】split_documents() 切分文档
    ↓
【UPLOAD_FLOW-5】vector_store_manager.add_documents() 添加到向量库
```

**用户查询流程（QUERY_FLOW）：**
```
【QUERY_FLOW-1】main.py query() 接收查询
    ↓
【QUERY_FLOW-2】RAGEngine.query() 处理查询
    ↓
【QUERY_FLOW-3】get_vector_store() 获取向量存储
    ↓
【QUERY_FLOW-4】similarity_search() 相似度搜索
    ↓
【QUERY_FLOW-5】format_docs() 格式化文档
    ↓
【QUERY_FLOW-6】requests.post() 调用大模型
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
