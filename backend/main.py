# ========== 标准库 ==========
import os       # 操作系统模块，用于读取环境变量、处理文件路径
import sys      # 系统模块，用于操作Python运行环境和导入路径
import shutil   # 文件操作模块，用于复制文件
import logging  # 日志模块，用于记录运行日志
import httpx    # HTTP客户端库，用于调用Ollama API获取模型列表
import requests  # HTTP客户端库，用于调用大模型API

# 将项目根目录添加到Python路径，确保可以导入core模块
# 这样可以让backend/main.py找到core/目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== FastAPI框架 ==========
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks  # FastAPI核心组件
from fastapi.responses import RedirectResponse  # 重定向响应
# - FastAPI: 创建Web应用的主类
# - File/UploadFile: 处理文件上传
# - HTTPException: 抛出HTTP错误
# - BackgroundTasks: 后台任务（目前未使用）

from fastapi.staticfiles import StaticFiles  # 静态文件服务，用于托管前端HTML/CSS/JS文件
from fastapi.middleware.cors import CORSMiddleware  # CORS中间件，允许前端跨域访问API

# ========== 数据验证 ==========
from pydantic import BaseModel  # Pydantic数据验证库，用于定义API请求/响应的数据模型

# ========== 环境配置 ==========
from dotenv import load_dotenv  # dotenv库，用于从.env文件加载环境变量配置

# ========== OpenAI接口 ==========
import openai  # OpenAI官方库，用于调用大模型API（兼容Ollama等本地模型）

# ========== 项目内部模块 ==========
# 从core包导入核心组件
from core import vector_store_manager, DocumentProcessor, RAGEngine
# - vector_store_manager: 向量数据库管理器单例，负责文档的向量存储和检索
# - DocumentProcessor: 文档处理器，负责加载和切分文档
# - RAGEngine: RAG引擎，负责接收用户问题并返回RAG增强后的回答

# 加载.env配置文件
load_dotenv()


# #################################################################################
# #                           程序入口说明                                        #
# #################################################################################
#
# 【启动方式】
#   1. 命令行启动: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
#   2. 直接运行: python backend/main.py
#
# 【应用启动流程】
#
#   python backend/main.py
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  第36行: load_dotenv()                                      │
#   │  作用: 从 .env 文件加载环境变量到 os.environ                  │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  第46行: app = FastAPI(...)                                │
#   │  作用: 创建FastAPI应用实例，注册中间件                       │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  第63-64行: document_processor = DocumentProcessor()         │
#   │             rag_engine = RAGEngine()                        │
#   │  作用: 初始化文档处理器和RAG引擎单例                         │
#   │        （这两个对象会在首次API调用时真正初始化）               │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  第200-203行: uvicorn.run(app, ...)                        │
#   │  作用: 启动Web服务器，监听 0.0.0.0:8000                      │
#   └─────────────────────────────────────────────────────────────┘
#
#
# #################################################################################
# #                           API调用流程说明                                      #
# #################################################################################
#
# 【1. 文件上传流程】POST /api/upload
#
#   前端上传文件
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  upload_file(file)                                          │
#   │  - 接收上传的文件                                            │
#   │  - 验证文件格式 (.txt/.pdf/.docx)                           │
#   │  - 保存文件到 uploads/ 目录                                  │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  document_processor.process_file(file_path)                 │
#   │  【调用的类/方法链】                                         │
#   │     DocumentProcessor.load_document()                        │
#   │        │                                                    │
#   │        ├── TextLoader.load()      (txt文件)                  │
#   │        ├── PyPDFLoader.load()     (pdf文件)                  │
#   │        └── Docx2txtLoader.load()  (docx文件)                  │
#   │        │                                                    │
#   │        ▼                                                    │
#   │     DocumentProcessor.split_documents()                      │
#   │        │                                                    │
#   │        └── RecursiveCharacterTextSplitter.split_documents()   │
#   │             将长文档切分成500字符的片段                       │
#   │        │                                                    │
#   │        ▼                                                    │
#   │     返回: List[Document] 包含所有切片                        │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  vector_store_manager.add_documents(documents)               │
#   │  【调用的类/方法链】                                         │
#   │     VectorStoreManager.get_vector_store()                    │
#   │        │                                                    │
#   │        ├── chromadb.PersistentClient.get_collection()        │
#   │        │    创建或获取 ChromaDB collection                  │
#   │        ▼                                                    │
#   │     Chroma.add_documents()                                   │
#   │        │                                                    │
#   │        ├── SentenceTransformerEmbeddings.embed_documents()  │
#   │        │    将每个文本片段转换为384维向量                     │
#   │        ▼                                                    │
#   │        └── chromadb.Client.upsert()                         │
#   │             存储: (id, vector, document, metadata)           │
#   └─────────────────────────────────────────────────────────────┘
#
#
# 【2. 用户查询流程】POST /api/query
#
#   前端发送问题
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  query_rag(request)                                        │
#   │  - 验证问题不为空                                            │
#   │  - 调用 rag_engine.query(question)                          │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  rag_engine.query(question)                                 │
#   │  【调用的类/方法链】                                         │
#   │     VectorStoreManager.get_vector_store().as_retriever()   │
#   │        │                                                    │
#   │        ▼                                                    │
#   │     retriever.get_relevant_documents(question)              │
#   │        │                                                    │
#   │        ├── SentenceTransformerEmbeddings.embed_query()      │
#   │        │    将用户问题转换为384维向量                         │
#   │        ▼                                                    │
#   │        └── Chroma.similarity_search()                       │
#   │             在向量数据库中找到最相似的Top-K个片段            │
#   │        │                                                    │
#   │        ▼                                                    │
#   │     返回: List[Document] 最相关的文档片段                     │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  RAGEngine.format_docs(docs)                                │
#   │  - 将多个Document片段用 "\n\n" 连接成上下文字符串            │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  PromptTemplate.format(context=..., question=...)          │
#   │  - 构建完整提示词:                                           │
#   │    "基于以下上下文信息回答问题：\n\n{context}\n\n问题：{question}"│
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  openai.ChatCompletion.create()                             │
#   │  【调用外部服务】                                            │
#   │  - 向配置的LLM API发送请求                                   │
#   │  - 模型: self.llm_model_name (如 llama3.2)                  │
#   │  - 地址: self.llm_api_base (如 http://localhost:11434/v1)  │
#   │  返回: 大模型生成的回答                                      │
#   └─────────────────────────────────────────────────────────────┘
#          │
#          ▼
#   返回给前端: {answer, sources, success}
#
#
# 【3. 模型测试流程】POST /api/test-model
#
#   后台配置页面点击"测试模型连接"
#          │
#          ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  test_model(request)                                        │
#   │  - 保存当前openai配置                                        │
#   │  - 设置为请求中的新配置                                      │
#   │  - 发送测试问题给大模型                                      │
#   │  - 恢复原配置                                               │
#   │  - 返回测试结果                                             │
#   └─────────────────────────────────────────────────────────────┘
#
#
# #################################################################################
# #                           类关系图                                            #
# #################################################################################
#
#                        ┌─────────────────┐
#                        │   main.py       │
#                        │  (FastAPI应用)  │
#                        └────────┬────────┘
#                                 │
#            ┌────────────────────┼────────────────────┐
#            │                    │                    │
#            ▼                    ▼                    ▼
#   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
#   │DocumentProcessor│  │  RAGEngine      │  │VectorStoreManager│
#   │  (文档处理)      │  │  (RAG检索生成)  │  │  (向量数据库)    │
#   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
#            │                    │                    │
#            │                    │                    │
#            ▼                    ▼                    │
#   ┌─────────────────┐  ┌─────────────────┐            │
#   │TextLoader       │  │VectorStoreManager│◄───────────┤
#   │PyPDFLoader      │  │  .get_vector_store()          │
#   │Docx2txtLoader   │  │  .retriever                  │
#   └────────┬────────┘  └────────┬────────┘            │
#            │                    │                    │
#            │                    │                    │
#            ▼                    ▼                    ▼
#   ┌─────────────────┐  ┌─────────────────────────────────┐
#   │RecursiveCharacter│  │         ChromaDB               │
#   │TextSplitter     │  │  (向量数据库)                    │
#   └────────┬────────┘  │  - 存储向量 + 文档              │
#            │          │  - 相似度搜索                   │
#            │          └────────┬────────────────────────┘
#            │                   │
#            │                   ▼
#            │          ┌─────────────────┐
#            │          │SentenceTransformer│
#            │          │Embeddings       │
#            │          │(文本→向量转换)   │
#            │          └─────────────────┘
#            │
#            ▼
#   ┌─────────────────┐
#   │  Document       │◄────────── LangChain文档对象
#   │  - page_content │            包含文本内容和元数据
#   │  - metadata     │
#   └─────────────────┘
#
#
# #################################################################################
# #                           数据流向图                                          #
# #################################################################################
#
# 【存储数据流】上传文件时
#
#   文件(.txt/.pdf/.docx)
#          │
#          ▼
#   ┌──────────────┐
#   │DocumentLoader │  读取文件内容
#   └──────┬───────┘
#          │
#          ▼
#   ┌──────────────┐
#   │   Document   │  page_content="长文本内容..."
#   └──────┬───────┘
#          │
#          ▼
#   ┌──────────────┐
#   │TextSplitter  │  切分成500字符片段
#   └──────┬───────┘
#          │
#          ▼
#   List[Document]                    List[Document]
#   ├── chunk_id=0, source="a.txt"   ├── chunk_id=0, source="b.pdf"
#   ├── chunk_id=1, source="a.txt"   └── ...
#   └── ...
#          │
#          ▼
#   ┌──────────────┐
#   │EmbeddingFunc │  SentenceTransformer
#   │ 文本→向量    │  转换
#   └──────┬───────┘
#          │
#          ▼
#   List[(id, vector[384维], document, metadata)]
#          │
#          ▼
#   ┌──────────────┐
#   │   ChromaDB   │  持久化存储
#   └──────────────┘
#
#
# 【检索数据流】用户查询时
#
#   用户问题: "什么是RAG？"
#          │
#          ▼
#   ┌──────────────┐
#   │EmbeddingFunc │  SentenceTransformer
#   │ 问题→向量    │  转换
#   └──────┬───────┘
#          │
#          ▼
#   vector[384维]
#          │
#          ▼
#   ┌──────────────┐
#   │   ChromaDB   │  相似度搜索
#   │  Top-K=3     │  余弦相似度
#   └──────┬───────┘
#          │
#          ▼
#   List[Document]  最相关的3个片段
#          │
#          ▼
#   ┌──────────────┐
#   │PromptTemplate│  构建提示词
#   │ context + ?  │
#   └──────┬───────┘
#          │
#          ▼
#   ┌──────────────┐
#   │   OpenAI     │  调用大模型
#   │   API        │  (Ollama)
#   └──────┬───────┘
#          │
#          ▼
#   回答文本 ──► 返回给前端
#
#
# #################################################################################
# #                           核心类说明                                          #
# #################################################################################
#
# 【VectorStoreManager】向量数据库管理器（单例）
#   - initialize(): 初始化ChromaDB连接和嵌入函数
#   - get_vector_store(collection): 获取向量存储
#   - add_documents(docs): 添加文档到向量库
#   - similarity_search(query, k): 向量相似性搜索
#   - get_collection_names(): 获取所有集合名
#   - delete_collection(name): 删除集合
#
# 【DocumentProcessor】文档处理器
#   - load_document(path): 加载文档
#   - split_documents(docs): 切分文档
#   - process_file(path): 完整处理流程
#   - get_supported_extensions(): 支持的格式列表
#
# 【RAGEngine】RAG引擎
#   - query(question): 处理用户问题
#   - format_docs(docs): 格式化文档列表
#   - update_config(...): 更新配置
#   - get_config(): 获取当前配置
#
# #################################################################################

# 创建FastAPI应用实例
app = FastAPI(title="RAG文档助手管理系统", version="1.0.0")

# 配置CORS中间件，允许跨域访问
# 这样前端页面就可以从不同域名/端口访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 允许所有来源的跨域请求
    allow_credentials=True,            # 允许携带凭证（cookies）
    allow_methods=["*"],               # 允许所有HTTP方法
    allow_headers=["*"],               # 允许所有HTTP头
)

# 配置上传文件存储目录
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
# 确保上传目录存在，不存在则自动创建
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化文档处理器和RAG引擎（单例，全局共享）
document_processor = DocumentProcessor()
rag_engine = RAGEngine()

# ========== 数据模型定义 ==========

class QueryRequest(BaseModel):
    """用户查询请求的数据模型"""
    question: str  # 用户的问题

class ConfigUpdateRequest(BaseModel):
    """配置更新请求的数据模型"""
    top_k: int = None              # 检索返回的文档数量
    temperature: float = None       # 大模型温度参数
    llm_api_base: str = None       # 大模型API地址
    llm_api_key: str = None        # 大模型API密钥
    llm_model_name: str = None     # 大模型名称

class UploadResponse(BaseModel):
    """文件上传响应的数据模型"""
    filename: str       # 上传后的文件名
    chunks_count: int   # 切分的片段数量
    message: str        # 响应消息

class ModelTestRequest(BaseModel):
    """模型测试请求的数据模型"""
    question: str        # 测试问题
    api_base: str       # API地址
    api_key: str        # API密钥
    model_name: str     # 模型名称

# ========== API接口定义 ==========

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    文件上传接口

    流程：
    1. 接收上传的文件
    2. 验证文件格式（仅支持.txt/.pdf/.docx）
    3. 保存文件到uploads目录
    4. 调用DocumentProcessor处理文件（加载+切片）
    5. 将切片后的文档存入向量数据库
    6. 返回处理结果

    参数:
        file: 上传的文件对象（通过FastAPI的File参数接收）

    返回:
        UploadResponse: 包含文件名、切片数量和消息
    """
    try:
        # 1. 获取文件扩展名并转小写
        ext = os.path.splitext(file.filename)[1].lower()

        # 2. 【调用入口】获取支持的文件格式列表
        #    调用 DocumentProcessor.get_supported_extensions() 方法
        supported_extensions = document_processor.get_supported_extensions()

        # 3. 验证文件格式是否支持
        if ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式。支持的格式: {', '.join(supported_extensions)}"
            )

        # 4. 构建文件保存路径
        # 使用与列表读取时一致的路径（基于BASE_DIR）
        file_path = os.path.join(BASE_DIR, UPLOAD_DIR, file.filename)

        # 5. 将上传的文件内容保存到磁盘
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 6. 【调用入口】处理文档：加载文档并切分成小片段
        #    调用 DocumentProcessor.process_file() 方法
        #    process_file()会返回切分后的Document列表
        documents = document_processor.process_file(file_path)

        # 7. 【调用入口】将文档添加到向量数据库
        #    调用 VectorStoreManager.add_documents() 方法
        #    这里会：
        #    - 使用SentenceTransformer将每个片段转换为向量
        #    - 将向量和文本内容存储到ChromaDB
        vector_store_manager.add_documents(documents)

        # 8. 返回成功响应
        return {
            "filename": file.filename,
            "chunks_count": len(documents),  # 切分成了多少个片段
            "message": "文件上传并处理成功"
        }
    except HTTPException:
        # 直接重新抛出HTTP异常（保留状态码）
        raise
    except Exception as e:
        # 其他异常返回500内部服务器错误
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """
    RAG问答接口

    流程：
    1. 接收用户问题
    2. 调用RAG引擎处理问题
    3. RAG引擎会检索相关文档并调用大模型生成回答
    4. 返回回答和参考来源

    参数:
        request: QueryRequest对象，包含用户的问题

    返回:
        包含回答内容、来源文档列表、是否成功的字典
    """
    # 验证问题不能为空
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 【调用入口】处理用户查询
    #    调用 RAGEngine.query() 方法
    #    RAG引擎会：检索向量数据库→获取相关文档→调用大模型生成回答
    result = rag_engine.query(request.question)
    return result

@app.post("/api/test-model")
async def test_model(request: ModelTestRequest):
    """
    测试模型连接接口

    用于在后台配置页面测试模型是否配置成功
    流程：
    1. 接收模型配置参数（API地址、密钥、模型名）
    2. 向模型发送一个简单的测试问题
    3. 返回模型的回答和连接状态

    参数:
        request: ModelTestRequest对象，包含测试问题和模型配置

    返回:
        包含回答内容、模型名称、是否成功的字典
    """
    try:
        # 使用requests直接调用API，兼容所有OpenAI兼容的API（包括DeepSeek、Qwen、OpenAI等）
        llm_url = f"{request.api_base}/chat/completions"
        payload = {
            "model": request.model_name,
            "messages": [
                {"role": "user", "content": request.question}
            ],
            "temperature": 0.7
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json"
        }
        if request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"
        
        # 发送请求
        response = requests.post(llm_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 返回结果
        return {
            "success": True,
            "answer": result["choices"][0]["message"]["content"],
            "model_name": request.model_name
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "error": str(e),
            "model_name": request.model_name
        }

@app.get("/api/config")
async def get_config():
    """
    获取当前RAG引擎配置接口

    返回:
        当前的所有配置参数
    """
    # 【调用入口】获取配置
    #    调用 RAGEngine.get_config() 方法
    return rag_engine.get_config()

@app.get("/api/models")
async def get_models():
    """
    获取支持的线上模型列表

    返回:
        包含各平台模型信息的字典
    """
    # 【调用入口】获取线上模型列表
    #    调用 RAGEngine.get_online_models() 方法
    return {"success": True, "models": rag_engine.get_online_models()}

@app.get("/api/ollama/models")
async def get_ollama_models():
    """
    获取本地Ollama服务器中的所有模型列表

    调用Ollama的API获取已安装的模型列表，用于后台管理页面的模型选择下拉框

    返回:
        包含模型列表的字典，每个模型包含name（模型名称）和details（模型详情）
    """
    # Ollama默认API地址
    ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    
    try:
        # 创建异步HTTP客户端
        async with httpx.AsyncClient() as client:
            # 调用Ollama的/api/tags接口获取模型列表
            response = await client.get(f"{ollama_base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                # 解析返回的JSON数据
                data = response.json()
                # 提取模型名称列表
                models = []
                for model in data.get("models", []):
                    models.append({
                        "name": model.get("name", ""),        # 模型完整名称（如 llama3.2:latest）
                        "details": model.get("details", {})    # 模型详情（包含大小、参数等信息）
                    })
                
                return {"success": True, "models": models}
            else:
                return {"success": False, "error": f"Ollama API返回错误: {response.status_code}"}
    
    except httpx.ConnectError:
        # Ollama服务未启动或无法连接
        return {"success": False, "error": "无法连接到Ollama服务，请确保Ollama已启动"}
    except Exception as e:
        # 其他未知错误
        return {"success": False, "error": f"获取模型列表失败: {str(e)}"}

@app.put("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """
    更新RAG引擎配置接口

    参数:
        request: ConfigUpdateRequest对象，包含要更新的配置参数

    返回:
        更新成功的消息和最新的配置
    """
    # 【调用入口】更新配置
    #    调用 RAGEngine.update_config() 方法
    rag_engine.update_config(
        top_k=request.top_k,
        temperature=request.temperature,
        llm_api_base=request.llm_api_base,
        llm_api_key=request.llm_api_key,
        llm_model_name=request.llm_model_name
    )
    
    # 【调用入口】获取更新后的配置
    #    调用 RAGEngine.get_config() 方法
    return {"message": "配置更新成功", "config": rag_engine.get_config()}

@app.get("/api/collections")
async def get_collections():
    """
    获取向量数据库中的所有集合（表）列表

    返回:
        包含所有集合名称列表的字典
    """
    # 【调用入口】获取集合列表
    #    调用 VectorStoreManager.get_collection_names() 方法
    return {"collections": vector_store_manager.get_collection_names()}

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    删除指定的向量集合

    参数:
        collection_name: 要删除的集合名称

    返回:
        删除成功消息
    """
    # 【调用入口】删除集合
    #    调用 VectorStoreManager.delete_collection() 方法
    success = vector_store_manager.delete_collection(collection_name)
    if success:
        return {"message": f"集合 {collection_name} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail="集合不存在")

# ========== 文件管理接口 ==========

@app.get("/api/files")
async def list_uploaded_files():
    """
    获取已上传文件列表

    返回:
        包含文件名列表的字典
    """
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    upload_path = os.path.join(BASE_DIR, upload_dir)
    
    if not os.path.exists(upload_path):
        return {"files": [], "message": "上传目录不存在"}
    
    try:
        # 获取目录中的所有文件
        files = []
        for item in os.listdir(upload_path):
            item_path = os.path.join(upload_path, item)
            if os.path.isfile(item_path):
                # 获取文件大小
                file_size = os.path.getsize(item_path)
                # 获取修改时间
                modified_time = os.path.getmtime(item_path)
                files.append({
                    "filename": item,
                    "size": file_size,
                    "size_formatted": format_file_size(file_size),
                    "modified": modified_time
                })
        
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"success": True, "files": files}
    
    except Exception as e:
        return {"success": False, "error": f"获取文件列表失败: {str(e)}"}

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """
    下载指定的已上传文件

    参数:
        filename: 要下载的文件名

    返回:
        文件内容流
    """
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    upload_path = os.path.join(BASE_DIR, upload_dir)
    file_path = os.path.join(upload_path, filename)
    
    # 安全检查：防止路径遍历攻击
    if not os.path.abspath(file_path).startswith(os.path.abspath(upload_path)):
        raise HTTPException(status_code=400, detail="非法文件路径")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="路径不是文件")
    
    # 读取文件内容
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    # 获取文件扩展名，设置合适的Content-Type
    ext = os.path.splitext(filename)[1].lower()
    content_type = get_content_type(ext)
    
    # 返回文件下载响应
    from fastapi.responses import Response
    return Response(
        content=file_content,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(file_content))
        }
    )

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """
    删除指定的已上传文件及其向量数据

    参数:
        filename: 要删除的文件名

    返回:
        删除成功消息
    """
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    upload_path = os.path.join(BASE_DIR, upload_dir)
    file_path = os.path.join(upload_path, filename)
    
    # 安全检查：防止路径遍历攻击
    if not os.path.abspath(file_path).startswith(os.path.abspath(upload_path)):
        raise HTTPException(status_code=400, detail="非法文件路径")
    
    # 1. 删除物理文件
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")
    
    # 2. 删除向量数据库中的数据
    try:
        success = vector_store_manager.delete_file_chunks(filename)
        # 如果集合不存在，视为已删除状态，不报错
        if not success:
            logging.warning(f"文件 {filename} 的向量数据不存在或已删除")
    except Exception as e:
        # 如果是集合不存在的错误，视为正常情况
        if "Collection" in str(e) and "does not exist" in str(e):
            logging.warning(f"向量数据库集合不存在，跳过向量数据删除")
        else:
            raise HTTPException(status_code=500, detail=f"删除向量数据失败: {str(e)}")
    
    return {"message": f"文件 {filename} 及其向量数据删除成功"}

@app.get("/api/uploaded-files")
async def list_uploaded_files_with_chunks():
    """
    获取已上传文件列表及其向量片段数量

    返回:
        包含文件列表的字典，每个文件包含:
            - filename: str - 文件名
            - size: int - 文件大小（字节）
            - size_formatted: str - 格式化后的大小
            - modified: float - 修改时间戳
            - chunks: int - 向量片段数量
    """
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    upload_path = os.path.join(BASE_DIR, upload_dir)
    
    if not os.path.exists(upload_path):
        return {"success": True, "files": []}
    
    try:
        # 获取向量数据库中的文件片段信息
        vector_files = vector_store_manager.list_uploaded_files()
        vector_files_dict = {f["source"]: f["chunks"] for f in vector_files}
        
        # 获取目录中的所有文件
        files = []
        for item in os.listdir(upload_path):
            item_path = os.path.join(upload_path, item)
            if os.path.isfile(item_path):
                file_size = os.path.getsize(item_path)
                modified_time = os.path.getmtime(item_path)
                files.append({
                    "filename": item,
                    "size": file_size,
                    "size_formatted": format_file_size(file_size),
                    "modified": modified_time,
                    "chunks": vector_files_dict.get(item, 0)
                })
        
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"success": True, "files": files}
    
    except Exception as e:
        return {"success": False, "error": f"获取文件列表失败: {str(e)}"}

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小为可读字符串
    
    参数:
        size_bytes: 文件大小（字节）
    
    返回:
        格式化后的文件大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def get_content_type(ext: str) -> str:
    """
    根据文件扩展名获取对应的Content-Type
    
    参数:
        ext: 文件扩展名（带点号，如 ".txt"）
    
    返回:
        Content-Type字符串
    """
    content_types = {
        ".txt": "text/plain; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pdf": "application/pdf",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    }
    return content_types.get(ext, "application/octet-stream")

# ========== 静态文件挂载 ==========

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMIN_STATIC_DIR = os.path.join(BASE_DIR, "frontend", "admin")
USER_STATIC_DIR = os.path.join(BASE_DIR, "frontend", "user")

# 打印路径调试信息
print(f"DEBUG - BASE_DIR: {BASE_DIR}")
print(f"DEBUG - ADMIN_STATIC_DIR: {ADMIN_STATIC_DIR}")
print(f"DEBUG - USER_STATIC_DIR: {USER_STATIC_DIR}")
print(f"DEBUG - ADMIN_DIR_EXISTS: {os.path.exists(ADMIN_STATIC_DIR)}")
print(f"DEBUG - USER_DIR_EXISTS: {os.path.exists(USER_STATIC_DIR)}")

# 重定向 /admin 到 /admin/（解决URL尾部斜杠问题）
@app.get("/admin")
async def admin_redirect():
    """
    将 /admin 重定向到 /admin/
    
    由于静态文件挂载需要尾部斜杠，此路由确保两种URL格式都能正常工作
    """
    return RedirectResponse(url="/admin/")

# 将前端页面挂载到/admin/路径（带斜杠）
# 使用html=True可以让目录作为HTML应用挂载，支持SPA路由
app.mount("/admin/", StaticFiles(directory=ADMIN_STATIC_DIR, html=True), name="admin")

# 将前端页面挂载到根路径
app.mount("/", StaticFiles(directory=USER_STATIC_DIR, html=True), name="user")

# ========== 应用启动入口 ==========

if __name__ == "__main__":
    import uvicorn
    # 启动FastAPI应用
    # host: 监听地址，0.0.0.0表示接受所有网络接口的连接
    # port: 监听端口，默认为8000
    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000))
    )