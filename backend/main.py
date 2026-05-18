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
from fastapi.responses import RedirectResponse, StreamingResponse  # 重定向响应和流式响应
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

# 导入 backend 配置（支持 bge-m3 默认配置）
from backend.config import Config, config
# 打印系统配置状态（启动时显示）
config.print_config_status()

# ========== 全局常量定义 ==========

# 获取项目根目录的绝对路径
# 项目结构:
#   langChainRag/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置上传文件存储目录（相对于项目根目录）
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
# 确保上传目录存在，不存在则自动创建
UPLOAD_DIR_PATH = os.path.join(BASE_DIR, UPLOAD_DIR)
os.makedirs(UPLOAD_DIR_PATH, exist_ok=True)

# 前端静态文件目录
ADMIN_STATIC_DIR = os.path.join(BASE_DIR, "frontend", "admin")
USER_STATIC_DIR = os.path.join(BASE_DIR, "frontend", "user")

# ========== 启动方式 ==========
# 1. 命令行: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# 2. 直接运行: python backend/main.py

# ========== API调用流程 ==========
# POST /api/upload: 文件上传→DocumentProcessor处理→VectorStoreManager向量化存储
# POST /api/query: 用户问题→RAGEngine检索→大模型生成回答
# POST /api/test-model: 测试模型连接配置

# ========== 文件上传大小限制配置 ==========
# 解决大文件上传时报错 "File is not a zip file" 的问题
# DOCX文件本质是ZIP压缩包，文件被截断后会报 "File is not a zip file" 错误
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# ========== 异步任务状态管理 ==========
# 用于跟踪文件上传任务的处理状态
# 键: task_id (UUID), 值: dict {status, filename, chunks_count, error, progress}
upload_tasks = {}

# 任务状态枚举
class UploadStatus:
    PENDING = "pending"      # 等待处理
    UPLOADING = "uploading"  # 正在上传
    PROCESSING = "processing" # 正在处理文档
    STORING = "storing"      # 正在存储向量
    COMPLETED = "completed"  # 处理完成
    FAILED = "failed"        # 处理失败

# 创建FastAPI应用实例
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.requests import Request

app = FastAPI(
    title="RAG文档助手管理系统", 
    version="1.0.0"
)

# 自定义中间件：限制请求体大小
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """
    中间件：限制请求体大小

    当上传的文件超过MAX_FILE_SIZE时，返回413错误
    防止文件被截断导致 "File is not a zip file" 错误
    """
    if request.method in ["POST", "PUT"]:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_FILE_SIZE:
            return Response(
                content=f"文件大小超过限制（最大 {MAX_FILE_SIZE // (1024 * 1024)}MB）",
                status_code=413,
                media_type="text/plain"
            )
    response = await call_next(request)
    return response

# 配置CORS中间件，允许跨域访问
# 这样前端页面就可以从不同域名/端口访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 允许所有来源的跨域请求
    allow_credentials=True,            # 允许携带凭证（cookies）
    allow_methods=["*"],               # 允许所有HTTP方法
    allow_headers=["*"],               # 允许所有HTTP头
)

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
    """文件上传响应的数据模型（同步模式）"""
    filename: str       # 上传后的文件名
    chunks_count: int   # 切分的片段数量
    message: str        # 响应消息

class AsyncUploadResponse(BaseModel):
    """异步文件上传响应的数据模型"""
    task_id: str        # 任务ID，用于查询处理状态
    filename: str       # 上传的文件名
    message: str        # 响应消息

class TaskStatusResponse(BaseModel):
    """任务状态响应的数据模型"""
    task_id: str        # 任务ID
    status: str         # 任务状态
    filename: str       # 文件名
    chunks_count: int   # 切分的片段数量（完成后才有值）
    progress: int       # 处理进度 (0-100)
    error: str          # 错误信息（失败时才有值）

class ModelTestRequest(BaseModel):
    """模型测试请求的数据模型"""
    question: str        # 测试问题
    api_base: str       # API地址
    api_key: str        # API密钥
    model_name: str     # 模型名称

# ========== API接口定义 ==========

# 【调用者】FastAPI后台任务（BackgroundTasks）通过start_background_task调用
# 【功能】后台任务处理上传的文件，包括文档解析、向量化存储
# 【参数】task_id: 任务ID（UUID）；file_path: 文件保存路径；filename: 原始文件名
async def process_uploaded_file(task_id: str, file_path: str, filename: str):
    global upload_tasks
    try:
        print(f"【后台任务】开始执行，task_id: {task_id}")
        upload_tasks[task_id]["status"] = UploadStatus.PROCESSING  # 【调用】UploadStatus.PROCESSING
        upload_tasks[task_id]["progress"] = 30
        print(f"【后台任务】状态更新: PROCESSING, 进度: 30%")
        print(f"【后台任务】任务ID: {task_id}")
        print(f"【后台任务】文件路径: {file_path}")

        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"【后台任务】文件存在，大小: {file_size} bytes")
            with open(file_path, 'rb') as f:
                header = f.read(8)
                print(f"【后台任务】文件头前8字节: {header.hex()}")
                if header[:2] == b'PK':
                    print(f"【后台任务】文件类型: ZIP格式（DOCX）")
                else:
                    print(f"【后台任务】警告：不是ZIP格式！文件头: {header}")
                if file_size < 100:
                    print(f"【后台任务】警告：文件太小 ({file_size} bytes)，可能不完整")
        else:
            print(f"【后台任务】错误：文件不存在: {file_path}")
            raise RuntimeError(f"文件不存在: {file_path}")

        # 【调用】document_processor.process_file()
        # 【功能】加载文档并切分成小片段，返回List[Document]
        print(f"【后台任务】开始调用 document_processor.process_file()")
        documents = document_processor.process_file(file_path)

        # 【更新】任务状态为"正在存储向量"
        upload_tasks[task_id]["status"] = UploadStatus.STORING
        upload_tasks[task_id]["progress"] = 70
        print(f"【后台任务】状态更新: STORING, 进度: 70%")

        # 【回调函数】progress_callback - 向量化进度回调
        def progress_callback(current, total, status):
            if total > 0:
                progress = 70 + int((current / total) * 25)  # 70% - 95%
                upload_tasks[task_id]["progress"] = progress
                upload_tasks[task_id]["status"] = UploadStatus.STORING
                print(f"【后台任务】向量化进度: {current}/{total} ({progress}%)")

        # 【获取配置】rag_engine.get_config()
        # 【功能】获取当前RAG引擎配置，用于向量化时的模型参数
        config = rag_engine.get_config()
        llm_api_base = config.get("llm_api_base")
        llm_api_key = config.get("llm_api_key", "").replace("******", "") if config.get("llm_api_key") else ""
        llm_model_name = config.get("llm_model_name")

        print(f"【后台任务】开始向量化，文档数量: {len(documents)}")
        print(f"【后台任务】大模型配置 - API Base: {llm_api_base}, Model: {llm_model_name}")

        # 【调用】vector_store_manager.add_documents()
        # 【功能】将文档向量化并存储到ChromaDB
        # 【强制使用本地Ollama】因为线上API如DeepSeek不支持embedding
        vector_store_manager.add_documents(documents, progress_callback=progress_callback,
                                          llm_api_base=llm_api_base,
                                          llm_api_key=llm_api_key,
                                          llm_model_name=llm_model_name,
                                          use_local_ollama=True)
        print(f"【后台任务】向量化完成")

        # 【更新】任务状态为"处理完成"
        upload_tasks[task_id]["status"] = UploadStatus.COMPLETED
        upload_tasks[task_id]["progress"] = 100
        upload_tasks[task_id]["chunks_count"] = len(documents)

    except Exception as e:
        # 【更新】任务状态为"处理失败"
        upload_tasks[task_id]["status"] = UploadStatus.FAILED
        upload_tasks[task_id]["error"] = str(e)
        logging.error(f"文件处理失败 {filename}: {str(e)}")

import asyncio
import json

# 【调用者】upload_file() - StreamingResponse生成器
# 【功能】生成上传进度事件流（SSE），实时推送处理进度给前端
# 【参数】file: 上传的FastAPI UploadFile对象
# 【返回】SSE格式的进度数据，包含progress、status、message字段
async def generate_upload_progress(file: UploadFile):
    try:
        # 1. 获取文件扩展名并转小写
        ext = os.path.splitext(file.filename)[1].lower()

        # 【SSE进度】10% - 开始验证文件格式
        yield f"data: {json.dumps({'progress': 10, 'status': '开始处理', 'message': '正在验证文件格式...'})}\n\n"
        await asyncio.sleep(0.1)

        # 【调用】document_processor.get_supported_extensions()
        # 【功能】获取支持的文件格式列表（.txt/.pdf/.docx/.doc）
        supported_extensions = document_processor.get_supported_extensions()

        # 【验证】检查文件格式是否支持
        if ext not in supported_extensions:
            yield f"data: {json.dumps({'progress': 0, 'status': 'failed', 'message': f'不支持的文件格式。支持的格式: {', '.join(supported_extensions)}'})}\n\n"
            return

        # 4. 构建文件保存路径
        file_path = os.path.join(UPLOAD_DIR_PATH, file.filename)

        # 【处理】先删除旧文件，确保文件写入完整性
        if os.path.exists(file_path):
            os.remove(file_path)

        # 【SSE进度】15% - 开始保存文件
        yield f"data: {json.dumps({'progress': 15, 'status': 'uploading', 'message': '正在保存文件...'})}\n\n"
        await asyncio.sleep(0.1)

        # 【流式写入】每次读取8KB，避免一次性读取大文件到内存
        file_size = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)  # 每次读取8KB
                if not chunk:
                    break
                buffer.write(chunk)
                file_size += len(chunk)

        # 【验证】检查文件写入完整性
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            if actual_size != file_size:
                yield f"data: {json.dumps({'progress': 0, 'status': 'failed', 'message': f'文件保存不完整'})}\n\n"
                return

        # 【SSE进度】30% - 文件保存完成
        yield f"data: {json.dumps({'progress': 30, 'status': 'processing', 'message': '文件保存完成，正在解析文档...'})}\n\n"
        await asyncio.sleep(0.1)

        # 【调用】document_processor.process_file()
        # 【功能】加载文档并切分成小片段，返回List[Document]
        documents = document_processor.process_file(file_path)
        chunks_count = len(documents)

        # 【SSE进度】50% - 文档解析完成
        yield f"data: {json.dumps({'progress': 50, 'status': 'processing', 'message': f'文档解析完成，共 {chunks_count} 个片段'})}\n\n"
        await asyncio.sleep(0.1)

        # 【SSE进度】60% - 开始向量化
        yield f"data: {json.dumps({'progress': 60, 'status': 'storing', 'message': '开始向量化...'})}\n\n"
        await asyncio.sleep(0.1)

        # 【计算】分批向量化，每批数量根据文档总数动态调整
        total_docs = len(documents)
        batch_size = max(1, min(5, total_docs // 10))

        # 【循环调用】vector_store_manager.add_documents()
        # 【功能】分批将文档向量化并存储到ChromaDB
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            vector_store_manager.add_documents(batch, use_local_ollama=True)

            # 【计算进度】60% - 95%
            current = min(i + batch_size, total_docs)
            progress = 60 + int((current / total_docs) * 35)
            yield f"data: {json.dumps({'progress': progress, 'status': 'storing', 'message': f'正在向量化... {current}/{total_docs}'})}\n\n"
            await asyncio.sleep(0.05)

        # 【SSE进度】100% - 处理完成
        yield f"data: {json.dumps({'progress': 100, 'status': 'completed', 'message': f'处理完成，切分了 {chunks_count} 个片段', 'chunks_count': chunks_count, 'filename': file.filename})}\n\n"

    except Exception as e:
        logging.error(f"【上传失败】错误: {str(e)}")
        yield f"data: {json.dumps({'progress': 0, 'status': 'failed', 'message': str(e)})}\n\n"

# 【调用者】前端上传组件（XMLHttpRequest/Fetch）
# 【功能】文件上传接口，通过SSE实时推送处理进度
# 【参数】file: 上传的FastAPI UploadFile对象（multipart/form-data）
# 【返回】StreamingResponse（SSE事件流，包含progress/status/message）
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # 【调用】generate_upload_progress()生成器
    # 【功能】返回SSE格式的上传进度事件流
    return StreamingResponse(
        generate_upload_progress(file),
        media_type="text/event-stream"
    )

# 【调用者】前端轮询任务状态（setInterval）
# 【功能】查询上传任务状态接口，用于获取异步处理进度
# 【参数】task_id: 任务ID（UUID）
# 【返回】TaskStatusResponse: {task_id, status, filename, chunks_count, progress, error}
@app.get("/api/upload/task/{task_id}", response_model=TaskStatusResponse)
async def get_upload_task_status(task_id: str):
    # 【访问】upload_tasks字典（内存中）
    task = upload_tasks.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return task
# 【调用者】前端用户界面（用户提交问题时）、第三方系统集成
# 【功能】RAG问答接口，接收用户问题并返回增强回答
# 【参数】request: QueryRequest对象，包含用户问题
# 【返回】{answer: 回答内容, sources: 来源文档列表, success: 是否成功}
@app.post("/api/query")
async def query_rag(request: QueryRequest):
    # 【验证】检查问题不能为空
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 【调用】rag_engine.query()
    # 【功能】检索向量数据库→获取相关文档→调用大模型生成回答
    # 【流程】VectorStoreManager.get_vector_store()→Chroma.similarity_search()→RAGEngine.format_docs()→LLM API
    result = rag_engine.query(request.question)
    return result

# 【调用者】管理员后台配置页面（测试大模型连接）
# 【功能】测试模型连接，验证API地址、密钥和模型名称是否有效
# 【参数】request: {question: 测试问题, api_base: API地址, api_key: API密钥, model_name: 模型名称}
# 【返回】{success: 是否成功, answer: 模型回答, model_name: 模型名称, error: 错误信息}
@app.post("/api/test-model")
async def test_model(request: ModelTestRequest):
    try:
        # 【构建URL】兼容所有OpenAI兼容API（DeepSeek/Qwen/OpenAI等）
        llm_url = f"{request.api_base}/chat/completions"
        payload = {
            "model": request.model_name,
            "messages": [{"role": "user", "content": request.question}],
            "temperature": 0.7
        }

        # 【设置请求头】API密钥通过Authorization头传递
        headers = {"Content-Type": "application/json"}
        if request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"

        # 【调用】requests.post()
        # 【功能】向大模型API发送测试请求
        response = requests.post(llm_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        # 【返回】成功响应
        return {
            "success": True,
            "answer": result["choices"][0]["message"]["content"],
            "model_name": request.model_name
        }
    except Exception as e:
        # 【返回】失败响应
        return {
            "success": False,
            "answer": "",
            "error": str(e),
            "model_name": request.model_name
        }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    处理favicon.ico请求，避免404错误
    """
    return Response(content="", media_type="image/x-icon")

# 【调用者】前端用户界面（初始化时获取配置）、管理员后台配置页面（显示当前配置）
# 【功能】获取当前RAG引擎配置接口
# 【返回】{llm_api_base, llm_api_key, llm_model_name, temperature, top_k, use_local_ollama}
@app.get("/api/config")
async def get_config():
    # 【调用】rag_engine.get_config()
    return rag_engine.get_config()

# 【调用者】管理员后台配置页面（模型选择下拉框）
# 【功能】获取支持的线上模型列表
# 【返回】{success: True, models: [{name, provider, description}, ...]}
@app.get("/api/models")
async def get_models():
    # 【调用】rag_engine.get_online_models()
    return {"success": True, "models": rag_engine.get_online_models()}

# 【调用者】管理员后台配置页面（本地模型选择下拉框）
# 【功能】获取本地Ollama服务器中的所有模型列表
# 【外部API】GET http://localhost:11434/api/tags
# 【返回】{success: True/False, models: [{name, details}, ...], error: 错误信息}
@app.get("/api/ollama/models")
async def get_ollama_models():
    # 【配置】Ollama默认API地址（可通过环境变量OLLAMA_API_BASE覆盖）
    ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    try:
        # 【调用】httpx.AsyncClient.get()
        # 【功能】异步调用Ollama API获取已安装的模型列表
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_base_url}/api/tags", timeout=10)

            if response.status_code == 200:
                data = response.json()
                # 【解析】提取模型名称和详情
                models = []
                for model in data.get("models", []):
                    models.append({
                        "name": model.get("name", ""),
                        "details": model.get("details", {})
                    })
                return {"success": True, "models": models}
            else:
                return {"success": False, "error": f"Ollama API返回错误: {response.status_code}"}

    except httpx.ConnectError:
        # 【异常处理】Ollama服务未启动或无法连接
        return {"success": False, "error": "无法连接到Ollama服务，请确保Ollama已启动"}
    except Exception as e:
        # 其他未知错误
        return {"success": False, "error": f"获取模型列表失败: {str(e)}"}

# 【调用者】管理员后台配置页面（保存配置时）
# 【功能】更新RAG引擎配置接口，修改大模型API地址、密钥、模型名称等参数
# 【参数】request: {top_k, temperature, llm_api_base, llm_api_key, llm_model_name}
# 【返回】{message: 成功消息, config: 更新后的完整配置}
@app.put("/api/config")
async def update_config(request: ConfigUpdateRequest):
    # 【调用】rag_engine.update_config()
    # 【功能】更新RAG引擎配置（大模型API地址、密钥、模型名称、温度参数、top_k）
    rag_engine.update_config(
        top_k=request.top_k,
        temperature=request.temperature,
        llm_api_base=request.llm_api_base,
        llm_api_key=request.llm_api_key,
        llm_model_name=request.llm_model_name
    )

    # 【调用】rag_engine.get_config()
    # 【功能】获取更新后的配置并返回给前端
    return {"message": "配置更新成功", "config": rag_engine.get_config()}

# 【调用者】管理员后台页面（管理向量数据）
# 【功能】获取向量数据库中的所有集合（表）列表
# 【返回】{collections: ["documents", "knowledge_base", ...]}
@app.get("/api/collections")
async def get_collections():
    # 【调用】vector_store_manager.get_collection_names()
    return {"collections": vector_store_manager.get_collection_names()}

# 【调用者】管理员后台页面（清理向量数据）
# 【功能】删除指定的向量集合
# 【参数】collection_name: 要删除的集合名称
# 【返回】{message: 成功消息}
@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    # 【调用】vector_store_manager.delete_collection()
    # 【功能】删除ChromaDB中的指定集合
    success = vector_store_manager.delete_collection(collection_name)
    if success:
        return {"message": f"集合 {collection_name} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail="集合不存在")

# ========== 文件管理接口 ==========

# 【调用者】前端用户界面（显示已上传文件）、管理员后台页面（文件管理）
# 【功能】获取已上传文件列表
# 【返回】{files: [{name, size, upload_date}, ...], message: 状态消息}
@app.get("/api/files")
async def list_uploaded_files():
    upload_path = UPLOAD_DIR_PATH

    # 【检查】上传目录是否存在
    if not os.path.exists(upload_path):
        return {"files": [], "message": "上传目录不存在"}

    try:
        # 【遍历】os.listdir() - 遍历上传目录收集文件信息
        files = []
        for item in os.listdir(upload_path):
            item_path = os.path.join(upload_path, item)

            # 【过滤】只处理文件，跳过子目录
            if os.path.isfile(item_path):
                # 获取文件大小（单位：字节）
                file_size = os.path.getsize(item_path)
                # 获取文件最后修改时间（Unix时间戳，从1970年1月1日至今的秒数）
                modified_time = os.path.getmtime(item_path)

                # 将文件信息添加到列表中
                files.append({
                    "filename": item,                    # 文件名（不含路径）
                    "size": file_size,                   # 文件大小（原始字节数）
                    "size_formatted": format_file_size(file_size),  # 格式化后的可读大小（如 "2.35 MB"）
                    "modified": modified_time            # 最后修改时间戳
                })

        # ========== 按修改时间排序 ==========
        # 使用lambda表达式按modified字段降序排列，最新修改的文件排在最前面
        files.sort(key=lambda x: x["modified"], reverse=True)

        # ========== 返回结果 ==========
        return {"success": True, "files": files}

    except Exception as e:
        # 捕获所有异常，返回错误信息
        return {"success": False, "error": f"获取文件列表失败: {str(e)}"}

@app.get("/api/files/{filename}")
async def download_file(filename: str):
    """
    下载指定的已上传文件

参数:
        filename: 要下载的文件名

返回:
        文件内容流

调用者:
        - 前端用户界面（下载已上传的文件）
        - 管理员后台页面（文件管理）

被调用的内部方法:
        - get_content_type() - 根据扩展名获取Content-Type
    """
    upload_path = UPLOAD_DIR_PATH
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
    import urllib.parse
    encoded_filename = urllib.parse.quote(filename)
    return Response(
        content=file_content,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={encoded_filename}; filename*=UTF-8''{encoded_filename}",
            "Content-Length": str(len(file_content))
        }
    )

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """
    删除指定的已上传文件及其向量数据
    """
    try:
        upload_path = UPLOAD_DIR_PATH
        file_path = os.path.join(upload_path, filename)

        # 安全检查：防止路径遍历攻击
        if not os.path.abspath(file_path).startswith(os.path.abspath(upload_path)):
            logging.error(f"【删除】非法文件路径: {filename}")
            raise HTTPException(status_code=400, detail="非法文件路径")

        logging.info(f"【删除】开始删除文件: {filename}")

        # 1. 删除物理文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"【删除】物理文件删除成功: {filename}")
            except Exception as e:
                logging.error(f"【删除】物理文件删除失败: {filename}, 错误: {str(e)}")
                raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")
        else:
            logging.warning(f"【删除】物理文件不存在，跳过: {filename}")

        # 2. 删除向量数据库中的数据
        try:
            success, deleted = vector_store_manager.delete_file_chunks(filename)
            if success and deleted:
                logging.info(f"【删除】向量数据删除成功: {filename}")
            elif success and not deleted:
                logging.info(f"【删除】向量数据不存在，跳过: {filename}")
            else:
                logging.warning(f"【删除】向量数据删除失败: {filename}")
        except Exception as e:
            error_str = str(e)
            # 如果是集合不存在的错误，视为正常情况
            if "Collection" in error_str and "does not exist" in error_str:
                logging.warning(f"【删除】向量数据库集合不存在，跳过向量数据删除")
            else:
                logging.error(f"【删除】向量数据删除失败: {filename}, 错误: {error_str}")
                raise HTTPException(status_code=500, detail=f"删除向量数据失败: {error_str}")

        logging.info(f"【删除】删除完成: {filename}")
        return {"success": True, "message": f"文件 {filename} 及其向量数据删除成功"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"【删除】未处理异常: {filename}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

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

调用者:
        - 前端用户界面（显示文件和向量片段信息）
        - 管理员后台页面（文件管理和统计）

被调用的内部方法:
        - VectorStoreManager.list_uploaded_files() - 获取向量数据库中的文件信息
        - format_file_size() - 格式化文件大小
    """
    upload_path = UPLOAD_DIR_PATH

    if not os.path.exists(upload_path):
        return {"success": True, "files": []}

    try:
        # ========== 获取向量数据库中的文件片段统计信息 ==========
        # 调用向量存储管理器获取每个文件被切分成多少个向量片段
        # 返回格式: [{"source": "文件名", "chunks": 片段数量}, ...]
        vector_files = vector_store_manager.list_uploaded_files()
        # 转换为字典，便于快速查找：{"文件名": 片段数量}
        vector_files_dict = {f["source"]: f["chunks"] for f in vector_files}

        # ========== 遍历上传目录，收集文件信息 ==========
        # 用于存储最终的文件列表
        files = []

        # 遍历上传目录中的所有条目（文件和文件夹）
        for item in os.listdir(upload_path):
            # 构建完整的文件路径
            item_path = os.path.join(upload_path, item)

            # 只处理文件，跳过子目录
            if os.path.isfile(item_path):
                # 获取文件大小（字节）
                file_size = os.path.getsize(item_path)
                # 获取文件最后修改时间戳（Unix时间戳）
                modified_time = os.path.getmtime(item_path)

                # 将文件信息添加到列表中
                files.append({
                    "filename": item,                    # 文件名（不含路径）
                    "size": file_size,                   # 文件大小（原始字节数）
                    "size_formatted": format_file_size(file_size),  # 格式化后的可读大小
                    "modified": modified_time,           # 最后修改时间戳
                    "chunks": vector_files_dict.get(item, 0)  # 向量片段数量（未处理则为0）
                })

        # ========== 按修改时间排序 ==========
        # 按modified字段降序排列，最新修改的文件排在最前面
        files.sort(key=lambda x: x["modified"], reverse=True)

        # ========== 返回结果 ==========
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

# 为 IIS 部署提供 WSGI 应用对象
# IIS + wfastcgi 需要 WSGI 应用，而不是 ASGI 应用
# 使用 a2wsgi 将 FastAPI (ASGI) 转换为 WSGI
try:
    from a2wsgi import ASGI2WSGI
    # 创建 WSGI 应用对象，供 IIS 使用
    # IIS 部署时会使用此对象
    wsgi_app = ASGI2WSGI(app)
except ImportError:
    # 如果没有安装 a2wsgi，则不提供 WSGI 应用
    # 这种情况下只能使用 uvicorn 直接运行
    wsgi_app = None
    print("警告：未安装 a2wsgi，IIS 部署将不可用。请运行: pip install a2wsgi")

if __name__ == "__main__":
    import uvicorn
    # 启动FastAPI应用
    # host: 监听地址，0.0.0.0表示接受所有网络接口的连接
    # port: 监听端口，默认为8000
    # 注意：请求体大小限制通过自定义中间件 limit_request_size 实现（见上方）
    # 解决大文件上传时报错 "File is not a zip file" 的问题
    # DOCX文件本质是ZIP压缩包，文件被截断后会报此错误
    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000))
    )