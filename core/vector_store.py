# ==============================================================================
# 向量存储模块 - Vector Store Module
# ==============================================================================
# 【模块功能】管理ChromaDB向量数据库的初始化、文档存储和语义检索
# 【核心作用】将文档转换为向量并存储，支持高效的语义相似度搜索
# 
# 【调用关系】
#   - 被调用方：core/rag_engine.py (query方法)、backend/main.py (文件上传/删除接口)
#   - 调用外部：chromadb、langchain_community.vectorstores.Chroma、requests
# 
# 【架构设计】
#   1. OllamaEmbeddings类：封装嵌入API调用，支持Ollama原生格式和OpenAI兼容格式
#   2. VectorStoreManager类：单例模式管理ChromaDB连接和嵌入函数实例
#   3. 延迟初始化：数据库连接和嵌入函数按需创建，优化启动性能
# ==============================================================================

import os
import json
from typing import List, Optional

# 第三方库导入
import chromadb                                           # ChromaDB向量数据库核心库
import requests                                           # HTTP请求库，用于调用嵌入API
from chromadb.config import Settings                       # ChromaDB配置类
from langchain_community.vectorstores import Chroma       # LangChain封装的Chroma向量存储
from langchain_core.embeddings import Embeddings          # LangChain嵌入接口标准
from langchain_core.documents import Document             # LangChain文档数据类型
from dotenv import load_dotenv                            # 从.env文件加载环境变量

# 本地模型支持（Sentence Transformers）
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# 加载环境变量配置（优先从.env文件读取配置）
load_dotenv()

# 导入 backend 配置（支持 bge-m3 默认配置）
try:
    # 尝试从 backend 导入配置
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).parent.parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from config import Config
    _config = Config()
    _USE_BACKEND_CONFIG = True
except ImportError:
    # 如果导入失败，使用传统环境变量方式
    _USE_BACKEND_CONFIG = False
    _config = None

# ==============================================================================
# OllamaEmbeddings类 - 嵌入向量生成器
# ==============================================================================
# 【设计目的】统一封装不同嵌入服务的调用方式
# 【支持格式】
#   - Ollama原生格式：http://localhost:11434/api/embeddings
#   - OpenAI兼容格式：http://xxx/v1/embeddings
# 【核心特点】
#   - 自动检测API格式（URL包含/v1则使用OpenAI格式）
#   - 支持多种配置优先级（传入参数 > 环境变量 > 默认值）
#   - 专门处理bge-m3模型的特殊响应格式
class OllamaEmbeddings(Embeddings):
    """
    支持Ollama原生API和OpenAI兼容API的嵌入向量生成类
    
    【类字段说明】
    - model_name: str - 嵌入模型名称（如"bge-m3"、"qwen2.5:7b-instruct"）
    - base_url: str - API服务地址（如"http://localhost:11434"）
    - api_key: str - API密钥（可选，OpenAI格式需要）
    - use_openai_format: bool - 是否使用OpenAI格式调用API
    
    【实现原理】
    继承LangChain的Embeddings接口，实现embed_documents和embed_query方法，
    底层通过requests库调用外部嵌入服务获取向量。
    """

    def __init__(self, model_name: str = "qwen2.5:7b-instruct", 
                 api_base: str = None, api_key: str = None, api_format: str = None):
        """
        初始化嵌入向量生成器
        
        【配置优先级】传入参数 > EMBEDDING_*环境变量 > LLM_*环境变量 > OLLAMA_* > 默认值
        
        【参数说明】
        :param model_name: 嵌入模型名称，默认"qwen2.5:7b-instruct"
        :param api_base: API服务地址，默认None（将从环境变量获取）
        :param api_key: API密钥，默认None（OpenAI格式需要）
        :param api_format: API格式，可选"openai"或"ollama"，默认None（自动检测）
        """
        
        # 1. 设置嵌入模型名称（配置优先级：传入参数 > EMBEDDING_MODEL > LLM_MODEL_NAME > 默认值）
        self.model_name = model_name if model_name else \
            os.getenv("EMBEDDING_MODEL", os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct"))
        
        # 2. 设置API地址（配置优先级：传入参数 > EMBEDDING_API_BASE > LLM_API_BASE > OLLAMA_API_BASE > 默认值）
        if api_base:
            self.base_url = api_base.rstrip('/')  # 移除末尾斜杠，统一格式
        else:
            embed_api_base = os.getenv("EMBEDDING_API_BASE")
            if embed_api_base:
                self.base_url = embed_api_base.rstrip('/')
            else:
                llm_base_url = os.getenv("LLM_API_BASE")
                if llm_base_url:
                    self.base_url = llm_base_url.rstrip('/')
                else:
                    self.base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip('/')
        
        # 3. 设置API密钥（配置优先级：传入参数 > EMBEDDING_API_KEY > LLM_API_KEY > 默认空字符串）
        if api_key:
            self.api_key = api_key
        else:
            embed_api_key = os.getenv("EMBEDDING_API_KEY")
            if embed_api_key:
                self.api_key = embed_api_key
            else:
                self.api_key = os.getenv("LLM_API_KEY", "")
        
        # 4. 设置API格式（显式指定 > URL自动检测）
        # OpenAI格式URL通常包含/v1路径（如http://localhost:11434/v1）
        # Ollama原生格式使用/api/embeddings端点
        if api_format == "openai":
            self.use_openai_format = True
        elif api_format == "ollama":
            self.use_openai_format = False
        elif api_format is None:
            # 仅当未指定api_format时才根据URL路径自动检测
            self.use_openai_format = "/v1" in self.base_url
        else:
            self.use_openai_format = "/v1" in self.base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量为文档文本生成嵌入向量
        
        【调用场景】向向量数据库添加文档时，为每个文档片段生成向量
        
        【参数说明】
        :param texts: List[str] - 需要生成向量的文本列表
        :return: List[List[float]] - 向量列表，每个元素是一个浮点数组
        
        【实现细节】
        遍历文本列表，逐个调用_get_embedding方法获取向量，
        适合批量处理文档片段。
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量
        
        【调用场景】用户提问时，为查询语句生成向量用于相似度搜索
        
        【参数说明】
        :param text: str - 查询文本
        :return: List[float] - 嵌入向量（浮点数组）
        
        【实现细节】
        直接调用_get_embedding方法，适合单次查询场景。
        """
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """
        调用嵌入API获取文本向量（核心私有方法）
        
        【功能说明】
        根据配置的API格式，构建请求并解析响应，提取嵌入向量
        
        【参数说明】
        :param text: str - 需要生成向量的文本
        :return: List[float] - 嵌入向量
        
        【异常处理】
        捕获HTTP请求异常，打印错误信息后重新抛出RuntimeError
        """
        if self.use_openai_format:
            # OpenAI格式API调用
            # URL构建：如果已包含/v1则直接使用，否则添加/v1前缀
            url = f"{self.base_url}/embeddings" if "/v1" in self.base_url else f"{self.base_url}/v1/embeddings"
            
            # 请求体：符合OpenAI API规范
            payload = {
                "model": self.model_name,
                "input": text
            }
            # bge-m3模型在OpenAI格式下需要特殊选项
            if self.model_name == "bge-m3":
                payload["options"] = {"embedding_only": True}
            api_name = "OpenAI兼容"
        else:
            # Ollama原生格式API调用
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.model_name,
                "prompt": text,
                "options": {"embedding_only": False}  # bge-m3使用Ollama格式返回1024维向量
            }
            api_name = "Ollama"
        
        # 构建请求头
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"  # OpenAI格式需要认证
        headers["Content-Type"] = "application/json"
        
        try:
            # 发送POST请求，设置30秒超时（避免长时间阻塞）
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            # 检查HTTP错误（如4xx客户端错误、5xx服务器错误）
            response.raise_for_status()
            
            # 解析JSON响应
            result = response.json()
            
            # 根据API格式提取嵌入向量
            if self.use_openai_format:
                # OpenAI格式响应结构：{"data": [{"embedding": [0.1, 0.2, ...]}]}
                embedding = result.get("data", [{}])[0].get("embedding", [])
            else:
                # Ollama格式响应处理
                # bge-m3在embedding_only=False时返回{"embedding": {"dense": [...], "sparse": {...}}}
                # 需要提取dense部分作为最终向量
                raw_embedding = result.get("embedding", [])
                if isinstance(raw_embedding, dict) and "dense" in raw_embedding:
                    embedding = raw_embedding["dense"]
                else:
                    embedding = raw_embedding
            
            return embedding
        
        except requests.exceptions.RequestException as e:
            # 捕获所有HTTP请求异常（连接超时、DNS解析失败、HTTP错误等）
            print(f"ERROR - 嵌入API调用失败: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"ERROR - 响应内容: {e.response.text}")
            
            # 重新抛出异常，让上层调用者处理
            raise RuntimeError(f"调用{api_name}嵌入API失败: {str(e)}")

# ==============================================================================
# LocalEmbeddings类 - 本地离线模型嵌入向量生成器
# ==============================================================================
# 【设计目的】支持加载本地离线嵌入模型（如bge-small-zh-v1.5）
# 【支持模型】基于Sentence Transformers格式的模型
# 【核心特点】
#   - 无需网络连接，完全离线运行
#   - 支持从本地目录加载模型
#   - 自动处理模型缓存和加载
class LocalEmbeddings(Embeddings):
    """
    本地离线模型嵌入向量生成类
    
    【类字段说明】
    - model_path: str - 本地模型目录路径
    - model: SentenceTransformer - 加载的模型实例
    - _initialized: bool - 模型是否已初始化
    
    【实现原理】
    使用Sentence Transformers库加载本地模型，实现LangChain的Embeddings接口，
    支持embed_documents和embed_query方法。
    """

    def __init__(self, model_path: str):
        """
        初始化本地模型嵌入向量生成器
        
        【参数说明】
        :param model_path: str - 本地模型目录路径
        """
        self.model_path = model_path
        self._initialized = False
        self.model = None
        # bge-small-zh-v1.5 使用归一化向量，提高相似度计算准确性
        self.normalize_embeddings = True
        
        # 尝试预加载模型
        self._load_model()

    def _load_model(self):
        """
        加载本地模型（私有方法）
        
        【实现细节】
        使用Sentence Transformers库加载本地模型，设置trust_remote_code=True以支持自定义模型
        自动检测GPU，如果没有显卡则使用CPU
        """
        if self._initialized and self.model is not None:
            return
        
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("需要安装sentence_transformers库才能使用本地模型")
        
        # 自动检测设备（优先使用GPU）
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print("【向量化】检测到GPU，使用CUDA加速")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("【向量化】检测到Apple Silicon，使用MPS加速")
            else:
                device = "cpu"
                print("【向量化】未检测到GPU，使用CPU")
        except ImportError:
            device = "cpu"
            print("【向量化】PyTorch未安装，使用CPU")
        
        try:
            print(f"【向量化】正在加载本地模型: {self.model_path}")
            self.model = SentenceTransformer(
                self.model_path,
                trust_remote_code=True,
                device=device  # 指定设备
            )
            self._initialized = True
            print(f"【向量化】本地模型加载成功")
        except Exception as e:
            print(f"ERROR - 加载本地模型失败: {str(e)}")
            raise RuntimeError(f"加载本地模型失败: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量为文档文本生成嵌入向量
        
        【调用场景】向向量数据库添加文档时，为每个文档片段生成向量
        
        【参数说明】
        :param texts: List[str] - 需要生成向量的文本列表
        :return: List[List[float]] - 向量列表，每个元素是一个浮点数组
        
        【实现细节】
        使用 normalize_embeddings=True 对向量进行归一化，提高余弦相似度计算准确性
        """
        if not self._initialized:
            self._load_model()
        
        # 使用 Sentence Transformers 批量生成向量
        # convert_to_numpy=True: 返回 numpy 数组，便于转换为 Python 列表
        # normalize_embeddings=True: 对向量进行归一化，使其成为单位向量
        # 这样余弦相似度计算简化为向量点积，提高计算效率和准确性
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,  # 必须设置为 True，否则返回的是 PyTorch tensor
            normalize_embeddings=self.normalize_embeddings
        )
        # 将 numpy 数组转换为普通 Python 浮点数列表
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量
        
        【调用场景】用户提问时，为查询语句生成向量用于相似度搜索
        
        【参数说明】
        :param text: str - 查询文本
        :return: List[float] - 嵌入向量（浮点数组）
        
        【实现细节】
        使用 normalize_embeddings=True 对向量进行归一化，提高余弦相似度计算准确性
        """
        if not self._initialized:
            self._load_model()
        
        # 使用 Sentence Transformers 生成向量
        # convert_to_numpy=True: 返回 numpy 数组，便于转换为 Python 列表
        # normalize_embeddings=True: 对向量进行归一化，使其成为单位向量
        # 这样余弦相似度计算简化为向量点积，提高计算效率和准确性
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,  # 必须设置为 True，否则返回的是 PyTorch tensor
            normalize_embeddings=self.normalize_embeddings
        )
        # 将 numpy 数组转换为普通 Python 浮点数列表
        return embedding.tolist()

# ==============================================================================
# VectorStoreManager类 - 向量存储管理器（单例模式）
# ==============================================================================
# 【设计模式】单例模式（Singleton）
# 【设计目的】确保整个应用只有一个ChromaDB连接和嵌入函数实例，
#           避免重复创建连接，优化资源使用。
# 
# 【核心字段】
#   - _instance: 类属性，存储唯一实例
#   - _initialized: 标记是否已初始化数据库连接
#   - chroma_client: ChromaDB持久化客户端
#   - embedding_function: OllamaEmbeddings实例
class VectorStoreManager:
    """
    向量存储管理器（单例模式）
    
    【功能职责】
    1. 管理ChromaDB数据库连接（延迟初始化）
    2. 管理嵌入函数实例（按需创建）
    3. 提供文档添加、相似度搜索、集合管理等核心功能
    
    【使用方式】
    from core.vector_store import vector_store_manager
    vector_store = vector_store_manager.get_vector_store()
    """
    
    _instance = None                          # 单例实例（类级别的共享变量）
    _initialized: bool = False                # 数据库连接初始化标记
    chroma_client: Optional[chromadb.PersistentClient] = None  # ChromaDB客户端
    embedding_function: Optional[Embeddings] = None             # 嵌入函数实例（支持OllamaEmbeddings和LocalEmbeddings）
    
    def __new__(cls) -> 'VectorStoreManager':
        """
        实现单例模式的核心方法
        
        【设计原理】
        重写__new__方法，确保首次调用时创建实例，后续调用返回同一实例。
        
        :return: VectorStoreManager - 唯一的单例实例
        """
        if cls._instance is None:
            # 首次调用，创建新实例
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            # 初始化实例状态
            cls._instance._initialized = False
            cls._instance.chroma_client = None
            cls._instance.embedding_function = None
        # 返回已存在的实例（或刚创建的实例）
        return cls._instance
    
    def initialize(self) -> None:
        """
        初始化ChromaDB数据库连接（延迟初始化）
        
        【调用时机】
        首次调用get_vector_store时自动触发，避免应用启动时就建立数据库连接
        
        【实现细节】
        使用chromadb.PersistentClient创建持久化连接，数据存储在本地文件系统
        默认路径为./chroma_db，可通过CHROMA_DB_PATH环境变量配置
        """
        if not self._initialized:
            self.chroma_client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                settings=Settings(anonymized_telemetry=False)  # 禁用匿名遥测数据
            )
            self._initialized = True
    
    def _init_embedding(self, llm_api_base: str = None, llm_api_key: str = None, 
                        llm_model_name: str = None, use_local_ollama: bool = None) -> None:
        """
        初始化嵌入函数（延迟加载，私有方法）
        
        【调用时机】
        每次调用get_vector_store时触发，检查是否需要重新创建嵌入函数
        
        【设计目的】
        根据配置动态创建嵌入函数实例，支持运行时切换模型和API配置
        
        【配置逻辑】
        1. 如果 backend/config.py 可用，优先使用 Config 类的配置
        2. 如果未配置大模型，自动使用 bge-small-zh-v1.5 离线版作为默认嵌入模型
        3. 如果已配置大模型，使用大模型的配置作为嵌入模型
        4. 支持本地离线模型和远程API两种模式
        
        【参数说明】
        :param llm_api_base: LLM API地址（用于兼容旧接口）
        :param llm_api_key: LLM API密钥（用于兼容旧接口）
        :param llm_model_name: LLM模型名称（用于兼容旧接口）
        :param use_local_ollama: 是否使用本地Ollama服务
        """
        
        # 配置变量
        use_local = False
        local_model_path = None
        api_base = None
        api_key = ""
        model_name = "bge-small-zh-v1.5"
        api_format = "local"
        
        # 优先使用 backend/config.py 的配置（如果可用）
        if _USE_BACKEND_CONFIG and _config is not None:
            # 使用新的配置系统
            embedding_config = _config.get_embedding_config()
            api_base = embedding_config.get("api_base")
            api_key = embedding_config.get("api_key", "")
            model_name = embedding_config["model_name"]
            api_format = embedding_config["api_format"]
            use_local = embedding_config.get("use_local", False)
            local_model_path = embedding_config.get("local_model_path")
            
            # 打印配置来源（嵌入模型始终使用本地 bge-small-zh-v1.5）
            print(f"【配置】嵌入模型: {model_name} (本地离线模型)")
            if _config.is_llm_configured():
                llm_config = _config.get_llm_config()
                print(f"【配置】大模型: {llm_config['model_name']} (用于回答生成)")
            else:
                print(f"【配置】大模型: 未配置 (将直接返回检索结果)")
        else:
            # 使用传统的环境变量配置（向后兼容）
            # 1. 确定是否使用本地模型
            use_local_env = os.getenv("USE_LOCAL_EMBEDDING", "false").lower()
            use_local = use_local_env == "true"
            
            if use_local:
                # 使用本地模型
                api_format = "local"
                local_model_path = os.getenv("LOCAL_MODEL_PATH", "./backend/models/bge-small-zh-v1.5")
                model_name = "bge-small-zh-v1.5"
                print("【配置】使用本地离线模型")
            else:
                # 使用远程API
                # 1. 确定API地址
                embed_api_base = os.getenv("EMBEDDING_API_BASE")
                if embed_api_base:
                    api_base = embed_api_base
                elif llm_api_base:
                    api_base = llm_api_base
                else:
                    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
                
                # 2. 确定API密钥
                embed_api_key = os.getenv("EMBEDDING_API_KEY")
                if embed_api_key:
                    api_key = embed_api_key
                elif llm_api_key:
                    api_key = llm_api_key
                else:
                    api_key = os.getenv("LLM_API_KEY", "")
                
                # 3. 确定模型名称（默认使用 bge-small-zh-v1.5）
                embed_model_name = os.getenv("EMBEDDING_MODEL")
                if embed_model_name:
                    model_name = embed_model_name
                elif llm_model_name:
                    model_name = llm_model_name
                else:
                    model_name = os.getenv("LLM_MODEL_NAME", "bge-small-zh-v1.5")
                
                # 4. 确定API格式
                llm_api_base_env = os.getenv("LLM_API_BASE", "")
                if "/v1" in llm_api_base_env:
                    api_format = "openai"
                else:
                    api_format = "ollama"
        
        # 判断是否需要重新创建嵌入函数
        need_recreate = False
        current_config_hash = None
        
        if self.embedding_function is None:
            # 嵌入函数尚未创建，需要初始化
            need_recreate = True
        else:
            # 检查现有配置是否变化
            # 对于本地模型，比较模型路径
            if use_local:
                current_path = getattr(self.embedding_function, 'model_path', None)
                need_recreate = current_path != local_model_path
            else:
                # 对于远程API，比较API地址、密钥、模型名称
                current_base = getattr(self.embedding_function, 'base_url', None)
                current_key = getattr(self.embedding_function, 'api_key', None)
                current_model = getattr(self.embedding_function, 'model_name', None)
                
                need_recreate = (api_base and current_base != api_base.rstrip('/')) or \
                               (api_key and current_key != api_key) or \
                               (current_model != model_name)
        
        # 创建或更新嵌入函数实例
        if need_recreate:
            if use_local and local_model_path:
                # 使用本地离线模型
                print(f"【向量化】初始化本地嵌入模型 - 路径: {local_model_path}")
                self.embedding_function = LocalEmbeddings(model_path=local_model_path)
            else:
                # 使用远程API模型
                print(f"【向量化】初始化远程嵌入函数 - API: {api_base}, Model: {model_name}, Format: {api_format}")
                self.embedding_function = OllamaEmbeddings(
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    api_format=api_format
                )
    
    def get_vector_store(self, collection_name: str = "documents", 
                         llm_api_base: str = None, llm_api_key: str = None, 
                         llm_model_name: str = None, use_local_ollama: bool = False) -> Chroma:
        """
        获取或创建向量集合（主入口方法）
        
        【功能说明】
        返回指定名称的Chroma向量存储对象，如果集合不存在则自动创建
        
        【参数说明】
        :param collection_name: str - 集合名称，默认"documents"
        :param llm_api_base: str - LLM API地址（兼容参数）
        :param llm_api_key: str - LLM API密钥（兼容参数）
        :param llm_model_name: str - LLM模型名称（兼容参数）
        :param use_local_ollama: bool - 是否使用本地Ollama，默认False
        :return: Chroma - LangChain的Chroma向量存储对象
        
        【调用流程】
        1. 检查数据库连接是否初始化
        2. 初始化嵌入函数
        3. 获取或创建集合
        4. 返回Chroma向量存储对象
        """
        
        # 步骤1：延迟初始化数据库连接
        if not self._initialized:
            self.initialize()
        
        # 步骤2：初始化嵌入函数（按需创建）
        self._init_embedding(llm_api_base=llm_api_base, llm_api_key=llm_api_key, 
                            llm_model_name=llm_model_name, use_local_ollama=use_local_ollama)
        
        # 步骤3：获取或创建集合
        try:
            # 尝试获取已存在的集合
            self.chroma_client.get_collection(name=collection_name)
        except Exception:
            # 集合不存在，创建新集合
            self.chroma_client.create_collection(name=collection_name)
        
        # 步骤4：返回Chroma向量存储对象（封装了集合和嵌入函数）
        return Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[Document], collection_name: str = "documents", 
                      progress_callback=None, llm_api_base: str = None, 
                      llm_api_key: str = None, llm_model_name: str = None,
                      use_local_ollama: bool = False) -> None:
        """
        向向量数据库添加文档（分批处理）
        
        【功能说明】
        将文档列表转换为向量并存储到指定集合，支持进度回调
        
        【参数说明】
        :param documents: List[Document] - LangChain Document对象列表
        :param collection_name: str - 目标集合名称，默认"documents"
        :param progress_callback: Callable - 进度回调函数，签名：(current, total, message)
        :param llm_api_base: str - LLM API地址（兼容参数）
        :param llm_api_key: str - LLM API密钥（兼容参数）
        :param llm_model_name: str - LLM模型名称（兼容参数）
        :param use_local_ollama: bool - 是否使用本地Ollama，默认False
        
        【实现细节】
        - 分批处理：每批20个文档，避免一次性处理过多数据
        - 进度更新：每批处理完成后调用回调函数通知进度
        - 日志输出：打印处理进度信息
        """
        
        total_docs = len(documents)
        print(f"【向量化】开始向量化，共 {total_docs} 个文档")
        
        # 通知开始（如果有回调函数）
        if progress_callback:
            progress_callback(0, total_docs, "开始向量化...")
        
        # 获取向量存储对象
        vector_store = self.get_vector_store(collection_name, llm_api_base=llm_api_base, 
                                            llm_api_key=llm_api_key, llm_model_name=llm_model_name,
                                            use_local_ollama=use_local_ollama)
        
        # 根据文档数量动态调整批处理大小
        # 小文档集使用较大批次，大文档集使用较小批次，平衡性能和内存
        if total_docs <= 50:
            batch_size = 20  # 文档数较少，使用较大批次
        elif total_docs <= 200:
            batch_size = 15  # 文档数中等，使用中等批次
        else:
            batch_size = max(5, total_docs // 50)  # 文档数较多，动态计算批次大小
        
        print(f"【向量化】根据文档数量({total_docs}个)，设置批次大小为: {batch_size}")
        
        # 遍历处理每一批
        for i in range(0, total_docs, batch_size):
            # 截取当前批次的文档
            batch = documents[i:i+batch_size]
            # 将批次文档添加到向量数据库（内部会自动调用嵌入函数生成向量）
            vector_store.add_documents(batch)
            
            # 更新进度（如果有回调函数）
            if progress_callback:
                current = min(i + batch_size, total_docs)
                progress_callback(current, total_docs, f"正在向量化... {current}/{total_docs}")
                print(f"【向量化】进度回调: {current}/{total_docs}")
        
        # 通知完成（如果有回调函数）
        if progress_callback:
            progress_callback(total_docs, total_docs, "向量化完成")
            print(f"【向量化】进度回调: {total_docs}/{total_docs} (完成)")
        
        print(f"【向量化】向量化完成")
    
    def similarity_search(self, query: str, k: int = 3, collection_name: str = "documents",
                         llm_api_base: str = None, llm_api_key: str = None, 
                         llm_model_name: str = None, use_local_ollama: bool = False) -> List[Document]:
        """
        执行语义相似度搜索
        
        【功能说明】
        将查询文本转换为向量，在向量数据库中查找最相似的k个文档
        
        【参数说明】
        :param query: str - 查询文本
        :param k: int - 返回结果数量，默认3
        :param collection_name: str - 集合名称，默认"documents"
        :param llm_api_base: str - LLM API地址（兼容参数）
        :param llm_api_key: str - LLM API密钥（兼容参数）
        :param llm_model_name: str - LLM模型名称（兼容参数）
        :param use_local_ollama: bool - 是否使用本地Ollama，默认False
        :return: List[Document] - 匹配的文档列表（按相似度降序排列）
        
        【检索原理】
        1. 将查询文本转换为嵌入向量
        2. 使用余弦相似度算法计算与库中所有向量的相似度
        3. 返回相似度最高的前k个文档
        
        【应用场景】
        RAG问答场景：用户提问时，检索相关文档作为上下文
        """
        
        # 获取向量存储对象
        vector_store = self.get_vector_store(collection_name, llm_api_base=llm_api_base, 
                                            llm_api_key=llm_api_key, llm_model_name=llm_model_name,
                                            use_local_ollama=use_local_ollama)
        
        # 执行相似度搜索
        return vector_store.similarity_search(query, k=k)
    
    def delete_collection(self, collection_name: str = "documents") -> bool:
        """
        删除整个向量集合（谨慎使用）
        
        【功能说明】
        删除指定名称的集合及其所有数据，此操作不可恢复
        
        【参数说明】
        :param collection_name: str - 要删除的集合名称，默认"documents"
        :return: bool - 删除是否成功
        
        【返回值含义】
        - True: 删除成功
        - False: 删除失败（如集合不存在或权限不足）
        
        【调用场景】
        数据清理、重新初始化、更换嵌入模型时使用
        """
        
        # 确保数据库连接已初始化
        if not self._initialized:
            self.initialize()
        
        try:
            # 删除集合
            self.chroma_client.delete_collection(name=collection_name)
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    def get_collection_names(self) -> List[str]:
        """
        获取所有向量集合的名称列表
        
        【功能说明】
        返回数据库中所有已创建的集合名称
        
        :return: List[str] - 集合名称列表
        
        【应用场景】
        - 管理界面展示所有集合
        - 调试和监控
        """
        
        if not self._initialized:
            self.initialize()
        
        # 遍历所有集合并提取名称
        return [col.name for col in self.chroma_client.list_collections()]
    
    def get_all_documents(self, collection_name: str = "documents") -> List[Document]:
        """
        获取集合中的所有文档
        
        【功能说明】
        返回指定集合中的所有文档数据（包含内容和元数据）
        
        :param collection_name: str - 集合名称，默认"documents"
        :return: List[Document] - 文档列表
        
        【应用场景】
        - 数据导出
        - 调试和验证
        - 数据备份
        
        【注意事项】
        强制使用本地bge-m3模型确保兼容性
        """
        
        if not self._initialized:
            self.initialize()
        
        # 强制使用本地bge-m3模型
        self._init_embedding(use_local_ollama=True)
        
        # 获取向量存储并返回所有文档
        vector_store = self.get_vector_store(collection_name, use_local_ollama=True)
        return vector_store.get()
    
    def list_uploaded_files(self, collection_name: str = "documents") -> List[dict]:
        """
        获取已上传文件列表及其片段数量
        
        【功能说明】
        统计每个源文件在向量数据库中的文档片段数量
        
        :param collection_name: str - 集合名称，默认"documents"
        :return: List[dict] - 包含{"source": 文件名, "chunks": 片段数}的列表
        
        【实现原理】
        1. 获取集合中所有文档的元数据
        2. 按source字段（文件名）分组统计
        3. 返回统计结果
        
        【应用场景】
        管理界面展示已上传文件列表和处理状态
        """
        
        if not self._initialized:
            self.initialize()
        
        try:
            # 获取集合
            collection = self.chroma_client.get_collection(name=collection_name)
            # 获取所有文档的元数据
            all_data = collection.get(include=["metadatas"])
            
            # 检查数据有效性
            if not all_data or "metadatas" not in all_data:
                return []
            
            # 按文件名统计片段数量
            file_chunks = {}
            for metadata in all_data["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    if source not in file_chunks:
                        file_chunks[source] = 0
                    file_chunks[source] += 1
            
            # 转换为列表格式返回
            return [
                {"source": source, "chunks": count}
                for source, count in file_chunks.items()
            ]
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []
    
    def delete_file_chunks(self, filename: str, collection_name: str = "documents") -> tuple:
        """
        删除指定文件名的所有向量数据片段
        
        【功能说明】
        删除与指定文件相关的所有向量记录，支持精确匹配和模糊匹配
        
        【参数说明】
        :param filename: str - 要删除的文件名
        :param collection_name: str - 集合名称，默认"documents"
        :return: tuple - (是否删除成功, 是否执行了删除操作)
        
        【返回值含义】
        - (True, True): 成功删除了数据
        - (True, False): 没有可删除的数据（集合/文件不存在）
        - (False, _): 删除操作失败
        
        【匹配规则】
        支持三种匹配方式：
        1. 精确匹配：source == filename
        2. 文件名包含：filename in source
        3. 包含文件名：source in filename
        
        【应用场景】
        用户删除上传的文件时，同步删除对应的向量数据
        """
        
        if not self._initialized:
            self.initialize()
        
        try:
            # 获取所有集合名称
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            # 检查集合是否存在
            if collection_name not in collection_names:
                print(f"【删除】集合不存在，跳过向量数据删除")
                return (True, False)  # 成功，但没有执行删除
            
            # 获取集合对象
            collection = self.chroma_client.get_collection(name=collection_name)
            # 获取所有文档的ID和元数据
            all_data = collection.get(include=["metadatas", "documents"])
            
            # 检查数据有效性
            if not all_data or "ids" not in all_data or len(all_data["ids"]) == 0:
                print(f"【删除】集合为空，文件名: {filename}")
                return (True, False)  # 成功，但没有数据可删
            
            print(f"【删除】开始处理，总文档数: {len(all_data['ids'])}，文件名: {filename}")
            
            # 筛选出需要删除的文档ID
            ids_to_delete = []
            for i, metadata in enumerate(all_data["metadatas"]):
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    # 支持精确匹配和模糊匹配
                    if source == filename or filename in source or source in filename:
                        ids_to_delete.append(all_data["ids"][i])
                        print(f"【删除】匹配到文档: {source}")
            
            # 执行删除操作
            if ids_to_delete:
                print(f"【删除】准备删除 {len(ids_to_delete)} 个文档")
                collection.delete(ids=ids_to_delete)
                print(f"【删除】成功删除 {len(ids_to_delete)} 个文档")
                return (True, True)  # 成功删除了数据
            else:
                print(f"【删除】向量数据不存在，文件名: {filename}")
                return (True, False)  # 成功，但没有找到匹配的文档
        
        except Exception as e:
            print(f"【删除】失败: {e}")
            return (False, False)

# ==============================================================================
# 全局向量存储管理器实例
# ==============================================================================
# 【使用方式】
# from core.vector_store import vector_store_manager
# 
# # 获取向量存储
# vector_store = vector_store_manager.get_vector_store()
# 
# # 添加文档
# vector_store_manager.add_documents(documents)
# 
# # 相似度搜索
# results = vector_store_manager.similarity_search("你的问题")
# ==============================================================================
vector_store_manager: VectorStoreManager = VectorStoreManager()
