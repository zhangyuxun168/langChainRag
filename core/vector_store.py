"""
向量存储模块
============

核心功能：管理ChromaDB向量数据库的初始化、文档存储和检索

模块职责：
---------
1. 将文本片段转换为向量（嵌入）并存储到ChromaDB
2. 根据用户问题进行相似度搜索，找到最相关的文档片段
3. 管理向量数据库的集合（Collection）
4. 支持文档列表查询和删除操作

设计模式：
---------
- 单例模式：确保整个应用只有一个向量数据库连接
- 延迟初始化：嵌入函数在第一次使用时才创建，避免启动时下载模型失败

调用关系：
---------
- 被调用方：rag_engine.py (检索文档)、main.py (上传/删除文档)
- 调用外部：Ollama嵌入API、ChromaDB

配置依赖：
---------
- CHROMA_DB_PATH: 向量数据库存储路径（默认./chroma_db）
- OLLAMA_API_BASE: Ollama API地址（默认http://localhost:11434）
- EMBEDDING_MODEL: 嵌入模型名称（默认使用LLM_MODEL_NAME）

========================================
调用流程说明
========================================

【文件上传流程 UPLOAD_FLOW】
【UPLOAD_FLOW-5】main.py upload_file() 调用 vector_store_manager.add_documents()
    ↓
【UPLOAD_FLOW-6】VectorStoreManager.add_documents() 添加文档
    ↓
【UPLOAD_FLOW-7】VectorStoreManager.get_vector_store() 获取向量存储
    ↓
【UPLOAD_FLOW-8】VectorStoreManager.initialize() 初始化数据库连接
    ↓
【UPLOAD_FLOW-9】VectorStoreManager._init_embedding() 初始化嵌入函数
    ↓
【UPLOAD_FLOW-10】OllamaEmbeddings.embed_documents() 生成嵌入向量

【用户查询流程 QUERY_FLOW】
【QUERY_FLOW-3】RAGEngine.query() 调用 vector_store_manager.get_vector_store()
    ↓
【QUERY_FLOW-4】VectorStoreManager.get_vector_store() 获取向量存储
    ↓
【QUERY_FLOW-5】VectorStoreManager.similarity_search() 执行相似度搜索
    ↓
【QUERY_FLOW-6】OllamaEmbeddings.embed_query() 生成查询向量
"""

# ========== 标准库导入 ==========
import os  
"""操作系统模块，用于读取环境变量、处理文件路径"""

import json  
"""JSON处理模块，用于解析Ollama API响应"""

from typing import List, Optional  
"""类型提示模块，用于标注变量和函数的类型"""

# ========== 第三方库导入 ==========
import chromadb  
"""ChromaDB向量数据库客户端，用于存储和检索向量数据"""

import requests  
"""HTTP请求库，用于调用Ollama嵌入API"""

from chromadb.config import Settings  
"""ChromaDB配置类，用于设置数据库参数"""

# ========== LangChain相关导入 ==========
from langchain_community.vectorstores import Chroma  
"""LangChain的Chroma向量存储封装类，提供添加文档和相似性搜索功能"""

from langchain_core.embeddings import Embeddings  
"""LangChain嵌入基类，定义嵌入函数接口"""

from langchain_core.documents import Document  
"""LangChain的Document文档对象类型，表示一段文本及其元数据"""

# ========== 第三方库导入 ==========
from dotenv import load_dotenv  
"""dotenv库，用于从.env文件加载环境变量配置"""

# 加载环境变量配置
load_dotenv()

class OllamaEmbeddings(Embeddings):
    """
    使用Ollama嵌入API的自定义嵌入类
    
    核心功能：
    --------
    - 调用本地Ollama服务生成文本嵌入向量
    - 避免从HuggingFace下载大型嵌入模型
    - 实现LangChain Embeddings接口，可无缝集成
    
    字段说明：
    --------
    - model_name: str - 用于生成嵌入的Ollama模型名称
    - base_url: str - Ollama API基础地址（不带/v1后缀）
    
    Ollama嵌入API说明：
    ------------------
    - API地址：http://localhost:11434/api/embeddings
    - 请求格式：{"model": "模型名称", "prompt": "要嵌入的文本"}
    - 返回格式：{"embedding": [向量数组]}
    
    调用者：
    -------
    VectorStoreManager._init_embedding() 内部创建实例
    """

    def __init__(self, model_name: str = "qwen2.5:7b-instruct"):
        """
        初始化Ollama嵌入类
        
        参数：
        ------
        model_name: str - 用于生成嵌入的Ollama模型名称，默认为qwen2.5:7b-instruct
        
        初始化流程：
        ----------
        1. 设置模型名称
        2. 从环境变量读取Ollama API地址，默认为http://localhost:11434
        """
        self.model_name = model_name
        # Ollama原生嵌入API地址，不带/v1后缀（/v1是OpenAI兼容模式）
        self.base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip('/')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为多个文档生成嵌入向量
        
        参数：
        ------
        texts: List[str] - 要嵌入的文本列表
        
        返回：
        ------
        List[List[float]] - 每个文本对应的向量列表
        
        调用者：
        -------
        Chroma.add_documents() 内部调用（批量添加文档时）
        
        执行逻辑：
        --------
        遍历文本列表，逐个调用 _get_embedding() 获取向量
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量
        
        参数：
        ------
        text: str - 要嵌入的查询文本（用户问题）
        
        返回：
        ------
        List[float] - 文本对应的向量
        
        调用者：
        -------
        Chroma.similarity_search() 内部调用（检索时）
        """
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """
        调用Ollama嵌入API获取文本向量（私有方法）
        
        参数：
        ------
        text: str - 要嵌入的文本
        
        返回：
        ------
        List[float] - 文本对应的向量（浮点数数组）
        
        API调用详情：
        ------------
        - URL: {base_url}/api/embeddings
        - 请求方法: POST
        - 请求体: {"model": model_name, "prompt": text}
        - 响应格式: {"embedding": [向量数组]}
        
        调用者：
        -------
        OllamaEmbeddings.embed_documents() 和 embed_query()
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        # 调试日志：打印实际请求的URL和模型名称
        print(f"DEBUG - 调用Ollama嵌入API:")
        print(f"DEBUG -   URL: {url}")
        print(f"DEBUG -   Model: {self.model_name}")
        print(f"DEBUG -   Base URL: {self.base_url}")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()  # 检查HTTP错误（4xx/5xx）
            result = response.json()
            print(f"DEBUG - 嵌入API调用成功，向量维度: {len(result.get('embedding', []))}")
            return result.get("embedding", [])
        except requests.exceptions.RequestException as e:
            print(f"ERROR - 嵌入API调用失败: {str(e)}")
            raise RuntimeError(f"调用Ollama嵌入API失败: {str(e)}")

class VectorStoreManager:
    """
    向量数据库管理器类（单例模式）
    
    核心职责：
    --------
    - 管理ChromaDB向量数据库的连接和初始化
    - 将文本片段转换为向量并存储
    - 根据用户问题进行相似度搜索
    - 管理文档列表和删除操作
    
    字段说明：
    --------
    - _instance: VectorStoreManager - 单例实例（类属性）
    - _initialized: bool - 是否已初始化数据库连接
    - chroma_client: chromadb.PersistentClient - ChromaDB持久化客户端
    - embedding_function: OllamaEmbeddings - 嵌入函数实例（延迟初始化）
    
    设计模式：
    --------
    - 单例模式：确保整个应用只有一个数据库连接
    - 延迟初始化：嵌入函数在第一次使用时创建
    
    调用者：
    -------
    - rag_engine.py: 检索文档
    - main.py: 上传/删除文档、获取文件列表
    
    配置依赖：
    --------
    - CHROMA_DB_PATH: 数据库存储路径（默认./chroma_db）
    - OLLAMA_API_BASE: Ollama API地址
    - EMBEDDING_MODEL: 嵌入模型名称
    """
    
    # 单例实例（类属性）
    _instance = None
    # 是否已初始化数据库连接
    _initialized: bool = False
    # ChromaDB持久化客户端
    chroma_client: Optional[chromadb.PersistentClient] = None
    # 嵌入函数实例（延迟初始化）
    embedding_function: Optional[OllamaEmbeddings] = None
    
    def __new__(cls) -> 'VectorStoreManager':
        """
        实现单例模式，确保整个应用只有一个向量数据库连接
        
        返回：
        ------
        VectorStoreManager: 唯一的单例实例
        
        单例实现逻辑：
        ------------
        1. 检查类属性 _instance 是否为 None
        2. 如果为 None，创建新实例并初始化状态
        3. 返回已存在的实例或新创建的实例
        """
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.chroma_client = None
            cls._instance.embedding_function = None
        return cls._instance
    
    def initialize(self) -> None:
        """
        初始化向量数据库连接（延迟初始化）
        
        【UPLOAD_FLOW-8】被 VectorStoreManager.get_vector_store() 调用
        【QUERY_FLOW-4.1】被 VectorStoreManager.get_vector_store() 调用
        
        执行逻辑：
        --------
        【UPLOAD_FLOW-8.1 / QUERY_FLOW-4.1.1】检查是否已初始化
        【UPLOAD_FLOW-8.2 / QUERY_FLOW-4.1.2】如果未初始化，创建ChromaDB持久化客户端
        【UPLOAD_FLOW-8.3 / QUERY_FLOW-4.1.3】设置数据库存储路径和配置
        
        注意：
        ------
        - 嵌入函数会在第一次实际使用时（调用get_vector_store）才初始化
        - 这样可以避免服务器启动时下载嵌入模型失败
        
        配置参数：
        --------
        - path: 数据库文件存储路径，从环境变量CHROMA_DB_PATH读取，默认./chroma_db
        - anonymized_telemetry: 是否启用匿名遥测，默认False
        
        调用者：
        -------
        VectorStoreManager.get_vector_store() 内部调用
        """
        if not self._initialized:
            # 【UPLOAD_FLOW-8.2 / QUERY_FLOW-4.1.2】创建ChromaDB持久化客户端
            self.chroma_client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                settings=Settings(
                    anonymized_telemetry=False  # 禁用匿名遥测
                )
            )
            # 【UPLOAD_FLOW-8.3 / QUERY_FLOW-4.1.3】标记为已初始化
            self._initialized = True
    
    def _init_embedding(self) -> None:
        """
        初始化嵌入函数（延迟加载，私有方法）
        
        【UPLOAD_FLOW-9】被 VectorStoreManager.get_vector_store() 调用
        【QUERY_FLOW-4.2】被 VectorStoreManager.get_vector_store() 调用
        
        设计意图：
        --------
        只有在实际需要向量化时才创建嵌入函数，这样可以：
        1. 避免服务器启动时下载模型失败
        2. 允许用户先配置系统，再进行文档上传
        3. 节省内存资源
        
        实现逻辑：
        --------
        【UPLOAD_FLOW-9.1 / QUERY_FLOW-4.2.1】检查嵌入函数是否已初始化
        【UPLOAD_FLOW-9.2 / QUERY_FLOW-4.2.2】如果未初始化，创建OllamaEmbeddings实例
        【UPLOAD_FLOW-9.3 / QUERY_FLOW-4.2.3】从环境变量获取嵌入模型名称
        
        配置优先级：
        ----------
        1. EMBEDDING_MODEL 环境变量
        2. LLM_MODEL_NAME 环境变量
        3. 默认值 "qwen2.5:7b-instruct"
        
        调用者：
        -------
        VectorStoreManager.get_vector_store() 内部调用
        """
        if self.embedding_function is None:
            # 【UPLOAD_FLOW-9.2 / QUERY_FLOW-4.2.2】从环境变量获取嵌入模型名称
            embed_model_name = os.getenv("EMBEDDING_MODEL", os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct"))
            # 【UPLOAD_FLOW-9.3 / QUERY_FLOW-4.2.3】创建Ollama嵌入函数实例
            self.embedding_function = OllamaEmbeddings(model_name=embed_model_name)
    
    def get_vector_store(self, collection_name: str = "documents") -> Chroma:
        """
        获取或创建向量集合（主入口方法）
        
        【UPLOAD_FLOW-7】被 VectorStoreManager.add_documents() 调用
        【QUERY_FLOW-4】被 RAGEngine.query() 调用
        【QUERY_FLOW-5】被 VectorStoreManager.similarity_search() 调用
        
        参数：
        ------
        collection_name: str - 集合名称，用于区分不同类型的文档，默认为"documents"
        
        返回：
        ------
        Chroma - LangChain的Chroma向量存储对象，提供添加文档和相似性搜索功能
        
        执行流程：
        --------
        【UPLOAD_FLOW-7.1 / QUERY_FLOW-4.1】延迟初始化数据库连接（如果未初始化）
        【UPLOAD_FLOW-7.2 / QUERY_FLOW-4.2】延迟初始化嵌入函数（如果未初始化）
        【UPLOAD_FLOW-7.3 / QUERY_FLOW-4.3】尝试获取已存在的集合，不存在则创建
        【UPLOAD_FLOW-7.4 / QUERY_FLOW-4.4】返回Chroma向量存储对象
        
        调用关系：
        --------
        - 【调用 UPLOAD_FLOW-8 / QUERY_FLOW-4.1】VectorStoreManager.initialize() 初始化数据库连接
        - 【调用 UPLOAD_FLOW-9 / QUERY_FLOW-4.2】VectorStoreManager._init_embedding() 初始化嵌入函数
        
        调用者：
        -------
        - VectorStoreManager.add_documents()
        - VectorStoreManager.similarity_search()
        - RAGEngine.query()
        """
        # 【UPLOAD_FLOW-7.1 / QUERY_FLOW-4.1】延迟初始化数据库连接
        if not self._initialized:
            # 【调用 UPLOAD_FLOW-8 / QUERY_FLOW-4.1】VectorStoreManager.initialize()
            self.initialize()
        
        # 【UPLOAD_FLOW-7.2 / QUERY_FLOW-4.2】延迟初始化嵌入函数
        # 【调用 UPLOAD_FLOW-9 / QUERY_FLOW-4.2】VectorStoreManager._init_embedding()
        self._init_embedding()
        
        # 【UPLOAD_FLOW-7.3 / QUERY_FLOW-4.3】获取或创建集合
        try:
            # 尝试获取已存在的集合
            self.chroma_client.get_collection(name=collection_name)  # type: ignore
        except Exception:
            # 如果集合不存在，则创建一个新的
            self.chroma_client.create_collection(name=collection_name)  # type: ignore
        
        # 【UPLOAD_FLOW-7.4 / QUERY_FLOW-4.4】创建并返回Chroma向量存储对象
        return Chroma(
            client=self.chroma_client,           # ChromaDB客户端实例
            collection_name=collection_name,       # 集合名称
            embedding_function=self.embedding_function  # 嵌入函数
        )
    
    def add_documents(self, documents: List[Document], collection_name: str = "documents") -> None:
        """
        向向量数据库添加文档
        
        【UPLOAD_FLOW-6】被 main.py upload_file() 调用
        
        参数：
        ------
        documents: List[Document] - LangChain的Document对象列表，每个Document包含：
            - page_content: str - 文档文本内容
            - metadata: dict - 元数据（如来源文件名、片段ID等）
        collection_name: str - 要添加到的集合名称，默认为"documents"
        
        执行流程：
        --------
        【UPLOAD_FLOW-6.1】获取或创建指定名称的向量集合
        【UPLOAD_FLOW-6.2】使用embedding_function将每个文档的文本转换成向量
        【UPLOAD_FLOW-6.3】将向量和文档内容存储到ChromaDB
        
        调用关系：
        --------
        - 【调用 UPLOAD_FLOW-7】VectorStoreManager.get_vector_store() 获取向量存储
        
        调用者：
        -------
        main.py 的 upload_file 接口
        
        注意：
        ------
        - 文档会自动被向量化并存储
        - 元数据会被保留，用于后续检索时显示来源
        """
        # 【调用 UPLOAD_FLOW-7】VectorStoreManager.get_vector_store(collection_name)
        vector_store = self.get_vector_store(collection_name)
        # 添加文档到向量存储
        vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 3, collection_name: str = "documents") -> List[Document]:
        """
        执行相似性搜索（向量检索）
        
        【QUERY_FLOW-5】被 RAGEngine.query() 调用
        
        参数：
        ------
        query: str - 搜索查询文本（用户的问题）
        k: int - 返回最相似的文档数量，默认为3
        collection_name: str - 要搜索的集合名称，默认为"documents"
        
        返回：
        ------
        List[Document] - 与查询最相关的k个Document对象列表，按相似度降序排列
        
        执行流程：
        --------
        【QUERY_FLOW-5.1】使用embedding_function将query文本转换成向量
        【QUERY_FLOW-5.2】在ChromaDB中计算query向量与所有文档向量的余弦相似度
        【QUERY_FLOW-5.3】返回相似度最高的k个文档
        
        调用关系：
        --------
        - 【调用 QUERY_FLOW-4】VectorStoreManager.get_vector_store() 获取向量存储
        
        调用者：
        -------
        RAGEngine.query() 方法
        
        检索原理：
        --------
        - 使用余弦相似度衡量向量之间的相似性
        - 返回的文档按相似度从高到低排列
        - 相似度分数越高，表示文档与查询越相关
        """
        # 【调用 QUERY_FLOW-4】VectorStoreManager.get_vector_store(collection_name)
        vector_store = self.get_vector_store(collection_name)
        # 【QUERY_FLOW-5.3】执行相似性搜索
        return vector_store.similarity_search(query, k=k)
    
    def delete_collection(self, collection_name: str = "documents") -> bool:
        """
        删除整个向量集合（相当于删除数据库中的表）
        
        参数：
        ------
        collection_name: str - 要删除的集合名称，默认为"documents"
        
        返回：
        ------
        bool - 是否成功删除
        
        调用者：
        -------
        main.py 的 /api/clear-vectors 接口
        
        注意：
        ------
        - 此操作会删除集合中的所有文档，谨慎使用
        - 删除后无法恢复
        """
        if not self._initialized:
            self.initialize()
        try:
            self.chroma_client.delete_collection(name=collection_name)  # type: ignore
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    def get_collection_names(self) -> List[str]:
        """
        获取所有向量集合的名称列表
        
        返回：
        ------
        List[str] - 所有集合名称组成的列表
        
        调用者：
        -------
        主要用于调试和管理
        """
        if not self._initialized:
            self.initialize()
        return [col.name for col in self.chroma_client.list_collections()]  # type: ignore
    
    def get_all_documents(self, collection_name: str = "documents") -> List[Document]:
        """
        获取集合中的所有文档
        
        参数：
        ------
        collection_name: str - 集合名称，默认为"documents"
        
        返回：
        ------
        List[Document] - 所有Document对象列表
        
        调用者：
        -------
        主要用于调试和数据导出
        """
        if not self._initialized:
            self.initialize()
        self._init_embedding()
        
        vector_store = self.get_vector_store(collection_name)
        return vector_store.get()
    
    def list_uploaded_files(self, collection_name: str = "documents") -> List[dict]:
        """
        获取已上传文件列表及其片段数量
        
        参数：
        ------
        collection_name: str - 集合名称，默认为"documents"
        
        返回：
        ------
        List[dict] - 文件列表，每个元素包含:
            - source: str - 文件名
            - chunks: int - 该文件的片段数量
        
        执行逻辑：
        --------
        1. 获取集合中的所有文档元数据
        2. 按source字段（文件名）分组统计片段数量
        3. 返回文件列表
        
        调用者：
        -------
        main.py 的 /api/uploaded-files 接口
        
        用途：
        ------
        供前端展示已上传的文件列表
        """
        if not self._initialized:
            self.initialize()
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)  # type: ignore
            all_data = collection.get(include=["metadatas"])
            
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
            
            return [
                {"source": source, "chunks": count}
                for source, count in file_chunks.items()
            ]
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []
    
    def delete_file_chunks(self, filename: str, collection_name: str = "documents") -> bool:
        """
        删除指定文件名的所有向量数据片段
        
        参数：
        ------
        filename: str - 要删除的文件名（不含路径）
        collection_name: str - 集合名称，默认为"documents"
        
        返回：
        ------
        bool - 是否成功删除
        
        执行逻辑：
        --------
        1. 获取集合中的所有文档及其元数据
        2. 筛选出source字段匹配filename的文档ID
        3. 删除这些文档的向量数据
        
        调用者：
        -------
        main.py 的 DELETE /api/files/{filename} 接口
        
        用途：
        ------
        删除文件时同时清理向量数据库中的相关数据
        """
        if not self._initialized:
            self.initialize()
        
        try:
            collection = self.chroma_client.get_collection(name=collection_name)  # type: ignore
            all_data = collection.get(include=["metadatas", "documents"])
            
            if not all_data or "ids" not in all_data:
                return True  # 没有数据可删，视为成功
            
            # 筛选出需要删除的文档ID
            ids_to_delete = []
            for i, metadata in enumerate(all_data["metadatas"]):
                if metadata and metadata.get("source") == filename:
                    ids_to_delete.append(all_data["ids"][i])
            
            # 执行删除
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            
            return True
        except Exception as e:
            print(f"删除文件向量数据失败: {e}")
            return False

# ========== 全局单例实例 ==========
"""
创建全局向量存储管理器实例，供其他模块导入使用

使用方式：
--------
from core.vector_store import vector_store_manager

设计意图：
--------
- 确保整个应用只有一个向量数据库连接
- 简化调用方式，无需每次创建实例
"""
vector_store_manager: VectorStoreManager = VectorStoreManager()