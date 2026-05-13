"""
RAG引擎模块
===========

核心功能：接收用户问题 -> 检索相关文档 -> 调用大模型生成回答

RAG（Retrieval-Augmented Generation）工作原理：
1. 用户提出问题
2. 将问题向量化后在向量数据库中检索相关文档片段
3. 将检索到的文档作为上下文拼接到提示词中
4. 调用大模型，让模型基于提供的上下文回答问题

模块职责：
---------
- 管理大模型配置（API地址、密钥、模型名称等）
- 执行文档检索和答案生成
- 支持线上模型和本地Ollama模型
- 配置持久化到.env文件

调用关系：
---------
- 被调用方：main.py (query 接口、config 相关接口)
- 调用外部：vector_store.py (向量检索)、requests (API调用)

配置依赖：
---------
- TOP_K: 检索返回的文档数量（默认3）
- TEMPERATURE: 大模型温度参数（默认0.7）
- LLM_API_BASE: 大模型API地址
- LLM_API_KEY: 大模型API密钥
- LLM_MODEL_NAME: 大模型名称

========================================
调用流程说明（用户查询流程 QUERY_FLOW）
========================================

【QUERY_FLOW-1】main.py query() 接收用户查询请求
    ↓
【QUERY_FLOW-2】RAGEngine.query() 处理查询（本模块主入口）
    ↓
【QUERY_FLOW-3】vector_store_manager.get_vector_store() 获取向量存储
    ↓
【QUERY_FLOW-4】VectorStoreManager.similarity_search() 执行相似度搜索
    ↓
【QUERY_FLOW-5】OllamaEmbeddings.embed_query() 生成查询向量
    ↓
【QUERY_FLOW-6】RAGEngine.format_docs() 格式化检索到的文档
    ↓
【QUERY_FLOW-7】requests.post() 调用大模型API生成回答
"""

# ========== 标准库导入 ==========
import os  
"""操作系统模块，用于文件路径处理和环境变量读取"""

import requests  
"""HTTP请求库，用于调用大模型API"""

# ========== LangChain相关导入 ==========
from langchain_core.prompts import PromptTemplate  
"""提示词模板类，用于构建结构化的LLM输入"""

from langchain_core.documents import Document  
"""LangChain的Document文档对象类型"""

# ========== 第三方库导入 ==========
from dotenv import load_dotenv  
"""dotenv库，用于从.env文件加载环境变量"""

from typing import List, Optional, Dict, Any  
"""类型提示模块"""

# ========== 内部模块导入 ==========
from .vector_store import vector_store_manager  
"""向量数据库管理器，用于文档检索"""

# 加载环境变量配置
load_dotenv()

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# .env配置文件的完整路径
ENV_FILE_PATH = os.path.join(BASE_DIR, '.env')

# ========== 线上模型配置映射 ==========
"""
预定义的线上模型提供商配置，便于快速切换不同模型服务

结构说明：
- key: 提供商名称（如 deepseek、qwen、openai、ollama）
- value: 包含以下字段的字典：
    - api_base: API基础地址
    - models: 可用模型名称列表
"""
ONLINE_MODELS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-r1-chat"]
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"]
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    },
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "models": ["qwen2.5:7b-instruct", "llama3.2", "mistral"]
    }
}

class RAGEngine:
    """
    RAG（检索增强生成）引擎类
    
    核心职责：
    --------
    - 接收用户问题并进行语义检索
    - 将检索到的文档作为上下文传递给大模型
    - 生成基于文档内容的准确回答
    
    字段说明：
    --------
    - top_k: int - 检索时返回最相关的文档数量（默认3）
    - temperature: float - 大模型温度参数，控制生成随机性（0-1，默认0.7）
    - llm_api_base: str - 大模型API基础地址
    - llm_api_key: str - 大模型API密钥（敏感信息）
    - llm_model_name: str - 大模型名称
    - vector_store: Any - 向量存储实例（延迟初始化）
    - prompt_template: PromptTemplate - RAG提示词模板
    
    RAG工作流程：
    ------------
    1. 用户提问 → 调用 query() 方法
    2. 问题向量化 → 调用向量数据库
    3. 检索相关文档 → 返回Top-K个相似片段
    4. 构建提示词 → 拼接上下文和问题
    5. 调用大模型 → 生成回答
    6. 返回结果 → 包含回答和来源文档
    
    调用者：
    -------
    main.py 的 /api/query 接口和配置相关接口
    """
    
    def __init__(self):
        """
        初始化RAG引擎
        
        从环境变量读取配置：
        -------------------
        - TOP_K: 检索时返回最相关的文档数量（默认3）
        - TEMPERATURE: 大模型生成温度（默认0.7）
        - LLM_API_BASE: 大模型API地址（默认Ollama本地地址）
        - LLM_API_KEY: 大模型API密钥
        - LLM_MODEL_NAME: 大模型名称
        
        初始化策略：
        ----------
        - 延迟初始化向量存储：避免服务器启动时下载嵌入模型失败
        - 默认配置指向本地Ollama服务，便于快速启动
        """
        # Top-K: 检索时返回最相关的K个文档片段，影响回答的丰富度
        self.top_k: int = int(os.getenv("TOP_K", 3))
        
        # Temperature: 控制生成随机性（0-1），值越大回答越随机、富有创造性
        self.temperature: float = float(os.getenv("TEMPERATURE", 0.7))
        
        # 大模型的API基础地址，支持线上API和本地Ollama服务
        self.llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
        
        # 大模型的API密钥，用于身份验证（线上模型必填）
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "sk-xxx")
        
        # 大模型的名称（如 llama3.2、deepseek-chat、gpt-4等）
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "local-model")
        
        # 向量存储实例，延迟初始化（首次查询时创建）
        # 这样可以避免服务器启动时下载嵌入模型失败
        self.vector_store = None
        
        # RAG提示词模板，定义大模型的输入格式
        # 模板变量：
        #   {context} - 检索到的相关文档内容
        #   {question} - 用户的问题
        self.prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下上下文信息回答问题：

{context}

问题：{question}

请根据提供的上下文信息，用简洁明了的语言回答问题。如果上下文信息不足以回答问题，请说明无法回答。"""
        )
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        将检索到的多个文档片段格式化为单个字符串
        
        【QUERY_FLOW-6】被 RAGEngine.query() 调用
        
        参数：
        ------
        docs: List[Document] - 检索到的Document对象列表（相关文档片段）
        
        返回：
        ------
        str - 用双换行符连接的文档内容字符串
        
        调用者：
        -------
        RAGEngine.query() 内部调用
        
        设计意图：
        --------
        将多个文档片段合并为一个字符串，作为大模型的上下文输入，
        保持文档之间的清晰分隔（使用两个换行符）。
        """
        # 将每个文档的page_content提取出来，用\n\n连接
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        处理用户问题的核心方法（主入口）
        
        【QUERY_FLOW-2】被 main.py query() 接口调用
        
        参数：
        ------
        question: str - 用户提出的问题
        
        返回：
        ------
        Dict[str, Any] - 包含以下字段的字典：
            - answer: str - 大模型生成的回答
            - sources: List[str] - 参考的文档来源列表（去重后）
            - success: bool - 是否成功
        
        处理流程：
        --------
        【QUERY_FLOW-2.1】延迟初始化向量存储（首次查询时）
        【QUERY_FLOW-2.2】文档检索：使用向量数据库进行相似度搜索
        【QUERY_FLOW-2.3】格式化上下文：将多个文档片段合并为字符串
        【QUERY_FLOW-2.4】构建提示词：拼接上下文和问题
        【QUERY_FLOW-2.5】调用大模型：发送请求并获取回答
        【QUERY_FLOW-2.6】返回结果：包含回答和来源文档
        
        调用关系：
        --------
        - 【调用 QUERY_FLOW-3】vector_store_manager.get_vector_store() 获取向量存储
        - 【调用 QUERY_FLOW-4】vector_store.similarity_search() 执行检索
        - 【调用 QUERY_FLOW-6】RAGEngine.format_docs() 格式化文档
        - 【调用 QUERY_FLOW-7】requests.post() 调用大模型API
        
        调用者：
        -------
        main.py 的 /api/query 接口
        """
        try:
            # ========== 【QUERY_FLOW-2.1】延迟初始化向量存储 ==========
            # 只有在首次查询时才初始化向量存储
            # 这样可以避免服务器启动时下载嵌入模型失败
            if self.vector_store is None:
                # 【调用 QUERY_FLOW-3】vector_store_manager.get_vector_store()
                self.vector_store = vector_store_manager.get_vector_store()
            
            # ========== 【QUERY_FLOW-2.2】文档检索 ==========
            # 使用向量数据库进行相似度搜索，找到最相关的K个文档片段
            # 例如问题"什么是RAG"，会找到向量数据库中与这个问题最相似的文档
            try:
                # 【调用 QUERY_FLOW-4】vector_store.similarity_search()
                docs: List[Document] = self.vector_store.similarity_search(question, k=self.top_k)
            except Exception as e:
                # 如果集合不存在或其他向量数据库错误，返回友好提示
                if "Collection" in str(e) and "does not exist" in str(e):
                    return {
                        "answer": "抱歉，当前没有已上传的文档。请先在后台管理页面上传文档后再进行查询。",
                        "sources": [],
                        "success": True
                    }
                raise  # 其他异常继续抛出
            
            # ========== 检查是否找到相关文档 ==========
            # 如果没有检索到任何文档，返回友好提示
            if not docs or len(docs) == 0:
                return {
                    "answer": "抱歉，在已上传的文档中没有找到相关信息。请尝试更换查询关键词或上传更多相关文档。",
                    "sources": [],
                    "success": True
                }
            
            # ========== 【QUERY_FLOW-2.3】格式化上下文 ==========
            # 将检索到的多个文档片段合并成一个字符串，作为大模型的参考资料
            # 【调用 QUERY_FLOW-6】RAGEngine.format_docs(docs)
            context: str = self.format_docs(docs)
            
            # ========== 【QUERY_FLOW-2.4】构建提示词 ==========
            # 使用提示词模板，将上下文和问题组合成完整的prompt
            prompt: str = self.prompt_template.format(context=context, question=question)
            
            # ========== 【QUERY_FLOW-2.5】调用大模型 ==========
            # 构建API请求URL（OpenAI兼容格式）
            llm_url = f"{self.llm_api_base}/chat/completions"
            
            # 构建请求体
            payload = {
                "model": self.llm_model_name,        # 模型名称
                "messages": [                        # 消息列表
                    {"role": "user", "content": prompt}  # 用户角色的消息
                ],
                "temperature": self.temperature       # 温度参数
            }
            
            # 调试日志：输出调用信息
            print(f"DEBUG - 调用大模型API:")
            print(f"DEBUG -   URL: {llm_url}")
            print(f"DEBUG -   Model: {self.llm_model_name}")
            print(f"DEBUG -   API Key: {'已设置' if self.llm_api_key else '未设置'}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json"
            }
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            # 【调用 QUERY_FLOW-7】requests.post() 发送请求（超时时间120秒）
            response = requests.post(llm_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()  # 检查HTTP错误
            result = response.json()
            
            # 从API响应中提取大模型的回答
            answer: str = result["choices"][0]["message"]["content"]
            
            # 提取回答参考的文档来源（去重）
            source_files: List[str] = [doc.metadata.get("source", "unknown") for doc in docs]
            
            # ========== 【QUERY_FLOW-2.6】返回结果 ==========
            return {
                "answer": answer,                          # 大模型生成的回答
                "sources": list(set(source_files)),       # 参考的文档来源列表（去重）
                "success": True                            # 是否成功
            }
        except Exception as e:
            # 如果发生任何错误，返回错误信息
            return {
                "answer": f"查询失败: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def update_config(self, 
                      top_k: Optional[int] = None, 
                      temperature: Optional[float] = None, 
                      llm_api_base: Optional[str] = None, 
                      llm_api_key: Optional[str] = None, 
                      llm_model_name: Optional[str] = None) -> None:
        """
        更新RAG引擎的配置参数
        
        参数：
        ------
        top_k: Optional[int] - 检索返回的相关文档数量（默认None，不更新）
        temperature: Optional[float] - 回答随机性参数（0-1之间，默认None）
        llm_api_base: Optional[str] - 大模型API地址（默认None）
        llm_api_key: Optional[str] - 大模型API密钥（默认None，空字符串不更新）
        llm_model_name: Optional[str] - 大模型名称（默认None）
        
        调用者：
        -------
        main.py 的 /api/config 接口（POST请求）
        
        注意事项：
        --------
        - 所有参数都是可选的，只更新传入的参数
        - API密钥为空字符串时不更新，避免意外清空
        - 更新后自动持久化到.env文件
        
        配置更新流程：
        ------------
        1. 更新内存中的配置
        2. 调用 _save_config_to_env() 持久化到文件
        """
        # 更新Top-K（检索返回数量）
        if top_k is not None:
            self.top_k = top_k
        
        # 更新温度参数（控制生成随机性）
        if temperature is not None:
            self.temperature = temperature
        
        # 更新API地址
        if llm_api_base is not None:
            self.llm_api_base = llm_api_base
        
        # 更新API密钥（空字符串不更新，避免意外清空）
        if llm_api_key is not None and llm_api_key.strip():
            self.llm_api_key = llm_api_key
        
        # 更新模型名称
        if llm_model_name is not None:
            self.llm_model_name = llm_model_name
        
        # 持久化配置到.env文件（确保重启后配置不丢失）
        self._save_config_to_env()
    
    def _save_config_to_env(self) -> None:
        """
        将当前配置持久化到.env文件（私有方法）
        
        执行逻辑：
        --------
        1. 读取当前.env文件内容（如果存在）
        2. 更新现有配置项或添加新配置项
        3. 写入更新后的内容
        
        配置项映射：
        ----------
        - TOP_K → self.top_k
        - TEMPERATURE → self.temperature
        - LLM_API_BASE → self.llm_api_base
        - LLM_API_KEY → self.llm_api_key（敏感信息，明文存储）
        - LLM_MODEL_NAME → self.llm_model_name
        
        调用者：
        -------
        RAGEngine.update_config() 内部调用
        """
        try:
            # 读取当前.env文件内容
            if os.path.exists(ENV_FILE_PATH):
                with open(ENV_FILE_PATH, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                lines = []
            
            # 配置项映射：环境变量名 → 当前值
            config_map = {
                'TOP_K': str(self.top_k),
                'TEMPERATURE': str(self.temperature),
                'LLM_API_BASE': self.llm_api_base,
                'LLM_API_KEY': self.llm_api_key,
                'LLM_MODEL_NAME': self.llm_model_name
            }
            
            # 标记哪些配置项已经更新
            updated_keys = set()
            
            # 更新现有配置行
            for i, line in enumerate(lines):
                for key in config_map:
                    if line.startswith(f'{key}='):
                        lines[i] = f'{key}={config_map[key]}\n'
                        updated_keys.add(key)
                        break
            
            # 添加未找到的配置项（追加到文件末尾）
            for key, value in config_map.items():
                if key not in updated_keys:
                    lines.append(f'{key}={value}\n')
            
            # 写入更新后的内容
            with open(ENV_FILE_PATH, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"DEBUG - 配置已持久化到 {ENV_FILE_PATH}")
        except Exception as e:
            print(f"ERROR - 配置持久化失败: {str(e)}")
    
    def _mask_api_key(self, api_key: str) -> str:
        """
        将API密钥转换为掩码形式（私有方法）
        
        参数：
        ------
        api_key: str - 原始API密钥
        
        返回：
        ------
        str - 掩码形式的密钥，如 "sk-8e8a**********25c3"
        
        掩码规则：
        --------
        - 显示前4位和后4位
        - 中间部分用*代替
        - 如果密钥长度<=8，全部用*代替
        
        调用者：
        -------
        RAGEngine.get_config() 内部调用
        """
        if not api_key:
            return ""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]

    def get_config(self) -> Dict[str, Any]:
        """
        获取当前RAG引擎的配置（API密钥以掩码形式返回）
        
        返回：
        ------
        Dict[str, Any] - 包含当前配置的字典：
            - top_k: int - 检索返回数量
            - temperature: float - 温度参数
            - llm_api_base: str - API地址
            - llm_model_name: str - 模型名称
            - llm_api_key: str - 掩码形式的API密钥
        
        调用者：
        -------
        main.py 的 /api/config 接口（GET请求）
        
        安全注意：
        --------
        API密钥以掩码形式返回，避免敏感信息泄露
        """
        masked_key = self._mask_api_key(self.llm_api_key)
        return {
            "top_k": self.top_k,
            "temperature": self.temperature,
            "llm_api_base": self.llm_api_base,
            "llm_model_name": self.llm_model_name,
            "llm_api_key": masked_key
        }
    
    @staticmethod
    def get_online_models() -> Dict[str, Any]:
        """
        获取支持的线上模型列表（静态方法）
        
        返回：
        ------
        Dict[str, Any] - 包含各平台模型信息的字典：
            - key: 提供商名称（deepseek、qwen、openai、ollama）
            - value: 包含 api_base 和 models 字段的字典
        
        调用者：
        -------
        main.py 的 /api/models 接口
        
        用途：
        ------
        供前端展示可用的模型选项，帮助用户选择配置
        """
        return ONLINE_MODELS