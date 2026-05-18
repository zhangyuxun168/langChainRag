# ==============================================================================
# Backend 配置文件 - 系统配置管理
# ==============================================================================
# 【文件功能】
# 1. 统一管理 backend 目录下的所有配置
# 2. 提供 bge-small-zh-v1.5 离线版默认配置（当系统未配置大模型时使用）
# 3. 支持从环境变量和 .env 文件读取配置
# 4. 支持加载本地离线嵌入模型
# 
# 【配置优先级】
# 传入参数 > 环境变量 > .env 文件 > 本文件默认值
# 
# 【使用方式】
# from backend.config import Config
# config = Config()
# embedding_config = config.get_embedding_config()
# ==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# 获取 backend 目录路径
BACKEND_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = BACKEND_DIR.parent

# 本地模型目录
LOCAL_MODELS_DIR = BACKEND_DIR / "models"

# 加载项目根目录的 .env 文件（如果存在）
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)


class Config:
    """
    系统配置管理类
    
    【功能说明】
    集中管理所有配置项，包括：
    - 大模型配置（LLM）
    - 嵌入模型配置（Embedding）
    - 向量数据库配置（ChromaDB）
    - 文档处理配置
    
    【核心特性】
    - 当未配置大模型时，自动使用 bge-small-zh-v1.5 离线版作为默认嵌入模型
    - 支持从环境变量覆盖配置
    - 支持加载本地离线模型
    """
    
    # ==================== bge-small-zh-v1.5 离线版默认配置 ====================
    # 当系统未配置大模型时，使用以下默认配置
    DEFAULT_EMBEDDING_MODEL = "bge-small-zh-v1.5"
    DEFAULT_EMBEDDING_DIMENSION = 512  # bge-small-zh-v1.5 返回的向量维度
    DEFAULT_LOCAL_MODEL_PATH = str(LOCAL_MODELS_DIR / "bge-small-zh-v1.5")
    
    # ==================== 大模型默认配置 ====================
    DEFAULT_LLM_MODEL = "qwen2.5:7b-instruct"
    DEFAULT_LLM_API_BASE = "http://localhost:11434/v1"
    
    # ==================== 向量数据库默认配置 ====================
    DEFAULT_CHROMA_DB_PATH = "./chroma_db"
    DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
    
    # ==================== 文档处理默认配置 ====================
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_TOP_K = 3
    DEFAULT_TEMPERATURE = 0.7
    
    @classmethod
    def is_llm_configured(cls) -> bool:
        """
        检查是否已配置大模型
        
        【判断逻辑】
        检查 LLM_API_BASE 或 LLM_MODEL_NAME 环境变量是否已设置
        
        :return: bool - True 表示已配置，False 表示未配置
        """
        llm_api_base = os.getenv("LLM_API_BASE")
        llm_model_name = os.getenv("LLM_MODEL_NAME")
        
        # 如果环境变量存在且不为空，则认为已配置
        return bool(llm_api_base and llm_api_base.strip()) or \
               bool(llm_model_name and llm_model_name.strip())
    
    @classmethod
    def is_local_model_available(cls) -> bool:
        """
        检查本地离线模型是否可用
        
        【判断逻辑】
        检查 backend/models/bge-small-zh-v1.5 目录是否存在且包含模型文件
        
        :return: bool - True 表示本地模型可用，False 表示不可用
        """
        model_path = Path(cls.DEFAULT_LOCAL_MODEL_PATH)
        if not model_path.exists():
            return False
        
        # 检查是否包含必要的模型文件
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
        return all((model_path / f).exists() for f in required_files)
    
    @classmethod
    def get_embedding_config(cls) -> dict:
        """
        获取嵌入模型配置
        
        【配置逻辑】
        1. 始终使用本地的 bge-small-zh-v1.5 离线模型进行向量嵌入
        2. 大模型（LLM）仅用于回答生成，不影响嵌入配置
        3. 嵌入模型与回答模型完全分离，保证嵌入的稳定性和离线可用性
        
        :return: dict - 嵌入模型配置字典
        """
        # 始终使用本地的 bge-small-zh-v1.5 离线模型进行向量嵌入
        # 嵌入模型与大模型完全分离，大模型仅用于回答生成
        return {
            "model_name": cls.DEFAULT_EMBEDDING_MODEL,
            "api_base": None,
            "api_key": "",
            "api_format": "local",  # 使用本地模型格式
            "dimension": cls.DEFAULT_EMBEDDING_DIMENSION,
            "use_local": True,
            "local_model_path": cls.DEFAULT_LOCAL_MODEL_PATH
        }
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """
        获取大模型配置
        
        【配置逻辑】
        如果未配置大模型，返回 None（表示使用默认嵌入模型进行检索，不使用 LLM 生成）
        
        :return: dict or None - 大模型配置字典，未配置时返回 None
        """
        if not cls.is_llm_configured():
            return None
        
        return {
            "model_name": os.getenv("LLM_MODEL_NAME", cls.DEFAULT_LLM_MODEL),
            "api_base": os.getenv("LLM_API_BASE", cls.DEFAULT_LLM_API_BASE),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "temperature": float(os.getenv("TEMPERATURE", cls.DEFAULT_TEMPERATURE))
        }
    
    @classmethod
    def get_chroma_config(cls) -> dict:
        """
        获取向量数据库配置
        
        :return: dict - ChromaDB 配置字典
        """
        return {
            "db_path": os.getenv("CHROMA_DB_PATH", cls.DEFAULT_CHROMA_DB_PATH),
            "ollama_base": os.getenv("OLLAMA_API_BASE", cls.DEFAULT_OLLAMA_API_BASE)
        }
    
    @classmethod
    def get_document_config(cls) -> dict:
        """
        获取文档处理配置
        
        :return: dict - 文档处理配置字典
        """
        return {
            "chunk_size": int(os.getenv("CHUNK_SIZE", cls.DEFAULT_CHUNK_SIZE)),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", cls.DEFAULT_CHUNK_OVERLAP)),
            "top_k": int(os.getenv("TOP_K", cls.DEFAULT_TOP_K))
        }
    
    @classmethod
    def print_config_status(cls):
        """
        打印当前配置状态（用于调试）
        """
        print("=" * 60)
        print("系统配置状态")
        print("=" * 60)
        
        if cls.is_llm_configured():
            print("[OK] 大模型已配置")
            llm_config = cls.get_llm_config()
            print(f"  - 模型：{llm_config['model_name']}")
            print(f"  - API 地址：{llm_config['api_base']}")
        else:
            print("[未配置] 大模型未配置")
            if cls.is_local_model_available():
                print(f"  - 将使用本地离线模型：{cls.DEFAULT_EMBEDDING_MODEL}")
            else:
                print(f"  - 本地模型不可用，请下载 {cls.DEFAULT_EMBEDDING_MODEL} 到 {cls.DEFAULT_LOCAL_MODEL_PATH}")
        
        embedding_config = cls.get_embedding_config()
        print(f"\n嵌入模型配置:")
        print(f"  - 模型：{embedding_config['model_name']}")
        print(f"  - 使用本地模型：{'是' if embedding_config.get('use_local') else '否'}")
        if embedding_config.get('use_local'):
            print(f"  - 本地模型路径：{embedding_config.get('local_model_path')}")
            print(f"  - 本地模型可用：{'是' if cls.is_local_model_available() else '否'}")
        else:
            print(f"  - API 地址：{embedding_config.get('api_base', 'N/A')}")
            print(f"  - API 格式：{embedding_config.get('api_format', 'N/A')}")
        print(f"  - 向量维度：{embedding_config['dimension']}")
        
        print("=" * 60)


# 全局配置实例
config = Config()
