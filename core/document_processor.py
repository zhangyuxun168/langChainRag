"""
文档处理器模块
================

负责加载各种格式的文档，并将文档切分成小的文本片段，为后续的向量存储做准备。

模块功能：
---------
1. 支持多种文件格式加载（txt、pdf、docx）
2. 使用递归字符分割器将长文档切分成小片段
3. 为每个片段添加元数据（来源、序号等）

调用关系：
---------
- 被调用方：main.py (upload_file 接口)
- 调用外部：LangChain文档加载器和文本分割器

配置依赖：
---------
- CHUNK_SIZE: 每个文本片段的大小（默认500字符）
- CHUNK_OVERLAP: 相邻片段的重叠字符数（默认50字符）

========================================
调用流程说明（文件上传流程 UPLOAD_FLOW）
========================================

【UPLOAD_FLOW-1】main.py upload_file() 接收文件上传请求
    ↓
【UPLOAD_FLOW-2】DocumentProcessor.process_file() 处理文件（本模块主入口）
    ↓
【UPLOAD_FLOW-3】DocumentProcessor.load_document() 加载文档内容
    ↓
【UPLOAD_FLOW-4】DocumentProcessor.split_documents() 切分文档
    ↓
【UPLOAD_FLOW-5】vector_store_manager.add_documents() 添加到向量数据库
    ↓
【UPLOAD_FLOW-6】VectorStoreManager.get_vector_store() 获取向量存储
    ↓
【UPLOAD_FLOW-7】OllamaEmbeddings.embed_documents() 生成嵌入向量
"""

# ========== 标准库导入 ==========
import os  
"""操作系统模块，用于读取环境变量、处理文件路径和扩展名判断"""

# ========== LangChain相关导入 ==========
from langchain_text_splitters import RecursiveCharacterTextSplitter  
"""递归字符文本分割器，用于将长文档切分成小片段，保持语义完整性"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader  
"""文档加载器集合：
- TextLoader: 加载纯文本文件（.txt）
- PyPDFLoader: 加载PDF文件（.pdf）
- Docx2txtLoader: 加载Word文档（.docx）
"""

from langchain_core.documents import Document  
"""LangChain的Document文档对象类型，表示一段文本及其元数据"""

# ========== 第三方库导入 ==========
from dotenv import load_dotenv  
"""dotenv库，用于从.env文件加载环境变量配置"""

from typing import List  
"""类型提示模块，用于标注变量和函数的类型"""

# 加载环境变量配置
load_dotenv()


class DocumentProcessor:
    """
    文档处理器类
    
    核心职责：
    --------
    - 加载不同格式的文档文件
    - 将长文档切分成语义完整的小片段
    - 为每个片段添加追踪元数据
    
    字段说明：
    --------
    - chunk_size: int - 每个文本片段的目标大小（字符数），默认500
    - chunk_overlap: int - 相邻片段的重叠字符数，用于保持上下文连贯，默认50
    - text_splitter: RecursiveCharacterTextSplitter - 文本分割器实例
    
    调用者：
    -------
    main.py 的 upload_file 接口通过 DocumentProcessor.process_file() 调用
    """
    
    def __init__(self):
        """
        初始化文档处理器
        
        从环境变量读取配置：
        - CHUNK_SIZE: 每个文本片段的大小（默认500字符）
        - CHUNK_OVERLAP: 相邻片段的重叠字符数（默认50字符）
        
        初始化流程：
        ----------
        1. 读取环境变量配置
        2. 创建RecursiveCharacterTextSplitter实例
        3. 配置分隔符优先级：段落 > 行 > 空格 > 字符
        """
        # 每个文本片段的目标大小（字符数），从环境变量读取，默认500
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
        
        # 相邻片段的重叠字符数，用于保持上下文连贯性，默认50
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))
        
        # 文本分割器实例，延迟初始化
        self.text_splitter: RecursiveCharacterTextSplitter
        
        # 初始化递归字符文本分割器
        # 分隔符优先级（从高到低）：
        #   "\n\n" - 优先按段落分隔（两个换行符）
        #   "\n"   - 其次按单行分隔
        #   " "    - 再按空格分隔
        #   ""     - 最后按字符分隔
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,          # 每个片段的目标大小（字符数）
            chunk_overlap=self.chunk_overlap,    # 相邻片段之间的重叠字符数
            length_function=len,                 # 使用字符串长度作为计量单位
            separators=["\n\n", "\n", " ", ""]   # 分隔符列表，按优先级排序
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        加载指定路径的文档（根据扩展名自动选择加载器）
        
        【UPLOAD_FLOW-3】被 DocumentProcessor.process_file() 调用
        
        参数：
        ------
        file_path: str - 文件的完整路径（绝对路径或相对路径）
        
        返回：
        ------
        List[Document] - Document对象列表，每个Document包含：
            - page_content: str - 文档文本内容
            - metadata: dict - 元数据（默认包含来源路径等信息）
        
        支持的文件格式：
        --------------
        - .txt: 纯文本文件（UTF-8编码）
        - .pdf: PDF文档（使用PyPDF解析）
        - .docx: Microsoft Word文档
        
        调用者：
        -------
        DocumentProcessor.process_file() 内部调用
        
        异常处理：
        --------
        - ValueError: 不支持的文件格式
        - RuntimeError: 加载过程中的其他错误（文件不存在、权限问题等）
        """
        # 获取文件扩展名并转为小写，用于判断文件类型
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # 根据文件类型选择对应的加载器
            if ext == ".txt":
                # 文本文件：使用UTF-8编码读取，确保中文正确处理
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == ".pdf":
                # PDF文件：使用PyPDF加载器提取文本内容
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                # Word文件：使用docx2txt加载器提取文本
                loader = Docx2txtLoader(file_path)
            else:
                # 不支持的文件格式，抛出异常
                raise ValueError(f"不支持的文件格式: {ext}")
            
            # 使用加载器读取文档，返回Document对象列表
            documents: List[Document] = loader.load()
            return documents
        except ValueError:
            raise  # 重新抛出格式错误
        except Exception as e:
            # 其他异常统一包装为RuntimeError
            raise RuntimeError(f"加载文档失败: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表切分成更小的文本片段
        
        【UPLOAD_FLOW-4】被 DocumentProcessor.process_file() 调用
        
        参数：
        ------
        documents: List[Document] - 待切分的Document对象列表
        
        返回：
        ------
        List[Document] - 切分后的Document对象列表，每个片段大小约为chunk_size
        
        切分策略：
        --------
        使用RecursiveCharacterTextSplitter递归分割：
        1. 优先按段落分隔（两个换行符）
        2. 其次按行分隔（单个换行符）
        3. 再按空格分隔
        4. 最后按字符分隔
        
        调用者：
        -------
        DocumentProcessor.process_file() 内部调用
        
        设计意图：
        --------
        将长文档切分成小片段可以：
        - 提高向量检索的准确性
        - 控制提示词长度，避免超出模型上下文限制
        - 保持语义完整性，优先按自然边界分割
        """
        # 使用预配置的分片器对文档进行切分
        split_docs: List[Document] = self.text_splitter.split_documents(documents)
        return split_docs
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        处理单个文件的完整流程（主入口方法）
        
        【UPLOAD_FLOW-2】被 main.py upload_file() 调用
        
        参数：
        ------
        file_path: str - 文件的完整路径
        
        返回：
        ------
        List[Document] - 切分并添加元数据后的Document列表
        
        处理流程：
        --------
        【UPLOAD_FLOW-2.1】加载文档 → 调用 load_document()
        【UPLOAD_FLOW-2.2】切分文档 → 调用 split_documents()  
        【UPLOAD_FLOW-2.3】添加元数据 → 为每个片段添加chunk_id和source
        
        元数据说明：
        ----------
        - chunk_id: int - 该片段在原文档中的序号，从0开始
        - source: str - 原始文件名（不含路径），用于查询时显示来源
        
        调用者：
        -------
        main.py 的 upload_file 接口
        
        示例：
        ------
        processor = DocumentProcessor()
        docs = processor.process_file("/data/documents/report.pdf")
        # 返回多个Document对象，每个包含约500字符的文本片段
        """
        # ========== 【UPLOAD_FLOW-2.1】加载文档 ==========
        # 根据文件扩展名选择合适的加载器读取文件内容
        # 【调用 UPLOAD_FLOW-3】DocumentProcessor.load_document(file_path)
        documents: List[Document] = self.load_document(file_path)
        
        # ========== 【UPLOAD_FLOW-2.2】将长文档切分成小片段 ==========
        # 使用递归字符分割器将文档切分为多个小片段
        # 【调用 UPLOAD_FLOW-4】DocumentProcessor.split_documents(documents)
        split_docs: List[Document] = self.split_documents(documents)
        
        # ========== 【UPLOAD_FLOW-2.3】为每个片段添加元数据信息 ==========
        for i, doc in enumerate(split_docs):
            # chunk_id: 该片段在原文档中的序号，用于追踪位置和排序
            doc.metadata["chunk_id"] = i
            # source: 原始文件名（不含路径），便于查询时显示来源
            doc.metadata["source"] = os.path.basename(file_path)
        
        return split_docs
    
    def get_supported_extensions(self) -> List[str]:
        """
        获取支持的文件格式列表
        
        返回：
        ------
        List[str] - 支持的文件扩展名列表，包括：.txt, .pdf, .docx
        
        调用者：
        -------
        main.py 的 upload_file 接口（用于文件格式校验）
        """
        return [".txt", ".pdf", ".docx"]