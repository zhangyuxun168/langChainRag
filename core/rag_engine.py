# RAG引擎模块 - 负责文档检索和大模型生成回答
import os
import re
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

from .vector_store import vector_store_manager

load_dotenv()

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.doc', '.md']

def extract_file_names_from_question(question: str) -> List[str]:
    """
    从用户问题中提取文件名
    
    【功能】使用正则表达式从问题中提取可能的文件名
    【参数】question: 用户问题文本
    【返回】提取到的文件名列表
    
    【匹配规则】
    1. 匹配包含扩展名的文件名（如 "合同.pdf"）
    2. 匹配中文文件名（如 "测试文档.docx"）
    3. 匹配英文文件名（如 "test.txt"）
    """
    file_names = []
    
    # 匹配文件名模式：文件名.扩展名
    # 支持中文、英文、数字和下划线
    pattern = r'([\u4e00-\u9fa5a-zA-Z0-9_]+)\.(txt|pdf|docx|doc|md)'
    matches = re.findall(pattern, question)
    
    for match in matches:
        file_name = f"{match[0]}.{match[1]}"
        file_names.append(file_name)
    
    return list(set(file_names))  # 去重

# ========== 提示词模板 ==========
DEFAULT_PROMPT_TEMPLATE = """
你是一个专业的问答助手，请根据提供的参考文档回答用户的问题。

参考文档：
{context}

用户问题：
{question}

请严格按照以下规则回答：
1. 必须基于提供的参考文档内容进行回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"未找到相关信息"
3. 回答要简洁明了，不要冗长
4. 如果有多个相关文档，可以综合多个文档的内容进行回答
5. 请用中文回答
"""

# 【调用者】backend/main.py - query()接口、get_config()接口、update_config()接口
# 【功能】RAG引擎核心类：管理大模型配置、执行RAG检索和回答生成
# 【字段】top_k: 检索返回的文档数量(默认3)；temperature: 温度参数(0-1)；
#         llm_api_base: API地址；llm_api_key: API密钥；llm_model_name: 模型名称；
#         prompt_template: 提示词模板
# 【调用】core/vector_store.py: VectorStoreManager.similarity_search()
# 【调用】requests: HTTP请求调用大模型API
class RAGEngine:
    
    # 【调用者】模块加载时自动调用（创建全局实例rag_engine）
    # 【功能】初始化RAG引擎配置，从环境变量读取参数
    # 【配置来源】环境变量 > .env文件 > 默认值
    def __init__(self):
        self.top_k: int = int(os.getenv("TOP_K", 3))
        self.temperature: float = float(os.getenv("TEMPERATURE", 0.7))
        self.llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1").rstrip('/')
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct")
        
        self.prompt_template = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    # 【调用者】backend/main.py的query()接口（RAG引擎主入口）
    # 【功能】执行RAG查询：向量检索→格式化文档→构建提示词→调用大模型→返回回答
    # 【流程】VectorStoreManager.similarity_search()→format_docs()→PromptTemplate.format()→requests.post()
    # 【错误处理】空问题/集合不存在/无相关文档/API失败
    # 【特殊逻辑】如果未配置大模型，直接返回检索到的文档内容
    def query(self, question: str) -> Dict[str, Any]:
        try:
            if not question or not question.strip():
                return {
                    "answer": "请输入有效的问题",
                    "sources": [],
                    "success": True
                }
            
            try:
                docs: List[Document] = vector_store_manager.similarity_search(
                    question, 
                    k=self.top_k,
                    use_local_ollama=True
                )
            except Exception as e:
                if "Collection" in str(e) and "does not exist" in str(e):
                    return {
                        "answer": "抱歉，当前没有已上传的文档。请先在后台管理页面上传文档后再进行查询。",
                        "sources": [],
                        "success": True
                    }
                raise
            
            if not docs or len(docs) == 0:
                return {
                    "answer": "抱歉，在已上传的文档中没有找到相关信息。请尝试更换查询关键词或上传更多相关文档。",
                    "sources": [],
                    "success": True
                }
            
            # 获取文档来源列表
            source_files: List[str] = [doc.metadata.get("source", "unknown") for doc in docs]
            
            # 【完全匹配文件名逻辑】从用户问题中提取文件名，只返回完全匹配的文件
            # 如果用户问题中提到了具体的文件名，则只显示这些文件
            extracted_files = extract_file_names_from_question(question)
            
            if extracted_files:
                # 过滤出与问题中提到的文件名完全匹配的文档
                matched_files = []
                for source in source_files:
                    # 获取文件名（去掉路径）
                    source_filename = os.path.basename(source)
                    # 检查是否与提取的文件名完全匹配（不区分大小写）
                    if any(extracted.lower() == source_filename.lower() for extracted in extracted_files):
                        matched_files.append(source)
                
                if matched_files:
                    # 只保留完全匹配的文件
                    source_files = matched_files
                    print(f"【RAG调试】问题中提到的文件: {extracted_files}")
                    print(f"【RAG调试】完全匹配的文件: {source_files}")
                else:
                    # 没有找到完全匹配的文件，保持原来源列表
                    print(f"【RAG调试】未找到完全匹配的文件，问题中提到: {extracted_files}")
            
            # 定义已知的嵌入模型列表（这些模型不能用于聊天）
            embedding_models = ["bge-m3", "bge-small-zh", "all-minilm", "text-embedding", "gte-"]
            
            # 检查配置的模型是否是嵌入模型（不能用于聊天）
            is_embedding_model = any(
                model_name.lower() in self.llm_model_name.lower() 
                for model_name in embedding_models
            )
            
            # 检查是否配置了有效的大模型
            # 判断条件：API地址包含localhost:11434，且模型名称包含嵌入模型关键词，则视为未配置有效大模型
            is_local_ollama = self.llm_api_base.startswith("http://localhost:11434") or \
                             self.llm_api_base.startswith("http://127.0.0.1:11434")
            
            # 如果是本地Ollama且配置的是嵌入模型，视为未配置有效大模型
            if is_local_ollama and is_embedding_model:
                is_llm_configured = False
            else:
                # 其他情况：检查是否为默认配置
                is_llm_configured = not (
                    self.llm_api_base == "http://localhost:11434" and 
                    self.llm_model_name == "qwen2.5:7b-instruct"
                )
            
            # 【调试日志】打印配置信息
            print(f"【RAG调试】llm_api_base: {self.llm_api_base}")
            print(f"【RAG调试】llm_model_name: {self.llm_model_name}")
            print(f"【RAG调试】is_embedding_model: {is_embedding_model}")
            print(f"【RAG调试】is_local_ollama: {is_local_ollama}")
            print(f"【RAG调试】is_llm_configured: {is_llm_configured}")
            print(f"【RAG调试】条件判断(not is_llm_configured or is_embedding_model): {not is_llm_configured or is_embedding_model}")
            
            if not is_llm_configured or is_embedding_model:
                # 未配置大模型，或配置的是嵌入模型，使用本地 bge-small-zh-v1.5 模型生成回答
                context: str = self.format_docs(docs)
                
                # 使用简单的基于规则的方法生成回答
                # 将问题和上下文结合，生成一个简洁的回答
                answer = self._generate_simple_answer(question, context)
                
                return {
                    "answer": answer,
                    "sources": list(set(source_files)),
                    "success": True,
                    "llm_configured": False,
                    "embedding_model": "bge-small-zh-v1.5",
                    "answer_type": "local"
                }
            
            # 已配置大模型，调用大模型生成回答
            context: str = self.format_docs(docs)
            prompt: str = self.prompt_template.format(context=context, question=question)
            
            # 判断是否为 Ollama 模型
            # 条件：模型名称包含":"（如 qwen2.5:7b-instruct）或 API地址是本地 Ollama 默认地址
            is_ollama = ":" in self.llm_model_name or \
                       (self.llm_api_base.startswith("http://localhost:11434") or \
                        self.llm_api_base.startswith("http://127.0.0.1:11434"))
            
            if is_ollama:
                # Ollama 固定使用 /api/chat 路径，不受配置的 llm_api_base 影响
                llm_url = "http://localhost:11434/api/chat"
                payload = {
                    "model": self.llm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": self.temperature
                }
                headers = {"Content-Type": "application/json"}
            else:
                llm_url = f"{self.llm_api_base}/chat/completions"
                payload = {
                    "model": self.llm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature
                }
                headers = {"Content-Type": "application/json"}
                if self.llm_api_key:
                    headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            try:
                response = requests.post(llm_url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                result = response.json()
                
                answer: str = result["message"]["content"] if is_ollama else result["choices"][0]["message"]["content"]
                
                return {
                    "answer": answer,
                    "sources": list(set(source_files)),
                    "success": True,
                    "llm_configured": True,
                    "llm_model_name": self.llm_model_name,
                    "embedding_model": "bge-small-zh-v1.5"
                }
            except Exception as llm_error:
                # 大模型调用失败，回退到本地模型
                print(f"【RAG调试】大模型调用失败，回退到本地模型: {str(llm_error)}")
                context: str = self.format_docs(docs)
                answer = self._generate_simple_answer(question, context)
                
                return {
                    "answer": answer,
                    "sources": list(set(source_files)),
                    "success": True,
                    "llm_configured": False,
                    "embedding_model": "bge-small-zh-v1.5",
                    "answer_type": "local_fallback",
                    "fallback_reason": f"大模型调用失败: {str(llm_error)}"
                }
        except Exception as e:
            return {
                "answer": f"查询失败: {str(e)}",
                "sources": [],
                "success": False,
                "llm_configured": False,
                "embedding_model": "bge-small-zh-v1.5"
            }
    
    # 【调用者】本类内部query()方法
    # 【功能】使用简单基于规则的方法生成回答（当没有配置大模型时使用）
    # 【原理】从上下文中提取与问题相关的内容，进行简单的摘要和组织
    # 【输入】question: 用户问题；context: 检索到的文档内容
    # 【输出】生成的回答字符串
    def _generate_simple_answer(self, question: str, context: str) -> str:
        if not context or not context.strip():
            return "未找到相关信息"
        
        # 简单的问答逻辑：在上下文中查找问题关键词相关的内容
        # 将上下文按段落分割
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        # 找出与问题最相关的段落
        question_keywords = set(question.lower().replace('?', '').replace('？', '').split())
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            # 检查段落是否包含问题中的关键词
            matched_keywords = [kw for kw in question_keywords if kw in paragraph_lower]
            if matched_keywords:
                relevant_paragraphs.append((len(matched_keywords), paragraph))
        
        # 按匹配关键词数量排序
        relevant_paragraphs.sort(key=lambda x: -x[0])
        
        if not relevant_paragraphs:
            # 如果没有找到直接相关的内容，返回前几个段落
            answer = "\n\n".join(paragraphs[:2])
        else:
            # 返回最相关的段落
            answer = "\n\n".join([p[1] for p in relevant_paragraphs[:2]])
        
        # 添加前缀说明
        answer = f"根据文档内容，关于您的问题 \"{question}\"，相关信息如下：\n\n{answer}\n\n【说明】当前使用本地嵌入模型进行回答，如需更智能的总结，请配置大语言模型。"
        
        return answer
    
    # 【调用者】backend/main.py的update_config()接口
    # 【功能】运行时动态更新RAG引擎配置参数（支持部分更新）
    # 【验证】top_k必须>0，temperature必须在0-1之间
    def update_config(self,
                      top_k: Optional[int] = None,
                      temperature: Optional[float] = None,
                      llm_api_base: Optional[str] = None,
                      llm_api_key: Optional[str] = None,
                      llm_model_name: Optional[str] = None) -> None:
        # 更新检索文档数量
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k必须大于0")
            self.top_k = top_k
        
        # 更新温度参数
        if temperature is not None:
            if temperature < 0 or temperature > 1:
                raise ValueError("temperature必须在0-1之间")
            self.temperature = temperature
        
        # 更新API地址
        if llm_api_base is not None:
            self.llm_api_base = llm_api_base.rstrip('/')
        
        # 更新API密钥
        # 特殊标记 __KEEP__ 表示保持原有密钥不变（用于前端输入框为空但不想修改密钥的情况）
        if llm_api_key is not None and llm_api_key != "__KEEP__":
            self.llm_api_key = llm_api_key
        
        # 更新模型名称
        if llm_model_name is not None:
            self.llm_model_name = llm_model_name
    
    # 【调用者】backend/main.py的get_config()接口
    # 【功能】获取当前RAG引擎配置，对API密钥进行脱敏处理
    # 【安全】API密钥只返回前6位+***，保护敏感信息
    def get_config(self) -> Dict[str, Any]:
        # 判断是否配置了大模型
        is_llm_configured = not (
            self.llm_api_base == "http://localhost:11434" and 
            self.llm_model_name == "qwen2.5:7b-instruct"
        )
        
        return {
            "top_k": self.top_k,
            "temperature": self.temperature,
            "llm_api_base": self.llm_api_base,
            "llm_api_key": self.mask_api_key(self.llm_api_key),
            "llm_model_name": self.llm_model_name,
            "llm_configured": is_llm_configured,
            "embedding_model_name": "bge-small-zh-v1.5",
            "embedding_dimension": 512,
            "embedding_type": "local"
        }

    # 【调用者】get_config()
    # 【功能】对API密钥进行脱敏处理，只显示前6位+***
    # 【安全】保护敏感信息，避免API密钥泄露
    @staticmethod
    def mask_api_key(api_key: str) -> str:
        if not api_key:
            return ""
        if len(api_key) <= 6:
            return api_key
        return api_key[:6] + "***"
    

    
    # 【调用者】RAGEngine.query()
    # 【功能】将文档列表格式化为上下文字符串，用于构建提示词
    # 【格式】--- [来源] ---\n内容，多个文档用空行分隔
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"文档{i+1}")
            content = doc.page_content
            formatted_doc = f"--- [{source}] ---\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    # 【调用者】backend/main.py的get_models()接口
    # 【功能】获取支持的线上模型列表（预留接口，当前返回空列表）
    def get_online_models(self) -> List[str]:
        return []
    
    # 【调用者】core/document_processor.py - 在文档向量化前调用
    # 【功能】对完整文档内容进行总结，生成文档摘要
    # 【输入】document_text: 文档完整内容；filename: 原始文件名（用于调试日志）
    # 【输出】总结文本字符串
    # 【逻辑】如果配置了有效大模型，调用大模型总结；否则使用简单规则生成摘要
    def summarize_document(self, document_text: str, filename: str = "") -> str:
        print(f"【文档总结】开始总结文档: {filename}")
        print(f"【文档总结】文档长度: {len(document_text)} 字符")
        
        # 如果文档内容过短，直接返回原文作为总结
        if len(document_text) < 200:
            print(f"【文档总结】文档内容过短，直接返回原文")
            return document_text
        
        # 定义已知的嵌入模型列表（这些模型不能用于聊天）
        embedding_models = ["bge-m3", "bge-small-zh", "all-minilm", "text-embedding", "gte-"]
        
        # 检查配置的模型是否是嵌入模型（不能用于聊天）
        is_embedding_model = any(
            model_name.lower() in self.llm_model_name.lower() 
            for model_name in embedding_models
        )
        
        # 检查是否为本地Ollama
        is_local_ollama = self.llm_api_base.startswith("http://localhost:11434") or \
                         self.llm_api_base.startswith("http://127.0.0.1:11434")
        
        # 判断是否配置了有效的大模型
        is_llm_configured = not (
            self.llm_api_base == "http://localhost:11434" and 
            self.llm_model_name == "qwen2.5:7b-instruct"
        )
        
        # 如果是本地Ollama且配置的是嵌入模型，视为未配置有效大模型
        if is_local_ollama and is_embedding_model:
            is_llm_configured = False
        
        if is_llm_configured and not is_embedding_model:
            # 使用配置的大模型进行总结
            print(f"【文档总结】使用配置的大模型: {self.llm_model_name}")
            return self._summarize_with_llm(document_text)
        else:
            # 使用简单规则生成总结
            print(f"【文档总结】使用本地模型生成总结")
            return self._summarize_simple(document_text)
    
    # 【调用者】summarize_document()
    # 【功能】使用配置的大模型生成文档总结
    # 【输入】document_text: 文档完整内容
    # 【输出】总结文本字符串
    def _summarize_with_llm(self, document_text: str) -> str:
        # 构建总结提示词
        summary_prompt = f"""请对以下文档内容进行总结，输出详细的文档摘要：

文档内容：
{document_text[:8000]}

总结要求：
1. 概括文档的主要内容和核心观点
2. 列出文档中的关键数据和结论
3. 保持总结的完整性和准确性
4. 使用中文输出，格式清晰
"""
        
        # 判断是否为 Ollama 模型
        is_ollama = ":" in self.llm_model_name or \
                   (self.llm_api_base.startswith("http://localhost:11434") or \
                    self.llm_api_base.startswith("http://127.0.0.1:11434"))
        
        if is_ollama:
            llm_url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.llm_model_name,
                "messages": [{"role": "user", "content": summary_prompt}],
                "stream": False,
                "temperature": 0.3  # 低温度，保持总结的准确性
            }
            headers = {"Content-Type": "application/json"}
        else:
            llm_url = f"{self.llm_api_base}/chat/completions"
            payload = {
                "model": self.llm_model_name,
                "messages": [{"role": "user", "content": summary_prompt}],
                "temperature": 0.3
            }
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        try:
            response = requests.post(llm_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            summary = result["message"]["content"] if is_ollama else result["choices"][0]["message"]["content"]
            print(f"【文档总结】大模型总结成功，总结长度: {len(summary)} 字符")
            return summary
        except Exception as e:
            print(f"【文档总结】大模型总结失败，回退到简单总结: {str(e)}")
            return self._summarize_simple(document_text)
    
    # 【调用者】summarize_document()、_summarize_with_llm()
    # 【功能】使用简单规则生成文档总结（当没有配置大模型或大模型调用失败时使用）
    # 【输入】document_text: 文档完整内容
    # 【输出】总结文本字符串
    def _summarize_simple(self, document_text: str) -> str:
        # 将文档按段落分割
        paragraphs = [p.strip() for p in document_text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return "文档内容为空"
        
        # 如果段落较少，直接返回所有段落
        if len(paragraphs) <= 3:
            return "\n\n".join(paragraphs)
        
        # 提取关键段落：首段、末段和最长的几个段落
        # 首段通常是引言或摘要
        # 末段通常是结论
        # 最长段落通常包含重要内容
        
        # 按长度排序段落
        sorted_paragraphs = sorted(paragraphs, key=lambda x: len(x), reverse=True)
        
        # 选取关键段落：首段 + 最长的3个段落 + 末段
        key_paragraphs = set()
        key_paragraphs.add(paragraphs[0])  # 首段
        key_paragraphs.add(paragraphs[-1])  # 末段
        
        # 添加最长的段落（排除已添加的）
        for p in sorted_paragraphs:
            if len(key_paragraphs) >= 5:
                break
            if p not in key_paragraphs:
                key_paragraphs.add(p)
        
        # 按原始顺序排列
        result_paragraphs = [p for p in paragraphs if p in key_paragraphs]
        
        summary = "\n\n".join(result_paragraphs)
        
        # 添加总结前缀
        summary = f"【文档摘要】\n\n{summary}\n\n【说明】当前使用本地模型进行总结，如需更智能的总结，请配置大语言模型。"
        
        print(f"【文档总结】简单总结完成，总结长度: {len(summary)} 字符")
        return summary


# 创建RAG引擎实例（全局单例）
rag_engine = RAGEngine()
