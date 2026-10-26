"""
RAG 服务层
功能：整合向量检索和 LLM，提供完整的 RAG 对话服务
支持：查询扩展、意图识别、领域过滤、参考片段展示
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from contextlib import asynccontextmanager
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from enum import Enum

# 加载环境变量
load_dotenv()

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✓ RAG 服务启动")
    yield
    print("✓ RAG 服务关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 对话服务",
    description="基于检索增强生成的智能对话服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置
MILVUS_API_URL = "http://localhost:8000"  # Milvus API 地址
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ========== 数据模型 ==========

class Domain(str, Enum):
    """领域枚举"""
    LAW = "law"
    FINANCE = "finance"
    AI = "ai"
    GENERAL = "general"

class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., description="用户问题")
    use_rag: bool = Field(True, description="是否使用 RAG")
    top_k: int = Field(5, description="检索文档数量")
    search_type: str = Field("hybrid", description="检索类型")
    stream: bool = Field(False, description="是否流式输出")
    enable_query_expansion: bool = Field(True, description="是否启用查询扩展")
    enable_intent_detection: bool = Field(True, description="是否启用意图识别")

class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(..., description="回答内容")
    sources: List[Dict] = Field([], description="参考来源")
    search_results: Optional[List[Dict]] = Field(None, description="检索结果")
    domain: Optional[str] = Field(None, description="识别的领域")
    expanded_queries: Optional[List[str]] = Field(None, description="扩展的查询")
    referenced_fragments: Optional[List[Dict]] = Field(None, description="引用的文本片段")

class RAGPipeline:
    """RAG 处理流水线"""

    @staticmethod
    def detect_intent_with_llm(question: str) -> Tuple[Domain, float, str]:
        """
        使用 LLM 进行意图识别
        返回: (领域, 置信度, 理由)
        """
        try:
            prompt = f"""请分析以下问题属于哪个领域，并给出判断理由。

问题：{question}

可选领域：
1. law（法律）：法律法规、合同、诉讼、权利义务等相关问题
2. finance（金融）：银行、投资、股票、理财、贷款等金融相关问题
3. ai（人工智能）：AI技术、机器学习、深度学习、算法模型等相关问题
4. general（通用）：不属于以上特定领域的一般性问题

请按以下JSON格式回答：
{{
    "domain": "领域名称（law/finance/ai/general）",
    "confidence": 置信度（0.0-1.0的小数）,
    "reason": "判断理由"
}}

只返回JSON，不要其他内容。"""

            messages = [
                {"role": "system", "content": "你是一个意图识别专家，精通领域分类。"},
                {"role": "user", "content": prompt}
            ]

            completion = llm_client.chat.completions.create(
                model="qwen3.8-27b",
                messages=messages,
                temperature=0.1,  # 低温度使结果更确定
                max_tokens=200
            )

            # 解析 JSON 响应
            response_text = completion.choices[0].message.content.strip()
            # 提取 JSON 部分
            import json
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                domain_str = result.get("domain", "general")
                confidence = float(result.get("confidence", 0.5))
                reason = result.get("reason", "")

                # 映射到 Domain 枚举
                domain_map = {
                    "law": Domain.LAW,
                    "finance": Domain.FINANCE,
                    "ai": Domain.AI,
                    "general": Domain.GENERAL
                }
                domain = domain_map.get(domain_str, Domain.GENERAL)

                return domain, confidence, reason

        except Exception as e:
            print(f"意图识别失败: {e}")

        return Domain.GENERAL, 0.5, "默认归类"

    @staticmethod
    def expand_query_with_llm(question: str, domain: Domain) -> List[str]:
        """
        使用 LLM 进行查询扩展
        """
        try:
            domain_context = {
                Domain.LAW: "法律领域，关注法规条文、案例、法律程序等",
                Domain.FINANCE: "金融领域，关注市场、产品、风险、监管等",
                Domain.AI: "人工智能领域，关注技术原理、应用、发展趋势等",
                Domain.GENERAL: "通用领域"
            }

            prompt = f"""请为以下问题生成3个相关的扩展查询，帮助更全面地检索信息。

原始问题：{question}
领域背景：{domain_context.get(domain, "通用领域")}

要求：
1. 从不同角度扩展原问题
2. 保持与原问题的相关性
3. 每个扩展查询独立一行
4. 直接输出查询内容，不要编号

扩展查询："""

            messages = [
                {"role": "system", "content": "你是查询扩展专家，擅长从多角度分析问题。"},
                {"role": "user", "content": prompt}
            ]

            completion = llm_client.chat.completions.create(
                model="qwen3.8-27b",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )

            # 解析生成的查询
            expanded = completion.choices[0].message.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip() and len(q.strip()) > 5][:3]

            # 返回原始查询 + 扩展查询
            return [question] + expanded

        except Exception as e:
            print(f"查询扩展失败: {e}")
            return [question]

    @staticmethod
    def search_knowledge_with_filter(queries: List[str], domain: Domain,
                                    top_k: int = 5, search_type: str = "hybrid"):
        """
        批量检索多个查询，支持领域过滤
        """
        all_results = []
        seen_ids = set()  # 去重

        for query in queries:
            try:
                # 构建请求
                request_data = {
                    "query": query,
                    "top_k": top_k,
                    "search_type": search_type,
                    "vector_weight": 0.7
                }

                # 根据领域添加过滤条件
                # 注意：这需要你的数据有相应的元数据字段
                # if domain == Domain.LAW:
                #     request_data["filter"] = 'source == "law.md"'
                # elif domain == Domain.FINANCE:
                #     request_data["filter"] = 'source == "finance.md"'
                # elif domain == Domain.AI:
                #     request_data["filter"] = 'source == "ai_dev.md"'

                response = requests.post(
                    f"{MILVUS_API_URL}/query",
                    json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    # 去重并合并结果
                    for item in result.get("results", []):
                        if item["id"] not in seen_ids:
                            seen_ids.add(item["id"])
                            all_results.append(item)

            except Exception as e:
                print(f"检索查询 '{query}' 失败: {e}")

        # 按分数排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]

    @staticmethod
    def search_knowledge(query: str, top_k: int = 5, search_type: str = "hybrid"):
        """从向量数据库检索相关知识（保留原方法）"""
        try:
            response = requests.post(
                f"{MILVUS_API_URL}/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "search_type": search_type,
                    "vector_weight": 0.7
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"检索失败: {response.text}")
                return None
        except Exception as e:
            print(f"检索错误: {e}")
            return None

    @staticmethod
    def build_context_with_fragments(search_results):
        """构建上下文并提取引用片段"""
        if not search_results:
            return "", []

        # 处理不同格式的输入
        if isinstance(search_results, dict) and "results" in search_results:
            results = search_results["results"]
        elif isinstance(search_results, list):
            results = search_results
        else:
            return "", []

        if not results:
            return "", []

        context_parts = []
        fragments = []

        for i, result in enumerate(results[:5], 1):  # 最多使用5个文档
            text = result.get('text', '')
            # 截取相关片段
            fragment_length = 300
            if len(text) > fragment_length:
                # 取前300字作为片段
                fragment = text[:fragment_length] + "..."
            else:
                fragment = text

            context_parts.append(f"[参考{i}]\n{fragment}")

            # 保存引用片段信息
            fragments.append({
                "index": i,
                "source": result.get('source', 'unknown'),
                "section": result.get('section', ''),
                "score": result.get('score', 0),
                "fragment": fragment[:200] + "..." if len(fragment) > 200 else fragment
            })

        return "\n\n".join(context_parts), fragments

    @staticmethod
    def build_context(search_results):
        """构建上下文（兼容旧方法）"""
        context, _ = RAGPipeline.build_context_with_fragments(search_results)
        return context

    @staticmethod
    def build_prompt(question: str, context: str = ""):
        """构建提示词"""
        if context:
            prompt = f"""你是一个专业的助手，请基于以下参考内容回答用户问题。
如果参考内容不足以回答问题，请诚实地说明，并提供你所知道的相关信息。

参考内容：
{context}

用户问题：{question}

请提供准确、专业的回答："""
        else:
            prompt = f"""你是一个专业的助手，请回答以下问题：

{question}

请提供准确、专业的回答："""

        return prompt

    @staticmethod
    def generate_answer(prompt: str, stream: bool = False):
        """生成回答"""
        try:
            messages = [
                {"role": "system", "content": "你是一个专业、严谨的助手。"},
                {"role": "user", "content": prompt}
            ]

            if stream:
                # 流式输出
                completion = llm_client.chat.completions.create(
                    model="qwen3.8-27b",
                    messages=messages,
                    stream=True
                )
                return completion
            else:
                # 非流式输出
                completion = llm_client.chat.completions.create(
                    model="qwen3.8-27b",
                    messages=messages,
                    stream=False
                )
                return completion.choices[0].message.content

        except Exception as e:
            print(f"生成回答错误: {e}")
            return f"抱歉，生成回答时出错：{str(e)}"

# ========== API 路由 ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "RAG 对话服务",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "智能对话",
            "/health": "健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    # 检查 Milvus API
    try:
        response = requests.get(f"{MILVUS_API_URL}/health")
        milvus_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        milvus_status = "disconnected"

    return {
        "status": "healthy",
        "milvus_api": milvus_status,
        "llm": "ready"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能对话接口（增强版）"""

    # RAG 流水线
    pipeline = RAGPipeline()

    # 1. 意图识别（如果启用）
    domain = Domain.GENERAL
    intent_reason = ""
    if request.enable_intent_detection:
        domain, confidence, intent_reason = pipeline.detect_intent_with_llm(request.question)
        print(f"识别领域: {domain.value} (置信度: {confidence:.2f})")

        # 如果是通用领域且置信度低，直接用 LLM 回答
        if domain == Domain.GENERAL and confidence >= 0.5:
            prompt = pipeline.build_prompt(request.question)
            answer = pipeline.generate_answer(prompt, stream=request.stream)

            return ChatResponse(
                answer=answer,
                sources=[],
                search_results=None,
                domain=domain.value,
                expanded_queries=None,
                referenced_fragments=None
            )

    # 2. 判断是否使用 RAG
    if request.use_rag:
        # 3. 查询扩展（如果启用）
        if request.enable_query_expansion:
            expanded_queries = pipeline.expand_query_with_llm(request.question, domain)
            print(f"扩展查询: {expanded_queries}")
        else:
            expanded_queries = [request.question]

        # 4. 检索相关知识（使用扩展查询）
        if len(expanded_queries) > 1:
            # 使用多个查询检索
            search_results = pipeline.search_knowledge_with_filter(
                queries=expanded_queries,
                domain=domain,
                top_k=request.top_k,
                search_type=request.search_type
            )
            # 格式化为标准结果
            search_results_dict = {"results": search_results}
        else:
            # 单个查询
            search_results_dict = pipeline.search_knowledge(
                query=request.question,
                top_k=request.top_k,
                search_type=request.search_type
            )
            search_results = search_results_dict.get("results", []) if search_results_dict else []

        # 5. 构建上下文和提取片段
        context, fragments = pipeline.build_context_with_fragments(search_results_dict)

        # 6. 构建增强提示词
        if context:
            enhanced_prompt = f"""你是一个专业的{domain.value}领域助手。

用户问题：{request.question}

领域背景：{intent_reason}

参考资料：
{context}

请基于参考资料回答用户问题。在回答中：
1. 明确指出使用了哪些参考资料（如[参考1]、[参考2]）
2. 如果参考资料不足，诚实说明并提供你的知识
3. 保持专业、准确、有条理

回答："""
        else:
            enhanced_prompt = pipeline.build_prompt(request.question)

        # 7. 生成回答
        answer = pipeline.generate_answer(enhanced_prompt, stream=request.stream)

        # 8. 整理来源
        sources = []
        for frag in fragments[:3]:
            sources.append({
                "source": frag["source"],
                "section": frag["section"],
                "score": frag["score"]
            })

        return ChatResponse(
            answer=answer,
            sources=sources,
            search_results=search_results[:10] if isinstance(search_results, list) else None,
            domain=domain.value,
            expanded_queries=expanded_queries if len(expanded_queries) > 1 else None,
            referenced_fragments=fragments
        )

    else:
        # 不使用 RAG，直接回答
        prompt = pipeline.build_prompt(request.question)
        answer = pipeline.generate_answer(prompt, stream=request.stream)

        return ChatResponse(
            answer=answer,
            sources=[],
            search_results=None,
            domain=domain.value if request.enable_intent_detection else None,
            expanded_queries=None,
            referenced_fragments=None
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口（SSE）"""
    from fastapi.responses import StreamingResponse
    import asyncio

    async def generate():
        pipeline = RAGPipeline()

        # 检索知识
        if request.use_rag:
            search_results = pipeline.search_knowledge(
                query=request.question,
                top_k=request.top_k,
                search_type=request.search_type
            )
            context = pipeline.build_context(search_results)
            prompt = pipeline.build_prompt(request.question, context)

            # 先发送检索结果
            yield f"data: {json.dumps({'type': 'sources', 'data': search_results}, ensure_ascii=False)}\n\n"
        else:
            prompt = pipeline.build_prompt(request.question)

        # 流式生成回答
        stream = pipeline.generate_answer(prompt, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'content', 'data': content}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ========== 对话历史管理（可选）==========

class ConversationManager:
    """对话管理器"""

    def __init__(self):
        self.conversations = {}

    def create_session(self, session_id: str):
        """创建会话"""
        self.conversations[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in self.conversations:
            self.create_session(session_id)

        self.conversations[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str, limit: int = 10):
        """获取历史"""
        if session_id in self.conversations:
            return self.conversations[session_id][-limit:]
        return []

# 创建全局对话管理器
conversation_manager = ConversationManager()

@app.post("/chat/with-history")
async def chat_with_history(
    session_id: str,
    request: ChatRequest
):
    """带历史记录的对话"""

    # 获取历史
    history = conversation_manager.get_history(session_id)

    # RAG 检索
    pipeline = RAGPipeline()

    if request.use_rag:
        search_results = pipeline.search_knowledge(
            query=request.question,
            top_k=request.top_k,
            search_type=request.search_type
        )
        context = pipeline.build_context(search_results)
    else:
        search_results = None
        context = ""

    # 构建带历史的消息
    messages = [
        {"role": "system", "content": "你是一个专业、严谨的助手。"}
    ]

    # 添加历史
    messages.extend(history)

    # 添加当前问题
    if context:
        user_prompt = f"基于以下参考内容回答问题：\n\n{context}\n\n问题：{request.question}"
    else:
        user_prompt = request.question

    messages.append({"role": "user", "content": user_prompt})

    # 生成回答
    try:
        completion = llm_client.chat.completions.create(
            model="qwen3.8-27b",
            messages=messages,
            stream=False
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"生成回答时出错：{str(e)}"

    # 保存到历史
    conversation_manager.add_message(session_id, "user", request.question)
    conversation_manager.add_message(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": search_results["results"][:3] if search_results else [],
        "history_length": len(conversation_manager.get_history(session_id))
    }

# ========== 启动服务 ==========

def main():
    """启动 RAG 服务"""
    import uvicorn

    print("="*60)
    print("RAG 对话服务")
    print("="*60)
    print(f"服务地址: http://localhost:8001")
    print(f"API 文档: http://localhost:8001/docs")
    print("="*60)

    uvicorn.run(
        "rag_service:app",
        host="0.0.0.0",
        port=8001,  # 使用不同端口
        reload=True
    )

if __name__ == "__main__":
    main()