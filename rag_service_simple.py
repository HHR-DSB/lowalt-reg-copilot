"""
简化版 RAG 服务 - 只包含最基础的 RAG 功能
Query 进去，Answer 出来
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# FastAPI 应用
app = FastAPI(
    title="Simple RAG Service",
    description="简化版 RAG 服务 - 基础问答功能",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
MILVUS_API_URL = "http://localhost:8000"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("请在 .env 文件中设置 DASHSCOPE_API_KEY")

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 请求模型
class SimpleQueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SimpleQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Simple RAG"}

# 从 Milvus 检索相关文档
def retrieve_documents(query: str, top_k: int = 5) -> List[Dict]:
    """从向量数据库检索相关文档"""
    try:
        # 调用 Milvus API
        response = requests.post(
            f"{MILVUS_API_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "search_type": "vector"  # 只用向量搜索
            }
        )

        if response.status_code == 200:
            results = response.json()
            return results.get("results", [])
        else:
            logger.error(f"检索失败: {response.text}")
            return []

    except Exception as e:
        logger.error(f"检索出错: {e}")
        return []

# 生成答案
def generate_answer(question: str, contexts: List[str]) -> str:
    """基于检索结果生成答案"""

    # 如果没有检索到相关内容
    if not contexts:
        prompt = f"请回答以下问题：\n\n{question}"
    else:
        # 构建带上下文的提示
        context_text = "\n\n".join([f"参考 {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        prompt = f"""基于以下参考信息回答问题。如果参考信息不足，可以基于你的知识补充。

参考信息：
{context_text}

问题：{question}

请提供准确、清晰的回答。"""

    try:
        # 调用 LLM11111
        completion = llm_client.chat.completions.create(
            model="qwen3.8-27b",
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        return f"抱歉，生成答案时发生错误：{str(e)}"

# 主要的 RAG 端点
@app.post("/chat", response_model=SimpleQueryResponse)
async def simple_rag_chat(request: SimpleQueryRequest):
    """
    简单 RAG 对话接口
    1. 接收问题
    2. 检索相关文档
    3. 生成答案
    4. 返回结果
    """

    try:
        logger.info(f"收到问题: {request.question}")

        # 1. 检索相关文档
        retrieved_docs = retrieve_documents(request.question, request.top_k)
        logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")

        # 2. 提取文本内容
        contexts = []
        sources = []
        for doc in retrieved_docs:
            text = doc.get("text", "")
            if text:
                contexts.append(text)
                # 记录来源
                source = doc.get("source", "Unknown")
                section = doc.get("section", "")
                sources.append(f"{source} - {section} (相关度: {doc.get('score', 0):.2f})")

        # 3. 生成答案
        answer = generate_answer(request.question, contexts)

        # 4. 返回结果
        return SimpleQueryResponse(
            question=request.question,
            answer=answer,
            sources=sources[:3]  # 只返回前3个最相关的来源
        )

    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 快速测试端点
@app.get("/test")
async def test_endpoint():
    """快速测试端点"""
    test_response = await simple_rag_chat(
        SimpleQueryRequest(
            question="什么是人工智能？",
            top_k=3
        )
    )
    return test_response

# 启动服务
if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("启动简化版 RAG 服务")
    print("=" * 50)
    print(f"API 文档: http://localhost:8002/docs")
    print(f"测试端点: http://localhost:8002/test")
    print("=" * 50)

    # 开发模式：如果需要自动重载，请使用命令行：
    # uvicorn rag_service_simple:app --reload --port 8002

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002  # 使用不同的端口避免冲突
    )