"""
LowAlt-Reg Copilot MVP：低空规章 RAG 问答服务
端口 :8004

复制自 rag_service.py（:8001 增强版），六处法规域改造：
1. TARGET_COLLECTION 指向 lowalt_reg_kb（通过 :8000 的 collection 参数，见 milvus_api.QueryRequest）
2. 意图分类四域（law/finance/ai/general）→ 二元（regulation/general），旁路判断更干净
3. 查询扩展的领域背景定向：口语 → 规章术语
4. 检索 payload 全部携带 collection 参数
5. ★核心：参考片段带出处（source｜section），模型才能回答"依据《XX》第X章"——
   入库时建的 section 章节标签，在这里兑现价值
6. 生成 Prompt 强化诚实条款：不得虚构条文编号（法规问答的生命线）

原 :8001 服务保持不动，作为阶段 2 消融实验的对照组。
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
    print("✓ LowAlt-Reg RAG 服务启动")
    yield
    print("✓ LowAlt-Reg RAG 服务关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="LowAlt-Reg Copilot",
    description="低空空域规章智能问答（基于 lowalt_reg_kb 知识库）",
    version="1.0.0",
    lifespan=lifespan
)

# 配置
MILVUS_API_URL = "http://localhost:8000"  # Milvus API 地址
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 本服务的目标集合：milvus_api.py 已支持 /query 传 collection 参数（默认 None 走旧库）。
# 定义成模块级常量而不是散落在各个函数里 —— 集合名是"部署配置"，改一处即全局生效
#
# v2 = 条文边界切块库（10-27 对比实验后定为生产配置）：
#   vs 滑窗版（lowalt_reg_kb）：faithfulness 0.749→0.781、precision 0.890→0.909、
#   recall 持平——两升一平无下降 + Q7 满分质变（完整条文在场→引用逐字可查证）
#   + 产品价值：引用粒度=检索粒度（问"第X条"直接命中整条）。老库保留作对照组
TARGET_COLLECTION = "lowalt_reg_kb_v2"

# ---- Rerank 精排配置（两阶段检索的第二阶段）----
# 注意端点不是 OpenAI 兼容格式，是 DashScope 原生 API —— rerank 不是 chat 任务，没有 messages 结构
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = "qwen3.7-text-rerank"       # 免费额度（侦察时验证可用）
RECALL_MULTIPLIER = 4                      # 召回放大倍数：top_k=5 → 每查询召回 20 条
                                           # 依据实测：Q7 的目标切片 hybrid 排位第 7，
                                           # top_k=5 直接漏掉——放大候选池是 rerank 生效的前提

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ========== 数据模型 ==========

class Domain(str, Enum):
    """领域枚举 —— 从四域二元化

    原 rag_service.py 是 law/finance/ai/general 四域（为行业报告知识库设计）。
    低空规章库是单一垂直域，四分类是历史包袱：
    - 分类越多，边界 case 越多，意图识别的 token 开销越大
    - 旁路判断只需要一个信息："这问题在不在我的知识域内"
    二元化后，general + 高置信 → 跳过检索 的旁路逻辑更干净
    """
    REGULATION = "regulation"   # 低空规章域内
    GENERAL = "general"         # 域外

class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., description="用户问题")
    use_rag: bool = Field(True, description="是否使用 RAG")
    top_k: int = Field(5, description="检索文档数量")
    search_type: str = Field("hybrid", description="检索类型")
    stream: bool = Field(False, description="是否流式输出")
    # 消融结论落地（10-26 实验）：查询扩展在朴素分数融合下单独使用为负贡献
    # （C 象限 top1 3/9 < 基线 4/9），rerank 以原始问题为唯一基准打分后无增量收益
    # ——生产默认关闭。显式传 True 可重新启用（消融实验复现用）
    enable_query_expansion: bool = Field(False, description="是否启用查询扩展（默认关：消融实验结论）")
    enable_intent_detection: bool = Field(True, description="是否启用意图识别")
    # Rerank 开关——与上面两个开关并列，消融实验的四象限就此齐了：
    #   扩展 on/off × 精排 on/off，用同一批验收问题对比，量化每个模块的独立贡献
    enable_rerank: bool = Field(True, description="是否启用 Rerank 精排（两阶段检索）")

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
        使用 LLM 进行意图识别（二元：低空规章域内 / 域外）
        返回: (领域, 置信度, 理由)
        """
        try:
            prompt = f"""请分析以下问题是否属于"低空空域规章"领域，并给出判断理由。

问题：{question}

判断标准：
1. regulation（低空规章）：与无人驾驶航空器、低空空域、飞行活动管理、
空域协同管理、目视飞行规则、民航规章条款等相关的问题
2. general（通用）：与低空规章无关的一般性问题
（例如：编程技术、天气常识、闲聊、其他行业问题）

请按以下JSON格式回答：
{{
    "domain": "领域名称（regulation/general）",
    "confidence": 置信度（0.0-1.0的小数）,
    "reason": "判断理由"
}}

只返回JSON，不要其他内容。"""

            messages = [
                {"role": "system", "content": "你是一个意图识别专家，精通领域分类。"},
                {"role": "user", "content": prompt}
            ]

            completion = llm_client.chat.completions.create(
                model="qwen3.8-max",
                messages=messages,
                temperature=0.1,  # 低温度使结果更确定——分类任务要稳定不要创意
                max_tokens=200
            )

            # 解析 JSON 响应
            response_text = completion.choices[0].message.content.strip()
            # 提取 JSON 部分：正则匹配第一个 {...}（防模型输出前后缀噪声）
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                domain_str = result.get("domain", "general")
                confidence = float(result.get("confidence", 0.5))
                reason = result.get("reason", "")

                # 映射到 Domain 枚举。
                # get() 的默认值是安全网：模型偶尔输出旧类目（如 "law"）或
                # 自创类目时，统一回落到 GENERAL——宁可错杀（走旁路不检索）
                # 不可错放（域外问题硬检索一通胡拼）
                domain_map = {
                    "regulation": Domain.REGULATION,
                    "general": Domain.GENERAL
                }
                domain = domain_map.get(domain_str, Domain.GENERAL)

                return domain, confidence, reason

        except Exception as e:
            print(f"意图识别失败: {e}")

        # 异常兜底：general + 0.5，让主流程的旁路分支自己决定走哪条路
        return Domain.GENERAL, 0.5, "默认归类"

    @staticmethod
    def expand_query_with_llm(question: str, domain: Domain) -> List[str]:
        """
        使用 LLM 进行查询扩展。
        领域背景定向为"口语 → 规章术语"改写——
        消融实验已证明这是检索主力（top 分数 0.305 → 0.889）
        """
        try:
            domain_context = {
                Domain.REGULATION: (
                    "低空空域规章领域。扩展时把口语表达转写为规章术语"
                    "（如'无人机能飞多高'→'微型/轻型无人驾驶航空器真高限制'），"
                    "并从不同角度扩展：国家条例、民航规章（CCAR-92）、四川试点规则"
                ),
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
                model="qwen3.8-max",
                messages=messages,
                temperature=0.7,   # 扩展要多样性，高温度——和分类任务相反
                max_tokens=200
            )

            # 解析生成的查询
            expanded = completion.choices[0].message.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip() and len(q.strip()) > 5][:3]

            # 返回原始查询 + 扩展查询（原始永远排第一，保底）
            return [question] + expanded

        except Exception as e:
            print(f"查询扩展失败: {e}")
            return [question]

    @staticmethod
    def rerank_results(query: str, documents: List[str], top_n: int = 5) -> Optional[List[Dict]]:
        """两阶段检索的精排：用"原始问题"对候选文档统一重打分

        为什么打分基准必须用原始问题而不是扩展查询：
        - 扩展查询是为"召回多样性"服务的（每个查询负责一个角度）
        - 精排要回答的问题是"哪个文档最能回答用户真正想问的"——基准必须唯一且忠于用户
        若用扩展查询各自打分再合并，就退回到了"多路各自为政"的老问题

        为什么放在服务层而不是 milvus_api 层：
        - milvus_api 每次只收到一个查询，看不到"多查询融合"的全局池
        - 精排必须对融合后的并集统一进行——这是编排层的职责

        返回: [{"index": documents 的序号, "relevance_score": 相关性分}]，失败返回 None
        （失败不抛异常——调用方负责降级，一路精排挂了不该瘫掉整条链路）
        """
        try:
            resp = requests.post(
                RERANK_URL,
                headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
                json={
                    "model": RERANK_MODEL,
                    "input": {"query": query, "documents": documents},
                    "parameters": {"return_documents": False, "top_n": top_n},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["output"]["results"]
            print(f"rerank 响应异常: {resp.status_code} {resp.text[:150]}")
        except Exception as e:
            print(f"rerank 调用失败: {e}")
        return None

    @staticmethod
    def search_knowledge_with_filter(queries: List[str], domain: Domain,
                                    top_k: int = 5, search_type: str = "hybrid",
                                    enable_rerank: bool = True):
        """
        两阶段检索：召回（粗排）→ 精排（rerank）

        阶段一 召回：多查询 × top_k×RECALL_MULTIPLIER，按 Milvus 主键去重求并集 —— 宽进
        阶段二 精排：原始问题（queries[0]）对并集统一重打分，取 top_k —— 严出

        为什么召回要放大（这是本改造的关键参数）：
        embedding 检索存在语义鸿沟（口语"相向飞行" vs 条文"对头相遇"），
        实测目标切片 hybrid 排位第 7 而 top_k=5 —— 直接漏掉。
        宽召回把候选池放大到 20+，rerank 才有"从沙里挑金"的余地；
        没有放大的 rerank 是无米之炊（池里没有的永远排不上来）。

        降级设计：rerank 失败 → 回落到 hybrid 分数排序（本函数的旧排序逻辑），
        功能从"更准"退化为"和从前一样"，但不中断服务

        注：requests 是同步调用，在 async 路由里会阻塞事件循环——
        原代码如此，MVP 不动（单用户场景感知不到），并发场景再换 httpx.AsyncClient
        """
        all_results = []
        seen_ids = set()  # 多查询间按 Milvus 主键去重

        # 精排开启时召回放大（给 rerank 备粮）；关闭时维持原 top_k（纯召回模式）
        recall_k = top_k * RECALL_MULTIPLIER if enable_rerank else top_k

        for query in queries:
            try:
                request_data = {
                    "query": query,
                    "top_k": recall_k,                # ★ 放大后的召回数
                    "search_type": search_type,
                    "vector_weight": 0.7,
                    "collection": TARGET_COLLECTION
                }

                response = requests.post(
                    f"{MILVUS_API_URL}/query",
                    json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    for item in result.get("results", []):
                        if item["id"] not in seen_ids:
                            seen_ids.add(item["id"])
                            all_results.append(item)

            except Exception as e:
                print(f"检索查询 '{query}' 失败: {e}")

        # 按 hybrid 分数排序——这个排序同时是 rerank 失败时的降级序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # ---- 阶段二：精排 ----
        if enable_rerank and all_results:
            docs = [r["text"] for r in all_results]
            # queries[0] 是原始问题（expand_query_with_llm 保证原句永远排第一）
            reranked = RAGPipeline.rerank_results(queries[0], docs, top_n=top_k)
            if reranked:
                final = []
                for r_ in reranked:
                    item = dict(all_results[r_["index"]])       # index 指回候选池
                    item["hybrid_score"] = item.get("score", 0) # 留档：召回阶段的旧分数
                    item["score"] = r_["relevance_score"]       # score 语义统一为"最终排序依据"
                    final.append(item)
                return final
            print("⚠ rerank 失败，降级为 hybrid 分数排序")

        # 未开 rerank 或降级路径：走原逻辑（截断 top_k）
        return all_results[:top_k]

    @staticmethod
    def search_knowledge(query: str, top_k: int = 5, search_type: str = "hybrid"):
        """从法规知识库检索（单查询，保留原方法结构）"""
        try:
            response = requests.post(
                f"{MILVUS_API_URL}/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "search_type": search_type,
                    "vector_weight": 0.7,
                    "collection": TARGET_COLLECTION   # ★ 同上：指向法规库
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
        """构建上下文并提取引用片段

        ★ 本次改造的核心：参考块从
            [参考1]\n正文...
        变为
            [参考1]（出处：民用无人驾驶航空器运行安全管理规则（CCAR-92部）｜B 章 操控员管理）\n正文...

        为什么这是核心：原来模型只能回答"根据参考1"，用户无法核对；
        带上出处后模型能标注"依据 CCAR-92 B 章"——引用从"内部编号"
        升级为"可查证的法律引用"。入库时辛苦建的 section 章节标签，
        到这里才真正兑现价值（数据 → 标签 → 用户体验，缺一环都白建）
        """
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
            # fragment = 完整 chunk（10-27 取消 300 字截断）：
            # 截断是"上下文超预算"的手段，但 5 块 × 500 字 ≈ 2500 字对
            # 128k 窗口是零头——本没有预算问题。300 字截断是滑窗时代的节俭
            # 设计（chunk 400 砍到 300 代价可忽略），条文块时代 chunk 最长 500
            # 字，且完整条文是引用可查证性的载体（Q1 案例：五类定义条文超
            # 300 字被砍 → context_recall=0）。技术债形态：前提变了设计没跟上
            fragment = text

            source = result.get('source', '未知来源')
            section = result.get('section', '')
            # 出处随参考块一起喂给模型（见 docstring：这是本次改造核心）
            context_parts.append(f"[参考{i}]（出处：{source}｜{section}）\n{fragment}")

            # 保存引用片段信息（API 响应里的 referenced_fragments 字段）
            fragments.append({
                "index": i,
                "source": source,
                "section": section,
                "score": result.get('score', 0),
                # 完整片段（取消 200 字二次截断）：RAGAS 的 contexts 正是用
                # 这个字段组装——这里砍一刀，评估就拿残料算，分数失真
                "fragment": fragment
            })

        return "\n\n".join(context_parts), fragments

    @staticmethod
    def build_context(search_results):
        """构建上下文（兼容旧方法，内部走带出处的新实现）"""
        context, _ = RAGPipeline.build_context_with_fragments(search_results)
        return context

    @staticmethod
    def build_prompt(question: str, context: str = ""):
        """构建提示词（旁路直答场景：无参考资料时使用）"""
        if context:
            prompt = f"""你是低空空域规章咨询助手，请基于以下参考内容回答用户问题。
如果参考内容不足以回答问题，请诚实地说明，并提供你所知道的相关信息。

参考内容：
{context}

用户问题：{question}

请提供准确、专业的回答："""
        else:
            prompt = f"""你是低空空域规章咨询助手，请回答以下问题：

{question}

请提供准确、专业的回答："""

        return prompt

    @staticmethod
    def generate_answer(prompt: str, stream: bool = False):
        """生成回答"""
        try:
            messages = [
                {"role": "system", "content": "你是低空空域规章咨询助手，专业、严谨，引用条文时注明出处。"},
                {"role": "user", "content": prompt}
            ]

            if stream:
                completion = llm_client.chat.completions.create(
                    model="qwen3.8-max",
                    messages=messages,
                    stream=True
                )
                return completion
            else:
                completion = llm_client.chat.completions.create(
                    model="qwen3.8-max",
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
        "service": "LowAlt-Reg Copilot",
        "version": "1.0.0",
        "collection": TARGET_COLLECTION,
        "endpoints": {
            "/chat": "智能对话（意图识别+查询扩展+带出处引用）",
            "/chat/stream": "流式对话（SSE，暂为旧管线）",
            "/chat/with-history": "多轮对话（暂为旧管线）",
            "/health": "健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        response = requests.get(f"{MILVUS_API_URL}/health")
        milvus_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        milvus_status = "disconnected"

    return {
        "status": "healthy",
        "milvus_api": milvus_status,
        "collection": TARGET_COLLECTION,
        "llm": "ready"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能对话接口（低空规章增强版）"""

    pipeline = RAGPipeline()

    # 1. 意图识别（如果启用）
    domain = Domain.GENERAL
    intent_reason = ""
    if request.enable_intent_detection:
        domain, confidence, intent_reason = pipeline.detect_intent_with_llm(request.question)
        print(f"识别领域: {domain.value} (置信度: {confidence:.2f})")

        # 旁路逻辑：general 且高置信 → 域外问题，跳过检索直接 LLM 回答。
        # （沿用 rag_service.py 中已修复的 >= 0.5 方向：高置信域外不浪费检索，
        #  低置信说明拿不准，仍然检索兜底）
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

        # 4. 检索法规知识库（两阶段：宽召回 + 精排）
        # 统一走融合管线（原来单查询走 search_knowledge 的分支已合并）：
        # 单查询 queries=[原问题] 同样受益于"召回放大 + rerank"，
        # 且消融组合（扩展 on/off × 精排 on/off）四种情况行为一致，实验才可比
        search_results = pipeline.search_knowledge_with_filter(
            queries=expanded_queries,
            domain=domain,
            top_k=request.top_k,
            search_type=request.search_type,
            enable_rerank=request.enable_rerank
        )
        search_results_dict = {"results": search_results}

        # 5. 构建上下文（带出处）和提取片段
        context, fragments = pipeline.build_context_with_fragments(search_results_dict)

        # 6. 构建增强提示词
        #    诚实条款针对法规域强化：条文编号必须来自参考资料，不得凭记忆虚构——
        #    法规问答的引用可信度是生命线（宁可承认"参考资料未注明"也不编造）
        if context:
            enhanced_prompt = f"""你是低空空域规章咨询助手。

用户问题：{request.question}

领域背景：{intent_reason}

参考资料：
{context}

请基于参考资料回答用户问题。在回答中：
1. 引用依据时标注出处（如"依据《无人驾驶航空器飞行管理暂行条例》第二章"），
   每段参考资料开头的（出处：...）就是出处信息
2. 条文编号、章节号必须来自参考资料原文，不得凭记忆虚构；
   参考资料未注明的内容，明确说"参考资料中未注明"
3. 如果参考资料与问题无关或不足以回答，诚实说明，不要强行拼凑
4. 保持专业、准确、有条理

回答："""
        else:
            enhanced_prompt = pipeline.build_prompt(request.question)

        # 7. 生成回答
        answer = pipeline.generate_answer(enhanced_prompt, stream=request.stream)

        # 8. 整理来源（前 3 条给 sources 字段）
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
    """流式对话接口（SSE）。

    注意：此端点仍是旧管线（单查询 + 简单 prompt，无意图/扩展/出处标注），
    MVP 主打 /chat；流式与增强管线的合并留给阶段 2
    """
    from fastapi.responses import StreamingResponse
    import asyncio

    async def generate():
        pipeline = RAGPipeline()

        if request.use_rag:
            # search_knowledge 已带 TARGET_COLLECTION，检索的仍是法规库
            search_results = pipeline.search_knowledge(
                query=request.question,
                top_k=request.top_k,
                search_type=request.search_type
            )
            context = pipeline.build_context(search_results)
            prompt = pipeline.build_prompt(request.question, context)

            yield f"data: {json.dumps({'type': 'sources', 'data': search_results}, ensure_ascii=False)}\n\n"
        else:
            prompt = pipeline.build_prompt(request.question)

        stream = pipeline.generate_answer(prompt, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'content', 'data': content}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ========== 对话历史管理（可选）==========

class ConversationManager:
    """对话管理器（内存版，服务重启即失——MVP 够用，持久化留给后续）"""

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
    """带历史记录的对话（暂为旧管线，同 /chat/stream 的说明）"""

    history = conversation_manager.get_history(session_id)

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

    messages = [
        {"role": "system", "content": "你是低空空域规章咨询助手，专业、严谨。"}
    ]

    messages.extend(history)

    if context:
        user_prompt = f"基于以下参考内容回答问题：\n\n{context}\n\n问题：{request.question}"
    else:
        user_prompt = request.question

    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = llm_client.chat.completions.create(
            model="qwen3.8-max",
            messages=messages,
            stream=False
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"生成回答时出错：{str(e)}"

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
    print("LowAlt-Reg Copilot（低空规章 RAG 问答）")
    print(f"目标集合: {TARGET_COLLECTION}")
    print(f"依赖服务: {MILVUS_API_URL} (需先启动 milvus_api.py)")
    print("="*60)
    print(f"服务地址: http://localhost:8004")
    print(f"API 文档: http://localhost:8004/docs")
    print("="*60)

    uvicorn.run(
        "lowalt_rag_service:app",     # 模块名必须与本文件名一致，否则 reload 找不到 app
        host="0.0.0.0",
        port=8004,                    # 8001 是原增强版（对照组），8002 是 Simple 版，本服务 8004
        reload=True,
        # reload_excludes 是血泪教训（2026-10-26 实录）：
        # volumes/ 是 MinIO 数据卷（海量小文件 + Milvus 写入时动态增删），
        # StatReload 扫描到"扫描中途消失的路径"会 FileNotFoundError 崩掉
        # reloader 主进程 —— worker 变孤儿继续服务旧代码：
        # 服务活着、请求正常、日志照打，唯独改代码永远不生效，症状极具迷惑性。
        # 数据目录必须从 reload 监测中排除
        reload_excludes=["volumes*", "ocr_cache*", "milvus.doc*", "*.log"],
    )

if __name__ == "__main__":
    main()
