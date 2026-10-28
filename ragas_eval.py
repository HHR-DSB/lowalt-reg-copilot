"""RAGAS 评估管线（阶段 2 评估体系）——消融实验的自动化/规模化续篇

定位：
  消融四象限（ablation_experiment.py）量化的是"检索命中"（人工标注的分类指标，top1/top3）；
  本脚本用 RAGAS 补齐"生成质量"维度（LLM-as-judge 的连续指标，0~1 分）：
    faithfulness       答案是否忠于检索内容——对应我们的"0 幻觉"诉求的量化版
    answer_relevancy   答案与问题的相关性
    context_precision  检索精确度（相关文档是否排在前面——与消融 top1/top3 交叉验证）
    context_recall     召回覆盖度（ground_truth 要点被检索到多少——Q9 波动性主战场）

两阶段设计（评估工程实践：被评数据与评估过程分离）：
  阶段 A collect：调 :8004 /chat 跑 9 题 → 样本存 ragas_samples.json
                  答案只收集一次；RAGAS 后续调参/换指标重跑，不用重新生成
  阶段 B eval   ：读样本 → 构造 judge → 四指标评估 → 分数表 + 存档

用法：
  python ragas_eval.py collect   # 阶段 A（约 5-10 分钟：9 次意图+检索+生成）
  python ragas_eval.py eval      # 阶段 B（约 10-20 分钟：judge LLM 数百次调用）

设计说明：
  - Q10（域外旁路题）不入 RAGAS：无检索上下文，四个指标都不适用，保持人工判定
  - Q9 是拒答/波动题：指标照常算，但解读要注意——诚实拒答的 faithfulness 高、
    answer_relevancy 天然偏低（拒答文本对"问题本身"的信息量就是低），这不是缺陷是语义
  - ground_truth 全部锚定阶段 1 验收人工判读的参考要点扩写，独立于被评管线——
    避免循环论证（不能用被评对象生成的答案当标准答案）
  - contexts 取 referenced_fragments 的 fragment（实际进 prompt 的 300 字片段），
    而非检索全文——评估"系统真实行为"而不是"理想行为"
"""
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ---- 配置区 ----
BASE = "http://127.0.0.1:8004"      # 用 127.0.0.1 而非 localhost：避开系统代理拦截
SAMPLES_FILE = "ragas_samples.json"
RESULTS_FILE = "ragas_results.json"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# judge 配置：judge 是"阅卷老师"（RAGAS 指标内部靠 LLM 结构化输出打分，
# 对指令遵循有要求，模型不能太弱）
# ⚠ 10-27 晚 qwen3.8-27b 免费额度耗尽（403 FreeTierOnly），应急切 qwen3.8-max
# （max 系与 27b 系额度池独立）——27b 额度恢复后可切回；注意换 judge 后
# 前后分数不可直接对比（判卷标准变了）
JUDGE_MODEL = "qwen3.8-max"
EMBED_MODEL = "qwen3.7-text-embedding-flash"

# requests 全局绕开系统代理（PowerShell 代理拦 127.0.0.1 的坑，Python 同样适用）
SESSION = requests.Session()
SESSION.trust_env = False

# ---- 评估集：9 题（Q1-Q9）----
# ground_truth = 参考答案，context_recall 的锚点。基于 test_lowalt_mvp.py 的
# 参考要点（当初人工判读确认过的）扩写成通顺句子，不额外编造法条细节——
# GT 写错会系统性歪曲 context_recall，宁保守勿杜撰
EVAL_SET = [
    {
        "qid": "Q1",
        "question": "无人驾驶航空器按性能指标分为哪几类？",
        "ground_truth": "依据《无人驾驶航空器飞行管理暂行条例》第二章，无人驾驶航空器按性能指标分为微型、轻型、小型、中型、大型五类。",
    },
    {
        "qid": "Q2",
        "question": "操控小型、中型、大型民用无人驾驶航空器的人员应当具备什么条件？",
        "ground_truth": "依据《无人驾驶航空器飞行管理暂行条例》第二章第十六条，操控小型、中型、大型民用无人驾驶航空器的人员应当具备相应条件，包括取得操控员执照、接受培训、无不良记录等。",
    },
    {
        "qid": "Q3",
        "question": "民用无人驾驶航空器操控员执照管理有哪些规定？",
        "ground_truth": "民用无人驾驶航空器操控员执照管理的规定主要见于《民用无人驾驶航空器运行安全管理规则》（CCAR-92部）B章操控员管理。",
    },
    {
        "qid": "Q4",
        "question": "民用无人驾驶航空器实名登记有什么要求？",
        "ground_truth": "民用无人驾驶航空器实名登记的要求主要规定于《民用无人驾驶航空器运行安全管理规则》（CCAR-92部）C章登记管理，涵盖登记的适用对象与登记要求等内容。",
    },
    {
        "qid": "Q5",
        "question": "什么是低空空域协同管理？",
        "ground_truth": "低空空域协同管理是在成都地区空管协调委授权下，对低空空域资源进行统筹配置、运行管理和服务保障的管理制度，依据《四川省低空空域协同管理与使用规定（暂行）》第一条、第二条。",
    },
    {
        "qid": "Q6",
        "question": "四川省低空空域协同管理试点依据哪个批复方案？",
        "ground_truth": "四川省低空空域协同管理试点依据国家空管委批复的《四川省低空空域协同管理试点方案》，见《四川省低空空域协同管理与使用规定（暂行）》第一条。",
    },
    {
        "qid": "Q7",
        "question": "相向飞行时应当如何避让？",
        "ground_truth": "相向飞行时应当各自向自己的右方避让，依据《四川省低空空域协同管理运行规则（暂行）》第五章。",
    },
    {
        "qid": "Q8",
        "question": "低空目视通道在什么时间开放？",
        "ground_truth": "低空目视通道在昼间开放，依据《四川省低空空域协同管理运行规则（暂行）》第二十二条。",
    },
    {
        "qid": "Q9",
        "question": "2018年颁发的《四川省低空空域协同管理运行规则（暂行）》现在还有效吗？",
        "ground_truth": "《四川省低空空域协同管理运行规则（暂行）》（2018年颁发）的效力状态可依据《四川省低空空域协同管理与使用规定（暂行）》第九章附则中的废止条款判断，新规定施行后旧规则相应废止。",
    },
]


def collect():
    """阶段 A：以服务默认配置（消融最优：扩展 off + rerank on）跑 9 题，
    存 answer + 实际进 prompt 的 contexts。"""
    samples = []
    for item in EVAL_SET:
        qid, q = item["qid"], item["question"]
        print(f"\n[{qid}] {q}")
        t0 = time.time()

        r = None
        for attempt in (1, 2):  # Q5 有 180s 超时前科，重试一次
            try:
                resp = SESSION.post(f"{BASE}/chat", json={"question": q}, timeout=300)
                resp.raise_for_status()
                r = resp.json()
                break
            except Exception as e:
                print(f"  第 {attempt} 次请求失败: {e}")
                time.sleep(3)
        if r is None:
            print(f"  [{qid}] 两次均失败，跳过")
            continue

        # 意图偏差检查：Q1-Q9 全部预期 regulation，判成 general 说明旁路了（无 RAG）
        if r.get("domain") != "regulation":
            print(f"  ⚠ domain={r.get('domain')}（预期 regulation，意图识别偏差）")

        # contexts 必须与实际进 prompt 的格式逐字一致（含出处头）：
        # 生成模型看到的是"[参考i]（出处：source｜section）\n片段"，
        # judge 也必须看到同样的东西——否则答案里的出处断言
        # （"依据《XX》第X章第三十四条"）会被判"上下文无依据"，
        # faithfulness 被系统性低估（Q7 实测 0.273，人工抽查近乎全忠实）
        contexts = [
            f"[参考{i}]（出处：{f['source']}｜{f['section']}）\n{f['fragment']}"
            for i, f in enumerate(r.get("referenced_fragments") or [], 1)
        ]
        srcs = "; ".join(f"{s['source']}/{s['section']}" for s in r.get("sources", []))
        print(f"  {time.time() - t0:.1f}s | {len(contexts)} 个片段 | {srcs}")

        samples.append({
            "qid": qid,
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": r["answer"],
            "reference": item["ground_truth"],
        })

    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n完成：{len(samples)}/{len(EVAL_SET)} 题已存 {SAMPLES_FILE}")


def run_eval():
    """阶段 B：读样本 → RAGAS 四指标。judge = qwen3.8-27b via LangchainLLMWrapper。

    ragas 0.4.3 踩坑记录（两套 metric 体系并行，混用必炸）：
      - evaluate() 主入口只认旧式模块级单例（ragas.metrics._faithfulness.faithfulness，
        过 isinstance(Metric) 检查），llm 通过 evaluate(llm=...) 统一注入
      - collections 下的新类（Faithfulness(llm=...) 构造注入式）服务于新的
        @experiment API，传给 evaluate() 报
        "All metrics must be initialised metric objects"
      - llm 参数类型 BaseRagasLLM | LangchainLLM：LangchainLLMWrapper 可用；
        InstructorLLM 不行（继承 InstructorBaseRagasLLM，是与 BaseRagasLLM
        平行的另一套体系）
    """
    # 延迟导入：collect 阶段不需要 ragas（依赖链大，import 慢）
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings as LCEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    # 旧式模块级单例（下划线模块）——evaluate() 唯一认的 metric 形态
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness

    key = os.getenv("DASHSCOPE_API_KEY")

    # judge LLM：LangchainLLMWrapper 包一个指向 DashScope 兼容端点的 ChatOpenAI。
    # 判卷要稳定，temperature=0（默认 0.7 会给分数引入随机抖动）。
    # ⚠ extra_body 关思考模式：qwen3.8 默认 enable_thinking=true，与 RAGAS 的
    # n=3 采样直接冲突（DashScope 400 "n must be 1 when enable_thinking is true"），
    # 且思考 token 拖慢调用引发 TimeoutError、污染结构化输出解析——判卷不需要思考
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            api_key=key,
            base_url=DASHSCOPE_BASE_URL,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
    )
    # embedding（answer_relevancy 需要）：旧式 metric 调 LangChain 风格的
    # embed_query()——必须 LangchainEmbeddingsWrapper 包 langchain 的
    # OpenAIEmbeddings；ragas.embeddings.OpenAIEmbeddings 是新体系接口，
    # 没有 embed_query，会 AttributeError（第二课：llm 和 embeddings 的
    # 新旧体系坑是同一个）
    embeddings = LangchainEmbeddingsWrapper(
        LCEmbeddings(
            model=EMBED_MODEL,
            api_key=key,
            base_url=DASHSCOPE_BASE_URL,
        )
    )

    # 直接用模块级单例；judge 不在构造时传，而是 evaluate(llm=...) 统一注入
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    with open(SAMPLES_FILE, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"载入 {len(samples)} 个样本，开始评估（judge 调用较多，耐心等）...\n")

    ds = EvaluationDataset.from_list([
        {k: s[k] for k in ("user_input", "retrieved_contexts", "response", "reference")}
        for s in samples
    ])
    result = evaluate(ds, metrics=metrics, llm=judge_llm, embeddings=embeddings)

    # 结果：DataFrame 每行一题，四列指标（NaN = 该题该指标计算失败，看服务端日志）
    df = result.to_pandas()
    cols = [m.name for m in metrics]

    print("\n==== 每题得分 ====")
    header = "qid    " + "".join(f"{c:<22}" for c in cols)
    print(header)
    for _, row in df.iterrows():
        qid = samples[int(row.name)]["qid"] if str(row.name).isdigit() else "?"
        vals = "".join(
            f"{(f'{row[c]:.3f}' if row[c] == row[c] else 'NaN'):<22}" for c in cols
        )
        print(f"{qid:<6} {vals}")

    print("\n==== 指标均值 ====")
    for c in cols:
        print(f"  {c:<20} {df[c].mean():.3f}")

    out = {
        "config": {"judge": JUDGE_MODEL, "embed": EMBED_MODEL, "n": len(samples)},
        "mean": {c: round(float(df[c].mean()), 4) for c in cols},
        "per_question": {
            samples[i]["qid"]: {c: (round(float(df.loc[i, c]), 4) if df.loc[i, c] == df.loc[i, c] else None) for c in cols}
            for i in range(len(samples)) if i in df.index
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存 {RESULTS_FILE}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "collect":
        collect()
    elif mode == "eval":
        run_eval()
    else:
        print(__doc__)
