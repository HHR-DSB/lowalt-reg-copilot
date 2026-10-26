"""消融四象限实验：量化"查询扩展"与"Rerank 精排"的独立贡献

评估指标：检索命中（top1/top3 是否命中预期文档+章节）
—— 生成质量不在本实验范围（那是 RAGAS 的领域），消融的是检索模块

四象限：
  A 基线：无扩展 + 无精排（裸 hybrid 检索）
  B 仅精排：无扩展 + 精排（单查询 + 召回放大 + rerank）
  C 仅扩展：扩展 + 无精排（多查询融合，纯 hybrid 序 = 验收时的原状态）
  D 全开：扩展 + 精排（当前生产配置）

变量控制：
  - 意图识别仅 C/D 象限执行（domain 是查询扩展的输入；A/B 无扩展，
    且领域过滤已注释不参与检索逻辑）——跳过可省 18 次 LLM 调用
  - 每象限 top_k=5 恒定
"""
import re
import time

from lowalt_rag_service import RAGPipeline, Domain

# (qid, 问题, 预期 source 关键词正则, 预期 section 关键词列表)
# 预期来自阶段 1 验收的参考要点（人工判读结论）
QUESTIONS = [
    ("Q1", "无人驾驶航空器按性能指标分为哪几类？", "条例", ["第二章", "第六章"]),
    ("Q2", "操控小型、中型、大型民用无人驾驶航空器的人员应当具备什么条件？", "条例", ["第二章"]),
    ("Q3", "民用无人驾驶航空器操控员执照管理有哪些规定？", "CCAR", ["B 章"]),
    ("Q4", "民用无人驾驶航空器实名登记有什么要求？", "CCAR", ["C 章"]),
    ("Q5", "什么是低空空域协同管理？", "规定", ["第一章", "第二章"]),
    ("Q6", "四川省低空空域协同管理试点依据哪个批复方案？", "规定|运行规则", ["第一章", "前言"]),
    ("Q7", "相向飞行时应当如何避让？", "规定|运行规则", ["第五章"]),
    ("Q8", "低空目视通道在什么时间开放？", "运行规则|规定", ["第三章"]),
    ("Q9", "2018年颁发的《四川省低空空域协同管理运行规则（暂行）》现在还有效吗？", "规定", ["第九章"]),
]

QUADRANTS = [
    ("A 基线", False, False),
    ("B 仅精排", False, True),
    ("C 仅扩展", True, False),
    ("D 全开", True, True),
]


def is_hit(item, src_pat, sec_pats):
    """命中判定：source 匹配 且 section 匹配任一关键词"""
    if not item:
        return False
    return bool(re.search(src_pat, item["source"])) and \
           any(re.search(s, item["section"]) for s in sec_pats)


def main():
    summary = {}   # {象限: {"top1": n, "top3": n}}

    for quad_name, use_expansion, use_rerank in QUADRANTS:
        print("=" * 72)
        print(f"象限 {quad_name}（扩展={'on' if use_expansion else 'off'} 精排={'on' if use_rerank else 'off'}）")
        print("=" * 72)
        top1_n = top3_n = 0

        for qid, q, src_pat, sec_pats in QUESTIONS:
            t0 = time.time()
            try:
                if use_expansion:
                    # C/D：跑真实链路（意图 → 扩展），domain 传导给扩展 prompt
                    domain, conf, _ = RAGPipeline.detect_intent_with_llm(q)
                    queries = RAGPipeline.expand_query_with_llm(q, domain)
                else:
                    # A/B：无扩展，单查询直查（domain 不参与检索，跳过意图识别）
                    domain, queries = Domain.REGULATION, [q]

                results = RAGPipeline.search_knowledge_with_filter(
                    queries, domain, top_k=5, enable_rerank=use_rerank)

                top1 = results[0] if results else None
                h1 = is_hit(top1, src_pat, sec_pats)
                h3 = any(is_hit(r, src_pat, sec_pats) for r in results[:3])
                top1_n += h1
                top3_n += h3

                # 紧凑行：命中标记 + top1 详情（rerank 象限附 hybrid 对照）
                if top1:
                    extra = ""
                    if "hybrid_score" in top1:
                        extra = f"（hybrid {top1['hybrid_score']:.3f}→rerank {top1['score']:.3f}）"
                    print(f"  {qid} top1={'√' if h1 else '×'} top3={'√' if h3 else '×'}"
                          f" ｜{top1['source'][:10]}…｜{top1['section']}｜{top1['score']:.3f}{extra}"
                          f"（{time.time()-t0:.0f}s）")
                else:
                    print(f"  {qid} 无检索结果（{time.time()-t0:.0f}s）")
            except Exception as e:
                print(f"  {qid} 异常: {e}")

        summary[quad_name] = {"top1": top1_n, "top3": top3_n}
        print(f"  → 小计: top1 {top1_n}/9, top3 {top3_n}/9\n")

    # ---- 汇总表 ----
    print("=" * 72)
    print("消融四象限汇总（检索命中，n=9）")
    print("=" * 72)
    print(f"{'象限':<12}{'扩展':<6}{'精排':<6}{'top1命中':<10}{'top3命中':<10}")
    for quad_name, use_expansion, use_rerank in QUADRANTS:
        s = summary[quad_name]
        print(f"{quad_name:<12}{'on' if use_expansion else 'off':<6}"
              f"{'on' if use_rerank else 'off':<6}{s['top1']}/9{'':<5}{s['top3']}/9")


if __name__ == "__main__":
    main()
