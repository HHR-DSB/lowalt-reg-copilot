"""LowAlt-Reg Copilot 阶段 1 验收：10 问测试

覆盖结构：
  - 四份文档各 2 问（Q1-Q8）：验证各文档的检索与引用
  - 跨文档效力题（Q9）：验证引用可查证设计的核心价值（废止关系发现）
  - 域外旁路题（Q10）：验证意图识别 + 旁路逻辑

自动判定：domain 归属、旁路是否跳过检索、sources 是否非空
人工判读：回答内容与参考的真实对应（第四信号，防"答案正确≠RAG生效"）
"""
import time
import requests

BASE = "http://localhost:8004"

# (编号, 问题, 预期行为, 预期要点——供人工判读的参考答案)
# 预期行为: "regulation" = 域内走 RAG；"bypass" = 域外旁路直答
QUESTIONS = [
    ("Q1",  "无人驾驶航空器按性能指标分为哪几类？",
     "regulation", "条例·第二章：微型/轻型/小型/中型/大型 五类"),
    ("Q2",  "操控小型、中型、大型民用无人驾驶航空器的人员应当具备什么条件？",
     "regulation", "条例·第二章第十六条：执照/培训/无不良记录等条件"),
    ("Q3",  "民用无人驾驶航空器操控员执照管理有哪些规定？",
     "regulation", "CCAR-92·B 章操控员管理"),
    ("Q4",  "民用无人驾驶航空器实名登记有什么要求？",
     "regulation", "CCAR-92·C 章登记管理"),
    ("Q5",  "什么是低空空域协同管理？",
     "regulation", "川规定·第一/二条：成都空管协调委授权下的统筹配置、运行管理和服务保障"),
    ("Q6",  "四川省低空空域协同管理试点依据哪个批复方案？",
     "regulation", "川规定·第一条：国家空管委《四川省低空空域协同管理试点方案》批复"),
    ("Q7",  "相向飞行时应当如何避让？",
     "regulation", "川运行规则·第五章：各自向右避让（已知排序不精案例，复测）"),
    ("Q8",  "低空目视通道在什么时间开放？",
     "regulation", "第二十二条：昼间开放"),
    ("Q9",  "2018年颁发的《四川省低空空域协同管理运行规则（暂行）》现在还有效吗？",
     "regulation", "跨文档效力题：川规定·第九章附则的废止条款"),
    ("Q10", "红烧肉怎么做？",
     "bypass", "域外：应走旁路，sources 为空，直接 LLM 回答"),
]


def ask(question, timeout=180):
    r = requests.post(f"{BASE}/chat", json={"question": question}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main():
    stats = {"pass": 0, "warn": 0, "fail": 0}
    details = []

    for qid, question, expect, ref in QUESTIONS:
        print("=" * 70)
        print(f"{qid}  {question}")
        print(f"预期: {expect} ｜ 参考要点: {ref}")
        print("-" * 70)

        try:
            t0 = time.time()
            r = ask(question)
            dt = time.time() - t0
        except Exception as e:
            print(f"  ✗ 请求失败: {e}")
            stats["fail"] += 1
            details.append((qid, "FAIL", str(e)[:60]))
            continue

        domain = r["domain"]
        sources = r["sources"]
        expanded = r.get("expanded_queries")

        # ---- 自动判定 ----
        checks = []
        if expect == "bypass":
            checks.append(("domain=general", domain == "general"))
            checks.append(("sources为空", len(sources) == 0))
            checks.append(("无扩展", expanded is None))
        else:
            checks.append(("domain=regulation", domain == "regulation"))
            checks.append(("sources非空", len(sources) > 0))

        ok = all(c[1] for c in checks)
        mark = "PASS" if ok else "WARN"
        stats["pass" if ok else "warn"] += 1
        details.append((qid, mark, f"domain={domain}"))

        # f-string 嵌套同引号是 Python 3.12+ 语法，3.9 要先在表达式外拼好
        status = " ｜ ".join(n + ("✓" if v else "✗") for n, v in checks)
        print(f"  [{mark}] {status}  （{dt:.1f}s）")

        if expanded:
            print(f"  扩展({len(expanded)-1}条): {expanded[1:]}")
        for s in sources:
            print(f"  src: {s['source']}｜{s['section']}｜{s['score']:.4f}")
        # 回答全文留给人工判读（第四信号）
        print(f"  回答:\n{r['answer']}")
        print()

    # ---- 汇总 ----
    print("=" * 70)
    print("验收汇总")
    print("=" * 70)
    print(f"自动判定: PASS {stats['pass']} / WARN {stats['warn']} / FAIL {stats['fail']}（共 {len(QUESTIONS)} 问）")
    for qid, mark, note in details:
        print(f"  {qid}: {mark}  {note}")
    print()
    print("人工判读清单（第四信号：回答内容与参考要点的真实对应）——请逐题核对上方回答：")
    for qid, q, expect, ref in QUESTIONS:
        if expect == "regulation":
            print(f"  {qid}: {ref}")


if __name__ == "__main__":
    main()
