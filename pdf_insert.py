"""
PDF 入库管线 —— LowAlt-Reg Copilot 阶段 1（低空规章正式入库版）

本文件由"沙箱练习版"升级而来，三处核心变化（建议对着 git 历史或记忆 diff 理解）：
1. 目标集合：pdf_practice（沙箱） → lowalt_reg_kb（正式库，两者互不影响）
2. source：PDF 元数据 title（不可靠） → DOCS 里显式指定的规范名（双重职责，见下）
3. section："切片N"（无语义） → 真实章节名（"A 章 总则"级别，引用可解释）

新增 DRY_RUN 开关：先空跑检查数据质量，再真正灌库（数据工程习惯：先看数据再动手）
"""
import re
from pathlib import Path

import fitz
from bs4 import BeautifulSoup
import jieba.analyse                                   # 坑1：必须显式导入，否则 extract_keywords 内部调 jieba.analyse 会报 AttributeError
from milvus_insert import connect_milvus, generate_embedding, extract_keywords, create_index
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
import milvus_config as config

# ============================ 配置区 ============================

# ---- 切块策略开关（10-27 对比实验完成，条文切块胜出定为生产配置）----
# "sliding" = 滑窗 400+80（阶段 1 配置，对照组；老库 lowalt_reg_kb 保留）
# "article" = 条文边界切块（"第X条"原子单元 + 同章短条合并，生产配置）
# 实验结论（9 题 RAGAS，唯一变量=切块策略）：faithfulness 0.749→0.781、
# precision 0.890→0.909、recall 0.889 持平——两升一平无下降 + Q7 满分质变
# （0.846→1.000：完整条文在场→出处断言逐字可查证）；9 题样本下均值差异温和
# （不具统计显著性），决策依据另含产品价值：引用粒度=检索粒度
CHUNK_STRATEGY = "article"

COLLECTION_NAME = (
    "lowalt_reg_kb" if CHUNK_STRATEGY == "sliding"
    else "lowalt_reg_kb_v2"   # 条文切块实验库
)

DRY_RUN = False                            # True  = 空跑：只做提取/清洗/切块并打印样本
                                         #        不连 Milvus、不调 embedding API（零成本）
                                         # False = 正式入库（空跑检查通过后再切换）

# 待入库文档：(PDF 路径, source 规范名, 提取模式)
#
# 提取模式（同一份语料、三种形态，数据管线的现实）：
#   "pdf"  = PDF 自带文字层，fitz 直接提取
#   "ocr"  = 文字被转成矢量轮廓的 PDF（防拷贝处理，get_text 提不出字符），
#            读 ocr_cache/ 里此前用多模态大模型识别好的缓存（见 read_ocr_cache）
#   "html" = 政府网页存档（浏览器"另存为"的原始 HTML），
#            正文被 <div id="contxt"> 容器天然隔离，bs4 精准提取（见 read_html_text）
#
# source 的双重职责（一个字段两个用途，所以取名要讲究）：
#   ① 幂等判重键 —— 入库前按它查询，已存在则整文档跳过（重跑零重复）
#   ② 引用展示名 —— RAG 问答时 sources 字段展示给用户看的名字
# 因此取"官方全名（+文号）"而不是文件路径：路径以后一挪就变，判重随之失效
# ⚠ source 里不能含英文双引号 " ，否则打碎下方 expr 的 f'source == "..."' 语法（中文括号没事）
DOCS = [
    (r"c:\Users\admin\Desktop\低空资料\民用无人驾驶航空器运行安全管理规则.pdf",
     "民用无人驾驶航空器运行安全管理规则（CCAR-92部）", "pdf"),
    (r"c:\Users\admin\Desktop\低空资料\四川省低空空域协同管理运行规则.pdf",
     "四川省低空空域协同管理运行规则（暂行）", "pdf"),
    (r"c:\Users\admin\Desktop\低空资料\无人驾驶航空器飞行管理暂行条例.pdf",
     "无人驾驶航空器飞行管理暂行条例（国务院令第761号）", "ocr"),
    (r"c:\Users\admin\Desktop\低空资料\四川省低空空域协同管理与使用规定（暂行）.html",
     "四川省低空空域协同管理与使用规定（暂行）", "html"),
]

# 章标题正则：两个分支对应两份文档的编号风格
#   [A-Z]\s*章            → CCAR-92 的 "A 章  总   则"（民航规章按字母编章：A/B/C.../N）
#   第[一二三四五六七八九十百]+章 → 川运行规则的 "第一章  总则"（法规惯例：中文数字编章）
# [^\n]{0,25} 限制标题剩余长度：防止匹配到正文中恰好折行到行首、以"第X章"开头的长句
# （误识别的后果只是 section 标签错，不伤检索正文——可接受的妥协）
CHAPTER_PAT = re.compile(
    r"(?:^|\n)\s*([A-Z]\s*章[^\n]{0,25}|第[一二三四五六七八九十百]+章[^\n]{0,25})"
)

# 条文起点正则（article 策略）：两个分支对应两种编号风格
#   第[一二三四五六七八九十百千零]+条 → 条例/川规定/川运行规则的中文数字条（"第三十四条"）
#   第\d{2,3}\.\d{1,4}\s*条           → CCAR-92 的节号条（实测 "第92.329 条"带空格；
#     小数部分 \d{1,4} 上限 4 位——DRY_RUN 首轮抓到的 bug：92.1001（M 章首条）
#     是 4 位，\d{1,3} 漏匹配导致它跟在前一块尾部跨章混块）
# 关键设计：锚定行首（^|\n）——区分"条文开头"与"正文引用"。
# 法规排版惯例是每条另起一行，而正文句中引用（实测样本"按照本规则第92.329 条确定的…"）
# 不在行首。若不锚定，正文引用处会被误切成段边界，一条法规被引用句拦腰截断。
# 与 CHAPTER_PAT 同款行首锚定策略（包括同样的局限：OCR 断行不可靠时可能漏——靠 DRY_RUN 验证）
ARTICLE_PAT = re.compile(
    r"(?:^|\n)\s*(第[一二三四五六七八九十百千零]+条|第\s*\d{2,3}\.\d{1,4}\s*条)"
)


# ============================ 提取与清洗 ============================

def _is_noise(line: str, header: str = None) -> bool:
    """判断一行是否为待清除的噪声（页码 / 页眉）。返回 True = 丢掉这行

    四种噪声形态（对着语料实测归纳，且各自对应不同的清洗策略）：
      ① "12"       纯数字页码（CCAR-92）        → isdigit()
      ② "— 12 —"   破折号包数字页码（川运行规则） → 正则整行匹配
      ③ 页眉行      CCAR-92 每页首行重复文档全名   → 整行相等判断
      ④ "| ... |"  OCR 把网页表格识别成 markdown 表格行（条例末页的
                   "索引号/发文机关/发文字号"元数据表） → 行首是竖线
    另：空行也当噪声丢掉 —— 滑窗按字符数硬切，空行不携带语义，纯占 400 字的位

    关键原则：只做"整行精确匹配"，不做子串替换。
    正文里出现数字和标题词都太常见，只有整行恰好是页码/页眉才敢删——宁可漏杀，不可误杀。

    ⚠ 页眉为什么用 s == header 整行相等，而不是 startswith 前缀匹配？
    CCAR-92 正文里"民用无人驾驶航空器……"是高频行首主语（几乎每段都有），
    前缀匹配会把大量正文行误杀！页眉行的特征是"整行 strip 后恰好等于标题"，
    正文行则是"标题词 + 更多内容"，整行相等刚好区分两者。

    （调试技巧：如果某天某种页码清不掉，print(repr(line)) 看真实字符——
    破折号可能是 U+2014 "—"、全角"——"或其他码点，正则要照着真实字符写）
    """
    s = line.strip()
    if not s:                                          # 空行
        return True
    if s.isdigit():                                    # ① 纯数字页码
        return True
    if re.match(r"^—\s*\d+\s*—$", s):                 # ② "— 12 —" 式页码
        return True
    if header and s == header:                         # ③ 页眉（整行相等，理由见 docstring）
        return True
    if s.startswith("|"):                              # ④ OCR 生成的 markdown 表格行
        return True
    return False


def read_ocr_cache(pdf_path: str) -> str:
    """读取曲线化 PDF 的 OCR 缓存，按页拼接为全文

    背景：《暂行条例》这份 PDF 的文字被转成了矢量轮廓（防拷贝处理），
    fitz.get_text() 提不出任何字符。此前已用多模态大模型（百炼 qwen3.5-ocr）
    逐页"渲染成图 → 识别"并落盘，缓存布局：
        ocr_cache/<文件名去扩展名>/page_001.txt, page_002.txt, ...

    为什么读缓存而不是每次重新 OCR：
    - OCR 是 API 调用，有成本（钱 + 时间）；缓存落盘后，重跑入库、
      迭代清洗规则、调整切块参数都不再花一分钱
    - 这和 insert 前的 source 判重是同一个思想（幂等）在不同环节的体现：
      判重防"重复入库"，缓存防"重复计算"

    fail-fast 原则：缓存不存在直接抛异常，而不是静默返回空字符串——
    空文本会一路悄悄走到"没有数据需要插入"，问题被掩盖到很晚才暴露
    """
    cache_dir = Path("ocr_cache") / Path(pdf_path).stem
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"OCR 缓存不存在: {cache_dir}\n"
            f"本函数只读缓存不生成 —— 需要先对该 PDF 执行 OCR 并落盘到上述目录"
        )
    # sorted 按文件名排：page_001~page_024 因三位补零，字典序恰好等于页序
    # （如果不补零，page_10 会排在 page_2 前面——文件名排序的经典坑）
    parts = [p.read_text(encoding="utf-8") for p in sorted(cache_dir.glob("page_*.txt"))]
    return "\n".join(parts)


def read_html_text(html_path: str) -> str:
    """从政府网页存档提取正文（浏览器"另存为"保存的原始 HTML）

    为什么 HTML 是最好的源 —— 对比同一份《川规定》的三种形态：
    - 曲线化 PDF：文字被转矢量轮廓，fitz 提不出 → 只能 OCR
    - OCR 结果：混入网页导航，且逐行断字丢字（"制定本"后丢"规定。"）
    - 原始 HTML：正文被 <div id="contxt"> 容器天然隔离，find 一下就拿到纯正文，
      且段落连续不断行 —— 「换到好源」比任何下游清洗技巧都值钱

    contxt 这个容器名怎么找的：打印页面所有 <div id=...> 清单
    （nav/head/ly/copyrights 是导航噪声，contxt 是正文）——
    对接新数据源的第一步永远是先看结构，再写代码
    """
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", id="contxt")               # 政府网站正文容器（侦察结论）
    if body is None:
        raise ValueError(f"未找到正文容器 div#contxt: {html_path}")
    # get_text(separator)：所有标签的文本用换行衔接，即每个 <p>（一条法规）自成一行；
    # &nbsp; 等实体 bs4 自动还原，无需手工处理
    return body.get_text("\n", strip=True)


def extract_text(doc_path: str, header: str = None, mode: str = "pdf") -> str:
    """提取文档全文 + 逐行清洗。返回全文文本（单个字符串）

    （原名 extract_pdf_text，更名为 extract_text —— 它已经是三源提取器，
    函数名不能撒谎：参数叫 doc_path 而非 pdf_path 也是同理）

    mode 选择提取路径（三条路，殊途同归于同一套清洗）：
    - "pdf"：fitz 直接提取文字层（要求 PDF 有文字层）
    - "ocr"：读 ocr_cache/ 缓存（文字被曲线化的 PDF，见 read_ocr_cache）
    - "html"：解析网页存档正文容器（见 read_html_text）

    「多源提取、统一清洗」是数据管线的常见分层：数据源形态各异（变的部分），
    但下游清洗/切块/入库只面对一种格式（不变的部分）—— 把变化的波及范围控制在上游

    header：页眉前缀（传 source 规范名去掉括号部分，如"民用无人驾驶航空器运行安全管理规则"），
    None 则跳过页眉清洗（兼容没有页眉的文档）
    """
    # ---- 先按 mode 拿到"整篇字符串"（html/ocr 两路） ----
    if mode == "html":
        raw = read_html_text(doc_path)
    elif mode == "ocr":
        raw = read_ocr_cache(doc_path)
    else:
        raw = None

    if raw is not None:                                # html / ocr：整篇统一清洗
        lines = raw.split("\n")                        # 与 pdf 路径走完全相同的清洗
        kept = [ln for ln in lines if not _is_noise(ln, header)]
        return "\n".join(kept)

    # ---- pdf 路径：逐页提取逐页清洗 ----
    doc = fitz.open(doc_path)
    parts = []
    for page in doc:
        lines = page.get_text().split("\n")             # 拆行 → 逐行判噪 → 拼回
        kept = [ln for ln in lines if not _is_noise(ln, header)]
        parts.append("\n".join(kept))
    doc.close()
    return "\n".join(parts)                             # 页与页之间保留换行衔接


# ============================ 章节感知（本次升级的核心） ============================

def find_chapters(text: str) -> list:
    """扫描全文章标题 → 返回 [(章名, 该章在全文中的起始偏移)]

    用途：给每个切片打"章节"标签。滑窗切块只认字符数、不认文档结构，
    想知道"第 37 块属于哪一章"，必须先知道"每章从全文第几个字符开始"——
    这就是"位置信息"：切块时丢掉的位置，在这里先记下来。

    ("前言", 0) 是哨兵：公布令（"国务院令第761号……现公布……"）等
    第一章之前的内容统一归入"前言"，同时保证后续区间判断不会越界
    """
    chapters = [("前言", 0)]
    for m in CHAPTER_PAT.finditer(text):
        title = re.sub(r"\s+", " ", m.group(1)).strip()    # "A 章  总   则" → "A 章 总则"（压连续空白）
        chapters.append((title, m.start(1)))               # start(1)：第 1 个分组（章名）的起始偏移
    return chapters


def chapter_of(chapters: list, offset: int) -> str:
    """给定切片的起始偏移，返回它落在哪个章

    逻辑：chapters 本就按偏移升序排列，顺序扫描，
    记住最后一个 start <= offset 的章名即为答案
    （章节数 <20，顺序扫描足够快；写成二分属于过早优化）
    """
    current = chapters[0][0]
    for title, start in chapters:
        if start <= offset:
            current = title
        else:
            break
    return current


# ============================ 切块 ============================

def sliding_window_chunks(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """滑动窗口切块：每块 chunk_size 字，相邻块重叠 overlap 字

    相比沙箱版的变化：返回值从 [块文本] 升级为 [(起始偏移, 块文本)]
    —— 因为 section 章节标签需要知道每块"从全文第几个字符开始"，
    才能用 chapter_of 映射到章节。旧版返回纯文本列表，位置信息丢了

    步长 = chunk_size - overlap：相邻块恰好重叠 overlap 字。
    重叠的作用：关键句恰好在切块边界被腰斩时，相邻两块各持有它的一部分，
    保证至少有一块带足上下文被检索命中（代价：约 20% 存储冗余，可接受）
    """
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunks.append((i, text[i: i + chunk_size]))    # (块在全文中的偏移, 块内容)
        if i + chunk_size >= len(text):                # 本块已覆盖到文末，收工
            break
    return chunks


def article_chunks(text: str, min_len: int = 120, max_len: int = 500) -> list:
    """条文边界切块：一条法规 = 一个原子单元，同章内短条贪心合并

    与滑窗的哲学差异：
      滑窗保证"块大小均匀"（为 embedding 友好，向量表示质量稳定）
      条文切块保证"语义单元完整"（一条讲一件事；引用天然对齐"第X条"，
      用户问"第三十四条"时检索粒度 = 引用粒度）

    短条合并是两者的折中：单条太短（有的条文只有一行）embedding 上下文不足，
    所以同章内贪心合并到 min_len 以上——"章内"约束防止章末短条与下章内容
    混块导致 section 标签错位（这正是滑窗方案的老伤，见 Q9 错位案例）。

    超长条文（> max_len）保持独立不再切：条文完整性优先于块大小均匀——
    切半条比大块更伤（这就是要修的"法条腰斩"问题本身）。

    返回 [(偏移, 块文本)] 与滑窗同构，insert_pdf_data 无需感知差异。
    防御性回退：匹配不到任何条文 → 返回 []，调用方回退滑窗。
    """
    matches = list(ARTICLE_PAT.finditer(text))
    if not matches:
        return []

    chapters = find_chapters(text)
    raw = []

    # 第一条之前的"前言块"（公布令/总则导语）：单独成块，不与条文混
    if matches[0].start() > 0:
        head = text[:matches[0].start()].strip()
        if head:
            raw.append((0, head))

    # 按条文起点切段：每条 = 本条起点 → 下一条起点之间的全部文本
    # （包含条内款项"（一）…"和续行——它们不是行首"第X条"，不会被切开）
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw.append((start, text[start:end].strip()))

    # 同章内贪心合并：当前块过短 → 并入前块（前块不超 max_len 且同章）
    merged = []
    for offset, piece in raw:
        if (merged
                and len(piece) < min_len
                and len(merged[-1][1]) + len(piece) < max_len
                and chapter_of(chapters, merged[-1][0]) == chapter_of(chapters, offset)):
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + piece)
        else:
            merged.append((offset, piece))
    return merged


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """切块策略分发：按 CHUNK_STRATEGY 选路，两条路径返回同构 [(偏移, 块文本)]

    article 匹配不到条文时回退滑窗（防御性：语料形态变化不炸管线）——
    回退必须打印警告而非静默：静默回退会让"实验组其实混着滑窗数据"这类事故
    无声发生（今天 RAGAS 调试的同款教训：跑的东西必须是你以为在跑的东西）
    """
    if CHUNK_STRATEGY == "article":
        chunks = article_chunks(text)
        if chunks:
            return chunks
        print("  ⚠ 未匹配到任何条文起点，回退滑窗切块（该文档将混入滑窗数据，对比解读时注意）")
    return sliding_window_chunks(text, chunk_size, overlap)


# ============================ 集合与入库 ============================

def create_pdf_collection():
    """创建/获取正式集合（字段结构与 pdf_practice 完全一致：同一套 config.FIELDS）

    已存在则直接取用 —— 这本身也是一种幂等：脚本重跑不会重复建集合
    """
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 已存在")
        return Collection(COLLECTION_NAME)

    fields = []
    for field_config in config.FIELDS:
        if field_config["name"] == "id":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )
        elif field_config["name"] == "embedding":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.FLOAT_VECTOR,
                dim=field_config["dim"]
            )
        elif field_config["type"] == "VARCHAR":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.VARCHAR,
                max_length=field_config["max_length"]
            )
        fields.append(field)

    schema = CollectionSchema(fields, description="低空规章知识库（LowAlt-Reg Copilot 阶段1）")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"✓ 创建集合 {COLLECTION_NAME}（{len(fields)} 个字段）")
    return collection


def insert_pdf_data(collection, doc_path: str, source_name: str, mode: str = "pdf",
                    chunk_size: int = 400, overlap: int = 80):
    """单文档入库全流程：提取 → 清洗 → 章节感知切块 → 向量化 → 幂等插入

    相比沙箱版的三处升级：
    1. source_name 显式传入（为什么不靠元数据，见 DOCS 配置区注释）
    2. section 从 f"切片{i+1}" 改为真实章节名 —— 用户问"操控员要什么资质"，
       引用能标注"出自 CCAR-92 B 章 操控员管理"，而不是无意义的"切片 47"
    3. mode 双提取路径 —— 文本型 PDF 直接提取 / 曲线化 PDF 走 OCR 缓存
    """
    # ---- 幂等保护：同 source 已入库则整文档跳过 ----
    collection.load()                                  # 过滤查询在内存进行，query 前必须 load（忘了这行会报错）
    existing = collection.query(
        expr=f'source == "{source_name}"',             # 按 source 过滤：文档级唯一标识
        output_fields=["source"],
        limit=1
    )
    if existing:
        print(f"  ⚠ 该来源已入库，跳过: {source_name}")
        return

    header = source_name.split("（")[0]                # 页眉 = 规范名去括号（页眉不含"（CCAR-92部）"这类后缀）
    text = extract_text(doc_path, header, mode)
    chapters = find_chapters(text)
    chunks = chunk_text(text, chunk_size, overlap)     # 策略分发：sliding / article 同构返回

    print(f"\n开始处理: {source_name}")
    print(f"  清洗后全文 {len(text)} 字 → {len(chunks)} 个切片，识别出 {len(chapters)} 个章节")
    print(f"  章节分布: {[t for t, _ in chapters]}")

    # ---- 五列数据：与 schema 字段一一对应，顺序不能乱 ----
    texts, embeddings, sources, sections, keywords_list = [], [], [], [], []

    print("  正在生成向量（每块一次 API 调用，需要几分钟）...")
    for i, (offset, chunk) in enumerate(chunks):       # 解包：块的起始偏移 + 块文本
        embedding = generate_embedding(chunk)
        if embedding is None:                          # 网络抖动/配额问题 → 跳过该块，别让一颗老鼠屎打断全锅
            print(f"  ⚠ 第 {i+1} 块向量化失败，跳过")
            continue

        texts.append(chunk)
        embeddings.append(embedding)
        sources.append(source_name)
        sections.append(chapter_of(chapters, offset))  # 偏移 → 章节名（本次升级的核心一行）
        keywords_list.append(extract_keywords(chunk))

        if (i + 1) % 20 == 0:                          # 进度汇报：全量要好几分钟
            print(f"    已处理 {i+1}/{len(chunks)} 块")

    # ---- 批量插入：五个列表按列传入，Milvus 自动按行组装 ----
    if texts:
        collection.insert([texts, embeddings, sources, sections, keywords_list])
        collection.flush()                             # 强制落盘，确保立即可查
        print(f"  ✓ 成功插入 {len(texts)} 条数据")
    else:
        print("  没有数据需要插入")


# ============================ 主流程 ============================

if __name__ == "__main__":

    # ========== 模式一：DRY_RUN 空跑（先跑这个！） ==========
    # 不连 Milvus、不调 embedding API（零成本），纯本地验证数据质量：
    #   ① 清洗后字符数/切片数是否符合预期（跑之前先预测，偏差大 = 有蹊跷）
    #   ② 抽查切片样本：有没有页码/页眉残留？法条从哪里被切断？
    #      （法条被切断是已知问题，记录下来，阶段 2 做"条文边界切块"对比实验，现在不改）
    #   ③ 章节识别结果对不对（章名列表肉眼校对）
    if DRY_RUN:
        for doc_path, source_name, mode in DOCS:
            header = source_name.split("（")[0]
            text = extract_text(doc_path, header, mode)
            chapters = find_chapters(text)
            chunks = chunk_text(text)

            print(f"\n===== {source_name} =====")
            print(f"清洗后 {len(text)} 字 → {len(chunks)} 个切片")
            print(f"章节({len(chapters)}): {[t for t, _ in chapters]}")
            for idx in [0, len(chunks) // 2, len(chunks) - 1]:    # 抽头/中/尾三块
                offset, chunk = chunks[idx]
                print(f"\n--- 切片[{idx+1}]（偏移 {offset}，属于 [{chapter_of(chapters, offset)}]）---")
                print(chunk[:200])                                # 看 200 字足够判断干净与否
        print("\n>>> 检查无误后：把文件顶部 DRY_RUN 改为 False，重新运行即正式入库")

    # ========== 模式二：正式入库 ==========
    else:
        connect_milvus()                              # 第一站：连接
        collection = create_pdf_collection()          # 第二站：建/取正式集合
        create_index(collection)                      # 第三站：建索引
        # 注：nlist=128 vs 约 170 条数据——数据量刚过 nlist，索引能建但聚心质量一般
        # （数据量小是 MVP 常态，以后语料扩容自然缓解，现在不优化）

        for doc_path, source_name, mode in DOCS:      # 第四站：逐文档入库（内部自带幂等）
            insert_pdf_data(collection, doc_path, source_name, mode)

        # ---- 入库后验证 ----
        collection.load()
        print(f"\n集合统计: 共 {collection.num_entities} 条数据")

        # 三个验证问题，刻意覆盖两份文档 + 检验 section 章节标签：
        #   问题 1/2 答案在川运行规则（第34条：各自向右避让；第41条：提前 1 小时报备）
        #   问题 3 答案在 CCAR-92 —— 若 top 命中的 section 显示"B 章 操控员管理"，
        #   说明"偏移 → 章节"映射真的生效了（这才是本次升级的验收点）
        questions = [
            "相向飞行时应当如何避让",
            "使用目视走廊需要提前多久报备飞行计划",
            "无人驾驶航空器操控员执照管理有哪些要求",
        ]
        for q in questions:
            print(f"\n验证检索: '{q}'")
            query_vec = generate_embedding(q)
            results = collection.search(
                data=[query_vec],                     # 注意：是"向量的列表"，哪怕只查一条也要套 []
                anns_field="embedding",               # 在哪个字段上搜
                param=config.SEARCH_PARAMS,           # 现成的：L2 + nprobe=10
                limit=3,                              # top 3
                output_fields=["text", "source", "section"]
            )
            print("Top 3（L2 距离越小越相关）:")
            for hit in results[0]:
                print(f"  距离={hit.distance:.4f} ｜ {hit.entity.get('source')[:22]}… ｜ {hit.entity.get('section')}")
                print(f"    内容: {hit.entity.get('text')[:60]}...")
