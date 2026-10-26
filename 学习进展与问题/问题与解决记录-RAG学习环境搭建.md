# RAG 学习环境搭建：问题与解决记录

> 记录时间：2026-10-21｜最后更新：2026-10-25
> 项目：`C:\Users\admin\Desktop\RAG`（基于 Milvus 的 RAG 知识库问答系统）
> 环境：Windows + py39 conda 环境 + Docker Desktop（WSL2 后端）

---

## 一、conda 环境问题

### 1.1 不知道当前用的哪个环境
- **现象**：终端里 `python --version` 显示 3.9.13，不确定是不是 py39
- **原因**：未激活任何环境时默认使用 base（Anaconda 自带，Python 3.9.13）
- **解决**：每次开终端先 `conda activate py39`，看到命令行前缀出现 `(py39)` 再操作；验证用 `python --version`（应为 3.9.21）
- **知识点**：IDE 右下角显示的解释器（影响"运行"按钮/调试/代码提示）和终端里激活的 conda 环境是两回事，互不同步

### 1.2 `conda activate py39` 报错 CommandNotFoundError
- **现象**：提示 shell 未配置 conda activate
- **原因**：PowerShell 缺少 conda 初始化钩子
- **解决**：管理员执行 `conda init powershell`，关闭终端重新打开即生效
- **替代方案**：`conda.bat activate py39`，或直接用完整路径 `C:\Users\admin\anaconda3\envs\py39\python.exe xxx.py`

### 1.3 py39 缺少项目依赖
- **现象**：跑项目脚本会报 ModuleNotFoundError
- **解决**：激活 py39 后安装
  ```powershell
  pip install pymilvus fastapi uvicorn jieba requests python-multipart
  ```
- **注意**：安装时的 PATH 警告（uvicorn.exe not on PATH）可忽略，本项目都是 `python xxx.py` 方式启动
- **隐患记录**：pip 自动装了 pymilvus 3.0.2，而服务端是 Milvus v2.3.3，若遇兼容错误执行 `pip install "pymilvus==2.3.9"` 降级（实际验证 3.0.2 可正常连接查询）

---

## 二、Docker 安装（重点：网络问题规避）

### 2.1 背景
用户之前装 Docker 后网络出过严重问题，本次采取"最小侵入"策略重装。

### 2.2 为什么 Docker 会搞坏网络（调研结论）
1. Docker 在 Windows 上是"虚拟机方案"（WSL2/Hyper-V），安装时创建 vEthernet 虚拟网卡 + WinNAT + HNS，侵入网络栈
2. **网段冲突**：Docker 默认用 172.x 网段，与家庭/公司内网冲突时路由表被污染 → 全断
3. **DNS 劫持**：WSL2 NAT 模式接管 DNS，与 VPN 冲突 → 域名解析失败
4. **TUN 模式冲突**：Clash/VPN 的 TUN 模式和 Docker 都要接管路由表，二者打架 → 网络瘫痪（国内开发者重灾区）

### 2.3 规避措施（本次实际采用）
| 措施 | 说明 |
|------|------|
| 安装前做网络快照 | `ipconfig /all`、路由表等存到 `Desktop\docker-setup-backup\`，出问题可对比 |
| Per-user 安装 | 免管理员、WSL2 后端、不启用 Hyper-V 后端，侵入最小 |
| 安装期间关闭 TUN 代理 | 系统代理模式与 Docker 可共存 |
| 装后配置镜像加速 | 应对 Docker Hub 国内访问失败 |
| 不开机自启 | Settings → General 取消 "Start Docker Desktop when you sign in" |

### 2.4 安装过程踩的坑
| 问题 | 报错/现象 | 解决 |
|------|----------|------|
| 缺 Windows 组件 | 安装器提示需要 Virtual Machine Platform | 管理员执行 `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` 后重启电脑 |
| 重启后 docker 命令找不到 | `docker : 无法将"docker"项识别为 cmdlet...` | 安装器已写入用户 PATH，但 IDE/终端是安装前启动的（旧 PATH）。**彻底解决：完全重启 IDE**；临时解决：`$env:Path += ";C:\Users\admin\AppData\Local\Programs\DockerDesktop\resources\bin"` |

### 2.5 Docker 日常使用
```powershell
docker desktop status    # 查看状态
docker desktop start     # 启动（或开始菜单打开 Docker Desktop 等鲸鱼变绿）
docker desktop stop      # 关闭引擎
# 退出时也可：任务栏托盘右键鲸鱼图标 → Quit Docker Desktop
```

---

## 三、Milvus 部署问题

### 3.1 拉镜像失败 `error from registry: denied`
- **现象**：`docker compose up -d` 拉取 minio/milvus 镜像时报 denied
- **原因**：Docker Hub 官方源在国内直连不稳定（DNS 污染/限流）
- **解决**：
  1. 配置国内镜像加速：Docker Desktop → Settings → Docker Engine 的 JSON 中加
     ```json
     "registry-mirrors": ["https://docker.m.daocloud.io", "https://docker.1panel.live", "https://docker.1ms.run"]
     ```
  2. 部分镜像（如 minio 特定 tag）加速源也可能同步失败，多试几次或更换 tag
  3. 本项目最终将 `docker-compose.yml` 第 21 行的 MinIO 镜像 tag 更新为 `RELEASE.2024-05-10T01-41-38Z`（原 2023-03-20 tag 各源均拉取失败）

### 3.2 容器状态 `(health: starting)` 迟迟不 healthy
- **原因**：Milvus 健康检查有 90 秒预热期
- **解决**：等 1-2 分钟再 `docker compose ps`，看到 3 个 `(healthy)` 即就绪

### 3.3 验证部署成功的标志
- `docker compose ps`：milvus-etcd / milvus-minio / milvus-standalone 三个 healthy，milvus-attu Up
- 浏览器打开 `http://localhost:3000`（Attu 可视化）能连上 `milvus-standalone:19530`
- 知识库集合 `ai_knowledge_base` 可见

### 3.4 查询知识库里有哪些数据来源（实用命令）
```powershell
C:\Users\admin\anaconda3\envs\py39\python.exe -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://localhost:19530'); rows = c.query(collection_name='ai_knowledge_base', filter='id >= 0', output_fields=['source']); print(sorted({r['source'] for r in rows}))"
```
实测结果：`['ai_dev.md', 'fin.md', 'law.md', 'manual_input']`，共 164 条切片。
**结论：milvus.doc 下的 3 个 PDF 未入库**（milvus_insert.py 只处理 .md 文件）。

---

## 四、功能验证中的认知问题

### 4.1 Attu 里"生成随机向量"搜索结果驴唇不对马嘴
- **原因**：随机数向量不代表任何文本语义，结果必然随机；Attu 只做"给定向量查最近邻"，**不会调通义 API 把文本转成向量**
- **正确姿势**：文本语义搜索用 `python milvus_query.py`（内部先调 embedding API）
- **顺便学到**：dimension=1024（embedding 模型决定）、L2 距离越小越相关、IVF_FLAT 索引、nprobe 参数

### 4.2 三种检索模式的区别
| 模式 | 原理 | 适用 |
|------|------|------|
| vector | 文本转向量，找语义最近的切片 | 换说法也能命中（语义理解） |
| keyword | jieba 分词后关键词匹配 | 精确专有名词 |
| hybrid | 两者加权融合 | 兼顾两者 |

### 4.3 如何判断一次 /chat 请求是否真的用了知识库（三个信号）
1. `sources` 字段：非空 = 检索命中并喂给了模型；`[]` = 没命中
2. `answer` 里是否有"参考 1"式引用表述
3. 看服务终端的检索日志

### 4.4 LLM 报 403 AllocationQuota.FreeTierOnly
- **原因**：阿里云百炼 API Key 的 qwen-plus 模型免费额度用完（免费额度按模型独立计算）
- **解决**：改用其他模型名（如 `qwen-flash`/`qwen-turbo`/`qwen3.8-27b`，在 rag_service_simple.py 第 114 行）；或在百炼控制台充值/关闭"仅使用免费额度"
- **注意**：改完代码必须重启服务（启动时未开 --reload，改代码不重启不生效）

---

## 五、核心 Bug：Simple RAG 永远检索不到知识库（双 bug 叠加）⭐

### 5.1 症状
`POST :8002/chat` 返回 200，但 `sources` 永远是 `[]`，答案没有引用标记，内容是模型自己编的（细节还可能失真）。

### 5.2 排查过程
1. 怀疑检索结果被分数阈值过滤 → 读代码排除
2. 读 `retrieve_documents()` 发现它调用 `http://localhost:8000/search` → 而 milvus_api.py 只有 `/query` 端点 → **404**
3. 继续读字段映射：代码取 `doc["content"]` 和 `doc["metadata"]["source"]`，但 `/query` 实际返回 `text`、顶层 `source`/`section` → **字段名全对不上**

### 5.3 两个 bug 的位置与修复（rag_service_simple.py）
| # | 位置 | 错误 | 修复 |
|---|------|------|------|
| 1 | 第 73 行 | `requests.post(f"{MILVUS_API_URL}/search")` | 改为 `/query`，请求体 `"enable_keyword": False` 改为 `"search_type": "vector"` |
| 2 | 第 150-155 行 | `doc.get("content")`、`doc.get("metadata",{}).get("source")` | 改为 `doc.get("text")`、`doc.get("source")`、`doc.get("section")` |

### 5.4 教训（最值钱的部分）
- **双 bug 同症状**：两个 bug 的外在表现完全一样（sources 空）。只修一个去测试，发现没好，极易误判"修复无效"——症状相同不代表病因唯一
- **外部依赖必须核对接口契约**：调用别的服务前，确认它的真实路径、请求体字段、响应结构（直接看对方 /docs 或源码）
- **日志是最诚实的证人**：:8002 终端里每次请求都在打印 `检索失败: {"detail":"Not Found"}`，早翻日志早发现
- **手动复现外部调用**：手动 curl/Swagger 调一遍下游接口，能直接验证依赖是否正常（本次手动测 /search 得到 404，正是内部故障的根因）

### 5.5 修复后验证结果（成功标准）
```json
{
  "answer": "根据2025年中国法律服务行业市场报告估计为 3,200亿元人民币……",
  "sources": [
    "law.md - 01 法律服务行业发展概览 - 1.2 行业市场规模 (相关度: 0.74)",
    "law.md - 01 法律服务行业发展概览 - 1.5 行业竞争格局 (相关度: 0.70)"
  ]
}
```
- sources 精确到文件名+章节+相关度分数（score = 1/(1+L2距离) 转换的 0~1 相似度，越大越相关）
- 答案数据来自 law.md 而非模型编造
- 小现象：前两条 sources 重复（同章节切出多个相邻切片，均进 top3），增强版会做去重融合

---

## 六、增强版 RAG 设计缺陷：域外旁路条件逻辑反转（实验 B 发现）⭐

### 6.1 背景
实验 B：向 `:8001` 发域外问题 `"推荐几部科幻电影"`，验证意图识别 → 低置信旁路的分流逻辑。

### 6.2 症状
- 终端日志：`识别领域: general (置信度: 0.98)` —— 分类器**非常确定**这是域外问题
- 但响应里 `sources` 仍有 5 条低分结果（约 0.28）、`expanded_queries` 4 条 —— **检索照常执行，旁路未触发**
- 实际浪费：4 次向量化（原问题+3 扩展）+ 4 次 hybrid 检索 + 5 条无关 chunk 注入 Prompt

### 6.3 定位过程（三方证据对照法）
1. 终端日志拿到 domain + 置信度
2. 响应字段确认检索确实发生（sources 非空）
3. 读 rag_service.py 第 406-407 行源码：
   ```python
   # 如果是通用领域且置信度低，直接用 LLM 回答
   if domain == Domain.GENERAL and confidence < 0.5:
   ```
   `0.98 < 0.5` 为 False → 旁路分支不进 → 落入正常检索流程

### 6.4 问题本质：条件逻辑写反了
| 场景 | 当前行为 | 合理行为 |
|------|---------|---------|
| general + 高置信（如 0.98，确定域外） | 仍检索（纯浪费） | **应旁路** |
| general + 低置信（如 0.3，分类器拿不准） | 旁路（跳过知识库） | **应检索兜底**（可能其实是域内问题，只是表述怪异） |

越是确定域外，越浪费检索；越是模糊，越跳过知识库——与设计目标完全相反。

### 6.5 修复方案（候选）
| 方案 | 改法 | 语义 | 取舍 |
|------|------|------|------|
| A（推荐） | `confidence < 0.5` 改为 `confidence >= 0.5` | 确定域外→旁路省开销；不确定→检索兜底 | 保留兜底能力 |
| B（简单） | 条件只留 `domain == Domain.GENERAL` | 所有域外都旁路 | 丢掉对模糊问题的兜底 |

注：这是设计纠正而非 bug 修复——改完记得同步修改第 406 行注释，避免注释和代码继续打架。

### 6.6 验证标准（改后重启 :8001，重发同一电影问题）
1. `sources: []`、`search_results: null`、`expanded_queries: null`
2. 终端有"识别领域: general"日志，但**无**检索相关日志
3. 响应时间明显缩短（省掉向量化+检索开销）

### 6.7 附带发现（同一实验的额外收获）
- **检索权重公式实证**：`score = 0.7×vector_score + 0.3×keyword_score`（实测 0.4037×0.7 + 0×0.3 = 0.2826，分毫不差）
- **数据卫生问题现形**：top 5 中 `law.md "目录 - 02|法律服务行业增量客群洞察"` 出现 2 次（id 不同、内容相同——历史重复插入所致），top_k 名额浪费 2/5。生产环境需入库前去重或按内容 hash 去重
- **hybrid 的短板场景**：域外问题 `keyword_score` 全为 0（零词汇重叠），混合检索退化为纯向量检索

### 6.8 教训
- **条件表达式要对照语义意图复核**：`and confidence < 0.5` 这种边界条件，写反了不报错、不抛异常，行为却完全相反——只能靠"预测→实验→对照"暴露
- **看日志要看全**：只看 `识别领域: general` 会以为门控正常，必须追到置信度数值和后续检索日志才能发现"分类对了但分流错了"
- **四信号判据升级**：`domain` 字段正确 ≠ 流程正确，触发条件和分支走向才是最终事实

---

## 七、常用信息速查

| 服务/工具 | 地址/命令 |
|-----------|----------|
| Attu 可视化 | http://localhost:3000 |
| Milvus gRPC | localhost:19530 |
| MinIO 控制台 | http://localhost:9001（minioadmin/minioadmin） |
| 检索 API（:8000） | `python milvus_api.py`，文档 http://localhost:8000/docs |
| Simple RAG（:8002） | `python rag_service_simple.py`，文档 http://localhost:8002/docs |
| 启动 Milvus | `docker compose up -d` |
| 停止 Milvus（数据保留） | `docker compose down` |
| 交互查询 | `python milvus_query.py` |

**必中知识库的测试问题**：
- `2025年法律服务行业的市场规模有多大？`（law.md）
- `2025年四季度全球经济展望如何？`（fin.md）
- `AI+创业有哪些值得关注的案例？`（ai_dev.md）
