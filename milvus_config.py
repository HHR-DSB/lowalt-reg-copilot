"""
Milvus 配置文件
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Milvus 配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "ai_knowledge_base"
EMBEDDING_DIM = 1024  # qwen3.7-text-embedding-flash

# 阿里云配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 索引配置
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NPROBE = 10
NLIST = 128

# metric_type (距离度量类型)
# 指定计算向量之间相似度的方法。常见的类型包括:

# L2 - 欧几里得距离,值越小表示越相似
# IP - 内积(Inner Product),值越大表示越相似
# COSINE - 余弦相似度,值越大表示越相似(范围 -1 到 1)
# HAMMING - 汉明距离,用于二进制向量
# JACCARD - 杰卡德距离,用于二进制向量

# 选择哪种度量取决于你的向量类型和业务需求。例如,对于归一化的向量,IP 和 COSINE 效果类似。
# params (搜索参数)
# 包含搜索时的具体配置参数,这里的 nprobe 是其中一个:

# nprobe - 搜索时要查询的聚类单元(bucket)数量

# 值越大:搜索越精确,但速度越慢
# 值越小:搜索越快,但可能牺牲一些准确性
# 通常设置为索引的 nlist 参数的 10-20%
# 取值范围:[1, nlist]



# 其他可能的搜索参数还包括:

# ef - 用于 HNSW 索引
# search_k - 用于 ANNOY 索引

# 搜索配置
TOP_K = 10
SEARCH_PARAMS = {
    "metric_type": METRIC_TYPE,
    "params": {"nprobe": NPROBE}
}

# 字段定义
FIELDS = [
    {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True},
    {"name": "text", "type": "VARCHAR", "max_length": 65535},
    {"name": "embedding", "type": "FLOAT_VECTOR", "dim": EMBEDDING_DIM},
    {"name": "source", "type": "VARCHAR", "max_length": 500},
    {"name": "section", "type": "VARCHAR", "max_length": 500},
    {"name": "keywords", "type": "VARCHAR", "max_length": 1000}  # 用于混合检索的关键词
]