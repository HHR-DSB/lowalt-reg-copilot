"""
Milvus 数据插入脚本
功能：读取 Markdown 文件，切分内容，生成向量并插入到 Milvus
"""
import os
import re
import jieba
from openai import OpenAI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from dotenv import load_dotenv
import milvus_config as config
import jieba.analyse 

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（用于生成向量）
embedding_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.EMBEDDING_BASE_URL
)

def connect_milvus():
    """连接到 Milvus"""
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print(f"✓ 已连接到 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")

def create_collection():
    """创建 Milvus 集合"""
    # 检查集合是否存在
    if utility.has_collection(config.COLLECTION_NAME):
        print(f"集合 {config.COLLECTION_NAME} 已存在")
        return Collection(config.COLLECTION_NAME)

    # 定义字段
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

    # 创建集合
    schema = CollectionSchema(fields, description="AI 知识库集合")
    collection = Collection(config.COLLECTION_NAME, schema)

    print(f"✓ 创建集合 {config.COLLECTION_NAME}")
    return collection

def create_index(collection):
    """创建索引"""
# metric_type (距离度量类型)
# 与搜索时相同,指定如何计算向量相似度:

# L2 - 欧几里得距离
# IP - 内积
# COSINE - 余弦相似度
# 等等

# index_type (索引类型)
# 指定使用哪种索引算法。常见类型:

# IVF_FLAT - 倒排文件索引,精确但较慢
# IVF_SQ8 - 带标量量化的倒排索引,节省内存
# IVF_PQ - 带乘积量化的倒排索引,高度压缩
# HNSW - 分层导航小世界图,速度快但占内存
# FLAT - 暴力搜索,最精确但最慢
# ANNOY - 近似最近邻索引

# params.nlist (聚类中心数量)
# 仅用于 IVF 系列索引,表示:

# 将向量空间划分成多少个聚类单元(bucket/cell)
# 值越大:索引越精细,搜索越准确,但构建索引和内存消耗越大
# 典型值:1024, 2048, 4096
# 经验公式:nlist = 4 * sqrt(数据量)
    index_params = {
        "metric_type": config.METRIC_TYPE,
        "index_type": config.INDEX_TYPE,
        "params": {"nlist": config.NLIST}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✓ 创建向量索引")

def generate_embedding(text):
    """生成文本向量"""
    try:
        response = embedding_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text,
            dimensions=config.EMBEDDING_DIM,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"生成向量时出错: {e}")
        return None

def extract_keywords(text, top_n=10):
    # 我喜欢人工智能。   我 喜欢 人工智能
    """提取关键词（用于混合检索）"""
    words = jieba.analyse.extract_tags(text, topK=top_n)
    return " ".join(words)

def split_markdown(file_path):
    """
    切分 Markdown 文件
    规则：
    1. 先按 ## 切分
    2. 如果 ## 内有 ###，则按 ### 切分
    3. 否则保持 ## 级别的内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []

    # 按 ## 切分（二级标题）
    h2_sections = re.split(r'\n## ', content)

    for i, h2_section in enumerate(h2_sections):
        if i == 0 and not h2_section.startswith('## '):
            # 第一部分可能不是以 ## 开头的
            if h2_section.strip():
                chunks.append({
                    'text': h2_section.strip(),
                    'source': os.path.basename(file_path),
                    'section': '前言'
                })
            continue

        # 获取二级标题
        lines = h2_section.split('\n')
        h2_title = lines[0].strip()
        h2_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        # 检查是否包含三级标题 ###
        if '### ' in h2_content:
            # 按 ### 切分
            h3_sections = re.split(r'\n### ', h2_content)

            for j, h3_section in enumerate(h3_sections):
                if j == 0 and h3_section.strip():
                    # ## 下直接的内容（### 之前）
                    chunks.append({
                        'text': f"## {h2_title}\n{h3_section.strip()}",
                        'source': os.path.basename(file_path),
                        'section': h2_title
                    })
                elif j > 0:
                    # ### 级别的内容
                    h3_lines = h3_section.split('\n')
                    h3_title = h3_lines[0].strip()
                    h3_content = '\n'.join(h3_lines[1:]) if len(h3_lines) > 1 else ""

                    chunks.append({
                        'text': f"## {h2_title}\n### {h3_title}\n{h3_content.strip()}",
                        'source': os.path.basename(file_path),
                        'section': f"{h2_title} - {h3_title}"
                    })
        else:
            # 没有三级标题，保持二级标题内容
            chunks.append({
                'text': f"## {h2_title}\n{h2_content.strip()}",
                'source': os.path.basename(file_path),
                'section': h2_title
            })

    print(f"✓ 文档切分完成，共 {len(chunks)} 个片段")
    return chunks

def insert_data(collection, file_path):
    """插入数据到 Milvus"""
    # 切分文档
    chunks = split_markdown(file_path)

    # 准备数据
    texts = []
    embeddings = []
    sources = []
    sections = []
    keywords_list = []

    print("正在生成向量...")
    for i, chunk in enumerate(chunks):
        text = chunk['text'][:65000]  # 限制文本长度

        # 生成向量
        embedding = generate_embedding(text)
        if embedding is None:
            print(f"跳过第 {i+1} 个片段（向量生成失败）")
            continue

        # 提取关键词
        keywords = extract_keywords(text)

        texts.append(text)
        embeddings.append(embedding)
        sources.append(chunk['source'])
        sections.append(chunk['section'])
        keywords_list.append(keywords)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(chunks)} 个片段")

    # 插入数据
    if texts:
        data = [
            texts,
            embeddings,
            sources,
            sections,
            keywords_list
        ]

        collection.insert(data)
        collection.flush()
        print(f"✓ 成功插入 {len(texts)} 条数据")
    else:
        print("没有数据需要插入")

def main():
    """主函数"""
    try:
        # 连接 Milvus
        connect_milvus()

        # 创建集合
        collection = create_collection()

        # 创建索引
        create_index(collection)

        # 插入数据
        file_path = "./milvus.doc/law.md"  # 使用相对路径
        print(f"\n开始处理文件: {file_path}")
        insert_data(collection, file_path)

        # 加载集合到内存
        collection.load()
        print("✓ 集合已加载到内存")

        # 显示统计信息
        print(f"\n集合统计信息:")
        print(f"  总文档数: {collection.num_entities}")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭连接
        connections.disconnect("default")
        print("\n✓ 已断开 Milvus 连接")

if __name__ == "__main__":
    # 导入 jieba.analyse 用于关键词提取
    import jieba.analyse
    main()