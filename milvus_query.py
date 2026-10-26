"""
Milvus 查询召回脚本
功能：支持向量检索、关键词检索和混合检索
"""
import os
from openai import OpenAI
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import milvus_config as config
import jieba
import jieba.analyse
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（用于生成查询向量）
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

def generate_query_embedding(query_text):
    """生成查询文本的向量"""
    try:
        response = embedding_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=query_text,
            dimensions=config.EMBEDDING_DIM,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"生成查询向量时出错: {e}")
        return None

def extract_query_keywords(query_text, top_n=5):
    """提取查询关键词"""
    words = jieba.analyse.extract_tags(query_text, topK=top_n)
    return words

def vector_search(collection, query_text, top_k=10):
    """向量检索"""
    # 生成查询向量
    query_embedding = generate_query_embedding(query_text)
    if query_embedding is None:
        return []

    # 执行向量搜索
    search_params = {
        "metric_type": config.METRIC_TYPE,
        "params": {"nprobe": config.NPROBE}
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "text", "source", "section", "keywords"]
    )

    return results[0] if results else []

def keyword_search(collection, query_text, top_k=10):
    """关键词检索"""
    # 提取关键词
    keywords = extract_query_keywords(query_text)

    if not keywords:
        print("未能提取关键词")
        return []

    # 构建查询表达式（Milvus 只支持前缀匹配）
    keyword_conditions = []
    for keyword in keywords:
        keyword_conditions.append(f'keywords like "{keyword}%"')

    expr = " or ".join(keyword_conditions)

    try:
        results = collection.query(
            expr=expr,
            output_fields=["id", "text", "source", "section", "keywords"],
            limit=top_k
        )
        return results
    except Exception as e:
        print(f"关键词搜索失败: {e}")
        return []

def hybrid_search(collection, query_text, top_k=10, vector_weight=0.7):
    """
    混合检索（向量 + 关键词）
    vector_weight: 向量检索权重 (0-1)，关键词权重为 1-vector_weight
    """
    print(f"\n执行混合检索 (向量权重: {vector_weight}, 关键词权重: {1-vector_weight})")

    # 向量检索
    vector_results = vector_search(collection, query_text, top_k * 2)

    # 关键词检索
    keyword_results = keyword_search(collection, query_text, top_k * 2)

    # 合并结果并计算混合分数
    result_map = {}

    # 处理向量检索结果
    for i, hit in enumerate(vector_results):
        doc_id = hit.id
        # 向量相似度分数（距离越小越好，需要转换）
        vector_score = 1 / (1 + hit.distance)  # 将距离转换为相似度

        result_map[doc_id] = {
            "id": doc_id,
            "text": hit.entity.get("text", ""),
            "source": hit.entity.get("source", ""),
            "section": hit.entity.get("section", ""),
            "keywords": hit.entity.get("keywords", ""),
            "vector_score": vector_score,
            "vector_rank": i + 1,
            "keyword_score": 0,
            "keyword_rank": top_k * 2 + 1  # 默认排名很低
        }

    # 处理关键词检索结果
    for i, hit in enumerate(keyword_results):
        doc_id = hit["id"]
        # 关键词相关性分数（基于排名）
        keyword_score = 1 - (i / len(keyword_results))

        if doc_id in result_map:
            result_map[doc_id]["keyword_score"] = keyword_score
            result_map[doc_id]["keyword_rank"] = i + 1
        else:
            result_map[doc_id] = {
                "id": doc_id,
                "text": hit.get("text", ""),
                "source": hit.get("source", ""),
                "section": hit.get("section", ""),
                "keywords": hit.get("keywords", ""),
                "vector_score": 0,
                "vector_rank": top_k * 2 + 1,
                "keyword_score": keyword_score,
                "keyword_rank": i + 1
            }

    # 计算混合分数
    for doc_id in result_map:
        result = result_map[doc_id]
        result["hybrid_score"] = (
            vector_weight * result["vector_score"] +
            (1 - vector_weight) * result["keyword_score"]
        )

    # 按混合分数排序
    sorted_results = sorted(
        result_map.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return sorted_results[:top_k]

def rerank_results(results, query_text):
    """
    重排序结果（可以使用更复杂的重排序模型）
    这里使用简单的关键词匹配度重排序
    """
    query_keywords = set(jieba.cut(query_text.lower()))

    for result in results:
        text = result.get("text", "").lower()
        text_keywords = set(jieba.cut(text))

        # 计算关键词重叠度
        overlap = len(query_keywords & text_keywords)
        result["rerank_score"] = overlap / max(len(query_keywords), 1)

    # 结合原始分数和重排序分数
    for result in results:
        original_score = result.get("hybrid_score", 0)
        rerank_score = result.get("rerank_score", 0)
        result["final_score"] = 0.8 * original_score + 0.2 * rerank_score

    # 重新排序
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def display_results(results, query_text, search_type="混合检索"):
    """显示搜索结果"""
    print("\n" + "="*60)
    print(f"{search_type}结果")
    print(f"查询: {query_text}")
    print("="*60)

    if not results:
        print("没有找到相关结果")
        return

    for i, result in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(f"章节: {result.get('section', 'N/A')}")
        print(f"来源: {result.get('source', 'N/A')}")

        # 显示分数信息
        if "hybrid_score" in result:
            print(f"混合分数: {result['hybrid_score']:.4f}")
            print(f"  向量分数: {result.get('vector_score', 0):.4f} (排名: {result.get('vector_rank', 'N/A')})")
            print(f"  关键词分数: {result.get('keyword_score', 0):.4f} (排名: {result.get('keyword_rank', 'N/A')})")

        if "final_score" in result:
            print(f"最终分数: {result['final_score']:.4f}")

        # 显示关键词
        keywords = result.get('keywords', '')
        if keywords:
            print(f"关键词: {keywords[:100]}...")

        # 显示文本片段
        text = result.get('text', '')
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"内容: {text}")

def interactive_query():
    """交互式查询"""
    if not utility.has_collection(config.COLLECTION_NAME):
        print(f"集合 {config.COLLECTION_NAME} 不存在，请先运行插入脚本")
        return

    collection = Collection(config.COLLECTION_NAME)
    collection.load()
    print(f"✓ 集合 {config.COLLECTION_NAME} 已加载")

    while True:
        print("\n" + "="*50)
        print("查询模式选择:")
        print("1. 向量检索")
        print("2. 关键词检索")
        print("3. 混合检索（推荐）")
        print("4. 混合检索 + 重排序")
        print("0. 退出")

        mode = input("\n请选择模式 (0-4): ").strip()

        if mode == "0":
            print("退出查询")
            break

        query_text = input("\n请输入查询内容: ").strip()
        if not query_text:
            print("查询内容不能为空")
            continue

        try:
            top_k = input("返回结果数量 (默认10): ").strip()
            top_k = int(top_k) if top_k else 10

            if mode == "1":
                # 向量检索
                results = vector_search(collection, query_text, top_k)
                # 转换格式
                formatted_results = []
                for hit in results:
                    formatted_results.append({
                        "id": hit.id,
                        "text": hit.entity.get("text", ""),
                        "source": hit.entity.get("source", ""),
                        "section": hit.entity.get("section", ""),
                        "keywords": hit.entity.get("keywords", ""),
                        "vector_score": 1 / (1 + hit.distance)
                    })
                display_results(formatted_results, query_text, "向量检索")

            elif mode == "2":
                # 关键词检索
                results = keyword_search(collection, query_text, top_k)
                display_results(results, query_text, "关键词检索")

            elif mode == "3":
                # 混合检索
                weight = input("向量权重 (0-1, 默认0.7): ").strip()
                weight = float(weight) if weight else 0.7
                results = hybrid_search(collection, query_text, top_k, weight)
                display_results(results, query_text, "混合检索")

            elif mode == "4":
                # 混合检索 + 重排序
                weight = input("向量权重 (0-1, 默认0.7): ").strip()
                weight = float(weight) if weight else 0.7
                results = hybrid_search(collection, query_text, top_k * 2, weight)
                results = rerank_results(results, query_text)[:top_k]
                display_results(results, query_text, "混合检索+重排序")

            else:
                print("无效的模式选择")

        except Exception as e:
            print(f"查询出错: {e}")

def main():
    """主函数"""
    try:
        # 连接 Milvus
        connect_milvus()

        # 交互式查询
        interactive_query()

    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭连接
        connections.disconnect("default")
        print("\n✓ 已断开 Milvus 连接")

if __name__ == "__main__":
    main()