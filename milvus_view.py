"""
Milvus 数据库查看脚本
功能：查看集合信息、数据统计、样本数据等
"""
from pymilvus import connections, Collection, utility, list_collections
import milvus_config as config
import pandas as pd
from tabulate import tabulate

def connect_milvus():
    """连接到 Milvus"""
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print(f"✓ 已连接到 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")

def list_all_collections():
    """列出所有集合"""
    collections = list_collections()
    print("\n" + "="*50)
    print("所有集合列表:")
    print("="*50)

    if collections:
        for i, name in enumerate(collections, 1):
            collection = Collection(name)
            print(f"{i}. {name} (实体数: {collection.num_entities})")
    else:
        print("当前没有任何集合")

    return collections

def view_collection_schema(collection_name):
    """查看集合结构"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    schema = collection.schema

    print("\n" + "="*50)
    print(f"集合 '{collection_name}' 的结构:")
    print("="*50)
    print(f"描述: {schema.description}")
    print(f"字段数: {len(schema.fields)}")

    print("\n字段详情:")
    field_data = []
    for field in schema.fields:
        field_info = {
            "字段名": field.name,
            "数据类型": field.dtype.name,
            "是否主键": "是" if field.is_primary else "否",
            "自动ID": "是" if field.auto_id else "否"
        }

        # 添加特定字段的额外信息
        if field.dtype.name == "FLOAT_VECTOR":
            field_info["维度"] = field.dim
        elif field.dtype.name == "VARCHAR":
            field_info["最大长度"] = field.params.get("max_length", "N/A")

        field_data.append(field_info)

    # 使用 pandas 创建表格
    df = pd.DataFrame(field_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

def view_collection_stats(collection_name):
    """查看集合统计信息"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    collection.load()

    print("\n" + "="*50)
    print(f"集合 '{collection_name}' 的统计信息:")
    print("="*50)

    # 基本统计
    print(f"总实体数: {collection.num_entities}")

    # 索引信息
    indexes = collection.indexes
    if indexes:
        print("\n索引信息:")
        for index in indexes:
            print(f"  - 字段: {index.field_name}")
            print(f"    类型: {index.params.get('index_type', 'N/A')}")
            print(f"    度量: {index.params.get('metric_type', 'N/A')}")
            print(f"    参数: {index.params.get('params', {})}")

    # 分区信息
    partitions = collection.partitions
    print(f"\n分区数: {len(partitions)}")
    for partition in partitions:
        print(f"  - {partition.name}: {partition.num_entities} 实体")

    # 获取来源统计
    try:
        sources_result = collection.query(
            expr="id > 0",
            output_fields=["source"],
            limit=999999
        )

        if sources_result:
            sources = [r["source"] for r in sources_result]
            source_counts = pd.Series(sources).value_counts()

            print("\n文档来源分布:")
            for source, count in source_counts.items():
                print(f"  - {source}: {count} 条")
    except Exception as e:
        print(f"获取来源统计失败: {e}")

    # 获取章节统计
    try:
        sections_result = collection.query(
            expr="id > 0",
            output_fields=["section"],
            limit=100  # 只获取前100条来展示
        )

        if sections_result:
            sections = [r["section"] for r in sections_result]
            unique_sections = list(set(sections))[:10]  # 显示前10个不同的章节

            print(f"\n章节示例（前{len(unique_sections)}个）:")
            for section in unique_sections:
                print(f"  - {section}")
    except Exception as e:
        print(f"获取章节统计失败: {e}")

def view_sample_data(collection_name, limit=5):
    """查看样本数据"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    collection.load()

    print("\n" + "="*50)
    print(f"集合 '{collection_name}' 的样本数据（前{limit}条）:")
    print("="*50)

    # 查询样本数据
    results = collection.query(
        expr="id > 0",
        output_fields=["id", "text", "source", "section", "keywords"],
        limit=limit
    )

    if results:
        for i, result in enumerate(results, 1):
            print(f"\n--- 记录 {i} ---")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"来源: {result.get('source', 'N/A')}")
            print(f"章节: {result.get('section', 'N/A')}")
            print(f"关键词: {result.get('keywords', 'N/A')}")

            # 显示文本的前200个字符
            text = result.get('text', '')
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"文本内容: {text}")
    else:
        print("集合中没有数据")

def search_by_text(collection_name, query_text):
    """根据文本内容搜索（精确匹配）"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    collection.load()

    # 使用包含搜索
    expr = f'text like "%{query_text}%"'

    try:
        results = collection.query(
            expr=expr,
            output_fields=["id", "text", "section"],
            limit=10
        )

        print(f"\n搜索 '{query_text}' 的结果（最多10条）:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n--- 结果 {i} ---")
                print(f"ID: {result['id']}")
                print(f"章节: {result['section']}")
                text = result['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"内容: {text}")
        else:
            print("没有找到匹配的结果")
    except Exception as e:
        print(f"搜索失败: {e}")

def interactive_view():
    """交互式查看"""
    while True:
        print("\n" + "="*50)
        print("Milvus 数据库查看工具")
        print("="*50)

        print("\n请选择操作:")
        print("1. 列出所有集合")
        print("2. 查看集合结构")
        print("3. 查看集合统计")
        print("4. 查看样本数据")
        print("5. 文本搜索")
        print("0. 退出")

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == "0":
            print("退出程序")
            break

        elif choice == "1":
            list_all_collections()

        elif choice == "2":
            view_collection_schema(config.COLLECTION_NAME)

        elif choice == "3":
            view_collection_stats(config.COLLECTION_NAME)

        elif choice == "4":
            try:
                limit = input("显示多少条样本数据？(默认5): ").strip()
                limit = int(limit) if limit else 5
                view_sample_data(config.COLLECTION_NAME, limit)
            except ValueError:
                print("请输入有效的数字")

        elif choice == "5":
            query_text = input("请输入搜索文本: ").strip()
            if query_text:
                search_by_text(config.COLLECTION_NAME, query_text)
            else:
                print("搜索文本不能为空")

        else:
            print("无效选项")

def main():
    """主函数"""
    try:
        # 连接 Milvus
        connect_milvus()

        # 交互式查看
        interactive_view()

    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭连接
        connections.disconnect("default")
        print("\n✓ 已断开 Milvus 连接")

if __name__ == "__main__":
    # 安装 tabulate（如果需要）
    try:
        import tabulate
    except ImportError:
        print("正在安装 tabulate...")
        import subprocess
        subprocess.check_call(["pip3", "install", "tabulate", "--user"])
        import tabulate

    main()