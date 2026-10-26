"""
Milvus 数据删除脚本
功能：删除指定条件的数据或清空集合
"""
from pymilvus import connections, Collection, utility
import milvus_config as config

# # 传入进行delete 或者query
# expr = "id == 123"
# expr = "age > 30"
# expr = "id in [1, 2, 3] and status == 'active'"

def connect_milvus():
    """连接到 Milvus"""
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print(f"✓ 已连接到 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")

def delete_by_ids(collection_name, ids):
    """根据 ID 删除数据"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    expr = f"id in {ids}"

    collection.delete(expr)
    print(f"✓ 已删除 ID 为 {ids} 的数据")

def delete_by_source(collection_name, source):
    """根据来源删除数据"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)

    # 先查询符合条件的数据
    collection.load()
    expr = f'source == "{source}"'

    # 注意：Milvus 2.x 版本的删除需要使用主键
    # 先查询获取 ID，然后删除
    results = collection.query(
        expr=expr,
        output_fields=["id"]
    )

    if results:
        ids = [r["id"] for r in results]
        delete_expr = f"id in {ids}"
        collection.delete(delete_expr)
        print(f"✓ 已删除来源为 '{source}' 的 {len(ids)} 条数据")
    else:
        print(f"没有找到来源为 '{source}' 的数据")

def delete_by_section(collection_name, section):
    """根据章节删除数据"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    collection.load()

    expr = f'section == "{section}"'
    results = collection.query(
        expr=expr,
        output_fields=["id"]
    )

    if results:
        ids = [r["id"] for r in results]
        delete_expr = f"id in {ids}"
        collection.delete(delete_expr)
        print(f"✓ 已删除章节为 '{section}' 的 {len(ids)} 条数据")
    else:
        print(f"没有找到章节为 '{section}' 的数据")

def clear_collection(collection_name):
    """清空集合（删除所有数据）"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    collection = Collection(collection_name)
    collection.load()

    # 获取所有数据的 ID
    results = collection.query(
        expr="id > 0",  # 获取所有数据
        output_fields=["id"],
        limit=999999
    )

    if results:
        ids = [r["id"] for r in results]
        delete_expr = f"id in {ids}"
        collection.delete(delete_expr)
        collection.flush()
#         在 Milvus 中,flush() 操作的作用是:将内存中的数据持久化到磁盘。
        # 详细说明
        # 当你向 Milvus 插入数据时:

        # 数据首先写入内存缓冲区
        # 调用 flush() 后,数据从内存刷写到磁盘
        # 只有 flush 之后,数据才能被搜索到
        print(f"✓ 已清空集合，删除了 {len(ids)} 条数据")
    else:
        print("集合已经是空的")

def drop_collection(collection_name):
    """删除整个集合"""
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")
        return

    utility.drop_collection(collection_name)
    print(f"✓ 已删除集合 {collection_name}")

def interactive_delete():
    """交互式删除"""
    print("\n" + "="*50)
    print("Milvus 数据删除工具")
    print("="*50)

    print("\n请选择删除方式:")
    print("1. 根据 ID 删除")
    print("2. 根据来源删除")
    print("3. 根据章节删除")
    print("4. 清空集合（保留结构）")
    print("5. 删除集合（删除所有）")
    print("0. 退出")

    choice = input("\n请输入选项 (0-5): ").strip()

    if choice == "0":
        print("退出程序")
        return

    elif choice == "1":
        ids_str = input("请输入要删除的 ID（多个 ID 用逗号分隔）: ").strip()
        try:
            ids = [int(id.strip()) for id in ids_str.split(",")]
            delete_by_ids(config.COLLECTION_NAME, ids)
        except ValueError:
            print("ID 格式错误，请输入整数")

    elif choice == "2":
        source = input("请输入要删除的来源文件名: ").strip()
        if source:
            delete_by_source(config.COLLECTION_NAME, source)
        else:
            print("来源不能为空")

    elif choice == "3":
        section = input("请输入要删除的章节名: ").strip()
        if section:
            delete_by_section(config.COLLECTION_NAME, section)
        else:
            print("章节名不能为空")

    elif choice == "4":
        confirm = input(f"确定要清空集合 {config.COLLECTION_NAME} 吗？(yes/no): ").strip().lower()
        if confirm == "yes":
            clear_collection(config.COLLECTION_NAME)
        else:
            print("操作已取消")

    elif choice == "5":
        confirm = input(f"确定要删除集合 {config.COLLECTION_NAME} 吗？这将删除所有数据和结构！(yes/no): ").strip().lower()
        if confirm == "yes":
            drop_collection(config.COLLECTION_NAME)
        else:
            print("操作已取消")

    else:
        print("无效选项")

def main():
    """主函数"""
    try:
        # 连接 Milvus
        connect_milvus()

        # 交互式删除
        interactive_delete()

        # 如果集合还存在，显示剩余数据量
        if utility.has_collection(config.COLLECTION_NAME):
            collection = Collection(config.COLLECTION_NAME)
            collection.flush()
            # print(f"\n当前集合中剩余数据: {collection.num_entities} 条")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭连接
        connections.disconnect("default")
        print("\n✓ 已断开 Milvus 连接")

if __name__ == "__main__":
    main()