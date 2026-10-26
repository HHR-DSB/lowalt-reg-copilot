"""
Milvus FastAPI 服务
功能：提供 RESTful API 接口来访问 Milvus RAG 系统
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import os
import tempfile
from openai import OpenAI
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import milvus_config as config
import jieba
import jieba.analyse
import uvicorn
import re

# 加载环境变量
load_dotenv()

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接 Milvus
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    print("✓ Milvus 连接已建立")
    yield
    # 关闭时断开连接
    connections.disconnect("default")
    print("✓ Milvus 连接已关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="Milvus RAG API",
    description="基于 Milvus 的 RAG 知识库服务",
    version="1.0.0",
    lifespan=lifespan
)

# 初始化 OpenAI 客户端
embedding_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.EMBEDDING_BASE_URL
)

# ========== 数据模型 ==========

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(10, description="返回结果数量")
    search_type: str = Field("hybrid", description="搜索类型: vector, keyword, hybrid")
    vector_weight: float = Field(0.7, description="向量权重(混合搜索时)")
    rerank: bool = Field(False, description="是否重排序")
    # 多集合支持：请求可指定目标集合（如 lowalt_reg_kb 法规库）。
    # 默认值必须是 None 而不是某个集合名 —— 这样旧调用方（:8002 Simple 版、
    # 既有测试脚本）不传该字段时，行为和从前完全一样。
    # "新增参数的向后兼容" = "不传 = 与过去一模一样"，这是 API 演进的基本功。
    # 这个参数同时是阶段 2 消融实验的开关：对照组指哪个库，由调用方决定。
    collection: Optional[str] = Field(None, description="目标集合名；不传则用默认集合")

class InsertRequest(BaseModel):
    """插入请求模型"""
    text: str = Field(..., description="文本内容")
    source: str = Field(..., description="来源")
    section: str = Field(..., description="章节")

class DeleteRequest(BaseModel):
    """删除请求模型"""
    delete_type: str = Field(..., description="删除类型: id, source, section")
    value: Any = Field(..., description="删除条件值")

class CollectionInfo(BaseModel):
    """集合信息模型"""
    name: str
    num_entities: int
    fields: List[Dict]
    indexes: List[Dict]

# ========== 辅助函数 ==========

def generate_embedding(text: str):
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
        raise HTTPException(status_code=500, detail=f"生成向量失败: {str(e)}")

def extract_keywords(text: str, top_n: int = 10):
    """提取关键词"""
    words = jieba.analyse.extract_tags(text, topK=top_n)
    return " ".join(words)

def split_markdown_content(content: str, source: str):
    """切分 Markdown 内容"""
    chunks = []

    # 按 ## 切分
    h2_sections = re.split(r'\n## ', content)

    for i, h2_section in enumerate(h2_sections):
        if i == 0 and not h2_section.startswith('## '):
            if h2_section.strip():
                chunks.append({
                    'text': h2_section.strip(),
                    'source': source,
                    'section': '前言'
                })
            continue

        lines = h2_section.split('\n')
        h2_title = lines[0].strip()
        h2_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        if '### ' in h2_content:
            h3_sections = re.split(r'\n### ', h2_content)

            for j, h3_section in enumerate(h3_sections):
                if j == 0 and h3_section.strip():
                    chunks.append({
                        'text': f"## {h2_title}\n{h3_section.strip()}",
                        'source': source,
                        'section': h2_title
                    })
                elif j > 0:
                    h3_lines = h3_section.split('\n')
                    h3_title = h3_lines[0].strip()
                    h3_content = '\n'.join(h3_lines[1:]) if len(h3_lines) > 1 else ""

                    chunks.append({
                        'text': f"## {h2_title}\n### {h3_title}\n{h3_content.strip()}",
                        'source': source,
                        'section': f"{h2_title} - {h3_title}"
                    })
        else:
            chunks.append({
                'text': f"## {h2_title}\n{h2_content.strip()}",
                'source': source,
                'section': h2_title
            })

    return chunks

# ========== API 路由 ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Milvus RAG API",
        "version": "1.0.0",
        "endpoints": {
            "查询": "/query",
            "插入": "/insert",
            "删除": "/delete",
            "集合信息": "/collection/info",
            "健康检查": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查 Milvus 连接
        collections = utility.list_collections()
        return {
            "status": "healthy",
            "milvus": "connected",
            "collections_count": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {str(e)}")

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    """查询知识库（支持 collection 参数指定目标集合，默认走 config.COLLECTION_NAME）"""
    try:
        # ---- 多集合分发：请求带 collection 就用它，否则回落到默认集合 ----
        # `x or default` 惯用法：None 是 falsy，触发右边的默认值；
        # 注意别写成 `x if x else default` 的冗长版，也别用 `x ?? default`（那是 JS）
        # （字符串 "lowalt_reg_kb" 是 truthy，会正确选中）
        collection_name = request.collection or config.COLLECTION_NAME

        # 下面两处从 config.COLLECTION_NAME 改为局部变量 collection_name：
        # 集合的"存在性检查"和"实例获取"必须指向同一个名字，
        # 只改一处会出现"检查了 A 却打开了 B"的错位 bug
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"集合 {collection_name} 不存在")

        collection = Collection(collection_name)
        collection.load()

        results = []

        if request.search_type == "vector":
            # 向量检索
            query_embedding = generate_embedding(request.query)

            search_params = {
                "metric_type": config.METRIC_TYPE,
                "params": {"nprobe": config.NPROBE}
            }

            search_results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=request.top_k,
                output_fields=["id", "text", "source", "section", "keywords"]
            )

            for hits in search_results:
                for hit in hits:
                    results.append({
                        "id": hit.id,
                        "score": float(1 / (1 + hit.distance)),
                        "text": hit.entity.get("text", ""),
                        "source": hit.entity.get("source", ""),
                        "section": hit.entity.get("section", ""),
                        "keywords": hit.entity.get("keywords", "")
                    })

        elif request.search_type == "keyword":
            # 关键词检索
            keywords = jieba.analyse.extract_tags(request.query, topK=5)
            # Milvus 只支持前缀匹配，不支持 %keyword% 模式
            # 改为使用包含关键词的精确匹配或前缀匹配
            keyword_conditions = []
            for kw in keywords:
                # 使用前缀匹配
                keyword_conditions.append(f'keywords like "{kw}%"')
            expr = " or ".join(keyword_conditions) if keyword_conditions else "id > 0"

            query_results = collection.query(
                expr=expr,
                output_fields=["id", "text", "source", "section", "keywords"],
                limit=request.top_k
            )

            for i, hit in enumerate(query_results):
                results.append({
                    "id": hit["id"],
                    "score": float(1 - (i / len(query_results))),
                    "text": hit.get("text", ""),
                    "source": hit.get("source", ""),
                    "section": hit.get("section", ""),
                    "keywords": hit.get("keywords", "")
                })

        else:  # hybrid
            # 混合检索
            query_embedding = generate_embedding(request.query)

            # 向量搜索
            search_params = {
                "metric_type": config.METRIC_TYPE,
                "params": {"nprobe": config.NPROBE}
            }

            vector_results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=request.top_k * 2,
                output_fields=["id", "text", "source", "section", "keywords"]
            )

            # 关键词搜索
            keywords = jieba.analyse.extract_tags(request.query, topK=5)
            if keywords:
                keyword_conditions = []
                for kw in keywords:
                    # 使用前缀匹配
                    keyword_conditions.append(f'keywords like "{kw}%"')
                expr = " or ".join(keyword_conditions)

                keyword_results = collection.query(
                    expr=expr,
                    output_fields=["id", "text", "source", "section", "keywords"],
                    limit=request.top_k * 2
                )
            else:
                keyword_results = []

            # 合并结果
            result_map = {}

            # 处理向量结果
            for hits in vector_results:
                for i, hit in enumerate(hits):
                    doc_id = hit.id
                    vector_score = 1 / (1 + hit.distance)

                    result_map[doc_id] = {
                        "id": doc_id,
                        "text": hit.entity.get("text", ""),
                        "source": hit.entity.get("source", ""),
                        "section": hit.entity.get("section", ""),
                        "keywords": hit.entity.get("keywords", ""),
                        "vector_score": float(vector_score),
                        "keyword_score": 0
                    }

            # 处理关键词结果
            for i, hit in enumerate(keyword_results):
                doc_id = hit["id"]
                keyword_score = 1 - (i / max(len(keyword_results), 1))

                if doc_id in result_map:
                    result_map[doc_id]["keyword_score"] = float(keyword_score)
                else:
                    result_map[doc_id] = {
                        "id": doc_id,
                        "text": hit.get("text", ""),
                        "source": hit.get("source", ""),
                        "section": hit.get("section", ""),
                        "keywords": hit.get("keywords", ""),
                        "vector_score": 0,
                        "keyword_score": float(keyword_score)
                    }

            # 计算混合分数
            for doc_id in result_map:
                result = result_map[doc_id]
                result["score"] = float(
                    request.vector_weight * result["vector_score"] +
                    (1 - request.vector_weight) * result["keyword_score"]
                )

            # 排序
            results = sorted(
                result_map.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:request.top_k]

        return {
            "query": request.query,
            "search_type": request.search_type,
            "total": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/insert")
async def insert_document(request: InsertRequest):
    """插入文档"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            # 创建集合
            from pymilvus import FieldSchema, CollectionSchema, DataType

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

            schema = CollectionSchema(fields, description="AI 知识库集合")
            collection = Collection(config.COLLECTION_NAME, schema)

            # 创建索引
            index_params = {
                "metric_type": config.METRIC_TYPE,
                "index_type": config.INDEX_TYPE,
                "params": {"nlist": config.NLIST}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(config.COLLECTION_NAME)

        # 生成向量和关键词
        embedding = generate_embedding(request.text)
        keywords = extract_keywords(request.text)

        # 插入数据
        data = [
            [request.text[:65000]],  # text
            [embedding],  # embedding
            [request.source],  # source
            [request.section],  # section
            [keywords]  # keywords
        ]

        collection.insert(data)
        collection.flush()
        collection.load()

        return {
            "message": "文档插入成功",
            "source": request.source,
            "section": request.section
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"插入失败: {str(e)}")

@app.post("/insert_file")
async def insert_file(file: UploadFile = File(...)):
    """上传并插入文件"""
    try:
        # 检查文件类型
        if not file.filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="仅支持 Markdown 文件")

        # 读取文件内容
        content = await file.read()
        content = content.decode('utf-8')

        # 切分内容
        chunks = split_markdown_content(content, file.filename)

        if not utility.has_collection(config.COLLECTION_NAME):
            # 创建集合（代码同上）
            from pymilvus import FieldSchema, CollectionSchema, DataType
            # ... (创建集合的代码)

        collection = Collection(config.COLLECTION_NAME)

        # 批量插入
        texts = []
        embeddings = []
        sources = []
        sections = []
        keywords_list = []

        for chunk in chunks:
            text = chunk['text'][:65000]
            embedding = generate_embedding(text)
            keywords = extract_keywords(text)

            texts.append(text)
            embeddings.append(embedding)
            sources.append(chunk['source'])
            sections.append(chunk['section'])
            keywords_list.append(keywords)

        if texts:
            data = [texts, embeddings, sources, sections, keywords_list]
            collection.insert(data)
            collection.flush()
            collection.load()

        return {
            "message": "文件插入成功",
            "filename": file.filename,
            "chunks_count": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件插入失败: {str(e)}")

@app.delete("/delete")
async def delete_documents(request: DeleteRequest):
    """删除文档"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")

        collection = Collection(config.COLLECTION_NAME)
        collection.load()

        if request.delete_type == "id":
            # 按 ID 删除
            if isinstance(request.value, list):
                expr = f"id in {request.value}"
            else:
                expr = f"id == {request.value}"

        elif request.delete_type == "source":
            # 按来源删除
            # 先查询获取 ID
            results = collection.query(
                expr=f'source == "{request.value}"',
                output_fields=["id"]
            )
            if not results:
                return {"message": "没有找到匹配的文档", "deleted_count": 0}

            ids = [r["id"] for r in results]
            expr = f"id in {ids}"

        elif request.delete_type == "section":
            # 按章节删除
            results = collection.query(
                expr=f'section == "{request.value}"',
                output_fields=["id"]
            )
            if not results:
                return {"message": "没有找到匹配的文档", "deleted_count": 0}

            ids = [r["id"] for r in results]
            expr = f"id in {ids}"

        else:
            raise HTTPException(status_code=400, detail="无效的删除类型")

        # 执行删除
        collection.delete(expr)
        collection.flush()

        return {
            "message": "删除成功",
            "delete_type": request.delete_type,
            "value": request.value
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@app.get("/collection/info")
async def get_collection_info():
    """获取集合信息"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")

        collection = Collection(config.COLLECTION_NAME)
        collection.load()

        # 获取字段信息
        fields = []
        for field in collection.schema.fields:
            field_info = {
                "name": field.name,
                "type": field.dtype.name,
                "is_primary": field.is_primary,
                "auto_id": field.auto_id
            }
            if field.dtype.name == "FLOAT_VECTOR":
                field_info["dim"] = field.dim
            elif field.dtype.name == "VARCHAR":
                field_info["max_length"] = field.params.get("max_length")
            fields.append(field_info)

        # 获取索引信息
        indexes = []
        for index in collection.indexes:
            indexes.append({
                "field_name": index.field_name,
                "index_type": index.params.get("index_type"),
                "metric_type": index.params.get("metric_type"),
                "params": index.params.get("params", {})
            })

        return {
            "name": config.COLLECTION_NAME,
            "num_entities": collection.num_entities,
            "fields": fields,
            "indexes": indexes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取信息失败: {str(e)}")

@app.get("/collection/stats")
async def get_collection_stats():
    """获取集合统计信息"""
    try:
        if not utility.has_collection(config.COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 {config.COLLECTION_NAME} 不存在")

        collection = Collection(config.COLLECTION_NAME)
        collection.load()

        # 获取来源统计
        sources_result = collection.query(
            expr="id > 0",
            output_fields=["source"],
            limit=999999
        )

        source_stats = {}
        for r in sources_result:
            source = r["source"]
            source_stats[source] = source_stats.get(source, 0) + 1

        return {
            "total_documents": collection.num_entities,
            "source_distribution": source_stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

# ========== 启动服务 ==========

def main():
    """启动 FastAPI 服务"""
    print("="*60)
    print("Milvus RAG API 服务")
    print("="*60)
    print(f"服务地址: http://localhost:8000")
    print(f"API 文档: http://localhost:8000/docs")
    print(f"OpenAPI: http://localhost:8000/openapi.json")
    print("="*60)

    uvicorn.run(
        "milvus_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # 同 lowalt_rag_service：排除 MinIO 数据卷等目录，防止 StatReload
        # 扫描到动态增删的路径而崩溃（reloader 一死，改代码就永远不生效）
        reload_excludes=["volumes*", "ocr_cache*", "milvus.doc*", "*.log"],
        log_level="info"
    )

if __name__ == "__main__":
    main()