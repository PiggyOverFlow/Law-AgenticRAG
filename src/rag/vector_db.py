from typing import List, Dict, Any, Optional, Sequence
import logging
from pathlib import Path
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from config import get_config
from src.rag.chunker import LawChunk


logger = logging.getLogger(__name__)


class VectorDBManager:
    """Milvus 向量数据库管理器"""
    def __init__(self):
        self.config = get_config()
        self._init_db()

    def _init_db(self):
        # Milvus-lite 使用 uri 连接本地文件或内存
        uri = getattr(self.config.rag.vector_db, 'uri', "./data/milvus_local.db")
        token = getattr(self.config.rag.vector_db, 'password', "") # 如果有 token/password

        # 确保数据目录存在
        file_path = Path(uri.replace("sqlite:///", "")) if "sqlite" in uri else Path(uri)
        if not file_path.parent.exists() and str(file_path.parent) != ".":
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # 连接 Milvus-lite (通常只需 uri)
        try:
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
            logger.info(f"成功连接到 Milvus-lite: {uri}")
        except Exception as e:
            logger.error(f"连接 Milvus-lite 失败: {e}")
            raise

        collection_name = self.config.rag.vector_db.collection_name

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)

            # 兼容处理：检查是否存在索引，如果没有则创建
            if not self.collection.has_index():
                logger.warning(f"集合 {collection_name} 缺少索引，正在补充创建...")
                index_params = {
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                    "params": {}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)

            self.collection.load()
            logger.info(f"加载已存在的 Milvus 集合: {collection_name}")
        else:
            # 创建新集合
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.rag.vector_db.dimension),
                FieldSchema(name="law_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="article_num", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="repeal_date", dtype=DataType.VARCHAR, max_length=20),
            ]

            schema = CollectionSchema(fields, f"Legal laws collection: {collection_name}")
            self.collection = Collection(name=collection_name, schema=schema)

            # 创建索引 (Milvus-lite 本地模式不支持 HNSW，改用 AUTOINDEX)
            index_params = {
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE",  # 推荐使用余弦相似度
                "params": {}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            logger.info(f"创建并加载新 Milvus 集合: {collection_name}，维度: {self.config.rag.vector_db.dimension}")

    def insert_chunks(self, chunks: List[LawChunk], embeddings: List[List[float]]):
        """插入法律片段到 Milvus"""
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append({
                "id": str(chunk.chunk_id)[:100] if chunk.chunk_id else "",
                "embedding": [float(x) for x in embedding],
                "law_name": str(chunk.law_name)[:255] if chunk.law_name else "",
                "article_num": str(chunk.article_num)[:50] if chunk.article_num else "",
                "content": str(chunk.content) if chunk.content else "",
                "level": str(chunk.level)[:20] if chunk.level else "",
                "metadata": chunk.metadata or {},
                "effective_date": str(chunk.metadata.get("effective_date") or "")[:20],
                "repeal_date": str(chunk.metadata.get("repeal_date") or "")[:20],
            })

        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"成功插入 {len(chunks)} 个片段到 Milvus 集合: {self.collection.name}")

    def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """在 Milvus 中进行向量搜索"""
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        expr = self._build_expr(filters)

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["law_name", "article_num", "content", "level", "metadata", "effective_date", "repeal_date"]
        )

        formatted_results = []
        for result in results[0]:
            formatted_results.append({
                "chunk_id": result.id,
                "score": result.score,
                "law_name": result.entity.get("law_name"),
                "article_num": result.entity.get("article_num"),
                "content": result.entity.get("content"),
                "level": result.entity.get("level"),
                "metadata": result.entity.get("metadata"),
                "effective_date": result.entity.get("effective_date"),
                "repeal_date": result.entity.get("repeal_date"),
            })

        return formatted_results

    def keyword_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """基于关键词的稀疏检索（Milvus 标量过滤 + 词项重叠打分）。"""
        terms = [t.strip() for t in query.split() if t.strip()]
        if not terms:
            # 中文检索常见无空格场景，退化为按连续字串切分
            terms = [query[i:i + 2] for i in range(0, max(len(query) - 1, 1))][:16]

        fields = ["id", "law_name", "article_num", "content", "level", "metadata", "effective_date", "repeal_date"]
        expr = self._build_expr(filters)

        # 通过多关键词 OR 提升召回，再按词项重叠进行稀疏重打分
        keyword_expr = None
        safe_terms = [t.replace("'", "\\'") for t in terms if t]
        if safe_terms:
            like_parts = [f"content like '%{t}%'" for t in safe_terms[:12]]
            keyword_expr = "(" + " or ".join(like_parts) + ")"

        final_expr = keyword_expr if keyword_expr else None
        if expr and keyword_expr:
            final_expr = f"({expr}) and {keyword_expr}"
        elif expr:
            final_expr = expr

        try:
            rows = self.collection.query(
                expr=final_expr,
                output_fields=fields,
                limit=max(top_k * 4, top_k)
            )
        except Exception:
            # 某些 Milvus-lite 版本对 like 支持有限，退化为仅 metadata 过滤后查询
            rows = self.collection.query(
                expr=expr,
                output_fields=fields,
                limit=max(top_k * 4, top_k)
            )

        scored = []
        lower_terms = [t.lower() for t in safe_terms if t]
        for row in rows:
            content = str(row.get("content", ""))
            content_lower = content.lower()
            overlap = sum(1 for t in lower_terms if t in content_lower)
            if overlap == 0 and lower_terms:
                continue

            scored.append({
                "chunk_id": row.get("id"),
                "score": float(overlap),
                "law_name": row.get("law_name"),
                "article_num": row.get("article_num"),
                "content": row.get("content"),
                "level": row.get("level"),
                "metadata": row.get("metadata"),
                "effective_date": row.get("effective_date"),
                "repeal_date": row.get("repeal_date"),
            })

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[:top_k]

    def _build_expr(self, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """构建 Milvus 过滤表达式，仅保留集合内真实字段。"""
        if not filters:
            return None

        allowed_fields = {"law_name", "article_num", "level", "effective_date", "repeal_date"}
        expressions = []
        for key, value in filters.items():
            if key not in allowed_fields:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                safe = value.replace("'", "\\'")
                expressions.append(f"{key} == '{safe}'")
            else:
                expressions.append(f"{key} == {value}")

        return " and ".join(expressions) if expressions else None

    def delete_collection(self):
        """删除当前集合"""
        utility.drop_collection(self.collection.name)
        logger.info(f"成功删除 Milvus 集合: {self.collection.name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        stats = self.collection.num_entities
        return {
            "type": "milvus",
            "collection": self.collection.name,
            "num_entities": stats,
            "description": self.collection.description,
            "vectors_count": stats
        }
