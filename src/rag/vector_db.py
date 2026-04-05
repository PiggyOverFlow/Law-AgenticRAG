from typing import List, Dict, Any, Optional, Sequence
import logging
from pathlib import Path
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from datetime import datetime

from config import get_config
from src.rag.chunker import LawChunk


logger = logging.getLogger(__name__)


class VectorDBManager:
    """Milvus 向量数据库管理器，用于存储和检索法律文本向量"""
    
    def __init__(self):
        """初始化向量数据库管理器"""
        self.config = get_config()
        self._init_db()

    def _init_db(self):
        """初始化数据库连接"""
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
        self.metric_type = str(getattr(self.config.rag.vector_db, "metric_type", "COSINE")).upper()

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)

            # 兼容处理：检查是否存在索引，如果没有则创建
            if not self.collection.has_index():
                logger.warning(f"集合 {collection_name} 缺少索引，正在补充创建...")
                index_params = {
                    "index_type": "AUTOINDEX",
                    "metric_type": self.metric_type,
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
                "metric_type": self.metric_type,
                "params": {}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
            logger.info(
                f"创建并加载新 Milvus 集合: {collection_name}，维度: {self.config.rag.vector_db.dimension}，"
                f"metric={self.metric_type}"
            )

    def normalize_case_date(self, case_date: Optional[str]) -> str:
        """规范化案件时间，支持 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD。"""
        raw = str(case_date or "").strip()
        if not raw:
            return ""

        normalized = raw.replace("/", "-").replace(".", "-")
        try:
            return datetime.strptime(normalized, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            logger.warning("case_date 格式无法解析: %s，预期格式为 YYYY-MM-DD", raw)
            return raw

    def is_effective_for_date(
        self,
        effective_date: Optional[str],
        repeal_date: Optional[str],
        case_date: Optional[str],
    ) -> bool:
        """判断某条规范在案件日期是否有效。"""
        normalized_case_date = self.normalize_case_date(case_date)
        if not normalized_case_date:
            return True

        effective = self.normalize_case_date(effective_date)
        repeal = self.normalize_case_date(repeal_date)
        if effective and normalized_case_date < effective:
            return False
        if repeal and normalized_case_date > repeal:
            return False
        return True

    def insert_chunks(self, chunks: List[LawChunk], embeddings: List[List[float]]):
        """插入法律片段到 Milvus
        
        Args:
            chunks: 法律文本块列表
            embeddings: 向量列表
        """
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

    def _query_all_rows(self, output_fields: List[str], expr: str = "id != ''") -> List[Dict[str, Any]]:
        """分批读取集合数据，避免单次 query 过大。"""
        total = int(self.collection.num_entities or 0)
        if total <= 0:
            return []

        batch_size = 5000
        rows: List[Dict[str, Any]] = []
        offset = 0
        while offset < total:
            batch = self.collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=min(batch_size, total - offset),
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < batch_size:
                break
        return rows

    def list_indexed_sources(self) -> Dict[str, Dict[str, Any]]:
        """聚合当前索引中的法律来源状态，用于增量同步。"""
        rows = self._query_all_rows(
            ["id", "law_name", "article_num", "metadata", "effective_date", "repeal_date"]
        )
        summary: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            source_law_id = str(metadata.get("source_law_id") or row.get("law_name") or "").strip()
            if not source_law_id:
                continue

            item = summary.setdefault(
                source_law_id,
                {
                    "source_law_id": source_law_id,
                    "law_name": str(row.get("law_name") or "").strip(),
                    "source_hash": str(metadata.get("source_hash") or "").strip(),
                    "version_id": str(metadata.get("version_id") or "").strip(),
                    "effective_date": str(metadata.get("effective_date") or row.get("effective_date") or "").strip(),
                    "repeal_date": str(metadata.get("repeal_date") or row.get("repeal_date") or "").strip(),
                    "chunk_ids": [],
                    "chunk_count": 0,
                },
            )
            chunk_id = str(row.get("id") or "").strip()
            if chunk_id:
                item["chunk_ids"].append(chunk_id)
            item["chunk_count"] += 1
        return summary

    def delete_chunks_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """按 chunk id 删除现有记录。"""
        safe_ids = [str(chunk_id).replace("'", "\\'") for chunk_id in chunk_ids if str(chunk_id).strip()]
        if not safe_ids:
            return 0

        batch_size = 200
        deleted = 0
        for i in range(0, len(safe_ids), batch_size):
            batch = safe_ids[i:i + batch_size]
            expr = "id in [" + ", ".join([f"'{chunk_id}'" for chunk_id in batch]) + "]"
            self.collection.delete(expr=expr)
            deleted += len(batch)

        self.collection.flush()
        logger.info("已从 Milvus 删除 %s 个旧片段", deleted)
        return deleted

    def delete_chunks_by_source_law_ids(self, source_law_ids: Sequence[str]) -> int:
        """删除指定法律来源对应的所有 chunk。"""
        targets = {str(law_id).strip() for law_id in source_law_ids if str(law_id).strip()}
        if not targets:
            return 0

        indexed = self.list_indexed_sources()
        delete_ids: List[str] = []
        for law_id in targets:
            entry = indexed.get(law_id)
            if entry:
                delete_ids.extend(entry.get("chunk_ids", []))

        return self.delete_chunks_by_ids(delete_ids)

    def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """在 Milvus 中进行向量搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        search_params = {"metric_type": self.metric_type, "params": {}}
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
        case_date = (filters or {}).get("case_date")
        for result in results[0]:
            raw_score = float(result.score)
            distance = self._to_distance(raw_score)
            effective_date = result.entity.get("effective_date")
            repeal_date = result.entity.get("repeal_date")
            if case_date and not self.is_effective_for_date(effective_date, repeal_date, case_date):
                continue
            formatted_results.append({
                "chunk_id": result.id,
                "score": distance,
                "raw_score": raw_score,
                "law_name": result.entity.get("law_name"),
                "article_num": result.entity.get("article_num"),
                "content": result.entity.get("content"),
                "level": result.entity.get("level"),
                "metadata": result.entity.get("metadata"),
                "effective_date": effective_date,
                "repeal_date": repeal_date,
            })

        # 统一语义：score 为距离，越小越相关。
        formatted_results.sort(key=lambda x: float(x.get("score", 9999.0)))
        return formatted_results

    def _to_distance(self, score: float) -> float:
        """统一把底层分值转换为"距离"，确保上层逻辑稳定：距离越小越相关
        
        Args:
            score: 原始分数
            
        Returns:
            float: 距离值
        """
        if self.metric_type == "COSINE":
            return 1.0 - score
        return score

    def keyword_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """基于关键词的稀疏检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
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
        case_date = (filters or {}).get("case_date")
        for row in rows:
            content = str(row.get("content", ""))
            metadata = row.get("metadata") or {}
            retrieval_text = str(metadata.get("retrieval_text", ""))
            path_text = str(metadata.get("structure_path_text", ""))
            effective_date = row.get("effective_date")
            repeal_date = row.get("repeal_date")
            if case_date and not self.is_effective_for_date(effective_date, repeal_date, case_date):
                continue
            content_lower = " ".join([content, retrieval_text, path_text]).lower()
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
                "effective_date": effective_date,
                "repeal_date": repeal_date,
            })

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[:top_k]

    def _build_expr(self, filters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """构建 Milvus 过滤表达式，仅保留集合内真实字段
        
        Args:
            filters: 过滤条件字典
            
        Returns:
            Optional[str]: 过滤表达式
        """
        if not filters:
            return None

        allowed_fields = {"law_name", "article_num", "level", "effective_date", "repeal_date"}
        expressions = []
        case_date = self.normalize_case_date(filters.get("case_date"))
        if case_date:
            expressions.append(f"(effective_date == '' or effective_date <= '{case_date}')")
            expressions.append(f"(repeal_date == '' or repeal_date >= '{case_date}')")

        for key, value in filters.items():
            if key == "case_date":
                continue
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
        """获取集合统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.collection.num_entities
        return {
            "type": "milvus",
            "collection": self.collection.name,
            "metric_type": self.metric_type,
            "num_entities": stats,
            "description": self.collection.description,
            "vectors_count": stats,
            "indexed_sources": len(self.list_indexed_sources()) if stats else 0,
        }
