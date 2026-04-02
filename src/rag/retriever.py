from typing import List, Dict, Any, Optional, Sequence, Callable, Protocol, runtime_checkable
import logging
import re
import json
import time
import math
from dataclasses import dataclass, field

import requests
import torch

from config import get_config
from src.rag.chunker import LawChunk, RetrievalResult
from src.rag.vector_db import VectorDBManager


logger = logging.getLogger(__name__)


# =============================================================================
# 轻量级 StateGraph 兜底实现 — 兼容 LangGraph StateGraph 接口，零外部依赖
# =============================================================================
class _StateGraph:
    def __init__(self, state_class: type):
        self._state_class = state_class
        self._nodes: Dict[str, Callable] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional: Dict[str, Callable] = {}
        self._entry: str = ""

    def add_node(self, name: str, func: Callable) -> None:
        self._nodes[name] = func

    def add_edge(self, src: str, dst: str) -> None:
        self._edges.setdefault(src, []).append(dst)

    def add_conditional_edges(
        self, src: str, router: Callable, mapping: Dict[str, str]
    ) -> None:
        self._conditional[src] = (router, mapping)

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def compile(self) -> "_CompiledGraph":
        return _CompiledGraph(
            state_class=self._state_class,
            nodes=self._nodes,
            edges=self._edges,
            conditional=self._conditional,
            entry=self._entry,
        )


class _END:
    pass


END = _END()


class _CompiledGraph:
    def __init__(
        self,
        state_class: type,
        nodes: Dict[str, Callable],
        edges: Dict[str, List[str]],
        conditional: Dict[str, tuple[Callable, Dict[str, str]]],
        entry: str,
    ):
        self._state_class = state_class
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry

    def invoke(self, initial_state) -> dict:
        if isinstance(initial_state, dict):
            state = self._state_class(**initial_state)
        else:
            state = initial_state

        current = self._entry
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            if isinstance(current, _END):
                break

            node_func = self._nodes.get(current)
            if node_func is None:
                break

            result = node_func(state)
            if result is not None:
                if hasattr(state, "__dict__"):
                    state.__dict__.update(result if isinstance(result, dict) else {})
                else:
                    state.update(result if isinstance(result, dict) else {})

            if current in self._conditional:
                router, mapping = self._conditional[current]
                next_key = router(state)
                current = mapping.get(next_key, next_key)
                continue

            next_list = self._edges.get(current, [])
            current = next_list[0] if next_list else None

        return state.__dict__ if hasattr(state, "__dict__") else dict(state)


StateGraph = _StateGraph


@runtime_checkable
class Embeddings(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


@dataclass
class RetrievalTrace:
    round_index: int
    thought: str
    query_used: str
    rewritten_from: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    vector_hits: int = 0
    kept_hits: int = 0


class QueryAgent:
    def __init__(self):
        self.config = get_config()

    def should_rewrite(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"should_rewrite": False, "reason": "empty_query"}

        llm_result = self._llm_judge_query_completeness(q)
        if llm_result is not None:
            return llm_result

        incomplete_markers = ["这个", "这种", "该", "上述", "前述", "他", "她", "它", "怎么办", "如何处理","如何"]
        should_rewrite = any(marker in q for marker in incomplete_markers) and len(q) <= 24
        return {
            "should_rewrite": should_rewrite,
            "reason": "heuristic_incomplete_query" if should_rewrite else "heuristic_complete_query",
        }

    def rewrite_query(self, query: str, force: bool = False) -> str:
        q = (query or "").strip()
        if not q:
            return q

        if not force:
            decision = self.should_rewrite(q)
            if not decision.get("should_rewrite", False):
                return q

        rewritten = self._llm_rewrite_query(q)
        if rewritten:
            return rewritten

        fallback = q
        fallback = fallback.replace("怎么办", "适用哪些法律条款以及如何处理")
        fallback = fallback.replace("怎么判", "司法实践中如何认定与裁判")
        if fallback == q:
            fallback = f"{q} 相关法律条文 司法解释 适用要点"
        return re.sub(r"\s+", " ", fallback).strip()

    def extract_keywords(self, query: str, max_keywords: int = 8) -> List[str]:
        q = (query or "").strip()
        if not q:
            return []

        llm_keywords = self._llm_extract_keywords(q, max_keywords=max_keywords)
        if llm_keywords:
            return llm_keywords

        return self._heuristic_extract_keywords(q, max_keywords=max_keywords)

    def _llm_judge_query_completeness(self, query: str) -> Optional[Dict[str, Any]]:
        prompt = (
            "你是法律检索查询分析器。判断 query 是否信息完整，是否需要先改写。"
            "若包含模糊指代、上下文缺失、语义过短，建议改写。"
            "只输出 JSON：{\"should_rewrite\": bool, \"reason\": \"...\"}。\n"
            f"query: {query}"
        )
        data = self._call_llm_json(prompt, temperature=0.1, max_tokens=200)
        if not data:
            return None
        return {
            "should_rewrite": bool(data.get("should_rewrite", False)),
            "reason": str(data.get("reason", "llm_decision")),
        }

    def _llm_rewrite_query(self, query: str) -> Optional[str]:
        prompt = (
            "你是法律检索改写器。将 query 改写成更完整、更可检索的一句话，"
            "保持原意，不添加新事实。只输出 JSON：{\"rewrite\": \"...\"}。\n"
            f"query: {query}"
        )
        data = self._call_llm_json(prompt, temperature=0.2, max_tokens=256)
        if not data:
            return None
        rewritten = str(data.get("rewrite", "")).strip()
        return rewritten or None

    def _llm_extract_keywords(self, query: str, max_keywords: int = 8) -> List[str]:
        prompt = (
            "你是法律检索的事实要素抽取器。目标是输出高信息密度的检索要素，而不是宽泛领域词。\n"
            "请从用户问题中抽取：\n"
            "1) dispute_cause: 争议案由（可为空）\n"
            "2) evidence_have: 已有证据（如转账记录、合同、病历、聊天记录）\n"
            "3) evidence_missing: 缺失证据（如无借条、无签字）\n"
            "4) legal_focus: 具体法律判断点（如举证责任、合同成立、过错认定、时效）\n"
            "5) facts: 关键事实动作（如未付款、拒不履行、解除合同）\n"
            "输出 JSON：\n"
            "{\"dispute_cause\":\"...\",\"evidence_have\":[...],\"evidence_missing\":[...],\"legal_focus\":[...],\"facts\":[...]}\n"
            "约束：\n"
            "- 不要输出泛化词（如：法律、纠纷、处理、相关规定）\n"
            "- 优先输出可区分条文的细粒度词组（2-10字）\n"
            f"- 最多输出 {max_keywords} 个最终要素\n"
            f"query: {query}"
        )
        data = self._call_llm_json(prompt, temperature=0.1, max_tokens=256)
        if not data:
            return []

        candidates: List[str] = []

        # 兼容老格式：{"keywords": [...]} 或 {"elements": [...]}。
        for legacy_field in ("keywords", "elements"):
            raw = data.get(legacy_field, [])
            if isinstance(raw, list):
                candidates.extend([str(x) for x in raw])

        cause = data.get("dispute_cause") or data.get("争议案由")
        if isinstance(cause, str) and cause.strip():
            candidates.append(cause)

        for field in ("evidence_have", "evidence_missing", "legal_focus", "facts"):
            value = data.get(field)
            if isinstance(value, list):
                candidates.extend([str(x) for x in value])

        if not candidates:
            return []

        return self._post_process_terms(candidates, max_keywords=max_keywords)

    def _heuristic_extract_keywords(self, query: str, max_keywords: int = 8) -> List[str]:
        candidates: List[str] = []

        # 证据缺失/存在模式：尽量提炼成可检索的短语。
        for m in re.finditer(r"(?:无|没有|未|缺少)([\u4e00-\u9fa5A-Za-z0-9]{2,12})", query):
            candidates.append(f"无{m.group(1)}")
            candidates.append(m.group(1))

        for m in re.finditer(r"(?:只有|仅有|提供了|提交了|持有)([\u4e00-\u9fa5A-Za-z0-9]{2,12})", query):
            candidates.append(m.group(1))

        split_tokens = [
            t.strip() for t in re.split(r"[\s,，。；;、:：()（）]+", query)
            if t.strip() and len(t.strip()) >= 2
        ]
        candidates.extend(split_tokens)

        return self._post_process_terms(candidates, max_keywords=max_keywords)

    def _post_process_terms(self, terms: List[str], max_keywords: int) -> List[str]:
        cleaned: List[str] = []
        seen = set()

        for item in terms:
            token = self._normalize_term(item)
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            cleaned.append(token)

        # 优先高信息密度短语：证据缺失/存在、法律动作、长度更长的短语。
        scored = []
        for token in cleaned:
            score = 1.0
            if token.startswith(("无", "未", "缺少")):
                score += 0.5
            if len(token) >= 4:
                score += 0.25
            if any(ch.isdigit() for ch in token):
                score += 0.2
            scored.append((score, token))

        scored.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
        return [token for _, token in scored[:max_keywords]]

    def _normalize_term(self, term: str) -> str:
        token = re.sub(r"\s+", "", str(term or "")).strip()
        token = token.strip("，。；;、:：()（）[]【】{}\"'“”‘’")
        if len(token) < 2 or len(token) > 14:
            return ""
        if self._is_low_information_token(token):
            return ""
        return token

    def _is_low_information_token(self, token: str) -> bool:
        low_info = {
            "法律", "法规", "条文", "规定", "相关", "问题", "情况", "处理", "如何", "怎么办",
            "纠纷", "案件", "起诉", "诉讼", "请求", "支持", "认定", "成立", "是否",
        }
        if token in low_info:
            return True
        # 纯数字或非常短的功能词，不作为检索要素。
        if token.isdigit():
            return True
        return False

    def _call_llm_json(self, prompt: str, temperature: float, max_tokens: int) -> Optional[Dict[str, Any]]:
        llm_cfg = self.config.llm
        base_url = str(llm_cfg.base_url).rstrip("/")
        model = llm_cfg.primary_model
        api_key = llm_cfg.api_key
        if not base_url or not model or not api_key or str(api_key).startswith("${"):
            return None

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": "你只能输出 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=20,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_json_object(content)
        except Exception as e:
            logger.warning(f"QueryAgent 调用 LLM 失败，使用规则回退: {e}")
            return None

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.replace("json", "", 1).strip()

        try:
            return json.loads(stripped)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text or "")
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}


class EmbeddingModel(Embeddings):
    def __init__(self):
        self.config = get_config()
        self._init_model()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.encode(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.encode_single(text)

    def _init_model(self):
        if self.config.rag.use_ollama:
            self.base_url = self.config.rag.ollama.base_url
            self.model_name = self.config.rag.ollama.embedding_model
            logger.info(f"使用 Ollama 加载嵌入模型: {self.model_name}")
            return

        use_local = self.config.rag.use_local_model
        local_path = self.config.rag.embedding_model_path
        model_name = self.config.rag.embedding_model

        if use_local and local_path:
            model_path = local_path
            logger.info(f"从本地加载嵌入模型: {model_path}")
        else:
            model_path = model_name
            logger.info(f"从 Hugging Face 加载嵌入模型: {model_path}")

        try:
            from pathlib import Path
            from transformers import AutoTokenizer, AutoModel

            if use_local and local_path and not Path(local_path).exists():
                model_path = model_name
                logger.warning(f"本地模型路径不存在，回退 Hugging Face: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.model.eval()

            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("使用 GPU 加速嵌入计算")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.model = None
            self.tokenizer = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.config.rag.use_ollama:
            return self._encode_ollama(texts)

        if self.model is None or self.tokenizer is None:
            dim = int(self.config.rag.vector_db.dimension)
            return [[0.0] * dim for _ in texts]

        try:
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"编码失败: {e}")
            dim = int(self.config.rag.vector_db.dimension)
            return [[0.0] * dim for _ in texts]

    def _encode_ollama(self, texts: List[str]) -> List[List[float]]:
        dim = int(self.config.rag.vector_db.dimension)
        cleaned_texts = []
        for text in texts:
            if not text or not text.strip():
                cleaned_texts.append("")
                continue
            sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
            cleaned_texts.append(sanitized)

        non_empty = [t for t in cleaned_texts if t]
        if not non_empty:
            return [[0.0] * dim for _ in texts]

        headers = {"Connection": "close"}

        try:
            with requests.Session() as session:
                response = session.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": non_empty},
                    headers=headers,
                    timeout=60,
                )
            if response.status_code == 200:
                valid_embeddings = response.json().get("embeddings", [])
                mapped = []
                valid_idx = 0
                for text in cleaned_texts:
                    if text and valid_idx < len(valid_embeddings):
                        mapped.append(valid_embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        mapped.append([0.0] * dim)
                return mapped
        except Exception as e:
            logger.warning(f"Ollama 批量编码失败，降级逐条编码: {e}")

        embeddings = []
        for text in cleaned_texts:
            if not text:
                embeddings.append([0.0] * dim)
                continue

            emb = None
            for _ in range(3):
                try:
                    with requests.Session() as session:
                        response = session.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.model_name, "prompt": text},
                            headers=headers,
                            timeout=20,
                        )
                    if response.status_code == 200:
                        emb = response.json().get("embedding")
                        break
                except Exception:
                    pass
                time.sleep(0.3)

            embeddings.append(emb if emb else [0.0] * dim)

        return embeddings

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]


class Reranker:
    """
    关键词重排器。

    核心规则：
    1. 关键词权重采用候选集内的自适应区分度（类似 IDF），高频词自动降权。
    2. 多关键词共同命中时给予阶梯奖励，增强“证据组合”信号。
    3. 距离越小越相关，最终按距离升序排序。
    """

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        ranked = []
        normalized_keywords = self._normalize_keywords(keywords or [])
        text_blobs = [self._build_text_blob(c) for c in chunks]

        keyword_df: Dict[str, int] = {}
        keyword_idf: Dict[str, float] = {}
        candidate_count = len(text_blobs)
        for kw in normalized_keywords:
            df = sum(1 for text in text_blobs if kw in text)
            keyword_df[kw] = df
            # 平滑 IDF：在当前候选集内出现越少，区分度越高。
            keyword_idf[kw] = math.log((candidate_count + 1) / (df + 1)) + 1.0

        # 高频词（在大多数候选中出现）自动降权，避免 Stop-word Trap。
        effective_keywords = []
        for kw in normalized_keywords:
            ratio = keyword_df.get(kw, 0) / max(1, candidate_count)
            if ratio >= 0.65:
                continue
            effective_keywords.append(kw)

        # 若全部被判定为高频词，则保留区分度最高的少量词，避免完全失去重排信号。
        if not effective_keywords and normalized_keywords:
            effective_keywords = sorted(
                normalized_keywords,
                key=lambda x: keyword_idf.get(x, 0.0),
                reverse=True,
            )[:3]

        for idx, chunk in enumerate(chunks):
            base_distance = float(chunk.get("score", 9999.0))
            text_blob = text_blobs[idx]

            matched_keywords = [kw for kw in effective_keywords if kw in text_blob]
            specificity_sum = sum(keyword_idf.get(kw, 0.0) for kw in matched_keywords)
            coverage = len(matched_keywords) / max(1, len(effective_keywords))

            # 自适应奖励：区分度越高、命中越多，奖励越大；上限防止过度改写排序。
            keyword_boost = min(
                0.35,
                0.06 * specificity_sum + 0.04 * max(0, len(matched_keywords) - 1) + 0.02 * coverage,
            )

            adjusted_distance = base_distance - keyword_boost

            item = dict(chunk)
            item["raw_distance"] = base_distance
            item["keyword_hits"] = matched_keywords
            item["keyword_boost"] = keyword_boost
            item["rerank_score"] = adjusted_distance
            item["score"] = adjusted_distance
            item["keyword_idf"] = {k: round(keyword_idf.get(k, 0.0), 3) for k in matched_keywords}
            item["effective_keywords"] = effective_keywords
            ranked.append(item)

        ranked.sort(key=lambda x: float(x.get("score", 9999.0)))
        return ranked[:top_k]

    def _build_text_blob(self, chunk: Dict[str, Any]) -> str:
        return " ".join(
            [
                str(chunk.get("law_name", "")),
                str(chunk.get("article_num", "")),
                str(chunk.get("content", "")),
            ]
        ).lower()

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        low_info = {
            "法律", "法规", "条文", "规定", "相关", "问题", "情况", "处理", "如何", "怎么办",
            "纠纷", "案件", "起诉", "诉讼", "请求", "支持", "认定", "成立", "是否",
        }
        low_info_lower = {x.lower() for x in low_info}
        dedup = []
        seen = set()
        for item in keywords:
            kw = re.sub(r"\s+", "", str(item or "")).strip().lower()
            if len(kw) < 2 or kw in seen:
                continue
            if kw in low_info_lower:
                continue
            seen.add(kw)
            dedup.append(kw)
        return dedup


class LegalRAG:
    def __init__(self):
        self.config = get_config()
        self.embedding_model = EmbeddingModel()
        self.query_agent = QueryAgent()
        self.vector_db = VectorDBManager()
        self.reranker = Reranker()

    def build_index(self, chunks: List[LawChunk], batch_size: int = 32):
        logger.info(f"开始构建索引，共 {len(chunks)} 个 chunks")
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch_chunks]
            embeddings = self.embedding_model.encode(texts)
            self.vector_db.insert_chunks(batch_chunks, embeddings)
            logger.info(f"索引写入进度: {min(i + batch_size, len(chunks))}/{len(chunks)}")
        logger.info("索引构建完成")

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        result = self.search_with_trace(query, filters)
        return result.get("results", [])

    def search_with_trace(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        original_query = (query or "").strip()
        if not original_query:
            return {"results": [], "trace": [], "query_used": "", "keywords": []}

        merged_filters = filters or {}
        retrieval_cfg = getattr(self.config.rag, "retrieval", None)
        top_k_initial = max(1, int(getattr(retrieval_cfg, "top_k_initial", 10)))
        top_k_final = max(1, int(getattr(retrieval_cfg, "top_k_final", 5)))
        max_rounds = max(1, int(getattr(retrieval_cfg, "agentic_max_rounds", 3)))
        min_rounds = max(1, int(getattr(retrieval_cfg, "agentic_min_rounds", 1)))
        min_rounds = min(min_rounds, max_rounds)
        min_results_to_stop = max(1, int(getattr(retrieval_cfg, "min_results_to_stop", top_k_final)))

        trace: List[RetrievalTrace] = []
        current_query = original_query
        previous_query = None
        final_keywords: List[str] = []
        total_vector_hits = 0
        final_reranked_results: List[Dict[str, Any]] = []
        aggregated_candidates: Dict[str, Dict[str, Any]] = {}

        rewrite_decision = self.query_agent.should_rewrite(original_query)
        if rewrite_decision.get("should_rewrite", False):
            rewritten = self.query_agent.rewrite_query(original_query, force=True)
            if rewritten and rewritten != original_query:
                previous_query = original_query
                current_query = rewritten

        for round_idx in range(1, max_rounds + 1):
            thought = (
                f"第{round_idx}轮：先向量召回 {top_k_initial} 条，再做关键词命中重排，"
                "若命中不足则继续改写 query。"
            )
            keywords = self.query_agent.extract_keywords(current_query, max_keywords=8)
            final_keywords = keywords

            dense = self._dense_search(current_query, merged_filters, top_k_initial)
            # 每轮先保留更大的候选池，跨轮聚合后再截断为 top_k_final，
            # 避免“局部前十”过早截断造成有效条文丢失。
            reranked = self.reranker.rerank(current_query, dense, top_k_initial, keywords=keywords)
            total_vector_hits += len(dense)

            trace.append(
                RetrievalTrace(
                    round_index=round_idx,
                    thought=thought,
                    query_used=current_query,
                    rewritten_from=previous_query,
                    keywords=keywords,
                    vector_hits=len(dense),
                    kept_hits=len(reranked),
                )
            )

            for item in reranked:
                key = self._dedup_candidate_key(item)
                old = aggregated_candidates.get(key)
                if old is None or float(item.get("score", 9999.0)) < float(old.get("score", 9999.0)):
                    aggregated_candidates[key] = item

            final_reranked_results = sorted(
                aggregated_candidates.values(),
                key=lambda x: float(x.get("score", 9999.0)),
            )[:top_k_final]

            has_enough_results = len(aggregated_candidates) >= min_results_to_stop
            has_completed_min_rounds = round_idx >= min_rounds
            if has_completed_min_rounds and has_enough_results:
                break

            if round_idx >= max_rounds:
                break

            next_query = self.query_agent.rewrite_query(current_query, force=True)
            if not next_query:
                next_query = current_query
            if next_query == current_query:
                diversified = f"{current_query} 相关法律依据 司法解释".strip()
                if diversified != current_query:
                    next_query = diversified

            if next_query == current_query and has_completed_min_rounds:
                break

            previous_query = current_query
            current_query = next_query

        retrieval_results = self._to_retrieval_results(
            final_reranked_results,
            query_used=current_query,
            keywords=final_keywords,
            route_focus="law_article",
        )

        logger.info(
            f"Agentic-RAG 完成，轮次: {len(trace)}，召回: {total_vector_hits}，"
            f"返回: {len(retrieval_results)}"
        )

        return {
            "results": retrieval_results,
            "trace": [t.__dict__ for t in trace],
            "query_used": current_query,
            "keywords": final_keywords,
        }

    def answer_with_citations(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        search_result = self.search_with_trace(query, filters)
        results = search_result.get("results", [])

        if not results:
            return {
                "answer": "未检索到相关法条，建议补充案件主体、行为、时间等关键信息后重试。",
                "citations": [],
                "trace": search_result.get("trace", []),
                "query_used": search_result.get("query_used", query),
                "keywords": search_result.get("keywords", []),
            }

        citations = []
        evidence_blocks = []
        for idx, item in enumerate(results, 1):
            chunk = item.chunk
            citations.append(
                {
                    "id": idx,
                    "law_name": chunk.law_name,
                    "article_num": chunk.article_num,
                    "distance": item.score,
                    "snippet": chunk.content[:220],
                }
            )
            evidence_blocks.append(
                f"[{idx}] {chunk.law_name} {chunk.article_num}\n{chunk.content[:1200]}"
            )

        answer = self._synthesize_answer(query, evidence_blocks)
        return {
            "answer": answer,
            "citations": citations,
            "trace": search_result.get("trace", []),
            "query_used": search_result.get("query_used", query),
            "keywords": search_result.get("keywords", []),
        }

    def _dense_search(self, query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        embedding = self.embedding_model.encode_single(query)
        return self.vector_db.search(embedding, top_k, filters)

    def _dedup_candidate_key(self, item: Dict[str, Any]) -> str:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id:
            return f"id::{chunk_id}"

        law_name = str(item.get("law_name", "")).strip()
        article_num = str(item.get("article_num", "")).strip()
        if law_name or article_num:
            return f"law::{law_name}::{article_num}"

        content = str(item.get("content", "")).strip()
        return f"content::{content[:120]}"

    def _to_retrieval_results(
        self,
        reranked: List[Dict[str, Any]],
        query_used: str,
        keywords: List[str],
        route_focus: str,
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for idx, item in enumerate(reranked, 1):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "route_focus": route_focus,
                    "query_rewrite": query_used,
                    "context_tier": self._context_tier_by_rank(idx - 1),
                    "keywords": keywords,
                    "keyword_hits": item.get("keyword_hits", []),
                    "raw_distance": item.get("raw_distance", item.get("score", 0.0)),
                    "keyword_boost": item.get("keyword_boost", 0.0),
                }
            )

            chunk = LawChunk(
                chunk_id=str(item.get("chunk_id", "")),
                law_name=str(item.get("law_name", "")),
                article_num=str(item.get("article_num", "")),
                content=str(item.get("content", "")),
                level=str(item.get("level", "")),
                metadata=metadata,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(item.get("score", 0.0)),
                    rank=idx,
                )
            )
        return results

    def _synthesize_answer(self, query: str, evidence_blocks: List[str]) -> str:
        llm_cfg = self.config.llm
        base_url = str(llm_cfg.base_url).rstrip("/")
        model = llm_cfg.primary_model
        api_key = llm_cfg.api_key

        if not base_url or not model or not api_key or str(api_key).startswith("${"):
            return self._fallback_answer(query, evidence_blocks)

        prompt = (
            "你是法律分析助手。请严格基于给定证据回答用户问题，并在句末使用 [编号] 引用。\n"
            "不要编造未提供的法律依据。\n\n"
            f"用户问题：{query}\n\n"
            "证据：\n"
            + "\n\n".join(evidence_blocks)
        )

        payload = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": min(int(llm_cfg.max_tokens), 1200),
            "messages": [
                {
                    "role": "system",
                    "content": "你是专业法律助手，回答必须引用证据编号。",
                },
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=getattr(self.config.performance, "request_timeout", 120),
            )
            response.raise_for_status()
            content = (
                response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            return content or self._fallback_answer(query, evidence_blocks)
        except Exception as e:
            logger.error(f"基于证据生成回答失败: {e}")
            return self._fallback_answer(query, evidence_blocks)

    def _fallback_answer(self, query: str, evidence_blocks: List[str]) -> str:
        lines = [f"问题：{query}", "", "可参考法条："]
        for block in evidence_blocks[:5]:
            first_line = block.splitlines()[0] if block else ""
            lines.append(first_line)
        lines.append("")
        lines.append("说明：当前为降级回答，请结合上述法条原文进行人工研判。")
        return "\n".join(lines)

    def _context_tier_by_rank(self, rank_index: int) -> int:
        if rank_index == 0:
            return 1
        if rank_index <= 2:
            return 2
        return 3

    def search_by_event(self, event_description: str, event_time: Optional[str] = None) -> List[RetrievalResult]:
        filters = {}
        if event_time:
            filters["effective_date"] = event_time
        return self.search(event_description, filters)

    def get_relevant_laws(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.search(query)

        law_groups = {}
        for result in results:
            law_name = result.chunk.law_name
            if law_name not in law_groups:
                law_groups[law_name] = []
            law_groups[law_name].append(result)

        relevant_laws = []
        for law_name, law_results in law_groups.items():
            relevant_laws.append(
                {
                    "law_name": law_name,
                    "articles": [r.chunk.article_num for r in law_results],
                    "contents": [r.chunk.content for r in law_results],
                    "min_distance": min(r.score for r in law_results),
                }
            )

        relevant_laws.sort(key=lambda x: x["min_distance"])
        return relevant_laws[:top_k]

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.vector_db.get_collection_stats()

    def reset_index(self):
        logger.warning("重置向量数据库索引")
        self.vector_db.delete_collection()
        self.vector_db = VectorDBManager()

    def as_retriever(self, top_k: int = 5) -> "HybridLegalRetriever":
        return HybridLegalRetriever(self, top_k)


class HybridLegalRetriever:
    def __init__(self, legal_rag: LegalRAG, top_k: int = 5):
        self._rag = legal_rag
        self._top_k = top_k

    def invoke(self, query: str) -> List["RetrievedDoc"]:
        results = self._rag.search(query)
        return [
            RetrievedDoc(
                page_content=r.chunk.content,
                metadata={
                    "law_name": r.chunk.law_name,
                    "article_num": r.chunk.article_num,
                    "level": r.chunk.level,
                    "distance": r.score,
                    "rank": r.rank,
                    **r.chunk.metadata,
                },
            )
            for r in results[: self._top_k]
        ]

    def get_relevant_documents(self, query: str) -> List["RetrievedDoc"]:
        return self.invoke(query)

    def __call__(self, query: str) -> List["RetrievedDoc"]:
        return self.invoke(query)


class RetrievedDoc:
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<RetrievedDoc: {self.page_content[:50]}...>"

    def __str__(self):
        return self.page_content

    def to_langchain_doc(self) -> Dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}
