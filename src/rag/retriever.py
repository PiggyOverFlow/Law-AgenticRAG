from typing import List, Dict, Any, Optional
import logging
import re
import time
from datetime import datetime
import torch
import numpy as np

from config import get_config
from src.rag.chunker import LawChunk, RetrievalResult
from src.rag.vector_db import VectorDBManager


logger = logging.getLogger(__name__)


class QueryRewriterRouter:
    """检索前 Query 改写与路由决策。"""

    def __init__(self):
        self.config = get_config()

    def rewrite_and_route(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        decision = self._decide_rewrite_strategy(query)
        rewrites = self._execute_rewrite_by_decision(query, decision)
        route = self._route_query(query, rewrites, decision)

        metadata_hints = {}
        law_match = re.findall(r"《([^》]+)》", query)
        if law_match:
            metadata_hints["law_name"] = law_match[0]

        # 从 query 抽取时间信息作为后置时效过滤输入
        year_match = re.search(r"(19\d{2}|20\d{2})年", query)
        if year_match:
            metadata_hints["event_date"] = f"{year_match.group(1)}-01-01"

        if filters:
            metadata_hints.update({k: v for k, v in filters.items() if v is not None})

        return {
            "original_query": query,
            "rewrites": rewrites,
            "rewrite_primary": rewrites[0] if rewrites else query,
            "route": route,
            "metadata_hints": metadata_hints,
            "rewrite_decision": decision,
        }

    def _decide_rewrite_strategy(self, query: str) -> Dict[str, Any]:
        """
        Agentic 决策阶段：只负责判断“是否有必要改写”，不执行改写。
        使用 LLM 动态判断 query 是否需要改写。
        """
        q = query.strip()
        if not q:
            return {
                "should_rewrite": False,
                "action": "skip_rewrite",
                "reason": "empty_query",
                "signals": {"agentic_decision": False},
            }

        try:
            import requests

            base_url = str(self.config.llm.base_url).rstrip("/")
            model = self.config.llm.primary_model
            api_key = self.config.llm.api_key

            prompt = (
                "你是一个法律检索系统的 Query 路由分析 Agent。\n"
                f"分析问题：【{q}】\n"
                "请判断该查询是否需要大模型改写。如果它是短而不依赖上下文的专业词汇（如故意伤害罪）或已经是明确完整的事实，则不需要改写 (false)，避免过度发散；"
                "如果它包含代词、模糊指代（如“这种情况”）、省略了主体/客体，或者高度依赖上文语境才能检索，则需要改写 (true)。\n"
                "请严格输出 JSON，包含两个字段：\n"
                '{"should_rewrite": boolean, "reason": "分析原因"}'
            )
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "messages": [
                        {"role": "system", "content": "你必须且只输出标准的 JSON，不能包含别的文字。"},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=10,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = self._extract_json_object(content)

            should_rewrite = bool(data.get("should_rewrite", False))
            return {
                "should_rewrite": should_rewrite,
                "action": "contextual_rewrite" if should_rewrite else "skip_rewrite",
                "reason": str(data.get("reason", "llm_decision_to_rewrite" if should_rewrite else "llm_decision_to_skip")),
                "signals": {"agentic_decision": True},
            }
        except Exception as e:
            logger.warning(f"决策 Agent 调用失败: {e}，回退到保守策略：不改写")
            return {
                "should_rewrite": False,
                "action": "skip_rewrite",
                "reason": "llm_fallback_decision",
                "signals": {"agentic_decision": False},
            }

    def _execute_rewrite_by_decision(self, query: str, decision: Dict[str, Any]) -> List[str]:
        if not decision.get("should_rewrite", False):
            return [query]
        return self._rewrite_with_llm(query)

    def _rewrite_with_llm(self, query: str) -> List[str]:
        # 优先尝试 LLM 生成多路改写，失败时回退到规则改写
        try:
            import requests

            base_url = str(self.config.llm.base_url).rstrip("/")
            model = self.config.llm.primary_model
            api_key = self.config.llm.api_key

            if not base_url or not model or not api_key:
                raise ValueError("LLM 配置不完整")

            prompt = (
                "你是法律检索查询改写器。请输出严格 JSON，字段 rewrites 为字符串数组，"
                "包含 2-4 个中文改写，覆盖：关键词扩展、法律术语标准化、同义问法。"
                "要求：与原始语义严格等价，不得引入新的案由、争议焦点或额外事实。"
                f"原始问题：{query}"
            )
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0.2,
                    "max_tokens": 500,
                    "messages": [
                        {"role": "system", "content": "你只输出 JSON，不要输出其他文本。"},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=20,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = self._extract_json_object(content)
            rewrites = [str(x).strip() for x in data.get("rewrites", []) if str(x).strip()]
            if rewrites:
                return list(dict.fromkeys([query] + rewrites))[:4]
        except Exception as e:
            logger.warning(f"Query 改写 LLM 调用失败，使用规则回退: {e}")

        normalized = query
        normalized = normalized.replace("怎么判", "法律责任如何认定")
        normalized = normalized.replace("怎么办", "适用法律条款与处理路径")
        normalized = normalized.replace("赔多少", "损害赔偿责任范围与计算标准")

        fallback = [
            query,
            f"{query} 适用法律条款",
            f"{normalized}",
            f"{query} 司法解释",
        ]
        return list(dict.fromkeys([x.strip() for x in fallback if x.strip()]))[:4]

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        import json

        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.replace("json", "", 1).strip()

        try:
            return json.loads(stripped)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    def _route_query(
        self,
        original_query: str,
        rewrites: List[str],
        decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = " ".join([original_query] + rewrites)
        short_query = self._is_short_query(original_query)
        decision = decision or self._decide_rewrite_strategy(original_query)

        need_retrieval = True
        if any(x in text for x in ["你好", "谢谢", "再见", "hi", "hello"]):
            need_retrieval = False

        if any(x in text for x in ["案例", "判例", "类案", "裁判"]):
            focus = "case"
        elif any(x in text for x in ["事实", "证据", "时间线", "经过", "行为"]):
            focus = "fact_extraction"
        else:
            focus = "law_article"

        keywords = [
            token for token in re.split(r"[\s,，。；;、:：()（）]+", original_query)
            if token and len(token) >= 2
        ]

        return {
            "need_retrieval": need_retrieval,
            "focus": focus,
            "keywords": keywords[:12],
            "use_hybrid": True,  # 强制开启混合检索，短文本专业词更需要 BM25 的精确召回补充
            "rewrite_mode": "contextual" if decision.get("should_rewrite") else "skip",
            "rewrite_reason": decision.get("reason", "unknown"),
            "rewrite_action": decision.get("action", "skip_rewrite"),
        }

    def _is_short_query(self, query: str) -> bool:
        q = query.strip()
        if not q:
            return True

        tokens = [t for t in re.split(r"[\s,，。；;、:：()（）]+", q) if t]
        return len(q) <= 12 or len(tokens) <= 3


class EmbeddingModel:
    def __init__(self):
        self.config = get_config()
        self._init_model()

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
            logger.info(f"从本地加载嵌入模型: {local_path}")
            model_path = local_path
        else:
            logger.info(f"从 Hugging Face 加载嵌入模型: {model_name}")
            model_path = model_name
        
        try:
            from transformers import AutoTokenizer, AutoModel
            from pathlib import Path
            
            if use_local and local_path and not Path(local_path).exists():
                logger.warning(f"本地模型路径不存在: {local_path}")
                logger.info(f"回退到 Hugging Face: {model_name}")
                model_path = model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("使用 GPU 加速嵌入计算")
            
            logger.info("BGE 嵌入模型加载成功")
            
        except Exception as e:
            logger.error(f"加载 BGE 模型失败: {e}")
            self.model = None
            self.tokenizer = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.config.rag.use_ollama:
            return self._encode_ollama(texts)

        if self.model is None or self.tokenizer is None:
            return [[0.0] * self.config.rag.vector_db.dimension for _ in texts]
        
        try:
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # BGE-M3 通常使用 [CLS] token 或 mean pooling
                embeddings = model_output.last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"编码失败: {e}")
            return [[0.0] * self.config.rag.vector_db.dimension for _ in texts]

    def _encode_ollama(self, texts: List[str]) -> List[List[float]]:
        import requests
        import time
        embeddings = []
        
        # 预处理：过滤出有效的文本，避免发向服务器导致崩溃
        import re
        valid_texts = []
        for t in texts:
            if t and t.strip():
                # 过滤掉无法见字符、极度生僻的控制字符（这些可能导致 LLM 分词出 NaN）
                cleaned_t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', t)
                valid_texts.append(cleaned_t)
            else:
                valid_texts.append("")
                
        # 提取真正需要发送的非空文本
        texts_to_send = [t for t in valid_texts if t]
        
        if not texts_to_send:
            return [[0.0] * self.config.rag.vector_db.dimension for _ in texts]
            
        # 使用独立的 Session 并禁用 Keep-Alive，防止长连接引起的 Ollama HTTP 500
        headers = {"Connection": "close"}
        
        try:
            # 尝试使用最新的 /api/embed (支持批量编码)
            # Ollama 对于非常大的批量可能会超时，甚至在后端崩溃并返回 500
            # 减小发送批次大小，由上层的批处理保护
            with requests.Session() as session:
                response = session.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": texts_to_send},
                    headers=headers,
                    timeout=60 # 重新增加一些超时时间
                )
            if response.status_code == 200:
                valid_embeddings = response.json().get("embeddings", [])
                
                # 重新映射回包含空字符串的原始列表顺序
                return_embeddings = []
                v_idx = 0
                for t in valid_texts:
                    if t and t.strip() and v_idx < len(valid_embeddings):
                        return_embeddings.append(valid_embeddings[v_idx])
                        v_idx += 1
                    else:
                        return_embeddings.append([0.0] * self.config.rag.vector_db.dimension)
                return return_embeddings
            else:
                logger.warning(f"Ollama /api/embed 发生异常: HTTP {response.status_code}，尝试降级为逐条请求")
        except Exception as e:
            logger.warning(f"Ollama /api/embed 发生异常: {e}，尝试降级为逐条请求")
            
        # 降级到逐条循环请求方案
        for text in texts:
            if not text or not text.strip():
                # 处理空字符串，避免 Ollama 因为空 prompt 返回 500
                embeddings.append([0.0] * self.config.rag.vector_db.dimension)
                continue
                
            retry_count = 0
            while retry_count < 3:
                try:
                    with requests.Session() as session:
                        response = session.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.model_name, "prompt": text},
                            headers=headers,
                            timeout=15 # 设置超时，避免卡死死锁
                        )
                    
                    if response.status_code == 200:
                        embeddings.append(response.json()["embedding"])
                        break
                    else:
                        # 如遇到500等错误，可能引擎过载，等待一下再重试
                        logger.warning(f"Ollama 编码警告: HTTP {response.status_code}，将在 10 秒后重试...")
                        time.sleep(5)
                        retry_count += 1
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Ollama 连接超时/失败，正在重试 ({retry_count}/3): {e}")
                    time.sleep(5)
                    
            if retry_count == 3:
                logger.error(f"Ollama 编码彻底失败 ({len(text)} 字符)")
                embeddings.append([0.0] * self.config.rag.vector_db.dimension)
                
        return embeddings

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]


class Reranker:
    def __init__(self):
        self.config = get_config()
        self._ollama_rerank_supported: Optional[bool] = None
        self._init_model()

    def _init_model(self):
        if self.config.rag.use_ollama:
            self.base_url = self.config.rag.ollama.base_url
            self.model_name = self.config.rag.ollama.reranker_model
            logger.info(f"使用 Ollama 加载重排序模型: {self.model_name}")
            return

        use_local = self.config.rag.use_local_model
        local_path = self.config.rag.reranker_model_path
        model_name = self.config.rag.reranker_model
        
        if use_local and local_path:
            logger.info(f"从本地加载重排序模型: {local_path}")
            model_path = local_path
        else:
            logger.info(f"从 Hugging Face 加载重排序模型: {model_name}")
            model_path = model_name

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        if self.config.rag.use_ollama:
            return self._rerank_ollama(query, chunks)[:top_k]

        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        if model is None or tokenizer is None:
            for chunk in chunks:
                # 若无重排模型，回退到原始检索向量分（一般为 0.6~0.9 左右），避免暴露仅有 0.01 左右的 RRF 排序分
                chunk["rerank_score"] = float(chunk.get("original_score", chunk.get("score", 0.0)))
            sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return sorted_chunks[:top_k]

        try:
            pairs = [[query, chunk["content"]] for chunk in chunks]
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                scores = model(**inputs).logits.view(-1,).float()
                scores = torch.sigmoid(scores).cpu().numpy().tolist()

            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = score

            sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            return sorted_chunks[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            for chunk in chunks:
                chunk["rerank_score"] = float(chunk.get("original_score", chunk.get("score", 0.0)))
            sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return sorted_chunks[:top_k]

    def _rerank_ollama(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import requests

        # 标准 Ollama 默认不一定支持 /api/rerank。
        # 这里优先尝试 rerank 端点；若返回 404，自动回退到 /api/generate 打分重排。
        try:
            if self._ollama_rerank_supported is not False:
                response = requests.post(
                    f"{self.base_url}/api/rerank",
                    json={
                        "model": self.model_name,
                        "query": query,
                        "documents": [c["content"] for c in chunks]
                    },
                    timeout=20
                )

                if response.status_code == 200:
                    self._ollama_rerank_supported = True
                    results = response.json().get("results", [])
                    for res in results:
                        idx = res.get("index")
                        if isinstance(idx, int) and 0 <= idx < len(chunks):
                            chunks[idx]["rerank_score"] = float(res.get("relevance_score", 0.0))
                    for chunk in chunks:
                        if "rerank_score" not in chunk:
                            chunk["rerank_score"] = float(chunk.get("original_score", chunk.get("score", 0.0)))
                    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

                if response.status_code == 404:
                    self._ollama_rerank_supported = False
                    logger.warning("Ollama 不支持 /api/rerank，已切换为 /api/generate 重排模式")
                else:
                    logger.warning(f"Ollama /api/rerank 异常: HTTP {response.status_code}，尝试 generate 重排")

            return self._rerank_with_ollama_generate(query, chunks)
        except Exception as e:
            logger.error(f"Ollama 重排序异常: {e}")
            return self._rerank_with_ollama_generate(query, chunks)

    def _rerank_with_ollama_generate(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import requests

        headers = {"Connection": "close"}
        with requests.Session() as session:
            for chunk in chunks:
                prompt = self._build_rerank_prompt(query, str(chunk.get("content", "")))

                score = None
                for attempt in range(2):
                    try:
                        response = session.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0,
                                    "num_predict": 8,
                                }
                            },
                            headers=headers,
                            timeout=25,
                        )
                        if response.status_code == 200:
                            text = str(response.json().get("response", ""))
                            score = self._parse_score(text)
                            break

                        # 429/5xx 做一次轻量重试
                        if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                            time.sleep(0.2)
                            continue
                        break
                    except Exception:
                        if attempt == 0:
                            time.sleep(0.2)
                            continue
                        break

                base_score = float(chunk.get("original_score", chunk.get("score", 0.0)))
                if score is None:
                    chunk["rerank_score"] = base_score
                else:
                    # 仍保留向量分作为先验，避免生成式评分抖动。
                    chunk["rerank_score"] = (0.8 * base_score) + (0.2 * score)

                # 避免高并发短间隔请求打满 Ollama
                time.sleep(0.05)

        return sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    def _build_rerank_prompt(self, query: str, document: str) -> str:
        return (
            "你是法律检索重排器。请评估 Document 对 Query 的相关性，"
            "只输出 0 到 1 之间的小数，不要输出其它内容。\n"
            f"Query: {query}\n"
            f"Document: {document}\n"
            "Score:"
        )

    def _parse_score(self, text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            score = float(match.group(0))
            # 兼容输出 0-100 的情况
            if score > 1.0:
                score = score / 100.0
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            return score
        except Exception:
            return None

    def _fallback_rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """当外部 rerank 不可用时，使用词项覆盖率 + 初始分数进行稳健降级。"""
        terms = self._extract_terms(query)
        term_count = max(len(terms), 1)

        for chunk in chunks:
            base_score = float(chunk.get("original_score", chunk.get("score", 0.0)))
            doc = " ".join([
                str(chunk.get("law_name", "")),
                str(chunk.get("article_num", "")),
                str(chunk.get("content", "")),
            ]).lower()

            hit = 0
            for t in terms:
                if t and t in doc:
                    hit += 1
            lexical_score = hit / term_count

            # 稳健融合：保留向量相似度主导，同时用词项命中修正排序。
            chunk["rerank_score"] = (0.9 * base_score) + (0.1 * lexical_score)

        return sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    def _extract_terms(self, query: str) -> List[str]:
        tokens = [t.strip().lower() for t in re.split(r"[\s,，。；;、:：()（）]+", query) if t.strip()]
        if tokens:
            return list(dict.fromkeys(tokens))[:16]

        # 中文无空格时回退到双字词
        chars = [query[i:i + 2].lower() for i in range(0, max(len(query) - 1, 1))]
        return list(dict.fromkeys([c for c in chars if c.strip()]))[:16]

class LegalRAG:
    def __init__(self):
        self.config = get_config()
        self.embedding_model = EmbeddingModel()
        self.reranker = Reranker()
        self.query_router = QueryRewriterRouter()
        self.vector_db = VectorDBManager()

    def build_index(self, chunks: List[LawChunk], batch_size: int = 32):
        logger.info(f"开始构建索引，共 {len(chunks)} 个 chunks")
        
        # 分批处理以防止长连接超时和显存溢出
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch_chunks]
            
            logger.info(f"正在处理批次: {i}/{len(chunks)}")
            embeddings = self.embedding_model.encode(texts)
            self.vector_db.insert_chunks(batch_chunks, embeddings)
            
        logger.info("索引构建完成")

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        logger.info(f"执行检索查询: {query[:50]}...")

        query_plan = self.query_router.rewrite_and_route(query, filters)
        route_info = query_plan.get("route", {})
        if not route_info.get("need_retrieval", True):
            logger.info("路由判定无需检索，直接返回空结果")
            return []

        rewritten_queries = query_plan.get("rewrites", [])
        primary_query = query_plan.get("rewrite_primary", query)
        use_hybrid = bool(route_info.get("use_hybrid", True))

        query_tokens = [t for t in re.split(r"[\s,，。；;、:：()（）]+", query) if t]
        max_rewrites = 1 if not use_hybrid else (2 if (len(query) <= 8 and len(query_tokens) <= 2) else 4)
        active_queries = rewritten_queries[:max_rewrites]
        logger.info(
            f"改写查询数量: {len(active_queries)}，路由焦点: {route_info.get('focus')}，"
            f"混合检索: {'开启' if use_hybrid else '关闭'}，改写模式: {route_info.get('rewrite_mode', 'unknown')}"
        )

        merged_filters = self._merge_filters(filters, query_plan.get("metadata_hints", {}))

        top_k_initial = self.config.rag.retrieval.top_k_initial
        dense_results = []
        sparse_results = []
        for q in active_queries:
            query_embedding = self.embedding_model.encode_single(q)
            dense = self.vector_db.search(query_embedding, top_k_initial, merged_filters)
            for item in dense:
                item["retrieval_channel"] = "dense"
            dense_results.extend(dense)

            if use_hybrid:
                sparse = self.vector_db.keyword_search(q, top_k_initial, merged_filters)
                for item in sparse:
                    item["retrieval_channel"] = "sparse"
                sparse_results.extend(sparse)

        initial_results = self._rrf_fuse(
            dense_results,
            sparse_results,
            top_k_initial * 3,
            dense_weight=1.0,
            sparse_weight=0.2,
        )
        initial_results = self._apply_temporal_filter(initial_results, query_plan.get("metadata_hints", {}))

        logger.info(
            f"初步检索到 {len(initial_results)} 个结果 (dense={len(dense_results)}, sparse={len(sparse_results)})"
        )
        
        top_k_final = self.config.rag.retrieval.top_k_final
        reranked_results = self.reranker.rerank(primary_query, initial_results, top_k_final)
        
        retrieval_results = []
        for idx, result in enumerate(reranked_results):
            metadata = result.get("metadata") or {}
            metadata = dict(metadata)
            metadata["route_focus"] = route_info.get("focus", "law_article")
            metadata["context_tier"] = self._context_tier_by_rank(idx)
            metadata["query_rewrite"] = primary_query

            chunk = LawChunk(
                chunk_id=result["chunk_id"],
                law_name=result["law_name"],
                article_num=result["article_num"],
                content=result["content"],
                level=result["level"],
                metadata=metadata
            )
            retrieval_result = RetrievalResult(
                chunk=chunk,
                score=result.get("rerank_score", result.get("score", 0.0)),
                rank=idx + 1
            )
            retrieval_results.append(retrieval_result)
        
        logger.info(f"最终返回 {len(retrieval_results)} 个结果")
        return retrieval_results

    def _merge_filters(self, user_filters: Optional[Dict[str, Any]], hints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {}
        if user_filters:
            merged.update(user_filters)
        if hints:
            merged.update(hints)
        return merged

    def _rrf_fuse(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        limit: int,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        def _ranked(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)

        fused = {}
        for source_name, items in [("dense", _ranked(dense_results)), ("sparse", _ranked(sparse_results))]:
            source_weight = dense_weight if source_name == "dense" else sparse_weight
            for rank_idx, item in enumerate(items, start=1):
                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue
                key = str(chunk_id)
                if key not in fused:
                    fused[key] = dict(item)
                    fused[key]["original_score"] = float(item.get("score", 0.0))  # 记录原始检索分数（通常在 0.6 ~ 0.9 左右）
                    fused[key]["score"] = 0.0
                fused[key]["score"] += source_weight * (1.0 / (rrf_k + rank_idx))
                fused[key][f"{source_name}_rank"] = rank_idx

        merged = list(fused.values())
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:limit]

    def _apply_temporal_filter(self, results: List[Dict[str, Any]], hints: Dict[str, Any]) -> List[Dict[str, Any]]:
        event_date = hints.get("event_date") if hints else None
        if not event_date:
            return results

        event_dt = self._parse_date(event_date)
        if not event_dt:
            return results

        filtered = []
        for item in results:
            eff = self._parse_date(str(item.get("effective_date", "")))
            rep = self._parse_date(str(item.get("repeal_date", "")))
            if eff and event_dt < eff:
                continue
            if rep and event_dt > rep:
                continue
            filtered.append(item)
        return filtered

    def _parse_date(self, text: str) -> Optional[datetime]:
        if not text:
            return None
        candidates = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%Y-%m",
            "%Y/%m",
            "%Y",
        ]
        for fmt in candidates:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return None

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
            relevant_laws.append({
                "law_name": law_name,
                "articles": [r.chunk.article_num for r in law_results],
                "contents": [r.chunk.content for r in law_results],
                "max_score": max(r.score for r in law_results)
            })
        
        relevant_laws.sort(key=lambda x: x["max_score"], reverse=True)
        return relevant_laws[:top_k]

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.vector_db.get_collection_stats()

    def reset_index(self):
        logger.warning("重置向量数据库索引")
        self.vector_db.delete_collection()
        self._init_vector_db()

    def _init_vector_db(self):
        self.vector_db = VectorDBManager()