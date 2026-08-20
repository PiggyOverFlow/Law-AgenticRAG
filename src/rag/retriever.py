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
from src.llm import LLMBackend
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

    def _state_get(self, state, key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _state_update(self, state, payload: Any) -> None:
        if payload is None:
            return

        if isinstance(payload, dict):
            data = payload
        elif hasattr(payload, "__dict__"):
            data = payload.__dict__
        else:
            return

        if isinstance(state, dict):
            state.update(data)
            return

        for key, value in data.items():
            setattr(state, key, value)

    def invoke(self, initial_state) -> dict:
        if isinstance(initial_state, dict):
            state = self._state_class(**initial_state)
        else:
            state = initial_state

        current = self._entry
        steps = 0
        visit_counts: Dict[str, int] = {}
        configured_limit = self._state_get(state, "graph_max_steps")
        fallback_limit = max(8, int(self._state_get(state, "max_iterations", 5)) * 6)
        max_steps = max(1, int(configured_limit or fallback_limit))

        while current:
            if isinstance(current, _END):
                break

            steps += 1
            if steps > max_steps:
                logger.warning("StateGraph 提前终止：超过最大执行步数 %s", max_steps)
                if isinstance(state, dict):
                    state["graph_terminated_reason"] = "max_steps_exceeded"
                    state["graph_steps"] = steps - 1
                    state["graph_visit_counts"] = dict(visit_counts)
                break

            node_func = self._nodes.get(current)
            if node_func is None:
                break

            visit_counts[current] = visit_counts.get(current, 0) + 1
            result = node_func(state)
            self._state_update(state, result)

            if current in self._conditional:
                router, mapping = self._conditional[current]
                next_key = router(state)
                current = mapping.get(next_key, next_key)
                continue

            next_list = self._edges.get(current, [])
            current = next_list[0] if next_list else None

        if isinstance(state, dict):
            state["graph_steps"] = min(steps, max_steps)
            state["graph_visit_counts"] = dict(visit_counts)
            return dict(state)

        return state.__dict__


StateGraph = _StateGraph


@runtime_checkable
class Embeddings(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


@dataclass
class RetrievalTrace:
    """检索轨迹，记录每次检索的详细信息"""
    round_index: int
    thought: str
    query_used: str
    retrieval_query: str = ""
    rewritten_from: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    vector_hits: int = 0
    kept_hits: int = 0
    contrast_query: str = ""
    delta: str = ""
    focus_terms: List[str] = field(default_factory=list)
    trap_terms: List[str] = field(default_factory=list)


@dataclass
class ContrastiveExample:
    """对比学习示例，用于区分相似查询"""
    target_query: str
    contrast_query: str = ""
    delta: str = ""
    focus_terms: List[str] = field(default_factory=list)
    trap_terms: List[str] = field(default_factory=list)
    contrast_type: str = "semantic_boundary"
    enhanced_query: str = ""


@dataclass
class IssuePlan:
    """问题查询计划，包含主问题、子问题和检索查询"""
    main_issue: str = ""
    sub_issues: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_queries: List[str] = field(default_factory=list)
    focus_terms: List[str] = field(default_factory=list)
    ignore_terms: List[str] = field(default_factory=list)
    primary_claim: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalElements:
    """法律要素，包含主体、关系和请求权"""
    subjects: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)


class QueryAgent:
    """查询智能体，负责查询重写、关键词提取、法律要素提取和查询规划"""
    
    def __init__(self):
        """初始化查询智能体"""
        self.config = get_config()
        self.llm_backend = LLMBackend()

    def should_rewrite(self, query: str, context: str = "") -> Dict[str, Any]:
        """判断查询是否需要重写
        
        Args:
            query: 查询文本
            
        Returns:
            Dict[str, Any]: 包含 should_rewrite 和 reason 的字典
        """
        q = (query or "").strip()
        if not q:
            return {"should_rewrite": False, "reason": "empty_query"}

        llm_result = self._llm_judge_query_completeness(q, context=context)
        if llm_result is not None:
            return llm_result

        incomplete_markers = ["这个", "这种", "该", "上述", "前述", "他", "她", "它", "怎么办", "如何处理","如何"]
        context_hint = bool(str(context or "").strip())
        should_rewrite = any(marker in q for marker in incomplete_markers) and (len(q) <= 24 or context_hint)
        return {
            "should_rewrite": should_rewrite,
            "reason": "heuristic_incomplete_query" if should_rewrite else "heuristic_complete_query",
        }

    def rewrite_query(self, query: str, context: str = "", force: bool = False) -> str:
        """重写查询文本，使其更完整和具体
        
        Args:
            query: 原始查询文本
            force: 是否强制重写
            
        Returns:
            str: 重写后的查询文本
        """
        q = (query or "").strip()
        if not q:
            return q

        if not force:
            decision = self.should_rewrite(q, context=context)
            if not decision.get("should_rewrite", False):
                return q

        rewritten = self._llm_rewrite_query(q, context=context)
        if rewritten:
            return rewritten

        fallback = q
        fallback = fallback.replace("怎么办", "适用哪些法律条款以及如何处理")
        fallback = fallback.replace("怎么判", "司法实践中如何认定与裁判")
        if context.strip():
            fallback = f"基于以下上下文：{context[:240]}；当前问题：{fallback}"
        if fallback == q:
            fallback = f"{q} 相关法律条文 司法解释 适用要点"
        return re.sub(r"\s+", " ", fallback).strip()

    def extract_keywords(self, query: str, max_keywords: int = 8) -> List[str]:
        """从查询中提取关键词
        
        Args:
            query: 查询文本
            max_keywords: 最大关键词数量
            
        Returns:
            List[str]: 关键词列表
        """
        q = (query or "").strip()
        if not q:
            return []

        llm_keywords = self._llm_extract_keywords(q, max_keywords=max_keywords)
        if llm_keywords:
            return llm_keywords

        return self._heuristic_extract_keywords(q, max_keywords=max_keywords)

    def build_contrastive_example(
        self,
        query: str,
        legal_elements: Optional[LegalElements] = None,
        claim: Optional[Dict[str, Any]] = None,
    ) -> ContrastiveExample:
        """构建对比学习示例，包含目标词和陷阱词
        
        Args:
            query: 查询文本
            legal_elements: 法律要素
            claim: 请求权
            
        Returns:
            ContrastiveExample: 对比学习示例
        """
        q = (query or "").strip()
        if not q:
            return ContrastiveExample(target_query="")

        llm_example = self._llm_build_contrastive_example(q, legal_elements=legal_elements, claim=claim)
        if llm_example:
            return llm_example

        return self._heuristic_build_contrastive_example(q, legal_elements=legal_elements, claim=claim)

    def extract_legal_elements(self, query: str, context: str = "") -> LegalElements:
        """从查询中提取法律要素（主体、关系、请求权）
        
        Args:
            query: 查询文本
            context: 上下文信息
            
        Returns:
            LegalElements: 法律要素对象
        """
        q = (query or "").strip()
        if not q:
            return LegalElements()

        llm_elements = self._llm_extract_legal_elements(q, context=context)
        if llm_elements:
            return llm_elements

        return self._heuristic_extract_legal_elements(q, context=context)

    def plan_issue_queries(
        self,
        query: str,
        context: str = "",
        legal_elements: Optional[LegalElements] = None,
    ) -> IssuePlan:
        """规划问题查询，生成多个检索查询以提高召回率
        
        Args:
            query: 原始查询文本
            context: 上下文信息
            legal_elements: 法律要素
            
        Returns:
            IssuePlan: 问题查询计划
        """
        q = (query or "").strip()
        if not q:
            return IssuePlan()

        if legal_elements is None:
            legal_elements = self.extract_legal_elements(q, context=context)

        llm_plan = self._llm_plan_issue_queries(q, context=context, legal_elements=legal_elements)
        if llm_plan:
            return llm_plan

        return self._heuristic_plan_issue_queries(q, context=context, legal_elements=legal_elements)

    def _llm_judge_query_completeness(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        prompt = (
            "你是法律检索查询分析器。判断 query 是否信息完整，是否需要先改写。"
            "若包含模糊指代、上下文缺失、语义过短，建议改写。"
            "只输出 JSON：{\"should_rewrite\": bool, \"reason\": \"...\"}。\n"
            f"query: {query}\n"
            f"context: {context}"
        )
        data = self._call_llm_json(prompt, temperature=0.1, max_tokens=200)
        if not data:
            return None
        return {
            "should_rewrite": bool(data.get("should_rewrite", False)),
            "reason": str(data.get("reason", "llm_decision")),
        }

    def _llm_rewrite_query(self, query: str, context: str = "") -> Optional[str]:
        prompt = (
            "你是法律检索改写器。将 query 改写成更完整、更可检索的一句话，"
            "保持原意，不添加新事实。只输出 JSON：{\"rewrite\": \"...\"}。\n"
            f"query: {query}\n"
            f"context: {context}"
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

    def _llm_extract_legal_elements(self, query: str, context: str = "") -> Optional[LegalElements]:
        prompt = (
            "你是法律案件结构化分析器。请从问题与上下文中抽取主体、主体关系、请求项。只输出 JSON。\n"
            "格式：\n"
            "{\n"
            '  "subjects":[{"name":"...","type":"自然人/公司/机构/其他","role":"..."}],\n'
            '  "relations":[{"from":"...","to":"...","type":"...","description":"..."}],\n'
            '  "claims":[{"claimant":"...","against":"...","type":"...","object":"...","basis":"...","priority":1}]\n'
            "}\n"
            "约束：\n"
            "- role 必须尽量体现法律身份，如债务人、借款人、受让人、保证人、股东、公司等\n"
            "- relations 描述法律关系，不要只写自然语言叙事\n"
            "- claims 是谁向谁主张什么，priority 越小越优先\n"
            f"问题: {query}\n"
            f"上下文: {context}"
        )
        data = self._call_llm_json(prompt, temperature=0.1, max_tokens=800)
        if not data:
            return None
        return self._normalize_legal_elements(data)

    def _llm_build_contrastive_example(
        self,
        query: str,
        legal_elements: Optional[LegalElements] = None,
        claim: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContrastiveExample]:
        contrast_cfg = getattr(self.config.rag, "contrastive", None)
        max_focus = max(2, int(getattr(contrast_cfg, "max_focus_terms", 6)))
        max_trap = max(2, int(getattr(contrast_cfg, "max_trap_terms", 6)))
        subject_context = self._format_legal_elements_for_prompt(legal_elements)
        claim_context = self._format_claim_for_prompt(claim)
        prompt = (
            "你是法律检索中的对比样例构造器。请围绕给定 claim 构造一个“主体关系相近但结论可能不同”的法律对比问题，"
            "并指出真正决定法条适用分歧的关键差异。\n"
            "只输出 JSON，字段如下：\n"
            "{\n"
            '  "contrast_query": "...",\n'
            '  "delta": "...",\n'
            '  "focus_terms": ["..."],\n'
            '  "trap_terms": ["..."],\n'
            '  "contrast_type": "法律关系错配/责任主体错配/构成要件错配/证据效力错配/程序阶段错配",\n'
            '  "enhanced_query": "..."\n'
            "}\n"
            "约束：\n"
            "- contrast_query 必须与原问题高度相似，但检索目标应不同\n"
            f"- focus_terms 最多 {max_focus} 个，trap_terms 最多 {max_trap} 个\n"
            "- focus_terms 是应优先关注的法律事实、证据或裁判要点\n"
            "- trap_terms 是容易误召回的干扰概念、错误法律路径或泛化表述\n"
            "- 如果存在主体或请求信息，必须优先围绕 claim 里的主体资格、主体关系、请求边界构造对比\n"
            "- enhanced_query 写成一句检索意图摘要，强调该关注什么、避免什么\n"
            f"原始问题: {query}\n"
            f"主体结构: {subject_context}\n"
            f"核心请求: {claim_context}"
        )
        data = self._call_llm_json(prompt, temperature=0.2, max_tokens=512)
        if not data:
            return None

        contrast_query = str(data.get("contrast_query", "")).strip()
        delta = str(data.get("delta", "")).strip()
        focus_terms = self._post_process_terms(data.get("focus_terms", []), max_keywords=max_focus)
        trap_terms = self._post_process_terms(data.get("trap_terms", []), max_keywords=max_trap)
        enhanced_query = str(data.get("enhanced_query", "")).strip()
        contrast_type = str(data.get("contrast_type", "semantic_boundary")).strip() or "semantic_boundary"
        if not any([contrast_query, delta, focus_terms, trap_terms, enhanced_query]):
            return None

        return ContrastiveExample(
            target_query=query,
            contrast_query=contrast_query,
            delta=delta,
            focus_terms=focus_terms,
            trap_terms=trap_terms,
            contrast_type=contrast_type,
            enhanced_query=enhanced_query or self._compose_enhanced_query(query, delta, focus_terms, trap_terms),
        )

    def _heuristic_build_contrastive_example(
        self,
        query: str,
        legal_elements: Optional[LegalElements] = None,
        claim: Optional[Dict[str, Any]] = None,
    ) -> ContrastiveExample:
        contrast_cfg = getattr(self.config.rag, "contrastive", None)
        max_focus = max(2, int(getattr(contrast_cfg, "max_focus_terms", 6)))
        max_trap = max(2, int(getattr(contrast_cfg, "max_trap_terms", 6)))
        claim_query = self._build_claim_query(claim)
        target_query = claim_query or query
        terms = self.extract_keywords(target_query, max_keywords=max(max_focus + max_trap, 8))
        focus_terms = terms[:max_focus]
        trap_terms = self._infer_trap_terms(target_query, max_trap=max_trap)
        delta = self._infer_delta(target_query, focus_terms, trap_terms)
        contrast_query = self._infer_contrast_query(target_query, trap_terms)

        return ContrastiveExample(
            target_query=target_query,
            contrast_query=contrast_query,
            delta=delta,
            focus_terms=focus_terms,
            trap_terms=trap_terms,
            contrast_type="heuristic_legal_boundary",
            enhanced_query=self._compose_enhanced_query(target_query, delta, focus_terms, trap_terms),
        )

    def _llm_plan_issue_queries(
        self,
        query: str,
        context: str = "",
        legal_elements: Optional[LegalElements] = None,
    ) -> Optional[IssuePlan]:
        subject_context = self._format_legal_elements_for_prompt(legal_elements)
        prompt = (
            "你是法律 Agentic-RAG 的争点规划器。请基于主体、主体关系、请求项，先确定主争点和子争点，"
            "再为每个高优先级 claim 设计检索 query。只输出 JSON。\n"
            "JSON 格式：\n"
            "{\n"
            '  "main_issue": "...",\n'
            '  "sub_issues": [{"issue":"...","query":"...","priority":1}],\n'
            '  "retrieval_queries": ["..."],\n'
            '  "focus_terms": ["..."],\n'
            '  "ignore_terms": ["..."],\n'
            '  "primary_claim": {"claimant":"...","against":"...","type":"...","object":"...","basis":"...","priority":1}\n'
            "}\n"
            "约束：\n"
            "- 必须优先围绕 claims 规划，不要只按表面事实拆题\n"
            "- main_issue 必须是整案最需要优先检索的法律问题\n"
            "- sub_issues 只保留 2-5 个高价值子争点，priority 越小优先级越高\n"
            "- retrieval_queries 输出 3-8 条，用于法律法规检索，避免泛化词\n"
            "- focus_terms 是应重点覆盖的概念；ignore_terms 是应避免被误召回的概念\n"
            f"问题: {query}\n"
            f"上下文: {context}\n"
            f"主体结构: {subject_context}"
        )
        data = self._call_llm_json(prompt, temperature=0.2, max_tokens=700)
        if not data:
            return None

        main_issue = str(data.get("main_issue", "")).strip()
        sub_issues = data.get("sub_issues", [])
        retrieval_queries = data.get("retrieval_queries", [])
        focus_terms = self._post_process_terms(data.get("focus_terms", []), max_keywords=8)
        ignore_terms = self._post_process_terms(data.get("ignore_terms", []), max_keywords=8)
        primary_claim = self._normalize_claim(data.get("primary_claim", {}))

        normalized_sub_issues: List[Dict[str, Any]] = []
        if isinstance(sub_issues, list):
            for item in sub_issues[:6]:
                if not isinstance(item, dict):
                    continue
                issue = str(item.get("issue", "")).strip()
                issue_query = str(item.get("query", "")).strip()
                priority = item.get("priority", 9)
                try:
                    priority = int(priority)
                except Exception:
                    priority = 9
                if not issue and not issue_query:
                    continue
                normalized_sub_issues.append(
                    {"issue": issue, "query": issue_query, "priority": priority}
                )

        normalized_queries: List[str] = []
        if isinstance(retrieval_queries, list):
            for item in retrieval_queries[:10]:
                value = str(item).strip()
                if value:
                    normalized_queries.append(value)

        if not main_issue and normalized_sub_issues:
            normalized_sub_issues.sort(key=lambda x: (x.get("priority", 9), x.get("issue", "")))
            main_issue = normalized_sub_issues[0].get("issue", "") or normalized_sub_issues[0].get("query", "")

        if not any([main_issue, normalized_sub_issues, normalized_queries, focus_terms, ignore_terms]):
            return None

        if main_issue and main_issue not in normalized_queries:
            normalized_queries.insert(0, main_issue)
        for item in sorted(normalized_sub_issues, key=lambda x: (x.get("priority", 9), x.get("issue", ""))):
            issue_query = str(item.get("query", "")).strip()
            if issue_query and issue_query not in normalized_queries:
                normalized_queries.append(issue_query)

        return IssuePlan(
            main_issue=main_issue,
            sub_issues=normalized_sub_issues,
            retrieval_queries=normalized_queries[:10],
            focus_terms=focus_terms,
            ignore_terms=ignore_terms,
            primary_claim=primary_claim,
        )

    def _heuristic_plan_issue_queries(
        self,
        query: str,
        context: str = "",
        legal_elements: Optional[LegalElements] = None,
    ) -> IssuePlan:
        seed = " ".join([query, context]).strip()
        focus_terms = self.extract_keywords(seed, max_keywords=8)
        sub_issues: List[Dict[str, Any]] = []
        retrieval_queries: List[str] = []
        claims = sorted((legal_elements.claims if legal_elements else []), key=lambda x: x.get("priority", 9))
        primary_claim = claims[0] if claims else {}

        def add_issue(issue: str, issue_query: str, priority: int) -> None:
            if not issue and not issue_query:
                return
            sub_issues.append({"issue": issue, "query": issue_query, "priority": priority})
            if issue_query and issue_query not in retrieval_queries:
                retrieval_queries.append(issue_query)

        if any(k in seed for k in ["借款", "借贷", "本金", "借条"]):
            add_issue("民间借贷债权本体及本金返还", "民间借贷 借款合同 本金返还 法律依据", 1)
        if "债权转让" in seed:
            add_issue("债权转让对债务人的效力", "债权转让 通知债务人 受让人直接起诉 法律依据", 1)
        if any(k in seed for k in ["时效", "起诉", "催收", "诉讼时效"]):
            add_issue("诉讼时效起算与中断", "民间借贷 债权转让 诉讼时效 中断 法律依据", 2)
        if any(k in seed for k in ["利息", "%", "24%", "年利率"]):
            add_issue("利息与逾期利息支持边界", "民间借贷 年利率24% 逾期利息 是否支持", 1)

        main_issue = ""
        if sub_issues:
            sub_issues.sort(key=lambda x: (x.get("priority", 9), x.get("issue", "")))
            main_issue = sub_issues[0].get("issue", "") or sub_issues[0].get("query", "")
        elif focus_terms:
            main_issue = f"{focus_terms[0]} 相关法律适用"
            retrieval_queries.append(f"{main_issue} 法律依据")

        if main_issue and main_issue not in retrieval_queries:
            retrieval_queries.insert(0, main_issue)
        claim_query = self._build_claim_query(primary_claim)
        if claim_query and claim_query not in retrieval_queries:
            retrieval_queries.insert(0, claim_query)
        if claim_query:
            main_issue = claim_query

        ignore_terms = self._infer_trap_terms(seed, max_trap=6)
        return IssuePlan(
            main_issue=main_issue,
            sub_issues=sub_issues,
            retrieval_queries=retrieval_queries[:10],
            focus_terms=focus_terms[:8],
            ignore_terms=ignore_terms,
            primary_claim=primary_claim,
        )

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

    def _heuristic_extract_legal_elements(self, query: str, context: str = "") -> LegalElements:
        seed = " ".join([query, context]).strip()
        subjects: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []

        def add_subject(name: str, role: str, subject_type: str = "其他") -> None:
            normalized = str(name or "").strip()
            if not normalized:
                return
            for item in subjects:
                if item.get("name") == normalized:
                    if role and not item.get("role"):
                        item["role"] = role
                    return
            subjects.append({"name": normalized, "type": subject_type, "role": role})

        for name in re.findall(r"[张李王赵钱孙周吴郑冯陈褚卫蒋沈韩杨朱秦尤许何吕施孔曹严华金魏陶姜][\u4e00-\u9fa5]{1,2}", seed):
            add_subject(name, "")

        role_patterns = [
            ("受让人", r"([\u4e00-\u9fa5]{2,4})作为?受让人"),
            ("债务人", r"向([\u4e00-\u9fa5]{2,4})起诉"),
            ("债权人", r"([\u4e00-\u9fa5]{2,4})将这笔债权转让"),
        ]
        for role, pattern in role_patterns:
            for name in re.findall(pattern, seed):
                add_subject(name, role)

        if "债权转让" in seed and len(subjects) >= 2:
            relations.append(
                {
                    "from": subjects[0]["name"],
                    "to": subjects[1]["name"],
                    "type": "债权转让/基础债权关系",
                    "description": "存在债权转让或基础债权关系",
                }
            )

        if "起诉" in seed and len(subjects) >= 2:
            claims.append(
                {
                    "claimant": subjects[-1]["name"],
                    "against": subjects[0]["name"],
                    "type": "请求履行债务",
                    "object": "本金及利息",
                    "basis": "基础债权关系",
                    "priority": 1,
                }
            )

        return LegalElements(subjects=subjects, relations=relations, claims=claims)

    def _normalize_legal_elements(self, data: Dict[str, Any]) -> Optional[LegalElements]:
        if not isinstance(data, dict):
            return None
        subjects_raw = data.get("subjects", [])
        relations_raw = data.get("relations", [])
        claims_raw = data.get("claims", [])

        subjects: List[Dict[str, Any]] = []
        if isinstance(subjects_raw, list):
            for item in subjects_raw[:12]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                subjects.append(
                    {
                        "name": name,
                        "type": str(item.get("type", "其他")).strip() or "其他",
                        "role": str(item.get("role", "")).strip(),
                    }
                )

        relations: List[Dict[str, Any]] = []
        if isinstance(relations_raw, list):
            for item in relations_raw[:16]:
                if not isinstance(item, dict):
                    continue
                relations.append(
                    {
                        "from": str(item.get("from", "")).strip(),
                        "to": str(item.get("to", "")).strip(),
                        "type": str(item.get("type", "")).strip(),
                        "description": str(item.get("description", "")).strip(),
                    }
                )

        claims: List[Dict[str, Any]] = []
        if isinstance(claims_raw, list):
            for item in claims_raw[:12]:
                normalized = self._normalize_claim(item)
                if normalized:
                    claims.append(normalized)

        return LegalElements(subjects=subjects, relations=relations, claims=claims)

    def _normalize_claim(self, claim: Any) -> Dict[str, Any]:
        if not isinstance(claim, dict):
            return {}
        try:
            priority = int(claim.get("priority", 9))
        except Exception:
            priority = 9
        normalized = {
            "claimant": str(claim.get("claimant", "")).strip(),
            "against": str(claim.get("against", "")).strip(),
            "type": str(claim.get("type", "")).strip(),
            "object": str(claim.get("object", "")).strip(),
            "basis": str(claim.get("basis", "")).strip(),
            "priority": priority,
        }
        if not any([normalized["claimant"], normalized["against"], normalized["type"], normalized["object"]]):
            return {}
        return normalized

    def _build_claim_query(self, claim: Optional[Dict[str, Any]]) -> str:
        if not claim:
            return ""
        parts = [
            str(claim.get("claimant", "")).strip(),
            "向",
            str(claim.get("against", "")).strip(),
            str(claim.get("type", "")).strip(),
            str(claim.get("object", "")).strip(),
            str(claim.get("basis", "")).strip(),
            "法律依据",
        ]
        value = " ".join([p for p in parts if p]).strip()
        return re.sub(r"\s+", " ", value).strip()

    def _format_legal_elements_for_prompt(self, legal_elements: Optional[LegalElements]) -> str:
        if not legal_elements:
            return "无"
        return json.dumps(
            {
                "subjects": legal_elements.subjects,
                "relations": legal_elements.relations,
                "claims": legal_elements.claims,
            },
            ensure_ascii=False,
        )

    def _format_claim_for_prompt(self, claim: Optional[Dict[str, Any]]) -> str:
        if not claim:
            return "无"
        return json.dumps(claim, ensure_ascii=False)

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

    def _infer_trap_terms(self, query: str, max_trap: int) -> List[str]:
        candidates: List[str] = []
        mapping = {
            "借款": ["货款", "代付款", "往来款"],
            "借贷": ["买卖货款", "投资款", "代付款"],
            "劳动": ["劳务关系", "合作关系", "承揽"],
            "解除劳动合同": ["协商解除", "自动离职", "劳务终止"],
            "交通事故": ["工伤", "意外事件", "治安案件"],
            "违约": ["侵权", "不当得利", "缔约过失"],
            "保证": ["共同借款", "债务加入", "代偿"],
            "夫妻共同债务": ["个人债务", "公司债务", "职务行为"],
            "仲裁": ["直接起诉", "行政复议", "调解协议"],
            "转账记录": ["付款凭证", "报销", "货款结算"],
            "借条": ["收据", "对账单", "付款申请"],
        }
        for key, values in mapping.items():
            if key in query:
                candidates.extend(values)
        if not candidates:
            candidates = ["一般付款纠纷", "普通合同争议", "无关程序规则"]
        return self._post_process_terms(candidates, max_keywords=max_trap)

    def _infer_delta(self, query: str, focus_terms: List[str], trap_terms: List[str]) -> str:
        focus = "、".join(focus_terms[:4]) or "关键构成要件"
        trap = "、".join(trap_terms[:3]) or "表面相近但法律性质不同的路径"
        return (
            f"检索时应优先围绕{focus}识别真正决定责任与法条适用的事实，"
            f"避免被{trap}等表面相似但法律性质不同的干扰信息带偏。"
        )

    def _infer_contrast_query(self, query: str, trap_terms: List[str]) -> str:
        trap = trap_terms[0] if trap_terms else "其他法律关系"
        return f"{query} 但争议焦点实际属于{trap}时应如何认定"

    def _compose_enhanced_query(
        self,
        query: str,
        delta: str,
        focus_terms: List[str],
        trap_terms: List[str],
    ) -> str:
        parts = [query]
        if focus_terms:
            parts.append("重点关注：" + "、".join(focus_terms))
        if trap_terms:
            parts.append("避免混淆：" + "、".join(trap_terms))
        if delta:
            parts.append("差异提示：" + delta)
        return "；".join(parts)

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
        if not self.llm_backend.is_available():
            return None

        try:
            content = self.llm_backend.generate(
                messages=[
                    {"role": "system", "content": "你只能输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=20,
            )
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
    """嵌入模型，用于将文本转换为向量表示"""
    
    def __init__(self):
        """初始化嵌入模型"""
        self.config = get_config()
        self._init_model()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """将文档列表转换为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        return self.encode(list(texts))

    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为向量
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 向量表示
        """
        return self.encode_single(text)

    def _init_model(self):
        """初始化嵌入模型"""
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
        """将文本列表编码为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
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
    关键词重排器，基于关键词匹配和多种评分策略对检索结果进行重排序。

    核心规则：
    1. 关键词权重采用候选集内的自适应区分度（类似 IDF），高频词自动降权。
    2. 多关键词共同命中时给予阶梯奖励，增强"证据组合"信号。
    3. 距离越小越相关，最终按距离升序排序。
    4. 支持对比学习示例，对目标词给予奖励，对陷阱词给予惩罚。
    5. 考虑路径匹配和法律优先级，提供更精准的排序。
    """

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        keywords: Optional[List[str]] = None,
        contrastive_example: Optional[ContrastiveExample] = None,
    ) -> List[Dict[str, Any]]:
        """对检索结果进行重排序
        
        Args:
            query: 查询文本
            chunks: 候选块列表
            top_k: 返回结果数量
            keywords: 关键词列表
            contrastive_example: 对比学习示例，包含目标词和陷阱词
            
        Returns:
            List[Dict[str, Any]]: 重排序后的结果列表
        """
        if not chunks:
            return []

        ranked = []
        normalized_keywords = self._normalize_keywords(keywords or [])
        contrast_cfg = getattr(get_config().rag, "contrastive", None)
        normalized_focus = self._normalize_keywords(
            (contrastive_example.focus_terms if contrastive_example else []) or []
        )
        normalized_traps = self._normalize_keywords(
            (contrastive_example.trap_terms if contrastive_example else []) or []
        )
        query_profile = self._build_query_profile(
            query=query,
            keywords=normalized_keywords,
            focus_terms=normalized_focus,
        )
        target_hit_bonus = float(getattr(contrast_cfg, "target_hit_bonus", 0.05))
        trap_hit_penalty = float(getattr(contrast_cfg, "trap_hit_penalty", 0.06))
        max_bonus = float(getattr(contrast_cfg, "max_bonus", 0.22))
        max_penalty = float(getattr(contrast_cfg, "max_penalty", 0.28))
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
            metadata = chunk.get("metadata") or {}
            path_text = " ".join(
                [
                    str(metadata.get("structure_path_text", "")),
                    str(metadata.get("structure_hierarchy_path", "")),
                    str(metadata.get("structure_locator", "")),
                ]
            ).lower()

            matched_keywords = [kw for kw in effective_keywords if kw in text_blob]
            specificity_sum = sum(keyword_idf.get(kw, 0.0) for kw in matched_keywords)
            coverage = len(matched_keywords) / max(1, len(effective_keywords))
            focus_hits = [kw for kw in normalized_focus if kw in text_blob]
            trap_hits = [kw for kw in normalized_traps if kw in text_blob]
            path_hits = [kw for kw in effective_keywords if kw in path_text]
            path_focus_hits = [kw for kw in normalized_focus if kw in path_text]
            path_match_score = min(
                0.20,
                0.04 * len(path_hits) + 0.03 * len(path_focus_hits),
            )
            legal_priority_score, legal_priority_reasons = self._compute_legal_priority(
                metadata=metadata,
                query_profile=query_profile,
            )

            # 自适应奖励：区分度越高、命中越多，奖励越大；上限防止过度改写排序。
            keyword_boost = min(
                0.35,
                0.06 * specificity_sum + 0.04 * max(0, len(matched_keywords) - 1) + 0.02 * coverage,
            )
            contrastive_bonus = min(max_bonus, target_hit_bonus * len(focus_hits))
            contrastive_penalty = min(max_penalty, trap_hit_penalty * len(trap_hits))

            adjusted_distance = (
                base_distance
                - keyword_boost
                - path_match_score
                - legal_priority_score
                - contrastive_bonus
                + contrastive_penalty
            )

            item = dict(chunk)
            item["raw_distance"] = base_distance
            item["keyword_hits"] = matched_keywords
            item["keyword_boost"] = keyword_boost 
            item["path_hits"] = path_hits
            item["path_focus_hits"] = path_focus_hits
            item["path_match_score"] = path_match_score
            item["legal_priority_score"] = legal_priority_score
            item["legal_priority_reasons"] = legal_priority_reasons
            item["focus_hits"] = focus_hits
            item["trap_hits"] = trap_hits
            item["contrastive_bonus"] = contrastive_bonus
            item["contrastive_penalty"] = contrastive_penalty
            item["rerank_score"] = adjusted_distance
            item["score"] = base_distance
            item["keyword_idf"] = {k: round(keyword_idf.get(k, 0.0), 3) for k in matched_keywords}
            item["effective_keywords"] = effective_keywords
            ranked.append(item)

        ranked.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 9999.0))))
        return ranked[:top_k]

    def _build_text_blob(self, chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata") or {}
        return " ".join(
            [
                str(chunk.get("law_name", "")),
                str(chunk.get("article_num", "")),
                str(chunk.get("content", "")),
                str(metadata.get("retrieval_text", "")),
                str(metadata.get("context_text", "")),
                str(metadata.get("structure_path_text", "")),
                str(metadata.get("structure_locator", "")),
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

    def _build_query_profile(
        self,
        query: str,
        keywords: List[str],
        focus_terms: List[str],
    ) -> Dict[str, Any]:
        query_text = " ".join([str(query or ""), " ".join(keywords), " ".join(focus_terms)]).lower()

        domain_markers = {
            "finance_amc": ["金融资产管理公司", "不良贷款", "国有银行", "资产管理公司", "amc"],
            "consumer_prepaid": ["预付式消费", "预付卡", "消费者", "经营者", "充值卡"],
            "sale_contract": ["买卖合同", "买卖", "货物", "交付", "价款"],
            "execution": ["执行", "被执行人", "司法赔偿", "国家赔偿", "财产调查"],
            "labor": ["劳动", "用人单位", "工资", "解除劳动合同"],
            "company": ["公司", "股东", "法定代表人", "出资"],
            "private_lending": ["民间借贷", "借款", "出借人", "本金", "利息"],
            "limitation": ["诉讼时效", "时效中断", "中断"],
            "contract_general": ["合同编", "合同", "债权转让", "让与人", "受让人", "债务人", "通知义务"],
        }

        active_domains = {
            domain
            for domain, markers in domain_markers.items()
            if any(marker.lower() in query_text for marker in markers)
        }

        return {
            "text": query_text,
            "active_domains": active_domains,
        }

    def _compute_legal_priority(
        self,
        metadata: Dict[str, Any],
        query_profile: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        law_name = str(metadata.get("law_name", "")).lower()
        category = str(metadata.get("category", "")).lower()
        level = str(metadata.get("level", "")).lower()
        scope = str(metadata.get("applicability_scope", "")).lower()
        tags = [str(tag).lower() for tag in (metadata.get("tags", []) or [])]
        blob = " ".join([law_name, category, level, scope, " ".join(tags)])
        active_domains = set(query_profile.get("active_domains", set()) or set())

        score = 0.0
        reasons: List[str] = []

        # 通用规范优先：法律本体和合同编通则优先于专项解释。
        if "民法典" in law_name:
            score += 0.08
            reasons.append("民法典基础规范优先")
        if "合同编通则" in law_name or ("合同编" in law_name and "通则" in law_name):
            score += 0.08
            reasons.append("合同编通则优先")
        elif "合同编" in law_name:
            score += 0.05
            reasons.append("合同编规则前置")
        if "司法解释" in level or "解释" in law_name:
            score += 0.03
            reasons.append("司法解释具有直接适用价值")

        if "private_lending" in active_domains and "民间借贷" in blob:
            score += 0.06
            reasons.append("命中民间借贷场景")
        if "limitation" in active_domains and "诉讼时效" in blob:
            score += 0.06
            reasons.append("命中诉讼时效场景")
        if "contract_general" in active_domains and any(term in blob for term in ["债权转让", "让与人", "受让人", "债务人"]):
            score += 0.05
            reasons.append("命中债权转让一般规则")

        special_domains = {
            "finance_amc": (["金融资产管理公司", "不良贷款", "国有银行", "资产管理公司"], 0.11, "金融不良资产专项规则后置"),
            "consumer_prepaid": (["预付式消费", "预付卡", "经营者", "消费者"], 0.10, "预付式消费专项规则后置"),
            "sale_contract": (["买卖合同", "买卖"], 0.06, "买卖合同专项规则后置"),
            "execution": (["执行", "司法赔偿", "国家赔偿", "财产调查", "被执行人"], 0.12, "执行/赔偿专项规则后置"),
            "labor": (["劳动", "用人单位"], 0.08, "劳动专项规则后置"),
            "company": (["公司", "股东"], 0.07, "公司专项规则后置"),
        }

        for domain, (markers, penalty, reason) in special_domains.items():
            if any(marker.lower() in blob for marker in markers) and domain not in active_domains:
                score -= penalty
                reasons.append(reason)

        # 限幅，避免优先级规则完全覆盖语义相关性。
        score = max(-0.18, min(0.18, score))
        return round(score, 4), reasons


class LegalRAG:
    """法律检索增强生成系统，整合向量检索、查询处理和重排序功能"""
    
    def __init__(self):
        """初始化法律 RAG 系统"""
        self.config = get_config()
        self.embedding_model = EmbeddingModel()
        self.query_agent = QueryAgent()
        self.vector_db = VectorDBManager()
        self.reranker = Reranker()
        self.llm_backend = LLMBackend()

    def build_index(
        self,
        chunks: List[LawChunk],
        batch_size: int = 32,
        replace_existing_sources: bool = True,
    ):
        """构建向量索引
        
        Args:
            chunks: 法律文本块列表
            batch_size: 批处理大小
            replace_existing_sources: 是否替换已存在的源
        """
        if replace_existing_sources:
            chunk_ids = sorted(
                {
                    str(chunk.chunk_id or "").strip()
                    for chunk in chunks
                    if str(chunk.chunk_id or "").strip()
                }
            )
            if chunk_ids:
                deleted = self.vector_db.delete_chunks_by_ids(chunk_ids)
                logger.info("索引写入前已清理 %s 个同版本片段，共删除 %s 个旧片段", len(chunk_ids), deleted)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [
                str(chunk.metadata.get("retrieval_text") or chunk.content or "")
                for chunk in batch_chunks
            ]
            embeddings = self.embedding_model.encode(texts)
            self.vector_db.insert_chunks(batch_chunks, embeddings)
            logger.info(f"索引写入进度: {min(i + batch_size, len(chunks))}/{len(chunks)}")
        logger.info("索引构建完成")

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        legal_elements: Optional[Dict[str, Any]] = None,
        primary_claim: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """检索相关法律法规
        
        Args:
            query: 查询文本
            filters: 过滤条件
            legal_elements: 法律要素
            primary_claim: 主要请求权
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        result = self.search_with_trace(query, filters, legal_elements=legal_elements, primary_claim=primary_claim)
        return result.get("results", [])

    def search_with_trace(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        legal_elements: Optional[Dict[str, Any]] = None,
        primary_claim: Optional[Dict[str, Any]] = None,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """检索相关法律法规并返回完整轨迹
        
        Args:
            query: 查询文本
            filters: 过滤条件
            legal_elements: 法律要素
            primary_claim: 主要请求权
            
        Returns:
            Dict[str, Any]: 包含结果、轨迹、查询、关键词等的字典
        """
        original_query = (query or "").strip()
        if not original_query:
            return {
                "results": [],
                "trace": [],
                "query_used": "",
                "keywords": [],
                "contrastive_example": {},
                "contrastive_cot": {},
            }

        merged_filters = filters or {}
        retrieval_cfg = getattr(self.config.rag, "retrieval", None)
        contrast_cfg = getattr(self.config.rag, "contrastive", None)
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
        use_contrastive = bool(getattr(contrast_cfg, "enabled", False))
        planning_context = str(conversation_context or "").strip()
        normalized_elements = (
            self.query_agent._normalize_legal_elements(legal_elements or {})
            if legal_elements
            else None
        )
        if normalized_elements is None and planning_context:
            try:
                extracted_elements = self.query_agent.extract_legal_elements(
                    original_query,
                    context=planning_context,
                )
                normalized_elements = (
                    extracted_elements if isinstance(extracted_elements, LegalElements) else None
                )
            except Exception as exc:
                logger.warning("会话上下文法律要素抽取失败，回退空要素: %s", exc)
                normalized_elements = None
        normalized_claim = self.query_agent._normalize_claim(primary_claim or {})
        contrastive_example = (
            self.query_agent.build_contrastive_example(
                original_query,
                legal_elements=normalized_elements,
                claim=normalized_claim,
            )
            if use_contrastive
            else ContrastiveExample(target_query=original_query)
        )

        rewrite_decision = self.query_agent.should_rewrite(original_query, context=planning_context)
        if rewrite_decision.get("should_rewrite", False):
            rewritten = self.query_agent.rewrite_query(original_query, context=planning_context, force=True)
            if rewritten and rewritten != original_query:
                previous_query = original_query
                current_query = rewritten

        for round_idx in range(1, max_rounds + 1):
            thought = (
                f"第{round_idx}轮：先向量召回 {top_k_initial} 条，再做关键词命中与对比边界微调重排，"
                "若命中不足则继续改写 query。"
            )
            keywords = self.query_agent.extract_keywords(current_query, max_keywords=8)
            final_keywords = keywords
            retrieval_query = current_query
            if use_contrastive and contrastive_example.enhanced_query:
                retrieval_query = contrastive_example.enhanced_query

            dense = self._dense_search(retrieval_query, merged_filters, top_k_initial)
            # 每轮先保留更大的候选池，跨轮聚合后再截断为 top_k_final，
            # 避免“局部前十”过早截断造成有效条文丢失。
            reranked = self.reranker.rerank(
                retrieval_query,
                dense,
                top_k_initial,
                keywords=keywords,
                contrastive_example=contrastive_example if use_contrastive else None,
            )
            total_vector_hits += len(dense)

            trace.append(
                RetrievalTrace(
                    round_index=round_idx,
                    thought=thought,
                    query_used=current_query,
                    retrieval_query=retrieval_query,
                    rewritten_from=previous_query,
                    keywords=keywords,
                    vector_hits=len(dense),
                    kept_hits=len(reranked),
                    contrast_query=contrastive_example.contrast_query,
                    delta=contrastive_example.delta,
                    focus_terms=list(contrastive_example.focus_terms),
                    trap_terms=list(contrastive_example.trap_terms),
                )
            )

            for item in reranked:
                key = self._dedup_candidate_key(item)
                old = aggregated_candidates.get(key)
                current_rank_score = float(item.get("rerank_score", item.get("score", 9999.0)))
                old_rank_score = (
                    float(old.get("rerank_score", old.get("score", 9999.0)))
                    if old is not None
                    else 9999.0
                )
                if old is None or current_rank_score < old_rank_score:
                    aggregated_candidates[key] = item

            final_reranked_results = sorted(
                aggregated_candidates.values(),
                key=lambda x: float(x.get("rerank_score", x.get("score", 9999.0))),
            )[:top_k_final]

            has_enough_results = len(aggregated_candidates) >= min_results_to_stop
            has_completed_min_rounds = round_idx >= min_rounds
            if has_completed_min_rounds and has_enough_results:
                break

            if round_idx >= max_rounds:
                break

            next_query = self.query_agent.rewrite_query(current_query, context=planning_context, force=True)
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
            contrastive_example=contrastive_example if use_contrastive else None,
        )
        contrastive_cot = self._build_contrastive_cot_from_results(
            query=original_query,
            results=retrieval_results,
            contrastive_example=contrastive_example if use_contrastive else None,
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
            "contrastive_example": contrastive_example.__dict__ if use_contrastive else {},
            "contrastive_cot": contrastive_cot,
        }

    def answer_with_citations(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "",
        memory_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        search_result = self.search_with_trace(
            query,
            filters,
            conversation_context=conversation_context,
        )
        results = search_result.get("results", [])

        if not results:
            return {
                "answer": "未检索到相关法条，建议补充案件主体、行为、时间等关键信息后重试。",
                "citations": [],
                "trace": search_result.get("trace", []),
                "query_used": search_result.get("query_used", query),
                "keywords": search_result.get("keywords", []),
                "contrastive_example": search_result.get("contrastive_example", {}),
                "contrastive_cot": search_result.get("contrastive_cot", {}),
                "conversation_context": conversation_context,
                "memory_snapshot": memory_snapshot or {},
            }

        citations = []
        evidence_items = []
        for idx, item in enumerate(results, 1):
            chunk = item.chunk
            locator = str(chunk.metadata.get("structure_locator", "")).strip()
            hierarchy_path = str(chunk.metadata.get("structure_hierarchy_path", "")).strip()
            snippet = chunk.content[:220]
            citations.append(
                {
                    "id": idx,
                    "law_name": chunk.law_name,
                    "article_num": chunk.article_num,
                    "distance": item.score,
                    "rerank_score": chunk.metadata.get("rerank_score", item.score),
                    "locator": locator,
                    "effective_date": chunk.metadata.get("effective_date", ""),
                    "repeal_date": chunk.metadata.get("repeal_date", ""),
                    "snippet": snippet,
                }
            )
            evidence_items.append(
                {
                    "id": idx,
                    "law_name": chunk.law_name,
                    "article_num": chunk.article_num,
                    "locator": locator,
                    "hierarchy_path": hierarchy_path,
                    "snippet": snippet,
                    "content": chunk.content[:1200],
                    "keyword_hits": list(chunk.metadata.get("keyword_hits", [])),
                    "path_hits": list(chunk.metadata.get("path_hits", [])),
                    "path_focus_hits": list(chunk.metadata.get("path_focus_hits", [])),
                    "legal_priority_reasons": list(chunk.metadata.get("legal_priority_reasons", [])),
                }
            )
        issue_outline = self._build_issue_outline(query, evidence_items)
        contrastive_cot = self._build_contrastive_cot_from_results(
            query=query,
            results=results,
            contrastive_example=search_result.get("contrastive_example", {}),
            issue_outline=issue_outline,
        )
        answer = self._synthesize_answer(
            query,
            evidence_items,
            issue_outline,
            contrastive_cot=contrastive_cot,
        )
        return {
            "answer": answer,
            "citations": citations,
            "issue_outline": issue_outline,
            "trace": search_result.get("trace", []),
            "query_used": search_result.get("query_used", query),
            "keywords": search_result.get("keywords", []),
            "contrastive_example": search_result.get("contrastive_example", {}),
            "contrastive_cot": contrastive_cot,
            "conversation_context": conversation_context,
            "memory_snapshot": memory_snapshot or {},
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
        contrastive_example: Optional[ContrastiveExample] = None,
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
                    "rerank_score": item.get("rerank_score", item.get("score", 0.0)),
                    "keyword_boost": item.get("keyword_boost", 0.0),
                    "path_match_score": item.get("path_match_score", 0.0),
                    "legal_priority_score": item.get("legal_priority_score", 0.0),
                    "legal_priority_reasons": item.get("legal_priority_reasons", []),
                    "focus_hits": item.get("focus_hits", []),
                    "path_hits": item.get("path_hits", []),
                    "path_focus_hits": item.get("path_focus_hits", []),
                    "trap_hits": item.get("trap_hits", []),
                    "contrastive_bonus": item.get("contrastive_bonus", 0.0),
                    "contrastive_penalty": item.get("contrastive_penalty", 0.0),
                }
            )
            if contrastive_example is not None:
                metadata.update(
                    {
                        "contrast_query": contrastive_example.contrast_query,
                        "delta": contrastive_example.delta,
                        "focus_terms": list(contrastive_example.focus_terms),
                        "trap_terms": list(contrastive_example.trap_terms),
                        "contrast_type": contrastive_example.contrast_type,
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
                    score=float(item.get("raw_distance", item.get("score", 0.0))),
                    rank=idx,
                )
            )
        return results

    def _build_issue_outline(self, query: str, evidence_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """根据 query 和证据命中情况构建争点提纲。"""
        normalized_query = (query or "").strip()
        evidence_text = "\n".join(
            [
                " ".join(
                    [
                        str(item.get("law_name", "")),
                        str(item.get("article_num", "")),
                        str(item.get("locator", "")),
                        str(item.get("hierarchy_path", "")),
                        str(item.get("snippet", "")),
                        " ".join(item.get("keyword_hits", [])),
                        " ".join(item.get("path_hits", [])),
                        " ".join(item.get("path_focus_hits", [])),
                    ]
                )
                for item in evidence_items[:8]
            ]
        )
        merged_text = f"{normalized_query}\n{evidence_text}"
        answer_mode = self._detect_answer_mode(normalized_query, merged_text)

        if answer_mode == "criminal_procedure_basis":
            outline: List[Dict[str, str]] = [
                {
                    "title": "问题一：适用情形与法律依据",
                    "focus": "先说明该问题属于何种刑事程序措施或侦查措施，并列出最直接的法律、司法解释或办案规定依据。",
                },
                {
                    "title": "问题二：启动条件与决定机关",
                    "focus": "明确由谁决定、在什么条件下可以启动，以及是否需要满足特定程序前提。",
                },
                {
                    "title": "问题三：办理程序与发布要求",
                    "focus": "按程序步骤说明如何发布、公告、执行或送达，并提示文书、审批、范围等要求。",
                },
                {
                    "title": "结论：直接可引用的法条",
                    "focus": "简要列出回答该问题最核心的法条和条号，方便用户直接引用。",
                },
            ]
            return outline

        if answer_mode == "legal_basis_lookup":
            outline = [
                {
                    "title": "问题一：直接法律依据",
                    "focus": "优先回答用户问题对应的核心法条和条号，不展开无关争议。",
                },
                {
                    "title": "问题二：适用条件与限制",
                    "focus": "简要说明这些法条在什么条件下适用，以及常见的适用边界。",
                },
                {
                    "title": "结论：检索结论与出处",
                    "focus": "概括最值得引用的规范出处，方便直接用于检索、学习或写作。",
                },
            ]
            return outline

        outline: List[Dict[str, str]] = []
        outline.append(
            {
                "title": "争点一：权利基础与请求主体",
                "focus": "先判断请求人是否基于现有证据享有主张本金、利息或其他给付请求的资格，明确基础法律关系和请求权来源。",
            }
        )

        if self._contains_any(
            merged_text,
            ["债权转让", "受让人", "让与人", "转让通知", "通知到达", "债权让与"],
        ):
            outline.append(
                {
                    "title": "争点二：债权转让及通知效力",
                    "focus": "分析债权转让是否已经对债务人发生效力，通知到达前后对履行对象、抗辩和受让人请求权的影响。",
                }
            )

        if self._contains_any(
            merged_text,
            ["时效", "起诉", "诉讼时效", "中断", "届满", "到期后", "何时起诉"],
        ):
            outline.append(
                {
                    "title": "争点三：诉讼时效与时间线",
                    "focus": "结合事实中的借款时间、到期时间、通知时间、起诉时间，分析诉讼时效的起算、中断及是否仍在保护期内。",
                }
            )

        if self._contains_any(
            merged_text,
            ["利息", "利率", "逾期", "借款", "年利率", "违约责任", "LPR", "本金"],
        ):
            outline.append(
                {
                    "title": "争点四：本金、借期内利息与逾期责任",
                    "focus": "区分本金、借期内利息、逾期利息或违约责任的请求边界，明确哪些部分可直接按约主张，哪些需要结合司法解释进一步调整。",
                }
            )

        if self._contains_any(
            merged_text,
            ["抗辩", "抵销", "撤销", "无效", "不存在", "履行", "免责"],
        ):
            outline.append(
                {
                    "title": "争点五：可能的抗辩与裁判风险",
                    "focus": "提示债务人可能提出的效力、通知、履行、抗辩或请求范围方面的争议点，并说明其对主张成功率的影响。",
                }
            )

        outline.append(
            {
                "title": "结论：请求支持范围",
                "focus": "以简短结论概括哪些请求大概率可获支持，哪些请求需要限缩、拆分或补充事实后再判断。",
            }
        )
        return outline

    def _detect_answer_mode(self, query: str, merged_text: str) -> str:
        """识别问答任务类型，避免把所有问题都套成民商事争点模板。"""
        query_text = str(query or "")
        full_text = str(merged_text or "")

        criminal_markers = [
            "犯罪嫌疑人", "被告人", "公安机关", "检察院", "人民检察院", "刑事", "侦查",
            "抓捕", "逮捕", "拘留", "通缉", "通缉令", "悬赏通告", "立案侦查", "追逃",
            "刑事诉讼法", "刑诉法", "程序规定",
        ]
        basis_lookup_markers = [
            "依据哪些法律条款", "依据哪些法律", "法律依据", "法条依据", "适用哪些条文",
            "有哪些规定", "如何规定", "条文依据", "规定在哪里", "根据哪些条款",
        ]
        civil_request_markers = [
            "本金", "利息", "违约责任", "赔偿", "偿还", "履行", "借款", "合同", "起诉",
            "诉讼时效", "债权转让", "请求支持",
        ]

        if self._contains_any(query_text, criminal_markers) or self._contains_any(full_text, ["刑事诉讼法", "公安机关办理刑事案件程序规定"]):
            if self._contains_any(query_text, basis_lookup_markers) or self._contains_any(query_text, ["如何发布", "如何办理", "怎么办理", "程序", "发布要求"]):
                return "criminal_procedure_basis"
            return "criminal_procedure_basis"

        if self._contains_any(query_text, basis_lookup_markers) and not self._contains_any(query_text, civil_request_markers):
            return "legal_basis_lookup"

        return "civil_issue_analysis"

    def _normalize_contrastive_example_payload(
        self,
        contrastive_example: Optional[Any],
    ) -> Dict[str, Any]:
        if contrastive_example is None:
            return {}
        if isinstance(contrastive_example, ContrastiveExample):
            return {
                "target_query": contrastive_example.target_query,
                "contrast_query": contrastive_example.contrast_query,
                "delta": contrastive_example.delta,
                "focus_terms": list(contrastive_example.focus_terms),
                "trap_terms": list(contrastive_example.trap_terms),
                "contrast_type": contrastive_example.contrast_type,
                "enhanced_query": contrastive_example.enhanced_query,
            }
        if isinstance(contrastive_example, dict):
            return {
                "target_query": str(contrastive_example.get("target_query", "")).strip(),
                "contrast_query": str(contrastive_example.get("contrast_query", "")).strip(),
                "delta": str(contrastive_example.get("delta", "")).strip(),
                "focus_terms": list(contrastive_example.get("focus_terms", []) or []),
                "trap_terms": list(contrastive_example.get("trap_terms", []) or []),
                "contrast_type": str(contrastive_example.get("contrast_type", "")).strip(),
                "enhanced_query": str(contrastive_example.get("enhanced_query", "")).strip(),
            }
        return {}

    def _build_contrastive_cot_from_results(
        self,
        query: str,
        results: List[RetrievalResult],
        contrastive_example: Optional[Any] = None,
        issue_outline: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        payload = self._normalize_contrastive_example_payload(contrastive_example)
        focus_terms = [str(x).strip() for x in payload.get("focus_terms", []) if str(x).strip()]
        trap_terms = [str(x).strip() for x in payload.get("trap_terms", []) if str(x).strip()]
        issue_titles = [
            str(item.get("title", "")).strip()
            for item in (issue_outline or [])
            if str(item.get("title", "")).strip()
        ]

        evidence_cards: List[Dict[str, Any]] = []
        for idx, result in enumerate(results[:4], 1):
            chunk = result.chunk
            metadata = chunk.metadata or {}
            evidence_cards.append(
                {
                    "id": idx,
                    "law_name": chunk.law_name,
                    "article_num": chunk.article_num,
                    "locator": str(metadata.get("structure_locator", "")).strip(),
                    "focus_hits": list(metadata.get("focus_hits", []) or []),
                    "trap_hits": list(metadata.get("trap_hits", []) or []),
                    "keyword_hits": list(metadata.get("keyword_hits", []) or []),
                    "snippet": str(chunk.content or "")[:180],
                }
            )

        support_refs = [
            f"[{item['id']}] {item['law_name']} {item['article_num']}"
            for item in evidence_cards
        ]
        support_text = "；".join(support_refs) if support_refs else "当前无稳定支持法条"
        issue_text = "；".join(issue_titles[:3]) if issue_titles else "围绕核心法条适用边界展开"

        reasoning_steps = [
            {
                "step": "争点定位",
                "content": (
                    f"原问题是“{query}”，当前优先围绕 {issue_text} 判断。"
                    if issue_titles
                    else f"原问题是“{query}”，需先锁定最直接的法条适用边界。"
                ),
            },
            {
                "step": "支持依据",
                "content": f"应优先使用以下命中法条作为正向依据：{support_text}。",
            },
            {
                "step": "混淆排除",
                "content": (
                    f"重点关注 {('、'.join(focus_terms) if focus_terms else '案件决定性事实')}；"
                    f"避免被 {('、'.join(trap_terms) if trap_terms else '表面相似但法律关系不同的概念')} 误导。"
                ),
            },
            {
                "step": "结论边界",
                "content": (
                    f"若关键差异“{payload.get('delta') or '暂无明确 delta'}”无法被现有证据证明，"
                    "结论应收敛为有限支持或提示证据不足。"
                ),
            },
        ]

        return {
            "target_query": query,
            "contrast_query": payload.get("contrast_query", ""),
            "delta": payload.get("delta", ""),
            "focus_terms": focus_terms,
            "trap_terms": trap_terms,
            "contrast_type": payload.get("contrast_type", ""),
            "issue_titles": issue_titles[:4],
            "supporting_evidence": evidence_cards,
            "reasoning_steps": reasoning_steps,
        }

    def _render_contrastive_cot_block(
        self,
        contrastive_cot: Optional[Dict[str, Any]],
    ) -> str:
        cot = contrastive_cot or {}
        steps = cot.get("reasoning_steps", []) or []
        evidence = cot.get("supporting_evidence", []) or []
        focus_terms = cot.get("focus_terms", []) or []
        trap_terms = cot.get("trap_terms", []) or []

        evidence_lines = []
        for item in evidence[:4]:
            evidence_lines.append(
                f"- [{item.get('id', '?')}] {item.get('law_name', '未知法律')} {item.get('article_num', '')}"
                f" | focus_hits: {'、'.join(item.get('focus_hits', []) or []) or '无'}"
                f" | trap_hits: {'、'.join(item.get('trap_hits', []) or []) or '无'}"
            )

        step_lines = []
        for item in steps:
            step_lines.append(f"- {item.get('step', '步骤')}：{item.get('content', '')}")

        return (
            "Contrastive CoT 约束：\n"
            f"- 对比问题：{cot.get('contrast_query') or '无'}\n"
            f"- 关键差异：{cot.get('delta') or '无'}\n"
            f"- 应重点论证：{'、'.join([str(x) for x in focus_terms]) or '无'}\n"
            f"- 应避免误用：{'、'.join([str(x) for x in trap_terms]) or '无'}\n"
            "- 支持证据：\n"
            f"{chr(10).join(evidence_lines) if evidence_lines else '- 暂无'}\n"
            "- 推理骨架：\n"
            f"{chr(10).join(step_lines) if step_lines else '- 暂无'}"
        )

    def _synthesize_answer(
        self,
        query: str,
        evidence_items: List[Dict[str, Any]],
        issue_outline: List[Dict[str, str]],
        contrastive_cot: Optional[Dict[str, Any]] = None,
    ) -> str:
        llm_cfg = self.config.llm
        if not self.llm_backend.is_available():
            return self._fallback_answer(
                query,
                evidence_items,
                issue_outline,
                contrastive_cot=contrastive_cot,
            )

        evidence_blocks = []
        for item in evidence_items:
            reasons = item.get("legal_priority_reasons", [])
            reason_text = f"优先级提示：{'；'.join(reasons)}\n" if reasons else ""
            evidence_blocks.append(
                f"[{item['id']}] {item['law_name']} {item['article_num']}\n"
                f"体系定位：{item.get('locator') or '未知'}\n"
                f"父级路径：{item.get('hierarchy_path') or '未知'}\n"
                f"{reason_text}"
                f"{item.get('content', '')}"
            )

        outline_text = "\n".join(
            [f"- {item['title']}：{item['focus']}" for item in issue_outline]
        )
        answer_mode = self._detect_answer_mode(query, f"{query}\n{outline_text}")
        contrastive_block = self._render_contrastive_cot_block(contrastive_cot)

        if answer_mode == "criminal_procedure_basis":
            task_guidance = (
                "回答必须使用“刑事程序法检索问答”的语气，不要使用民商事请求权分析模板。\n"
                "不要出现“权利基础与请求主体”“本金、利息”“请求支持范围”等民商事措辞，除非用户问题本身涉及这些内容。\n"
                "应优先回答：1. 直接法律依据；2. 启动条件和决定机关；3. 办理程序与发布要求；4. 可直接引用的条号。\n"
                "你必须先参考 Contrastive CoT 约束，区分真正适用的程序规则与表面相似但不应误用的规范。\n"
            )
            output_requirements = (
                "1. 先给出“检索结论”，用 2-4 句直接回答用户问的程序和法条依据。\n"
                "2. 然后按提纲逐项说明，每一段都尽量引用最直接的法条编号，如 [1][2]。\n"
                "3. 增加“对比辨析”小节，简要说明哪些规则适用、哪些规则容易误用。\n"
                "4. 最后给出“可直接引用的法条”，列出名称和条号。\n"
            )
        elif answer_mode == "legal_basis_lookup":
            task_guidance = (
                "回答应偏向“法条依据检索说明”，优先列出条文依据和适用条件，不要强行扩展成案件争点裁判分析。\n"
                "你必须参考 Contrastive CoT 约束，明确适用边界并排除易混淆规范。\n"
            )
            output_requirements = (
                "1. 先给出“检索结论”，简明回答核心规范依据。\n"
                "2. 然后按提纲说明直接依据和适用条件。\n"
                "3. 增加“对比辨析”小节，说明与相似规范的边界差异。\n"
                "4. 最后给出“可直接引用的法条”。\n"
            )
        else:
            task_guidance = (
                "回答必须按争点分段，优先处理请求权基础、时间线、利息边界和裁判风险。\n"
                "如果证据不足以直接得出结论，请明确写出“现有证据不足以判断”。\n"
                "涉及金钱给付时，尽量拆分为本金、借期内利息、逾期利息/违约责任分别分析。\n"
                "你必须参考 Contrastive CoT 约束，显式排除不适用或容易误用的法条路径。\n"
            )
            output_requirements = (
                "1. 先给出“争点摘要”，用 2-4 句概括核心结论。\n"
                "2. 然后逐个争点展开，每个争点单独成段，标题保持与提纲一致。\n"
                "3. 每一段都尽量引用最直接的法条编号，如 [1][3]。\n"
                "4. 增加“对比辨析”小节，说明应适用法条与不应误用法条的边界。\n"
                "5. 最后给出“结论与建议”，明确哪些请求大概率支持、哪些需要调整。\n"
            )

        prompt = (
            "你是法律分析助手。请严格基于给定证据回答用户问题，并在句末使用 [编号] 引用。\n"
            "不要编造未提供的法律依据，不要把未命中的规范当作已检索到的依据。\n"
            "你要先在内部执行 Contrastive CoT 推理，再输出经过压缩后的结论，不要暴露冗长草稿。\n"
            f"{task_guidance}\n"
            f"用户问题：{query}\n\n"
            "请严格按照以下提纲组织回答：\n"
            f"{outline_text}\n\n"
            "输出要求：\n"
            f"{output_requirements}\n"
            f"{contrastive_block}\n\n"
            "证据：\n"
            + "\n\n".join(evidence_blocks)
        )

        try:
            content = self.llm_backend.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是专业法律助手，回答必须引用证据编号。"
                            "你必须根据问题类型切换回答格式，不能把刑事程序法检索问题套用民商事请求权分析模板。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=min(int(llm_cfg.max_tokens), 1200),
                timeout=getattr(self.config.performance, "request_timeout", 120),
            )
            return content or self._fallback_answer(
                query,
                evidence_items,
                issue_outline,
                contrastive_cot=contrastive_cot,
            )
        except Exception as e:
            logger.error(f"基于证据生成回答失败: {e}")
            return self._fallback_answer(
                query,
                evidence_items,
                issue_outline,
                contrastive_cot=contrastive_cot,
            )

    def _fallback_answer(
        self,
        query: str,
        evidence_items: List[Dict[str, Any]],
        issue_outline: List[Dict[str, str]],
        contrastive_cot: Optional[Dict[str, Any]] = None,
    ) -> str:
        answer_mode = self._detect_answer_mode(query, f"{query}\n" + "\n".join([item.get("title", "") for item in issue_outline]))
        summary_label = "检索结论" if answer_mode != "civil_issue_analysis" else "争点摘要"
        lines = [f"问题：{query}", "", f"{summary_label}："]
        top_refs = [f"[{item['id']}]" for item in evidence_items[:3]]
        if top_refs:
            if answer_mode == "civil_issue_analysis":
                lines.append(
                    f"现有检索结果显示，本问题可优先围绕 {'、'.join([item['title'] for item in issue_outline[:3]])} 展开，"
                    f"核心参考依据见 {' '.join(top_refs)}。"
                )
            else:
                lines.append(
                    f"现有检索结果显示，可优先依据 {' '.join(top_refs)} 回答该问题，并结合命中的法条说明具体适用条件和办理要求。"
                )
        else:
            lines.append("现有检索结果较少，建议补充案件主体、时间线和争议焦点。")

        lines.append("")
        for section in issue_outline:
            title = section.get("title", "").strip()
            focus = section.get("focus", "").strip()
            lines.append(title)
            if title.startswith("结论"):
                if answer_mode == "civil_issue_analysis":
                    lines.append("现阶段建议优先依据前述命中法条判断请求支持范围，并对利息、时效、通知到达等关键事实做进一步核对。")
                else:
                    lines.append("现阶段建议优先引用前述命中法条，并按条文要求说明适用情形、决定机关和办理程序。")
            else:
                refs = self._select_issue_citations(title, evidence_items)
                ref_text = " ".join([f"[{item['id']}]" for item in refs]) or "暂无直接命中条文"
                lines.append(f"分析重点：{focus}")
                if refs:
                    primary = refs[0]
                    lines.append(
                        f"当前可优先参考 {primary['law_name']} {primary['article_num']} {ref_text}，"
                        "并结合条文原文进一步论证。"
                    )
                else:
                    lines.append("当前检索结果中缺少与该争点直接对应的法条，建议补充更明确的争议描述。")
            lines.append("")

        cot = contrastive_cot or {}
        lines.append("对比辨析：")
        lines.append(
            f"应重点关注：{'、'.join([str(x) for x in cot.get('focus_terms', [])]) or '现有命中法条的直接适用条件'}。"
        )
        lines.append(
            f"应避免误用：{'、'.join([str(x) for x in cot.get('trap_terms', [])]) or '表面相似但法律关系不同的规范路径'}。"
        )
        if cot.get("delta"):
            lines.append(f"关键差异：{cot.get('delta')}")
        lines.append("")

        lines.append("可参考法条：")
        for item in evidence_items[:5]:
            lines.append(
                f"[{item['id']}] {item['law_name']} {item['article_num']} "
                f"- {item.get('locator') or item.get('hierarchy_path') or '未知定位'}"
            )
        lines.append("")
        lines.append("说明：当前为结构化降级回答，请结合上述法条原文和案件事实进行人工研判。")
        return "\n".join(lines)

    def _select_issue_citations(
        self,
        issue_title: str,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selectors = {
            "权利基础": ["借款", "本金", "债权", "请求", "履行"],
            "债权转让": ["债权转让", "通知", "受让人", "让与人"],
            "诉讼时效": ["时效", "中断", "起诉", "期限"],
            "利息": ["利息", "利率", "逾期", "借款", "LPR"],
            "抗辩": ["抗辩", "无效", "撤销", "不存在", "履行"],
            "适用情形": ["通缉", "悬赏", "逃脱", "侦查", "抓捕", "刑事", "公安机关"],
            "法律依据": ["法律依据", "法条", "条款", "规定", "程序规定", "刑事诉讼法"],
            "启动条件": ["决定", "批准", "机关", "条件", "发布", "通缉令", "悬赏通告"],
            "办理程序": ["程序", "发布", "公告", "办理", "通报", "执行", "审批"],
            "直接法律依据": ["法律依据", "法条", "条款", "规定"],
            "适用条件": ["条件", "适用", "限制", "前提"],
            "结论": [],
        }
        matched_keywords: List[str] = []
        for marker, keywords in selectors.items():
            if marker in issue_title:
                matched_keywords = keywords
                break

        ranked: List[Dict[str, Any]] = []
        for item in evidence_items:
            haystack = " ".join(
                [
                    str(item.get("law_name", "")),
                    str(item.get("article_num", "")),
                    str(item.get("locator", "")),
                    str(item.get("hierarchy_path", "")),
                    str(item.get("snippet", "")),
                    " ".join(item.get("keyword_hits", [])),
                    " ".join(item.get("path_hits", [])),
                    " ".join(item.get("path_focus_hits", [])),
                ]
            )
            score = sum(1 for keyword in matched_keywords if keyword and keyword in haystack)
            if score > 0:
                ranked.append({"score": score, "item": item})

        ranked.sort(key=lambda x: (-x["score"], x["item"]["id"]))
        if ranked:
            return [entry["item"] for entry in ranked[:2]]
        return evidence_items[:1]

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        source = str(text or "")
        return any(keyword in source for keyword in keywords if keyword)

    def _context_tier_by_rank(self, rank_index: int) -> int:
        if rank_index == 0:
            return 1
        if rank_index <= 2:
            return 2
        return 3

    def search_by_event(self, event_description: str, event_time: Optional[str] = None) -> List[RetrievalResult]:
        filters = {}
        if event_time:
            filters["case_date"] = event_time
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
