from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass, field
import logging
import requests
import re

from config import get_config
from src.models.legal_event import LegalEvent, CaseFacts
from src.rag.retriever import LegalRAG, RetrievalResult, _StateGraph, END
StateGraph = _StateGraph  # 本地别名


logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph State — 替代隐式的 self.steps / self.current_step
# =============================================================================
class AgentState(dict):
    """LegalAgent 的状态结构，LangGraph 在节点间传递此字典。"""
    task: str = ""
    case_facts: Any = ""
    document_type: str = ""
    user_request: str = ""
    max_iterations: int = 5

    # ReAct 循环状态
    step: int = 0
    thoughts: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    # 中间结果
    retrieved_laws: list[dict] = field(default_factory=list)
    law_filter_report: dict = field(default_factory=dict)
    template: str = ""
    extracted_facts: dict = field(default_factory=dict)

    # 最终输出
    generated_document: str | None = None


@dataclass
class Thought:
    content: str
    step: int


@dataclass
class Action:
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class Observation:
    tool_name: str
    result: Any


@dataclass
class AgentStep:
    thought: Thought
    action: Optional[Action] = None
    observation: Optional[Observation] = None


class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class RAGSearchTool(Tool):
    def __init__(self, rag: LegalRAG):
        super().__init__(
            name="rag_search",
            description="根据案件描述检索相关的法律法规条文",
            func=self._search
        )
        self.rag = rag

    def _search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        results = self.rag.search(query, filters)
        return [
            {
                "law_name": r.chunk.law_name,
                "article_num": r.chunk.article_num,
                "content": r.chunk.content,
                "score": r.score,
                "metadata": r.chunk.metadata,
                "context_tier": r.chunk.metadata.get("context_tier", 3),
                "route_focus": r.chunk.metadata.get("route_focus", "law_article")
            }
            for r in results
        ]


class TemplateRetrievalTool(Tool):
    def __init__(self, template_dir: str):
        super().__init__(
            name="template_retrieval",
            description="根据文书类型检索对应的文书模板",
            func=self._retrieve_template
        )
        self.template_dir = template_dir

    def get_template(self, document_type: str) -> str:
        """直接获取模板，供 LangGraph 节点调用。"""
        from pathlib import Path

        template_path = Path(self.template_dir) / f"{document_type}.txt"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return self._get_default_template(document_type)

    def _retrieve_template(self, document_type: str) -> str:
        from pathlib import Path
        
        template_path = Path(self.template_dir) / f"{document_type}.txt"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        else:
            return self._get_default_template(document_type)

    def _get_default_template(self, document_type: str) -> str:
        templates = {
            "起诉书": """# 起诉书

原告：{plaintiff_info}
被告：{defendant_info}

## 诉讼请求
{claims}

## 事实与理由
{facts_and_reasons}

此致
{court}

具状人：{plaintiff}
{date}
""",
            "答辩状": """# 答辩状

答辩人：{defendant_info}
被答辩人：{plaintiff_info}

## 答辩请求
{defense_requests}

## 事实与理由
{facts_and_reasons}

此致
{court}

答辩人：{defendant}
{date}
""",
            "上诉状": """# 上诉状

上诉人：{appellant_info}
被上诉人：{appellee_info}

## 上诉请求
{appeal_requests}

## 事实与理由
{facts_and_reasons}

此致
{court}

上诉人：{appellant}
{date}
""",
            "申请书": """# 申请书

申请人：{applicant_info}
被申请人：{respondent_info}

## 申请事项
{application_matters}

## 事实与理由
{facts_and_reasons}

此致
{court}

申请人：{applicant}
{date}
""",
            "代理词": """# 代理词

尊敬的审判长、审判员：

{opening_statement}

## 代理意见
{arguments}

## 结语
{conclusion}

代理人：{agent}
{date}
"""
        }
        return templates.get(document_type, "模板不存在")


class FactExtractionTool(Tool):
    def __init__(self):
        super().__init__(
            name="fact_extraction",
            description="从案件事实中提取关键信息",
            func=self._extract_facts
        )

    def _extract_facts(self, case_facts: CaseFacts) -> Dict[str, Any]:
        key_info = {
            "events_count": len(case_facts.events),
            "evidence_summary": case_facts.evidence_summary,
            "key_disputes": case_facts.key_disputes,
            "timeline": self._build_timeline(case_facts.events)
        }
        return key_info

    def _build_timeline(self, events: List[LegalEvent]) -> List[Dict[str, Any]]:
        timeline = []
        for event in events:
            timeline.append({
                "time": event.time,
                "place": event.place,
                "cause": event.cause,
                "process": event.process,
                "result": event.result,
                "source": event.source_file
            })
        return timeline


class LegalAgent:
    def __init__(self):
        self.config = get_config()
        self._init_tools()
        self._build_graph()

    def _init_tools(self):
        self.tools = {
            "rag_search": RAGSearchTool(None),
            "template_retrieval": TemplateRetrievalTool(self.config.document.template_dir),
            "fact_extraction": FactExtractionTool()
        }
        self.rag_tool: RAGSearchTool = self.tools["rag_search"]

    def set_rag(self, rag: LegalRAG):
        self.tools["rag_search"] = RAGSearchTool(rag)
        self.rag_tool = self.tools["rag_search"]

    # =============================================================================
    # LangGraph 节点 — 替代 think/act/observe 的手动循环
    # =============================================================================
    def _node_fact_extraction(self, state: AgentState) -> AgentState:
        """提取案件关键事实。"""
        tool = self.tools["fact_extraction"]
        case_facts = state.get("case_facts")
        if hasattr(case_facts, "evidence_summary"):
            extracted = tool(case_facts=case_facts)
        else:
            extracted = {}
        state["extracted_facts"] = extracted
        state["observations"].append({"tool": "fact_extraction", "result": extracted})
        return state

    def _node_rag_search(self, state: AgentState) -> AgentState:
        """检索法律法规（Agentic-RAG：多轮查询 + 回退策略）。"""
        if not self.rag_tool or not self.rag_tool.rag:
            logger.warning("RAG 工具未初始化，跳过法条检索")
            state["retrieved_laws"] = []
            state["observations"].append({"tool": "rag_search", "result": [], "warning": "rag_not_initialized"})
            return state

        queries = self._build_agentic_rag_queries(state)
        retrieval_cfg = getattr(self.config.rag, "retrieval", None)
        target_results = max(1, int(getattr(retrieval_cfg, "top_k_final", 5)))
        min_attempts = max(1, int(getattr(retrieval_cfg, "agentic_min_rounds", 1)))

        dedup: Dict[str, Dict[str, Any]] = {}
        attempts: List[Dict[str, Any]] = []

        for idx, query in enumerate(queries[:8], 1):
            if not query.strip():
                continue

            try:
                results = self.rag_tool(query=query)
            except Exception as e:
                logger.warning(f"RAG 第{idx}轮检索失败: {e}")
                attempts.append({"attempt": idx, "query": query, "hits": 0, "error": str(e)})
                continue

            hits = 0
            for r in results:
                key = f"{r.get('law_name', '')}::{r.get('article_num', '')}"
                old = dedup.get(key)
                if old is None or float(r.get("score", 9999.0)) < float(old.get("score", 9999.0)):
                    dedup[key] = r
                hits += 1

            attempts.append({"attempt": idx, "query": query, "hits": hits})
            if len(dedup) >= target_results and idx >= min_attempts:
                break

        retrieved = sorted(dedup.values(), key=lambda x: float(x.get("score", 9999.0)))[:target_results]

        logger.info(f"Agentic-RAG 检索轮次: {len(attempts)}，命中条文: {len(retrieved)}")
        state["retrieved_laws"] = retrieved
        state["observations"].append({
            "tool": "rag_search",
            "result": retrieved,
            "agentic_attempts": attempts,
        })
        return state

    def _node_evidence_filter(self, state: AgentState) -> AgentState:
        """对候选法条做证据级筛选，抑制错引与长上下文膨胀。"""
        laws = state.get("retrieved_laws", []) or []
        if not laws:
            state["law_filter_report"] = {"candidates": 0, "kept": 0}
            state["observations"].append(
                {"tool": "evidence_filter", "result": [], "warning": "no_candidates"}
            )
            return state

        retrieval_cfg = getattr(self.config.rag, "retrieval", None)
        max_laws = max(1, int(getattr(retrieval_cfg, "top_k_final", 5)))
        min_support = float(getattr(retrieval_cfg, "evidence_filter_min_score", 0.18))
        total_char_budget = max(600, int(getattr(retrieval_cfg, "context_char_budget", 2600)))
        per_law_char_budget = max(120, int(getattr(retrieval_cfg, "per_law_char_budget", 700)))

        signals = self._build_evidence_signals(state)
        ranked = [self._score_law_by_evidence(item, signals) for item in laws]
        ranked.sort(
            key=lambda x: (-float(x.get("evidence_support_score", 0.0)), float(x.get("score", 9999.0)))
        )

        kept = [item for item in ranked if float(item.get("evidence_support_score", 0.0)) >= min_support]
        if not kept:
            kept = ranked[: max(1, min(3, len(ranked)))]

        compressed: List[Dict[str, Any]] = []
        used_chars = 0
        for law in kept:
            if len(compressed) >= max_laws:
                break

            remaining = total_char_budget - used_chars
            if remaining <= 80:
                break

            clipped = self._clip_text(
                str(law.get("content", "")),
                min(per_law_char_budget, remaining),
            )

            item = dict(law)
            item["content"] = clipped

            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "evidence_support_score": item.get("evidence_support_score", 0.0),
                    "evidence_hits": item.get("evidence_hits", []),
                    "evidence_filter_applied": True,
                }
            )
            item["metadata"] = metadata

            used_chars += len(clipped)
            compressed.append(item)

        avg_support = (
            sum(float(x.get("evidence_support_score", 0.0)) for x in compressed) / len(compressed)
            if compressed
            else 0.0
        )
        report = {
            "candidates": len(laws),
            "kept": len(compressed),
            "dropped": max(0, len(laws) - len(compressed)),
            "signals": len(signals),
            "avg_support": round(avg_support, 4),
            "char_budget": total_char_budget,
            "char_used": used_chars,
        }

        state["retrieved_laws"] = compressed
        state["law_filter_report"] = report
        state["observations"].append(
            {
                "tool": "evidence_filter",
                "result": compressed,
                "report": report,
            }
        )

        logger.info(
            "证据级筛选完成，候选: %s，保留: %s，平均支持分: %.3f，字符预算: %s/%s",
            report["candidates"],
            report["kept"],
            report["avg_support"],
            report["char_used"],
            report["char_budget"],
        )
        return state

    def _build_agentic_rag_queries(self, state: AgentState) -> List[str]:
        facts = state.get("extracted_facts", {})
        evidence_summary = self._sanitize_evidence_summary(facts.get("evidence_summary", ""))
        disputes = facts.get("key_disputes", []) or []
        user_request = (state.get("user_request") or "").strip()
        doc_type = (state.get("document_type") or "法律文书").strip()

        base_candidates = [
            " ".join([user_request, "；".join(disputes[:4])]).strip(),
            " ".join([user_request, evidence_summary[:300]]).strip(),
            " ".join(["；".join(disputes[:6]), evidence_summary[:280]]).strip(),
            user_request,
            evidence_summary[:320],
        ]

        expanded: List[str] = []
        for q in base_candidates:
            if not q:
                continue
            expanded.append(q)
            expanded.append(f"{q} 相关法律依据")
            expanded.append(f"{q} 司法解释")

        # 当语义过于个案化导致召回为 0 时，追加案由级兜底检索。
        domain_fallback = self._build_domain_fallback_queries(user_request, disputes, doc_type)
        expanded.extend(domain_fallback)

        unique = []
        seen = set()
        for q in expanded:
            normalized = re.sub(r"\s+", " ", q).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)

        return unique

    def _sanitize_evidence_summary(self, summary: str) -> str:
        if not summary:
            return ""
        lines = []
        for line in str(summary).splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("文件:") or s.startswith("类型:"):
                continue
            lines.append(s)
        return " ".join(lines)

    def _build_domain_fallback_queries(self, user_request: str, disputes: List[str], doc_type: str) -> List[str]:
        seed = " ".join([user_request, "；".join(disputes[:4])])

        if any(k in seed for k in ["借款", "民间借贷", "本金", "利息", "乙公司", "追加"]):
            return [
                "民间借贷 起诉书 共同被告 追加被告 法律依据",
                "民法典 借款合同 还本付息 逾期利息",
                "民间借贷司法解释 共同债务 追加被告",
            ]

        if any(k in seed for k in ["劳动", "赔偿金", "解除劳动合同"]):
            return [
                "劳动争议 违法解除 劳动合同 赔偿金 法律依据",
                "劳动合同法 解除劳动合同 经济补偿",
            ]

        return [
            f"{doc_type} 常用法律依据",
            "民事诉讼 起诉 条件 法律依据",
        ]

    def _build_evidence_signals(self, state: AgentState) -> List[str]:
        facts = state.get("extracted_facts", {}) or {}
        summary = self._sanitize_evidence_summary(facts.get("evidence_summary", ""))
        disputes = facts.get("key_disputes", []) or []
        timeline = facts.get("timeline", []) or []

        parts = [
            str(state.get("user_request", "")),
            summary,
            "；".join([str(x) for x in disputes[:10]]),
        ]

        for event in timeline[:15]:
            if not isinstance(event, dict):
                continue
            for key in ("cause", "process", "result", "place", "time"):
                value = event.get(key)
                if value:
                    parts.append(str(value))

        corpus = " ".join(parts)
        terms = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9%]{2,16}", corpus)
        terms.extend(
            re.findall(
                r"(?:20\d{2}年\d{1,2}月\d{1,2}日|\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:万|千|百)?元)",
                corpus,
            )
        )

        priority_words = (
            "借款", "借贷", "利息", "本金", "违约", "逾期", "合同", "转账", "借条",
            "赔偿", "劳动", "解除", "交通事故", "侵权", "举证", "时效",
        )
        for word in priority_words:
            if word in corpus:
                terms.append(word)

        dedup: List[str] = []
        seen = set()
        for token in terms:
            normalized = re.sub(r"\s+", "", str(token or "")).strip().lower()
            if len(normalized) < 2 or len(normalized) > 20:
                continue
            if normalized in seen:
                continue
            if self._is_low_information_term(normalized):
                continue
            seen.add(normalized)
            dedup.append(normalized)

        return dedup[:24]

    def _score_law_by_evidence(self, law: Dict[str, Any], signals: List[str]) -> Dict[str, Any]:
        item = dict(law)
        text_blob = " ".join(
            [
                str(item.get("law_name", "")),
                str(item.get("article_num", "")),
                str(item.get("content", "")),
            ]
        ).lower()

        base_distance = float(item.get("score", 9999.0))
        base_relevance = 1.0 / (1.0 + max(0.0, base_distance))

        hits = []
        for signal in signals:
            if signal in text_blob:
                hits.append(signal)
            if len(hits) >= 10:
                break

        overlap = len(hits) / max(1, min(len(signals), 12))
        context_tier = int(item.get("context_tier", 3))
        tier_bonus = 0.08 if context_tier == 1 else 0.04 if context_tier == 2 else 0.0

        support_score = min(1.0, 0.62 * base_relevance + 0.30 * overlap + tier_bonus)

        item["evidence_support_score"] = round(support_score, 4)
        item["evidence_hits"] = hits
        return item

    def _is_low_information_term(self, token: str) -> bool:
        low_info = {
            "法律", "法规", "条文", "规定", "相关", "问题", "情况", "处理", "如何", "怎么办",
            "纠纷", "案件", "起诉", "诉讼", "请求", "支持", "认定", "成立", "是否", "文书",
            "事实", "理由", "原告", "被告", "申请", "法院", "依据",
        }
        return token in low_info or token.isdigit()

    def _clip_text(self, text: str, max_chars: int) -> str:
        value = (text or "").strip()
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max(0, max_chars - 3)] + "..."

    def _node_template_retrieval(self, state: AgentState) -> AgentState:
        """获取文书模板。"""
        tool: TemplateRetrievalTool = self.tools["template_retrieval"]
        template = tool.get_template(state.get("document_type", "起诉书"))
        state["template"] = template
        state["observations"].append({"tool": "template_retrieval", "result": template})
        return state

    def _node_generate(self, state: AgentState) -> AgentState:
        """综合观察结果生成最终文书。"""
        laws = state.get("retrieved_laws", [])
        prompt = self._build_generation_prompt(
            {
                "document_type": state.get("document_type", ""),
                "user_request": state.get("user_request", ""),
                "extracted_facts": state.get("extracted_facts", {}),
                "template": state.get("template", ""),
                "case_facts": state.get("case_facts"),
            },
            laws,
        )
        state["generated_document"] = self._generate_document(prompt)
        return state

    # =============================================================================
    # 条件边路由 — 替代 _decide_action 中的 if-elif 链
    # =============================================================================
    def _route_after_extraction(self, state: AgentState) -> Literal["rag_search", "template_retrieval"]:
        """提取完成后，决定下一步是检索法律还是获取模板。"""
        if not state.get("retrieved_laws"):
            return "rag_search"
        return "template_retrieval"

    def _route_after_rag(self, state: AgentState) -> Literal["evidence_filter", "template_retrieval", "generate"]:
        """RAG 检索完成后，优先进入证据级筛选，再决定模板或生成。"""
        if state.get("retrieved_laws"):
            return "evidence_filter"
        if not state.get("template"):
            return "template_retrieval"
        return "generate"

    def _route_after_filter(self, state: AgentState) -> Literal["template_retrieval", "generate"]:
        """证据筛选后，进入模板获取或直接生成。"""
        if not state.get("template"):
            return "template_retrieval"
        return "generate"

    def _route_continue(self, state: AgentState) -> Literal["fact_extraction", "generate"]:
        """判断是继续迭代还是进入生成阶段。"""
        if state["step"] >= state["max_iterations"]:
            return "generate"
        if state.get("retrieved_laws") and state.get("template"):
            return "generate"
        return "fact_extraction"

    # =============================================================================
    # 构建 StateGraph
    # =============================================================================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("fact_extraction", self._node_fact_extraction)
        workflow.add_node("rag_search", self._node_rag_search)
        workflow.add_node("evidence_filter", self._node_evidence_filter)
        workflow.add_node("template_retrieval", self._node_template_retrieval)
        workflow.add_node("generate", self._node_generate)

        # 条件边路由
        workflow.add_conditional_edges(
            "fact_extraction",
            self._route_after_extraction,
            {
                "rag_search": "rag_search",
                "template_retrieval": "template_retrieval",
            },
        )

        workflow.add_conditional_edges(
            "rag_search",
            self._route_after_rag,
            {
                "evidence_filter": "evidence_filter",
                "template_retrieval": "template_retrieval",
                "generate": "generate",
            },
        )

        workflow.add_conditional_edges(
            "evidence_filter",
            self._route_after_filter,
            {
                "template_retrieval": "template_retrieval",
                "generate": "generate",
            },
        )

        workflow.add_conditional_edges(
            "template_retrieval",
            self._route_continue,
            {
                "fact_extraction": "fact_extraction",
                "generate": "generate",
            },
        )

        workflow.set_entry_point("fact_extraction")
        workflow.add_edge("generate", END)

        self._graph = workflow.compile()

    # =============================================================================
    # 对外接口 — 保持原有签名，内部委托给 Graph
    # =============================================================================
    def run(self, context: Dict[str, Any]) -> str:
        """执行 Agent，外部调用入口。"""
        state = AgentState(
            task=context.get("task", ""),
            case_facts=context.get("case_facts", ""),
            document_type=context.get("document_type", "起诉书"),
            user_request=context.get("user_request", ""),
            max_iterations=context.get("max_iterations", self.config.agent.max_iterations),
            step=0,
            thoughts=[],
            actions=[],
            observations=[],
            retrieved_laws=[],
            law_filter_report={},
            template="",
            extracted_facts={},
            generated_document=None,
        )

        result = self._graph.invoke(state)
        return result.get("generated_document") or ""

    # 保留原有方法供兼容（内部不再被 run 调用）
    def think(self, context: Dict[str, Any]) -> Thought:
        self.current_step = getattr(self, "current_step", 0) + 1
        return Thought(content="分析案件事实，准备检索相关法律法规", step=self.current_step)

    def act(self, thought: Thought, context: Dict[str, Any]) -> Optional[Action]:
        return None

    def observe(self, action: Action) -> Observation:
        return Observation(tool_name="", result=None)

    def generate(self, context: Dict[str, Any], observations: List[Observation]) -> str:
        return ""

    def _build_generation_prompt(self, context: Dict[str, Any], laws: List[Dict[str, Any]]) -> str:
        doc_type = context.get("document_type", "法律文书")
        user_request = context.get("user_request", "")
        template = context.get("template", "")
        facts = context.get("extracted_facts", {})
        case_facts = context.get("case_facts")

        fact_timeline = []
        if hasattr(case_facts, "events") and case_facts.events:
            for idx, event in enumerate(case_facts.events[:20], 1):
                event_line = (
                    f"{idx}. 时间:{event.time or '未知'}；地点:{event.place or '未知'}；"
                    f"起因:{event.cause or '未知'}；经过:{event.process or '未知'}；"
                    f"结果:{event.result or '未知'}"
                )
                fact_timeline.append(event_line)

        law_context = self._format_law_results(laws)
        evidence_summary = facts.get("evidence_summary", "")
        key_disputes = "\n".join([f"- {d}" for d in facts.get("key_disputes", [])])

        return f"""你是一名资深中国执业律师，请根据给定信息生成{doc_type}。

硬性要求：
1. 只输出最终文书正文，不要解释过程。
2. 严禁虚构当事人身份信息、金额、时间、地点；缺失信息请用“待补充”。
3. 法律依据必须来自提供的“相关法条”，并在“事实与理由”中结合案情论证。
4. 格式尽量贴合文书模板，结构完整、用语规范。
5. 若证据中出现明确利率/期限/本金，必须逐字沿用，不得改写为其他数值。
6. 涉及利息请求时，必须明确写出计算基数、利率类型（月利率或年利率）、起算时间与截止时间。

文书类型：{doc_type}
用户诉求：{user_request or '无'}

证据摘要：
{evidence_summary or '无'}

争议焦点：
{key_disputes or '无'}

事实时间线：
{chr(10).join(fact_timeline) if fact_timeline else '无'}

相关法条：
{law_context}

文书模板：
{template or '无模板'}
"""

    def _format_law_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "未找到相关法律条文"

        retrieval_cfg = getattr(self.config.rag, "retrieval", None)
        max_items = max(1, int(getattr(retrieval_cfg, "top_k_final", 5)))
        total_char_budget = max(600, int(getattr(retrieval_cfg, "context_char_budget", 2600)))
        per_law_char_budget = max(120, int(getattr(retrieval_cfg, "per_law_char_budget", 700)))

        lines = []
        used_chars = 0
        for law in results[: max_items * 2]:
            if len(lines) >= max_items:
                break

            remaining = total_char_budget - used_chars
            if remaining <= 80:
                break

            snippet = self._clip_text(
                str(law.get("content", "")),
                min(per_law_char_budget, remaining),
            )
            used_chars += len(snippet)

            support_score = law.get("evidence_support_score")
            if support_score is None:
                support_score = (law.get("metadata") or {}).get("evidence_support_score")

            support_text = ""
            try:
                if support_score is not None:
                    support_text = f" | 证据支持分: {float(support_score):.3f}"
            except Exception:
                support_text = ""

            idx = len(lines) + 1
            lines.append(
                f"{idx}. {law.get('law_name', '未知法律')} {law.get('article_num', '')}"
                f" | 距离: {float(law.get('score', 0.0)):.4f} (越小越相关)\n"
                f"   {snippet}{support_text}"
            )
        return "\n".join(lines)

    def _build_fallback_document(self, prompt: str) -> str:
        return (
            "# 法律文书（生成降级结果）\n\n"
            "模型调用失败，以下为系统根据现有证据与模板生成的草稿，请人工完善后使用。\n\n"
            "## 事实与理由\n"
            "待补充：请根据证据材料补充事实经过、争议焦点和法律依据。\n\n"
            "## 诉讼请求/申请事项\n"
            "待补充：请根据案件目标填写具体请求。\n\n"
            "## 参考上下文\n"
            f"{prompt[:2000]}"
        )

    def _generate_document(self, prompt: str) -> str:
        llm_cfg = self.config.llm
        base_url = str(llm_cfg.base_url).rstrip("/")
        api_key = llm_cfg.api_key
        model = llm_cfg.primary_model

        if not base_url or not model or not api_key or str(api_key).startswith("${"):
            logger.warning("LLM 配置不完整，返回降级草稿")
            return self._build_fallback_document(prompt)

        payload = {
            "model": model,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "你是中国法律文书写作助手。必须严格基于输入事实与法条生成，输出中文正式文书正文。",
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

            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                logger.warning("LLM 返回为空，返回降级草稿")
                return self._build_fallback_document(prompt)
            return content
        except Exception as e:
            logger.error(f"文书生成调用失败: {e}")
            return self._build_fallback_document(prompt)