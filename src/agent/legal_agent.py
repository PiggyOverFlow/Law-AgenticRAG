from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass, field
import logging
import requests

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
        """检索法律法规。"""
        facts = state.get("extracted_facts", {})
        evidence_summary = facts.get("evidence_summary", "")
        key_disputes = "；".join(facts.get("key_disputes", [])[:5]) if facts.get("key_disputes") else ""
        query = " ".join([x for x in [evidence_summary, key_disputes] if x]).strip()
        if state.get("user_request"):
            query = f"{query} {state['user_request']}".strip()

        if not query:
            query = state.get("document_type", "法律文书")

        if not self.rag_tool or not self.rag_tool.rag:
            logger.warning("RAG 工具未初始化，跳过法条检索")
            state["retrieved_laws"] = []
            state["observations"].append({"tool": "rag_search", "result": [], "warning": "rag_not_initialized"})
            return state

        results = self.rag_tool.rag.search(query)
        retrieved = [
            {
                "law_name": r.chunk.law_name,
                "article_num": r.chunk.article_num,
                "content": r.chunk.content,
                "score": r.score,
            }
            for r in results
        ]
        state["retrieved_laws"] = retrieved
        state["observations"].append({"tool": "rag_search", "result": retrieved})
        return state

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

    def _route_after_rag(self, state: AgentState) -> Literal["template_retrieval", "generate"]:
        """RAG 检索完成后，进入模板获取或直接生成。"""
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

        lines = []
        for idx, law in enumerate(results[:8], 1):
            lines.append(
                f"{idx}. {law.get('law_name', '未知法律')} {law.get('article_num', '')}"
                f" | 分数: {float(law.get('score', 0.0)):.4f}\n"
                f"   {law.get('content', '')}"
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