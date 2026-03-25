from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging
import json

from config import get_config
from src.models.legal_event import LegalEvent, CaseFacts
from src.rag.retriever import LegalRAG, RetrievalResult


logger = logging.getLogger(__name__)


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
        self.steps: List[AgentStep] = []
        self.current_step = 0

    def _init_tools(self):
        self.tools = {
            "rag_search": RAGSearchTool(None),
            "template_retrieval": TemplateRetrievalTool(self.config.document.template_dir),
            "fact_extraction": FactExtractionTool()
        }

    def set_rag(self, rag: LegalRAG):
        self.tools["rag_search"] = RAGSearchTool(rag)

    def think(self, context: Dict[str, Any]) -> Thought:
        prompt = self._build_thought_prompt(context)
        
        thought_content = self._generate_thought(prompt)
        
        self.current_step += 1
        return Thought(content=thought_content, step=self.current_step)

    def _build_thought_prompt(self, context: Dict[str, Any]) -> str:
        available_tools = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""你是一个专业的法律文书生成助手。当前任务：{context.get('task', '')}

可用工具：
{available_tools}

当前信息：
- 案件事实：{context.get('case_facts', '无')}
- 文书类型：{context.get('document_type', '无')}
- 用户需求：{context.get('user_request', '无')}

请分析当前情况，决定下一步行动。"""
        
        return prompt

    def _generate_thought(self, prompt: str) -> str:
        return "分析案件事实，准备检索相关法律法规"

    def act(self, thought: Thought, context: Dict[str, Any]) -> Optional[Action]:
        action_name = self._decide_action(thought, context)
        
        if action_name and action_name in self.tools:
            action_input = self._prepare_action_input(action_name, context)
            return Action(tool_name=action_name, tool_input=action_input)
        
        return None

    def _decide_action(self, thought: Thought, context: Dict[str, Any]) -> Optional[str]:
        if "检索" in thought.content or "法律" in thought.content:
            return "rag_search"
        elif "模板" in thought.content or "格式" in thought.content:
            return "template_retrieval"
        elif "提取" in thought.content or "事实" in thought.content:
            return "fact_extraction"
        return None

    def _prepare_action_input(self, action_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if action_name == "rag_search":
            case_facts = context.get("case_facts", "")
            if hasattr(case_facts, "evidence_summary"):
                query_text = case_facts.evidence_summary or str(case_facts)
            else:
                query_text = str(case_facts)

            if context.get("user_request"):
                query_text = f"{query_text} {context.get('user_request')}".strip()

            return {
                "query": query_text,
                "filters": context.get("filters")
            }
        elif action_name == "template_retrieval":
            return {
                "document_type": context.get("document_type", "")
            }
        elif action_name == "fact_extraction":
            return {
                "case_facts": context.get("case_facts")
            }
        return {}

    def observe(self, action: Action) -> Observation:
        tool = self.tools.get(action.tool_name)
        if tool:
            result = tool(**action.tool_input)
            return Observation(tool_name=action.tool_name, result=result)
        return Observation(tool_name=action.tool_name, result=None)

    def generate(self, context: Dict[str, Any], observations: List[Observation]) -> str:
        prompt = self._build_generation_prompt(context, observations)
        return self._generate_document(prompt)

    def _build_generation_prompt(self, context: Dict[str, Any], observations: List[Observation]) -> str:
        rag_results = []
        template = ""
        fact_info = {}
        
        for obs in observations:
            if obs.tool_name == "rag_search":
                rag_results = obs.result
            elif obs.tool_name == "template_retrieval":
                template = obs.result
            elif obs.tool_name == "fact_extraction":
                fact_info = obs.result

            structured_context = self._format_law_results(rag_results)
        
        prompt = f"""根据以下信息生成法律文书：

            文书类型：{context.get('document_type', '')}

            案件事实：
            {fact_info.get('evidence_summary', '')}

            相关法条与案例上下文（按重要性层级组织）：
            {structured_context}

            文书模板：
            {template}

            请根据模板格式，结合案件事实和相关法条，生成完整的法律文书。
            """
        
        return prompt

    def _format_law_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "未找到相关法条"

        def _score(item: Dict[str, Any]) -> float:
            return float(item.get("score", 0.0))

        ranked = sorted(results, key=_score, reverse=True)

        # 按用户要求组织顺序：保留最重要结果在首位，剩余结果逆序拼接。
        # 示例：1-2-3-4-5 -> 1-5-4-3-2
        if len(ranked) <= 1:
            ordered = ranked
        else:
            ordered = [ranked[0]] + list(reversed(ranked[1:]))

        lines = []
        lines.append("上下文顺序：最重要优先，其余按逆序补充")
        for idx, item in enumerate(ordered, start=1):
            tier = int(item.get("context_tier", 3))
            lines.append(
                f"{idx}. [层级{tier}] {item.get('law_name', '')} {item.get('article_num', '')} | 分数: {_score(item):.4f}"
            )
            lines.append(f"   {item.get('content', '')}")

        return "\n".join(lines)

    def _generate_document(self, prompt: str) -> str:
        return "生成的法律文书内容"

    def run(self, context: Dict[str, Any]) -> str:
        max_iterations = self.config.agent.max_iterations
        observations = []
        
        for iteration in range(max_iterations):
            thought = self.think(context)
            logger.info(f"步骤 {thought.step}: {thought.content}")
            
            action = self.act(thought, context)
            
            if action is None:
                logger.info("无需执行工具，直接生成文书")
                break
            
            logger.info(f"执行工具: {action.tool_name}")
            observation = self.observe(action)
            observations.append(observation)
            
            logger.info(f"工具执行完成，结果: {len(str(observation.result))} 字符")
            
            self.steps.append(AgentStep(thought=thought, action=action, observation=observation))
        
        return self.generate(context, observations)