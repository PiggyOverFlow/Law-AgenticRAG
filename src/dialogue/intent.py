from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List

from src.llm import LLMBackend
from src.models.dialogue import IntentResult, MemoryItem, MemoryOperation, SessionState


logger = logging.getLogger(__name__)


class IntentRecognizer:
    """LLM 驱动的会话解析器。"""

    def __init__(self):
        self.llm_backend = LLMBackend()

    def detect(
        self,
        query: str,
        session: SessionState,
        recalled_topics: List[Dict[str, Any]] | None = None,
        recalled_memories: List[Dict[str, Any]] | None = None,
    ) -> IntentResult:
        text = str(query or "").strip()
        if not text:
            return IntentResult(rationale="empty_query")

        recalled_topics = recalled_topics or []
        recalled_memories = recalled_memories or []

        llm_result = self._llm_parse_turn(
            query=text,
            session=session,
            recalled_topics=recalled_topics,
            recalled_memories=recalled_memories,
        )
        if llm_result is not None:
            return llm_result

        return self._fallback_parse_turn(text, session, recalled_topics, recalled_memories)

    def _llm_parse_turn(
        self,
        query: str,
        session: SessionState,
        recalled_topics: List[Dict[str, Any]],
        recalled_memories: List[Dict[str, Any]],
    ) -> IntentResult | None:
        if not self.llm_backend.is_available():
            return None

        recent_turns = session.turns[-6:]
        turn_lines = []
        for turn in recent_turns:
            role = "user" if turn.role == "user" else "assistant"
            turn_lines.append(f"{role}: {turn.content}")

        topic_lines = []
        for topic in recalled_topics[:4]:
            topic_lines.append(
                json.dumps(
                    {
                        "topic_id": topic.get("topic_id", ""),
                        "topic_label": topic.get("topic_label", ""),
                        "status": topic.get("status", ""),
                        "summary": topic.get("summary", ""),
                    },
                    ensure_ascii=False,
                )
            )

        memory_lines = []
        for item in recalled_memories[:12]:
            memory_lines.append(
                json.dumps(
                    {
                        "memory_id": item.get("memory_id", ""),
                        "topic_id": item.get("topic_id", ""),
                        "memory_type": item.get("memory_type", ""),
                        "content": item.get("content", ""),
                        "importance": item.get("importance", 0.0),
                        "status": item.get("status", "active"),
                    },
                    ensure_ascii=False,
                )
            )

        prompt = (
            "你是法律多轮对话管理器。请分析当前用户输入在会话中的作用，并给出结构化的记忆操作建议。"
            "你不能只判断一个意图标签，还要判断：当前轮要写入哪些长期记忆、是否与已有记忆冲突、是否需要让旧记忆失效。"
            "只输出 JSON。\n"
            "JSON 格式：\n"
            "{\n"
            '  "dialogue_intent": "new_question/follow_up/clarification/fact_append/fact_correction/topic_switch/document_generation",\n'
            '  "task_intent": "civil_issue_analysis/legal_basis_lookup/criminal_procedure_basis/document_drafting",\n'
            '  "needs_history": true,\n'
            '  "needs_rewrite": true,\n'
            '  "topic_switch": false,\n'
            '  "topic_label": "...",\n'
            '  "retrieval_mode": "active_topic/topic_recall/refresh",\n'
            '  "memory_operations": [\n'
            '    {"op":"invalidate|mark_stale|replace|noop","target_id":"memory_x","payload":{},"reason":"..."}\n'
            "  ],\n"
            '  "extracted_memories": [\n'
            '    {\n'
            '      "memory_type":"fact/claim/question/preference/correction/topic_hint",\n'
            '      "content":"...",\n'
            '      "importance":0.91,\n'
            '      "structured_payload":{"fact_type":"...","subject":"...","subject_role":"...","object":"...","object_role":"...","relation":"...","time":"...","claim":"...","basis":"...","amount":"...","evidence":"...","procedure_stage":"...","value":"..."},\n'
            '      "topic_id":"",\n'
            '      "source_turn_id":"current"\n'
            "    }\n"
            "  ],\n"
            '  "confidence":0.88,\n'
            '  "rationale":"..."\n'
            "}\n"
            "约束：\n"
            "- 如果用户纠正了前文事实，必须输出 invalidate 或 replace 操作。\n"
            "- 如果当前轮是在延续旧 topic，needs_history 必须为 true。\n"
            "- 如果当前轮更像在恢复较早的旧 topic，retrieval_mode 应为 topic_recall。\n"
            "- extracted_memories 只保留影响法律适用的高价值信息，如主体、时间、法律关系、请求、纠正、明确偏好。\n"
            "- structured_payload 尽量使用严格法律字段：subject、subject_role、object、object_role、relation、time、claim、basis、amount、evidence、procedure_stage。\n"
            "- 如果无法确定某字段，可留空，但不要编造。\n"
            "- importance 为 0 到 1，越高表示越应该进入长期记忆。\n"
            "- 只能引用已提供的 recalled memory_id，不要编造 target_id。\n"
            f"当前 active topic: {json.dumps(session.active_topic.model_dump(), ensure_ascii=False) if session.active_topic else 'null'}\n"
            f"历史摘要: {session.memory.rolling_summary}\n"
            f"近期对话: {turn_lines}\n"
            f"召回 topic: {topic_lines}\n"
            f"召回 memory: {memory_lines}\n"
            f"当前用户输入: {query}"
        )

        try:
            content = self.llm_backend.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "你是法律会话状态解析器，只输出 JSON，不要输出解释文本。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            data = self._parse_json(content)
            if not data:
                return None
            return self._normalize_intent_result(data)
        except Exception as exc:
            logger.warning("LLM 会话解析失败，回退启发式解析: %s", exc)
            return None

    def _fallback_parse_turn(
        self,
        query: str,
        session: SessionState,
        recalled_topics: List[Dict[str, Any]],
        recalled_memories: List[Dict[str, Any]],
    ) -> IntentResult:
        lowered = query.strip()
        has_history = bool(session.turns)
        memory_ops: List[MemoryOperation] = []
        extracted: List[MemoryItem] = []

        dialogue_intent = "new_question"
        task_intent = "civil_issue_analysis"
        retrieval_mode = "active_topic"
        topic_switch = False

        if any(word in lowered for word in ("刑事", "侦查", "公安机关", "检察院")):
            task_intent = "criminal_procedure_basis"
        elif any(word in lowered for word in ("法律依据", "法条依据", "适用哪些条文")):
            task_intent = "legal_basis_lookup"

        if any(word in lowered for word in ("换个问题", "另一个问题", "再问一个")):
            dialogue_intent = "topic_switch"
            topic_switch = True
            retrieval_mode = "refresh"
        elif any(word in lowered for word in ("不是", "更正", "说错了")) and has_history:
            dialogue_intent = "fact_correction"
            retrieval_mode = "refresh"
            for item in recalled_memories[:2]:
                memory_ops.append(
                    MemoryOperation(
                        op="invalidate",
                        target_id=str(item.get("memory_id", "")),
                        reason="fallback_fact_correction",
                    )
                )
        elif any(word in lowered for word in ("补充", "还有", "另外")) and has_history:
            dialogue_intent = "fact_append"
            retrieval_mode = "topic_recall"
        elif has_history and len(lowered) <= 28:
            dialogue_intent = "follow_up"
            retrieval_mode = "topic_recall"

        if any(word in lowered for word in ("发生在", "已经", "存在", "请求", "要求", "利息", "本金", "担保", "借款")):
            extracted.append(
                MemoryItem(
                    memory_id=uuid.uuid4().hex,
                    memory_type="fact",
                    content=query,
                    importance=0.75,
                    structured_payload=self._infer_legal_payload(query),
                    source_turn_id="current",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )

        return IntentResult(
            dialogue_intent=dialogue_intent,
            task_intent=task_intent,
            needs_history=has_history and dialogue_intent != "new_question",
            needs_rewrite=has_history and dialogue_intent in {"follow_up", "clarification", "fact_append", "fact_correction"},
            topic_switch=topic_switch,
            topic_label=self._clip_text(query, 40),
            retrieval_mode=retrieval_mode,
            memory_operations=memory_ops,
            extracted_memories=extracted,
            confidence=0.45,
            rationale="fallback_parser",
        )

    def _normalize_intent_result(self, data: Dict[str, Any]) -> IntentResult:
        memory_operations: List[MemoryOperation] = []
        for item in data.get("memory_operations", []) or []:
            if not isinstance(item, dict):
                continue
            memory_operations.append(
                MemoryOperation(
                    op=str(item.get("op", "noop")).strip() or "noop",
                    target_id=str(item.get("target_id", "")).strip(),
                    payload=dict(item.get("payload", {}) or {}),
                    reason=str(item.get("reason", "")).strip(),
                )
            )

        extracted_memories: List[MemoryItem] = []
        for item in data.get("extracted_memories", []) or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            importance = item.get("importance", 0.5)
            try:
                importance = float(importance)
            except Exception:
                importance = 0.5
            extracted_memories.append(
                MemoryItem(
                    memory_id=uuid.uuid4().hex,
                    topic_id=str(item.get("topic_id", "")).strip(),
                    memory_type=str(item.get("memory_type", "fact")).strip() or "fact",
                    content=content,
                    structured_payload=self._normalize_legal_payload(dict(item.get("structured_payload", {}) or {})),
                    importance=max(0.0, min(1.0, importance)),
                    source_turn_id=str(item.get("source_turn_id", "current")).strip() or "current",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return IntentResult(
            dialogue_intent=str(data.get("dialogue_intent", "new_question")).strip() or "new_question",
            task_intent=str(data.get("task_intent", "civil_issue_analysis")).strip() or "civil_issue_analysis",
            needs_history=bool(data.get("needs_history", False)),
            needs_rewrite=bool(data.get("needs_rewrite", False)),
            topic_switch=bool(data.get("topic_switch", False)),
            topic_label=self._clip_text(str(data.get("topic_label", "")).strip(), 60),
            retrieval_mode=str(data.get("retrieval_mode", "active_topic")).strip() or "active_topic",
            memory_operations=memory_operations,
            extracted_memories=extracted_memories,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=str(data.get("rationale", "")).strip(),
        )

    def _parse_json(self, text: str) -> Dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _clip_text(self, text: str, max_chars: int) -> str:
        raw = str(text or "").strip()
        if len(raw) <= max_chars:
            return raw
        return raw[: max(0, max_chars - 3)] + "..."

    def _normalize_legal_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload or {})
        allowed_fields = {
            "fact_type",
            "subject",
            "subject_role",
            "object",
            "object_role",
            "relation",
            "time",
            "claim",
            "basis",
            "amount",
            "evidence",
            "procedure_stage",
            "value",
            "court",
            "location",
        }
        return {key: normalized.get(key) for key in allowed_fields if key in normalized}

    def _infer_legal_payload(self, query: str) -> Dict[str, Any]:
        text = str(query or "").strip()
        payload: Dict[str, Any] = {"value": text}
        if "借款" in text:
            payload["relation"] = "借款"
            payload["fact_type"] = "legal_relation"
        elif "担保" in text:
            payload["relation"] = "担保"
            payload["fact_type"] = "legal_relation"
        elif "利息" in text:
            payload["claim"] = "利息"
            payload["fact_type"] = "claim"
        elif "本金" in text:
            payload["claim"] = "本金"
            payload["fact_type"] = "claim"
        return payload
