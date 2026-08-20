from __future__ import annotations

from typing import List

from src.models.dialogue import IntentResult, MemoryItem, SessionState, TopicState


class ContextBuilder:
    def __init__(self, recent_turn_window: int = 6, max_history_chars: int = 4000):
        self.recent_turn_window = max(1, int(recent_turn_window))
        self.max_history_chars = max(800, int(max_history_chars))

    def build(
        self,
        session: SessionState,
        intent: IntentResult,
        query: str,
        recalled_topics: List[TopicState] | None = None,
        recalled_memories: List[MemoryItem] | None = None,
    ) -> str:
        sections: List[str] = []
        memory = session.memory
        recalled_topics = recalled_topics or []
        recalled_memories = recalled_memories or []

        if session.active_topic and session.active_topic.topic_label:
            sections.append(f"当前主题：{session.active_topic.topic_label}")
        if session.active_topic and session.active_topic.summary:
            sections.append(f"当前主题摘要：{session.active_topic.summary}")

        if memory.rolling_summary:
            sections.append(f"历史摘要：{memory.rolling_summary}")

        if memory.confirmed_facts:
            sections.append("已确认事实：" + "；".join(memory.confirmed_facts[:12]))

        if memory.unresolved_questions:
            sections.append("未解决问题：" + "；".join(memory.unresolved_questions[:8]))

        if memory.last_answer_summary and intent.needs_history:
            sections.append(f"上一轮结论：{memory.last_answer_summary}")

        if memory.last_citations and intent.needs_history:
            refs = [
                f"{item.get('law_name', '')}{item.get('article_num', '')}".strip()
                for item in memory.last_citations[:5]
            ]
            refs = [item for item in refs if item]
            if refs:
                sections.append("上一轮引用法条：" + "；".join(refs))

        recent_turns = session.turns[-self.recent_turn_window :]
        recent_lines = []
        for turn in recent_turns:
            role = "用户" if turn.role == "user" else "助手"
            recent_lines.append(f"{role}：{turn.content}")
        if recent_lines and intent.needs_history:
            sections.append("近期对话：\n" + "\n".join(recent_lines))

        recalled_topic_lines = []
        for topic in recalled_topics[:4]:
            if not topic.topic_label and not topic.summary:
                continue
            recalled_topic_lines.append(
                f"- [{topic.topic_id}] {topic.topic_label or '未命名主题'}：{topic.summary or '无摘要'}"
            )
        if recalled_topic_lines:
            sections.append("召回主题：\n" + "\n".join(recalled_topic_lines))

        recalled_memory_lines = []
        for item in recalled_memories[:10]:
            if item.status == "invalidated":
                continue
            recalled_memory_lines.append(
                f"- [{item.memory_id}] ({item.memory_type}, importance={item.importance:.2f}, status={item.status}) {item.content}"
            )
        if recalled_memory_lines:
            sections.append("召回记忆：\n" + "\n".join(recalled_memory_lines))

        sections.append(f"当前问题：{query}")
        merged = "\n\n".join([item for item in sections if item]).strip()
        if len(merged) <= self.max_history_chars:
            return merged
        return merged[-self.max_history_chars :]
