from __future__ import annotations

import re
from typing import List

from src.models.dialogue import IntentResult, MemoryItem, QueryResolution, SessionState, TopicState


class QueryResolver:
    def resolve(
        self,
        query: str,
        intent: IntentResult,
        session: SessionState,
        context: str,
        recalled_topics: List[TopicState] | None = None,
        recalled_memories: List[MemoryItem] | None = None,
    ) -> QueryResolution:
        normalized = re.sub(r"\s+", " ", str(query or "").strip())
        if not normalized:
            return QueryResolution(resolved_query="", resolution_context=context)

        recalled_topics = recalled_topics or []
        recalled_memories = recalled_memories or []

        if intent.dialogue_intent == "topic_switch":
            return QueryResolution(
                resolved_query=normalized,
                resolution_context=f"新话题开始。\n当前问题：{normalized}",
                rewritten_from_history=False,
            )

        referenced_turn_ids: List[str] = [turn.turn_id for turn in session.turns[-4:]]
        retrieved_topic_ids = [topic.topic_id for topic in recalled_topics]
        retrieved_memory_ids = [item.memory_id for item in recalled_memories]

        if intent.needs_rewrite and (session.turns or recalled_memories or recalled_topics):
            prefix_parts = []
            if recalled_topics:
                prefix_parts.append(
                    "相关主题：" + "；".join(
                        [
                            f"{topic.topic_label}:{topic.summary or '无摘要'}"
                            for topic in recalled_topics[:2]
                            if topic.topic_label or topic.summary
                        ]
                    )
                )
            if recalled_memories:
                prefix_parts.append(
                    "相关记忆：" + "；".join(
                        [item.content for item in recalled_memories[:6] if item.status != "invalidated"]
                    )
                )
            elif session.memory.last_answer_summary:
                prefix_parts.append(f"上轮结论：{session.memory.last_answer_summary}")
            if session.memory.confirmed_facts:
                prefix_parts.append("已确认事实：" + "；".join(session.memory.confirmed_facts[:8]))
            prefix_parts.append(f"当前追问：{normalized}")
            resolved_query = " ".join(prefix_parts).strip()
            return QueryResolution(
                resolved_query=resolved_query,
                resolution_context=context,
                rewritten_from_history=True,
                referenced_turn_ids=referenced_turn_ids,
                retrieved_topic_ids=retrieved_topic_ids,
                retrieved_memory_ids=retrieved_memory_ids,
            )

        return QueryResolution(
            resolved_query=normalized,
            resolution_context=context,
            rewritten_from_history=False,
            referenced_turn_ids=referenced_turn_ids,
            retrieved_topic_ids=retrieved_topic_ids,
            retrieved_memory_ids=retrieved_memory_ids,
        )
