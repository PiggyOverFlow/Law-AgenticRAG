from __future__ import annotations

from datetime import datetime
from typing import List

from src.models.dialogue import MemoryItem, SessionState, TopicState, TurnRecord


class ConversationCompressor:
    def __init__(
        self,
        recent_turn_window: int = 6,
        summary_trigger_turns: int = 8,
        max_summary_chars: int = 1200,
    ):
        self.recent_turn_window = max(1, int(recent_turn_window))
        self.summary_trigger_turns = max(self.recent_turn_window + 1, int(summary_trigger_turns))
        self.max_summary_chars = max(300, int(max_summary_chars))

    def compress_if_needed(self, session: SessionState) -> SessionState:
        self._refresh_topic_summaries(session)
        if len(session.turns) <= self.summary_trigger_turns:
            return session

        keep_count = self.recent_turn_window * 2
        if len(session.turns) <= keep_count:
            return session

        archive_turns = session.turns[:-keep_count]
        retained_turns = session.turns[-keep_count:]
        summary_lines = self._summarize_turns(archive_turns)

        if session.memory.rolling_summary:
            summary_lines.insert(0, session.memory.rolling_summary)

        session.memory.rolling_summary = self._clip_text(" ".join(summary_lines), self.max_summary_chars)
        session.turns = retained_turns
        session.updated_at = datetime.utcnow()
        return session

    def _refresh_topic_summaries(self, session: SessionState) -> None:
        all_topics = []
        if session.active_topic:
            all_topics.append(session.active_topic)
        all_topics.extend(session.topic_history)
        for topic in all_topics:
            topic.summary = self._build_topic_summary(topic, session.memory.memory_items)

    def _build_topic_summary(self, topic: TopicState, memory_items: List[MemoryItem]) -> str:
        items = [
            item
            for item in memory_items
            if item.topic_id == topic.topic_id and item.status == "active"
        ]
        items.sort(key=lambda x: (-float(x.importance), x.updated_at.isoformat()))
        lines = []
        for item in items[:6]:
            lines.append(f"{item.memory_type}:{self._clip_text(item.content, 80)}")
        merged = "；".join(lines)
        return self._clip_text(merged, 260)

    def _summarize_turns(self, turns: List[TurnRecord]) -> List[str]:
        lines: List[str] = []
        for turn in turns:
            if turn.role == "user":
                lines.append(f"用户曾提问：{self._clip_text(turn.content, 120)}")
            elif turn.role == "assistant":
                lines.append(f"助手曾回答：{self._clip_text(turn.content, 160)}")
        return lines[-12:]

    def _clip_text(self, text: str, max_chars: int) -> str:
        raw = str(text or "").strip()
        if len(raw) <= max_chars:
            return raw
        return raw[: max(0, max_chars - 3)] + "..."
