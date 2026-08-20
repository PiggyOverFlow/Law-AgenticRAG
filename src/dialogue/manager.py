from __future__ import annotations

import logging
import math
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import get_config
from src.dialogue.compressor import ConversationCompressor
from src.dialogue.context_builder import ContextBuilder
from src.dialogue.intent import IntentRecognizer
from src.dialogue.memory_store import ConcurrencyError, SQLiteSessionStore
from src.dialogue.resolver import QueryResolver
from src.models.dialogue import (
    EpisodeRecord,
    IntentResult,
    MemoryEdge,
    MemoryItem,
    MemoryOperation,
    SessionState,
    TopicState,
    TurnRecord,
)


logger = logging.getLogger(__name__)


class DialogueManager:
    def __init__(self, rag):
        self.config = get_config()
        self.rag = rag
        dialogue_cfg = self.config.dialogue
        db_path = dialogue_cfg.session_db_path or self.config.database.path
        self.store = SQLiteSessionStore(
            db_path=db_path,
            table_name=dialogue_cfg.session_table,
        )
        self.intent_recognizer = IntentRecognizer()
        self.context_builder = ContextBuilder(
            recent_turn_window=dialogue_cfg.recent_turn_window,
            max_history_chars=dialogue_cfg.max_history_chars,
        )
        self.resolver = QueryResolver()
        self.compressor = ConversationCompressor(
            recent_turn_window=dialogue_cfg.recent_turn_window,
            summary_trigger_turns=dialogue_cfg.summary_trigger_turns,
            max_summary_chars=dialogue_cfg.max_summary_chars,
        )
        self.embedding_model = getattr(self.rag, "embedding_model", None)

    def handle_qa_turn(
        self,
        session_id: str,
        query: str,
        case_date: Optional[str] = None,
        reset_session: bool = False,
    ) -> Dict[str, Any]:
        safe_session_id = str(session_id or "default").strip() or "default"
        if reset_session:
            self.store.delete(safe_session_id)
        last_conflict = None
        for _ in range(2):
            session = self.store.get(safe_session_id) or SessionState(session_id=safe_session_id)
            if case_date:
                session.case_date = case_date

            recalled_topics = self._retrieve_relevant_topics(session, query)
            recalled_memories = self._retrieve_relevant_memories(session, query, recalled_topics)
            intent = self.intent_recognizer.detect(
                query=query,
                session=session,
                recalled_topics=[topic.model_dump() for topic in recalled_topics],
                recalled_memories=[item.model_dump() for item in recalled_memories],
            )

            active_topic = self._select_topic_for_turn(session, intent, recalled_topics, query)
            self._ensure_episode_for_turn(session, active_topic, intent, query)
            context = self.context_builder.build(
                session=session,
                intent=intent,
                query=query,
                recalled_topics=recalled_topics,
                recalled_memories=recalled_memories,
            )
            resolved = self.resolver.resolve(
                query=query,
                intent=intent,
                session=session,
                context=context,
                recalled_topics=recalled_topics,
                recalled_memories=recalled_memories,
            )

            filters = {"case_date": case_date or session.case_date} if (case_date or session.case_date) else None
            result = self.rag.answer_with_citations(
                resolved.resolved_query,
                filters=filters,
                conversation_context=resolved.resolution_context,
                memory_snapshot=session.memory.model_dump(),
            )

            self._update_session_after_turn(
                session=session,
                query=query,
                result=result,
                intent=intent,
                rewritten_query=resolved.resolved_query if resolved.rewritten_from_history else "",
                active_topic=active_topic,
                recalled_memories=recalled_memories,
            )
            session = self.compressor.compress_if_needed(session)
            self._refresh_all_topic_embeddings(session)
            self._promote_semantic_memories(session)
            self._run_forgetting_scheduler(session)
            try:
                self.store.save(session)
                result["session_id"] = safe_session_id
                result["dialogue_intent"] = intent.dialogue_intent
                result["task_intent"] = intent.task_intent
                result["resolved_query"] = resolved.resolved_query
                result["memory"] = session.memory.model_dump()
                result["recalled_topic_ids"] = resolved.retrieved_topic_ids
                result["recalled_memory_ids"] = resolved.retrieved_memory_ids
                result["episode_id"] = session.current_episode_id
                result["session_version"] = session.version
                return result
            except ConcurrencyError as exc:
                last_conflict = exc
                logger.warning("session %s 保存冲突，重试一次: %s", safe_session_id, exc)
                continue
        raise RuntimeError(f"session {safe_session_id} 保存失败: {last_conflict}")

    def get_session(self, session_id: str) -> SessionState:
        safe_session_id = str(session_id or "default").strip() or "default"
        return self.store.get(safe_session_id) or SessionState(session_id=safe_session_id)

    def reset_session(self, session_id: str) -> None:
        safe_session_id = str(session_id or "default").strip() or "default"
        self.store.delete(safe_session_id)

    def _retrieve_relevant_topics(self, session: SessionState, query: str) -> List[TopicState]:
        topics: List[TopicState] = []
        if session.active_topic:
            topics.append(session.active_topic)
        topics.extend(session.topic_history)
        query_embedding = self._embed_text(query)

        scored = []
        for topic in topics:
            score = self._semantic_score(query, query_embedding, " ".join([topic.topic_label, topic.summary]), topic.embedding)
            if session.active_topic and topic.topic_id == session.active_topic.topic_id:
                score += 0.35
            scored.append((score, topic))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, int(getattr(self.config.dialogue, "topic_recall_top_k", 3)))
        return [topic for score, topic in scored[:top_k] if score > 0]

    def _retrieve_relevant_memories(
        self,
        session: SessionState,
        query: str,
        topics: List[TopicState],
    ) -> List[MemoryItem]:
        topic_ids = {topic.topic_id for topic in topics if topic.topic_id}
        query_embedding = self._embed_text(query)
        scored = []
        for item in session.memory.memory_items:
            if item.status == "invalidated":
                continue
            memory_text = self._memory_recall_text(item)
            score = self._semantic_score(query, query_embedding, memory_text, item.embedding)
            score += float(item.importance) * 0.6
            if item.topic_id and item.topic_id in topic_ids:
                score += 0.3
            if item.status == "stale":
                score -= 0.2
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, int(getattr(self.config.dialogue, "memory_recall_top_k", 8)))
        results = [item for score, item in scored[:top_k] if score > 0]
        for item in results:
            item.access_count += 1
            item.last_accessed_at = datetime.utcnow()
        return results

    def _select_topic_for_turn(
        self,
        session: SessionState,
        intent: IntentResult,
        recalled_topics: List[TopicState],
        query: str,
    ) -> TopicState:
        if intent.topic_switch:
            self._close_active_topic(session, close_status="closed")
            new_topic = TopicState(
                topic_id=uuid.uuid4().hex,
                topic_label=intent.topic_label or self._clip_text(query, 40),
                status="active",
            )
            new_topic.embedding = self._embed_text(new_topic.topic_label)
            session.active_topic = new_topic
            return new_topic

        if intent.retrieval_mode == "topic_recall" and recalled_topics:
            chosen = recalled_topics[0]
            self._activate_existing_topic(session, chosen.topic_id)
            if session.active_topic:
                if intent.topic_label:
                    session.active_topic.topic_label = intent.topic_label
                return session.active_topic

        if session.active_topic is None:
            session.active_topic = TopicState(
                topic_id=uuid.uuid4().hex,
                topic_label=intent.topic_label or self._clip_text(query, 40),
                status="active",
            )
            session.active_topic.embedding = self._embed_text(session.active_topic.topic_label)
        elif intent.topic_label:
            session.active_topic.topic_label = intent.topic_label
            session.active_topic.embedding = self._embed_text(
                " ".join([session.active_topic.topic_label, session.active_topic.summary]).strip()
            )
        return session.active_topic

    def _ensure_episode_for_turn(
        self,
        session: SessionState,
        active_topic: TopicState,
        intent: IntentResult,
        query: str,
    ) -> None:
        current = self._get_episode_by_id(session, session.current_episode_id)
        if current and current.topic_id == active_topic.topic_id and current.status == "active":
            current.updated_at = datetime.utcnow()
            if not current.goal:
                current.goal = self._clip_text(query, 80)
            return

        if current and current.status == "active":
            current.status = "paused"
            current.updated_at = datetime.utcnow()

        for episode in session.episodes:
            if episode.topic_id == active_topic.topic_id and episode.status in {"paused", "active"}:
                episode.status = "active"
                episode.updated_at = datetime.utcnow()
                episode.goal = episode.goal or self._clip_text(query, 80)
                session.current_episode_id = episode.episode_id
                return

        episode = EpisodeRecord(
            episode_id=uuid.uuid4().hex,
            topic_id=active_topic.topic_id,
            title=active_topic.topic_label or self._clip_text(query, 40),
            goal=self._clip_text(query, 100),
            status="active",
        )
        session.episodes.append(episode)
        session.current_episode_id = episode.episode_id

    def _update_session_after_turn(
        self,
        session: SessionState,
        query: str,
        result: Dict[str, Any],
        intent: IntentResult,
        rewritten_query: str,
        active_topic: TopicState,
        recalled_memories: List[MemoryItem],
    ) -> None:
        now = datetime.utcnow()
        user_turn_id = uuid.uuid4().hex
        session.turns.append(
            TurnRecord(
                turn_id=user_turn_id,
                role="user",
                content=str(query or "").strip(),
                intent=intent.dialogue_intent,
                rewritten_query=rewritten_query,
                created_at=now,
            )
        )

        self._apply_memory_operations(
            session=session,
            operations=intent.memory_operations,
            active_topic=active_topic,
        )
        self._register_turn_to_episode(session, user_turn_id)
        self._write_extracted_memories(
            session=session,
            extracted_memories=intent.extracted_memories,
            active_topic=active_topic,
            source_turn_id=user_turn_id,
        )

        answer_text = str(result.get("answer", "") or "").strip()
        assistant_turn_id = uuid.uuid4().hex
        session.turns.append(
            TurnRecord(
                turn_id=assistant_turn_id,
                role="assistant",
                content=answer_text,
                intent="answer",
                citations=list(result.get("citations", []) or []),
                created_at=now,
            )
        )

        active_memory_ids = [
            item.memory_id
            for item in session.memory.memory_items
            if item.topic_id == active_topic.topic_id and item.status == "active"
        ]
        self._write_result_memories(
            session=session,
            result=result,
            active_topic=active_topic,
            assistant_turn_id=assistant_turn_id,
            derived_from=active_memory_ids,
        )
        self._register_turn_to_episode(session, assistant_turn_id)
        self._sync_snapshot_fields(session=session, result=result, active_topic=active_topic)
        active_topic.last_turn_id = assistant_turn_id
        session.updated_at = now

    def _apply_memory_operations(
        self,
        session: SessionState,
        operations: List[MemoryOperation],
        active_topic: TopicState,
    ) -> None:
        for operation in operations:
            if operation.op == "invalidate" and operation.target_id:
                self._invalidate_memory(session, operation.target_id)
            elif operation.op == "mark_stale" and operation.target_id:
                self._mark_memory_stale(session, operation.target_id)
            elif operation.op == "replace" and operation.target_id:
                self._invalidate_memory(session, operation.target_id)
                payload = dict(operation.payload or {})
                content = str(payload.get("content", "")).strip()
                if content:
                    self._upsert_memory_item(
                        session=session,
                        item=MemoryItem(
                            memory_id=uuid.uuid4().hex,
                            topic_id=active_topic.topic_id,
                            memory_type=str(payload.get("memory_type", "fact")).strip() or "fact",
                            memory_scope=str(payload.get("memory_scope", "episodic")).strip() or "episodic",
                            content=content,
                            structured_payload=dict(payload.get("structured_payload", {}) or {}),
                            importance=float(payload.get("importance", 0.85) or 0.85),
                            source_turn_id="current",
                        ),
                    )

    def _write_extracted_memories(
        self,
        session: SessionState,
        extracted_memories: List[MemoryItem],
        active_topic: TopicState,
        source_turn_id: str,
    ) -> None:
        for item in extracted_memories:
            item.topic_id = item.topic_id or active_topic.topic_id
            item.source_turn_id = source_turn_id
            item.episode_id = session.current_episode_id
            item.memory_scope = item.memory_scope or "episodic"
            item.updated_at = datetime.utcnow()
            item.created_at = item.created_at or datetime.utcnow()
            self._upsert_memory_item(session, item)

    def _write_result_memories(
        self,
        session: SessionState,
        result: Dict[str, Any],
        active_topic: TopicState,
        assistant_turn_id: str,
        derived_from: List[str],
    ) -> None:
        answer_text = self._clip_text(str(result.get("answer", "") or "").strip(), 240)
        if answer_text:
            self._upsert_memory_item(
                session=session,
                item=MemoryItem(
                    memory_id=uuid.uuid4().hex,
                    topic_id=active_topic.topic_id,
                    memory_type="conclusion",
                    memory_scope="episodic",
                    content=answer_text,
                    importance=0.82,
                    source_turn_id=assistant_turn_id,
                    episode_id=session.current_episode_id,
                    derived_from=list(derived_from),
                ),
            )

        for citation in list(result.get("citations", []) or [])[:5]:
            citation_text = f"{citation.get('law_name', '')} {citation.get('article_num', '')}".strip()
            if not citation_text:
                continue
            self._upsert_memory_item(
                session=session,
                item=MemoryItem(
                    memory_id=uuid.uuid4().hex,
                    topic_id=active_topic.topic_id,
                    memory_type="citation",
                    memory_scope="episodic",
                    content=citation_text,
                    structured_payload=dict(citation or {}),
                    importance=0.76,
                    source_turn_id=assistant_turn_id,
                    episode_id=session.current_episode_id,
                    derived_from=list(derived_from),
                ),
            )
        episode = self._get_episode_by_id(session, session.current_episode_id)
        if episode:
            episode.result_summary = answer_text
            episode.citations = list(result.get("citations", []) or [])[:5]
            episode.updated_at = datetime.utcnow()

    def _sync_snapshot_fields(
        self,
        session: SessionState,
        result: Dict[str, Any],
        active_topic: TopicState,
    ) -> None:
        memory = session.memory
        active_items = [
            item
            for item in memory.memory_items
            if item.topic_id == active_topic.topic_id and item.status == "active"
        ]
        active_items.sort(key=lambda x: (-float(x.importance), x.updated_at.isoformat()))

        memory.confirmed_facts = [
            item.content
            for item in active_items
            if item.memory_type in {"fact", "claim", "correction"}
        ][: self.config.dialogue.max_confirmed_facts]
        memory.unresolved_questions = [
            item.content
            for item in active_items
            if item.memory_type == "question"
        ][: self.config.dialogue.max_unresolved_questions]
        memory.last_answer_summary = self._clip_text(str(result.get("answer", "") or "").strip(), 240)
        memory.last_citations = list(result.get("citations", []) or [])[:5]
        topic_preferences = [
            item
            for item in active_items
            if item.memory_type == "preference" and item.status == "active"
        ]
        for item in topic_preferences:
            key = str(item.structured_payload.get("key", "")).strip()
            value = item.structured_payload.get("value")
            if key:
                memory.user_preferences[key] = value

    def _upsert_memory_item(self, session: SessionState, item: MemoryItem) -> None:
        item.updated_at = datetime.utcnow()
        existing = None
        for memory in session.memory.memory_items:
            if (
                memory.topic_id == item.topic_id
                and memory.memory_type == item.memory_type
                and memory.content.strip() == item.content.strip()
                and memory.status != "invalidated"
            ):
                existing = memory
                break

        if existing:
            existing.importance = max(float(existing.importance), float(item.importance))
            existing.structured_payload.update(item.structured_payload)
            existing.updated_at = datetime.utcnow()
            existing.status = "active"
            if item.source_turn_id:
                existing.source_turn_id = item.source_turn_id
            for dep in item.derived_from:
                if dep and dep not in existing.derived_from:
                    existing.derived_from.append(dep)
            if item.episode_id:
                existing.episode_id = item.episode_id
            existing.memory_scope = item.memory_scope or existing.memory_scope
            existing.embedding = self._embed_text(self._memory_recall_text(existing))
            target_memory = existing
        else:
            if not item.memory_id:
                item.memory_id = uuid.uuid4().hex
            item.embedding = self._embed_text(self._memory_recall_text(item))
            session.memory.memory_items.append(item)
            target_memory = item

        topic = self._get_topic_by_id(session, target_memory.topic_id)
        if topic and target_memory.memory_id not in topic.memory_ids:
            topic.memory_ids.append(target_memory.memory_id)
            self._refresh_topic_embedding(topic, session)
        elif topic:
            self._refresh_topic_embedding(topic, session)
        self._register_memory_to_episode(session, target_memory.memory_id)
        self._link_memory_graph(session, target_memory)

    def _invalidate_memory(self, session: SessionState, memory_id: str) -> None:
        target = self._find_memory(session, memory_id)
        if not target:
            return
        target.status = "invalidated"
        target.updated_at = datetime.utcnow()
        self._mark_edges_for_memory(session, memory_id, "invalidated")
        self._propagate_invalidation(session, target.memory_id)

    def _mark_memory_stale(self, session: SessionState, memory_id: str) -> None:
        target = self._find_memory(session, memory_id)
        if not target or target.status == "invalidated":
            return
        target.status = "stale"
        target.updated_at = datetime.utcnow()
        self._mark_edges_for_memory(session, memory_id, "stale")

    def _propagate_invalidation(self, session: SessionState, memory_id: str) -> None:
        for item in session.memory.memory_items:
            if item.status == "invalidated":
                continue
            if memory_id in item.derived_from:
                item.status = "stale"
                item.updated_at = datetime.utcnow()
                self._mark_edges_for_memory(session, item.memory_id, "stale")

    def _find_memory(self, session: SessionState, memory_id: str) -> MemoryItem | None:
        for item in session.memory.memory_items:
            if item.memory_id == memory_id:
                return item
        return None

    def _activate_existing_topic(self, session: SessionState, topic_id: str) -> None:
        if session.active_topic and session.active_topic.topic_id == topic_id:
            session.active_topic.status = "active"
            return
        if session.active_topic:
            self._close_active_topic(session, close_status="paused")
        for idx, topic in enumerate(session.topic_history):
            if topic.topic_id == topic_id:
                topic.status = "active"
                session.active_topic = topic
                del session.topic_history[idx]
                return

    def _close_active_topic(self, session: SessionState, close_status: str = "closed") -> None:
        if not session.active_topic:
            return
        session.active_topic.status = close_status
        session.topic_history.append(session.active_topic)
        session.active_topic = None

    def _get_topic_by_id(self, session: SessionState, topic_id: str) -> TopicState | None:
        if session.active_topic and session.active_topic.topic_id == topic_id:
            return session.active_topic
        for topic in session.topic_history:
            if topic.topic_id == topic_id:
                return topic
        return None

    def _refresh_all_topic_embeddings(self, session: SessionState) -> None:
        if session.active_topic:
            self._refresh_topic_embedding(session.active_topic, session)
        for topic in session.topic_history:
            self._refresh_topic_embedding(topic, session)

    def _refresh_topic_embedding(self, topic: TopicState, session: SessionState) -> None:
        memory_texts = []
        for item in session.memory.memory_items:
            if item.topic_id == topic.topic_id and item.status == "active":
                memory_texts.append(self._memory_recall_text(item))
                self._collect_topic_statistics(topic, item)
        topic_text = " ".join([topic.topic_label, topic.summary] + memory_texts[:6]).strip()
        topic.embedding = self._embed_text(topic_text)

    def _collect_topic_statistics(self, topic: TopicState, item: MemoryItem) -> None:
        entity_candidates = [
            str(item.structured_payload.get("subject", "") or ""),
            str(item.structured_payload.get("object", "") or ""),
        ]
        for value in entity_candidates:
            if value and value not in topic.key_entities:
                topic.key_entities.append(value)
        claim = str(item.structured_payload.get("claim", "") or "")
        if claim and claim not in topic.key_claims:
            topic.key_claims.append(claim)
        if item.memory_type in {"fact", "claim", "question"} and item.content not in topic.key_disputes:
            topic.key_disputes.append(item.content[:80])

    def _score_text(self, query: str, text: str) -> float:
        q_terms = set(self._extract_terms(query))
        t_terms = set(self._extract_terms(text))
        if not q_terms or not t_terms:
            return 0.0
        overlap = q_terms.intersection(t_terms)
        score = float(len(overlap))
        raw_query = str(query or "").strip()
        raw_text = str(text or "").strip()
        if raw_query and raw_query in raw_text:
            score += 2.0
        score += sum(min(len(term), 6) * 0.05 for term in overlap)
        return score

    def _semantic_score(
        self,
        query: str,
        query_embedding: List[float],
        text: str,
        target_embedding: List[float],
    ) -> float:
        lexical = self._score_text(query, text)
        semantic = self._cosine_similarity(query_embedding, target_embedding)
        if semantic <= 0:
            return lexical
        return lexical * 0.35 + semantic * 2.5

    def _memory_recall_text(self, item: MemoryItem) -> str:
        parts = [
            item.content,
            str(item.structured_payload.get("subject", "") or ""),
            str(item.structured_payload.get("object", "") or ""),
            str(item.structured_payload.get("relation", "") or ""),
            str(item.structured_payload.get("claim", "") or ""),
            str(item.structured_payload.get("time", "") or ""),
            str(item.structured_payload.get("basis", "") or ""),
            str(item.structured_payload.get("evidence", "") or ""),
        ]
        return " ".join([part for part in parts if part]).strip()

    def _embed_text(self, text: str) -> List[float]:
        raw = str(text or "").strip()
        if not raw:
            return []
        if self.embedding_model is None:
            return []
        try:
            vector = self.embedding_model.encode_single(raw)
            if isinstance(vector, list):
                return [float(x) for x in vector]
        except Exception as exc:
            logger.warning("会话向量化失败，回退词法 recall: %s", exc)
        return []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _extract_terms(self, text: str) -> List[str]:
        return re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,16}", str(text or ""))

    def _clip_text(self, text: str, max_chars: int) -> str:
        raw = str(text or "").strip()
        if len(raw) <= max_chars:
            return raw
        return raw[: max(0, max_chars - 3)] + "..."

    def _get_episode_by_id(self, session: SessionState, episode_id: str) -> EpisodeRecord | None:
        if not episode_id:
            return None
        for episode in session.episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def _register_turn_to_episode(self, session: SessionState, turn_id: str) -> None:
        episode = self._get_episode_by_id(session, session.current_episode_id)
        if not episode:
            return
        if turn_id not in episode.turn_ids:
            episode.turn_ids.append(turn_id)
            episode.updated_at = datetime.utcnow()

    def _register_memory_to_episode(self, session: SessionState, memory_id: str) -> None:
        episode = self._get_episode_by_id(session, session.current_episode_id)
        if not episode:
            return
        if memory_id not in episode.memory_ids:
            episode.memory_ids.append(memory_id)
            episode.updated_at = datetime.utcnow()

    def _link_memory_graph(self, session: SessionState, item: MemoryItem) -> None:
        if item.topic_id:
            self._upsert_edge(
                session,
                source_id=item.memory_id,
                target_id=item.topic_id,
                relation="belongs_to_topic",
            )
        for dep in item.derived_from:
            if dep:
                self._upsert_edge(
                    session,
                    source_id=item.memory_id,
                    target_id=dep,
                    relation="derived_from",
                )

    def _upsert_edge(self, session: SessionState, source_id: str, target_id: str, relation: str) -> None:
        for edge in session.memory.memory_edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == relation:
                return
        session.memory.memory_edges.append(
            MemoryEdge(
                edge_id=uuid.uuid4().hex,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
            )
        )

    def _mark_edges_for_memory(self, session: SessionState, memory_id: str, edge_status: str) -> None:
        for edge in session.memory.memory_edges:
            if edge.source_id == memory_id or edge.target_id == memory_id:
                edge.status = edge_status

    def _promote_semantic_memories(self, session: SessionState) -> None:
        counts: Dict[str, List[MemoryItem]] = {}
        for item in session.memory.memory_items:
            if item.status != "active" or item.memory_scope == "semantic":
                continue
            key = self._normalize_memory_key(item)
            counts.setdefault(key, []).append(item)

        for _, items in counts.items():
            source = max(items, key=lambda x: (float(x.importance), x.updated_at.isoformat()))
            repeated_across_episodes = len({item.episode_id for item in items if item.episode_id}) >= 2
            if float(source.importance) < 0.88 and not repeated_across_episodes:
                continue
            semantic_item = MemoryItem(
                memory_id=uuid.uuid4().hex,
                topic_id="",
                memory_type=source.memory_type,
                memory_scope="semantic",
                content=source.content,
                structured_payload=dict(source.structured_payload or {}),
                importance=max(0.9, float(source.importance)),
                source_turn_id=source.source_turn_id,
                derived_from=[source.memory_id],
                episode_id=source.episode_id,
            )
            self._upsert_memory_item(session, semantic_item)
            self._update_semantic_profile(session, semantic_item)

    def _update_semantic_profile(self, session: SessionState, item: MemoryItem) -> None:
        semantic = session.memory.semantic_memory
        if item.memory_type == "preference":
            key = str(item.structured_payload.get("key", "")).strip()
            value = item.structured_payload.get("value")
            if key:
                semantic.user_style_preferences[key] = value
        relation = str(item.structured_payload.get("relation", "") or "")
        claim = str(item.structured_payload.get("claim", "") or "")
        if relation and relation not in semantic.stable_fact_rules:
            semantic.stable_fact_rules.append(relation)
        if claim and claim not in semantic.recurring_patterns:
            semantic.recurring_patterns.append(claim)
        strategy = f"{item.memory_type}:{item.content[:80]}"
        if strategy not in semantic.reusable_strategies:
            semantic.reusable_strategies.append(strategy)
        semantic.updated_at = datetime.utcnow()

    def _run_forgetting_scheduler(self, session: SessionState) -> None:
        now = datetime.utcnow()
        budget = 200
        for item in session.memory.memory_items:
            age_days = max(0.0, (now - item.updated_at).total_seconds() / 86400.0)
            recency_score = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
            access_score = min(float(item.access_count) / 5.0, 1.0)
            status_penalty = -0.25 if item.status == "stale" else -1.0 if item.status == "invalidated" else 0.0
            item.retention_score = (
                float(item.importance) * 0.6
                + recency_score * 0.25
                + access_score * 0.15
                + status_penalty
            )

        semantic_items = [item for item in session.memory.memory_items if item.memory_scope == "semantic"]
        other_items = [item for item in session.memory.memory_items if item.memory_scope != "semantic"]
        other_items.sort(key=lambda x: (float(x.retention_score), x.updated_at.isoformat()))
        while len(semantic_items) + len(other_items) > budget and other_items:
            candidate = other_items.pop(0)
            if candidate.status == "invalidated" or float(candidate.retention_score) < 0.15:
                self._remove_memory(session, candidate.memory_id)
            else:
                candidate.status = "stale"
                break

    def _remove_memory(self, session: SessionState, memory_id: str) -> None:
        session.memory.memory_items = [
            item for item in session.memory.memory_items if item.memory_id != memory_id
        ]
        session.memory.memory_edges = [
            edge
            for edge in session.memory.memory_edges
            if edge.source_id != memory_id and edge.target_id != memory_id
        ]
        for topic in [session.active_topic] + list(session.topic_history):
            if topic and memory_id in topic.memory_ids:
                topic.memory_ids = [item for item in topic.memory_ids if item != memory_id]
        for episode in session.episodes:
            if memory_id in episode.memory_ids:
                episode.memory_ids = [item for item in episode.memory_ids if item != memory_id]

    def _normalize_memory_key(self, item: MemoryItem) -> str:
        parts = [
            item.memory_type,
            item.content.strip().lower(),
            str(item.structured_payload.get("subject", "")).strip().lower(),
            str(item.structured_payload.get("object", "")).strip().lower(),
            str(item.structured_payload.get("relation", "")).strip().lower(),
            str(item.structured_payload.get("claim", "")).strip().lower(),
        ]
        return "|".join(parts)
