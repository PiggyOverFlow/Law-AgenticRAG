from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    turn_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    intent: str = ""
    rewritten_query: str = ""
    citations: List[Dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TopicState(BaseModel):
    topic_id: str
    topic_label: str = ""
    status: Literal["active", "paused", "closed"] = "active"
    summary: str = ""
    embedding: List[float] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_claims: List[str] = Field(default_factory=list)
    key_disputes: List[str] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)
    last_turn_id: str = ""


class MemoryItem(BaseModel):
    memory_id: str
    topic_id: str = ""
    memory_type: str = "fact"
    content: str
    embedding: List[float] = Field(default_factory=list)
    structured_payload: Dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.0
    retention_score: float = 0.0
    status: Literal["active", "stale", "invalidated"] = "active"
    source_turn_id: str = ""
    derived_from: List[str] = Field(default_factory=list)
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryOperation(BaseModel):
    op: Literal["add", "invalidate", "mark_stale", "replace", "noop"] = "noop"
    target_id: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class MemoryEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: Literal["supports", "derived_from", "contradicts", "belongs_to_topic", "replaced_by"] = "supports"
    weight: float = 1.0
    status: Literal["active", "stale", "invalidated"] = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentSemanticMemory(BaseModel):
    user_style_preferences: Dict[str, Any] = Field(default_factory=dict)
    recurring_patterns: List[str] = Field(default_factory=list)
    reusable_strategies: List[str] = Field(default_factory=list)
    stable_fact_rules: List[str] = Field(default_factory=list)
    preferred_query_styles: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EpisodeRecord(BaseModel):
    episode_id: str
    topic_id: str = ""
    title: str = ""
    goal: str = ""
    turn_ids: List[str] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)
    result_summary: str = ""
    status: Literal["active", "completed", "aborted", "paused"] = "active"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class MemorySnapshot(BaseModel):
    rolling_summary: str = ""
    confirmed_facts: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    user_preferences: Dict = Field(default_factory=dict)
    last_answer_summary: str = ""
    last_citations: List[Dict] = Field(default_factory=list)
    memory_items: List[MemoryItem] = Field(default_factory=list)
    memory_edges: List[MemoryEdge] = Field(default_factory=list)
    semantic_memory: AgentSemanticMemory = Field(default_factory=AgentSemanticMemory)


class IntentResult(BaseModel):
    dialogue_intent: str = "new_question"
    task_intent: str = "civil_issue_analysis"
    needs_history: bool = False
    needs_rewrite: bool = False
    topic_switch: bool = False
    topic_label: str = ""
    retrieval_mode: str = "active_topic"
    memory_operations: List[MemoryOperation] = Field(default_factory=list)
    extracted_memories: List[MemoryItem] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""


class QueryResolution(BaseModel):
    resolved_query: str
    resolution_context: str = ""
    rewritten_from_history: bool = False
    referenced_turn_ids: List[str] = Field(default_factory=list)
    retrieved_topic_ids: List[str] = Field(default_factory=list)
    retrieved_memory_ids: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str
    version: int = 0
    turns: List[TurnRecord] = Field(default_factory=list)
    active_topic: Optional[TopicState] = None
    topic_history: List[TopicState] = Field(default_factory=list)
    episodes: List[EpisodeRecord] = Field(default_factory=list)
    current_episode_id: str = ""
    memory: MemorySnapshot = Field(default_factory=MemorySnapshot)
    case_date: str = ""
    document_context: Dict = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
