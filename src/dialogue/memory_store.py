from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Protocol

from src.models.dialogue import SessionState


class ConcurrencyError(RuntimeError):
    pass


class SessionStore(Protocol):
    def get(self, session_id: str) -> Optional[SessionState]:
        ...

    def save(self, state: SessionState) -> None:
        ...

    def delete(self, session_id: str) -> None:
        ...


class SQLiteSessionStore:
    def __init__(self, db_path: str, table_name: str = "dialogue_sessions"):
        self.db_path = Path(db_path)
        self.table_name = table_name
        if self.db_path.parent and str(self.db_path.parent) != ".":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_column(conn, "version", "INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, column_name: str, column_def: str) -> None:
        columns = conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()
        column_names = {str(row[1]) for row in columns}
        if column_name not in column_names:
            conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {column_name} {column_def}")

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload, version FROM {self.table_name} WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        state = SessionState.model_validate_json(row[0])
        try:
            state.version = int(row[1] or 0)
        except Exception:
            state.version = 0
        return state

    def save(self, state: SessionState) -> None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT version FROM {self.table_name} WHERE session_id = ?",
                (state.session_id,),
            ).fetchone()
            current_version = int(row[0]) if row else None
            expected_version = int(getattr(state, "version", 0) or 0)

            if row is None:
                next_version = 1
                state.version = next_version
                payload = state.model_dump_json()
                updated_at = state.updated_at.isoformat()
                conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (session_id, payload, updated_at, version)
                    VALUES (?, ?, ?, ?)
                    """,
                    (state.session_id, payload, updated_at, next_version),
                )
            else:
                if current_version != expected_version:
                    raise ConcurrencyError(
                        f"session {state.session_id} version conflict: expected {expected_version}, current {current_version}"
                    )
                next_version = current_version + 1
                state.version = next_version
                payload = state.model_dump_json()
                updated_at = state.updated_at.isoformat()
                cursor = conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET payload = ?, updated_at = ?, version = ?
                    WHERE session_id = ? AND version = ?
                    """,
                    (payload, updated_at, next_version, state.session_id, current_version),
                )
                if cursor.rowcount == 0:
                    raise ConcurrencyError(
                        f"session {state.session_id} optimistic update failed at version {current_version}"
                    )
            conn.commit()

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
