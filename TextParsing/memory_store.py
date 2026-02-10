from __future__ import annotations

"""
SQLite 기반 대화 이력 저장소

역할:
- 질문/답변/피드백/날짜만 저장한다.
"""

import os
import sqlite3
import time


class SessionMemoryStore:
    """
    SessionMemoryStore는 질문/답변/피드백을 SQLite에 저장한다.

    Args:
        db_path: SQLite 파일 경로
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """
        SQLite 연결을 생성한다.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        """
        대화 로그 테이블을 초기화한다.
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_turn(self, question: str, answer: str) -> None:
        """
        질문/답변을 저장한다. rating은 NULL로 둔다.
        """
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_log (question, answer, rating, created_at)
                VALUES (?, ?, NULL, ?)
                """,
                (question, answer, created_at),
            )

    def update_rating(self, question: str, answer: str, rating: int) -> bool:
        """
        특정 질문/답변 행에 rating을 기록한다.

        Returns:
            bool: 업데이트 성공 여부
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM chat_log
                WHERE question = ? AND answer = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (question, answer),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE chat_log SET rating = ? WHERE id = ?",
                (int(rating), row[0]),
            )
        return True
