from __future__ import annotations

"""
SQLite 기반 세션 메모리 저장소

역할:
- 멀티턴에서 이전 턴 문서 집합을 재사용하기 위한 캐시
- 세션별 최근 질문/답변/문서 ID 기록
"""

import os
import sqlite3
import time
from typing import Iterable, List, Optional


class SessionMemoryStore:
    """
    SessionMemoryStore는 세션별 문서/상태를 SQLite에 저장한다.

    Args:
        db_path: SQLite 파일 경로
    """

    def __init__(self, db_path: str, clear_on_start: bool = False) -> None:
        self.db_path = db_path
        # SQLite 파일 경로가 속한 디렉토리를 미리 생성한다.
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # 테이블 스키마를 준비한다.
        self._init_schema()
        # 프로세스 시작 시 전체 세션 데이터를 초기화할지 결정한다.
        if clear_on_start:
            self.clear_all()

    def _connect(self) -> sqlite3.Connection:
        """
        SQLite 연결을 생성한다.
        """
        # sqlite3.connect는 DB 파일이 없으면 자동 생성한다.
        conn = sqlite3.connect(self.db_path)
        # WAL은 동시 읽기/쓰기에 유리하고, NORMAL은 속도를 우선한다.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        """
        세션/문서 테이블을 초기화한다.
        """
        with self._connect() as conn:
            # 세션별 마지막 질문/답변/유형을 보관하는 테이블
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    last_question TEXT,
                    last_answer TEXT,
                    last_question_type TEXT,
                    updated_at REAL
                )
                """
            )
            # 세션별 문서 ID 목록을 보관하는 테이블
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_docs (
                    session_id TEXT,
                    doc_id TEXT,
                    rank INTEGER,
                    score REAL,
                    created_at REAL,
                    PRIMARY KEY (session_id, doc_id)
                )
                """
            )

    def load_doc_ids(self, session_id: str, limit: Optional[int] = None) -> List[str]:
        """
        세션에 저장된 문서 ID를 순서대로 반환한다.

        Args:
            session_id: 세션 식별자
            limit: 최대 반환 개수

        Returns:
            List[str]: 문서 ID 리스트
        """
        # rank 순서대로 doc_id를 반환한다.
        sql = "SELECT doc_id FROM session_docs WHERE session_id = ? ORDER BY rank ASC"
        params = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [row[0] for row in rows]

    def save_doc_ids(self, session_id: str, doc_ids: Iterable[str]) -> None:
        """
        세션의 문서 ID 목록을 저장한다.

        Args:
            session_id: 세션 식별자
            doc_ids: 문서 ID 리스트
        """
        # 저장 시각을 기록한다.
        now = time.time()
        with self._connect() as conn:
            # 세션별로 문서 목록을 완전히 덮어쓴다.
            conn.execute("DELETE FROM session_docs WHERE session_id = ?", (session_id,))
            for rank, doc_id in enumerate(doc_ids):
                # rank는 검색 결과 순서를 고정하기 위한 값이다.
                conn.execute(
                    """
                    INSERT OR REPLACE INTO session_docs
                        (session_id, doc_id, rank, score, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, doc_id, rank, None, now),
                )

    def update_state(
        self,
        session_id: str,
        last_question: Optional[str] = None,
        last_answer: Optional[str] = None,
        last_question_type: Optional[str] = None,
    ) -> None:
        """
        세션의 마지막 질문/답변/유형을 업데이트한다.
        """
        # 업데이트 시각을 기록한다.
        now = time.time()
        with self._connect() as conn:
            # 세션이 없으면 INSERT, 있으면 UPDATE로 최신 상태를 반영한다.
            conn.execute(
                """
                INSERT INTO session_state
                    (session_id, last_question, last_answer, last_question_type, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_question = excluded.last_question,
                    last_answer = excluded.last_answer,
                    last_question_type = excluded.last_question_type,
                    updated_at = excluded.updated_at
                """,
                (session_id, last_question, last_answer, last_question_type, now),
            )

    def get_last_question(self, session_id: str) -> Optional[str]:
        """
        세션의 마지막 질문을 반환한다.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_question FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def get_last_answer(self, session_id: str) -> Optional[str]:
        """
        세션의 마지막 답변을 반환한다.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_answer FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def get_last_question_type(self, session_id: str) -> Optional[str]:
        """
        세션의 마지막 질문 유형을 반환한다.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_question_type FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def get_last_turn(self, session_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        세션의 마지막 질문/답변을 함께 반환한다.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_question, last_answer FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None, None
        return row[0], row[1]

    def has_session(self, session_id: str) -> bool:
        """
        세션 상태가 존재하는지 확인한다.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM session_state WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def clear_session_docs(self, session_id: str) -> None:
        """
        특정 세션의 문서 ID 기록을 삭제한다.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM session_docs WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        """
        모든 세션 데이터를 삭제한다.
        """
        with self._connect() as conn:
            # 모든 문서 기록 삭제
            conn.execute("DELETE FROM session_docs")
            # 모든 세션 상태 삭제
            conn.execute("DELETE FROM session_state")
