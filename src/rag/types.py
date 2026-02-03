from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# # 메타데이터의 공통 타입을 정의
Metadata = Dict[str, Any]


@dataclass
class Document:
    """
    Document는 원본 문서의 텍스트와 메타데이터를 함께 보관한다.

    Args:
        id: 문서 식별자
        text: 문서 전체 텍스트
        metadata: 문서 메타데이터
    """

    # 문서 식별자 저장
    id: str
    # 문서 텍스트 저장.
    text: str
    # 문서 메타데이터 저장
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Chunk는 청킹된 텍스트와 메타데이터 보관

    Args:
        id: 청크 식별자
        text: 청크 텍스트다
        metadata: 청크 메타데이터
    """

    # 청크 식별자 저장
    id: str
    # 청크 텍스트 저장
    text: str
    # 청크 메타데이터 저장
    metadata: Metadata = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    """
    RetrievalPlan은 검색 전략과 파라미터를 보관

    Args:
        query: 사용자 질문
        top_k: 검색 결과 수
        strategy: 검색 전략 키워드
        metadata_filter: 메타데이터 필터
        needs_multi_doc: 다문서 처리 여부
        notes: 분석 메모
    """

    # 질문 저장
    query: str
    # 검색 결과 수 저장
    top_k: int
    # 검색 전략 저장
    strategy: str
    # 메타데이터 필터 저장
    metadata_filter: Metadata
    # 다문서 처리 여부 저장
    needs_multi_doc: bool
    # 분석 메모 저장
    notes: str = ""


@dataclass
class RetrievalResult:
    """
    RetrievalResult는 검색 결과 청크와 점수를 보관

    Args:
        chunks: 검색된 청크 리스트
        scores: 유사도 점수 리스트
        plan: 사용된 검색 계획
    """

    # 청크 리스트 저장
    chunks: List[Chunk]
    # 점수 리스트 저장
    scores: Optional[List[float]] = None
    # 사용된 플랜 저장
    plan: Optional[RetrievalPlan] = None


@dataclass
class ConversationTurn:
    """
    ConversationTurn은 대화 한 턴의 정보를 보관

    Args:
        role: 발화자 역할
        content: 발화 내용
    """

    # 역할 저장
    role: str
    # 발화 내용 저장
    content: str


@dataclass
class ConversationState:
    """
    ConversationState는 대화 히스토리를 보관

    Args:
        history: 대화 턴 리스트
    """

    # 히스토리를 저장
    history: List[ConversationTurn] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        """
        append는 히스토리에 새 턴을 추가

        Args:
            role: 발화자 역할
            content: 발화 내용
        """
        # 턴을 추가
        self.history.append(ConversationTurn(role=role, content=content))

    def last_user_message(self) -> Optional[str]:
        """
        last_user_message는 마지막 사용자 메시지를 반환

        Returns:
            Optional[str]: 최근 사용자 메시지
        """
        # 뒤에서부터 탐색
        for turn in reversed(self.history):
            # 사용자 턴인지 확인
            if turn.role == "user":
                # 내용을 반환
                return turn.content
        # 없으면 None 반환
        return None
