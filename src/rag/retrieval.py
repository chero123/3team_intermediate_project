from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from .config import RAGConfig
from .indexing import FaissVectorStore
from .types import Chunk, ConversationState, RetrievalPlan, RetrievalResult

QUESTION_TYPE_SINGLE = "single_doc"
QUESTION_TYPE_MULTI = "multi_doc"
QUESTION_TYPE_COMPARE = "compare"
QUESTION_TYPE_FOLLOWUP = "follow_up"

COMPARE_KEYWORDS = ["비교", "차이", "서로", "vs", "대비"]
FOLLOWUP_KEYWORDS = ["그럼", "그렇다면", "또", "추가로", "더", "이어서", "방금", "앞서"]


@dataclass
class QueryAnalysis:
    """
    QueryAnalysis는 질문 분석 결과 보관

    Args:
        question_type: 질문 유형
        needs_multi_doc: 다문서 여부
        metadata_filter: 필터 조건
        top_k: 검색 결과 수
        strategy: 검색 전략
        notes: 분석 메모
    """

    question_type: str
    needs_multi_doc: bool
    metadata_filter: Dict[str, str]
    top_k: int
    strategy: str
    notes: str = ""


class QueryAnalysisAgent:
    """
    QueryAnalysisAgent는 질문을 분석하고 검색 전략을 결정

    Args:
        config: 설정 객체

    Returns:
        None
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    def analyze(self, question: str, state: Optional[ConversationState] = None) -> QueryAnalysis:
        """
        analyze는 질문을 분석해 QueryAnalysis를 만든다.

        Args:
            question: 사용자 질문
            state: 대화 상태

        Returns:
            QueryAnalysis: 분석 결과
        """
        question_type = self._classify_question(question, state)
        metadata_filter = self._extract_metadata_filters(question)

        if question_type == QUESTION_TYPE_COMPARE:
            top_k = min(self.config.max_top_k, 10)
            strategy = self.config.rrf_strategy
            needs_multi_doc = True
            notes = "comparison detected"
        elif question_type == QUESTION_TYPE_MULTI:
            top_k = min(self.config.max_top_k, 8)
            strategy = self.config.rrf_strategy
            needs_multi_doc = True
            notes = "multi-document request"
        elif question_type == QUESTION_TYPE_FOLLOWUP:
            top_k = max(self.config.min_top_k, 4)
            strategy = self.config.similarity_strategy
            needs_multi_doc = False
            notes = "follow-up request"
        else:
            top_k = max(self.config.min_top_k, 5)
            strategy = self.config.similarity_strategy
            needs_multi_doc = False
            notes = "single-document request"

        if metadata_filter:
            top_k = max(self.config.min_top_k, min(top_k, 6))
            strategy = self.config.similarity_strategy
            notes += "; metadata filter applied"

        return QueryAnalysis(
            question_type=question_type,
            needs_multi_doc=needs_multi_doc,
            metadata_filter=metadata_filter,
            top_k=top_k,
            strategy=strategy,
            notes=notes,
        )

    def _classify_question(self, question: str, state: Optional[ConversationState]) -> str:
        """
        _classify_question는 질문 유형을 분류

        Args:
            question: 사용자 질문
            state: 대화 상태

        Returns:
            str: 질문 유형
        """
        if any(keyword in question for keyword in COMPARE_KEYWORDS):
            return QUESTION_TYPE_COMPARE
        if any(keyword in question for keyword in FOLLOWUP_KEYWORDS):
            return QUESTION_TYPE_FOLLOWUP
        if state and state.last_user_message() and len(question) < 30:
            return QUESTION_TYPE_FOLLOWUP
        if any(token in question for token in ["여러", "모든", "각각", "기관"]):
            return QUESTION_TYPE_MULTI
        return QUESTION_TYPE_SINGLE

    def _extract_metadata_filters(self, question: str) -> Dict[str, str]:
        """
        _extract_metadata_filters는 질문에서 메타데이터 조건 추출

        Args:
            question: 사용자 질문

        Returns:
            Dict[str, str]: 필터 딕셔너리
        """
        filters: Dict[str, str] = {}

        issuer_match = re.search(
            r"([가-힣A-Za-z0-9\s]+?(공단|연구원|대학교|과학기술원|재단|청|부))",
            question,
        )
        if issuer_match:
            filters["issuer"] = issuer_match.group(1).strip()

        project_match = re.search(r"([가-힣A-Za-z0-9\s]+?)\s*(사업|프로젝트|시스템)", question)
        if project_match:
            filters["project_name"] = project_match.group(1).strip()

        return filters


class RetrievalOrchestrator:
    """
    RetrievalOrchestrator는 QueryAnalysis 결과를 RetrievalPlan으로 변환

    Args:
        config: 설정 객체

    Returns:
        None
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.analyzer = QueryAnalysisAgent(config)

    def plan(self, question: str, state: Optional[ConversationState] = None) -> RetrievalPlan:
        """
        plan은 질문 분석 결과를 바탕으로 RetrievalPlan을 만든다.

        Args:
            question: 사용자 질문
            state: 대화 상태

        Returns:
            RetrievalPlan: 검색 계획
        """
        analysis = self.analyzer.analyze(question, state)
        return RetrievalPlan(
            query=question,
            top_k=analysis.top_k,
            strategy=analysis.strategy,
            metadata_filter=analysis.metadata_filter,
            needs_multi_doc=analysis.needs_multi_doc,
            notes=analysis.notes,
        )


def reciprocal_rank_fusion(ranked_lists: List[List[Chunk]], k: int = 60) -> List[Chunk]:
    """
    reciprocal_rank_fusion은 RRF 결합 수행

    Args:
        ranked_lists: 랭킹 리스트 모음
        k: 보정 상수

    Returns:
        List[Chunk]: 결합된 랭킹
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            chunk_map[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[chunk_id] for chunk_id, _ in fused]


class Reranker:
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        rerank는 검색 결과 재정렬

        Args:
            query: 사용자 질문
            chunks: 청크 리스트

        Returns:
            List[Chunk]: 재정렬된 청크 리스트
        """
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """
    CrossEncoderReranker는 CrossEncoder로 재정렬

    Args:
        model_path: 리랭커 모델 경로
        device: 디바이스
        top_n: 상위 N개 유지

    Returns:
        None
    """

    def __init__(self, model_path: str, device: str, top_n: int) -> None:
        self.model = CrossEncoder(model_path, device=device)
        self.top_n = top_n

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        rerank는 CrossEncoder 점수 재정렬

        Args:
            query: 사용자 질문
            chunks: 청크 리스트

        Returns:
            List[Chunk]: 재정렬된 청크 리스트
        """
        if not chunks:
            return []
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[: self.top_n]]


class Retriever:
    """
    Retriever는 검색 전략에 따라 검색 수행

    Args:
        store: 벡터 스토어
        config: 설정 객체
        reranker: 리랭커

    Returns:
        None
    """

    def __init__(self, store: FaissVectorStore, config: RAGConfig, reranker: Optional[Reranker] = None) -> None:
        self.store = store
        self.config = config
        self.reranker = reranker

    def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        retrieve는 검색 계획에 따라 검색 수행

        Args:
            plan: 검색 계획

        Returns:
            RetrievalResult: 검색 결과
        """
        if plan.strategy == self.config.rrf_strategy:
            chunks = self._retrieve_with_rrf(plan)
            scores = None
        elif plan.strategy == self.config.mmr_strategy:
            chunks = self._retrieve_with_mmr(plan)
            scores = None
        else:
            chunks, scores = self.store.similarity_search(
                query=plan.query,
                top_k=plan.top_k,
                metadata_filter=plan.metadata_filter,
            )

        if self.reranker is not None:
            chunks = self.reranker.rerank(plan.query, chunks)[: plan.top_k]

        return RetrievalResult(chunks=chunks, scores=scores, plan=plan)

    def _retrieve_with_rrf(self, plan: RetrievalPlan) -> List[Chunk]:
        """
        _retrieve_with_rrf는 RRF 전략 수행

        Args:
            plan: 검색 계획

        Returns:
            List[Chunk]: 검색 결과
        """
        sim_chunks, _ = self.store.similarity_search(
            query=plan.query,
            top_k=max(plan.top_k, self.config.mmr_candidate_pool),
            metadata_filter=plan.metadata_filter,
            fetch_k=self.config.mmr_candidate_pool,
        )

        mmr_chunks = self._retrieve_with_mmr(plan)

        fused = reciprocal_rank_fusion(
            [sim_chunks, mmr_chunks],
            k=self.config.rrf_k,
        )
        return fused[: plan.top_k]

    def _retrieve_with_mmr(self, plan: RetrievalPlan) -> List[Chunk]:
        """
        _retrieve_with_mmr는 MMR 전략을 수행한다.

        Args:
            plan: 검색 계획

        Returns:
            List[Chunk]: 검색 결과
        """
        return self.store.mmr_search(
            query=plan.query,
            top_k=plan.top_k,
            fetch_k=self.config.mmr_candidate_pool,
            lambda_mult=self.config.mmr_lambda,
            metadata_filter=plan.metadata_filter,
        )
