from __future__ import annotations

"""
RAG 검색/분석 흐름 모듈

섹션 구성:
- 상수/키워드
- QueryAnalysis + 분석 에이전트
- 검색 계획(Orchestrator)
- 검색/리랭킹 구현
"""

import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# CrossEncoder는 문장쌍 점수를 다시 계산해 검색 결과를 재정렬한다.
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from .config import RAGConfig
from .indexing import FaissVectorStore
from .llm import LLM, classify_query_type
from .types import Chunk, ConversationState, RetrievalPlan, RetrievalResult

QUESTION_TYPE_SINGLE = "single"
QUESTION_TYPE_MULTI = "multi"
QUESTION_TYPE_FOLLOWUP = "followup"

# 질문 유형 판별에 쓰는 키워드(휴리스틱) 목록이다.
FOLLOWUP_KEYWORDS = ["그럼", "그렇다면", "또", "추가로", "더", "이어서", "방금", "앞서"]
MULTI_KEYWORDS = ["여러", "모든", "각각", "기관", "비교", "차이", "서로", "vs", "대비"]


# Query analysis result
@dataclass
class QueryAnalysis:
    """
    QueryAnalysis는 질문 분석 결과 보관

    Args:
        question_type: 질문 유형
        needs_multi_doc: 다문서 여부
        top_k: 검색 결과 수
        strategy: 검색 전략
        notes: 분석 메모
    """

    question_type: str
    needs_multi_doc: bool
    top_k: int
    strategy: str
    notes: str = ""


# Retrieval utilities
def reciprocal_rank_fusion(ranked_lists: List[tuple[List[Chunk], float]], k: int = 60) -> List[Chunk]:
    """
    reciprocal_rank_fusion은 RRF 결합 수행

    Args:
        ranked_lists: 랭킹 리스트 모음
        k: 보정 상수

    Returns:
        List[Chunk]: 결합된 랭킹
    """
    # RRF는 여러 랭킹을 합쳐 안정적인 상위 결과를 만든다.
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked, weight in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            chunk_map[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)

    # 점수가 큰 순서대로 정렬해 최종 랭킹을 만든다.
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[chunk_id] for chunk_id, _ in fused]


def _lc_doc_to_chunk(doc: LCDocument) -> Chunk:
    """
    LangChain Document를 Chunk로 변환한다.

    Args:
        doc: LangChain 문서

    Returns:
        Chunk: 변환된 청크
    """
    chunk_id = doc.metadata.get("chunk_id", "")
    metadata = dict(doc.metadata)
    metadata.pop("chunk_id", None)
    return Chunk(id=chunk_id, text=doc.page_content, metadata=metadata)


def _load_bm25_from_file(path: str, top_k: int) -> Optional[BM25Retriever]:
    """
    저장된 BM25 인덱스를 로드한다.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        bm25.k = top_k
        return bm25
    except Exception:
        return None


# Reranker implementation
class CrossEncoderReranker:
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
        # CrossEncoder는 문장쌍 입력을 받아 relevance 점수를 출력한다.
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
        # (질문, 청크) 쌍을 만들어 재랭킹 점수를 계산한다.
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        # 점수 내림차순으로 정렬해 상위 top_n만 반환한다.
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[: self.top_n]]


# Query analysis agent (LLM 분류 + 휴리스틱 폴백)
class QueryAnalysisAgent:
    """
    QueryAnalysisAgent는 질문을 분석하고 검색 전략을 결정

    Args:
        config: 설정 객체

    Returns:
        None
    """

    def __init__(self, config: RAGConfig, llm: Optional[LLM] = None) -> None:
        self.config = config
        self.llm = llm

    def analyze(self, question: str, state: Optional[ConversationState] = None) -> QueryAnalysis:
        """
        analyze는 질문을 분석해 QueryAnalysis를 만든다.

        Args:
            question: 사용자 질문
            state: 대화 상태

        Returns:
            QueryAnalysis: 분석 결과
        """
        # 질문 타입을 먼저 추정한다.
        question_type = self._classify_question(question, state)

        if question_type == QUESTION_TYPE_MULTI:
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

        return QueryAnalysis(
            question_type=question_type,
            needs_multi_doc=needs_multi_doc,
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
        # LLM이 주입되면 우선 분류를 시도하고, 실패 시 휴리스틱으로 폴백한다.
        if self.llm is not None:
            label = classify_query_type(self.llm, question)
            if label:
                return label
        # 비교/후속/다문서 요청 여부를 휴리스틱으로 분류한다.
        if any(keyword in question for keyword in FOLLOWUP_KEYWORDS):
            return QUESTION_TYPE_FOLLOWUP
        # 짧은 질문 + 직전 대화가 있으면 후속 질문으로 본다.
        if state and state.last_user_message() and len(question) < 30:
            return QUESTION_TYPE_FOLLOWUP
        if any(keyword in question for keyword in MULTI_KEYWORDS):
            return QUESTION_TYPE_MULTI
        return QUESTION_TYPE_SINGLE


# Retriever (strategy dispatcher)
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

    def __init__(
        self,
        store: FaissVectorStore,
        config: RAGConfig,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.store = store
        self.config = config
        self.reranker = reranker
        self._bm25: Optional[BM25Retriever] = None

    def retrieve_single(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        단일 문서용 검색을 수행한다. (similarity 전용)

        Args:
            plan: 검색 계획

        Returns:
            RetrievalResult: 검색 결과
        """
        chunks = self.store.similarity_search(
            query=plan.query,
            top_k=plan.top_k,
        )
        chunks = self._apply_rerank(plan, chunks)
        return RetrievalResult(chunks=chunks, plan=plan)

    def retrieve_multi(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        다문서용 검색을 수행한다. (RRF)
        """
        chunks = self._retrieve_with_rrf(plan)
        chunks = self._apply_rerank(plan, chunks)
        return RetrievalResult(chunks=chunks, plan=plan)

    def retrieve_followup_single(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        후속 질문(단일 문서)의 검색을 수행한다.
        doc_id_filter는 파이프라인에서 주입된다.
        """
        chunks = self.store.similarity_search(
            query=plan.query,
            top_k=plan.top_k,
        )
        chunks = self._apply_doc_id_filter(chunks, plan.doc_id_filter)
        chunks = self._apply_rerank(plan, chunks)
        return RetrievalResult(chunks=chunks, plan=plan)

    def retrieve_followup_multi(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        후속 질문(다문서)의 검색을 수행한다. (RRF)
        doc_id_filter는 파이프라인에서 주입된다.
        """
        chunks = self._retrieve_with_rrf(plan)
        chunks = self._apply_rerank(plan, chunks)
        return RetrievalResult(chunks=chunks, plan=plan)

    def _retrieve_with_rrf(self, plan: RetrievalPlan) -> List[Chunk]:
        """
        _retrieve_with_rrf는 RRF 전략 수행

        Args:
            plan: 검색 계획

        Returns:
            List[Chunk]: 검색 결과
        """
        # RRF는 서로 다른 랭킹(유사도/MMR)을 합쳐 안정성을 높인다.
        sim_chunks = self.store.similarity_search(
            query=plan.query,
            top_k=max(plan.top_k, self.config.mmr_candidate_pool),
            fetch_k=self.config.mmr_candidate_pool,
        )

        mmr_chunks = self._retrieve_with_mmr(plan)

        bm25_chunks: List[Chunk] = []
        bm25 = self._get_bm25(top_k=self.config.bm25_top_k)
        if bm25 is not None:
            docs = bm25._get_relevant_documents(  # type: ignore[attr-defined]
                plan.query,
                run_manager=None,
            )
            bm25_chunks = [_lc_doc_to_chunk(doc) for doc in docs]

        # 세션 메모리에서 온 문서 ID 필터가 있으면 각 랭킹을 제한한다.
        sim_chunks = self._apply_doc_id_filter(sim_chunks, plan.doc_id_filter)
        mmr_chunks = self._apply_doc_id_filter(mmr_chunks, plan.doc_id_filter)
        bm25_chunks = self._apply_doc_id_filter(bm25_chunks, plan.doc_id_filter)

        ranked_lists: List[tuple[List[Chunk], float]] = [
            (sim_chunks, self.config.rrf_dense_weight),
            (mmr_chunks, self.config.rrf_mmr_weight),
            (bm25_chunks, self.config.rrf_bm25_weight),
        ]

        fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k)
        return fused[: plan.top_k]

    def _retrieve_with_mmr(self, plan: RetrievalPlan) -> List[Chunk]:
        """
        _retrieve_with_mmr는 MMR 전략을 수행한다.

        Args:
            plan: 검색 계획

        Returns:
            List[Chunk]: 검색 결과
        """
        # MMR은 다양성과 관련성을 함께 고려해 중복을 줄인다.
        return self.store.mmr_search(
            query=plan.query,
            top_k=plan.top_k,
            fetch_k=self.config.mmr_candidate_pool,
            lambda_mult=self.config.mmr_lambda,
        )

    def _get_bm25(self, top_k: int) -> Optional[BM25Retriever]:
        """
        BM25 리트리버를 필요 시 로딩한다.

        Args:
            top_k: BM25 반환 개수

        Returns:
            Optional[BM25Retriever]: BM25 리트리버
        """
        if self._bm25 is None:
            self._bm25 = _load_bm25_from_file(self.config.bm25_index_path, top_k)
        if self._bm25 is not None:
            self._bm25.k = top_k
        return self._bm25

    def _apply_doc_id_filter(self, chunks: List[Chunk], doc_id_filter: Optional[List[str]]) -> List[Chunk]:
        """
        doc_id_filter가 있으면 해당 문서 ID만 남긴다.
        """
        if not doc_id_filter:
            return chunks
        allow = set(doc_id_filter)
        filtered = [chunk for chunk in chunks if chunk.metadata.get("doc_id") in allow]
        return filtered or chunks

    def _apply_rerank(self, plan: RetrievalPlan, chunks: List[Chunk]) -> List[Chunk]:
        """
        리랭커가 있으면 검색 결과를 한 번 더 정렬한다.
        """
        if self.reranker is None:
            return chunks
        return self.reranker.rerank(plan.query, chunks)[: plan.top_k]
