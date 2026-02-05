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
QUESTION_TYPE_COMPARE = "compare"
QUESTION_TYPE_FOLLOWUP = "followup"

# 질문 유형 판별에 쓰는 키워드(휴리스틱) 목록이다.
# Heuristic keywords for question classification (LLM 분류 폴백)
COMPARE_KEYWORDS = ["비교", "차이", "서로", "vs", "대비"]
FOLLOWUP_KEYWORDS = ["그럼", "그렇다면", "또", "추가로", "더", "이어서", "방금", "앞서"]
MULTI_KEYWORDS = ["여러", "모든", "각각", "기관"]


# Query analysis result
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
        # 질문 타입과 메타데이터 필터를 먼저 추정한다.
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

        # 메타데이터 필터가 있으면 후보 수를 줄이고 유사도 검색으로 단순화한다.
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
        # LLM이 주입되면 우선 분류를 시도하고, 실패 시 휴리스틱으로 폴백한다.
        if self.llm is not None:
            label = classify_query_type(self.llm, question)
            if label:
                return label
        # 비교/후속/다문서 요청 여부를 휴리스틱으로 분류한다.
        if any(keyword in question for keyword in COMPARE_KEYWORDS):
            return QUESTION_TYPE_COMPARE
        if any(keyword in question for keyword in FOLLOWUP_KEYWORDS):
            return QUESTION_TYPE_FOLLOWUP
        # 짧은 질문 + 직전 대화가 있으면 후속 질문으로 본다.
        if state and state.last_user_message() and len(question) < 30:
            return QUESTION_TYPE_FOLLOWUP
        if any(keyword in question for keyword in MULTI_KEYWORDS):
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

        # 발주 기관/프로젝트명 등 메타데이터 필터를 정규식으로 추출한다.
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


# Retrieval plan builder
class RetrievalOrchestrator:
    """
    RetrievalOrchestrator는 QueryAnalysis 결과를 RetrievalPlan으로 변환

    Args:
        config: 설정 객체
    """

    def __init__(self, config: RAGConfig, llm: Optional[LLM] = None) -> None:
        self.config = config
        # QueryAnalysisAgent가 질문 분류/필터 추출을 담당한다.
        self.analyzer = QueryAnalysisAgent(config, llm=llm)

    def plan(self, question: str, state: Optional[ConversationState] = None) -> RetrievalPlan:
        """
        plan은 질문 분석 결과를 바탕으로 RetrievalPlan을 만든다.

        Args:
            question: 사용자 질문
            state: 대화 상태

        Returns:
            RetrievalPlan: 검색 계획
        """
        # 분석 결과를 RetrievalPlan으로 변환해 하위 검색 로직에 전달한다.
        analysis = self.analyzer.analyze(question, state)
        return RetrievalPlan(
            query=question,
            top_k=analysis.top_k,
            strategy=analysis.strategy,
            metadata_filter=analysis.metadata_filter,
            needs_multi_doc=analysis.needs_multi_doc,
            notes=analysis.notes,
        )


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


# Reranker implementations
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

    def __init__(self, store: FaissVectorStore, config: RAGConfig, reranker: Optional[Reranker] = None) -> None:
        self.store = store
        self.config = config
        self.reranker = reranker
        self._bm25: Optional[BM25Retriever] = None

    def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        """
        retrieve는 검색 계획에 따라 검색 수행

        Args:
            plan: 검색 계획

        Returns:
            RetrievalResult: 검색 결과
        """
        # 단일 문서는 similarity, 다문서는 RRF(similarity+mmr+bm25)로 실행한다.
        if plan.needs_multi_doc:
            chunks = self._retrieve_with_rrf(plan)
            scores = None
        else:
            chunks, scores = self.store.similarity_search(
                query=plan.query,
                top_k=plan.top_k,
                metadata_filter=plan.metadata_filter,
            )

        # 리랭커가 있으면 검색 결과를 한 번 더 정렬한다.
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
        # RRF는 서로 다른 랭킹(유사도/MMR)을 합쳐 안정성을 높인다.
        sim_chunks, _ = self.store.similarity_search(
            query=plan.query,
            top_k=max(plan.top_k, self.config.mmr_candidate_pool),
            metadata_filter=plan.metadata_filter,
            fetch_k=self.config.mmr_candidate_pool,
        )

        mmr_chunks = self._retrieve_with_mmr(plan)

        bm25_chunks: List[Chunk] = []
        bm25 = self._get_bm25(top_k=self.config.bm25_top_k)
        if bm25 is not None:
            docs = bm25.get_relevant_documents(plan.query)
            bm25_chunks = [_lc_doc_to_chunk(doc) for doc in docs]

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
            metadata_filter=plan.metadata_filter,
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
