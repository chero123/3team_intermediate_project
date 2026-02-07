from __future__ import annotations

"""
OpenAI 전용 검색/분석 흐름 모듈

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

from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument

from .config import RAGConfig
from .indexing import FaissVectorStore
from .openai_llm import OpenAILLM, classify_query_type
from .types import Chunk, ConversationState, RetrievalPlan, RetrievalResult

QUESTION_TYPE_SINGLE = "single"
QUESTION_TYPE_MULTI = "multi"
QUESTION_TYPE_FOLLOWUP = "followup"

# Heuristic keywords for question classification (LLM 분류 폴백)
FOLLOWUP_KEYWORDS = ["그럼", "그렇다면", "또", "추가로", "더", "이어서", "방금", "앞서"]
MULTI_KEYWORDS = ["여러", "모든", "각각", "기관", "비교", "차이", "서로", "vs", "대비"]


# Query analysis result
@dataclass
class QueryAnalysis:
    """
    QueryAnalysis는 질문 분석 결과를 보관한다.

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


# Query analysis agent (LLM 분류 + 휴리스틱 폴백)
class QueryAnalysisAgent:
    """
    OpenAI 기반 질문 분류 + 휴리스틱 폴백을 수행한다.

    Args:
        config: 설정 객체
        llm: OpenAI LLM
    """

    def __init__(self, config: RAGConfig, llm: Optional[OpenAILLM] = None) -> None:
        self.config = config
        self.llm = llm

    def analyze(self, question: str, state: Optional[ConversationState] = None) -> QueryAnalysis:
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
        if self.llm is not None:
            label = classify_query_type(self.llm, question)
            if label:
                return label
        if any(keyword in question for keyword in FOLLOWUP_KEYWORDS):
            return QUESTION_TYPE_FOLLOWUP
        if state and state.last_user_message() and len(question) < 30:
            return QUESTION_TYPE_FOLLOWUP
        if any(keyword in question for keyword in MULTI_KEYWORDS):
            return QUESTION_TYPE_MULTI
        return QUESTION_TYPE_SINGLE



# Retrieval utilities
def reciprocal_rank_fusion(ranked_lists: List[tuple[List[Chunk], float]], k: int = 60) -> List[Chunk]:
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked, weight in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            chunk_map[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[chunk_id] for chunk_id, _ in fused]


def _lc_doc_to_chunk(doc: LCDocument) -> Chunk:
    chunk_id = doc.metadata.get("chunk_id", "")
    metadata = dict(doc.metadata)
    metadata.pop("chunk_id", None)
    return Chunk(id=chunk_id, text=doc.page_content, metadata=metadata)



def _load_bm25_from_file(path: str, top_k: int) -> Optional[BM25Retriever]:
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
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, model_path: str, device: str, top_n: int) -> None:
        self.model = CrossEncoder(model_path, device=device)
        self.top_n = top_n

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[: self.top_n]]


# Retriever (strategy dispatcher)
class Retriever:
    """
    Retriever는 검색 전략에 따라 검색을 수행한다.

    Args:
        store: 벡터 스토어
        config: 설정 객체
        reranker: 리랭커
    """

    def __init__(self, store: FaissVectorStore, config: RAGConfig, reranker: Optional[Reranker] = None) -> None:
        self.store = store
        self.config = config
        self.reranker = reranker
        self._bm25: Optional[BM25Retriever] = None

    def retrieve(self, plan: RetrievalPlan) -> RetrievalResult:
        # 단일 문서는 similarity, 다문서는 RRF(similarity+mmr+bm25)로 단순화한다.
        if plan.needs_multi_doc:
            chunks = self._retrieve_with_rrf(plan)
            scores = None
        else:
            chunks, scores = self.store.similarity_search(
                query=plan.query,
                top_k=plan.top_k,
            )
            # 세션 메모리에서 온 문서 ID 필터가 있으면 결과를 제한한다.
            chunks = self._apply_doc_id_filter(chunks, plan.doc_id_filter)

        if self.reranker is not None:
            chunks = self.reranker.rerank(plan.query, chunks)[: plan.top_k]

        return RetrievalResult(chunks=chunks, scores=scores, plan=plan)

    def _retrieve_with_rrf(self, plan: RetrievalPlan) -> List[Chunk]:
        sim_chunks, _ = self.store.similarity_search(
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
        return self.store.mmr_search(
            query=plan.query,
            top_k=plan.top_k,
            fetch_k=self.config.mmr_candidate_pool,
            lambda_mult=self.config.mmr_lambda,
        )

    def _get_bm25(self, top_k: int) -> Optional[BM25Retriever]:
        if self._bm25 is None:
            self._bm25 = _load_bm25_from_file(self.config.bm25_index_path, top_k)
        if self._bm25 is not None:
            self._bm25.k = top_k
        return self._bm25

    def _apply_doc_id_filter(self, chunks: List[Chunk], doc_id_filter: Optional[List[str]]) -> List[Chunk]:
        # 세션 메모리에서 전달된 문서 ID만 남겨 후속 질문 범위를 좁힌다.
        if not doc_id_filter:
            return chunks
        allow = set(doc_id_filter)
        filtered = [chunk for chunk in chunks if chunk.metadata.get("doc_id") in allow]
        return filtered or chunks
