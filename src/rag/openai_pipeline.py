from __future__ import annotations

"""
OpenAI RAG 파이프라인 모듈

흐름:
- analyze_query → retrieve → generate → rewrite
"""

import uuid
from typing import Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .embeddings import create_embeddings
from .indexing import FaissVectorStore
from .memory_store import SessionMemoryStore
from .openai_llm import OpenAILLM, generate_answer, rewrite_answer, rewrite_query
from .openai_retrieval import CrossEncoderReranker, QueryAnalysisAgent, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    """
    RAGState는 그래프 노드 간에 전달되는 상태 구조를 정의한다.

    각 필드는 단계별 결과를 누적해서 파이프라인 전체 흐름을 공유한다.
    """

    # 사용자 질문 원문
    question: str
    # 세션 식별자
    session_id: str
    # 질문 분석 결과로 생성된 검색 계획
    plan: RetrievalPlan
    # 벡터 검색 결과(청크 + 점수/플랜)
    retrieval: RetrievalResult
    # 1차 생성 답변
    answer: str
    # 리라이트된 최종 답변
    rewritten: str
    # 라우팅 키(single/multi/followup)
    route_key: str


class OpenAIRAGPipeline:
    """
    OpenAIRAGPipeline은 OpenAI LLM을 사용하는 RAG 파이프라인이다.

    Args:
        config: 설정 객체
        llm: OpenAI LLM 인스턴스
        reranker: 리랭커 인스턴스
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[OpenAILLM] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        """
        OpenAI 기반 RAG 파이프라인을 초기화한다.

        Args:
            config: 설정 객체 (None이면 기본값 사용)
            llm: 본문 생성용 OpenAI LLM
            reranker: 리랭커 인스턴스
        """
        # 설정 객체 준비
        self.config = config or RAGConfig()

        # 임베딩 모델 준비
        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )

        # FAISS 벡터 스토어 로드
        self.store = FaissVectorStore(self.embeddings)
        self.store.load(self.config.index_dir)

        # 본문 생성은 큰 모델 사용
        self.llm = llm or OpenAILLM(model=self.config.openai_model)
        # 분류/리라이트는 작은 모델 사용
        self.small_llm = OpenAILLM(model="gpt-4o-mini")

        # 리랭커 준비 (검색 품질 보정)
        self.reranker = reranker or CrossEncoderReranker(
            model_path=self.config.rerank_model_path,
            device=self.config.device,
            top_n=self.config.max_top_k,
        )

        # 검색기 및 오케스트레이터 구성
        self.retriever = Retriever(self.store, self.config, self.reranker)
        self.analyzer = QueryAnalysisAgent(self.config, llm=self.small_llm)
        # 대화 히스토리 저장소
        self.state = ConversationState()
        # 세션별 문서 기억용 SQLite 저장소
        self.memory = SessionMemoryStore(
            self.config.memory_db_path,
            clear_on_start=self.config.memory_clear_on_start,
        )
        # 기본 세션 ID
        self._session_id = uuid.uuid4().hex
        # LangGraph 파이프라인 빌드
        self.graph = self._build_graph()

    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        """
        질문을 받아 RAG 파이프라인으로 처리하고 최종 답변을 반환한다.

        Args:
            question: 사용자 질문

        Returns:
            str: 리라이트된 최종 답변
        """
        sid = session_id or self._session_id
        # 그래프 실행 → 단계별 결과를 합친 상태 딕셔너리 반환
        result = self.graph.invoke({"question": question, "session_id": sid})
        # 리라이트 결과가 있으면 우선 사용
        answer = result.get("rewritten") or result.get("answer", "")
        # 대화 히스토리 업데이트
        self.state.append("user", question)
        self.state.append("assistant", answer)
        plan = result.get("plan")
        if plan is not None:
            self.memory.update_state(
                sid,
                last_question=question,
                last_answer=answer,
                last_question_type=plan.question_type,
            )
        return answer

    def _build_graph(self) -> StateGraph:
        """
        LangGraph 기반 파이프라인 그래프를 구성한다.

        Returns:
            StateGraph: 컴파일된 실행 그래프
        """
        # 상태 타입을 명시해 그래프를 초기화한다.
        builder = StateGraph(RAGState)
        # 각 노드를 등록한다.
        builder.add_node("analyze_query", self._node_analyze_query)
        builder.add_node("route_by_plan", self._route_by_plan)
        builder.add_node("retrieve_single", self._node_retrieve)
        builder.add_node("retrieve_multi", self._node_retrieve)
        builder.add_node("retrieve_followup", self._node_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("rewrite", self._node_rewrite)

        # 실행 순서를 연결한다.
        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "route_by_plan")
        builder.add_conditional_edges(
            "route_by_plan",
            lambda s: s["route_key"],
            {
                "single": "retrieve_single",
                "multi": "retrieve_multi",
                "followup": "retrieve_followup",
            },
        )
        builder.add_edge("retrieve_single", "generate")
        builder.add_edge("retrieve_multi", "generate")
        builder.add_edge("retrieve_followup", "generate")
        builder.add_edge("generate", "rewrite")
        builder.add_edge("rewrite", END)
        return builder.compile()

    def _route_by_plan(self, state: RAGState) -> dict:
        """
        질문 유형에 따라 검색 노드를 분기한다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"route_key": str}
        """
        question_type = state["plan"].question_type
        if question_type in {"multi", "followup"}:
            return {"route_key": question_type}
        return {"route_key": "single"}

    def _node_analyze_query(self, state: RAGState) -> dict:
        """
        질문을 분석해 검색 계획을 만든다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"plan": RetrievalPlan}
        """
        # 오케스트레이터가 질문/대화 상태를 보고 플랜을 생성한다.
        analysis = self.analyzer.analyze(state["question"], self.state)
        plan = RetrievalPlan(
            query=state["question"],
            top_k=analysis.top_k,
            strategy=analysis.strategy,
            metadata_filter=analysis.metadata_filter,
            question_type=analysis.question_type,
            needs_multi_doc=analysis.needs_multi_doc,
            notes=analysis.notes,
        )
        # 질문에 기관명/사업명/파일명이 명시되면 새 문맥으로 강제 리셋한다.
        if analysis.metadata_filter:
            self.memory.clear_session_docs(state["session_id"])
        last_question_type = self.memory.get_last_question_type(state["session_id"])
        # 세션 메모리 기반의 followup 판정(명시 키워드 + 짧은 질문 + 유사도) 혼합 규칙
        last_question = self.memory.get_last_question(state["session_id"])
        last_question_type = self.memory.get_last_question_type(state["session_id"])
        if last_question:
            q = state["question"]
            # 사용자가 명시적으로 문맥 리셋을 요청하면 필터를 비우고 새 질문으로 처리한다.
            if any(keyword in q for keyword in self.config.memory_reset_keywords):
                self.memory.clear_session_docs(state["session_id"])
            else:
                # 직전 질문이 multi이면 followup도 multi 전략을 강제한다.
                if last_question_type == "multi":
                    plan.question_type = "multi"
                    plan.needs_multi_doc = True
                    plan.strategy = self.config.rrf_strategy
                    plan.metadata_filter = {}
                    plan.notes = f"{plan.notes}; followup via session memory (force multi)"
                    doc_ids = self.memory.load_doc_ids(state["session_id"])
                    if doc_ids:
                        plan.doc_id_filter = doc_ids
                    last_q, last_a = self.memory.get_last_turn(state["session_id"])
                    rewritten = rewrite_query(
                        self.small_llm,
                        self.config,
                        state["question"],
                        previous_question=last_q,
                        previous_answer=last_a,
                    )
                    if rewritten:
                        plan.query = rewritten
                    return {"plan": plan}
                followup_hint = any(keyword in q for keyword in self.config.memory_followup_keywords)
                if len(q) < 30:
                    followup_hint = True
                # followup으로 확정되면 기존 문서 범위를 재사용한다.
                if plan.question_type != "multi":
                    plan.question_type = "followup"
                    plan.needs_multi_doc = False
                    plan.strategy = self.config.similarity_strategy
                else:
                    plan.needs_multi_doc = True
                    plan.strategy = self.config.rrf_strategy
                plan.metadata_filter = {}
                plan.notes = f"{plan.notes}; followup via session memory"
                # SQLite 세션 메모리에서 직전 검색 문서 ID를 불러와 후속 질문 범위를 제한한다.
                doc_ids = self.memory.load_doc_ids(state["session_id"])
                if doc_ids:
                    plan.doc_id_filter = doc_ids
                last_q, last_a = self.memory.get_last_turn(state["session_id"])
                rewritten = rewrite_query(
                    self.small_llm,
                    self.config,
                    state["question"],
                    previous_question=last_q,
                    previous_answer=last_a,
                )
                if rewritten:
                    plan.query = rewritten
        # 멀티/후속 질문은 검색용 쿼리로 재작성해 검색 품질을 높인다.
        if plan.question_type in {"multi", "followup"}:
            last_q, last_a = self.memory.get_last_turn(state["session_id"])
            rewritten = rewrite_query(
                self.small_llm,
                self.config,
                state["question"],
                previous_question=last_q,
                previous_answer=last_a,
            )
            if rewritten:
                plan.query = rewritten
        return {"plan": plan}

    def _node_retrieve(self, state: RAGState) -> dict:
        """
        검색 계획에 따라 문서 청크를 검색한다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"retrieval": RetrievalResult}
        """
        # 검색기에서 청크/점수 결과를 가져온다.
        retrieval = self.retriever.retrieve(state["plan"])
        doc_ids: list[str] = []
        for chunk in retrieval.chunks:
            doc_id = chunk.metadata.get("doc_id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        if doc_ids:
            # SQLite 세션 메모리에 현재 검색 문서 ID를 저장해 다음 턴에서 재사용한다.
            self.memory.save_doc_ids(state["session_id"], doc_ids)
        return {"retrieval": retrieval}

    def _node_generate(self, state: RAGState) -> dict:
        """
        검색된 컨텍스트로 1차 답변을 생성한다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"answer": str}
        """
        # 이전 턴은 참고용 문맥으로만 제공한다.
        last_question, last_answer = self.memory.get_last_turn(state["session_id"])
        previous_turn = ""
        if last_question and last_answer:
            previous_turn = f"질문: {last_question}\n답변: {last_answer}"
        previous_docs = self.memory.load_doc_ids(state["session_id"])
        # 본문 생성은 큰 모델로 수행한다.
        answer = generate_answer(
            self.llm,
            self.config,
            state["question"],
            state["retrieval"].chunks,
            previous_turn=previous_turn or None,
            previous_docs=previous_docs or None,
        )
        return {"answer": answer}

    def _node_rewrite(self, state: RAGState) -> dict:
        """
        생성된 답변을 스타일 규칙에 맞게 리라이트한다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"rewritten": str}
        """
        # 리라이트는 작은 모델로 수행한다.
        rewritten = rewrite_answer(self.small_llm, self.config, state["answer"])
        return {"rewritten": rewritten}
