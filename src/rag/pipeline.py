from __future__ import annotations

"""
로컬(vLLM) RAG 파이프라인 모듈

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
from .llm import LLM, VLLMLLM, generate_answer, rewrite_answer
from .memory_store import SessionMemoryStore
from .retrieval import CrossEncoderReranker, RetrievalOrchestrator, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    question: str
    session_id: str
    plan: RetrievalPlan
    retrieval: RetrievalResult
    answer: str
    rewritten: str


class RAGPipeline:
    """
    RAGPipeline은 질의 처리를 담당

    Args:
        config: 설정 객체
        llm: LLM 인스턴스
        reranker: 리랭커 인스턴스
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm: Optional[LLM] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        # 설정이 없으면 기본 설정으로 파이프라인을 초기화
        self.config = config or RAGConfig()

        # 임베딩 모델을 생성
        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )

        # 이미 생성된 인덱스를 로드해 질의 응답에 사용
        self.store = FaissVectorStore(self.embeddings)
        self.store.load(self.config.index_dir)

        # LLM이 주입되지 않으면 vLLM 기반 구현으로 생성
        self.llm = llm or VLLMLLM(
            model_path=self.config.llm_model_path,
            device=self.config.device,
            cache_dir=self.config.tokenizer_cache_dir,
            quantization=self.config.llm_quantization,
        )

        # 리랭커는 검색 결과 재정렬을 통해 답변 정확도를 보강
        self.reranker = reranker or CrossEncoderReranker(
            model_path=self.config.rerank_model_path,
            device=self.config.device,
            top_n=self.config.max_top_k,
        )

        # Retriever는 벡터 검색과 리랭킹을 묶어 단일 검색 API를 제공
        self.retriever = Retriever(self.store, self.config, self.reranker)
        # Orchestrator는 질문에 맞는 검색 전략을 선택
        self.orchestrator = RetrievalOrchestrator(self.config, llm=self.llm)
        # 대화 히스토리는 계획 수립 시 참고 자료로 사용된다.
        self.state = ConversationState()
        # 세션별 검색 문서 기억을 위해 SQLite 저장소를 사용한다.
        self.memory = SessionMemoryStore(
            self.config.memory_db_path,
            clear_on_start=self.config.memory_clear_on_start,
        )
        # 세션 ID는 인스턴스별로 기본값을 유지한다.
        self._session_id = uuid.uuid4().hex
        # LangGraph로 단계 실행 흐름을 고정한다.
        self.graph = self._build_graph()

    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        """
        ask는 질문을 받아 답변을 생성

        Args:
            question: 사용자 질문
            session_id: 세션 식별자

        Returns:
            str: 생성된 답변
        """
        sid = session_id or self._session_id
        # LangGraph는 상태 기반으로 분석→검색→생성 순서로 실행한다.
        result = self.graph.invoke({"question": question, "session_id": sid})
        answer = result.get("rewritten") or result["answer"]
        # 대화 히스토리를 갱신해 후속 질문의 문맥으로 활용한다.
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
        LangGraph 실행 그래프 구축

        Returns:
            StateGraph: 컴파일된 그래프
        """
        # StateGraph는 각 노드를 상태 갱신 함수로 연결한다.
        builder = StateGraph(RAGState)
        # 각 노드는 상태 일부를 계산해 반환한다.
        builder.add_node("analyze_query", self._node_analyze_query)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("rewrite", self._node_rewrite)

        # 실행 순서를 START -> analyze -> retrieve -> generate -> rewrite -> END로 고정한다.
        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "rewrite")
        builder.add_edge("rewrite", END)
        return builder.compile()

    def _node_analyze_query(self, state: RAGState) -> dict:
        """
        질문 분석 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        # 질문과 대화 히스토리를 기반으로 검색 계획을 만든다.
        plan = self.orchestrator.plan(state["question"], self.state)
        # 세션 메모리 기반의 followup 판정(명시 키워드 + 짧은 질문 + 유사도) 혼합 규칙
        last_question = self.memory.get_last_question(state["session_id"])
        if last_question:
            q = state["question"]
            # 사용자가 명시적으로 문맥 리셋을 요청하면 필터를 비우고 새 질문으로 처리한다.
            if any(keyword in q for keyword in self.config.memory_reset_keywords):
                self.memory.clear_session_docs(state["session_id"])
            else:
                followup_hint = any(keyword in q for keyword in self.config.memory_followup_keywords)
                if len(q) < 30:
                    followup_hint = True
                if not followup_hint:
                    # 질문 유사도가 낮으면 문맥 전환으로 보고 필터를 해제한다.
                    q_vec = np.asarray(self.embeddings.embed_query(q), dtype=np.float32)
                    last_vec = np.asarray(self.embeddings.embed_query(last_question), dtype=np.float32)
                    denom = float(np.linalg.norm(q_vec) * np.linalg.norm(last_vec)) or 1.0
                    similarity = float(np.dot(q_vec, last_vec) / denom)
                    if similarity < self.config.memory_similarity_threshold:
                        self.memory.clear_session_docs(state["session_id"])
                        return {"plan": plan}
                # followup으로 확정되면 기존 문서 범위를 재사용한다.
                plan.question_type = "followup"
                plan.needs_multi_doc = False
                plan.strategy = self.config.similarity_strategy
                plan.metadata_filter = {}
                plan.notes = f"{plan.notes}; followup via session memory"
                # SQLite 세션 메모리에서 직전 검색 문서 ID를 불러와 후속 질문 범위를 제한한다.
                doc_ids = self.memory.load_doc_ids(state["session_id"])
                if doc_ids:
                    plan.doc_id_filter = doc_ids
        return {"plan": plan}

    def _node_retrieve(self, state: RAGState) -> dict:
        """
        검색 실행 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        # 검색 계획에 따라 실제 벡터 검색 + 재랭킹을 수행한다.
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
        답변 생성 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        # 이전 턴은 참고용 문맥으로만 제공한다.
        last_question, last_answer = self.memory.get_last_turn(state["session_id"])
        previous_turn = ""
        if last_question and last_answer:
            previous_turn = f"질문: {last_question}\n답변: {last_answer}"
        previous_docs = self.memory.load_doc_ids(state["session_id"])
        # 검색 결과 청크를 넣어 LLM 답변을 생성한다.
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
        rewrite 노드는 생성된 답변을 3문장 요약 형식으로 리라이팅한다.

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        rewritten = rewrite_answer(self.llm, self.config, state["answer"])
        return {"rewritten": rewritten}
