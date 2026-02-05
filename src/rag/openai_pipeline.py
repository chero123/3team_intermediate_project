from __future__ import annotations

"""
OpenAI RAG 파이프라인 모듈

흐름:
- analyze_query → retrieve → generate → rewrite
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .embeddings import create_embeddings
from .indexing import FaissVectorStore
from .openai_llm import OpenAILLM, generate_answer, rewrite_answer
from .openai_retrieval import CrossEncoderReranker, RetrievalOrchestrator, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    """
    RAGState는 그래프 노드 간에 전달되는 상태 구조를 정의한다.

    각 필드는 단계별 결과를 누적해서 파이프라인 전체 흐름을 공유한다.
    """

    # 사용자 질문 원문
    question: str
    # 질문 분석 결과로 생성된 검색 계획
    plan: RetrievalPlan
    # 벡터 검색 결과(청크 + 점수/플랜)
    retrieval: RetrievalResult
    # 1차 생성 답변
    answer: str
    # 리라이트된 최종 답변
    rewritten: str


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
        self.orchestrator = RetrievalOrchestrator(self.config, llm=self.small_llm)
        # 대화 히스토리 저장소
        self.state = ConversationState()
        # LangGraph 파이프라인 빌드
        self.graph = self._build_graph()

    def ask(self, question: str) -> str:
        """
        질문을 받아 RAG 파이프라인으로 처리하고 최종 답변을 반환한다.

        Args:
            question: 사용자 질문

        Returns:
            str: 리라이트된 최종 답변
        """
        # 그래프 실행 → 단계별 결과를 합친 상태 딕셔너리 반환
        result = self.graph.invoke({"question": question})
        # 리라이트 결과가 있으면 우선 사용
        answer = result.get("rewritten") or result.get("answer", "")
        # 대화 히스토리 업데이트
        self.state.append("user", question)
        self.state.append("assistant", answer)
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
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("rewrite", self._node_rewrite)

        # 실행 순서를 연결한다.
        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "rewrite")
        builder.add_edge("rewrite", END)
        return builder.compile()

    def _node_analyze_query(self, state: RAGState) -> dict:
        """
        질문을 분석해 검색 계획을 만든다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"plan": RetrievalPlan}
        """
        # 오케스트레이터가 질문/대화 상태를 보고 플랜을 생성한다.
        plan = self.orchestrator.plan(state["question"], self.state)
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
        return {"retrieval": retrieval}

    def _node_generate(self, state: RAGState) -> dict:
        """
        검색된 컨텍스트로 1차 답변을 생성한다.

        Args:
            state: 그래프 상태

        Returns:
            dict: {"answer": str}
        """
        # 본문 생성은 큰 모델로 수행한다.
        answer = generate_answer(
            self.llm,
            self.config,
            state["question"],
            state["retrieval"].chunks,
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
