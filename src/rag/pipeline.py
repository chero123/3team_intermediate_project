from __future__ import annotations

"""
로컬(vLLM) RAG 파이프라인 모듈

흐름:
- analyze_query → retrieve → generate → rewrite
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .embeddings import create_embeddings
from .indexing import FaissVectorStore
from .llm import LLM, VLLMLLM, generate_answer, rewrite_answer
from .retrieval import CrossEncoderReranker, RetrievalOrchestrator, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    question: str
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
        # LangGraph로 단계 실행 흐름을 고정한다.
        self.graph = self._build_graph()

    def ask(self, question: str) -> str:
        """
        ask는 질문을 받아 답변을 생성

        Args:
            question: 사용자 질문

        Returns:
            str: 생성된 답변
        """
        # LangGraph는 상태 기반으로 분석→검색→생성 순서로 실행한다.
        result = self.graph.invoke({"question": question})
        answer = result.get("rewritten") or result["answer"]
        # 대화 히스토리를 갱신해 후속 질문의 문맥으로 활용한다.
        self.state.append("user", question)
        self.state.append("assistant", answer)
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
        return {"retrieval": retrieval}

    def _node_generate(self, state: RAGState) -> dict:
        """
        답변 생성 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        # 검색 결과 청크를 넣어 LLM 답변을 생성한다.
        answer = generate_answer(
            self.llm,
            self.config,
            state["question"],
            state["retrieval"].chunks,
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
