from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .embeddings import create_embeddings
from .indexing import FaissVectorStore
from .llm import LLM, VLLMLLM, generate_answer
from .retrieval import CrossEncoderReranker, RetrievalOrchestrator, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    question: str
    plan: RetrievalPlan
    retrieval: RetrievalResult
    answer: str


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
        self.config = config or RAGConfig()

        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )

        self.store = FaissVectorStore(self.embeddings)
        self.store.load(self.config.index_dir)

        self.llm = llm or VLLMLLM(
            model_path=self.config.llm_model_path,
            device=self.config.device,
            cache_dir=self.config.tokenizer_cache_dir,
            quantization=self.config.llm_quantization,
        )

        self.reranker = reranker or CrossEncoderReranker(
            model_path=self.config.rerank_model_path,
            device=self.config.device,
            top_n=self.config.max_top_k,
        )

        self.retriever = Retriever(self.store, self.config, self.reranker)
        self.orchestrator = RetrievalOrchestrator(self.config)
        self.state = ConversationState()
        self.graph = self._build_graph()

    def ask(self, question: str) -> str:
        """
        ask는 질문을 받아 답변을 생성

        Args:
            question: 사용자 질문

        Returns:
            str: 생성된 답변
        """
        result = self.graph.invoke({"question": question})
        answer = result["answer"]
        self.state.append("user", question)
        self.state.append("assistant", answer)
        return answer

    def _build_graph(self) -> StateGraph:
        """
        _build_graph는 LangGraph 실행 그래프를 만든다.

        Returns:
            StateGraph: 컴파일된 그래프
        """
        builder = StateGraph(RAGState)
        builder.add_node("analyze_query", self._node_analyze_query)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("generate", self._node_generate)

        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def _node_analyze_query(self, state: RAGState) -> dict:
        """
        _node_analyze_query는 질문 분석 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        plan = self.orchestrator.plan(state["question"], self.state)
        return {"plan": plan}

    def _node_retrieve(self, state: RAGState) -> dict:
        """
        _node_retrieve는 검색 실행 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        retrieval = self.retriever.retrieve(state["plan"])
        return {"retrieval": retrieval}

    def _node_generate(self, state: RAGState) -> dict:
        """
        _node_generate는 답변 생성 노드

        Args:
            state: LangGraph 상태

        Returns:
            dict: 업데이트할 상태 조각
        """
        answer = generate_answer(
            self.llm,
            self.config,
            state["question"],
            state["retrieval"].chunks,
        )
        return {"answer": answer}
