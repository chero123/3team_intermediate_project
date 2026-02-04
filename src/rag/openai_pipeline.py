from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import RAGConfig
from .embeddings import create_embeddings
from .indexing import FaissVectorStore
from .openai_llm import OpenAILLM, classify_query_type, generate_answer, rewrite_answer
from .retrieval import CrossEncoderReranker, RetrievalOrchestrator, Retriever, Reranker
from .types import ConversationState, RetrievalPlan, RetrievalResult


class RAGState(TypedDict):
    question: str
    plan: RetrievalPlan
    retrieval: RetrievalResult
    answer: str
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
        self.config = config or RAGConfig()

        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )

        self.store = FaissVectorStore(self.embeddings)
        self.store.load(self.config.index_dir)

        self.llm = llm or OpenAILLM(model=self.config.openai_model)

        self.reranker = reranker or CrossEncoderReranker(
            model_path=self.config.rerank_model_path,
            device=self.config.device,
            top_n=self.config.max_top_k,
        )

        self.retriever = Retriever(self.store, self.config, self.reranker)
        self.orchestrator = RetrievalOrchestrator(self.config, llm=self.llm)
        self.state = ConversationState()
        self.graph = self._build_graph()

    def ask(self, question: str) -> str:
        result = self.graph.invoke({"question": question})
        answer = result.get("rewritten") or result["answer"]
        self.state.append("user", question)
        self.state.append("assistant", answer)
        return answer

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(RAGState)
        builder.add_node("analyze_query", self._node_analyze_query)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("generate", self._node_generate)
        builder.add_node("rewrite", self._node_rewrite)

        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "rewrite")
        builder.add_edge("rewrite", END)
        return builder.compile()

    def _node_analyze_query(self, state: RAGState) -> dict:
        plan = self.orchestrator.plan(state["question"], self.state)
        return {"plan": plan}

    def _node_retrieve(self, state: RAGState) -> dict:
        retrieval = self.retriever.retrieve(state["plan"])
        return {"retrieval": retrieval}

    def _node_generate(self, state: RAGState) -> dict:
        answer = generate_answer(
            self.llm,
            self.config,
            state["question"],
            state["retrieval"].chunks,
        )
        return {"answer": answer}

    def _node_rewrite(self, state: RAGState) -> dict:
        rewritten = rewrite_answer(self.llm, state["answer"])
        return {"rewritten": rewritten}
