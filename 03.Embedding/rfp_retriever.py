# ============================================
# rfp_retriever.py - LangGraph 기반 RAG Retriever
# ============================================
# 파이프라인: text_parsing.py → text_chunking.py → rfp_retriever.py
#
# text_chunking.py에서 구축한 ChromaDB 벡터DB를 로드하여
# 질문 분류(Router) → 검색(Retriever) → 답변 생성(Generator)을 수행합니다.
#
# [필수 패키지]
# pip install langgraph langchain-core langchain-openai chromadb openai python-dotenv
# ============================================

import os
import re
import json
import numpy as np
from typing import Optional, Annotated
from typing_extensions import TypedDict

import dotenv

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from embeddings import OpenAIEmbedding
from vector_store import ChromaVectorDB


# ============================================
# State 정의
# ============================================


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    mode: str  # "single" | "multi" | "compare"
    target_doc_id: Optional[str]
    target_doc_ids: list  # compare 모드: 비교할 doc_id 목록
    context: list  # [{text, metadata, score}, ...]
    citation_list: list  # ["문서명 (p.XX)", ...]
    answer: str


# ============================================
# RAG 그래프 빌더
# ============================================


class RFPRetriever:
    """LangGraph 기반 RFP 검색 + 답변 생성 시스템"""

    def __init__(
        self,
        persist_dir: str = "data/vector_db/rfp_integrated",
        collection_name: str = "rfp_hybrid",
        llm_model: str = "gpt-5",
    ):
        # 벡터DB 로드 (text_chunking.py에서 구축한 것)
        self.vector_db = ChromaVectorDB(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        print(f"[로드] 벡터DB: {persist_dir} ({self.vector_db.count}개 벡터)")

        # 임베딩 모델 (text_chunking.py와 동일)
        self.embedding = OpenAIEmbedding()

        # LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        # 문서 목록 캐시
        self._doc_list = None

        # 대화 히스토리 (질문/답변/출처 누적)
        self.history = []

        # 그래프 컴파일
        self.graph = self._build_graph()

    def get_doc_list(self) -> list[str]:
        """벡터DB에 저장된 고유 doc_id 목록 조회"""
        if self._doc_list is not None:
            return self._doc_list

        # ChromaDB에서 모든 메타데이터의 doc_id를 가져옴
        all_metadata = self.vector_db.collection.get(include=["metadatas"], limit=10000)
        doc_ids = set()
        for meta in all_metadata["metadatas"]:
            if meta and "doc_id" in meta:
                doc_ids.add(meta["doc_id"])

        self._doc_list = sorted(doc_ids)
        return self._doc_list

    # ============================================
    # Node 1: 질문 분류기 (Router)
    # ============================================

    def _fuzzy_match_doc_ids(self, query: str) -> list[str]:
        """질문에서 기관명/사업명 키워드를 추출하여 doc_id를 fuzzy 매칭"""
        doc_list = self.get_doc_list()
        matched = []

        for doc_id in doc_list:
            # doc_id에서 기관명 추출 (첫 번째 '_' 앞부분)
            parts = doc_id.split("_", 1)
            org_name = parts[0].strip()

            # 기관명이 질문에 포함되어 있으면 매칭
            if org_name and org_name in query:
                matched.append(doc_id)
                continue

            # 사업명(뒷부분)의 주요 키워드가 질문에 포함되어 있는지 확인
            if len(parts) > 1:
                project_name = parts[1].strip()
                # 사업명에서 의미 있는 키워드 추출 (2글자 이상 한글 단어)
                keywords = re.findall(r"[가-힣]{2,}", project_name)
                # 3글자 이상 키워드 중 2개 이상 매칭되면 해당 문서로 판단
                significant_keywords = [kw for kw in keywords if len(kw) >= 3]
                if significant_keywords:
                    match_count = sum(1 for kw in significant_keywords if kw in query)
                    if match_count >= 2:
                        matched.append(doc_id)

        return matched

    def router_node(self, state: GraphState) -> dict:
        """질문이 특정 문서에 대한 것인지, 비교인지, 전체인지 판별"""
        query = state["query"]

        # 이미 mode가 지정된 경우 (외부에서 doc_id를 직접 지정한 경우)
        if state.get("target_doc_id"):
            return {"mode": "single", "target_doc_id": state["target_doc_id"]}
        if state.get("target_doc_ids"):
            return {"mode": "compare", "target_doc_ids": state["target_doc_ids"]}

        # 1단계: fuzzy matching으로 먼저 후보 doc_id 추출
        fuzzy_matched = self._fuzzy_match_doc_ids(query)
        print(f"[Router] fuzzy 매칭 결과: {fuzzy_matched}")

        doc_list = self.get_doc_list()
        doc_list_str = "\n".join(f"- {doc_id}" for doc_id in doc_list[:50])

        # 2단계: LLM 라우팅 (개선된 프롬프트)
        router_prompt = f"""당신은 RFP(제안요청서) 질문 분류기입니다.
사용자의 질문을 분석하여 3가지 모드 중 하나로 분류하세요.

[등록된 문서 목록]
{doc_list_str}

[판별 기준]
- 질문에 특정 사업명/기관명이 1개만 포함 → single (해당 doc_id 반환)
- 질문에 특정 사업명/기관명이 2개 이상 포함되고 비교/차이 등의 표현 → compare (해당 doc_id들 반환)
- "모든 문서", "전체" 등 불특정 다수 → multi
- 판별이 어려우면 → multi

[중요 매칭 규칙]
- 기관명만 언급되어도 해당 기관의 doc_id와 매칭하세요.
  예: "고려대학교" → "고려대학교_차세대 포털·학사 정보시스템 구축사업"
  예: "전북대학교" → "전북대학교_JST 공유대학(원) xAPI기반 LRS시스템 구축"
- 약칭이나 부분 명칭도 매칭하세요 (예: "벤처협회" → "(사)벤처기업협회_...")
- 한 기관에 여러 문서가 있으면 가장 관련 있는 것을 선택하세요.
- 비교 질문에서는 반드시 언급된 모든 기관의 doc_id를 포함하세요.

반드시 아래 JSON 형식으로만 답변하세요:
{{"mode": "single" 또는 "compare" 또는 "multi", "doc_id": "매칭된 doc_id 또는 null", "doc_ids": ["매칭된 doc_id 목록"] 또는 null}}"""

        response = self.llm.invoke(
            [
                SystemMessage(content=router_prompt),
                HumanMessage(content=query),
            ]
        )

        try:
            result = json.loads(response.content)
            mode = result.get("mode", "multi")
            doc_id = result.get("doc_id")
            doc_ids = result.get("doc_ids") or []
        except (json.JSONDecodeError, AttributeError):
            mode = "multi"
            doc_id = None
            doc_ids = []

        # 3단계: fuzzy 매칭 결과와 LLM 결과 병합
        # LLM이 놓친 doc_id를 fuzzy 매칭으로 보완
        if fuzzy_matched:
            if mode == "single" and doc_id:
                # single인데 fuzzy에서 2개 이상 매칭 → compare로 승격
                all_ids = list(dict.fromkeys([doc_id] + fuzzy_matched))
                if len(all_ids) >= 2:
                    mode = "compare"
                    doc_ids = all_ids
                    doc_id = None
            elif mode == "compare":
                # compare에서 LLM이 놓친 doc_id 보완
                all_ids = list(dict.fromkeys(doc_ids + fuzzy_matched))
                doc_ids = all_ids
            elif mode == "single" and not doc_id:
                # LLM이 매칭 실패했지만 fuzzy가 찾은 경우
                if len(fuzzy_matched) == 1:
                    doc_id = fuzzy_matched[0]
                else:
                    mode = "compare"
                    doc_ids = fuzzy_matched
            elif mode == "multi":
                # multi인데 fuzzy에서 특정 문서를 찾은 경우
                if len(fuzzy_matched) == 1:
                    mode = "single"
                    doc_id = fuzzy_matched[0]
                elif len(fuzzy_matched) >= 2:
                    mode = "compare"
                    doc_ids = fuzzy_matched

        print(f"[Router] mode={mode}, doc_id={doc_id}, doc_ids={doc_ids}")
        return {"mode": mode, "target_doc_id": doc_id, "target_doc_ids": doc_ids}

    # ============================================
    # Node 2: 검색기 (Retriever)
    # ============================================

    def _parse_query_results(self, results) -> tuple[list, list]:
        """ChromaDB 쿼리 결과를 context, citation_list로 변환"""
        context = []
        citation_list = []

        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                context.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": metadata,
                        "score": score,
                    }
                )

                source = metadata.get("source_display", "출처 미상")
                if source not in citation_list:
                    citation_list.append(source)

        return context, citation_list

    def retriever_node(self, state: GraphState) -> dict:
        """벡터DB에서 관련 청크를 검색"""
        query = state["query"]
        mode = state.get("mode", "multi")
        target_doc_id = state.get("target_doc_id")
        target_doc_ids = state.get("target_doc_ids") or []
        top_k = 5

        # 쿼리 임베딩
        query_embedding = self.embedding.embed_query(query)

        if mode == "single" and target_doc_id:
            # 단일 문서 모드: doc_id 필터링
            results = self.vector_db.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"doc_id": target_doc_id},
                include=["documents", "metadatas", "distances"],
            )
            context, citation_list = self._parse_query_results(results)

        elif mode == "compare" and target_doc_ids:
            # 비교 모드: 각 문서별로 따로 검색 후 합침
            context = []
            citation_list = []
            per_doc_k = max(3, top_k // len(target_doc_ids))

            for doc_id in target_doc_ids:
                results = self.vector_db.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=per_doc_k,
                    where={"doc_id": doc_id},
                    include=["documents", "metadatas", "distances"],
                )
                doc_context, doc_citations = self._parse_query_results(results)
                context.extend(doc_context)
                for c in doc_citations:
                    if c not in citation_list:
                        citation_list.append(c)

            # 점수 높은 순 정렬
            context.sort(key=lambda x: x["score"], reverse=True)

        else:
            # 전체 문서 모드
            results = self.vector_db.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            context, citation_list = self._parse_query_results(results)

        print(f"[Retriever] {len(context)}개 청크 검색 완료 (mode={mode})")
        return {"context": context, "citation_list": citation_list}

    # ============================================
    # Node 3: 답변 생성기 (Generator)
    # ============================================

    def generator_node(self, state: GraphState) -> dict:
        """검색된 컨텍스트를 바탕으로 답변 생성 (이전 대화 히스토리 포함)"""
        query = state["query"]
        context = state.get("context", [])
        citation_list = state.get("citation_list", [])

        if not context:
            return {
                "answer": "검색 결과가 없습니다. 질문을 다시 확인해 주세요.",
                "messages": [HumanMessage(content=query)],
            }

        # 컨텍스트 텍스트 조합
        context_text = ""
        for i, chunk in enumerate(context, 1):
            source = chunk["metadata"].get("source_display", "출처 미상")
            context_text += f"\n[참고자료 {i}] (출처: {source})\n{chunk['text']}\n"

        # 출처 목록
        citation_str = "\n".join(f"- {c}" for c in citation_list)

        # 이전 대화 히스토리 텍스트 조합
        history_text = ""
        if self.history:
            history_text = "\n[이전 대화 기록]\n"
            for h in self.history:
                mode_label = "단일문서" if h["mode"] == "single" else "다중문서"
                history_text += f"Q ({mode_label}): {h['query']}\n"
                history_text += f"A: {h['answer'][:300]}...\n\n"

        system_prompt = f"""당신은 공공기관 RFP(제안요청서) 분석 전문가입니다.
아래 검색된 참고자료를 바탕으로 사용자의 질문에 정확하게 답변하세요.

[규칙]
1. 반드시 참고자료에 있는 내용만을 근거로 답변하세요.
2. 표(Table) 형식 데이터는 마크다운 표로 유지하세요.
3. 답변 끝에 반드시 아래 형식으로 출처를 표기하세요:
   [출처: 문서명, p.XX]
4. 참고자료에 답이 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
5. 이전 대화 기록이 있다면 문맥을 이어서 답변하세요.

[사용 가능한 출처 목록]
{citation_str}
{history_text}
[검색된 참고자료]
{context_text}"""

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        )

        answer = response.content
        print(f"[Generator] 답변 생성 완료 ({len(answer)}자)")

        return {
            "answer": answer,
            "messages": [HumanMessage(content=query)],
        }

    # ============================================
    # 그래프 빌드
    # ============================================

    def _build_graph(self) -> StateGraph:
        """LangGraph 그래프 컴파일"""
        graph_builder = StateGraph(GraphState)

        # 노드 등록
        graph_builder.add_node("router", self.router_node)
        graph_builder.add_node("retriever", self.retriever_node)
        graph_builder.add_node("generator", self.generator_node)

        # 엣지 연결: START → router → retriever → generator → END
        graph_builder.add_edge(START, "router")
        graph_builder.add_edge("router", "retriever")
        graph_builder.add_edge("retriever", "generator")
        graph_builder.add_edge("generator", END)

        return graph_builder.compile()

    # ============================================
    # 편의 메서드
    # ============================================

    def ask(self, query: str, doc_id: str = None, doc_ids: list = None) -> dict:
        """
        질문에 대한 답변을 반환합니다.
        이전 대화 히스토리가 자동으로 포함됩니다.

        Args:
            query: 사용자 질문
            doc_id: 특정 문서 ID (None이면 자동 분류)
            doc_ids: 비교할 문서 ID 목록 (None이면 자동 분류)

        Returns:
            dict: {"answer": str, "citations": list, "context": list}
        """
        # 이전 대화 히스토리를 messages에 포함
        prev_messages = []
        for h in self.history:
            prev_messages.append(HumanMessage(content=h["query"]))
            prev_messages.append(SystemMessage(content=h["answer"]))

        # 모드 결정: doc_id > doc_ids > 자동 분류
        if doc_id:
            mode = "single"
        elif doc_ids:
            mode = "compare"
        else:
            mode = ""

        initial_state = {
            "messages": prev_messages,
            "query": query,
            "mode": mode,
            "target_doc_id": doc_id,
            "target_doc_ids": doc_ids or [],
            "context": [],
            "citation_list": [],
            "answer": "",
        }

        result = self.graph.invoke(initial_state)

        # 히스토리에 현재 대화 저장
        self.history.append(
            {
                "query": query,
                "mode": result.get("mode", ""),
                "answer": result["answer"],
                "citations": result["citation_list"],
            }
        )

        return {
            "answer": result["answer"],
            "citations": result["citation_list"],
            "context": result["context"],
        }

    def clear_history(self):
        """대화 히스토리를 초기화합니다."""
        self.history = []
        print("[히스토리] 대화 기록이 초기화되었습니다.")


# ============================================
# 편의 함수
# ============================================


def create_rag_graph(
    persist_dir: str = "data/vector_db/rfp_integrated",
    collection_name: str = "rfp_hybrid",
    llm_model: str = "gpt-5",
) -> RFPRetriever:
    """RAG 그래프 생성 편의 함수"""
    return RFPRetriever(
        persist_dir=persist_dir,
        collection_name=collection_name,
        llm_model=llm_model,
    )


# ============================================
# 메인 실행
# ============================================

if __name__ == "__main__":
    # 1. RAG 그래프 초기화 (text_chunking.py로 구축한 벡터DB 로드)
    retriever = create_rag_graph()

    # 2. 등록된 문서 목록 확인
    doc_list = retriever.get_doc_list()
    print(f"\n[등록 문서] 총 {len(doc_list)}개")
    for doc_id in doc_list[:5]:
        print(f"  - {doc_id}")
    if len(doc_list) > 5:
        print(f"  ... 외 {len(doc_list) - 5}개")

    # 3. 연속 질의 테스트 (single ↔ multi 번갈아 사용)
    test_queries = [
        {"query": "을지대학교와 사업의 목적이 뭐야?"},
        {"query": "을지대학교와 전북대학교 의 사업 차이를 알려줘"},
        {
            "query": "이 사업의 총 예산은 얼마인가요?",
            "doc_id": doc_list[0] if doc_list else None,
        },
        {"query": "전북대학교 2024년 사업에서 '고유번호 QUR-01' 가 뭐야? "},
        {"query": "지금까지 질문한 을지대학교와 전북대학교 내용을 비교 표로 정리해줘"},
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        doc_id = q.get("doc_id")
        mode_label = f"단일문서 ({doc_id})" if doc_id else "자동분류"
        print(f"[질문 {i}] ({mode_label}) {q['query']}")
        print("=" * 60)

        result = retriever.ask(q["query"], doc_id=doc_id)
        print(f"\n[답변]\n{result['answer']}")
        print(f"\n[출처] {result['citations']}")
        print(f"[누적 히스토리] {len(retriever.history)}건")

    # 히스토리 초기화가 필요하면:
    # retriever.clear_history()
