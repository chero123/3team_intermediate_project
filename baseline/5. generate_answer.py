# generate_answer.py
import os
import glob
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from context_builder import build_context


# =========================
# 경로/설정
# =========================
PARSING_DIR = "data/parsing_data"   # *_parsed.txt
FAISS_DIR   = "data/faiss_index"    # FAISS 인덱스 폴더

TOP_K = 5

# 하이브리드 후보/폭
CANDIDATE_K = 30
VECTOR_K = 40
RRF_K = 60


# =========================
# BM25용 문서 로드
# =========================
def load_parsed_documents(parsing_dir: str):
    txt_files = glob.glob(os.path.join(parsing_dir, "*_parsed.txt"))
    docs = []
    for path in txt_files:
        filename = os.path.basename(path)
        doc_id = filename.replace("_parsed.txt", "")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": filename, "doc_id": doc_id}))
    return docs


# =========================
# RRF 결합
# =========================
def rrf_fusion(doc_lists, k=60, top_k=30):
    scores = defaultdict(float)
    doc_by_key = {}

    for docs in doc_lists:
        for rank, d in enumerate(docs, start=1):
            key = d.metadata.get("chunk_id") or f"{d.metadata.get('source')}::{d.page_content[:80]}"
            scores[key] += 1.0 / (k + rank)
            if key not in doc_by_key:
                doc_by_key[key] = d

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_by_key[key] for key, _ in ranked[:top_k]]


# =========================
# 리랭커
# =========================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_rerank(query: str, docs, top_k: int = 5):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


# =========================
# Retriever + Rerank
# =========================
def retrieve_top_docs(question: str):
    # OpenAI Embedding + FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    # BM25
    bm25_docs = load_parsed_documents(PARSING_DIR)
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = TOP_K  # bm25는 TOP_K만 뽑아도 되지만, 필요하면 늘려도 됨

    # 1) 후보 넓게: vector 쪽은 많이 뽑기(리콜↑)
    vdocs = vectorstore.similarity_search(question, k=VECTOR_K)
    bdocs = bm25.invoke(question)

    # 2) RRF로 후보 합치기 → 30개
    cands = rrf_fusion([vdocs, bdocs], k=RRF_K, top_k=CANDIDATE_K)

    # 3) 리랭크로 최종 5개
    top_docs = cross_encoder_rerank(question, cands, top_k=TOP_K)

    return top_docs


# =========================
# LLM 생성
# =========================
def generate_answer(question: str, context: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
너는 문서 기반 질의응답 시스템이다.
반드시 아래 [문서] 안의 내용만 근거로 답해라.
문서에 없는 내용이면 '문서에서 찾을 수 없습니다' 라고 말해라.
가능하면 답변 끝에 출처(source)를 bullet로 적어라.

[질문]
{question}

[문서]
{context}
""".strip()

    return llm.invoke(prompt).content


def main():
    questions = [
        "벤처확인종합관리시스템 추진방안 구축 인터뷰내용 알려줘",
        "학사정보시스템 고도화 사업의 예산과 주요 과업 내용을 요약해줘",
    ]

    for q in questions:
        # 1) 검색 + 리랭크
        docs = retrieve_top_docs(q)

        # 2) context 구성
        context = build_context(docs, max_chars=6000, per_doc_chars=1500)

        # 3) 답변 생성
        answer = generate_answer(q, context)

        print("\n" + "=" * 100)
        print(f"Q: {q}\n")
        print("A:")
        print(answer)
        print("\n" + "-" * 100)
        print("CONTEXT PREVIEW:")
        print(context[:1200])  # 디버깅용


if __name__ == "__main__":
    main()
