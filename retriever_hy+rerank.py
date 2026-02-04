# retriever_hy+rerank.py
#하이브리드(BM25 + Vector + RRF)로 30개 선정 후 rerank로 5개 최종 결정

import os
import glob
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 리랭크(크로스 인코더)
from sentence_transformers import CrossEncoder


PARSING_DIR = "data/parsing_data"   # *_parsed.txt
FAISS_DIR   = "data/faiss_index"    # FAISS 인덱스 폴더

MODE = "hybrid_rrf_rerank"
# "mmr" / "bm25" / "hybrid_rrf" / "hybrid_rrf_rerank"

TOP_K = 5

# MMR
FETCH_K = 30
LAMBDA_MULT = 0.7

# RRF
RRF_K = 60

# ✅ 후보 개수 (하이브리드로 먼저 뽑을 후보)
CANDIDATE_K = 30

# ✅ 벡터 후보 폭 (하이브리드에서 vector쪽을 넓게 뽑아야 recall 올라감)
VECTOR_K = 40


def load_parsed_documents(parsing_dir: str):
    """BM25용: 파싱 텍스트 파일 전체를 Document로 로드"""
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


def rrf_fusion(doc_lists, k=60, top_k=5):
    """Reciprocal Rank Fusion: 순위 기반 결합"""
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


# ✅ 리랭커 준비 (한 번만)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_encoder_rerank(query: str, docs, top_k: int = 5):
    """Cross-Encoder로 후보 문서를 재정렬"""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


def main():
    # Vectorstore 로드
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    # MMR retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT},
    )

    # BM25 retriever
    bm25_docs = load_parsed_documents(PARSING_DIR)
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = TOP_K

    # 테스트 질문
    queries = [
        "벤처확인종합관리시스템 추진방안 구축 인터뷰내용 알려줘",
        "학사정보시스템 고도화 사업의 예산과 주요 과업 내용을 요약해줘",
    ]

    for q in queries:
        if MODE == "mmr":
            docs = mmr_retriever.invoke(q)

        elif MODE == "bm25":
            docs = bm25_retriever.invoke(q)

        elif MODE == "hybrid_rrf":
            # ✅ 하이브리드만: 후보 결합 결과에서 TOP_K 뽑기
            vdocs = vectorstore.similarity_search(q, k=VECTOR_K)
            bdocs = bm25_retriever.invoke(q)
            docs = rrf_fusion([vdocs, bdocs], k=RRF_K, top_k=TOP_K)

        elif MODE == "hybrid_rrf_rerank":
            # ✅ 1) 하이브리드로 후보 30개 생성
            vdocs = vectorstore.similarity_search(q, k=VECTOR_K)
            bdocs = bm25_retriever.invoke(q)
            cands = rrf_fusion([vdocs, bdocs], k=RRF_K, top_k=CANDIDATE_K)

            # (확인용) 후보 개수 출력
            print("\n" + "-" * 90)
            print(f"[Candidates] {len(cands)}개 (Hybrid RRF)")

            # ✅ 2) 후보 30개를 리랭크해서 최종 5개
            docs = cross_encoder_rerank(q, cands, top_k=TOP_K)

        else:
            raise ValueError("Unknown MODE")

        print("\n" + "=" * 90)
        print(f"MODE={MODE} | Q: {q}")
        for i, d in enumerate(docs, 1):
            print(f"\n[{i}] source={d.metadata.get('source')} chunk_id={d.metadata.get('chunk_id')} doc_id={d.metadata.get('doc_id')}")
            print(d.page_content[:350])


if __name__ == "__main__":
    main()
