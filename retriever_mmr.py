# run_retriever.py
# 리트리버 mmr , similarity 방식 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

FAISS_DIR = "data/faiss_index"

# =========================
# 리트리버 모드 선택
# =========================
RETRIEVER_MODE = "mmr"
# 옵션: "similarity", "mmr"

TOP_K = 5
FETCH_K = 20        # MMR용
LAMBDA_MULT = 0.7   # MMR용


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # ✅ retriever 생성
    if RETRIEVER_MODE == "similarity":
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )
    elif RETRIEVER_MODE == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K,
                "fetch_k": FETCH_K,
                "lambda_mult": LAMBDA_MULT,
            },
        )
    else:
        raise ValueError("Unknown RETRIEVER_MODE")

    # 질문
    queries = [
        "벤처확인종합관리시스템 추진방안 구축 인터뷰내용 알려줘",
        "학사정보시스템 고도화 사업의 예산과 주요 과업 내용을 요약해줘"

    ]

    for query in queries:
        docs = retriever.invoke(query)
        print("\n" + "=" * 80)
        print(f"질문: {query}")
        for i, d in enumerate(docs, 1):
            print(f"\n[{i}] source={d.metadata.get('source')} chunk_id={d.metadata.get('chunk_id')}")
            print(d.page_content[:300])


if __name__ == "__main__":
    main()
