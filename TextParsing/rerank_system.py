import os
import sys
import getpass
from dotenv import load_dotenv
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ==========================================
# 0. 환경 설정 및 초기화
# ==========================================
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

model_options = ["gpt-5-mini", "gpt-5-nano", "gpt-5"]
SELECTED_MODEL = model_options[0]  # 기본값: gpt-5-mini

print(f"시스템 초기화 중... (Model: {SELECTED_MODEL})")

# ==========================================
# 1. 하이브리드 리트리버 설정
# ==========================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(DB_PATH):
    print(f"오류: DB 경로({DB_PATH})가 존재하지 않습니다.")
    sys.exit()

# [1] Dense Retriever (Chroma)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="bid_rfp_collection"
)

dense_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 10, "fetch_k": 30} 
)

# [2] Sparse Retriever (BM25) 
print("BM25 인덱스 생성 중... (데이터 로드)")
try:
    raw_docs = vectorstore.get() 
    docs = []
    if raw_docs['ids']:
        for i in range(len(raw_docs['ids'])):
            content = raw_docs['documents'][i]
            if content: 
                docs.append(Document(
                    page_content=content,
                    metadata=raw_docs['metadatas'][i] if raw_docs['metadatas'] else {}
                ))
    
    if not docs:
        print("경고: DB에 문서가 없어 검색 기능이 제한됩니다.")
        sparse_retriever = None
    else:
        sparse_retriever = BM25Retriever.from_documents(docs)
        sparse_retriever.k = 10
        print(f"BM25 인덱스 생성 완료 (문서 수: {len(docs)})")

except Exception as e:
    print(f"BM25 초기화 실패: {e}")
    sparse_retriever = None

# [3] Ensemble Retriever (Hybrid)
if sparse_retriever:
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4] 
    )
else:
    ensemble_retriever = dense_retriever

# ==========================================
# 1.5. 리랭커 설정 
# ==========================================
print("리랭킹(Reranking) 모델 로딩 중... (FlashRank)")

try:
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    print("하이브리드 + 리랭킹 검색 엔진 준비 완료")
except Exception as e:
    print(f"리랭커 로딩 실패 (일반 검색으로 전환): {e}")
    compression_retriever = ensemble_retriever


# ==========================================
# 2. LLM & 프롬프트 설정
# ==========================================
try:
    llm = ChatOpenAI(model=SELECTED_MODEL, temperature=0)
except Exception as e:
    print(f"모델 초기화 실패 (API Key 또는 모델명 확인 필요): {e}")
    sys.exit()

# 2-1. 질문 재구성 (Contextualize Query)
contextualize_q_system_prompt = """
채팅 기록과 최신 사용자 질문이 주어지면, 
이 질문이 채팅 기록의 맥락을 참조하고 있을 경우 
채팅 기록 없이도 이해할 수 있는 '독립적인 질문'으로 재구성하세요.
질문에 답하지 말고, 질문을 재구성하거나 그대로 반환하기만 하세요.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_chain = contextualize_q_prompt | llm | StrOutputParser()

# 2-2. 답변 생성 (QA)
qa_system_prompt = """
당신은 공공 입찰(RFP) 분석 전문가 '입찰메이트'입니다.
아래의 [검색된 문서]를 사용하여 질문에 답변하세요.

규칙:
1. 문서를 기반으로 사실만 답변하고, 모르면 "문서에 해당 내용이 없습니다"라고 하세요.
2. 예산, 기간, 날짜 등 숫자는 정확히 기재하세요.
3. 답변은 보기 좋게 Markdown 형식(볼드체, 리스트 등)을 사용하세요.

[검색된 문서]:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ==========================================
# 3. 체인 조립 (LCEL)
# ==========================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_search_query(input_dict):
    if input_dict.get("chat_history"):
        return history_aware_chain
    else:
        return input_dict["input"]

# (1) 검색 체인
retrieval_chain = RunnableLambda(get_search_query) | compression_retriever

# (2) 답변 체인
answer_chain = (
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"]))
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# (3) 전체 RAG 체인 (답변 + 근거문서 반환)
rag_chain = RunnableParallel(
    {
        "context": retrieval_chain,
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history"),
    }
).assign(answer=answer_chain)


# ==========================================
# 4. 메인 실행 루프
# ==========================================
chat_history = [] 

print("\n" + "="*60)
print(f"입찰메이트 AI ({SELECTED_MODEL}) - Hybrid + Reranking Version")
print("="*60)

while True:
    query = input("\n질문 입력 (q로 종료): ")
    if query.lower() in ["q", "quit", "exit"]:
        print("종료합니다.")
        break
    
    if not query.strip():
        continue

    print(f"\n답변 생성 중... (Model: {SELECTED_MODEL})\n")
    
    full_response = ""
    source_documents = []

    try:
        for chunk in rag_chain.stream({"input": query, "chat_history": chat_history}):
            
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
                full_response += chunk["answer"]
            
            if "context" in chunk:
                source_documents = chunk["context"]

        print("\n")

        if source_documents:
            print("-" * 60)
            print("[참고 문서 (Top Result)]")
            seen_sources = set()
            
            for i, doc in enumerate(source_documents):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                filename = os.path.basename(source)
                score = doc.metadata.get("relevance_score", 0.0)
                preview = doc.page_content[:40].replace("\n", " ")
                
                source_key = f"{filename}_{page}"
                
                if source_key not in seen_sources:
                    print(f" {i+1}. [점수: {score:.4f}] {filename} (p.{page+1})")
                    print(f"     내용: {preview}...")
                    seen_sources.add(source_key)
            print("-" * 60)
        else:
            print("[참고 문서 없음]")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=full_response))

    except Exception as e:
        print(f"\n에러 발생: {e}")