import os
import sys
import getpass
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document              # 추가

# ==========================================
# 0. 환경 설정 및 초기화
# ==========================================
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

SELECTED_MODEL = "gpt-5-mini" 

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
    search_kwargs={"k": 5, "fetch_k": 20} # 하이브리드를 위해 k값 약간 조정
)

# [2] Sparse Retriever (BM25) 
print("BM25 인덱스 생성 중... (데이터 로드)")
try:
    # DB에 저장된 모든 문서를 가져와서 BM25 인덱스를 만듭니다.
    raw_docs = vectorstore.get() 
    docs = []
    for i in range(len(raw_docs['ids'])):
        if raw_docs['documents'][i]: 
            docs.append(Document(
                page_content=raw_docs['documents'][i],
                metadata=raw_docs['metadatas'][i] if raw_docs['metadatas'] else {}
            ))
    
    if not docs:
        print("오류: DB에 문서가 비어 있습니다.")
        sys.exit()
        
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 5  # 키워드 매칭 문서 5개
    print("BM25 인덱스 생성 완료")

except Exception as e:
    print(f"BM25 초기화 실패: {e}")
    sys.exit()

# [3] Ensemble Retriever (Hybrid) - 결합
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4]  # Dense(의미) 60%, Sparse(키워드) 40% 비중
)

# ==========================================
# 2. LLM & 프롬프트 설정
# ==========================================
try:
    llm = ChatOpenAI(model=SELECTED_MODEL, temperature=0)
except Exception as e:
    print(f"모델 초기화 오류: {e}")
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

# 질문 재구성 체인
history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser()

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
# 3. 체인 조립 (LCEL 방식)
# ==========================================

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# (1) 맥락 고려하여 검색 쿼리 결정
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return history_aware_retriever
    else:
        return input["input"]

# (2) 전체 RAG 체인 구성
# 변경점: retriever -> ensemble_retriever로 교체
setup_and_retrieval = RunnableParallel(
    {
        "context": contextualized_question | ensemble_retriever, 
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
    }
)

def format_context_for_prompt(input_dict):
    return {
        "context": format_docs(input_dict["context"]),
        "input": input_dict["input"],
        "chat_history": input_dict["chat_history"]
    }

# 최종 체인: 검색 -> 포맷팅 -> 답변생성 -> 파싱
rag_chain = setup_and_retrieval.assign(
    answer= format_context_for_prompt | qa_prompt | llm | StrOutputParser()
)

# ==========================================
# 4. 메인 실행 루프
# ==========================================
chat_history = [] 

print("\n" + "="*60)
print(f"입찰메이트 AI ({SELECTED_MODEL}) - Hybrid RAG Version")
print("="*60)

while True:
    query = input("\n질문 입력 (q로 종료): ")
    if query.lower() in ["q", "quit", "exit"]:
        print("종료합니다.")
        break
    
    if not query.strip():
        continue

    print("\n답변 생성 중...\n")
    
    full_response = ""
    source_documents = []

    try:
        # 스트리밍 실행
        for chunk in rag_chain.stream({"input": query, "chat_history": chat_history}):
            
            # 답변(answer) 스트리밍
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
                full_response += chunk["answer"]
            
            # 검색된 문서(context) 저장
            if "context" in chunk:
                source_documents = chunk["context"]

        print("\n")

        # 출처 표시
        if source_documents:
            print("-" * 60)
            print("[참고 문서 (Hybrid 검색)]")
            seen_sources = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                filename = os.path.basename(source)
                
                # 문서 내용 미리보기 (앞 30자)
                preview = doc.page_content[:30].replace("\n", " ")
                
                source_key = f"{filename} (p.{page+1})"
                if source_key not in seen_sources:
                    print(f"   • {filename} [Page: {page+1}] - {preview}...")
                    seen_sources.add(source_key)
            print("-" * 60)

        # 대화 기록 업데이트
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=full_response)
        ])

    except Exception as e:
        print(f"\n에러 발생: {e}")