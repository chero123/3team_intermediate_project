import os
import getpass
import sys
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 환경 및 모델 설정 
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# [프로젝트 가이드 기준 모델 선택]
# 1. gpt-5-mini: 최신 가성비 모델 (싸고 좋음)
# 2. gpt-5-nano: 초경량 모델 (속도 중요할 때)
# 3. gpt-5: 고성능 모델 (제한적 사용 권장 - 테스트용)

SELECTED_MODEL = "gpt-5-mini"  

print(f"사용 중인 모델: {SELECTED_MODEL}")

# ==========================================
# 2. 리트리버(Retriever) 설정
# ==========================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if not os.path.exists(DB_PATH):
    print("오류: DB가 없습니다. create_vectordb.py를 먼저 실행하세요.")
    sys.exit()

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="bid_rfp_collection"
)

# MMR 검색 사용 (다양한 정보 수집)
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "fetch_k": 10}
)

# ==========================================
# 3. 프롬프트 & LLM 설정
# ==========================================

try:
    llm = ChatOpenAI(model=SELECTED_MODEL, temperature=0)
except Exception as e:
    print(f"모델 설정 오류: {e}")
    sys.exit()

template = """
당신은 공공 입찰(RFP) 분석 전문가 '입찰메이트'입니다.
아래 제공된 [검색된 문서] 정보를 바탕으로 사용자의 질문에 답변하세요.

1. 문서에 있는 사실만 말하고, 없는 내용은 "정보가 문서에 없습니다"라고 답하세요.
2. 예산이나 기간 같은 숫자는 정확하게 명시하세요.
3. 답변 끝에 반드시 출처(공고번호 또는 사업명)를 괄호로 표기하세요.

[검색된 문서]:
{context}

질문: {question}
답변:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n--- [문서 {i+1}] ---\n"
        context += doc.page_content
        context += "\n"
    return context

# ==========================================
# 4. 체인 연결 및 실행
# ==========================================
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*50)
print(f"입찰메이트 AI ({SELECTED_MODEL}) 가 준비되었습니다.")
print("="*50)

while True:
    query = input("\n질문 입력 (종료: q): ")
    if query.lower() in ["q", "quit", "exit"]:
        print("종료합니다.")
        break
    
    if not query.strip():
        continue

    print("생각 중...", end="", flush=True)
    
    try:
        response = rag_chain.invoke(query)
        print("\r" + " " * 20 + "\r", end="")
        print(f"답변:\n{response}")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        if "404" in str(e):
            print("모델을 찾을 수 없습니다. API Key 권한을 확인하거나 모델명을 다시 확인하세요.")
        
    print("-" * 50)