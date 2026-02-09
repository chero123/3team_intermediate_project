import os
import getpass 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ==========================================
# 0. API 키 설정
# ==========================================
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# ==========================================
# 1. 경로 설정 (DB가 있는 곳)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# ==========================================
# 2. DB 불러오기
# ==========================================
print(f"DB 불러오는 중: {DB_PATH}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="bid_rfp_collection"
    )
    print("DB 연결 성공!")
except Exception as e:
    print(f"DB 연결 실패: {e}")
    exit()

# ==========================================
# 3. 검색 테스트
# ==========================================
# 공고문에 있을 법한 질문으로 테스트해보세요.
query = "사업 기간과 예산은 얼마인가요?" 
print(f"\n질문: '{query}'")

try:
    results = vectorstore.similarity_search(query, k=3) # 상위 3개 찾기

    if results:
        print(f"검색 성공! ({len(results)}개 찾음)\n")
        for i, doc in enumerate(results):
            print(f"--- [결과 {i+1}] ---")
            print(doc.page_content[:200]) # 내용 앞부분만 출력
            print(f"\n(출처: {doc.metadata.get('source', '알수없음')})")
            print("-" * 30)
    else:
        print("검색 결과가 없습니다. DB가 비어있거나, 검색어가 문서 내용과 관련이 없습니다.")
        
except Exception as e:
    print(f"검색 중 에러 발생: {e}")
    print("힌트: DB가 제대로 생성되지 않았거나, 경로가 틀렸을 수 있습니다.")