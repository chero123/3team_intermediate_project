import os
import sys
import shutil
import getpass
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 프로젝트 경로 및 API 설정
# ==========================================

# 현재 파일의 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 데이터 경로 설정
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "parsing_data")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

print(f"현재 스크립트 위치: {CURRENT_DIR}")
print(f"데이터 읽는 경로: {INPUT_DIR}")
print(f"DB 저장 경로:   {DB_PATH}")

# API 키 확인
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# ==========================================
# 2. 기존 DB 초기화 (중복 방지)
# ==========================================
if os.path.exists(DB_PATH):
    print(f"\n기존 벡터 DB를 삭제하고 새로 생성합니다... ({DB_PATH})")
    shutil.rmtree(DB_PATH)
else:
    print(f"\n새로운 벡터 DB를 생성합니다.")

# ==========================================
# 3. 문서 로드 (Load)
# ==========================================
if not os.path.exists(INPUT_DIR):
    print(f"오류: 데이터 폴더가 없습니다 -> {INPUT_DIR}")
    print("   (먼저 text_parsing.py를 실행했는지 확인해주세요.)")
    sys.exit()

print(f"\n텍스트 데이터 로딩 중...")

# .txt 파일만 골라서 로드
loader = DirectoryLoader(
    INPUT_DIR,
    glob="*_parsed.txt",
    loader_cls=TextLoader,
    show_progress=True
)

documents = loader.load()

if not documents:
    print("로드된 문서가 0개입니다. 폴더가 비어있습니다.")
    sys.exit()

print(f"총 {len(documents)}개의 문서 파일 로드 완료")

# ==========================================
# 4. 텍스트 청킹 (Split)
# ==========================================
print("\n텍스트 청킹(Chunking) 시작...")

# 입찰 공고문(한글)에 최적화된 분할 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 한 덩어리 크기 (약 300~500단어)
    chunk_overlap=200,    # 앞뒤로 겹치는 구간 (문맥 끊김 방지)
    # 자르는 우선순위: 문단(엔터2번) -> 줄바꿈 -> 마침표 -> 공백 -> 글자
    separators=["\n\n", "\n", ". ", " ", ""], 
    length_function=len,
)

split_docs = text_splitter.split_documents(documents)

print(f"청킹 완료: {len(documents)}개 원본 문서 -> {len(split_docs)}개 청크 생성됨")

# ==========================================
# 5. 임베딩 및 DB 저장 (Embed & Store)
# ==========================================
print(f"\n벡터 DB 생성 및 저장 시작 (Model: text-embedding-3-small)...")

# OpenAI 임베딩 모델 정의
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ChromaDB에 데이터 주입 및 디스크 저장
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="bid_rfp_collection"
)

print("-" * 50)
print(f"[성공] 모든 작업이 완료되었습니다!")
print(f"생성된 DB 위치: {DB_PATH}")
print("-" * 50)