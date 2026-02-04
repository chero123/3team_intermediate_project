import os
import getpass
import sys
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import tiktoken

# ==========================================
# 1. 경로 및 API 설정
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 데이터 경로 설정 (프로젝트 루트 기준)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "parsing_data")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

print(f"현재 스크립트 위치: {CURRENT_DIR}")
print(f"데이터 읽는 경로: {INPUT_DIR}")
print(f"DB 저장 경로:   {DB_PATH}")

# API 키 입력 (환경변수에 없으면 입력받음)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# ==========================================
# 2. 문서 로드 (Load)
# ==========================================
if not os.path.exists(INPUT_DIR):
    print(f"오류: 입력 폴더를 찾을 수 없습니다: {INPUT_DIR}")
    print("먼저 text_parsing.py를 실행해서 데이터를 생성해주세요.")
    sys.exit()

print(f"데이터 로딩 중... (경로: {INPUT_DIR})")

# _parsed.txt로 끝나는 파일만 로드
loader = DirectoryLoader(
    INPUT_DIR,
    glob="*_parsed.txt",
    loader_cls=TextLoader,
    show_progress=True
)

documents = loader.load()

if not documents:
    print("로드된 문서가 없습니다. 파싱된 텍스트 파일이 있는지 확인해주세요.")
    sys.exit()

print(f"총 {len(documents)}개의 문서 로드 완료")

# ==========================================
# 3. 스마트 청킹 (Split)
# ==========================================
print("\n스마트 청킹 시작...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,
)

split_docs = text_splitter.split_documents(documents)

print(f"청킹 완료: {len(split_docs)}개 청크 생성")

# ==========================================
# 4. 벡터 DB 저장 (Chroma)
# ==========================================
print(f"\n벡터 DB 저장 시작 (경로: {DB_PATH})...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="bid_documents"
)

print(f"성공! 벡터 DB가 '{DB_PATH}'에 저장되었습니다.")