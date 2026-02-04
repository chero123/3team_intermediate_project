import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 유종 님이 알려주신 정확한 데이터 경로
data_path = "/home/spai0630/workspace/data/original_data/files"
# 2. DB는 현재 프로젝트 폴더 안에 생성
db_path = "/home/spai0630/workspace/3team_intermediate_project/db"

print(f"1. 문서 로딩 시작 (경로: {data_path})...")

if not os.path.exists(data_path):
    print(f"❌ 에러: {data_path} 폴더가 없습니다. 경로를 다시 확인해주세요.")
    exit()

# PDF 로딩
loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    print(f"❌ 에러: {data_path} 안에 PDF 파일이 없습니다. 파일 확장자가 .pdf 인지 확인해주세요.")
    exit()

print(f"2. 문서 분할 중... (총 {len(documents)} 페이지)")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

print("3. 임베딩 및 DB 생성 중 (약 1~2분 소요)...")
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=db_path)

print(f"✅ DB 생성 완료! 위치: {db_path}")
