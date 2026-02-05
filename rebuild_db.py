import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 경로 설정
# 원본 PDF 경로와 파싱 텍스트 경로를 각각 지정합니다.
pdf_path = "/home/spai0630/workspace/3team_intermediate_project/data/original_data/"
txt_path = "/home/spai0630/workspace/3team_intermediate_project/data/original_data/parsing_data/"
db_path = "/home/spai0630/workspace/3team_intermediate_project/db"

print("1. 문서 로딩 시작...")

# 2. 문서 로더 초기화
# PDF 로더: 폴더 내 모든 PDF 파일을 읽어옵니다.
pdf_loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)

# 텍스트 로더: 파싱된 폴더 내 모든 TXT 파일을 UTF-8 인코딩으로 읽어옵니다.
txt_loader = DirectoryLoader(
    txt_path, 
    glob="*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}
)

# 3. 문서 통합
documents = []
if os.path.exists(pdf_path):
    documents.extend(pdf_loader.load())
if os.path.exists(txt_path):
    documents.extend(txt_loader.load())

if not documents:
    print("❌ 에러: 로드할 문서가 없습니다. 경로와 파일 확장자를 확인해주세요.")
    exit()

print(f"2. 문서 분할 중... (총 {len(documents)} 개 문서 로드 완료)")

# 4. 텍스트 분할 (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

print(f"3. 임베딩 및 벡터 DB 생성 중 (분할된 텍스트 수: {len(texts)})...")

# 5. 임베딩 모델 및 벡터 스토어 구축
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
vector_db = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings, 
    persist_directory=db_path
)

print(f"✅ 통합 DB 생성 완료! 위치: {db_path}")