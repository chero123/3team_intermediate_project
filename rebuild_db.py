import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker

# 1. 경로 설정
# 원본 PDF 경로와 파싱 텍스트 경로를 각각 지정합니다.
pdf_path = "/home/spai0630/workspace/3team_intermediate_project/data/original_data/"
txt_path = "/home/spai0630/workspace/3team_intermediate_project/data/original_data/parsing_data/"
db_path = "/home/spai0630/workspace/3team_intermediate_project/db"

# 임베딩 모델 및 스플리터 초기화 (수정사항)
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


# 의미론적 분할기 설정 (유사도 기반 분할 임계값 최적화)
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80.0
)

# 텍스트 길이 최적화를 위한 recursive splitter 설정
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 800자에서 예산등 세부질문에 답변을 못하길래 500으로 축소
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 문서 로더 초기화 및 데이터 로딩
print("문서 로딩 작업을 시작합니다...")
documents = []

# PDF 문서 로더
if os.path.exists(pdf_path):
    pdf_loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

# 텍스트 문서 로더 (UTF-8 인코딩 적용)
if os.path.exists(txt_path):
    txt_loader = DirectoryLoader(
        txt_path, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents.extend(txt_loader.load())

if not documents:
    print("로드된 문서가 없습니다. 경로 설정을 확인하십시오.")
    exit()
    
    
# 하이브리드 청킹 및 문맥 보강 수행
print(f"청킹 프로세스를 실행합니다 (총 {len(documents)}개 파일)...")
final_docs = []
for doc in documents:
    # 파일명 정보를 청크 내부에 직접 주입 (나연님 전략 적용)
    source_info = os.path.basename(doc.metadata.get('source', 'unknown'))
    doc.page_content = f"[[문서: {source_info}]]\n{doc.page_content}"
    
    # 1차 Semantic 분할 후 2차 500자 단위 재분할
    s_chunks = semantic_splitter.split_documents([doc])
    final_docs.extend(recursive_splitter.split_documents(s_chunks))

# 5. 벡터 데이터베이스 생성 및 로컬 저장
print(f"벡터 DB를 생성합니다 (최종 청크 수: {len(final_docs)})...")
vector_db = Chroma.from_documents(
    documents=final_docs,
    embedding=embeddings,
    persist_directory=db_path
)
vector_db.persist()

print(f"데이터베이스 생성이 완료되었습니다: {db_path}")


