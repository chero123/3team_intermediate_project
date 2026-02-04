import os
import sys
import shutil
import json
import getpass
import pandas as pd
import unicodedata
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# 1. 경로 및 API 설정
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# [경로 설정]
PARSING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "parsing_data")
MAPPING_FILE = os.path.join(PARSING_DATA_DIR, "parsed_mapping.json")
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "data_list.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

print(f"CSV 위치: {CSV_FILE_PATH}")
print(f"매핑 파일: {MAPPING_FILE}")

# API 키 확인
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key를 입력하세요: ")

# ==========================================
# 2. 데이터 로드 및 매핑 사전 준비
# ==========================================
print("\n데이터 로딩 및 매핑 준비 중...")

# 1) 매핑 파일 로드
if not os.path.exists(MAPPING_FILE):
    print("매핑 파일이 없습니다. text_parsing.py를 먼저 실행하세요.")
    sys.exit()

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

# 파일명 정규화 (NFC)
file_map = {}
for item in mapping_data:
    norm_name = unicodedata.normalize('NFC', item['original_filename'])
    file_map[norm_name] = item['saved_path']

# 2) CSV 파일 로드
if not os.path.exists(CSV_FILE_PATH):
    print("CSV 파일이 없습니다.")
    sys.exit()

try:
    try:
        df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')
    df = df.fillna("") # 빈 값 처리
    print(f"CSV 데이터 {len(df)}건 로드 완료")
except Exception as e:
    print(f"CSV 로드 중 에러: {e}")
    sys.exit()

# ==========================================
# 3. 문서 처리 및 Contextual Chunking
# ==========================================
print("\n전체 문서 처리 및 청킹(Contextual Chunking) 시작...")

# 텍스트 분할기 설정 (헤더 공간 고려하여 800자로 설정)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

documents = []
match_count = 0

# CSV 기준으로 루프
for idx, row in tqdm(df.iterrows(), total=len(df)):
    
    # 1. 파일명 매칭
    csv_filename = str(row.get('파일명', ''))
    if not csv_filename: continue
    
    norm_csv_filename = unicodedata.normalize('NFC', csv_filename)
    if norm_csv_filename not in file_map: continue
    
    txt_file_path = file_map[norm_csv_filename]
    if not os.path.exists(txt_file_path): continue

    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except: continue

    if not content or len(content) < 50: continue

    # 2. 메타데이터 추출
    
    notice_no = str(row.get('공고 번호', ''))
    title = str(row.get('사업명', ''))
    agency = str(row.get('발주 기관', ''))
    budget = str(row.get('사업 금액', '0'))
    deadline = str(row.get('입찰 참여 마감일', '정보없음'))
    summary = str(row.get('사업 요약', ''))

    # 금액 포맷팅 (130000000.0 -> 130,000,000)
    try:
        budget_fmt = f"{int(float(budget)):,}"
    except:
        budget_fmt = budget

    # 3. 모든 청크에 붙일 헤더 템플릿
    header_template = f"""<문서 정보>
공고번호: {notice_no}
사업명: {title}
발주기관: {agency}
사업금액: {budget_fmt}원
입찰마감: {deadline}
핵심요약: {summary[:100]}...
</문서 정보>

"""
    # 4. 원본 텍스트 분할 (Splitting)
    raw_chunks = text_splitter.split_text(content)

    # 5. 헤더 붙이기 (Injection)
    for chunk_text in raw_chunks:
        enriched_chunk_text = header_template + chunk_text
        
        # 메타데이터 저장 (나중에 필터링할 때 유용)
        metadata = {
            "source": csv_filename,
            "notice_no": notice_no,
            "title": title,
            "agency": agency,
            "budget": budget_fmt,
            "deadline": deadline
        }
        
        doc = Document(page_content=enriched_chunk_text, metadata=metadata)
        documents.append(doc)
        
    match_count += 1

print(f"\n처리 완료: 총 {match_count}개의 파일에서 -> {len(documents)}개의 '헤더 포함' 청크 생성됨.")

if match_count == 0:
    print("[주의] 매칭된 문서가 0개입니다. CSV 파일명과 파싱된 파일명을 확인하세요.")
    sys.exit()

# ==========================================
# 4. 벡터 DB 저장 (ChromaDB)
# ==========================================
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

print(f"\n벡터 DB 저장 시작 (ChromaDB)...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name="bid_rfp_collection"
)

print("-" * 50)
print(f"[성공] 모든 청크에 메타데이터가 주입된 DB 구축 완료")
print(f"저장 위치: {DB_PATH}")
print("-" * 50)