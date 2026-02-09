# ============================================================
# final_docs/*.md 를 전부(또는 CSV와 매칭되는 것만) 불러와서
# - (선택) CSV 메타데이터를 각 청크 헤더로 주입
# - MD의 <!-- page: N --> / <!-- tables: start/end page N --> 구조를 최대한 보존
# - OpenAI Embedding -> ChromaDB(persist) 구축
#
# ✅ OPENAI_API_KEY는 .env에서 로드 (getpass 입력 없음)
#   - 프로젝트 루트(=PROJECT_ROOT)에 .env 두는 걸 권장
#   - 예: PROJECT_ROOT/.env  안에  OPENAI_API_KEY=sk-...
# ============================================================

import os
import sys
import glob
import re
import shutil
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# =========================
# 0) 경로/환경변수(.env) 로드
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# .env 로드 (프로젝트 루트 우선, 없으면 현재 폴더도 시도)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# ✅ MD 폴더 / CSV / DB 경로
FINAL_DOCS_DIR = os.path.join(PROJECT_ROOT, "final_docs")                 # ✅ MD 폴더
CSV_FILE_PATH  = os.path.join(PROJECT_ROOT, "data", "data_list.csv")      # ✅ 메타데이터(선택)
DB_PATH        = os.path.join(PROJECT_ROOT, "data", "chroma_db")          # ✅ 저장 위치

# ✅ 동작 옵션
USE_CSV_METADATA = True   # True: CSV와 매칭되는 문서만 + 헤더주입 / False: final_docs MD 전부 인덱싱
REBUILD_DB = True         # True: 기존 DB 삭제 후 재생성

COLLECTION_NAME = "bid_rfp_collection"
EMBED_MODEL = "text-embedding-3-small"

# 청킹 파라미터
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

print(f"[INFO] PROJECT_ROOT     = {PROJECT_ROOT}")
print(f"[INFO] FINAL_DOCS_DIR   = {FINAL_DOCS_DIR}")
print(f"[INFO] CSV_FILE_PATH    = {CSV_FILE_PATH} (USE_CSV_METADATA={USE_CSV_METADATA})")
print(f"[INFO] DB_PATH          = {DB_PATH}")
print(f"[INFO] COLLECTION_NAME  = {COLLECTION_NAME}")
print(f"[INFO] EMBED_MODEL      = {EMBED_MODEL}")

# ✅ API KEY 체크 (.env 기반)
if not os.environ.get("OPENAI_API_KEY"):
    print("[ERROR] OPENAI_API_KEY가 설정되어 있지 않습니다.")
    print(" - PROJECT_ROOT/.env 또는 CURRENT_DIR/.env에 아래처럼 추가하세요:")
    print("   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx")
    sys.exit(1)


# =========================
# 1) 유틸: 파일명 정규화/매칭
# =========================
def norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())

def build_md_index(final_docs_dir: str) -> Dict[str, str]:
    """
    final_docs 폴더의 모든 md 파일을 key 여러 버전으로 매핑
    - basename(md)
    - basename_without_md
    - lowercase variants
    - (basename_without_md + ".md") variants
    """
    if not os.path.isdir(final_docs_dir):
        print(f"[ERROR] final_docs 폴더가 없습니다: {final_docs_dir}")
        sys.exit(1)

    md_paths = glob.glob(os.path.join(final_docs_dir, "*.md"))
    if not md_paths:
        print(f"[ERROR] final_docs에 md 파일이 없습니다: {final_docs_dir}")
        sys.exit(1)

    idx: Dict[str, str] = {}
    for p in md_paths:
        base = norm(os.path.basename(p))  # ex) "A.pdf.md" or "A.md"
        base_no_md = norm(re.sub(r"\.md$", "", base, flags=re.IGNORECASE))  # ex) "A.pdf" or "A"

        # 원형
        idx[base] = p
        idx[base_no_md] = p

        # 소문자
        idx[base.lower()] = p
        idx[base_no_md.lower()] = p

        # 흔한 케이스: csv가 "A.pdf"인데 md는 "A.pdf.md"
        idx[base_no_md + ".md"] = p
        idx[(base_no_md + ".md").lower()] = p

    return idx

def resolve_md_path(md_index: Dict[str, str], csv_filename: str) -> Optional[str]:
    """
    CSV 파일명으로 md 경로를 최대한 유연하게 찾는다.
    후보:
    - csv 그대로
    - csv + ".md"
    - basename(csv)
    - basename(csv) + ".md"
    - 확장자 제거한 이름 / 그 + ".md"
    """
    f = norm(csv_filename)
    if not f:
        return None

    candidates: List[str] = []
    candidates.append(f)
    candidates.append(f + ".md")

    base = norm(os.path.basename(f))
    candidates.append(base)
    candidates.append(base + ".md")

    # 확장자 제거(.pdf/.hwp/.docx 등) 후 재시도
    base_no_ext = re.sub(r"\.[^.]+$", "", base)
    if base_no_ext and base_no_ext != base:
        candidates.append(base_no_ext)
        candidates.append(base_no_ext + ".md")

    # 소문자 후보
    candidates += [c.lower() for c in candidates]

    for c in candidates:
        if c in md_index:
            return md_index[c]
    return None


# =========================
# 2) MD 청킹: 표 블록 보존 + 페이지 마커 활용
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

TABLE_BLOCK_RE = re.compile(
    r"<!--\s*tables:\s*start\s*page\s*\d+\s*-->.*?<!--\s*tables:\s*end\s*page\s*\d+\s*-->",
    re.DOTALL | re.IGNORECASE
)
PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)

def split_md_preserve_tables(md_text: str) -> List[str]:
    """
    - 표 블록( tables:start/end )은 통째로 보존
    - 가능하면 <!-- page: N --> 단위로 먼저 나누고, 각 섹션을 청킹
    """
    tables: List[str] = []

    def _table_repl(m: re.Match) -> str:
        tables.append(m.group(0))
        return f"[[[TABLE_BLOCK_{len(tables)-1}]]]"

    protected = TABLE_BLOCK_RE.sub(_table_repl, md_text)

    # 페이지 마커 기준 split: tokens = [pre, "9", content, "10", content, ...]
    tokens = PAGE_MARKER_RE.split(protected)
    page_sections: List[str] = []

    pre = tokens[0]
    if pre.strip():
        page_sections.append(pre)

    i = 1
    while i < len(tokens):
        page_no = tokens[i]
        content = tokens[i + 1] if i + 1 < len(tokens) else ""
        page_sections.append(f"<!-- page: {page_no} -->\n{content}".strip())
        i += 2

    # 페이지 마커가 없으면 전체를 한 덩어리로
    if not page_sections:
        page_sections = [protected]

    chunks: List[str] = []
    for sec in page_sections:
        sec = sec.strip()
        if not sec:
            continue

        raw_chunks = text_splitter.split_text(sec)

        # 표 토큰 복구
        for c in raw_chunks:
            def _restore(match: re.Match) -> str:
                idx = int(match.group(1))
                return tables[idx]
            restored = re.sub(r"\[\[\[TABLE_BLOCK_(\d+)\]\]\]", _restore, c)
            chunks.append(restored)

    return chunks


# =========================
# 3) (선택) CSV 로드
# =========================
def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV가 없습니다: {csv_path}")
        sys.exit(1)

    try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp949")
        return df.fillna("")
    except Exception as e:
        print(f"[ERROR] CSV 로드 실패: {e}")
        sys.exit(1)

def build_header_from_row(row: pd.Series, source_name: str) -> Tuple[str, Dict]:
    notice_no = str(row.get("공고 번호", "")) or str(row.get("공고번호", ""))
    title     = str(row.get("사업명", "")) or str(row.get("공고명", ""))
    agency    = str(row.get("발주 기관", "")) or str(row.get("발주기관", ""))
    budget    = str(row.get("사업 금액", "0")) or str(row.get("사업금액", "0"))
    deadline  = str(row.get("입찰 참여 마감일", "정보없음")) or str(row.get("마감일", "정보없음"))
    summary   = str(row.get("사업 요약", "")) or str(row.get("요약", ""))

    # 금액 포맷팅
    try:
        budget_fmt = f"{int(float(budget)):,}"
    except:
        budget_fmt = budget

    header = f"""<문서 정보>
공고번호: {notice_no}
사업명: {title}
발주기관: {agency}
사업금액: {budget_fmt}원
입찰마감: {deadline}
핵심요약: {summary[:100]}...
원본문서: {source_name}
</문서 정보>

"""
    metadata = {
        "source": source_name,
        "notice_no": notice_no,
        "title": title,
        "agency": agency,
        "budget": budget_fmt,
        "deadline": deadline,
    }
    return header, metadata


# =========================
# 4) 문서 로딩 -> 청킹 -> Document 생성
# =========================
md_index = build_md_index(FINAL_DOCS_DIR)

documents: List[Document] = []
match_count = 0
skipped_count = 0

if USE_CSV_METADATA:
    df = load_csv(CSV_FILE_PATH)
    print(f"[INFO] CSV rows = {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV→MD 매칭/청킹"):
        csv_filename = str(row.get("파일명", "")).strip()
        if not csv_filename:
            skipped_count += 1
            continue

        md_path = resolve_md_path(md_index, csv_filename)
        if not md_path or not os.path.exists(md_path):
            skipped_count += 1
            continue

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            skipped_count += 1
            continue

        if not content or len(content) < 50:
            skipped_count += 1
            continue

        header, base_meta = build_header_from_row(row, source_name=csv_filename)
        chunks = split_md_preserve_tables(content)

        for c in chunks:
            page_content = header + c
            meta = dict(base_meta)
            meta["source_md_path"] = md_path
            documents.append(Document(page_content=page_content, metadata=meta))

        match_count += 1

    print(f"\n[RESULT] 매칭 성공 문서 수 = {match_count}")
    print(f"[RESULT] 스킵(미매칭/에러) 수 = {skipped_count}")

else:
    # CSV 없이 final_docs/*.md 전부 인덱싱
    md_paths = sorted(set(md_index.values()))
    print(f"[INFO] final_docs md files = {len(md_paths)}")

    for md_path in tqdm(md_paths, desc="MD 전부 청킹"):
        base = os.path.basename(md_path)

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            continue

        if not content or len(content) < 50:
            continue

        header = f"""<문서 정보>
원본문서: {base}
</문서 정보>

"""
        base_meta = {"source": base, "source_md_path": md_path}

        chunks = split_md_preserve_tables(content)
        for c in chunks:
            documents.append(Document(page_content=header + c, metadata=dict(base_meta)))

        match_count += 1

    print(f"\n[RESULT] 처리한 MD 파일 수 = {match_count}")

if match_count == 0 or len(documents) == 0:
    print("[ERROR] 처리된 문서/청크가 0개입니다.")
    print(" - USE_CSV_METADATA=True면: CSV의 파일명과 final_docs의 md 파일명이 매칭되는지 확인하세요.")
    print(" - USE_CSV_METADATA=False면: final_docs에 md가 있는지 확인하세요.")
    sys.exit(1)

print(f"[RESULT] 총 생성된 청크(Document) 수 = {len(documents)}")


# =========================
# 5) ChromaDB 구축/저장
# =========================
if REBUILD_DB and os.path.exists(DB_PATH):
    print(f"[INFO] 기존 DB 삭제: {DB_PATH}")
    shutil.rmtree(DB_PATH)

print("[INFO] 임베딩/ChromaDB 생성 시작...")
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
)

print("-" * 60)
print("[SUCCESS] ChromaDB 구축 완료")
print(f" - 저장 위치: {DB_PATH}")
print(f" - 컬렉션: {COLLECTION_NAME}")
print(f" - 청크 수: {len(documents)}")
print("-" * 60)
