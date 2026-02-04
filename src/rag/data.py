from __future__ import annotations

"""
문서 로딩/파싱/청킹 모듈

섹션 구성:
- 메타데이터 로딩/정규화
- 파서(HWP/PDF/DOCX)
- 문서 로딩
- 청킹
"""

import csv
import os
import re
import hashlib
import struct
import unicodedata
import zlib
from typing import Dict, Iterable, List, Optional, Literal

import olefile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF 설치 필요: pip install pymupdf")
    fitz = None

try:
    import pdfplumber
except ImportError:
    print("pdfplumber 설치 필요: pip install pdfplumber")
    pdfplumber = None

from .types import Chunk, Document, Metadata

# 바이너리 확장자
SUPPORTED_BINARY_EXTENSIONS = {".pdf", ".hwp", ".docx"}

# CSV 텍스트 컬럼명을 지정
CSV_TEXT_FIELD = "텍스트"
CSV_FILENAME_FIELD = "파일명"

# PDF 파서 선택 (pdfplumber: 품질 우선, fitz: 속도 우선)
PDF_PARSER: Literal["pdfplumber", "fitz"] = "pdfplumber"


def safe_filename(original_name: str, suffix: str = "_parsed.txt", max_bytes: int = 180) -> str:
    """
    OS 파일명 길이 제한을 피하기 위해 안전한 파일명을 만든다.

    Args:
        original_name: 원본 파일명 (확장자 제외)
        suffix: 저장 파일 접미사
        max_bytes: 최대 바이트 길이

    Returns:
        str: 안전하게 변환된 파일명
    """
    name = unicodedata.normalize("NFC", original_name)
    base = re.sub(r"\.(hwp|pdf|docx)$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    tail = f"__{h}{suffix}"
    budget = max_bytes - len(tail.encode("utf-8"))
    if budget < 20:
        budget = 20
    b = base.encode("utf-8")
    if len(b) > budget:
        base = b[:budget].decode("utf-8", errors="ignore").rstrip()
    return f"{base}__{h}{suffix}"


def clean_text(text: str) -> str:
    """
    텍스트 정제: 불필요한 공백 및 깨진 문자를 정리한다.

    Args:
        text: 원본 텍스트

    Returns:
        str: 정제된 텍스트
    """
    if not text:
        return ""
    text = re.sub(
        r'[^\uAC00-\uD7A3\u2160-\u217Fa-zA-Z0-9\s.,!?():\-\[\]<>~·%/@\'"_=+○●■□▶◆※]',
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# Metadata loading/normalization
def load_metadata_csv(csv_path: str) -> Dict[str, Metadata]:
    """
    CSV 메타데이터를 파일명 기준으로 로드하는 함수

    Args:
        csv_path: CSV 파일 경로

    Returns:
        Dict[str, Metadata]: 파일명 기준 메타데이터 맵
    """
    if not os.path.exists(csv_path):
        # 파일이 없으면 빈 딕셔너리를 반환
        return {}

    # 결과 딕셔너리를 준비
    metadata_by_name: Dict[str, Metadata] = {}

    # CSV 파일을 열 때 나오는 이상한 특수 문자 방지를 위해 utf-8-sig 인코딩 사용
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        # DictReader를 생성
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get(CSV_FILENAME_FIELD) or row.get("filename") or row.get("file") or row.get("id")
            if not name:
                continue
            # None을 제거하고 저장
            metadata_by_name[name] = {k: v for k, v in row.items() if v is not None}
    return metadata_by_name


def normalize_metadata(row: Metadata) -> Metadata:
    """
    CSV 컬럼명을 표준 키로 정규화하는 함수

    Args:
        row: 원본 메타데이터 행

    Returns:
        Metadata: 표준화된 메타데이터
    """
    return {
        "notice_id": row.get("공고 번호"),
        "notice_round": row.get("공고 차수"),
        "project_name": row.get("사업명"),
        "project_amount": row.get("사업 금액"),
        "issuer": row.get("발주 기관"),
        "publish_date": row.get("공개 일자"),
        "bid_start": row.get("입찰 참여 시작일"),
        "bid_end": row.get("입찰 참여 마감일"),
        "summary": row.get("사업 요약"),
        "file_type": row.get("파일형식"),
        "filename": row.get("파일명") or row.get("filename"),
    }

# HWP parsing
def parse_hwp_all(path: str) -> str:
    """
    HWP 파일에서 BodyText를 파싱해 텍스트를 추출한다.

    Args:
        path: HWP 파일 경로

    Returns:
        str: 추출된 텍스트
    """
    if not os.path.exists(path):
        return ""
    try:
        f = olefile.OleFileIO(path)
        header = f.openstream("FileHeader").read()
        is_compressed = header[36] & 1
        texts: List[str] = []
        for entry in f.listdir():
            if entry[0] != "BodyText":
                continue
            data = f.openstream(entry).read()
            if is_compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    pass
            i = 0
            while i < len(data):
                if i + 4 > len(data):
                    break
                rec_header = struct.unpack_from("<I", data, i)[0]
                rec_type = rec_header & 0x3FF
                rec_len = (rec_header >> 20) & 0xFFF
                if rec_len == 0xFFF:
                    if i + 8 > len(data):
                        break
                    rec_len = struct.unpack_from("<I", data, i + 4)[0]
                    i += 8
                else:
                    i += 4
                if i + rec_len > len(data):
                    break
                if rec_type == 67 and rec_len > 0:
                    text_data = data[i : i + rec_len]
                    try:
                        text = text_data.decode("utf-16le", errors="ignore")
                        cleaned_chars = []
                        for char in text:
                            code = ord(char)
                            if code >= 32 or char in "\n\r\t":
                                cleaned_chars.append(char)
                            elif code in [13, 10]:
                                cleaned_chars.append("\n")
                        text = "".join(cleaned_chars).strip()
                        if text:
                            texts.append(text)
                    except UnicodeDecodeError:
                        pass
                i += rec_len
        f.close()
        if texts:
            return clean_text("\n".join(texts))
        return ""
    except Exception:
        return ""


def extract_text_from_hwp(path: str) -> str:
    """
    HWP 텍스트를 추출한다.

    Args:
        path: HWP 경로

    Returns:
        str: 추출된 텍스트
    """
    return parse_hwp_all(path)


# PDF parsing
def parse_pdf_with_pdfplumber(path: str) -> str:
    """
    pdfplumber로 PDF 텍스트를 추출한다.

    Args:
        path: PDF 경로

    Returns:
        str: 추출된 텍스트
    """
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(path) as pdf:
            text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        return clean_text(text) if text else ""
    except Exception:
        return ""


def parse_pdf_with_fitz(path: str) -> str:
    """
    PyMuPDF(fitz)로 PDF 텍스트를 추출한다.

    Args:
        path: PDF 경로

    Returns:
        str: 추출된 텍스트
    """
    if fitz is None:
        return ""
    try:
        doc = fitz.open(path)
        results = []
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                results.append(page_text.strip())
        doc.close()
        return clean_text("\n\n".join(results)) if results else ""
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> str:
    """
    PDF 텍스트를 추출한다 (parser 설정에 따라 선택).

    Args:
        path: PDF 경로

    Returns:
        str: 추출된 텍스트
    """
    if PDF_PARSER == "pdfplumber":
        text = parse_pdf_with_pdfplumber(path)
        if text:
            return text
    if PDF_PARSER == "fitz":
        text = parse_pdf_with_fitz(path)
        if text:
            return text
    # 파서가 없거나 실패하면 pypdf로 fallback
    try:
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return clean_text(text)
    except Exception:
        return ""


# DOCX parsing
def extract_text_from_docx(path: str) -> str:
    """
    docx 파일에서 텍스트를 추출하는 함수

    Args:
        path: docs 파일 경로

    Returns:
        str: 추출된 텍스트
    """
    # docx 문서를 로드
    doc = DocxDocument(path)
    # 문단 텍스트를 결합
    return clean_text("\n".join(p.text for p in doc.paragraphs if p.text))


# Document loading
def load_documents(data_dir: str, metadata_csv: str | None = None) -> List[Document]:
    """
    데이터 디렉토리에서 문서를 로드하는 함수

    Args:
        data_dir: 데이터 디렉토리 경로
        metadata_csv: 메타데이터 CSV 경로

    Returns:
        List[Document]: 문서 리스트
    """
    # 메타데이터를 로드
    metadata_map = load_metadata_csv(metadata_csv) if metadata_csv else {}

    # 결과 리스트를 준비
    documents: List[Document] = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            # 파일 경로를 구성
            path = os.path.join(root, filename)
            # 확장자를 추출
            ext = os.path.splitext(filename)[1].lower()

            # 파일별 메타데이터를 가져온다
            metadata = dict(metadata_map.get(filename, {}))
            # 메타데이터를 정규화한다
            normalized = normalize_metadata(metadata)

            # 바이너리 문서면 SV 텍스트를 확인
            if ext in SUPPORTED_BINARY_EXTENSIONS:
                csv_text = metadata.get(CSV_TEXT_FIELD)
                if csv_text and csv_text.strip():
                    text = csv_text
                # PDF라면 PDF를 파싱
                elif ext == ".pdf":
                    text = extract_text_from_pdf(path)
                # HWP라면 HWP를 파싱
                elif ext == ".hwp":
                    text = extract_text_from_hwp(path)
                # docx라면 docx를 파싱
                elif ext == ".docx":
                    text = extract_text_from_docx(path)
                else:
                    text = ""
            else:
                continue

            # 문서 ID를 만들기
            doc_id = os.path.relpath(path, data_dir)
            # 표준 메타데이터를 합친다
            metadata.update(normalized)
            # 경로 정보를 추가
            metadata.update({"path": path, "filename": filename})
            # 문서를 추가
            documents.append(Document(id=doc_id, text=text, metadata=metadata))
    return documents


# Chunking
def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    RecursiveCharacterTextSplitter 기반 청킹을 수행

    Args:
        text: 입력 텍스트
        chunk_size: 청크 길이
        overlap: 청크 겹침 길이

    Returns:
        List[str]: 청크 문자열 리스트
    """
    # 입력 검증을 수행
    if chunk_size <= 0:
        return []

    # 문장/문단 경계를 우선하고, 부족하면 더 작은 단위로 재귀 분할한다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "□", "。", ".", "!", "?", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_documents(docs: Iterable[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    """
    문서 리스트를 청크 리스트로 변환하는 함수

    Args:
        docs: 문서
        chunk_size: 청크 길이
        overlap: 청크 겹침 길이

    Returns:
        List[Chunk]: 청크 리스트
    """
    all_chunks: List[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(simple_chunk(doc.text, chunk_size, overlap)):
            # 청크 ID 생성
            chunk_id = f"{doc.id}::chunk_{i}"
            # 메타데이터를 복사
            metadata = dict(doc.metadata)
            # 청크 정보를 추가
            metadata.update({"chunk_index": i, "doc_id": doc.id})
            # 청크를 추가
            all_chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
    return all_chunks
