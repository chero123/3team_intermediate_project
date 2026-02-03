from __future__ import annotations

import csv
import os
import subprocess
from typing import Dict, Iterable, List, Optional

from docx import Document as DocxDocument
from hwp5 import hwp5txt
from pypdf import PdfReader

from .types import Chunk, Document, Metadata

# 텍스트 확장자
SUPPORTED_TEXT_EXTENSIONS = {".txt"}
# 바이너리 확장자
SUPPORTED_BINARY_EXTENSIONS = {".pdf", ".hwp", ".docx"}

# CSV 텍스트 컬럼명을 지정
CSV_TEXT_FIELD = "텍스트"
# CSV 파일명 컬럼명을 지정
CSV_FILENAME_FIELD = "파일명"


def load_metadata_csv(csv_path: str) -> Dict[str, Metadata]:
    """
    load_metadata_csv는 CSV 메타데이터를 파일명 기준으로 로드

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
    with open(csv_path, "r", encoding="utf-8") as f:
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
    normalize_metadata는 CSV 컬럼명을 표준 키로 정규화

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


def _read_text_file(path: str) -> str:
    """
    _read_text_file는 텍스트 파일을 읽는다.

    Args:
        path: 파일 경로

    Returns:
        str: 파일의 텍스트 내용
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(path: str) -> str:
    """
    extract_text_from_pdf는 PDF 텍스트를 추출

    Args:
        path: PDF 경로

    Returns:
        str: 추출된 텍스트
    """
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_text_with_hwp5txt(path: str) -> Optional[str]:
    """
    _extract_text_with_hwp5txt는 hwp5txt CLI를 사용해 HWP를 변환

    Args:
        path: HWP 경로

    Returns:
        Optional[str]: 성공 시 텍스트를 반환
    """
    try:
        # CLI를 호출
        result = subprocess.run(
            # 커맨드를 지정
            ["hwp5txt", path],
            # 실패 시 예외를 발생
            check=True,
            # 출력을 캡처
            capture_output=True,
            # 텍스트 모드로 실행
            text=True,
        )
        # stdout을 반환
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 실패 시 None을 반환
        return None


def extract_text_from_hwp(path: str) -> str:
    """
    extract_text_from_hwp는 HWP 텍스트를 추출

    Args:
        path: HWP 경로

    Returns:
        str: 추출된 텍스트
    """
    # CLI를 우선 시도
    text = _extract_text_with_hwp5txt(path)
    if text is not None:
        return text
    # 라이브러리로 변환
    return hwp5txt.get_text(path)


def extract_text_from_docx(path: str) -> str:
    """
    extract_text_from_docx는 DOCX 텍스트를 추출

    Args:
        path: DOCX 경로

    Returns:
        str: 추출된 텍스트
    """
    # DOCX 문서를 로드
    doc = DocxDocument(path)
    # 문단 텍스트를 결합
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def load_documents(data_dir: str, metadata_csv: str | None = None) -> List[Document]:
    """
    load_documents는 데이터 디렉토리에서 문서를 로드

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

            # 텍스트 파일이면 텍스트 파일 읽기
            if ext in SUPPORTED_TEXT_EXTENSIONS:
                text = _read_text_file(path)
            # 바이너리 문서면 SV 텍스트를 확인
            elif ext in SUPPORTED_BINARY_EXTENSIONS:
                csv_text = metadata.get(CSV_TEXT_FIELD)
                if csv_text and csv_text.strip():
                    text = csv_text
                # PDF라면 PDF를 파싱
                elif ext == ".pdf":
                    text = extract_text_from_pdf(path)
                # HWP라면 HWP를 파싱
                elif ext == ".hwp":
                    text = extract_text_from_hwp(path)
                # DOCX라면 DOCX를 파싱
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


def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    simple_chunk는 고정 길이 기반 청킹을 수행

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

    # 결과 리스트를 준비
    chunks: List[str] = []
    # 시작 위치를 초기화
    start = 0
    # 텍스트 끝까지 반복
    while start < len(text):
        # 끝 위치를 계산
        end = min(len(text), start + chunk_size)
        # 청크를 추출
        chunk = text[start:end]
        # 공백만 있는 청크를 제외
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        # 겹침만큼 이동.
        start = max(0, end - overlap)
    return chunks


def chunk_documents(docs: Iterable[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    """
    chunk_documents는 문서 리스트를 청크 리스트로 변환

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
            # 청크 ID를 만든다
            chunk_id = f"{doc.id}::chunk_{i}"
            # 메타데이터를 복사
            metadata = dict(doc.metadata)
            # 청크 정보를 추가
            metadata.update({"chunk_index": i, "doc_id": doc.id})
            # 청크를 추가
            all_chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
    return all_chunks
