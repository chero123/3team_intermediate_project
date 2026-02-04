# 라이브러리
import os
import glob
import zlib
import struct
import olefile
import re
import json
import hashlib
import unicodedata
from datetime import datetime
from typing import List, Literal
from dataclasses import dataclass

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


# ============================================
# PDF 파서 설정 (기본값: pdfplumber)
# ============================================
# "pdfplumber": 텍스트 품질 좋음, 표 추출 강력 (느림)
# "fitz": 속도 빠름 (2-5배)
PDF_PARSER: Literal["pdfplumber", "fitz"] = "pdfplumber"


# ============================================
# 데이터 클래스 정의
# ============================================


@dataclass
class Chunk:
    """청크 클래스"""

    id: str
    text: str
    metadata: dict


# ============================================
# 안전한 파일명 생성 함수
# ============================================


def safe_filename(
    original_name: str, suffix: str = "_parsed.txt", max_bytes: int = 180
) -> str:
    """
    OS 파일명 길이(바이트) 제한을 피하기 위해:
    - 원본 이름 정규화
    - 위험한 문자 제거
    - UTF-8 바이트 기준으로 자르기
    - 짧은 해시를 붙여 충돌 방지

    Args:
        original_name: 원본 파일명 (확장자 제외)
        suffix: 저장 파일 접미사 (기본값: "_parsed.txt")
        max_bytes: 최대 바이트 수 (기본값: 180)

    Returns:
        안전하게 변환된 파일명
    """
    name = unicodedata.normalize("NFC", original_name)

    # 확장자 제거 (혹시 .hwp/.pdf가 포함되어 있어도 안전하게)
    base = re.sub(r"\.(hwp|pdf)$", "", name, flags=re.IGNORECASE)

    # 파일명에 들어가면 곤란한 문자 제거/치환 (윈도우 금지문자 포함)
    base = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()

    # 충돌 방지용 짧은 해시
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]

    # suffix 고려해서 base를 바이트 기준으로 자르기
    # 최종 파일명: "{base}__{hash}{suffix}"
    tail = f"__{h}{suffix}"
    budget = max_bytes - len(tail.encode("utf-8"))
    if budget < 20:
        budget = 20

    b = base.encode("utf-8")
    if len(b) > budget:
        base = b[:budget].decode("utf-8", errors="ignore").rstrip()

    return f"{base}__{h}{suffix}"


# ============================================
# 텍스트 정제 함수
# ============================================


def clean_text(text: str) -> str:
    """텍스트 정제: 불필요한 공백 및 과도한 줄바꿈 제거"""
    if not text:
        return ""
    # 깨진 문자 제거 (한글, 영문, 숫자, 공백, 로마숫자, 기본 문장부호 유지)
    text = re.sub(
        r'[^\uAC00-\uD7A3\u2160-\u217Fa-zA-Z0-9\s.,!?():\-\[\]<>~·%/@\'"_=+○●■□▶◆※]',
        "",
        text,
    )
    # 연속 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ============================================
# HWP 파싱 함수
# ============================================


def parse_hwp_all(file_path: str) -> str:
    """HWP 파일에서 전체 텍스트를 추출합니다. (BodyText 파싱)"""
    if not os.path.exists(file_path):
        print(f"'{file_path}' 파일을 찾을 수 없습니다.")
        return ""
    try:
        f = olefile.OleFileIO(file_path)

        # 압축 여부 확인
        header = f.openstream("FileHeader").read()
        is_compressed = header[36] & 1

        texts = []

        # 모든 BodyText 섹션 처리
        for entry in f.listdir():
            if entry[0] == "BodyText":
                data = f.openstream(entry).read()

                # 압축 해제
                if is_compressed:
                    try:
                        data = zlib.decompress(data, -15)
                    except:
                        pass

                # 레코드 파싱
                i = 0
                while i < len(data):
                    if i + 4 > len(data):
                        break

                    rec_header = struct.unpack_from("<I", data, i)[0]
                    rec_type = rec_header & 0x3FF
                    rec_len = (rec_header >> 20) & 0xFFF

                    # 확장 길이 처리
                    if rec_len == 0xFFF:
                        if i + 8 > len(data):
                            break
                        rec_len = struct.unpack_from("<I", data, i + 4)[0]
                        i += 8
                    else:
                        i += 4

                    if i + rec_len > len(data):
                        break

                    # HWPTAG_PARA_TEXT (67)
                    if rec_type == 67 and rec_len > 0:
                        text_data = data[i : i + rec_len]
                        try:
                            text = text_data.decode("utf-16le", errors="ignore")
                            # HWP 제어문자 제거 (0x00~0x1F 범위의 특수 제어코드)
                            cleaned = []
                            for char in text:
                                code = ord(char)
                                if code >= 32 or char in "\n\r\t":
                                    cleaned.append(char)
                                elif code in [13, 10]:  # CR, LF
                                    cleaned.append("\n")
                            text = "".join(cleaned).strip()
                            if text:
                                texts.append(text)
                        except:
                            pass

                    i += rec_len

        f.close()

        if texts:
            result = "\n".join(texts)
            result = clean_text(result)
            print(f"'{file_path}' 파일 파싱 성공!")
            return result
        else:
            print(f"'{file_path}' 텍스트 추출 실패")
            return ""

    except Exception as e:
        print(f"'{file_path}' 파일 파싱 중 오류 발생: {e}")
        return ""


# ============================================
# PDF 파싱 함수
# ============================================


def parse_pdf_with_pdfplumber(file_path: str) -> str:
    """pdfplumber를 이용한 PDF 텍스트 추출 (텍스트 품질 좋음)"""
    if pdfplumber is None:
        print("pdfplumber가 설치되어 있지 않습니다.")
        return ""

    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"  총 {len(pdf.pages)}페이지 추출 중... (pdfplumber)")
            text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            if text:
                result = clean_text(text)
                print(f"'{file_path}' 파일 파싱 성공!")
                return result
            else:
                print(f"'{file_path}' 텍스트 추출 실패")
                return ""
    except Exception as e:
        print(f"'{file_path}' PDF 파싱 중 오류 발생: {e}")
        return ""


def parse_pdf_with_fitz(file_path: str) -> str:
    """PyMuPDF(fitz)를 이용한 PDF 텍스트 추출 (속도 빠름)"""
    if fitz is None:
        print("PyMuPDF가 설치되어 있지 않습니다.")
        return ""

    try:
        doc = fitz.open(file_path)
        results = []

        print(f"  총 {len(doc)}페이지 추출 중... (fitz)")

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                results.append(f"[페이지 {page_num + 1}]\n{page_text.strip()}")

        doc.close()

        if results:
            result = "\n\n".join(results)
            result = clean_text(result)
            print(f"'{file_path}' 파일 파싱 성공!")
            return result
        else:
            print(f"'{file_path}' 텍스트 추출 실패")
            return ""

    except Exception as e:
        print(f"'{file_path}' PDF 파싱 중 오류 발생: {e}")
        return ""


def parse_pdf(file_path: str, parser: str = None) -> str:
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        file_path: PDF 파일 경로
        parser: "pdfplumber" 또는 "fitz" (None이면 전역 설정 사용)

    Returns:
        str: 추출된 텍스트
    """
    if not os.path.exists(file_path):
        print(f"'{file_path}' 파일을 찾을 수 없습니다.")
        return ""

    # 파서 선택 (매개변수 > 전역 설정)
    selected_parser = parser or PDF_PARSER

    if selected_parser == "pdfplumber":
        return parse_pdf_with_pdfplumber(file_path)
    else:
        return parse_pdf_with_fitz(file_path)


# ============================================
# 확장자 자동 감지 함수
# ============================================


def load_file_content(file_path: str) -> str:
    """확장자 자동 감지 및 추출"""
    ext = file_path.split(".")[-1].lower()
    if ext == "hwp":
        return parse_hwp_all(file_path)
    elif ext == "pdf":
        return parse_pdf(file_path)
    return ""


# ============================================
# 청킹 함수 (기본)
# ============================================


def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    고정 길이 기반 청킹을 수행

    Args:
        text: 입력 텍스트
        chunk_size: 청크 길이
        overlap: 청크 겹침 길이

    Returns:
        List[str]: 청크 문자열 리스트
    """
    if chunk_size <= 0:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def chunk_text(
    text: str, doc_id: str, chunk_size: int, overlap: int, metadata: dict = None
) -> List[Chunk]:
    """
    텍스트를 청크 리스트로 변환

    Args:
        text: 입력 텍스트
        doc_id: 문서 ID
        chunk_size: 청크 길이
        overlap: 청크 겹침 길이
        metadata: 추가 메타데이터

    Returns:
        List[Chunk]: 청크 리스트
    """
    chunks: List[Chunk] = []
    pieces = simple_chunk(text, chunk_size, overlap)

    for i, piece in enumerate(pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        chunk_metadata = dict(metadata) if metadata else {}
        chunk_metadata.update({"chunk_index": i, "doc_id": doc_id})
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=chunk_metadata))

    return chunks


# ============================================
# 팀원별 청킹 함수
# ============================================


# 1. 안팀원 - RecursiveCharacterTextSplitter
def chunk_an(
    text: str,
    doc_id: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    metadata: dict = None,
) -> List[Chunk]:
    """
    안팀원 방식: RecursiveCharacterTextSplitter 기반 청킹
    - langchain 라이브러리 활용
    - 문단 → 줄 → 문장부호 → 공백 순으로 분할
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "□", "。", ".", "!", "?", " ", ""],
        )
        pieces = [c for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        print("  [경고] langchain-text-splitters 미설치, 기본 청킹 사용")
        pieces = simple_chunk(text, chunk_size, overlap)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        chunk_metadata = dict(metadata) if metadata else {}
        chunk_metadata.update(
            {"chunk_index": i, "doc_id": doc_id, "method": "an_recursive"}
        )
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=chunk_metadata))
    return chunks


# 2. 박팀원 - 문단 기반 청킹
def chunk_park(
    text: str,
    doc_id: str,
    min_chars: int = 200,
    max_chars: int = 800,
    overlap: int = 100,
    metadata: dict = None,
) -> List[Chunk]:
    """
    박팀원 방식: 문단 기반 적응형 청킹
    - 빈 줄 기준으로 문단 분리
    - 짧은 문단 합치기, 긴 문단 분할
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks = []

    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) <= max_chars:
            buffer += ("\n\n" + p) if buffer else p
        else:
            if len(buffer) >= min_chars:
                raw_chunks.append(buffer)
                buffer = p
            else:
                buffer += "\n\n" + p

    if buffer:
        raw_chunks.append(buffer)

    # 긴 청크 재분할 + overlap
    final_pieces = []
    for c in raw_chunks:
        if len(c) <= max_chars:
            final_pieces.append(c)
        else:
            start = 0
            while start < len(c):
                end = start + max_chars
                final_pieces.append(c[start:end])
                start = end - overlap

    chunks = []
    for i, piece in enumerate(final_pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        chunk_metadata = dict(metadata) if metadata else {}
        chunk_metadata.update(
            {"chunk_index": i, "doc_id": doc_id, "method": "park_paragraph"}
        )
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=chunk_metadata))
    return chunks


# 3. 서팀원 - SemanticChunker (의미론적 청킹)
def chunk_seo(
    text: str,
    doc_id: str,
    metadata: dict = None,
) -> List[Chunk]:
    """
    서팀원 방식: SemanticChunker 기반 의미론적 청킹
    - 임베딩 모델로 문장 간 유사도 분석
    - 의미 경계에서 분할
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("  [서팀원] 임베딩 모델 로딩 중...")
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        pieces = splitter.split_text(text)
        pieces = [p for p in pieces if p.strip()]
    except ImportError as e:
        print(f"  [경고] SemanticChunker 의존성 미설치: {e}")
        print(
            "  [경고] pip install langchain-experimental langchain-community sentence-transformers"
        )
        pieces = simple_chunk(text, 1000, 200)
    except Exception as e:
        print(f"  [경고] SemanticChunker 실패: {e}, 기본 청킹 사용")
        pieces = simple_chunk(text, 1000, 200)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        chunk_metadata = dict(metadata) if metadata else {}
        chunk_metadata.update(
            {"chunk_index": i, "doc_id": doc_id, "method": "seo_semantic"}
        )
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=chunk_metadata))
    return chunks


# 4. 김팀원 - Context Enrichment + 청킹
def chunk_kim(
    text: str,
    doc_id: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    metadata: dict = None,
    context_metadata: dict = None,
) -> List[Chunk]:
    """
    김팀원 방식: Context Enrichment + RecursiveCharacterTextSplitter
    - 청크 앞에 사업 개요 메타데이터 주입
    - 검색 시 문맥 유지
    """
    # Context Enrichment (메타데이터 주입)
    ctx = context_metadata or {}
    enriched_header = f"""[[사업 개요]]
사업명: {ctx.get('title', doc_id)}
발주기관: {ctx.get('agency', '-')}
공고번호: {ctx.get('notice_id', '-')}

[[본문]]
"""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        pieces = [c for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        print("  [경고] langchain-text-splitters 미설치, 기본 청킹 사용")
        pieces = simple_chunk(text, chunk_size, overlap)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        # 각 청크에 enriched_header 추가
        enriched_text = enriched_header + piece
        chunk_metadata = dict(metadata) if metadata else {}
        chunk_metadata.update(
            {
                "chunk_index": i,
                "doc_id": doc_id,
                "method": "kim_context_enrichment",
                "has_context_header": True,
            }
        )
        if context_metadata:
            chunk_metadata.update(context_metadata)
        chunks.append(Chunk(id=chunk_id, text=enriched_text, metadata=chunk_metadata))
    return chunks


# 5. 장팀원 - 계층 구조 기반 청킹
def chunk_jang(
    text: str,
    doc_id: str,
    chunk_size: int = 1000,
    overlap_ratio: float = 0.2,
    min_chunk_size: int = 200,
    metadata: dict = None,
) -> List[Chunk]:
    """
    장팀원 방식: HierarchicalChunkerV2 기반 계층 구조 청킹
    - 로마숫자(Ⅰ) → 숫자(1.) → 가나다(가.) 인식
    - 테이블 자동 감지
    - 계층 경로 추적
    """
    try:
        from hierarchical_chunker_v2 import HierarchicalChunkerV2

        chunker = HierarchicalChunkerV2(
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
            min_chunk_size=min_chunk_size,
        )
        h_chunks = chunker.chunk_document(text, doc_id, metadata)

        # HierarchicalChunkerV2의 Chunk를 현재 Chunk 형식으로 변환
        chunks = []
        for hc in h_chunks:
            chunk_metadata = hc.metadata.copy() if hc.metadata else {}
            chunk_metadata["method"] = "jang_hierarchical"
            chunk_metadata["tables_count"] = len(hc.tables) if hc.tables else 0
            chunks.append(Chunk(id=hc.chunk_id, text=hc.text, metadata=chunk_metadata))
        return chunks

    except ImportError:
        print("  [경고] hierarchical_chunker_v2 모듈 없음, 기본 청킹 사용")
        return chunk_text(
            text, doc_id, chunk_size, int(chunk_size * overlap_ratio), metadata
        )


# ============================================
# 전체 청킹 비교 테스트 함수
# ============================================


def run_chunking_comparison(
    input_dir: str = "data/parsing_data",
    base_output_dir: str = "data",
    chunk_size: int = 1000,
    overlap: int = 200,
):
    """
    모든 팀원의 청킹 방식을 비교 테스트하고 각각 다른 폴더에 저장

    Args:
        input_dir: 파싱된 텍스트 파일 디렉토리
        base_output_dir: 출력 기본 디렉토리
        chunk_size: 청크 크기
        overlap: 오버랩 크기

    Returns:
        dict: 각 방식별 통계
    """
    import glob as glob_module

    # 파싱된 파일 수집
    parsed_files = glob_module.glob(f"{input_dir}/*_parsed.txt")
    if not parsed_files:
        print(f"[오류] {input_dir}에 파싱된 파일이 없습니다.")
        return {}

    print(f"발견된 파싱 파일: {len(parsed_files)}개\n")

    # 청킹 방식 정의
    # 참고: chunk_park, chunk_seo, chunk_jang 함수는 run_chunking_comparison에 전달된
    # chunk_size, overlap 파라미터를 완전히 따르지 않고 자체 내부 로직 또는 고정 파라미터를 사용할 수 있습니다.
    chunking_methods = [
        {
            "folder": "chunking_data1",
            "name": "안팀원-RecursiveCharacterTextSplitter",
            "func": lambda t, d, m: chunk_an(t, d, chunk_size, overlap, m),
        },
        {
            "folder": "chunking_data2",
            "name": "박팀원-문단기반청킹",
            "func": lambda t, d, m: chunk_park(t, d, 200, 800, 100, m),
        },
        {
            "folder": "chunking_data3",
            "name": "서팀원-SemanticChunker",
            "func": lambda t, d, m: chunk_seo(t, d, m),
        },
        {
            "folder": "chunking_data4",
            "name": "김팀원-ContextEnrichment",
            "func": lambda t, d, m: chunk_kim(
                t, d, chunk_size, overlap, m, {"title": d}
            ),
        },
        {
            "folder": "chunking_data5",
            "name": "장팀원-HierarchicalChunker",
            "func": lambda t, d, m: chunk_jang(t, d, chunk_size, 0.2, 200, m),
        },
    ]

    all_stats = {}

    for method in chunking_methods:
        folder = method["folder"]
        name = method["name"]
        chunk_func = method["func"]

        output_dir = os.path.join(base_output_dir, folder)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{name}] 청킹 시작")
        print(f"출력 폴더: {output_dir}")
        print("=" * 60)

        stats = {
            "method": name,
            "total_files": len(parsed_files),
            "success_count": 0,
            "fail_count": 0,
            "total_chunks": 0,
            "files": [],
        }

        for file_path in parsed_files:
            file_name = os.path.basename(file_path)
            doc_id = file_name.replace("_parsed.txt", "")

            print(f"  처리 중: {file_name[:50]}...")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                metadata = {
                    "source_file": file_name,
                    "source_path": file_path,
                    "total_chars": len(text),
                }

                # 청킹 수행
                chunks = chunk_func(text, doc_id, metadata)

                # JSON 저장
                output_data = {
                    "doc_id": doc_id,
                    "source_file": file_name,
                    "method": name,
                    "total_chunks": len(chunks),
                    "chunks": [
                        {"id": c.id, "text": c.text, "metadata": c.metadata}
                        for c in chunks
                    ],
                }

                output_name = doc_id + "_chunked.json"
                output_path = os.path.join(output_dir, output_name)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                print(f"    → {len(chunks)}개 청크 생성")

                stats["success_count"] += 1
                stats["total_chunks"] += len(chunks)
                stats["files"].append({"file": file_name, "chunks": len(chunks)})

            except Exception as e:
                print(f"    → 오류: {e}")
                stats["fail_count"] += 1

        # 방식별 통계 저장
        stats_path = os.path.join(output_dir, "chunking_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(
            f"\n[{name}] 완료: 성공 {stats['success_count']}개, 총 청크 {stats['total_chunks']}개"
        )

        all_stats[folder] = stats

    # 전체 비교 요약
    print("\n" + "=" * 60)
    print("전체 비교 요약")
    print("=" * 60)
    print(f"{'방식':<40} {'청크 수':>10}")
    print("-" * 52)
    for folder, stats in all_stats.items():
        print(f"{stats['method']:<40} {stats['total_chunks']:>10}")

    # 전체 통계 저장
    summary_path = os.path.join(base_output_dir, "chunking_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n전체 요약 저장: {summary_path}")

    return all_stats


# ============================================
# 전체 파일 처리 함수
# ============================================


def process_all_files(
    input_dir: str = "data/original_data",
    output_dir: str = "data/parsing_data",
    save_to_file: bool = True,
    enable_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """
    입력 디렉토리의 모든 HWP, PDF 파일을 파싱합니다.

    Args:
        input_dir: 원본 파일 디렉토리
        output_dir: 파싱 결과 저장 디렉토리
        save_to_file: True이면 파일로 저장, False이면 저장하지 않음
        enable_chunking: True이면 청킹 수행
        chunk_size: 청크 크기 (글자 수)
        chunk_overlap: 청크 간 겹침 (글자 수)

    Returns:
        dict: {파일명: 추출된텍스트} 또는 {파일명: List[Chunk]} 형태의 딕셔너리
    """
    # 출력 폴더 생성 (저장할 경우에만)
    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)

    # HWP, PDF 파일 수집
    hwp_files = glob.glob(f"{input_dir}/*.hwp")
    pdf_files = glob.glob(f"{input_dir}/*.pdf")
    all_files = hwp_files + pdf_files

    print(f"발견된 파일: HWP {len(hwp_files)}개, PDF {len(pdf_files)}개")
    print(f"총 {len(all_files)}개 파일 변환 시작...")
    if enable_chunking:
        print(f"청킹 설정: chunk_size={chunk_size}, overlap={chunk_overlap}")
    print()

    parsed_docs = {}
    mapping = []  # 원본파일명 ↔ 저장파일명 매핑 (RAG 메타데이터용)
    success_count = 0
    fail_count = 0
    total_chunk_count = 0

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        doc_id = file_name.rsplit(".", 1)[0]
        print(f"--- {file_name} ---")

        # 확장자에 따라 파싱
        content = load_file_content(file_path)

        if content:
            print(f"[추출된 텍스트 미리보기]")
            print(content[:300])
            print("...\n")

            # 청킹 수행 (옵션)
            if enable_chunking:
                chunks = chunk_text(
                    text=content,
                    doc_id=doc_id,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    metadata={"filename": file_name, "source_path": file_path},
                )
                parsed_docs[file_name] = chunks
                total_chunk_count += len(chunks)
                print(f"  청크 생성: {len(chunks)}개")
            else:
                parsed_docs[file_name] = content

            # 파일 저장 (옵션) - safe_filename 적용
            if save_to_file:
                if enable_chunking:
                    # 청크 파일 저장
                    output_data = {
                        "doc_id": doc_id,
                        "source_file": file_name,
                        "method": "simple_chunk", # process_all_files는 기본 청킹만 수행
                        "total_chunks": len(chunks),
                        "chunks": [
                            {"id": c.id, "text": c.text, "metadata": c.metadata}
                            for c in chunks
                        ],
                    }
                    output_name = safe_filename(doc_id, suffix="_chunked.json")
                    output_path = os.path.join(output_dir, output_name)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                    print(f"저장 완료: {output_path} ({len(chunks)}개 청크)\n")
                else:
                    # 파싱 파일 저장
                    output_name = safe_filename(doc_id, suffix="_parsed.txt")
                    output_path = os.path.join(output_dir, output_name)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"저장 완료: {output_path} ({len(content)}자)\n")

                # 매핑 정보 기록
                mapping.append(
                    {
                        "original_filename": file_name,
                        "source_path": file_path,
                        "saved_filename": output_name,
                        "saved_path": output_path,
                        "chars": len(content),
                    }
                )

            success_count += 1
        else:
            print("텍스트 추출 실패\n")
            fail_count += 1

    print(f"\n{'='*50}")
    print(
        f"변환 완료: 성공 {success_count}개 / 실패 {fail_count}개 (총 {len(all_files)}개)"
    )
    if enable_chunking:
        print(f"총 청크 수: {total_chunk_count}개")
    if save_to_file:
        print(f"저장 위치: {output_dir}/")

        # 매핑 파일 저장 (원본 ↔ 저장 파일명 추적용)
        if mapping:
            mapping_path = os.path.join(output_dir, "parsed_mapping.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"매핑 저장: {mapping_path}")

    return parsed_docs


# ============================================
# 메인 실행
# ============================================


def set_pdf_parser(parser: str):
    """PDF 파서를 설정합니다."""
    global PDF_PARSER
    PDF_PARSER = parser


if __name__ == "__main__":
    start_time = datetime.now()

    # ============================================
    # 실행 모드 설정
    # ============================================
    # "parse": 파싱만 수행
    # "compare": 팀원별 청킹 비교 테스트
    RUN_MODE = "compare"  # 변경하여 실행 모드 선택

    if RUN_MODE == "compare":
        # ============================================
        # 팀원별 청킹 비교 테스트
        # ============================================
        print("=" * 60)
        print("팀원별 청킹 방식 비교 테스트")
        print("=" * 60)
        print("chunking_data1: 안팀원 - RecursiveCharacterTextSplitter")
        print("chunking_data2: 박팀원 - 문단 기반 청킹")
        print("chunking_data3: 서팀원 - SemanticChunker")
        print("chunking_data4: 김팀원 - Context Enrichment")
        print("chunking_data5: 장팀원 - HierarchicalChunker")
        print("=" * 60)

        stats = run_chunking_comparison(
            input_dir="data/parsing_data",
            base_output_dir="data",
            chunk_size=1000,
            overlap=200,
        )

    else:
        # ============================================
        # 기존 파싱 모드
        # ============================================
        INPUT_DIR = "data/original_data"
        OUTPUT_DIR = "data/parsing_data"

        # PDF 파서 설정
        set_pdf_parser("pdfplumber")

        # 청킹 설정
        ENABLE_CHUNKING = False
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200

        print(f"PDF 파서: {PDF_PARSER}")

        parsed_docs = process_all_files(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            save_to_file=True,
            enable_chunking=ENABLE_CHUNKING,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    # 사용 예시:
    # 1. 파싱만: RUN_MODE = "parse", ENABLE_CHUNKING = False
    # 2. 파싱+청킹: RUN_MODE = "parse", ENABLE_CHUNKING = True
    # 3. 팀원별 비교: RUN_MODE = "compare"

    elapsed = datetime.now() - start_time
    minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
    print(
        f"\n실행 문서: {os.path.basename(__file__)}, 완료 시간: {minutes:02d}분 {seconds:02d}초"
    )
