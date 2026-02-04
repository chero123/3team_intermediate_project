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
# 청킹 함수
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
                    output_name = safe_filename(doc_id, suffix="_chunked.txt")
                    output_path = os.path.join(output_dir, output_name)
                    with open(output_path, "w", encoding="utf-8") as f:
                        for chunk in chunks:
                            f.write(f"=== {chunk.id} ===\n")
                            f.write(chunk.text)
                            f.write("\n\n")
                    print(f"저장 완료: {output_path}\n")
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

    # 경로 설정
    INPUT_DIR = "data/original_data"
    OUTPUT_DIR = "data/parsing_data"

    # PDF 파서 설정
    # "pdfplumber": 텍스트 품질 좋음, 표 추출 강력 (느림)
    # "fitz": 속도 빠름 (2-5배)
    set_pdf_parser("pdfplumber")  # 기본값: pdfplumber

    # 청킹 설정 (테스트 후 지정)
    ENABLE_CHUNKING = False  # True: 청킹 수행, False: 파싱만
    CHUNK_SIZE = 1000  # 청크 크기 (글자 수)
    CHUNK_OVERLAP = 200  # 청크 간 겹침 (글자 수)

    print(f"PDF 파서: {PDF_PARSER}")

    # 전체 파일 처리
    parsed_docs = process_all_files(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        save_to_file=True,
        enable_chunking=ENABLE_CHUNKING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 사용 예시:
    # 1. 파싱만 (저장 O): ENABLE_CHUNKING = False
    # 2. 파싱+청킹 (저장 O): ENABLE_CHUNKING = True
    # 3. 파싱만 (저장 X): save_to_file=False, ENABLE_CHUNKING=False
    # 4. 파싱+청킹 (저장 X): save_to_file=False, ENABLE_CHUNKING=True
    # PDF 파서 변경: PDF_PARSER = "fitz" (속도 우선)

    elapsed = datetime.now() - start_time
    minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
    print(
        f"실행 문서: {os.path.basename(__file__)}, 완료 시간: {minutes:02d}분 {seconds:02d}초"
    )
