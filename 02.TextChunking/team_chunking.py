# ============================================
# team_chunking.py - 팀원별 청킹 비교 테스트
# ============================================
# 역할: 청킹만 담당 (파싱은 text_parsing.py 사용)
#
# 워크플로우:
#   1. text_parsing.py로 파싱 → data/parsing_data_openai/*.md (또는 .txt)
#   2. team_chunking.py로 청킹 비교 → data/chunking_data*/*.json
#
# 사용법:
#   python text_parsing.py   # 먼저 파싱 실행
#   python team_chunking.py  # 청킹 비교 실행
# ============================================

# 라이브러리
import os
import glob
import re
import json
import hashlib
import unicodedata
from datetime import datetime
from typing import List
from dataclasses import dataclass


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
    """
    name = unicodedata.normalize("NFC", original_name)
    base = re.sub(r"\.(hwp|pdf)$", "", name, flags=re.IGNORECASE)
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


# ============================================
# 청킹 함수 (기본)
# ============================================


def simple_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """고정 길이 기반 청킹을 수행"""
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
    """텍스트를 청크 리스트로 변환"""
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
    """안팀원 방식: RecursiveCharacterTextSplitter 기반 청킹"""
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
    """박팀원 방식: 문단 기반 적응형 청킹"""
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
    """서팀원 방식: SemanticChunker 기반 의미론적 청킹"""
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
    """김팀원 방식: Context Enrichment + RecursiveCharacterTextSplitter"""
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


# 5. 장팀원 - 계층적 섹션 기반 청킹 (Hierarchical Section-Aware Chunking)
def _is_markdown_table_line(line: str) -> bool:
    """마크다운 표 라인인지 확인"""
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_section_header(line: str) -> bool:
    """섹션 헤더인지 확인 (Ⅰ, Ⅱ, 1., 가. 등)"""
    stripped = line.strip()
    if not stripped:
        return False

    # 로마숫자 섹션 (Ⅰ, Ⅱ, Ⅲ, Ⅳ, Ⅴ 등)
    if stripped and stripped[0] in "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ":
        return True

    # 마크다운 헤더 (#, ##, ### 등)
    if stripped.startswith("#"):
        return True

    # 숫자 섹션 (1., 2., 10. 등) - 단, 너무 긴 것은 제외
    if len(stripped) < 50 and re.match(r"^\d+\.\s+\S", stripped):
        return True

    # 한글 섹션 (가., 나., 다. 등)
    if len(stripped) < 50 and re.match(r"^[가-힣]\.\s+\S", stripped):
        return True

    # 괄호 섹션 (1), 2), 가), 나) 등)
    if len(stripped) < 50 and re.match(r"^[\d가-힣]+\)\s+\S", stripped):
        return True

    return False


def _force_split_large_text(text: str, max_size: int, overlap: int = 100) -> List[str]:
    """
    큰 텍스트를 max_size 이내로 강제 분할
    가능하면 문장 경계에서, 안되면 단어 경계에서, 최후에는 문자 경계에서 분할
    """
    if len(text) <= max_size:
        return [text]

    pieces = []
    remaining = text

    while len(remaining) > max_size:
        # max_size 위치에서 분할점 찾기
        split_pos = max_size

        # 1순위: 문장 경계 (. ! ? 。) 찾기
        for end_char in [". ", "! ", "? ", "。", ".\n", "!\n", "?\n"]:
            pos = remaining[:max_size].rfind(end_char)
            if pos > max_size // 2:  # 절반 이상 위치에서만
                split_pos = pos + len(end_char)
                break
        else:
            # 2순위: 줄바꿈 찾기
            pos = remaining[:max_size].rfind("\n")
            if pos > max_size // 2:
                split_pos = pos + 1
            else:
                # 3순위: 공백 찾기
                pos = remaining[:max_size].rfind(" ")
                if pos > max_size // 2:
                    split_pos = pos + 1
                # 4순위: 그냥 max_size에서 자르기 (기본값 사용)

        pieces.append(remaining[:split_pos].strip())
        # 오버랩 적용: 이전 청크의 마지막 부분을 포함
        overlap_start = max(0, split_pos - overlap)
        remaining = remaining[overlap_start:].strip()

    if remaining.strip():
        pieces.append(remaining.strip())

    return pieces


def chunk_jang(
    text: str,
    doc_id: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    min_chunk_size: int = 200,
    metadata: dict = None,
) -> List[Chunk]:
    """
    장팀원 방식: 계층적 섹션 기반 청킹 (Hierarchical Section-Aware Chunking)

    특징:
    1. 섹션 헤더 감지하여 각 청크에 컨텍스트로 포함
    2. 어떤 청크도 chunk_size를 초과하지 않도록 강제 분할
    3. 문장/단어 경계에서 자연스럽게 분할
    4. 오버랩 적용으로 문맥 연결성 유지

    Args:
        text: 입력 텍스트
        doc_id: 문서 ID
        chunk_size: 최대 청크 크기 (기본 1000자)
        overlap: 오버랩 크기 (기본 100자)
        min_chunk_size: 최소 청크 크기 (기본 200자)
        metadata: 추가 메타데이터
    """
    # 1단계: 문단 단위로 분리
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    # 2단계: 섹션 헤더 추적하며 청크 생성
    final_pieces = []
    buffer = ""
    current_section_header = ""  # 현재 섹션 헤더 (컨텍스트용)

    for para in paragraphs:
        # 섹션 헤더 감지
        first_line = para.split("\n")[0] if para else ""
        if _is_section_header(first_line):
            # 이전 버퍼 저장
            if buffer.strip() and len(buffer.strip()) >= min_chunk_size:
                final_pieces.append(buffer.strip())
            elif buffer.strip():
                # 너무 작은 버퍼는 다음과 병합될 수 있도록 유지
                pass
            else:
                buffer = ""

            # 새 섹션 시작
            current_section_header = first_line
            buffer = para
            continue

        # 마크다운 표나 특수 블록 감지
        is_table = any(_is_markdown_table_line(line) for line in para.split("\n"))
        is_special = (
            para.startswith("[PAGE ")
            or para.startswith("[페이지 ")
            or para.startswith("[VLM ")
            or para.startswith("**[표]")
        )

        if is_table or is_special:
            # 이전 버퍼 저장
            if buffer.strip():
                final_pieces.append(buffer.strip())
                buffer = ""

            # 특수 블록은 크기 초과 시 강제 분할
            if len(para) <= chunk_size:
                final_pieces.append(para)
            else:
                # 섹션 헤더 포함하여 분할
                header_prefix = (
                    f"[{current_section_header}]\n\n" if current_section_header else ""
                )
                available_size = chunk_size - len(header_prefix)
                sub_pieces = _force_split_large_text(para, available_size, overlap)
                for sp in sub_pieces:
                    final_pieces.append(header_prefix + sp if header_prefix else sp)
            continue

        # 일반 텍스트: 버퍼에 추가하거나 분할
        potential_size = len(buffer) + len(para) + 2 if buffer else len(para)

        if potential_size <= chunk_size:
            # 버퍼에 추가
            buffer = (buffer + "\n\n" + para) if buffer else para
        else:
            # 버퍼가 충분히 크면 저장
            if len(buffer) >= min_chunk_size:
                final_pieces.append(buffer.strip())
                buffer = ""

            # 현재 문단이 chunk_size보다 크면 강제 분할
            if len(para) > chunk_size:
                header_prefix = (
                    f"[{current_section_header}]\n\n" if current_section_header else ""
                )
                available_size = chunk_size - len(header_prefix)
                sub_pieces = _force_split_large_text(para, available_size, overlap)
                for sp in sub_pieces:
                    final_pieces.append(header_prefix + sp if header_prefix else sp)
            else:
                # 섹션 헤더를 포함하여 새 버퍼 시작
                if current_section_header and not para.startswith(
                    current_section_header
                ):
                    header_prefix = f"[{current_section_header}]\n\n"
                    if len(header_prefix) + len(para) <= chunk_size:
                        buffer = header_prefix + para
                    else:
                        buffer = para
                else:
                    buffer = para

    # 마지막 버퍼 저장
    if buffer.strip():
        final_pieces.append(buffer.strip())

    # 3단계: 최종 검증 - 모든 청크가 chunk_size 이내인지 확인
    validated_pieces = []
    for piece in final_pieces:
        if len(piece) <= chunk_size:
            validated_pieces.append(piece)
        else:
            # 여전히 큰 청크는 강제 분할
            sub_pieces = _force_split_large_text(piece, chunk_size, overlap)
            validated_pieces.extend(sub_pieces)

    # 4단계: Chunk 객체로 변환
    chunks = []
    for i, piece in enumerate(validated_pieces):
        chunk_id = f"{doc_id}::chunk{i}"
        chunk_metadata = dict(metadata) if metadata else {}

        # 블록 타입 감지
        has_table = any(_is_markdown_table_line(line) for line in piece.split("\n"))
        has_image = "[PAGE " in piece or "[페이지 " in piece
        has_section_context = piece.startswith("[") and "]\n\n" in piece[:100]

        chunk_metadata.update(
            {
                "chunk_index": i,
                "doc_id": doc_id,
                "method": "jang_hierarchical_section",
                "has_table": has_table,
                "has_image": has_image,
                "has_section_context": has_section_context,
            }
        )
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=chunk_metadata))

    return chunks


# ============================================
# 전체 청킹 비교 테스트 함수
# ============================================


def run_chunking_comparison(
    input_dir: str = "data/parsing_data_openai",
    base_output_dir: str = "data",
    chunk_size: int = 1000,
    overlap: int = 200,
):
    """
    모든 팀원의 청킹 방식을 비교 테스트하고 각각 다른 폴더에 저장

    Args:
        input_dir: 파싱된 텍스트 파일 디렉토리 (.txt 또는 .md)
        base_output_dir: 출력 기본 디렉토리
        chunk_size: 청크 크기
        overlap: 오버랩 크기

    Returns:
        dict: 각 방식별 통계
    """
    # 파싱된 파일 수집 (.txt 와 .md 모두 지원)
    txt_files = glob.glob(f"{input_dir}/*_parsed.txt")
    md_files = glob.glob(f"{input_dir}/*_parsed.md")
    parsed_files = txt_files + md_files

    if not parsed_files:
        print(f"[오류] {input_dir}에 파싱된 파일이 없습니다.")
        print("  먼저 text_parsing.py를 실행하세요.")
        return {}

    print(
        f"발견된 파싱 파일: {len(parsed_files)}개 (.txt: {len(txt_files)}, .md: {len(md_files)})\n"
    )

    # 청킹 방식 정의
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
            "func": lambda t, d, m: chunk_jang(t, d, chunk_size, overlap, 200, m),
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
            # .txt 또는 .md 확장자 모두 처리
            doc_id = re.sub(r"_parsed\.(txt|md)$", "", file_name)

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
# 메인 실행
# ============================================

if __name__ == "__main__":
    start_time = datetime.now()

    # ============================================
    # 청킹 비교 테스트 실행
    # ============================================
    # 먼저 text_parsing.py로 파싱을 완료한 후 실행하세요.
    #
    # 파싱: python text_parsing.py
    # 청킹: python team_chunking.py
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

    # 파싱된 파일 디렉토리 (text_parsing.py 출력 경로와 일치해야 함)
    INPUT_DIR = "data/parsing_data_openai"

    stats = run_chunking_comparison(
        input_dir=INPUT_DIR,
        base_output_dir="data",
        chunk_size=1000,
        overlap=200,
    )

    elapsed = datetime.now() - start_time
    minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
    print(
        f"\n실행 문서: {os.path.basename(__file__)}, 완료 시간: {minutes:02d}분 {seconds:02d}초"
    )
