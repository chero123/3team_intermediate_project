import re
from typing import List
from dataclasses import dataclass


@dataclass
class Chunk:
    """청크 데이터 클래스"""

    id: str
    text: str  # 문맥(Header)이 포함된 최종 텍스트
    original_content: str  # 순수 본문
    metadata: dict


def is_section_header(line: str) -> bool:
    """RFP의 계층적 섹션 헤더 감지 (Ⅰ, 1., 가., □ 등)"""
    stripped = line.strip()
    if not stripped:
        return False
    patterns = [
        r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]",
        r"^#+",
        r"^\d+\.",
        r"^[가-힣]\.",
        r"^[\d가-힣]+\)",
        r"^□",
        r"^○",
    ]
    return any(re.match(p, stripped) for p in patterns)


def integrated_rfp_chunker(
    text: str, doc_id: str, context_metadata: dict = None
) -> List[Chunk]:
    """장팀원(계층 감지) + 김팀원(문서 정보 보강) 통합 청킹"""
    ctx = context_metadata or {}
    # 김팀원식 글로벌 헤더 (문서 식별)
    global_header = f"[[문서정보]]\n사업명: {ctx.get('title', doc_id)}\n발주기관: {ctx.get('agency', '미상')}\n\n"

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_section = "서론/기타"

    for i, para in enumerate(paragraphs):
        # 1. 장팀원식 섹션 추적
        lines = para.split("\n")
        first_line = lines[0] if lines else ""
        if is_section_header(first_line):
            current_section = first_line

        # 2. 문맥 보강 헤더 구성
        section_header = f"[[섹션정보]]\n위치: {current_section}\n\n"
        enriched_text = global_header + section_header + "[[본문]]\n" + para

        # 3. 메타데이터 구성
        chunk_metadata = {
            "doc_id": doc_id,
            "title": ctx.get("title", doc_id),
            "agency": ctx.get("agency", "미상"),
            "section": current_section,
            "page": ctx.get("page", "unknown"),
            "source_display": f"{ctx.get('title', doc_id)} (p.{ctx.get('page', '?')})",
            "method": "hybrid_jang_kim",
        }

        chunks.append(
            Chunk(
                id=f"{doc_id}::chunk{i}",
                text=enriched_text,
                original_content=para,
                metadata=chunk_metadata,
            )
        )
    return chunks
