# extract_tables_to_md.py
# ============================================
# PDF 전체 → VLM 표 추출 → (이미 존재하는) MD에 페이지별 삽입
# - md가 없으면 생성하지 않음 (에러 처리)
# - 재실행 시 같은 페이지 표 블록을 교체(중복 방지)
# ============================================

import base64
import json
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------- 경로 설정 --------
PDF_DIR = Path("pdf_out")
MD_DIR = Path("final_docs")

# -------- OpenAI 설정 --------
client = OpenAI()
MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """
너는 문서 페이지에서 '표'만 추출하는 도우미다.

규칙:
- 보이는 표만 추출한다 (추측 금지)
- 표는 Markdown table로 변환한다
- 표가 없으면 {"tables": []} 만 반환한다
- 설명 문장은 쓰지 않는다
- JSON만 출력한다

출력 형식:
{
  "tables": [
    {
      "caption": "표 제목 (없으면 빈 문자열)",
      "markdown": "| ... |"
    }
  ]
}
"""

# -------- 유틸: 안전 JSON 파싱 --------
def safe_json_loads(s: str) -> dict:
    """모델이 ```json ... ``` 같은 형식으로 줘도 최대한 JSON만 뽑아 파싱."""
    s = (s or "").strip()
    s = re.sub(r"^```json\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


# -------- 유틸: 페이지를 data URL로 --------
def page_to_data_url(page, zoom: float = 2.0) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -------- VLM: 페이지에서 표 추출 --------
def extract_tables_from_page(
    page,
    page_no: int,
    zoom: float = 2.0,
    retry: int = 2,
    sleep_sec: float = 1.0,
) -> dict:
    img_url = page_to_data_url(page, zoom=zoom)

    last_err = None
    for attempt in range(retry + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"{page_no}페이지에서 표만 추출해."},
                            {"type": "input_image", "image_url": img_url},
                        ],
                    },
                ],
            )
            return safe_json_loads(resp.output_text)
        except Exception as e:
            last_err = e
            # 레이트리밋/일시 오류 대비
            time.sleep(sleep_sec * (attempt + 1))

    raise last_err


# -------- MD에 페이지별 표 삽입(중복 방지: 교체 방식) --------
def insert_tables_into_md(md_path: Path, tables_by_page: list):
    """
    md 파일에서 <!-- page: N --> 블록의 '맨 아래'(다음 page 주석 직전)에 표를 삽입.
    재실행 시 같은 페이지 표 블록을 찾아 교체.
    """
    if not md_path.exists():
        raise FileNotFoundError(md_path)

    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 페이지 시작 라인 인덱스 찾기
    page_starts = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$", line.strip())
        if m:
            page_starts.append((int(m.group(1)), idx))

    # page 주석이 없으면 삽입 기준이 없으니 실패 처리
    if not page_starts:
        raise ValueError(f"MD에 <!-- page: N --> 주석이 없습니다: {md_path}")

    # 끝 sentinel
    page_starts_sorted = sorted(page_starts, key=lambda x: x[1])
    page_to_range = {}
    for k in range(len(page_starts_sorted)):
        page_no, start_idx = page_starts_sorted[k]
        end_idx = (
            page_starts_sorted[k + 1][1]
            if k + 1 < len(page_starts_sorted)
            else len(lines)
        )
        page_to_range[page_no] = (start_idx, end_idx)

    # 삽입할 데이터 빠른 조회용
    tables_map = {item["page"]: item.get("tables", []) for item in tables_by_page}

    new_lines = lines[:]
    # 뒤에서부터 처리(인덱스 꼬임 방지)
    for page_no in sorted(tables_map.keys(), reverse=True):
        if page_no not in page_to_range:
            # PDF 페이지가 md에 없으면 그냥 스킵
            continue

        start_idx, end_idx = page_to_range[page_no]
        block = new_lines[start_idx:end_idx]

        # 기존에 우리가 넣은 표 블록이 있으면 제거(교체)
        start_tag = f"<!-- tables: start page {page_no} -->"
        end_tag = f"<!-- tables: end page {page_no} -->"
        if start_tag in block and end_tag in block:
            s = block.index(start_tag)
            e = block.index(end_tag)
            block = block[:s] + block[e + 1 :]

        tables = tables_map.get(page_no, [])
        # 이번에 넣을 표가 없으면(빈 리스트) 삭제만 반영하고 종료
        if not tables:
            new_lines[start_idx:end_idx] = block
            continue

        # 페이지 맨 아래에 붙이기: block 끝쪽 공백 정리
        while block and block[-1].strip() == "":
            block.pop()

        insert = ["", start_tag]
        for t in tables:
            caption = (t.get("caption") or "").strip()
            md_table = (t.get("markdown") or "").strip()

            if caption:
                insert.append(f"**[표] {caption}**")
            if md_table:
                insert.append(md_table)
            insert.append("")  # 표 간 공백

        insert.extend([end_tag, ""])
        new_lines[start_idx:end_idx] = block + insert

    md_path.write_text("\n".join(new_lines), encoding="utf-8")


# -------- PDF 하나 처리: 표 있는 페이지만 추출 + (이미 존재하는) MD에 삽입 --------
def process_one_pdf(
    pdf_path: Path,
    zoom: float = 2.0,
    progress_every: int = 5,
):
    md_path = MD_DIR / f"{pdf_path.stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"MD 없음: {md_path}")

    doc = fitz.open(pdf_path)
    tables_by_page = []

    for i in range(doc.page_count):
        page_no = i + 1
        page = doc.load_page(i)

        tables_json = extract_tables_from_page(page, page_no=page_no, zoom=zoom)
        tables = tables_json.get("tables", []) if isinstance(tables_json, dict) else []

        if tables:
            tables_by_page.append({"page": page_no, "tables": tables})

        if progress_every and (page_no % progress_every == 0):
            print(
                f"{pdf_path.name}: {page_no}/{doc.page_count} "
                f"(tables pages so far: {len(tables_by_page)})"
            )

    doc.close()

    insert_tables_into_md(md_path, tables_by_page)
    print(
        f"✅ 완료: {pdf_path.name} → {md_path.name} "
        f"(표 있는 페이지: {len(tables_by_page)}개)"
    )
    return md_path, tables_by_page


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    assert pdfs, f"{PDF_DIR} 폴더에 PDF가 없습니다."

    for pdf_path in pdfs:
        try:
            process_one_pdf(
                pdf_path,
                zoom=2.0,
                progress_every=5,
            )
        except Exception as e:
            print(f"❌ 실패: {pdf_path.name} / 에러: {repr(e)}")


if __name__ == "__main__":
    main()
