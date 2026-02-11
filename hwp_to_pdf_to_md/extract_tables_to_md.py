# extract_tables_to_md.py
# ============================================
# pdf_out/*.pdf → VLM 표 추출 → final_docs/*.md에 페이지별 삽입
#
# 개선:
# - JSONDecodeError(Invalid control character) 방지/복구
# - response_format json_object(지원 시)로 JSON 강제
# - 파싱 실패/호출 실패는 "페이지 스킵 + 로그" (전체 중단 방지)
# - JPEG + DPI 렌더링 + concurrency 제한 + gc
# - ENOSPC(디스크 부족) 시 즉시 중단
# ============================================

import asyncio
import base64
import errno
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# -------- 경로 설정 --------
PDF_DIR = Path("pdf_out")
MD_DIR = Path("final_docs")

# -------- OpenAI 설정 --------
client = AsyncOpenAI()
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

# -------- 성능/부하 튜닝 --------
DPI = 140               # 120~160 권장
IMG_FORMAT = "jpeg"     # "jpeg" 추천
CONCURRENCY = 4         # 3~5 권장
RETRY = 1               # 1~2 권장
SLEEP_SEC = 0.6         # 재시도 backoff base
GC_EVERY = 15           # N페이지마다 gc

# -------- 로그 저장 --------
LOG_DIR = Path("logs_tables")
LOG_DIR.mkdir(exist_ok=True)

# =========================
# 1) JSON 파싱(강화)
# =========================
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def safe_json_loads(s: str) -> Dict[str, Any]:
    """
    모델이 ```json ...``` 혹은 앞/뒤에 텍스트를 섞거나,
    제어문자(Invalid control character)가 들어가도 최대한 복구해서 파싱.
    """
    s = (s or "").strip()

    # 코드펜스 제거
    s = re.sub(r"^```json\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    # 가장 바깥 { ... } 만 골라내기
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)

    # 제어문자 제거(가장 흔한 JSONDecodeError 원인)
    s = _CONTROL_CHARS_RE.sub("", s)

    return json.loads(s)

# =========================
# 2) 페이지 → data URL
# =========================
def page_to_data_url(page, dpi: int = DPI, img_format: str = IMG_FORMAT) -> str:
    pix = page.get_pixmap(dpi=dpi, alpha=False)

    if img_format.lower() == "png":
        raw = pix.tobytes("png")
        mime = "image/png"
    else:
        raw = pix.tobytes("jpeg")
        mime = "image/jpeg"

    del pix
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# =========================
# 3) VLM 호출 (JSON 모드 시도 → 실패하면 일반 호출)
# =========================
async def call_vlm_tables(img_url: str, page_no: int) -> str:
    """
    output_text(str)만 반환.
    response_format json_object를 먼저 시도하고,
    모델/환경이 거부하면 fallback.
    """
    # 1) JSON 모드 시도
    try:
        resp = await client.responses.create(
            model=MODEL,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{page_no}페이지에서 표만 추출해."},
                        {"type": "input_image", "image_url": img_url},
                    ],
                },
            ],
        )
        return resp.output_text
    except Exception:
        # 2) fallback: 일반 호출
        resp = await client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{page_no}페이지에서 표만 추출해."},
                        {"type": "input_image", "image_url": img_url},
                    ],
                },
            ],
        )
        return resp.output_text

async def extract_tables_from_page(page, page_no: int, pdf_name: str) -> Dict[str, Any]:
    """
    - retry 포함
    - JSON 파싱 실패 시: tables=[]로 반환 + 로그 저장 (전체 중단 방지)
    """
    img_url = page_to_data_url(page, dpi=DPI)

    last_err: Optional[Exception] = None
    last_raw: Optional[str] = None

    for attempt in range(RETRY + 1):
        try:
            raw = await call_vlm_tables(img_url, page_no)
            last_raw = raw

            try:
                return safe_json_loads(raw)
            except json.JSONDecodeError as je:
                # JSON 깨짐 → 페이지 스킵 처리, raw 로그 저장
                log_path = LOG_DIR / f"jsondecode_{pdf_name}_p{page_no}.txt"
                log_path.write_text(raw or "", encoding="utf-8")
                return {"tables": [], "_error": f"JSONDecodeError: {repr(je)}", "_raw_saved": str(log_path)}

        except Exception as e:
            last_err = e
            await asyncio.sleep(SLEEP_SEC * (attempt + 1))

    # 호출 자체 실패 → 페이지 스킵 + 에러 로그
    if last_raw:
        log_path = LOG_DIR / f"callfail_{pdf_name}_p{page_no}.txt"
        log_path.write_text(last_raw, encoding="utf-8")

    return {"tables": [], "_error": f"call_failed: {repr(last_err)}"}

# =========================
# 4) MD 삽입(교체 방식)
# =========================
def insert_tables_into_md(md_path: Path, tables_by_page: List[Dict[str, Any]]):
    if not md_path.exists():
        raise FileNotFoundError(md_path)

    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    page_starts = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$", line.strip())
        if m:
            page_starts.append((int(m.group(1)), idx))

    if not page_starts:
        raise ValueError(f"MD에 <!-- page: N --> 주석이 없습니다: {md_path}")

    page_starts_sorted = sorted(page_starts, key=lambda x: x[1])
    page_to_range = {}
    for k in range(len(page_starts_sorted)):
        page_no, start_idx = page_starts_sorted[k]
        end_idx = page_starts_sorted[k + 1][1] if k + 1 < len(page_starts_sorted) else len(lines)
        page_to_range[page_no] = (start_idx, end_idx)

    tables_map = {item["page"]: item.get("tables", []) for item in tables_by_page}

    new_lines = lines[:]
    for page_no in sorted(tables_map.keys(), reverse=True):
        if page_no not in page_to_range:
            continue

        start_idx, end_idx = page_to_range[page_no]
        block = new_lines[start_idx:end_idx]

        start_tag = f"<!-- tables: start page {page_no} -->"
        end_tag = f"<!-- tables: end page {page_no} -->"

        # 기존 블록 제거(재실행 중복 방지)
        if start_tag in block and end_tag in block:
            s = block.index(start_tag)
            e = block.index(end_tag)
            block = block[:s] + block[e + 1 :]

        tables = tables_map.get(page_no, [])
        if not tables:
            new_lines[start_idx:end_idx] = block
            continue

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
            insert.append("")
        insert.extend([end_tag, ""])
        new_lines[start_idx:end_idx] = block + insert

    md_path.write_text("\n".join(new_lines), encoding="utf-8")

# =========================
# 5) PDF 하나 처리
# =========================
async def process_one_pdf(pdf_path: Path, concurrency: int = CONCURRENCY, progress_every: int = 10):
    md_path = MD_DIR / f"{pdf_path.stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"MD 없음: {md_path}")

    doc = fitz.open(pdf_path)
    tables_by_page: List[Dict[str, Any]] = []

    sem = asyncio.Semaphore(concurrency)
    pdf_name_safe = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", pdf_path.stem)

    async def run_page(i: int) -> Tuple[int, List[Dict[str, Any]]]:
        page_no = i + 1
        page = doc.load_page(i)
        try:
            async with sem:
                tables_json = await extract_tables_from_page(page, page_no=page_no, pdf_name=pdf_name_safe)

            tables = []
            if isinstance(tables_json, dict):
                tables = tables_json.get("tables", []) or []

            return (page_no, tables)
        finally:
            del page

    tasks = [asyncio.create_task(run_page(i)) for i in range(doc.page_count)]
    done_count = 0

    for coro in asyncio.as_completed(tasks):
        page_no, tables = await coro
        done_count += 1

        if tables:
            tables_by_page.append({"page": page_no, "tables": tables})

        if progress_every and (done_count % progress_every == 0):
            print(f"{pdf_path.name}: {done_count}/{doc.page_count} (tables pages so far: {len(tables_by_page)})")

        if GC_EVERY and (done_count % GC_EVERY == 0):
            gc.collect()

    doc.close()

    tables_by_page.sort(key=lambda x: x["page"])
    insert_tables_into_md(md_path, tables_by_page)

    print(f"✅ 완료: {pdf_path.name} → {md_path.name} (표 있는 페이지: {len(tables_by_page)}개)")
    return md_path, tables_by_page

# =========================
# 6) main
# =========================
def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    assert pdfs, f"{PDF_DIR} 폴더에 PDF가 없습니다."

    for pdf_path in pdfs:
        try:
            asyncio.run(process_one_pdf(pdf_path, concurrency=CONCURRENCY, progress_every=10))
        except OSError as e:
            if getattr(e, "errno", None) == errno.ENOSPC or "No space left" in str(e):
                print(f"💥 디스크 용량 부족(ENOSPC). 즉시 중단: {pdf_path.name}")
                raise
            print(f"❌ 실패: {pdf_path.name} / 에러: {repr(e)}")
        except Exception as e:
            print(f"❌ 실패: {pdf_path.name} / 에러: {repr(e)}")

if __name__ == "__main__":
    main()
