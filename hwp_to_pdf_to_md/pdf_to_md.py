## pdf 를 .md 파일로 바꾸면서 페이지마다 페이지 번호 추가 (vlm으로 표 추출한거 삽입 위함)

from pathlib import Path
import fitz  # pymupdf

PDF_DIR = Path("pdf_out")
MD_DIR = Path("final_docs")
MD_DIR.mkdir(exist_ok=True)

def pdf_to_md_with_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    lines = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")

        lines.append(f"<!-- page: {i+1} -->")
        lines.append(text.strip())
        lines.append("")  # 페이지 간 공백

    out_path = MD_DIR / f"{pdf_path.stem}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    doc.close()
    return out_path

pdfs = sorted(PDF_DIR.glob("*.pdf"))
print("PDF files:", len(pdfs))

for pdf in pdfs:
    md_path = pdf_to_md_with_pages(pdf)
    print("✅ md created:", md_path.name)
