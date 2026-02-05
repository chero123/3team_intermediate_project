# context_builder.py
from typing import List
from langchain_core.documents import Document


def build_context(
    docs: List[Document],
    max_chars: int = 6000,
    per_doc_chars: int = 1500,
):
    parts = []
    total = 0

    for i, d in enumerate(docs, start=1):
        src = (
            d.metadata.get("source")
            or d.metadata.get("source_file")
            or "unknown_source"
        )
        doc_id = d.metadata.get("doc_id", "unknown_doc")
        chunk_id = d.metadata.get("chunk_id")

        text = d.page_content.strip()
        if not text:
            continue

        if len(text) > per_doc_chars:
            text = text[:per_doc_chars] + "\n...(truncated)"

        header = f"[{i}] source={src} doc_id={doc_id}"
        if chunk_id:
            header += f" chunk_id={chunk_id}"

        block = header + "\n" + text

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)
