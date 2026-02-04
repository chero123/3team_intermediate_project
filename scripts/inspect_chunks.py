from __future__ import annotations

"""
청킹 결과 점검 스크립트

역할:
- 저장된 청크 샘플/통계를 빠르게 확인
"""
# =====================================================================
# 청킹 점검 도구
#
# 흐름:
# - 상태/샘플 JSON 로드
# - 통계/미리보기 출력
# =====================================================================

import argparse

from rag.config import RAGConfig
from rag.data import chunk_documents, load_documents


def main() -> None:
    """
    청킹 결과를 출력
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/files")
    parser.add_argument("--metadata", default="data/data_list.csv")
    parser.add_argument("--filter", default=None, help="filename substring filter")
    parser.add_argument("--limit", type=int, default=3, help="number of chunks to print")
    args = parser.parse_args()

    config = RAGConfig()
    docs = load_documents(args.data_dir, args.metadata)
    if args.filter:
        docs = [d for d in docs if args.filter in d.id or args.filter in d.metadata.get("filename", "")]

    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    lengths = [len(c.text) for c in chunks] if chunks else [0]

    print(f"documents: {len(docs)}")
    print(f"chunks: {len(chunks)}")
    print(f"min/avg/max: {min(lengths)}/{sum(lengths)/max(1,len(lengths)):.1f}/{max(lengths)}")

    for c in chunks[: args.limit]:
        print("---")
        print(f"id: {c.id}")
        print(f"len: {len(c.text)}")
        print(c.text[:500])


if __name__ == "__main__":
    main()
