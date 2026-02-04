from __future__ import annotations

import argparse

from rag.indexing import Indexer


def main() -> None:
    """
    인덱싱 생성 함수

    Indexer 객체를 생성하여 지정된 데이터 디렉토리와 메타데이터 파일을 사용하여 인덱스 빌드

    실행 방법
    uv run scripts/build_index.py --data-dir <데이터 디렉토리 경로> --metadata <메타데이터 파일 경로>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/files")
    parser.add_argument("--metadata", default="data/data_list.csv")
    args = parser.parse_args()

    # 인덱서 객체 생성 및 인덱스 빌드 (상세 내용은 src/rag/indexing.py의 Indexer 클래스 참조)
    indexer = Indexer()
    indexer.build_index(args.data_dir, args.metadata)


if __name__ == "__main__":
    main()
