from __future__ import annotations

import argparse

from rag.indexing import Indexer


def main() -> None:
    """
    인덱싱 생성 함수

    Indexer 객체를 생성하여 지정된 데이터 디렉토리와 메타데이터 파일을 사용하여 인덱스 빌드
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/files")
    parser.add_argument("--metadata", default="data/data_list.csv")
    args = parser.parse_args()

    indexer = Indexer()
    indexer.build_index(args.data_dir, args.metadata)


if __name__ == "__main__":
    main()
