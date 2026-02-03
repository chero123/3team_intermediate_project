from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from .config import RAGConfig
from .data import chunk_documents, load_documents
from .embeddings import create_embeddings
from .types import Chunk, Document


def _chunk_to_doc(chunk: Chunk) -> LCDocument:
    """
    _chunk_to_doc는 Chunk를 LangChain Document로 변환

    Args:
        chunk: 입력 청크

    Returns:
        LCDocument: 변환된 문서
    """
    return LCDocument(page_content=chunk.text, metadata={**chunk.metadata, "chunk_id": chunk.id})


def _doc_to_chunk(doc: LCDocument) -> Chunk:
    """
    _doc_to_chunk는 LangChain Document를 Chunk로 변환

    Args:
        doc: 입력 문서

    Returns:
        Chunk: 변환된 청크
    """
    chunk_id = doc.metadata.get("chunk_id", "")
    metadata = dict(doc.metadata)
    metadata.pop("chunk_id", None)
    return Chunk(id=chunk_id, text=doc.page_content, metadata=metadata)


class FaissVectorStore:
    """
    FaissVectorStore는 FAISS 인덱스를 래핑

    Args:
        embeddings: 임베딩 객체
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.store: Optional[FAISS] = None

    def build(self, chunks: List[Chunk]) -> None:
        """
        build는 청크 리스트로 인덱스를 구축

        Args:
            chunks: 청크 리스트
        """
        docs = [_chunk_to_doc(chunk) for chunk in chunks]
        self.store = FAISS.from_documents(docs, self.embeddings)

    def save(self, index_dir: str) -> None:
        """
        save는 인덱스를 로컬에 저장

        Args:
            index_dir: 저장 경로
        """
        if self.store is None:
            raise RuntimeError("FAISS store is not built")
        self.store.save_local(index_dir)

    def load(self, index_dir: str) -> None:
        """
        load는 로컬 인덱스를 로드

        Args:
            index_dir: 인덱스 경로
        """
        self.store = FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def similarity_search(self, query: str, top_k: int, metadata_filter: Optional[dict] = None, fetch_k: Optional[int] = None):
        """
        similarity_search는 유사도 검색을 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            metadata_filter: 메타데이터 필터
            fetch_k: 필터링 전 후보 수

        Returns:
            Tuple[List[Chunk], List[float]]: 청크와 점수 리스트
        """
        if self.store is None:
            return [], []
        fetch = fetch_k or top_k
        results = self.store.similarity_search_with_score(query, k=fetch)
        filtered = []
        for doc, score in results:
            chunk = _doc_to_chunk(doc)
            if metadata_filter:
                if any(str(chunk.metadata.get(k)) != str(v) for k, v in metadata_filter.items()):
                    continue
            filtered.append((chunk, float(score)))
        selected = filtered[:top_k]
        return [c for c, _ in selected], [s for _, s in selected]

    def mmr_search(self, query: str, top_k: int, fetch_k: int, lambda_mult: float, metadata_filter: Optional[dict] = None):
        """
        mmr_search는 MMR 검색을 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            fetch_k: 후보 풀 크기
            lambda_mult: MMR 가중치
            metadata_filter: 메타데이터 필터

        Returns:
            List[Chunk]: 청크 리스트
        """
        if self.store is None:
            return []
        docs = self.store.max_marginal_relevance_search(
            query,
            k=fetch_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        chunks = [_doc_to_chunk(doc) for doc in docs]
        if metadata_filter:
            chunks = [c for c in chunks if not any(str(c.metadata.get(k)) != str(v) for k, v in metadata_filter.items())]
        return chunks[:top_k]


@dataclass
class IndexStatus:
    """
    IndexStatus는 인덱싱 진행 상황을 기록한다.

    Args:
        step: 현재 단계
        total_documents: 문서 수
        total_chunks: 청크 수
        min_chunk_len: 최소 길이
        max_chunk_len: 최대 길이
        avg_chunk_len: 평균 길이
        updated_at: 업데이트 시간
        status: 상태 문자열
        error: 에러 메시지
    """

    step: str
    total_documents: int = 0
    total_chunks: int = 0
    min_chunk_len: int = 0
    max_chunk_len: int = 0
    avg_chunk_len: float = 0.0
    updated_at: str = ""
    status: str = "running"
    error: Optional[str] = None


class Indexer:
    """
    Indexer는 인덱싱을 수행

    Args:
        config: 설정 객체
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()
        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )
        self.store = FaissVectorStore(self.embeddings)

    def _write_status(self, status: IndexStatus) -> None:
        """
        _write_status는 인덱싱 상태를 기록

        Args:
            status: 상태 객체
        """
        status.updated_at = datetime.now(timezone.utc).isoformat()
        with open(self.config.index_status_path, "w", encoding="utf-8") as f:
            json.dump(asdict(status), f, ensure_ascii=False, indent=2)

    def _write_preview(self, chunks: List[Chunk]) -> None:
        """
        _write_preview는 청킹 샘플을 저장

        Args:
            chunks: 청크 리스트
        """
        preview = [
            {
                "id": c.id,
                "len": len(c.text),
                "text": c.text[:200],
                "metadata": c.metadata,
            }
            for c in chunks[:5]
        ]
        with open(self.config.chunk_preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)

    def build_index(self, data_dir: str, metadata_csv: str) -> None:
        """
        build_index는 문서 로드부터 인덱스 저장까지 수행

        Args:
            data_dir: 데이터 디렉토리
            metadata_csv: 메타데이터 CSV

        Returns:
            None
        """
        status = IndexStatus(step="load_documents")
        self._write_status(status)

        documents: List[Document] = load_documents(data_dir, metadata_csv)
        status.total_documents = len(documents)

        status.step = "chunk_documents"
        self._write_status(status)

        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        status.total_chunks = len(chunks)
        lengths = [len(c.text) for c in chunks] if chunks else [0]
        status.min_chunk_len = int(min(lengths))
        status.max_chunk_len = int(max(lengths))
        status.avg_chunk_len = float(sum(lengths) / max(1, len(lengths)))
        self._write_preview(chunks)

        status.step = "build_faiss"
        self._write_status(status)

        self.store.build(chunks)

        status.step = "save_faiss"
        self._write_status(status)

        self.store.save(self.config.index_dir)

        status.step = "done"
        status.status = "done"
        self._write_status(status)
