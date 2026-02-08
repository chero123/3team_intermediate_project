from __future__ import annotations

"""
FAISS 인덱싱 및 상태 관리 모듈

섹션 구성:
- 상태/통계 데이터 구조
- 벡터 스토어 래퍼
- 인덱서(인덱싱 파이프라인)
"""

import json
import pickle
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from .config import RAGConfig
from .data import chunk_documents, load_documents
from .embeddings import create_embeddings
from .types import Chunk, Document


def _chunk_to_doc(chunk: Chunk) -> LCDocument:
    """
    Chunk를 LangChain Document로 변환

    Args:
        chunk: 입력 청크

    Returns:
        LCDocument: 변환된 문서
    """
    # LangChain FAISS는 Document 형식을 쓰므로, 내부 Chunk를 Document로 변환
    return LCDocument(page_content=chunk.text, metadata={**chunk.metadata, "chunk_id": chunk.id})


def _doc_to_chunk(doc: LCDocument) -> Chunk:
    """
    LangChain Document를 Chunk로 변환

    Args:
        doc: 입력 문서

    Returns:
        Chunk: 변환된 청크
    """
    # 검색 결과의 메타데이터에서 원래 chunk_id 추출
    chunk_id = doc.metadata.get("chunk_id", "")
    # LangChain 문서 메타데이터를 복사해 원본 형태로 돌린다.
    metadata = dict(doc.metadata)
    # 내부용 키는 외부 노출 시 제거
    metadata.pop("chunk_id", None)
    # 검색 결과를 파이프라인에서 쓰는 Chunk 타입으로 변환한
    return Chunk(id=chunk_id, text=doc.page_content, metadata=metadata)

class FaissVectorStore:
    """
    FAISS 인덱스를 래핑

    Args:
        embeddings: 임베딩 객체
    """
    def __init__(self, embeddings: Embeddings) -> None:
        # 임베딩 모델은 인덱스 생성/검색 모두에 필요
        self.embeddings = embeddings
        # 실제 FAISS 인덱스는 build/load 후에만 유효
        self.store: Optional[FAISS] = None

    def build(self, chunks: List[Chunk]) -> None:
        """
        build는 청크 리스트로 FAISS 인덱스를 구축

        Args:
            chunks: 청크 리스트
        """
        # Chunk -> LangChain Document 변환 후 FAISS 인덱스를 생성한다.
        docs = [_chunk_to_doc(chunk) for chunk in chunks]
        self.store = FAISS.from_documents(docs, self.embeddings)

    def save(self, index_dir: str) -> None:
        """
        FAISS 인덱스를 로컬에 저장

        Args:
            index_dir: 저장 경로
        """
        # build/load 이전에는 저장할 인덱스가 없으므로 방어적으로 실패시킨다.
        if self.store is None:
            raise RuntimeError("FAISS store is not built")
        self.store.save_local(index_dir)

    def load(self, index_dir: str) -> None:
        """
        load는 로컬 인덱스를 로드

        Args:
            index_dir: 인덱스 경로
        """
        # 로컬 저장 인덱스를 로드해 동일 임베딩 설정으로 복원한다.
        self.store = FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def similarity_search(self, query: str, top_k: int, fetch_k: Optional[int] = None):
        """
        similarity_search는 유사도 검색을 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            fetch_k: 필터링 전 후보 수

        Returns:
            List[Chunk]: 검색된 청크 리스트
        """
        # 인덱스가 없으면 검색 결과도 없다.
        if self.store is None:
            return []
        # 필터 적용 전 후보를 더 뽑고 싶으면 fetch_k를 사용한다.
        fetch = fetch_k or top_k
        # LangChain이 반환하는 Document 리스트를 받아 후처리한다.
        results = self.store.similarity_search(query, k=fetch)
        chunks = [_doc_to_chunk(doc) for doc in results]
        # 최종 top_k만 반환한다.
        return chunks[:top_k]

    def mmr_search(self, query: str, top_k: int, fetch_k: int, lambda_mult: float):
        """
        mmr_search는 MMR 검색을 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            fetch_k: 후보 풀 크기
            lambda_mult: MMR 가중치

        Returns:
            List[Chunk]: 청크 리스트
        """
        # 인덱스가 없으면 MMR도 수행 불가하다.
        if self.store is None:
            return []
        # 다양성과 관련성을 같이 고려하는 MMR 검색을 수행한다.
        docs = self.store.max_marginal_relevance_search(
            query,
            k=fetch_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        # 검색 결과를 Chunk로 변환한다.
        chunks = [_doc_to_chunk(doc) for doc in docs]
        # 최종 top_k로 자른다.
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
    # 인덱싱 단계 문자열을 상태 파일에 기록한다.
    step: str
    total_documents: int = 0
    total_chunks: int = 0
    min_chunk_len: int = 0
    max_chunk_len: int = 0
    avg_chunk_len: float = 0.0
    # UTC ISO8601 형식으로 상태 갱신 시간을 저장한다.
    updated_at: str = ""
    status: str = "running"
    error: Optional[str] = None

# build_index에서 사용하는 인덱서 클래스
class Indexer:
    """
    Indexer는 인덱싱을 수행

    Args:
        config: 설정 객체
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        # 설정이 없으면 기본 RAGConfig를 사용한다.
        self.config = config or RAGConfig()
        # 인덱스 생성/검색에 동일한 임베딩 설정을 쓰도록 고정한다.
        self.embeddings = create_embeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.device,
            batch_size=self.config.embedding_batch_size,
            normalize=True,
        )
        # FAISS 스토어 래퍼를 초기화한다.
        self.store = FaissVectorStore(self.embeddings)

    def _write_status(self, status: IndexStatus) -> None:
        """
        _write_status는 인덱싱 상태를 기록

        Args:
            status: 상태 객체
        """
        # 상태 파일은 UTC 기준으로 갱신 시간을 기록한다.
        status.updated_at = datetime.now(timezone.utc).isoformat()
        # 상태는 JSON으로 기록한다.
        with open(self.config.index_status_path, "w", encoding="utf-8") as f:
            json.dump(asdict(status), f, ensure_ascii=False, indent=2)

    def _write_preview(self, chunks: List[Chunk]) -> None:
        """
        _write_preview는 청킹 샘플을 저장

        Args:
            chunks: 청크 리스트
        """
        # 청킹 샘플을 미리보기용으로 저장
        preview = [
            {
                "id": c.id,
                "len": len(c.text),
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        # 청크 미리보기는 디버깅/검증 용도로만 사용한다.
        with open(self.config.chunk_preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)

    def _build_bm25(self, chunks: List[Chunk]) -> None:
        """
        _build_bm25는 BM25 인덱스를 생성해 저장한다.

        Args:
            chunks: 청크 리스트
        """
        if not chunks:
            return
        # bm25 저장 경로가 없으면 생성한다.
        bm25_dir = os.path.dirname(self.config.bm25_index_path)
        if bm25_dir:
            os.makedirs(bm25_dir, exist_ok=True)
        docs = [
            LCDocument(page_content=c.text, metadata={**c.metadata, "chunk_id": c.id})
            for c in chunks
        ]
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.bm25_top_k
        with open(self.config.bm25_index_path, "wb") as f:
            pickle.dump(bm25, f)

    def build_index(self, data_dir: str, metadata_csv: str) -> None:
        """
        build_index는 문서 로드부터 인덱스 저장까지 수행

        Args:
            data_dir: 데이터 디렉토리
            metadata_csv: 메타데이터 CSV

        Returns:
            None
        """
        # 단계별로 진행 상태를 기록해 중간 실패 원인을 추적
        status = IndexStatus(step="load_documents")
        self._write_status(status)

        # 데이터 디렉토리와 메타데이터 CSV로부터 문서를 로드
        documents: List[Document] = load_documents(
            data_dir,
            metadata_csv,
            config=self.config,
        )
        status.total_documents = len(documents)

        # 문서 로딩 후 청킹 단계로 넘어간다.
        status.step = "chunk_documents"
        self._write_status(status)

        # 청크 크기/오버랩은 설정값을 사용한다.
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        print(f"[CHUNK] total_chunks={len(chunks)}")
        status.total_chunks = len(chunks)
        # 청크 길이 통계를 만들어 상태 파일에 남긴다.
        lengths = [len(c.text) for c in chunks] if chunks else [0]
        status.min_chunk_len = int(min(lengths))
        status.max_chunk_len = int(max(lengths))
        status.avg_chunk_len = float(sum(lengths) / max(1, len(lengths)))
        self._write_preview(chunks)

        # BM25 인덱스를 저장 
        print(f"[BM25] build/save start -> {self.config.bm25_index_path}")
        self._build_bm25(chunks)
        print("[BM25] build/save done")

        # FAISS 인덱스 빌드 단계로 전환
        status.step = "build_faiss"
        self._write_status(status)

        # 청크들을 벡터화해 FAISS 인덱스를 생성
        print("[FAISS] build start")
        self.store.build(chunks)
        print("[FAISS] build done")

        # 로컬 저장 단계로 전환
        status.step = "save_faiss"
        self._write_status(status)

        # 인덱스를 디스크에 저장
        print(f"[FAISS] save start -> {self.config.index_dir}")
        self.store.save(self.config.index_dir)
        print("[FAISS] save done")

        # 모든 단계가 완료되면 완료 상태로 기록
        status.step = "done"
        status.status = "done"
        self._write_status(status)
