import numpy as np
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from chunking import Chunk


@dataclass
class SearchResult:
    """검색 결과 클래스"""

    chunk_id: str
    text: str
    score: float
    metadata: dict


class BaseVectorDB(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        pass


class ChromaVectorDB(BaseVectorDB):
    def __init__(self, collection_name: str = "chunks", persist_directory: str = None):
        import chromadb

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(
                SearchResult(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i],
                    metadata=results["metadatas"][0][i],
                )
            )
        return search_results

    @property
    def count(self) -> int:
        return self.collection.count()


class FAISSVectorDB(BaseVectorDB):
    def __init__(self, dimension: int):
        import faiss

        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add((embeddings / norms).astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        q = (
            (query_embedding / np.linalg.norm(query_embedding))
            .reshape(1, -1)
            .astype(np.float32)
        )
        scores, indices = self.index.search(q, top_k)
        return [
            SearchResult(
                chunk_id=self.chunks[i].id,
                text=self.chunks[i].text,
                score=float(s),
                metadata=self.chunks[i].metadata,
            )
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    @property
    def count(self) -> int:
        return self.index.ntotal
