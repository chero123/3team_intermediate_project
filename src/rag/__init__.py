"""RAG 패키지를 초기화하는 모듈"""

__all__ = ["RAGConfig", "RAGPipeline", "Indexer"]


def __getattr__(name: str):
    if name == "RAGConfig":
        from .config import RAGConfig

        return RAGConfig
    if name == "RAGPipeline":
        from .pipeline import RAGPipeline

        return RAGPipeline
    if name == "Indexer":
        from .indexing import Indexer

        return Indexer
    raise AttributeError(name)
