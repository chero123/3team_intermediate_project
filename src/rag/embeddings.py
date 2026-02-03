from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_path: str, device: str, batch_size: int, normalize: bool = True) -> Embeddings:
    """
    create_embeddings는 HuggingFace 임베딩 로더를 생성

    Args:
        model_path: 로컬 임베딩 모델 경로
        device: 실행 디바이스
        batch_size: 배치 크기
        normalize: 정규화 여부

    Returns:
        Embeddings: LangChain Embeddings 인스턴스
    """
    return HuggingFaceEmbeddings(
        # 로컬 모델 경로를 지정
        model_name=model_path,
        model_kwargs={
            # 디바이스를 지정
            "device": device,
            # 로컬 파일만 사용
            "local_files_only": True,
        },
        encode_kwargs={
            # 정규화를 적용
            "normalize_embeddings": normalize,
            # 배치 크기를 적용
            "batch_size": batch_size,
        },
    )
