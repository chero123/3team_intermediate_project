from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    RAGConfig 데이터 클래스
    """
    # 청크 길이 정의
    chunk_size: int = 800 
    # 청크 간 중복 길이 정의
    chunk_overlap: int = 120

    # 최대 검색 결과 수 제한
    max_top_k: int = 12
    # 최소 검색 결과 수 보장
    min_top_k: int = 3

    # 단순 유사도 전략 키워드 지정
    similarity_strategy: str = "similarity"
    # MMR 전략 키워드 지정
    mmr_strategy: str = "mmr"
    # RRF 전략 키워드 지정
    rrf_strategy: str = "rrf"

    # 생성 응답 최대 토큰 수 설정
    response_max_tokens: int = 160
    # 생성 샘플링 온도 설정
    response_temperature: float = 0.2

    # LLM 로컬 경로 지정
    llm_model_path: str = "models/YanoljaNEXT-EEVE-7B-v2"
    # 임베딩 로컬 경로 지정
    embedding_model_path: str = "models/bge-m3-ko"
    # 랭커 로컬 경로 지정
    rerank_model_path: str = "models/bge-reranker-v2-m3"
    # 실행 디바이스 설정
    device: str = "cuda"
    # vLLM 4-bit(bitsandbytes) 양자화 사용
    llm_quantization: str = "bitsandbytes"
    # LLM 제공자 선택 (vllm | openai)
    llm_provider: str = "vllm"
    # OpenAI 모델 이름
    openai_model: str = "gpt-4.1"

    # 임베딩 배치 크기 설정
    embedding_batch_size: int = 32

    # RRF 결합 보정 상수 설정
    rrf_k: int = 60
    # MMR 가중치 설정
    mmr_lambda: float = 0.7
    # MMR 후보 풀 크기 설정
    mmr_candidate_pool: int = 30

    # 인덱스 저장 경로 지정
    index_dir: str = "data/index"
    # 인덱싱 상태 파일 경로 지정
    index_status_path: str = "data/index_status.json"
    # 청크 미리보기 파일 경로 지정
    chunk_preview_path: str = "data/chunk_preview.json"
    # vLLM용 토크나이저 캐시 경로 지정
    tokenizer_cache_dir: str = "data/tokenizer_cache"
