from __future__ import annotations

"""
RAG 전역 설정 모듈

역할:
- 인덱싱/검색/생성 파라미터를 한 곳에서 관리
"""

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
    response_max_tokens: int = 480
    # 생성 샘플링 온도 설정
    response_temperature: float = 0.1

    # LLM 로컬 경로 지정
    llm_model_path: str = "models/YanoljaNEXT-EEVE-7B-v2"
    # 임베딩 로컬 경로 지정
    embedding_model_path: str = "models/bge-m3-ko"
    # 랭커 로컬 경로 지정
    rerank_model_path: str = "models/bge-reranker-v2-m3-ko"
    # 실행 디바이스 설정
    device: str = "cuda"
    # vLLM 4-bit(bitsandbytes) 양자화 사용
    llm_quantization: str = "bitsandbytes"
    # LLM 제공자 선택 (vllm | openai)
    llm_provider: str = "vllm"


    # OpenAI 모델 이름
    openai_model: str = "gpt-5-mini"
    # OpenAI Gpt-5 계열용 맥스 토큰 설정
    openai_gpt5_max_tokens: int = 800

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

    # Qwen3-VL 로컬 모델 경로
    qwen3_vl_model_path: str = "models/qwen3-vl-8b"
    # Qwen3-VL 사용 여부
    qwen3_vl_enabled: bool = True
    # Qwen3-VL 최대 생성 토큰
    qwen3_vl_max_tokens: int = 512
    # Qwen3-VL vLLM GPU 메모리 사용량
    qwen3_vl_gpu_memory_utilization: float = 0.9
    # Qwen3-VL vLLM 최대 컨텍스트 길이
    qwen3_vl_max_model_len: int = 8192
    # Qwen3-VL 이미지 필터링(무의미 이미지 스킵)
    qwen3_vl_dedupe_images: bool = True
    qwen3_vl_min_image_pixels: int = 128 * 128
    qwen3_vl_min_nonwhite_ratio: float = 0.02
    qwen3_vl_min_variance: float = 15.0
    qwen3_vl_min_edge_energy: float = 0.01
