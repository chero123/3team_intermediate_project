from __future__ import annotations

"""
RAG 전역 설정 모듈

역할:
- 인덱싱/검색/생성 파라미터를 한 곳에서 관리
"""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """
    RAGConfig 데이터 클래스
    """
    # 청크 길이 정의 (한국어 문서 기준으로 문단 단위가 잘 끊기는 값)
    chunk_size: int = 800 
    # 청크 간 중복 길이 정의 (경계 정보 손실 방지용)
    chunk_overlap: int = 120

    # 최대 검색 결과 수 제한 (지나친 컨텍스트 팽창 방지)
    max_top_k: int = 5
    # 최소 검색 결과 수 보장 (컨텍스트 부족 방지)
    min_top_k: int = 3

    # 단순 유사도 전략 키워드 지정
    similarity_strategy: str = "similarity"
    # MMR 전략 키워드 지정
    mmr_strategy: str = "mmr"
    # RRF 전략 키워드 지정
    rrf_strategy: str = "rrf"

    # 생성 응답 최대 토큰 수 설정 (과도한 장문 방지)
    response_max_tokens: int = 640
    # 생성 샘플링 온도 설정 (요약 정확성 우선)
    response_temperature: float = 0.1
    # 생성 상위 확률 누적(top-p) 설정 (반복 억제/다양성 확보)
    response_top_p: float = 0.9
    # 생성 반복 페널티 설정 (동일 문장 반복 억제)
    response_repetition_penalty: float = 1.15
    # 생성 중단 토큰 (라벨/메타 발화 방지)
    response_stop: list[str] = field(
        default_factory=lambda: [
            "rewrite 결과",
            "원문",
        ]
    )
    # LLM 로컬 경로 지정
    llm_model_path: str = "models/A.X-4.0-Light"
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
    openai_gpt5_max_tokens: int = 1000

    # 임베딩 배치 크기 설정 (GPU 메모리와 속도 균형)
    embedding_batch_size: int = 32

    # RRF 결합 보정 상수 설정 (상위 랭크 편향 완화용)
    rrf_k: int = 60
    # RRF에서 dense(similarity) 가중치
    rrf_dense_weight: float = 1.0
    # RRF에서 MMR 가중치
    rrf_mmr_weight: float = 1.0
    # BM25 검색 결과 수 (키워드 매칭 신호 확보)
    bm25_top_k: int = 8
    # RRF에서 BM25 가중치 (dense 결과 대비 영향도)
    rrf_bm25_weight: float = 1.0
    # MMR 가중치 설정 (관련성 vs 다양성 균형)
    mmr_lambda: float = 0.7
    # MMR 후보 풀 크기 설정 (다양성 확보용 후보 수)
    mmr_candidate_pool: int = 10

    # 리라이트용 생성 토큰 수
    rewrite_max_tokens: int = 480
    # 리라이트용 온도 (말투/스타일 유지)
    rewrite_temperature: float = 0.6
    # 리라이트용 top-p
    rewrite_top_p: float = 0.9
    # 리라이트용 반복 페널티
    rewrite_repetition_penalty: float = 1.1
    # 리라이트 중단 토큰 (라벨/메타 발화 방지)
    rewrite_stop: list[str] = field(
        default_factory=lambda: [
            "rewrite 결과",
            "원문",
        ]
    )

    # 인덱스 저장 경로 지정
    index_dir: str = "data/index"
    # BM25 인덱스 저장 경로
    bm25_index_path: str = "data/index/bm25.pkl"
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
    # Qwen3-VL 최대 생성 토큰 (표/수치 요약에 필요한 길이)
    qwen3_vl_max_tokens: int = 512
    # Qwen3-VL vLLM GPU 메모리 사용량 (KV 캐시 부족 방지)
    qwen3_vl_gpu_memory_utilization: float = 0.9
    # Qwen3-VL vLLM 최대 컨텍스트 길이 (FP8+GPU 메모리 제약 고려)
    qwen3_vl_max_model_len: int = 8192
    # Qwen3-VL 이미지 필터링(무의미 이미지 스킵)
    qwen3_vl_dedupe_images: bool = True
    # 너무 작은 이미지/아이콘 제거 기준
    qwen3_vl_min_image_pixels: int = 128 * 128
    # 빈 페이지(거의 백지) 제거 기준
    qwen3_vl_min_nonwhite_ratio: float = 0.02
    # 단색/로고 제거 기준
    qwen3_vl_min_variance: float = 15.0
    # 도표/텍스트가 없는 이미지 제거 기준
    qwen3_vl_min_edge_energy: float = 0.01
