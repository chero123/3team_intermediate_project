# TTS RAG 초기 아키텍쳐

- 아키텍쳐에 관한 상세 내용은 [여기](docs/ARCHITECTURE.md)
- TTS 온닉스 재생에 관한 상세 내용은 [여기](docs/ONNX_TTS.md)
- SQLite에 관한 상세 내용은 [여기](docs/sqlite.md)
- Qwen3-VL-8B에 관한 상세 내용은 [여기](docs/QWEN3_VL.md)

## 설치 방법

- WSL UV 기준

```bash
uv venv --python 3.10
source .venv/bin/activate

uv sync
```

## 실행 순서

1. 가상 환경 생성 후 `uv sync` 및 `pip install -r requirements.txt`
2. `uv run initialize.py`로 모델 다운로드
3. `uv run scripts/build_index`로 백터 DB 생성
4. `uv run scripts/run_query`로 CLI 실행

## 현재 관찰된 것

- vLLM + 4bits(배포 최적화 옵션)로 실행할 시, GPT API 사용 버전에 거의 근접한 속도를 구현할 수 있다.
- 다만, 프롬프트 준수율은 GPT가 우수하다.
- 웹 UI로 vLLM을 실행 시 고성능 컴퓨터가 요구된다.
- GPT-5 계열은 추론 모델이라 GPT-4 계열과 달리 높은 토큰 수 상한이 필요
- CLI에서 run_query 실행 시 음성에 잡음이 끼는 현상이 있는데, 실제 생성된 음성 파일과 WebUI 등으로 확인 시에는 잡음이 없다.
- Gradio, Streamlit과 같은 데모 프레임워크로는 실시간 합성을 실현할 수 없다.
- 실시간 음성 합성을 재현하려면 FastAPI + 커스텀 프론트나 다른 방식을 사용해야 한다.
- Qwen3-VL 이미지 파싱(8b 기준)은 무겁지만 성능이 괜찮다. 

## 2026-02-04기준 CONFIG 스냅샷
- 아래는 `src/rag/config.py` 기준의 기본값이다.
- 컨텍스트 구성은 SIMILARITY, BM25(신규 추가), MMR, RERANK, RRF를 모두 사용한다.
- GPT API 환경에서는 문제 없으나, vLLM의 경우 RTX 5080 16GB 환경 기준 프롬프트 길이가 길어져 max_model_len 초과로 우주 폭발 당할 수 있다.
- BM25를 추가하기 전에는 아래 설정으로 vLLM RAG 시스템을 구동할 수 있었으나, 2026-02-05에 BM25를 추가하며 컨텍스트 길이가 증가하면서 프롬프트 길이가 max_model_len 초과를 일으키고 있다.

```text
chunk_size=800
chunk_overlap=120
max_top_k=12
min_top_k=3
similarity_strategy=similarity
mmr_strategy=mmr
rrf_strategy=rrf
response_max_tokens=640
response_temperature=0.1
response_top_p=0.9
response_repetition_penalty=1.15
response_stop=[rewrite 결과, 원문]
llm_model_path=models/A.X-4.0-Light
embedding_model_path=models/bge-m3-ko
rerank_model_path=models/bge-reranker-v2-m3-ko
device=cuda
llm_quantization=bitsandbytes
llm_provider=vllm
openai_model=gpt-5-mini
openai_gpt5_max_tokens=800
embedding_batch_size=32
rrf_k=60
rrf_dense_weight=1.0
rrf_mmr_weight=1.0
bm25_top_k=30
rrf_bm25_weight=1.0
mmr_lambda=0.7
mmr_candidate_pool=30
rewrite_max_tokens=480
rewrite_temperature=0.6
rewrite_top_p=0.9
rewrite_repetition_penalty=1.1
rewrite_stop=[rewrite 결과, 원문]
index_dir=data/index
bm25_index_path=data/index/bm25.pkl
index_status_path=data/index_status.json
chunk_preview_path=data/chunk_preview.json
tokenizer_cache_dir=data/tokenizer_cache
qwen3_vl_model_path=models/qwen3-vl-8b
qwen3_vl_enabled=True
qwen3_vl_max_tokens=512
qwen3_vl_gpu_memory_utilization=0.9
qwen3_vl_max_model_len=8192
qwen3_vl_dedupe_images=True
qwen3_vl_min_image_pixels=16384
qwen3_vl_min_nonwhite_ratio=0.02
qwen3_vl_min_variance=15.0
qwen3_vl_min_edge_energy=0.01
```

## 2026-02-05 CONFIG 변경 스냅샷
아래는 `src/rag/config.py`에서 **토큰 길이 초과 대응을 위해 K 값을 줄인 이후**의 기본값이다.

```text
max_top_k=5
bm25_top_k=8
mmr_candidate_pool=10
```

## 2026-02-06 시도 기록
아래는 2026-02-06 기준으로 진행한 주요 시도와 결과 요약이다.

- **LangChain SummaryMemory + SQLite 병행 사용 시도**
  - SQLChatMessageHistory로 SQLite에 대화 기록 저장
  - ConversationSummaryMemory로 요약 유지/갱신
- **문제: vLLM 엔진 이중 실행으로 VRAM 부족**
  - SummaryMemory가 vLLM 래퍼를 새로 띄우면서 **vLLM 엔진이 2개 실행**
  - RTX 5080 16GB 환경에서 **KV 캐시 메모리 부족으로 엔진 초기화 실패**

> **SQLITE만 사용 결정** 

## 추후 작업

- 웹 UI로 vLLM을 실행 시 고성능 컴퓨터가 요구된다.
(RAM 부담을 낮추려면 model max lengh 등 vLLM 설정을 조정하거나 k 값을 줄여 컨텍스트 길이를 낮추어야 한다)
- 성능을 어느 정도 양보해야 할 수 있다.
