# vLLM 개념/구현/기능 정리

## 개념
- vLLM은 **GPU 상에서 LLM 추론을 고속화**하기 위한 엔진이다.
- 핵심 아이디어는 **PagedAttention 기반 KV 캐시 관리**로, 긴 컨텍스트나 다중 요청 상황에서 **메모리 효율을 높이고 처리량(throughput)을 개선**한다.
- 일반적인 PyTorch `generate()` 대비 **동시성/스케줄링/캐시 관리가 강화**되어 서비스형 추론에 적합하다.

## 이 프로젝트에서 vLLM을 사용하는 이유
- 로컬 LLM 추론을 빠르게 수행하기 위해 vLLM을 사용한다.
- CLI/Gradio에서 **응답 지연을 줄이고 처리량을 확보**하기 위한 선택이다.

## 동작 방식 요약
- **모델 로드** 시 GPU 메모리에 가중치와 KV 캐시 메타 정보를 준비한다.
- **요청 단위로 프롬프트를 처리**하고, 생성 토큰을 스케줄러가 관리한다.
- KV 캐시는 **페이지 단위로 할당/해제**되어 긴 컨텍스트에 유리하다.

## 주요 기능
1. **PagedAttention(KV 캐시 관리)**
   - 긴 컨텍스트에서도 메모리 파편화를 줄인다.
2. **스케줄링**
   - 여러 요청을 배치로 묶어 처리량을 높인다.
3. **프리필/디코드 분리**
   - 프리필 단계와 디코드 단계가 분리되어 효율적으로 운영된다.
4. **양자화 지원**
   - bitsandbytes, fp8 등으로 VRAM 사용량을 줄일 수 있다.
5. **max_model_len 제어**
   - 최대 컨텍스트 길이를 지정해 메모리/성능 균형을 맞춘다.

## 프로젝트 적용 위치
- 로컬 LLM 실행:
  - `src/rag/llm.py`에서 vLLM 엔진을 생성해 사용한다.
  - `scripts/run_query.py`는 로컬 vLLM 기반 실행 경로다.
- VLM(Qwen3-VL)에도 vLLM 경로가 포함되어 있다.

## 설정 항목 (config 기준)
- `llm_model_path`: 로컬 LLM 경로
- `llm_quantization`: vLLM 양자화 설정 (예: bitsandbytes)
- `device`: 실행 디바이스 (cuda)
- `qwen3_vl_max_model_len`: VLM 컨텍스트 길이
- `qwen3_vl_gpu_memory_utilization`: VLM GPU 메모리 사용률

## 운영/디버깅 포인트
- **max_model_len 초과**:
  - 입력 토큰이 max_model_len을 넘으면 예외가 발생한다.
  - 해결책: 컨텍스트 길이 축소, top_k 축소, max_model_len 상향.
- **VRAM 부족**:
  - `gpu_memory_utilization`이 낮으면 KV 캐시 확보 실패 가능.
  - 해결책: 모델 크기/컨텍스트 길이 조정.
- **첫 실행 지연**:
  - 모델 로드/컴파일로 첫 응답 지연이 길 수 있다.
  - 이후 요청은 캐시 덕분에 빨라진다.

## 참고
- vLLM은 서비스형 추론에서 특히 효율적이나,
  **컨텍스트 길이와 VRAM 요구량이 비례**하므로
  설정을 보수적으로 관리해야 한다.
