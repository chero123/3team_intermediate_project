# TTS RAG 초기 아키텍쳐

아키텍쳐에 관한 상세 내용은 [여기](docs/ARCHITECTURE.md)

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

- vLLM + 4bits(배포 최적화 옵션)로 실행할 시, GPT API 사용 버전과 속도 차이가 거의 없음
- 다만, (EEVE 모델과 비교해서) 프롬프트 준수율은 GPT가 우수
- GPT-5 계열은 추론 모델이라 GPT-4 계열과 달리 높은 토큰 수 상한이 필요
- CLI에서 run_query 실행 시 음성에 잡음이 끼는 현상이 있는데, 실제 생성된 음성 파일에는 잡음이 없음을 보아 WebUI 등으로 확인 필요

## 추후 작업

- 우정님이 올리 주신 파싱 및 청킹 메카니즘 반영
- vLLM 생성 파이프라인 개선