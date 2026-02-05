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

- vLLM + 4bits(배포 최적화 옵션)로 실행할 시, GPT API 사용 버전에 거의 근접한 속도를 구현할 수 있다.
- 다만, 프롬프트 준수율은 GPT가 우수하다.
- GPT-5 계열은 추론 모델이라 GPT-4 계열과 달리 높은 토큰 수 상한이 필요
- CLI에서 run_query 실행 시 음성에 잡음이 끼는 현상이 있는데, 실제 생성된 음성 파일과 WebUI 등으로 확인 시에는 잡음이 없다.
- Gradio, Streamlit과 같은 데모 프레임워크로는 실시간 합성을 실현할 수 없다.
- 실시간 합성을 재현하려면 커스텀 UI나 다른 방식을 사용해야 한다.

## 추후 작업

- 현재 유사도, MMR, BM25, RRF, Cross Rerank를 전부 사용한 덕에 속도 측면에 병목이 확인된다.
- 웹 UI로 vLLM을 실행 시 고성능 컴퓨터가 요구된다. (RAM 부담을 낮추려면 model max lengh 등 vLLM 설정 조정 필요)
- 성능을 어느 정도 양보해야 할 수 있다.