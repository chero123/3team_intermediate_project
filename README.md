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

## 추후 작업

- 청킹 메카니즘 개선
- 자가 커스텀 TTS 모델 허깅페이스 업로드 후 다운로드받을 수 있게 할 예정