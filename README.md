# 3team_intermediate_project

입찰/공고 문서를 대상으로 하이브리드 검색 기반 질의응답(RAG)을 수행하고, 
답변을 TTS로 합성한 후 SQLite에 저장하는 프로젝트입니다.

## 0. 개인 로그

- [안호성](https://hobi2k.github.io/category/Intermediate_RAG/)
- [김나연](https://github.com/chero123/3team_intermediate_project/tree/kim-nayeon/logs)

## 1. 핵심 기능

프로젝트는 다음 기능을 중심으로 동작합니다.

1. 하이브리드 검색 RAG
- Dense 검색: Chroma + OpenAI 임베딩(`text-embedding-3-small`)
- Sparse 검색: BM25
- 결합 검색: EnsembleRetriever (가중치 기본값 Dense 0.6 / Sparse 0.4)

2. 대화형 응답
- OpenAI Chat 모델로 답변 생성
- 대화 이력을 반영해 후속 질문 처리

3. TTS 합성/재생
- ONNX 기반 TTS 모델 사용
- `TTSWorker`가 Queue + Thread 구조로 문장 단위 비동기 처리
- 새 질문 입력 시 이전 재생 즉시 중단 가능

4. 대화/피드백 저장
- SQLite(`data/chat_log.sqlite`)에 질문·답변·피드백 저장
- 피드백은 `1(좋아요)`, `-1(싫어요)`, `NULL(미평가)` 방식

## 2. 현재 주요 엔트리포인트

`TextParsing/rag_system.py`
- CLI 기반 하이브리드 RAG + 스트리밍 출력 + TTS

`TextParsing/app.py`
- Streamlit 기반 웹 UI
- 하이브리드 검색, 음성 재생, 피드백 저장

`TextParsing/create_vectordb.py`
- 문서 청킹 후 Chroma DB 구축

`TextParsing/text_parshing.py`
- 원본 문서(HWP/PDF 등) 텍스트 추출/정제 스크립트

`TextParsing/tts_worker.py`
- TTS 비동기 워커 구현

`TextParsing/memory_store.py`
- SQLite 저장소 구현

## 3. 폴더 구조 요약

```text
3team_intermediate_project/
├─ TextParsing/
│  ├─ app.py
│  ├─ rag_system.py
│  ├─ create_vectordb.py
│  ├─ text_parshing.py
│  ├─ tts_worker.py
│  ├─ memory_store.py
│  └─ tts_runtime/
│
├─ hwp_to_pdf_to_md/
│  ├─ hwp_to_pdf.py
│  ├─ pdf_to_md.py
│  └─ extract_tables_to_md.py
│
├─ data/
│  ├─ chroma_db/
│  ├─ answer/
│  └─ chat_log.sqlite
│
├─ models/
└─ docs/
   ├─ tts_worker.md
   └─ sqlite.md

```

## 4. 환경 준비

### 4.1 uv 사용

```bash
uv sync
```

### 4.2 pip 사용

```bash
pip install -r requirements.txt
```

## 5. 환경 변수

프로젝트 루트(`3team_intermediate_project/.env`)에 OpenAI 키를 설정합니다.

```env
OPENAI_API_KEY=sk-...
```

## 6. 데이터 파이프라인

### 6.0 기존(.txt) 기반 파싱 방식 (Legacy)
초기 버전에서는 PDF를 직접 `.txt` 형식으로 변환하여 사용하였습니다.

```bash
python TextParsing/text_parshing.py
```
기본 텍스트만 추출하는 .txt 기반 단순 파싱 방식

### 6.1 원본 문서 정규화 (HWP → PDF → MD)

원본 데이터(HWP 96개 + PDF 4개)는 다음 순서로 정규화됩니다.

① HWP → PDF 변환  
```bash
python hwp_to_pdf_to_md/hwp_to_pdf.py
```
HWP 문서를 PDF로 변환하여 입력 포맷을 통일

② PDF → Markdown 추출
```bash
python hwp_to_pdf_to_md/pdf_to_md.py
```
PDF를 Markdown으로 변환

페이지 시작에 다음 태그 삽입
`<!-- page: 1 -->`
페이지 단위 구조를 유지하여 이후 검색·근거 추적에 활용

### 6.2 VLM 기반 표 추출 및 삽입
```bash
python extract_tables_to_md.py
```
VLM을 사용해 PDF 내 표를 별도 추출

해당 페이지 하단에 Markdown 형태로 추가 삽입

표 영역은 다음 태그로 구분

`<!-- tables: start page 3 -->`
(표 내용)
`<!-- tables: end page 3 -->`
삭제/교체가 아닌 추가 삽입 방식을 사용하여 원문 손실을 방지

### 6.3 벡터 DB 생성

```bash
python TextParsing/create_vectordb.py
```

`create_vectordb.py`는 기본적으로 다음을 수행합니다.
- 문서로드
- 페이지 단위 분리 후 청킹
- 표 블록은 하나의 단위로 보호
- RecursiveCharacterTextSplitter 기반 청킹
- OpenAI 임베딩 생성
- Chroma DB(`data/chroma_db`) 저장



## 7. 실행 방법

### 7.1 CLI

```bash
python TextParsing/rag_system.py
```

실행 후 프롬프트에서 질문을 입력합니다.

### 7.2 웹 UI (Streamlit)

```bash
python streamlit run TextParsing/app.py
```

## 8. TTS 동작 개요

`rag_system.py`/`app.py`는 답변 텍스트를 문장 단위로 분리해 `TTSWorker`에 전달합니다.

`TTSWorker`는 다음 순서로 동작합니다.
- Queue(FIFO)에 문장 적재
- 워커 스레드에서 문장별 ONNX 추론
- 세션 단위 WAV 누적 저장
- 플레이어(`ffplay` 또는 `mpv`)로 재생
- 새 질문 시 기존 재생/큐를 취소

자세한 내용은 [여기](docs/tts_worker.md)를 참고하세요.

## 9. 리트리버 전략

검색 전략은 다음 3단계로 동작합니다.

1. 히스토리 기반 질의 재구성(History-aware Query Rewrite)
- `chat_history`가 있으면 현재 질문을 독립 질문으로 재구성합니다.
- 구현 위치: `TextParsing/rag_system.py`, `TextParsing/app.py`

2. 하이브리드 검색(Ensemble)
- Dense: Chroma + OpenAI 임베딩(`text-embedding-3-small`)
- Sparse: BM25
- 결합: `EnsembleRetriever(weights=[0.6, 0.4])`

3. 컨텍스트 결합 후 생성
- 검색 문서를 프롬프트 컨텍스트로 결합해 최종 답변을 생성합니다.

## 10. History 처리 방식

대화 이력은 아래 두 경로에서 동시에 사용됩니다.

1. 검색 단계
- 과거 대화를 이용해 현재 질문을 독립 질문으로 바꿉니다.
- 후속 질문의 검색 누락을 줄이는 목적입니다.

2. 생성 단계
- `MessagesPlaceholder("chat_history")`를 통해 프롬프트에 대화 이력을 함께 전달합니다.
- 같은 질문이라도 이전 문맥을 반영한 답변이 가능해집니다.

이력 저장 범위:
- 메모리: 실행 중 체인 컨텍스트용(`chat_history`)
- SQLite: 질문/답변/피드백 영속 저장(`data/chat_log.sqlite`)

## 11. SQLite 저장 구조

기본 테이블은 `chat_log` 하나를 사용합니다.
- `id` (PK, AUTOINCREMENT)
- `question`
- `answer`
- `rating` (NULL/1/-1)
- `created_at` (`YYYY-MM-DD HH:MM:SS`)

자세한 스키마와 조회 방법은 [여기](docs/sqlite.md)를 참고하세요.

## 12. 플레이어 관련 주의 사항

`ffplay`/`mpv`/PulseAudio 또는 PipeWire 환경에 따라 로그 경고가 출력될 수 있습니다.
필요 시 시스템 오디오 스택 설정을 확인하세요.
