# RAG 구조/로직 상세 문서

## 전체 구조 요약
- 데이터 처리 단계에서는 문서를 로드하고 청킹/임베딩 후 FAISS 인덱스를 저장한다.
- 에이전트 실행 단계에서는 저장된 인덱스를 로드하고 LangGraph로 검색 및 생성 흐름을 실행한다.

## 디렉토리 구조
- `src/rag/`
  - `config.py` 전역 설정을 담당한다.
  - `types.py` 데이터 구조를 정의한다.
  - `data.py` 문서 로딩/파싱/청킹을 처리한다.
  - `embeddings.py` 임베딩 로더를 제공한다.
  - `indexing.py` FAISS 인덱싱과 상태 기록을 담당한다.
  - `retrieval.py` 질문 분석/검색 전략/리랭크를 수행한다.
  - `llm.py` vLLM 기반 생성과 프롬프트 구성을 담당한다.
  - `pipeline.py` LangGraph 온라인 파이프라인을 정의한다.
- `scripts/`
  - `build_index.py` 인덱싱 실행 스크립트다.
  - `run_query.py` 질의 실행 스크립트(REPL 지원)다.
  - `inspect_chunks.py` 청킹 결과를 확인한다.

## 인덱싱 흐름
1. `scripts/build_index.py`가 실행된다.
2. `Indexer.build_index()`가 상태 파일을 생성한다.
3. `load_documents()`가 문서와 메타데이터를 로드한다.
4. `chunk_documents()`가 텍스트를 청킹한다.
5. `FaissVectorStore.build()`가 인덱스를 만든다.
6. `FaissVectorStore.save()`가 인덱스를 `data/index`에 저장한다.
7. 상태 파일 `data/index_status.json`이 완료 상태로 업데이트된다.
8. 샘플 청크가 `data/chunk_preview.json`에 저장된다.

## 실행 흐름
1. `scripts/run_query.py`가 실행된다.
2. `RAGPipeline`이 저장된 인덱스를 로드한다.
3. LangGraph가 다음 순서로 실행된다.
   - `analyze_query`: 질문을 분석하고 RetrievalPlan을 만든다.
   - `retrieve`: 전략에 맞게 검색하고 리랭크한다.
   - `generate`: 컨텍스트를 기반으로 답변을 생성한다.
4. 답변이 출력되고, 선택 시 TTS가 수행된다.

## 청킹 모니터링
- `data/index_status.json`에 문서 수, 청크 수, 길이 통계가 기록
- `data/chunk_preview.json`에 청크 샘플이 저장
- `scripts/inspect_chunks.py`로 즉시 출력 확인이 가능

## 추후 수정
- 청킹 기준 변경: `data.py`의 `simple_chunk()` 수정
- 검색 전략 변경: `retrieval.py`의 `Retriever` 수정
- 프롬프트 변경: `llm.py`의 `build_prompt()` 수정
- 인덱싱 통계 변경: `indexing.py`의 `IndexStatus` 확장
