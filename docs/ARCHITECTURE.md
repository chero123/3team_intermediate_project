# RAG 구조/로직 상세 문서

## 전체 구조 요약
- 데이터 처리 단계에서는 문서를 로드하고 청킹/임베딩 후 FAISS 인덱스를 저장한다.
- 에이전트 실행 단계에서는 저장된 인덱스를 로드하고 LangGraph로 검색 및 생성 흐름을 실행한다.
- 로컬 LLM(vLLM)과 OpenAI LLM은 파이프라인/LLM 모듈이 분리되어 있다.

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
  - `openai_llm.py` OpenAI Responses/Chat 기반 생성과 프롬프트 구성을 담당한다.
  - `openai_retrieval.py` OpenAI용 질문 분류/검색 전략/리랭크를 수행한다.
  - `openai_pipeline.py` OpenAI 파이프라인을 정의한다.
- `scripts/`
  - `build_index.py` 인덱싱 실행 스크립트다.
  - `run_query.py` 질의 실행 스크립트(REPL 지원)다.
  - `run_query_openai.py` OpenAI 질의 실행 스크립트(REPL 지원)다.
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
    - (OpenAI 전용) `rewrite`: 생성 답변을 스타일 규칙에 맞게 리라이트한다.
4. 답변이 출력되고, 선택 시 TTS가 수행된다.

## LangGraph 그래프 구조 (도식)
로컬 파이프라인(`pipeline.py`):
```
START
  │
  ▼
analyze_query
  │
  ▼
retrieve
  │
  ▼
generate
  │
  ▼
rewrite
  │
  ▼
END
```

OpenAI 파이프라인(`openai_pipeline.py`):
```
START
  │
  ▼
analyze_query
  │
  ▼
retrieve
  │
  ▼
generate
  │
  ▼
rewrite
  │
  ▼
END
```

## 질문 유형 분류 기준 (single / multi / compare / followup)
분류는 `retrieval.py`(로컬)와 `openai_retrieval.py`(OpenAI)에서 동일한 규칙을 사용한다.

- 기본 흐름
  1) LLM 분류가 가능하면 `classify_query_type()` 결과를 우선 사용한다.  
  2) LLM 분류가 비어 있으면 휴리스틱 키워드로 폴백한다.

- compare
  - 키워드 포함: `비교`, `차이`, `서로`, `vs`, `대비`
  - 전략: `rrf_strategy`, `top_k` 상향, `needs_multi_doc=True`

- followup
  - 키워드 포함: `그럼`, `그렇다면`, `또`, `추가로`, `더`, `이어서`, `방금`, `앞서`
  - 또는 대화 히스토리가 있고 질문 길이가 짧을 때
  - 전략: `similarity_strategy`, `top_k` 하향, `needs_multi_doc=False`

- multi
  - 키워드 포함: `여러`, `모든`, `각각`, `기관`
  - 전략: `rrf_strategy`, `top_k` 중간, `needs_multi_doc=True`

- single
  - 위 조건에 해당하지 않는 일반 질문
  - 전략: `similarity_strategy`, `needs_multi_doc=False`

## 검색 전략 상세
- similarity: 벡터 유사도 기반 상위 `top_k` 검색
- mmr: 다양성 기반 MMR 검색 (후보 풀 `mmr_candidate_pool`)
- rrf: similarity + mmr 결과를 RRF로 결합

## 리랭크
- `CrossEncoderReranker`가 쿼리/청크 쌍을 재점수화해 상위 `top_n`만 유지한다.
  
## OpenAI 전용 모델 분리
- 본문 생성: `config.openai_model` (예: `gpt-5-mini`)
- 분류/리라이트: `openai_pipeline.OpenAIRAGPipeline.small_llm` (예: `gpt-4.1-mini`)

## 청킹 모니터링
- `data/index_status.json`에 문서 수, 청크 수, 길이 통계가 기록
- `data/chunk_preview.json`에 청크 샘플이 저장
- `scripts/inspect_chunks.py`로 즉시 출력 확인이 가능

## 추후 수정
- 청킹 기준 변경: `data.py`의 `simple_chunk()` 수정
- 검색 전략 변경: `retrieval.py`의 `Retriever` 수정
- 프롬프트 변경: `llm.py`의 `build_prompt()` 수정
- 인덱싱 통계 변경: `indexing.py`의 `IndexStatus` 확장
