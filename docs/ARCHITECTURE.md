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
4. `load_documents()` 내부에서 PDF/HWP/DOCX를 파싱한다.
   - PDF는 `pdfplumber`(기본) 또는 `fitz`를 사용한다.
   - PDF는 이미지 추출 시 Qwen3-VL(vLLM)로 수치/표 정보를 추출한다.
   - 이미지가 비어 있거나 로고/단색 등 저정보 이미지는 필터링한다.
   - 중복 이미지는 해시로 스킵한다.
   - VLM 결과는 메타데이터(`vlm_image_text`, `vlm_image_text_present`)에 기록된다.
5. `chunk_documents()`가 텍스트를 청킹한다.
6. `FaissVectorStore.build()`가 인덱스를 만든다.
7. `FaissVectorStore.save()`가 인덱스를 `data/index`에 저장한다.
8. 상태 파일 `data/index_status.json`이 완료 상태로 업데이트된다.
9. 샘플 청크가 `data/chunk_preview.json`에 저장된다.

## 실행 흐름
1. `scripts/run_query.py`가 실행된다.
2. `RAGPipeline`이 저장된 인덱스를 로드한다.
3. LangGraph가 다음 순서로 실행된다.
   - `analyze_query`: 질문을 분석하고 RetrievalPlan을 만든다.
   - `retrieve`: 전략에 맞게 검색하고 리랭크한다.
   - `generate`: 컨텍스트를 기반으로 답변을 생성한다.
    - `rewrite`: 생성 답변을 스타일 규칙에 맞게 리라이트한다.
4. 답변이 출력되고, 선택 시 TTS가 수행된다.

## SQLite 멀티턴 메모리
- 멀티턴에서 이전 턴 문서 집합을 재사용하기 위해 **SQLite 기반 세션 메모리**를 사용한다.
- 세션이 존재하면 **명시적 리셋/후속 힌트/질문 유사도** 규칙으로 문맥 유지 여부를 결정한다.
- followup으로 판정되면 저장된 문서 ID를 필터로 적용해 문서 흐름이 바뀌지 않도록 한다.
- 프롬프트에는 **[이전 턴]**(직전 질문+답변 전체)과 **[이전 문서]**(doc_id 목록)가 함께 들어가며,
  이전 턴은 참고용으로만 제공된다.
- 구현/사용법/스키마 상세는 `docs/sqlite.md`를 참고한다.

## 사용법 및 주의사항 (로컬 vLLM vs OpenAI)
### 로컬 vLLM 버전
- 실행: `uv run scripts/run_query.py --tts --device cuda`
- 특징:
  - 로컬 모델(vLLM)을 직접 로드한다.
  - 첫 질의는 **모델 로드/컴파일** 때문에 느릴 수 있다.
  - 이후 질의는 캐시가 유지되어 속도가 안정된다.
- 주의사항:
  - GPU 메모리 여유가 부족하면 엔진 초기화가 실패할 수 있다.
  - `max_model_len`이 프롬프트 길이보다 작으면 오류가 난다.
  - 비교/다문서 질문은 입력 토큰이 커질 수 있어 길이 관리가 필요하다.

### OpenAI 버전
- 실행: `uv run scripts/run_query_openai.py --tts --device cuda`
- 특징:
  - OpenAI API를 사용하므로 로컬 모델 로드가 없다.
  - 네트워크 왕복이 포함된다.
  - 분류/리라이트에 다른 모델을 사용한다.
- 주의사항:
  - API 키가 필요하다 (`OPENAI_API_KEY`).
  - 네트워크 지연이나 레이트 제한에 영향을 받는다.
  - 모델별 지원 파라미터(temperature, max_output_tokens 등)가 다를 수 있다.

## 스트리밍 출력
### 개념
- 스트리밍은 모델 응답을 **완료 후 일괄 출력**하는 대신 **중간 결과를 점진적으로 출력**하는 방식이다.
- 사용자 체감 지연을 줄이고, 긴 답변에서도 **초반 문장을 빠르게 제공**한다.
- TTS는 문장 단위로 생성/재생해 “텍스트 -> 음성” 지연을 최소화한다.

### CLI 스트리밍 방식(run_query)
- 개념: CLI는 요약 텍스트를 먼저 출력하고 이후 **문장 단위로 TTS를 순차 처리**한다.
- 구현 위치: `scripts/run_query.py`, `scripts/run_query_openai.py`
- 흐름:
  1) `pipeline.ask(question)`으로 답변 생성
  2) `_clean_for_tts()`로 불필요한 문자 정리
  3) `_split_sentences()`로 문장 분리 (소수점 보호 포함)
  4) 문장별로 `_split_sentence_for_tts()` 호출
  5) 각 조각을 `infer_tts_onnx()`에 전달해 순차 재생
- 문장 분리 규칙:
  - 기본 구두점(.!? 등) 기준으로 분리
  - 숫자 소수점(`5.5개월`)은 분리 대상에서 제외
- 주의사항:
  - CLI는 텍스트를 한 번에 출력하므로 체감 지연이 존재
  - 문장 단위 TTS는 자연스럽지만 호출 횟수가 증가

### Gradio 출력 방식(UI)
- 개념: Gradio는 **완료 후 일괄 출력**이 기본이며, 현재 파일도 그 방식으로 동작한다.
- 구현 위치: `server/app.py`
- 동작 방식:
  - `chat_with_tts()`가 generator로 동작하지만 **TTS는 단 한 번만 합성**한다.
  - 흐름:
    1) 텍스트 답변 생성
    2) 텍스트를 즉시 출력
    3) 전체 답변을 한 번만 TTS 합성 후 오디오 출력
- Gradio Chatbot 포맷: `history`를 `{"role": "...", "content": "..."}` 형태로 유지
- 주의사항:
  - Gradio 기본 Audio 컴포넌트는 **연속 스트리밍 재생을 지원하지 않는다**
    (새 오디오가 들어오면 재생이 초기화됨)

## ONNX TTS 상세 문서
- `docs/ONNX_TTS.md` 참고

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

## 메타데이터 필터링
- 질문에서 기관명/사업명 등을 추출해 메타데이터 필터로 사용한다.
- 예시:
  - `issuer = "한국생산기술연구원"`
  - `project_name = "2세대 전자조달시스템"`
- 필터가 있으면 검색 결과 중 해당 값과 일치하는 청크만 유지한다.
- 다문서 질문(needs_multi_doc=True)에서는 필터를 끈다.
  - 비교/다문서 질문에서 한쪽 문서가 잘리는 것을 막기 위함이다.

## 검색 전략 상세
- similarity: 벡터 유사도 기반으로 가장 가까운 청크를 찾는다. 빠르고 안정적이다.
- mmr: 유사도와 다양성을 동시에 고려해 중복을 줄인다.
  - 여기서 “다양성”은 이미 선택된 청크들과의 내용 유사도가 낮은 청크를 우대한다는 뜻이다.
  - 즉, 같은 문단/같은 표 설명만 반복되는 것을 피하고 서로 다른 단락·섹션의 정보를 섞는다.
- bm25: 키워드/문구 일치에 강한 sparse 검색이다.
- rrf: 여러 검색 결과의 순위를 누적해 안정적인 상위 결과를 만든다.  
  - similarity + mmr + bm25를 결합  
  - bm25 인덱스가 없으면 similarity + mmr만 결합  
  - `bm25_weight`로 bm25 영향도를 조절함

## 전략 선택 이유
- single 질문: 단일 문서에서 정확한 답을 찾는 문제로 가정  
  - similarity만 사용해 노이즈를 줄이고 속도를 확보  
- multi/compare 질문: 여러 문서를 섞어야 하므로 다양성과 키워드 신호가 중요  
  - rrf로 similarity + mmr + bm25를 결합해 폭넓게 커버  
- followup 질문: 이전 맥락을 좁혀서 찾는 게 중요  
  - similarity로 집중된 검색을 수행

### MMR 예시
- 질문: “사업 목적과 범위 요약”
  - similarity만 쓰면 “목적” 관련 청크가 반복적으로 상위에 몰릴 수 있다.
  - MMR은 “목적” 청크 1개를 뽑은 뒤, 다음에는 “범위/일정/요구사항”처럼  
    서로 다른 내용의 청크를 섞어 중복을 줄인다.

## 리랭크
- `CrossEncoderReranker`가 쿼리/청크 쌍을 재점수화해 상위 `top_n`만 유지한다.
  
## OpenAI 전용 모델 분리
- 본문 생성: `config.openai_model` (`gpt-5-mini`)
- 분류/리라이트: `openai_pipeline.OpenAIRAGPipeline.small_llm` (`gpt-4o-mini`)
- 모델을 나눈 이유: gpt-5 계열은 기본적으로 추론 모델이라 토큰 수가 너무 많이 든다.
                  따라서 리라이트와 분류 같은 많은 토큰 수가 필요 없는 작업에는 gpt-4 계열을 사용한다.

## 청킹 모니터링
- `data/index_status.json`에 문서 수, 청크 수, 길이 통계가 기록
- `data/chunk_preview.json`에 청크 샘플이 저장
- `scripts/inspect_chunks.py`로 즉시 출력 확인이 가능

## VLM(Qwen3-VL) 파이프라인 (PDF 이미지)
- 모델: `models/qwen3-vl-8b` (로컬 경로)
- 실행: `data.py`에서 vLLM 기반으로 로드
- 입력: PDF 내 이미지 추출 -> PIL 변환 -> Qwen3-VL 프롬프트
- 출력: 수치/표/지표 중심 텍스트
- 필터링:
  - 최소 해상도
  - 저분산/단색
  - 낮은 엣지 에너지
  - 중복 이미지 해시 제거
- 설정 위치: `config.py`
  - `qwen3_vl_enabled`
  - `qwen3_vl_model_path`
  - `qwen3_vl_max_tokens`
  - `qwen3_vl_gpu_memory_utilization`
  - `qwen3_vl_max_model_len`
  - `qwen3_vl_min_image_pixels`, `qwen3_vl_min_variance`, `qwen3_vl_min_edge_energy`, `qwen3_vl_min_nonwhite_ratio`, `qwen3_vl_dedupe_images`

## VLM 이미지 필터링 도입 배경
- 초기에는 로고/빈 페이지/단색 이미지까지 전부 VLM에 전달되어
  추론 시간이 급격히 증가하고 "수치 없음" 같은 무의미 결과가 대량 생성됨.
- 이를 해결하기 위해 이미지 필터링을 추가해 의미 있는 이미지(표/그래프/지표 가능성)만 선별한다.
- 적용 기법:
  - 최소 픽셀 수로 저해상도/아이콘류 제거
  - 그레이스케일 분산(variance)로 단색/로고 제거
  - 비백색 비율로 빈 페이지 제거
  - 엣지 에너지로 텍스트/도표 없는 이미지 제거
  - 해시 기반 중복 이미지 제거

## 추후 수정
- 청킹 기준 변경: `data.py`의 `simple_chunk()` 수정
- 검색 전략 변경: `retrieval.py`의 `Retriever` 수정
- 프롬프트 변경: `llm.py`의 `build_prompt()` 수정
- 인덱싱 통계 변경: `indexing.py`의 `IndexStatus` 확장
