## SQLite 멀티턴 메모리

이 문서는 RAG 멀티턴 대화를 위해 도입한 SQLite 기반 세션 메모리의 목적, 구조, 동작 방식, 그리고 사용 방법을 정리한다.

### 목적
- 멀티턴 대화에서 **이전 턴에 사용된 문서 집합**을 재사용해 후속 질문의 문맥을 유지한다.
- "방금 말한 문서 기준으로 수행된 추가 질문" 같은 시나리오에서 **다른 문서로 흐르는 문제를 차단**한다.
- UI/CLI/REST에서 동일한 방식으로 동작하도록 **세션 단위의 저장소**를 제공한다.

## 핵심 개념

### 세션(Session)
- 하나의 대화 흐름을 식별하는 값
- 문자열 ID로 관리 (uuid 기반)
- 세션별로 "최근 문서 ID 집합"과 "최근 질문/답변"을 저장한다.
- 생성 프롬프트에는 **[이전 턴] + [이전 문서]**가 함께 들어가며,
  이전 턴은 **질문+답변 전체**를 그대로 전달한다.

### 문서 ID(doc_id)
- 인덱싱 시 문서 이름 기반으로 부여됨
- 청크의 메타데이터(`chunk.metadata["doc_id"]`)에 저장됨
- 멀티턴의 필터 기준 키로 사용
- 프롬프트의 **[이전 문서]** 섹션에도 그대로 노출됨

## 구현 위치

- 저장소 구현: `src/rag/memory_store.py`
- 설정값: `src/rag/config.py`
  - `memory_db_path = "data/session_memory.sqlite"`
- 파이프라인 통합:
  - 로컬: `src/rag/pipeline.py`
  - OpenAI: `src/rag/openai_pipeline.py`
- 검색 필터 적용:
  - 로컬: `src/rag/retrieval.py`
  - OpenAI: `src/rag/openai_retrieval.py`
- UI/REST 연동:
  - `server/app.py`

## DB 스키마

### session_state
세션의 마지막 대화 상태를 저장한다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| session_id | TEXT (PK) | 세션 식별자 |
| last_question | TEXT | 마지막 질문 |
| last_answer | TEXT | 마지막 답변 |
| last_question_type | TEXT | 분류된 질문 유형 (single/multi/compare/followup) |
| updated_at | REAL | 갱신 시각 (epoch) |

### session_docs
세션별 문서 ID 집합을 저장한다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| session_id | TEXT | 세션 식별자 |
| doc_id | TEXT | 문서 ID |
| rank | INTEGER | 저장 당시 순서 |
| score | REAL | 점수 (현재 미사용) |
| created_at | REAL | 저장 시각 |
| PK | (session_id, doc_id) | 복합 PK |


## 동작 흐름

### 1) 첫 질문 처리
1. 질문 -> 검색 수행
2. 검색 결과 청크에서 `doc_id` 추출
3. `session_docs`에 문서 ID 목록 저장
4. `session_state`에 마지막 질문/답변/질문유형 저장

### 2) 후속 질문(followup) 판단 규칙(혼합)
SQLite 세션이 존재할 때, 다음 규칙으로 문맥 유지/전환을 결정한다.

1. **명시적 리셋**  
   - 질문에 `memory_reset_keywords`가 포함되면 세션 문서 목록을 비우고 새 문맥으로 처리한다.
2. **후속 질문 힌트**  
   - 질문이 짧거나(`len < 30`)  
   - `memory_followup_keywords`가 포함되면 followup으로 본다.
3. **질문 유사도**  
   - 위 두 조건에 해당하지 않으면, 현재 질문과 이전 질문의 임베딩 유사도를 계산한다.  
   - 유사도가 `memory_similarity_threshold`보다 낮으면 문맥 전환으로 판단한다.  
   - 유사도가 충분히 높으면 followup으로 처리한다.

followup으로 판정되면:
1. `session_docs`에서 doc_id 리스트 로드
2. 검색 결과를 `doc_id_filter`로 제한
3. 다른 문서로 흐르지 않고 **기존 문서 집합 안에서만 검색**
4. 프롬프트에 `[이전 턴]`(질문+답변 전체)과 `[이전 문서]`를 추가해
   후속 질문이 문맥을 안정적으로 참조하도록 한다.


## 코드 동작 요약

### 저장소 구현 (SessionMemoryStore)
파일: `src/rag/memory_store.py`

주요 메서드
- `load_doc_ids(session_id) -> List[str]`  
  저장된 문서 ID를 순서대로 반환
- `save_doc_ids(session_id, doc_ids)`  
  세션 문서 ID 리스트를 저장
- `update_state(session_id, last_question, last_answer, last_question_type)`  
  세션 상태 업데이트
- `get_last_question(session_id) -> Optional[str]`  
  마지막 질문 반환
- `get_last_answer(session_id) -> Optional[str]`  
  마지막 답변 반환
- `get_last_turn(session_id) -> Tuple[Optional[str], Optional[str]]`  
  마지막 질문/답변 동시 반환
- `has_session(session_id) -> bool`  
  세션 존재 여부 확인
- `clear_session_docs(session_id)`  
  세션 문서 목록만 초기화(문맥 리셋용)

### 파이프라인 통합
파일: `src/rag/pipeline.py`, `src/rag/openai_pipeline.py`

- `ask(question, session_id)`에 session_id 전달
- `_node_analyze_query`에서 followup이면 저장된 doc_id 로드 → `plan.doc_id_filter`에 주입
- `_node_retrieve` 이후 doc_id 목록 저장
- `_node_generate`에서 `[이전 턴]` + `[이전 문서]`를 프롬프트에 포함

### 검색 필터 적용
파일: `src/rag/retrieval.py`, `src/rag/openai_retrieval.py`

- `_apply_doc_id_filter(chunks, doc_id_filter)`로 doc_id 기준 제한
- RRF(similarity/mmr/bm25)에도 동일 필터 적용


## 사용 방법

### CLI
CLI는 동일 프로세스 내에서 파이프라인이 유지되므로 **세션이 자동 유지**된다.

```
uv run scripts/run_query.py --tts --device cuda
```

또는 OpenAI 버전:

```
uv run scripts/run_query_openai.py --tts --device cuda
```

### REST API
REST에서는 `session_id`를 직접 넘겨야 한다.

요청 예시:

```json
POST /api/ask
{
  "question": "첫 질문",
  "provider": "local",
  "session_id": "user-session-001"
}
```

후속 질문도 동일 `session_id` 사용:

```json
POST /api/ask
{
  "question": "추가 질문",
  "provider": "local",
  "session_id": "user-session-001"
}
```

### Gradio UI
Gradio는 `gr.State`로 세션 ID를 유지한다.
- 페이지 새로고침 시 세션이 바뀜
- 동일 브라우저 세션에서는 멀티턴 유지


## 주의사항

- followup이 아닌 질문은 문서 필터를 사용하지 않는다.
- 질문 유사도/키워드 규칙이 오판되면 문서가 과도하게 제한될 수 있다.
- 문서 ID는 인덱싱 기준이므로, 인덱스 재생성 시 문서 ID가 바뀌면 세션 메모리와 불일치가 발생할 수 있다.
- SQLite 파일 위치는 `config.py`의 `memory_db_path`로 관리된다.


## 트러블슈팅

### 문서를 못 찾는 경우
- followup으로 분류되었는데 이전 문서가 비어있는지 확인
- `data/session_memory.sqlite` 삭제 후 재시도

### 멀티턴이 유지되지 않는 경우
- REST 사용 시 `session_id`가 동일한지 확인
- Gradio 새로고침 시 세션 ID가 변경됨
