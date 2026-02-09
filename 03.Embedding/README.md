# RFP 문서 분석 RAG 시스템

공공기관 RFP(제안요청서) 문서를 자동으로 파싱, 청킹, 임베딩하여 벡터 DB에 저장하고, LangGraph 기반 RAG로 질의응답을 수행하는 시스템입니다.

---

## 전체 파이프라인 흐름

```
[원본 문서]          [파싱]              [청킹]            [임베딩/저장]        [질의응답]
HWP / PDF / DOCX → text_parsing.py → chunking.py → rag_pipeline.py → rfp_retriever.py
                     ↓                  ↓               ↓                   ↓
                   .md 파일 생성      Chunk 객체 생성   ChromaDB 저장      LangGraph 그래프
                   (data/final_docs)  (문맥 보강)      (data/vector_db)   (Router→Retriever→Generator)
```

### STEP 1. 문서 파싱 (`text_parsing.py`)

원본 문서(HWP, PDF, DOCX)를 마크다운(.md)으로 변환합니다.

- **HWP**: olefile로 BodyText 레코드 추출 → UTF-16LE 디코딩
- **PDF**: pdfplumber(정확도) 또는 fitz/PyMuPDF(속도) 선택 가능
- **표 추출**: OpenAI GPT-4o-mini VLM으로 PDF 페이지 이미지에서 표를 마크다운으로 변환
- **이미지 분석**: Qwen2.5-VL-7B(로컬) 또는 OpenAI VLM으로 이미지 내 텍스트 추출

```bash
# 사용 예시
python text_parsing.py
```

### STEP 2. 텍스트 청킹 (`chunking.py`)

파싱된 문서를 의미 단위로 분할하고, 검색 품질을 높이기 위해 문맥 정보를 보강합니다.

- 섹션 헤더 자동 감지 (Ⅰ, 1., 가., □, ○ 등)
- 단락(`\n\n`) 기준 분할
- 각 청크에 문서정보 + 섹션정보 헤더 추가:

```
[[문서정보]]
사업명: 벤처확인종합관리시스템 기능 고도화 용역사업
발주기관: (사)벤처기업협회

[[섹션정보]]
위치: Ⅰ. 사업개요

[[본문]]
실제 내용...
```

### STEP 3. 임베딩 (`embeddings.py`)

청크 텍스트를 벡터로 변환합니다.

| 모델 | 차원 | 특징 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | API 기반, 기본값 |
| OpenAI text-embedding-3-large | 3072 | API 기반, 고품질 |
| KoSRoBERTa (ko-sroberta-multitask) | 768 | 로컬, GPU 가속 |

### STEP 4. 벡터 저장 (`vector_store.py`)

임베딩 벡터를 벡터 DB에 저장합니다.

| DB | 유형 | 유사도 | 특징 |
|----|------|--------|------|
| ChromaDB | 영구 저장 | Cosine | 기본값, `data/vector_db/rfp_integrated`에 저장 |
| FAISS | 인메모리 | Inner Product | 대규모 검색에 유리 |

### STEP 5. 질의응답 (`rfp_retriever.py`)

LangGraph 기반 3단계 RAG 그래프로 질문에 답변합니다.

```
START → Router → Retriever → Generator → END
```

| 노드 | 역할 |
|------|------|
| **Router** | 질문을 분석하여 `single` / `compare` / `multi` 모드 분류 |
| **Retriever** | 벡터 유사도 검색 (문서 필터링 가능) |
| **Generator** | LLM으로 검색 결과 기반 답변 생성 + 출처 표시 |

#### Router 상세 동작

질문에서 기관명/사업명을 인식하여 검색 대상 문서를 결정합니다.

**3단계 매칭 프로세스:**

1. **Fuzzy Matching** (`_fuzzy_match_doc_ids`) — LLM 호출 전 키워드 기반으로 후보 doc_id를 먼저 추출
   - 기관명 매칭: doc_id의 `_` 앞부분(기관명)이 질문에 포함되면 즉시 매칭
   - 사업명 키워드 매칭: 3글자 이상 한글 키워드 중 2개 이상 일치하면 매칭
2. **LLM 라우팅** — 개선된 프롬프트로 모드 분류 및 doc_id 매칭
   - 기관명만 언급되어도 매칭 (예: "전북대학교" → `전북대학교_JST 공유대학(원)...`)
   - 약칭/부분 명칭 매칭 지원
3. **결과 병합** — LLM이 놓친 doc_id를 fuzzy 결과로 보완
   - compare 모드에서 누락된 문서 자동 추가
   - single인데 2개 이상 매칭 시 compare로 자동 승격

| 모드 | 설명 | 검색 방식 |
|------|------|-----------|
| `single` | 특정 문서 1개에 대한 질문 | 해당 doc_id 필터링 검색 |
| `compare` | 2개 이상 문서 비교 질문 | 각 문서별 개별 검색 후 병합 |
| `multi` | 전체 문서 대상 질문 | 전체 벡터DB 검색 |

```bash
# 사용 예시
python rfp_retriever.py
```

---

## 통합 실행 (`rag_pipeline.py`)

STEP 2~4를 한 번에 실행합니다. `data/data_list.xlsx`에서 메타데이터를 읽고, `data/final_docs/*.md` 파일을 처리합니다.

```bash
python rag_pipeline.py
```

**출력물:**
- `data/vector_db/rfp_integrated/` — ChromaDB 벡터 DB
- `data/debug/*.json` — 청크 확인용 JSON 파일

---

## 프로젝트 구조

```
03.Embedding/
│
├── text_parsing.py         # STEP 1: 문서 파싱 (HWP/PDF/DOCX → MD)
├── chunking.py             # STEP 2: 텍스트 청킹 + Chunk 데이터 클래스
├── embeddings.py           # STEP 3: 임베딩 모델 (OpenAI / KoSRoBERTa)
├── vector_store.py         # STEP 4: 벡터 DB (ChromaDB / FAISS) + SearchResult 클래스
├── rag_pipeline.py         # STEP 2~4 통합 실행 (메인 파이프라인)
├── rfp_retriever.py        # STEP 5: LangGraph 기반 질의응답
│
│
├── data/
│   ├── original_data/      # 원본 RFP 문서 (HWP, PDF, DOCX)
│   ├── final_docs/         # 파싱 완료된 마크다운 파일
│   ├── vector_db/          # 벡터 DB 저장소
│   │   └── rfp_integrated/ # ChromaDB 컬렉션
│   ├── debug/              # 청크 디버그 JSON
│   ├── data_list.xlsx      # 문서 메타데이터 (공고번호, 사업명, 발주기관)
│
├── models/
│   └── qwen2.5-vl-7b/      # 로컬 VLM 모델
│
└── .env                    # API 키 설정
```

---

## 모듈 의존성

```
chunking.py ─────────────┐
  (Chunk 정의)            │
                          ▼
embeddings.py        vector_store.py
  (임베딩 모델)        (Chunk 사용, SearchResult 정의)
       │                  │
       └──────┬───────────┘
              ▼
        rag_pipeline.py
          (통합 파이프라인)
              │
              ▼
       rfp_retriever.py
         (질의응답 시스템)
```

---

## 설치 및 환경 설정

### 1. 패키지 설치

```bash
pip install python-dotenv pandas openpyxl

# 문서 파싱
pip install olefile pymupdf pdfplumber python-docx

# 임베딩 & 벡터 DB
pip install openai chromadb sentence-transformers

# LangGraph (질의응답)
pip install langgraph langchain-core langchain-openai

# (선택) FAISS
pip install faiss-cpu  # 또는 faiss-gpu

# (선택) 로컬 VLM
pip install vllm qwen-vl-utils transformers torch
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다.

```
OPENAI_API_KEY=sk-proj-...
```

### 3. 실행 순서

```bash
# 1) 문서 파싱 (원본 → 마크다운)
python text_parsing.py

# 2) 벡터 DB 구축 (마크다운 → 청킹 → 임베딩 → 저장)
python rag_pipeline.py

# 3) 질의응답
python rfp_retriever.py
```

---

## 팀원 기여

| 팀원 | 담당 영역 |
|------|-----------|
| 장 | 계층적 섹션 감지 기반 청킹 (`chunking.py`) |
| 김 | 문맥 보강 메타데이터 설계, HWP 파싱 (`kim_parsing_local.py`) |
| 박 | PDF 표 추출 및 MD 삽입 (`park_extract_tables_to_md.py`) |
| 안 | 데이터 로딩/파싱 모듈 (`an_data.py`) |
