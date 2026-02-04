# 3팀 청킹(Chunking) 전략 비교 분석
---

## 개요

RAG(Retrieval-Augmented Generation) 시스템에서 **청킹(Chunking)**은 검색 품질을 결정하는 핵심 요소입니다.
본 문서는 3팀 팀원들이 시도한 다양한 청킹 전략을 비교 분석합니다.

### 청킹이 중요한 이유
- **너무 작은 청크**: 문맥 손실, 의미 파악 어려움
- **너무 큰 청크**: 검색 정밀도 저하, 노이즈 증가
- **적절한 청크**: 의미 단위 보존 + 검색 효율 균형

---

## 팀원별 청킹 방식

### 안팀원 - RecursiveCharacterTextSplitter

**파일**: `an_chchunk.py`

#### 핵심 코드
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    separators=["\n\n", "\n", "□", "。", ".", "!", "?", " ", ""],
)
```

#### 특징
| 항목 | 내용 |
|------|------|
| 방식 | LangChain 라이브러리 활용 |
| 분할 우선순위 | 문단(`\n\n`) → 줄(`\n`) → 공고문 기호(`□`) → 문장부호 → 공백 → 글자 |
| 청크 크기 | 외부 파라미터로 주입 |
| Overlap | 외부 파라미터로 주입 |
| HWP 파싱 | `hwp5txt` CLI |
| 임베딩 모델 | `dragonkue/BGE-m3-ko` (한국어 특화) |
| Vector DB | FAISS |

#### 장점
- 라이브러리 검증된 안정성
- 재귀적 분할로 의미 단위 최대한 보존
- 한국어 공고문 특수문자(`□`) 지원

#### 단점
- 외부 의존성 필요 (langchain)

---

### 박팀원 - 커스텀 문단 기반 청킹

**파일**: `park_parsing.ipynb`, `park_parsing.py`

#### 핵심 코드
```python
def paragraph_chunking(
    text: str,
    min_chars: int = 200,
    max_chars: int = 800,
    overlap: int = 100
):
    # 1단계: 빈 줄 기준 문단 분리
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 2단계: 짧은 문단 합치기, 긴 문단 분리
    for p in paragraphs:
        if len(buffer) + len(p) <= max_chars:
            buffer += ("\n\n" + p)  # 합침
        else:
            if len(buffer) >= min_chars:
                chunks.append(buffer)  # 저장

    # 3단계: 긴 청크 재분할 + overlap
    for c in chunks:
        if len(c) > max_chars:
            # 800자씩 자르면서 100자 겹침
            start = end - overlap
```

#### 특징
| 항목 | 내용 |
|------|------|
| 방식 | 커스텀 구현 (적응형) |
| 분할 기준 | 빈 줄(`\n\n`) 기준 문단 |
| 청크 크기 | 200~800자 (가변) |
| Overlap | 100자 (긴 청크만) |
| HWP 파싱 | `olefile` + `zlib` 직접 구현 |

#### 장점
- 외부 의존성 없음
- 짧은 문단 합치기로 의미 단위 보존
- 적응형 청킹 (문단 길이에 따라 동적 처리)

#### 단점
- 문단 경계(`\n\n`)에만 의존
- 문장 중간 끊김 가능성

---

### 서팀원 - 의미론적 청킹 (SemanticChunker)

**파일**: `seo_paring.ipynb`

#### 핵심 코드
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 특화 임베딩 모델
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"  # 유사도 백분위 기준 분할
)
```

#### 특징
| 항목 | 내용 |
|------|------|
| 방식 | 임베딩 기반 의미 분석 |
| 분할 기준 | 문장 간 의미 유사도 |
| 청크 크기 | 가변 (의미 단위) |
| Overlap | 없음 (의미 경계에서 분할) |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` (한국어) |

#### 시도한 방식들
1. **길이 기반 청킹** (500자/50자 overlap) → 1021개 청크
2. **의미론적 청킹** (SemanticChunker) → 141개 청크

#### 장점
- 의미 단위 완벽 보존
- 문장 중간 끊김 없음
- 자연스러운 문맥 유지

#### 단점
- 임베딩 모델 필요 (느림)
- 청크 수가 적어 검색 정밀도 저하 가능
- HWP 파싱 실패 (PDF만 처리)

---

### 김팀원 - Context Enrichment + 청킹

**파일**: `kim_chunk.ipynb`, `kim_parsing_local.py`

#### 핵심 코드
```python
# 1. 텍스트 추출 (HWP: hwp5txt, PDF: PyMuPDF)
def get_hwp_text(file_path):
    result = subprocess.run(['hwp5txt', file_path], capture_output=True, text=True)
    return clean_text(result.stdout)

def get_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return clean_text(text)

# 2. Context Enrichment (문맥 보강)
enriched_content = f"""[[사업 개요]]
사업명: {metadata['title']}
발주기관: {metadata['agency']}
공고번호: {metadata['notice_id']}

[[본문]]
{content}"""

# 3. 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # 한국어 우선순위
)
split_docs = text_splitter.split_documents(documents)
```

#### 특징
| 항목 | 내용 |
|------|------|
| 방식 | 메타데이터 주입 + RecursiveCharacterTextSplitter |
| 분할 기준 | 문단 → 줄 → 문장 → 공백 |
| 청크 크기 | 1000자 |
| Overlap | 200자 |
| HWP 파싱 | `hwp5txt` CLI (pyhwp) |
| PDF 파싱 | `PyMuPDF (fitz)` |
| 임베딩 | `text-embedding-3-small` (OpenAI) |
| Vector DB | ChromaDB |
| 결과 | 98개 문서 → **3529개 청크** |

#### Context Enrichment 효과
```
일반 청크:
"본 사업은 시스템 구축을 목표로 한다..."

Enriched 청크:
"[[사업 개요]]
사업명: OO시스템 구축
발주기관: OO부
공고번호: 2024-001
[[본문]]
본 사업은 시스템 구축을 목표로 한다..."
```

#### 장점
- 청크 분리 후에도 **문맥 유지**
- 메타데이터 검색 가능
- 프로덕션급 파이프라인 (Google Colab 환경)
- RAG 체인 완성 (Retriever + GPT-4o-mini)

#### 단점
- OpenAI API 비용 발생
- 청크 크기 증가 (메타데이터 포함)

---

### 장팀원 - 계층 구조 기반 청킹 (HierarchicalChunker)

**파일**: `hierarchical_chunker_v2.py`

#### 핵심 코드
```python
class HierarchicalChunkerV2:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 200,
    ):
        # 계층 구조 패턴 (레벨 순서대로)
        self.hierarchy_patterns = [
            # Level 1: 로마 숫자 (Ⅰ. Ⅱ. Ⅲ.)
            (1, re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[\.\\s]")),
            # Level 2: 아라비아 숫자 (1. 2. 3.)
            (2, re.compile(r"^(\d+)[\.\\)]\\s")),
            # Level 3: 가나다 (가. 나. 다.)
            (3, re.compile(r"^([가나다라마바사아자차카타파하])[\.\\)]\\s")),
        ]

        self.table_detector = TableDetector()  # 표 감지기

    def chunk_document(self, text: str, doc_id: str, metadata: dict = None):
        # 1. 테이블 감지
        tables, table_lines = self.table_detector.detect_all_tables(text, doc_id)

        # 2. 섹션 분리 (계층 구조 기반)
        sections = self._split_sections(lines, table_lines)

        # 3. 섹션을 청크로 변환 (오버랩 적용)
        chunks = []
        for section in sections:
            if len(section_text) <= self.chunk_size:
                chunks.append(...)  # 완전한 섹션
            else:
                text_parts = self._split_with_overlap(section_text)
                ...

        # 4. 작은 청크 병합
        chunks = self._merge_small_chunks(chunks, doc_id)
        return chunks
```

#### 테이블 감지 (TableDetector)
```python
class TableDetector:
    def is_table_region(self, lines, start_idx, min_rows=3):
        # 탭 구분 테이블 감지
        if "\t" in line:
            col_count = self.count_columns(line, "tab")
            ...
        # 공백 정렬 테이블 감지
        col_count = self.count_columns(line, "space")
        if col_count >= 3:
            ...

    def detect_kv_table(self, lines, start_idx):
        # 요구사항 키-값 테이블 감지
        # "요구사항 번호", "요구사항 분류", "요구사항 명칭" 등
        ...
```

#### 특징
| 항목 | 내용 |
|------|------|
| 방식 | 계층 구조 인식 + 테이블 감지 + 섹션 분리 |
| 분할 기준 | 로마숫자(Ⅰ) → 숫자(1.) → 가나다(가.) |
| 청크 크기 | 1000자 (기본값) |
| Overlap | 20% (비율 기반) |
| 최소 청크 | 200자 (이하면 병합) |
| 테이블 감지 | 탭/공백 정렬, 키-값 테이블 |
| 임베딩 모델 | `dragonkue/BGE-m3-ko` (한국어 특화) |
| Vector DB | ChromaDB |

#### 계층 구조 인식 예시
```
Ⅰ. 사업의 안내              ← Level 1
  1. 사업설명                ← Level 2
    가. 용역명               ← Level 3
    나. 용역기간             ← Level 3
  2. 사업개요                ← Level 2
Ⅱ. 제안요청 내용            ← Level 1
```

#### 메타데이터 구조
```python
chunk.metadata = {
    "doc_id": "문서ID",
    "hierarchy_level": 2,           # 계층 레벨
    "hierarchy_path": ["Ⅰ. 사업의 안내", "1. 사업설명"],  # 계층 경로
    "section_title": "1. 사업설명",
    "is_complete_section": True,    # 완전한 섹션 여부
    "start_line": 10,
    "end_line": 25,
}
chunk.tables = [...]  # 해당 청크에 포함된 테이블 정보
```

#### 장점
- **공고문 구조 인식** (로마숫자, 가나다 등)
- **테이블 자동 감지** 및 구조화
- **계층 경로 추적**으로 검색 시 문맥 파악 용이
- 작은 청크 자동 병합
- 문장 경계에서 분할 (`다.` 또는 줄바꿈)

#### 단점
- 계층 패턴이 없는 문서에는 효과 제한
- 구현 복잡도 높음

---

## 종합 비교표

| 항목 | 안팀원 | 박팀원 | 서팀원 | 김팀원 | 장팀원 |
|------|--------|--------|--------|--------|--------|
| **청킹 방식** | RecursiveCharacterTextSplitter | 커스텀 문단 기반 | SemanticChunker | RecursiveCharacterTextSplitter | HierarchicalChunker |
| **분할 기준** | 문단→줄→문장→단어 | 빈 줄(`\n\n`) | 문장 간 의미 유사도 | 문단→줄→문장→공백 | 계층구조(Ⅰ→1.→가.) |
| **청크 크기** | 외부 파라미터 | 200~800자 | 가변 (의미 단위) | 1000자 | 1000자 |
| **Overlap** | 외부 파라미터 | 100자 | 없음 | 200자 | 20% (비율) |
| **HWP 파싱** | `hwp5txt` CLI | `olefile` 직접 | 실패 (PDF만) | `hwp5txt` CLI | - |
| **PDF 파싱** | - | - | `pdfplumber` | `PyMuPDF` | - |
| **임베딩** | `dragonkue/BGE-m3-ko` | `all-MiniLM-L6-v2` | `ko-sroberta` | `text-embedding-3-small` | `dragonkue/BGE-m3-ko` |
| **Vector DB** | FAISS | FAISS | - | ChromaDB | ChromaDB |
| **테이블 처리** | X | X | X | X | O (자동 감지) |
| **특이점** | 라이브러리 활용 | 적응형 청킹 | 의미 보존 | Context Enrichment | 계층 경로 추적 |

---

## 테스트 가이드

### 환경 설정

```bash
# 필수 패키지 설치
pip install olefile pdfplumber pymupdf
```

---

### A. text_parsing.py 테스트 (고정 길이 청킹)

#### 1. CLI 실행
```bash
# 기본 실행 (파싱만)
python text_parsing.py

# 설정 변경 (파일 내 527~529번 라인 수정)
# ENABLE_CHUNKING = True
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 100
```

#### 2. Python 코드로 테스트
```python
from text_parsing import process_all_files, set_pdf_parser

# PDF 파서 설정 (선택)
set_pdf_parser("pdfplumber")  # 품질 우선
# set_pdf_parser("fitz")      # 속도 우선

# 파싱 + 청킹 수행
parsed_docs = process_all_files(
    input_dir="data/original_data",
    output_dir="data/parsing_data",
    save_to_file=True,
    enable_chunking=True,
    chunk_size=1000,
    chunk_overlap=200,
)

# 결과 확인
for filename, chunks in parsed_docs.items():
    print(f"{filename}: {len(chunks)}개 청크")
```

#### 3. 개별 파일 테스트
```python
from text_parsing import load_file_content, chunk_text

# 단일 파일 파싱
text = load_file_content("data/original_data/sample.hwp")
print(f"추출된 텍스트: {len(text)}자")

# 청킹
chunks = chunk_text(
    text=text,
    doc_id="sample",
    chunk_size=800,
    overlap=100,
    metadata={"source": "sample.hwp"}
)
print(f"생성된 청크: {len(chunks)}개")
```

---

### B. hierarchical_chunker_v2.py 테스트 (계층 구조 청킹)

#### 1. CLI 실행
```bash
# 파싱된 파일이 data/parsing_data/ 에 있어야 함
python hierarchical_chunker_v2.py

# 설정 변경 (파일 내 753~756번 라인 수정)
# CHUNK_SIZE = 1000
# OVERLAP_RATIO = 0.2
# MIN_CHUNK_SIZE = 200
```

#### 2. Python 코드로 테스트
```python
from hierarchical_chunker_v2 import HierarchicalChunkerV2, process_parsed_files_v2

# 방법 1: 전체 파일 처리
stats = process_parsed_files_v2(
    input_dir="data/parsing_data",
    output_dir="data/chunking_data",
    chunk_size=1000,
    overlap_ratio=0.2,
    min_chunk_size=200,
)
print(f"총 청크 수: {stats['total_chunks']}개")
print(f"총 테이블 수: {stats['total_tables']}개")

# 방법 2: 개별 문서 청킹
chunker = HierarchicalChunkerV2(
    chunk_size=1000,
    overlap_ratio=0.2,
    min_chunk_size=200,
)

with open("data/parsing_data/sample_parsed.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunker.chunk_document(
    text=text,
    doc_id="sample",
    metadata={"source": "sample.hwp"}
)

# 결과 확인
for chunk in chunks[:3]:
    print(f"[{chunk.chunk_id}]")
    print(f"  계층: {chunk.metadata.get('hierarchy_path')}")
    print(f"  테이블: {len(chunk.tables)}개")
    print(f"  내용: {chunk.text[:100]}...")
    print()
```

#### 3. 테이블 감지 테스트
```python
from hierarchical_chunker_v2 import TableDetector

detector = TableDetector()

with open("data/parsing_data/sample_parsed.txt", "r", encoding="utf-8") as f:
    text = f.read()

tables, table_lines = detector.detect_all_tables(text, "sample")

print(f"감지된 테이블: {len(tables)}개")
for table in tables:
    print(f"  - {table.table_id}: {len(table.rows)}행, 감지방식: {table.detection_method}")
```

---

### C. 청킹 파라미터 실험

```python
from text_parsing import load_file_content, simple_chunk

text = load_file_content("data/original_data/sample.hwp")

# 다양한 설정 비교
configs = [
    {"chunk_size": 500, "overlap": 50},    # 정밀 검색용
    {"chunk_size": 800, "overlap": 100},   # 일반 검색용
    {"chunk_size": 1000, "overlap": 200},  # 문맥 보존용
    {"chunk_size": 1500, "overlap": 300},  # 요약용
]

for cfg in configs:
    chunks = simple_chunk(text, cfg["chunk_size"], cfg["overlap"])
    print(f"size={cfg['chunk_size']}, overlap={cfg['overlap']} → {len(chunks)}개 청크")
```

---

### D. 파이프라인 전체 실행 순서

```bash
# 1단계: 원본 파일 파싱 (HWP/PDF → TXT)
RUN_MODE == "parse": # 변경 후
python team_chunking.py
# 결과: data/parsing_data/*.txt

# 2단계: 계층 구조 청킹 (TXT → JSON)
RUN_MODE == "compare":
python team_chunking.py
# 결과: data/chunking_data/*.json

# 3단계 : 청킹 5가지 × 임베딩 4가지 = 20가지 조합 테스트
python embedding_evaluation.py
# 결과 : evaluation_results.json
```

---

## 권장 설정

### 공공 입찰 공고(RFP) 문서 기준

| 용도 | chunk_size | overlap | 비고 |
|------|------------|---------|------|
| **정밀 검색** | 500~800 | 100 | 세부 요구사항 검색 시 |
| **일반 검색** | 800~1000 | 150~200 | 범용 RAG 시스템 |
| **요약/개요** | 1500~2000 | 300 | 전체 문맥 파악 시 |

### 추천 조합

| 목적 | 추천 방식 | 파일 |
|------|-----------|------|
| **빠른 프로토타이핑** | 고정 길이 청킹 | `text_parsing.py` |
| **의미 보존 중시** | 의미론적 청킹 | 서팀원 방식 (SemanticChunker) |
| **공고문 구조 인식** | 계층 구조 청킹 | `hierarchical_chunker_v2.py` |
| **프로덕션 배포** | Context Enrichment + 임베딩 | 김팀원 방식 (`kim_chunk.ipynb`) |

### 청킹 방식 선택 가이드

`embedding_evaluation.py`을 통해서

Ground Truth `evaluation_dataset.json` 을 불러와서 점수로 확인할 예정입니다.


---
## 최종 수정 날짜
2026.02.04

함수 변경되면 말씀해주세요.