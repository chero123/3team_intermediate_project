# RAG 청킹 전략 평가 프로젝트

> 5가지 청킹 방식 × 3가지 임베딩 모델 = 15개 조합 벤치마크

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-GTX%201660%20SUPER-green.svg)](https://developer.nvidia.com/cuda-zone)

---

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [평가 결과 요약](#평가-결과-요약)
- [팀원별 청킹 방식](#팀원별-청킹-방식)
- [성능 비교 상세](#성능-비교-상세)
- [테스트 가이드](#테스트-가이드)
- [권장사항](#권장사항)

---

## 프로젝트 개요

RAG(Retrieval-Augmented Generation) 시스템에서 **청킹(Chunking)**과 **임베딩 모델** 선택은 검색 품질을 결정하는 핵심 요소입니다. 본 프로젝트는 3팀 팀원들이 개발한 5가지 청킹 전략과 3가지 임베딩 모델을 조합하여 실제 데이터셋으로 성능을 평가했습니다.

### 평가 환경
- **평가 데이터셋**: 2개 (각 40개 질문)
- **청킹 방식**: 5가지 (Recursive, Paragraph, Semantic, ContextEnriched, Hierarchical)
- **임베딩 모델**: 3가지 (MiniLM, ko-sroberta, OpenAI text-embedding-3-small)
- **총 조합**: 15가지
- **GPU**: NVIDIA GeForce GTX 1660 SUPER
- **Vector DB**: FAISS, ChromaDB

---

## 평가 결과 요약

### 최고 성능 조합

| 순위 | 청킹 방식 | 임베딩 모델 | Dataset 1 | Dataset 2 | 평균 Hit@1 | Latency |
|------|-----------|-------------|-----------|-----------|------------|---------|
| 1 | **김팀원-ContextEnriched** | **OpenAI** | 90.00% | 85.00% | **87.50%** | 318.8ms |
| 2 | 안팀원-Recursive | ko-sroberta | 90.00% | 65.00% | 77.50% | 69.2ms |
| 3 | 김팀원-ContextEnriched | ko-sroberta | 87.50% | 77.50% | 82.50% | 59.7ms |
| 4 | 서팀원-Semantic | ko-sroberta | 87.50% | 60.00% | 73.75% | 54.1ms |
| 5 | 장팀원-Hierarchical | ko-sroberta | 82.50% | 57.50% | 70.00% | 72.4ms |

### 주요 발견사항

#### 1. 임베딩 모델 성능
- **ko-sroberta**: 가장 안정적이고 균형잡힌 성능 (평균 66.8%)
  - Dataset 1: 평균 70.5%
  - Dataset 2: 평균 63.0%
- **OpenAI**: 최고 정확도이지만 데이터셋 간 편차 존재 (평균 66.5%)
  - Dataset 1: 평균 73.0%
  - Dataset 2: 평균 60.0%
- **MiniLM**: 한국어 도메인에서 현저히 낮은 성능 (평균 12.5%)
  - Dataset 1: 평균 21.0%
  - Dataset 2: 평균 4.0%

#### 2. 청킹 방식 효과
- **ContextEnriched (김팀원)**: 두 데이터셋 모두 최상위 (평균 77.5%)
  - 메타데이터 주입으로 청크 분리 후에도 문맥 유지
- **Recursive (안팀원)**: Dataset 1에서 우수 (90%), Dataset 2에서 중간 (65%)
- **Paragraph (박팀원)**: 청크 수가 많지만 (11,764개) 성능은 낮음
  - Dataset 1: 평균 29.2%
  - Dataset 2: 평균 35.8%

#### 3. 속도 vs 정확도 트레이드오프

| 조합 | Hit@1 | Latency | 특징 |
|------|-------|---------|------|
| 김팀원-ContextEnriched + OpenAI | 87.50% | 318.8ms | 최고 정확도 |
| 김팀원-ContextEnriched + ko-sroberta | 82.50% | 59.7ms | **균형점** |
| 서팀원-Semantic + ko-sroberta | 73.75% | 54.1ms | 최고 속도 |

---

## 팀원별 청킹 방식

### 안팀원 - RecursiveCharacterTextSplitter

**특징**: LangChain 라이브러리 활용, 재귀적 분할

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    separators=["\n\n", "\n", "□", "。", ".", "!", "?", " ", ""],
)
```

| 항목 | 내용 |
|------|------|
| 분할 우선순위 | 문단(`\n\n`) → 줄(`\n`) → 공고문 기호(`□`) → 문장부호 → 공백 |
| 청크 크기 | 외부 파라미터로 주입 |
| HWP 파싱 | `hwp5txt` CLI |
| 임베딩 모델 | `dragonkue/BGE-m3-ko` |
| Vector DB | FAISS |
| 결과 청크 수 | 9,625개 |

**평가 결과** (ko-sroberta 기준)
- Dataset 1: Hit@1 90.00%, MRR 0.9125
- Dataset 2: Hit@1 65.00%, MRR 0.7042

---

### 박팀원 - 커스텀 문단 기반 청킹

**특징**: 외부 의존성 없는 적응형 청킹

```python
def paragraph_chunking(
    text: str,
    min_chars: int = 200,
    max_chars: int = 800,
    overlap: int = 100
):
    # 1. 빈 줄 기준 문단 분리
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # 2. 짧은 문단 합치기, 긴 문단 분리
    # 3. overlap 적용
```

| 항목 | 내용 |
|------|------|
| 분할 기준 | 빈 줄(`\n\n`) 기준 문단 |
| 청크 크기 | 200~800자 (가변) |
| HWP 파싱 | `olefile` + `zlib` 직접 구현 |
| 임베딩 모델 | `all-MiniLM-L6-v2` |
| Vector DB | FAISS |
| 결과 청크 수 | 11,764개 |

**평가 결과** (OpenAI 기준)
- Dataset 1: Hit@1 80.00%, MRR 0.8708
- Dataset 2: Hit@1 52.50%, MRR 0.5725

---

### 서팀원 - 의미론적 청킹 (SemanticChunker)

**특징**: 임베딩 기반 의미 분석, 문장 중간 끊김 없음

```python
from langchain_experimental.text_splitter import SemanticChunker

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)
```

| 항목 | 내용 |
|------|------|
| 분할 기준 | 문장 간 의미 유사도 |
| 청크 크기 | 가변 (의미 단위) |
| Overlap | 없음 (의미 경계에서 분할) |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 결과 청크 수 | 8,622개 |

**평가 결과** (ko-sroberta 기준)
- Dataset 1: Hit@1 87.50%, MRR 0.8875
- Dataset 2: Hit@1 60.00%, MRR 0.6729

---

### 김팀원 - Context Enrichment + 청킹 ⭐

**특징**: 메타데이터 주입으로 문맥 보존

```python
# Context Enrichment
enriched_content = f"""[[사업 개요]]
사업명: {metadata['title']}
발주기관: {metadata['agency']}
공고번호: {metadata['notice_id']}

[[본문]]
{content}"""

# RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

| 항목 | 내용 |
|------|------|
| 분할 기준 | 문단 → 줄 → 문장 → 공백 |
| 청크 크기 | 1000자 |
| HWP 파싱 | `hwp5txt` CLI |
| PDF 파싱 | `PyMuPDF (fitz)` |
| 임베딩 모델 | `text-embedding-3-small` (OpenAI) |
| Vector DB | ChromaDB |
| 결과 청크 수 | 9,625개 |

**평가 결과** (OpenAI 기준)
- Dataset 1: Hit@1 **90.00%**, MRR **0.9375** 🏆
- Dataset 2: Hit@1 **85.00%**, MRR **0.8675** 🏆

---

### 장팀원 - 계층 구조 기반 청킹 (HierarchicalChunker)

**특징**: 공고문 구조 인식 (로마숫자, 가나다) + 테이블 자동 감지

```python
class HierarchicalChunkerV2:
    def __init__(self, chunk_size=1000, overlap_ratio=0.2):
        self.hierarchy_patterns = [
            (1, re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[\.\s]")),  # Level 1
            (2, re.compile(r"^(\d+)[\.\)]\s")),              # Level 2
            (3, re.compile(r"^([가나다라마바사][\.\)]\s")),    # Level 3
        ]
        self.table_detector = TableDetector()
```

| 항목 | 내용 |
|------|------|
| 분할 기준 | 로마숫자(Ⅰ) → 숫자(1.) → 가나다(가.) |
| 청크 크기 | 1000자 |
| Overlap | 20% (비율 기반) |
| 테이블 감지 | 탭/공백 정렬, 키-값 테이블 |
| 임베딩 모델 | `dragonkue/BGE-m3-ko` |
| Vector DB | ChromaDB |
| 결과 청크 수 | 12,240개 |

**평가 결과** (ko-sroberta 기준)
- Dataset 1: Hit@1 82.50%, MRR 0.8529
- Dataset 2: Hit@1 57.50%, MRR 0.6792

---

## 성능 비교 상세

### Dataset 1 결과 (질문 40개)

| 청킹 방식 | MiniLM | ko-sroberta | OpenAI | 청크 수 |
|----------|---------|-------------|--------|---------|
| 안팀원-Recursive | 20.00% | **90.00%** | 65.00% | 9,625 |
| 박팀원-Paragraph | 2.50% | 5.00% | **80.00%** | 11,764 |
| 서팀원-Semantic | 30.00% | **87.50%** | 65.00% | 8,622 |
| 김팀원-ContextEnriched | 22.50% | 87.50% | **90.00%** | 9,625 |
| 장팀원-Hierarchical | 30.00% | **82.50%** | 65.00% | 12,240 |

### Dataset 2 결과 (질문 40개)

| 청킹 방식 | MiniLM | ko-sroberta | OpenAI | 청크 수 |
|----------|---------|-------------|--------|---------|
| 안팀원-Recursive | 0.00% | 65.00% | 50.00% | 9,625 |
| 박팀원-Paragraph | 0.00% | 55.00% | 52.50% | 11,764 |
| 서팀원-Semantic | 0.00% | 60.00% | 50.00% | 8,622 |
| 김팀원-ContextEnriched | 10.00% | 77.50% | **85.00%** | 9,625 |
| 장팀원-Hierarchical | 10.00% | 57.50% | 50.00% | 12,240 |

### 세부 지표 (Top 5)

| 조합 | Dataset | Hit@1 | Hit@5 | MRR | Latency |
|------|---------|-------|-------|-----|---------|
| 김팀원-ContextEnriched + OpenAI | 1 | 90.00% | 97.50% | 0.9375 | 314.0ms |
| 김팀원-ContextEnriched + OpenAI | 2 | 85.00% | 90.00% | 0.8675 | 323.5ms |
| 안팀원-Recursive + ko-sroberta | 1 | 90.00% | 92.50% | 0.9125 | 77.1ms |
| 김팀원-ContextEnriched + ko-sroberta | 1 | 87.50% | 97.50% | 0.9037 | 60.7ms |
| 서팀원-Semantic + ko-sroberta | 1 | 87.50% | 90.00% | 0.8875 | 54.8ms |

---

## 테스트 가이드

### 환경 설정

```bash
# 필수 패키지 설치
pip install langchain langchain-text-splitters langchain-experimental
pip install sentence-transformers faiss-cpu chromadb
pip install olefile pdfplumber pymupdf openai

# HWP 파싱용 (Linux/Mac)
pip install pyhwp
```

### 파이프라인 전체 실행 순서

```bash
# 1단계: 원본 파일 파싱 (HWP/PDF → TXT)
# team_chunking.py에서 RUN_MODE = "parse"로 변경 후
python team_chunking.py
# 결과: data/parsing_data/*.txt

# 2단계: 계층 구조 청킹 (TXT → JSON)
# team_chunking.py에서 RUN_MODE = "compare"로 변경 후
python team_chunking.py
# 결과: data/chunking_data/*.json

# 3단계: 청킹 5가지 × 임베딩 3가지 = 15가지 조합 테스트
python embedding_evaluation.py
# 결과: evaluation_results.json
```

### 개별 청킹 방식 테스트

#### A. 고정 길이 청킹 (text_parsing.py)

```python
from text_parsing import process_all_files, chunk_text

# 전체 파일 처리
parsed_docs = process_all_files(
    input_dir="data/original_data",
    output_dir="data/parsing_data",
    enable_chunking=True,
    chunk_size=1000,
    chunk_overlap=200,
)

# 단일 파일 테스트
text = load_file_content("data/original_data/sample.hwp")
chunks = chunk_text(text, "sample", chunk_size=800, overlap=100)
print(f"생성된 청크: {len(chunks)}개")
```

#### B. 계층 구조 청킹 (hierarchical_chunker_v2.py)

```python
from hierarchical_chunker_v2 import HierarchicalChunkerV2

chunker = HierarchicalChunkerV2(
    chunk_size=1000,
    overlap_ratio=0.2,
    min_chunk_size=200,
)

chunks = chunker.chunk_document(
    text=text,
    doc_id="sample",
    metadata={"source": "sample.hwp"}
)

# 결과 확인
for chunk in chunks[:3]:
    print(f"계층: {chunk.metadata.get('hierarchy_path')}")
    print(f"테이블: {len(chunk.tables)}개")
```

---

## 권장사항

### 프로덕션 환경

| 시나리오 | 추천 조합 | 이유 |
|----------|-----------|------|
| **고정밀 요구** | 김팀원-ContextEnriched + OpenAI | 최고 정확도 (평균 87.5%) |
| **속도와 정확도 균형** | 김팀원-ContextEnriched + ko-sroberta | 82.5% 정확도, 60ms 응답 |
| **비용 최적화** | 서팀원-Semantic + ko-sroberta | 73.8% 정확도, 54ms 응답 |
| **안정성 우선** | 안팀원-Recursive + ko-sroberta | 검증된 라이브러리 |

### 청킹 파라미터 설정

공공 입찰 공고(RFP) 문서 기준

| 용도 | chunk_size | overlap | 비고 |
|------|------------|---------|------|
| **정밀 검색** | 500~800 | 100 | 세부 요구사항 검색 시 |
| **일반 검색** | 800~1000 | 150~200 | 범용 RAG 시스템 (권장) |
| **요약/개요** | 1500~2000 | 300 | 전체 문맥 파악 시 |

### 개선 방향

1. **MiniLM 사용 지양**: 한국어 특화 도메인에서 현저히 낮은 성능 (4-21%)
2. **청크 수 최적화**: 많다고 좋은 것이 아님 (Paragraph 11,764개 vs ContextEnriched 9,625개)
3. **Context Enrichment 적용**: 메타데이터 주입으로 청크 분리 후에도 문맥 유지
4. **OpenAI 임베딩 검증 필요**: 데이터셋 간 편차 존재 (Dataset 1: 73%, Dataset 2: 60%)
5. **ko-sroberta 추천**: 가장 안정적이고 균형잡힌 성능 (평균 66.8%)

---

## 프로젝트 구조

```
rag-evaluation/
├── data/
│   ├── original_data/          # 원본 HWP/PDF 파일
│   ├── parsing_data/           # 파싱된 TXT 파일
│   ├── chunking_data/          # 청킹된 JSON 파일
│   ├── evaluation_dataset.json # 평가 데이터셋 1
│   └── evaluation_dataset2.json# 평가 데이터셋 2
├── embedding_evaluation.py      # 전체 평가 스크립트
├── team_chunking.py            # 통합 실행 스크립트
└── README.md
```

---

## 종합 비교표

| 항목 | 안팀원 | 박팀원 | 서팀원 | 김팀원 | 장팀원 |
|------|--------|--------|--------|--------|--------|
| **청킹 방식** | RecursiveCharacter | 커스텀 문단 기반 | SemanticChunker | Context Enrichment | Hierarchical |
| **분할 기준** | 문단→줄→문장 | 빈 줄(`\n\n`) | 의미 유사도 | 문단→줄→문장 | 계층구조(Ⅰ→1.→가.) |
| **청크 크기** | 외부 파라미터 | 200~800자 | 가변 | 1000자 | 1000자 |
| **Overlap** | 외부 파라미터 | 100자 | 없음 | 200자 | 20% |
| **청크 수** | 9,625 | 11,764 | 8,622 | 9,625 | 12,240 |
| **Dataset 1** | 90.0% | 80.0% | 87.5% | **90.0%** | 82.5% |
| **Dataset 2** | 65.0% | 52.5% | 60.0% | **85.0%** | 57.5% |
| **평균 성능** | 77.5% | 66.3% | 73.8% | **87.5%** | 70.0% |
| **HWP 파싱** | `hwp5txt` | `olefile` | 실패 | `hwp5txt` | - |
| **Vector DB** | FAISS | FAISS | - | ChromaDB | ChromaDB |
| **테이블 처리** | X | X | X | X | O |

---

## 참고 문헌

- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Semantic Chunking: https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings

---

## 최종 수정 날짜
2025.02.05

## 라이선스
MIT License

## 기여자
AI6기 3팀 - 박팀원, 안팀원, 서팀원, 김팀원, 장팀원