# 문서 파싱 및 청킹 도구

HWP, PDF 파일을 텍스트로 변환하고, 선택적으로 청킹까지 수행하는 로컬 환경용 도구입니다.

---

## 설치

```bash
pip install olefile pymupdf pdfplumber
```

---

## text_parsing.py 사용법

### 기본 실행

```bash
python text_parsing.py
```

### 설정 변수

파일 하단의 `if __name__ == "__main__":` 블록에서 설정을 변경합니다.

```python
# 경로 설정
INPUT_DIR = "data/original_data"   # 원본 파일 경로
OUTPUT_DIR = "data/parsing_data"   # 결과 저장 경로

# PDF 파서 설정
set_pdf_parser("pdfplumber")  # 또는 "fitz"

# 청킹 설정
ENABLE_CHUNKING = False  # True: 청킹 수행, False: 파싱만
CHUNK_SIZE = 1000        # 청크 크기 (글자 수)
CHUNK_OVERLAP = 200      # 청크 간 겹침 (글자 수)
```

---

## 설정 옵션 상세

### 1. PDF 파서 선택

| 옵션 | 특징 | 추천 상황 |
|------|------|-----------|
| `"pdfplumber"` | 텍스트 품질 좋음, 표 추출 강력 | 표가 많은 문서, 품질 중시 |
| `"fitz"` | 속도 2-5배 빠름 | 대량 파일 처리, 속도 중시 |

```python
# 품질 우선 (기본값)
set_pdf_parser("pdfplumber")

# 속도 우선
set_pdf_parser("fitz")
```

### 2. 파일 저장 여부

```python
# 저장 O (기본값)
save_to_file=True

# 저장 X (메모리에서만 처리)
save_to_file=False
```

### 3. 청킹 설정

```python
# 파싱만 수행
ENABLE_CHUNKING = False

# 파싱 + 청킹 수행
ENABLE_CHUNKING = True
CHUNK_SIZE = 1000     # 청크당 1000자
CHUNK_OVERLAP = 200   # 200자 겹침
```

**청킹 예시** (CHUNK_SIZE=10, OVERLAP=3):
```
원본: "ABCDEFGHIJKLMNO"
청크1: "ABCDEFGHIJ"  (0-10)
청크2: "HIJKLMNO"    (7-15, 3자 겹침)
```

---

## 사용 시나리오

### 시나리오 1: 파싱만 (저장 O)

```python
set_pdf_parser("pdfplumber")
ENABLE_CHUNKING = False

parsed_docs = process_all_files(
    input_dir="data/original_data",
    output_dir="data/parsing_data",
    save_to_file=True,
    enable_chunking=False,
)
```

결과: `data/parsing_data/문서명_parsed.txt`

### 시나리오 2: 파싱 + 청킹 (저장 O)

```python
set_pdf_parser("fitz")  # 속도 우선
ENABLE_CHUNKING = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

parsed_docs = process_all_files(
    input_dir="data/original_data",
    output_dir="data/parsing_data",
    save_to_file=True,
    enable_chunking=True,
    chunk_size=500,
    chunk_overlap=100,
)
```

결과: `data/parsing_data/문서명_chunked.txt`

### 시나리오 3: 파싱만 (저장 X, 메모리 처리)

```python
parsed_docs = process_all_files(
    input_dir="data/original_data",
    output_dir="data/parsing_data",
    save_to_file=False,
    enable_chunking=False,
)

# parsed_docs = {"파일명.hwp": "추출된 텍스트...", ...}
```

---

## 함수 직접 호출

### 개별 파일 파싱

```python
from text_parsing import parse_hwp_all, parse_pdf, set_pdf_parser

# HWP 파싱
text = parse_hwp_all("문서.hwp")

# PDF 파싱 (전역 설정 사용)
set_pdf_parser("pdfplumber")
text = parse_pdf("문서.pdf")

# PDF 파싱 (파서 직접 지정)
text = parse_pdf("문서.pdf", parser="fitz")
```

### 청킹만 수행

```python
from text_parsing import simple_chunk, chunk_text

# 단순 청킹 (문자열 리스트 반환)
chunks = simple_chunk(text, chunk_size=1000, overlap=200)

# 메타데이터 포함 청킹 (Chunk 객체 리스트 반환)
chunks = chunk_text(text, doc_id="doc1", chunk_size=1000, overlap=200)
```

---

## 출력 형식

### 파싱 결과 (`_parsed.txt`)

```
추출된 텍스트 내용이 그대로 저장됩니다.
줄바꿈과 공백이 정리된 상태입니다.
```

### 청킹 결과 (`_chunked.txt`)

```
=== 문서명::chunk0 ===
첫 번째 청크 내용...

=== 문서명::chunk1 ===
두 번째 청크 내용...
```

---

## 반환값 구조

```python
# enable_chunking=False
{
    "문서1.hwp": "추출된 텍스트...",
    "문서2.pdf": "추출된 텍스트...",
}

# enable_chunking=True
{
    "문서1.hwp": [Chunk(id="문서1::chunk0", text="...", metadata={...}), ...],
    "문서2.pdf": [Chunk(id="문서2::chunk0", text="...", metadata={...}), ...],
}
```

---

## 팀원별 파일 비교

| 기능 | jang | kim | seo | an | park |
|------|------|-----|-----|-----|------|
| HWP 파싱 | O (olefile) | O (hwp5txt + olefile) | X | X | O (olefile) |
| PDF 파싱 | X | O (fitz) | O (pdfplumber) | X | X |
| 청킹 | X | X | X | O | O (paragraph) |
| 저장 옵션 | O | X | X | X | O |
| 해싱 (safe_filename) | X | X | X | X | O (sha1) |

---

## 안전한 파일명 생성 (safe_filename)

파일명이 길거나 OS/인코딩 차이로 충돌이 발생하는 것을 방지합니다.

### 문제 상황

```
원본: 수협중앙회_강릉어선안전조업국 상황관제시스템 구축.hwp
     ↓ UTF-8 인코딩 시 바이트 수 초과 또는 특수문자 문제
OSError: 파일명이 너무 길거나 잘못된 문자 포함
```

### 해결 방식

```python
from text_parsing import safe_filename

# 원본 파일명 → 안전한 파일명 변환
safe_name = safe_filename("수협중앙회_강릉어선안전조업국 상황관제시스템 구축", suffix="_parsed.txt")
# 결과: "수협중앙회_강릉어선안전조업국 상황관__a1b2c3d4e5_parsed.txt"
```

### 동작 원리

| 단계 | 처리 내용 |
|------|----------|
| 1. 정규화 | `unicodedata.normalize("NFC")` |
| 2. 위험 문자 제거 | 윈도우 금지문자 (`<>:"/\|?*`) 제거 |
| 3. 바이트 제한 | UTF-8 기준 180바이트 이내로 축약 |
| 4. 해시 추가 | SHA1 해시 10자리로 충돌 방지 |

### 매핑 파일 (parsed_mapping.json)

`process_all_files()` 실행 시 자동 생성됩니다.

```json
[
  {
    "original_filename": "수협중앙회_강릉어선안전조업국 상황관제시스템 구축.hwp",
    "source_path": "data/original_data/수협중앙회_...",
    "saved_filename": "수협중앙회_강릉어선안전조업국 상황관__a1b2c3d4e5_parsed.txt",
    "saved_path": "data/parsing_data/...",
    "chars": 12345
  }
]
```

### RAG에서 활용

```python
import json

# 매핑 파일 로드
with open("data/parsing_data/parsed_mapping.json") as f:
    mapping = json.load(f)

# 청킹 시 원본 파일명을 source로 사용
for item in mapping:
    with open(item["saved_path"], encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, metadata={
        "source": item["original_filename"],  # 원본 파일명
        "chars": item["chars"]
    })
```

검색 결과 출력 시:
```
Q: "어선안전 관련 내용 알려줘"
A: [출처: 수협중앙회_강릉어선안전조업국 상황관제시스템 구축.hwp]
         ↑ 해시 파일명이 아닌 원본 파일명 표시
```
