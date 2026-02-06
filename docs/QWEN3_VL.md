# Qwen3-VL 8B (로컬 VLM) 기능/사용법

이 문서는 **로컬 Qwen3-VL 8B 모델**을 사용해 PDF 내 이미지(표/그래프/도식) 정보를 텍스트로 추출하는 기능과 사용법을 정리한다.

## 목적
- PDF 안에 **텍스트+이미지(표/그래프/수치)**가 혼재된 문서에서
  **이미지 영역의 수치/표 정보를 추가로 추출**해 RAG 답변 품질을 개선한다.
- OCR만으로는 부족한 **레이아웃/표 구조 해석**을 보완한다.

## 전체 흐름 (요약)
1) PDF 텍스트는 `pdfplumber` 또는 `fitz`로 기본 추출
2) PDF 페이지 이미지를 렌더링
3) 이미지 필터링(무의미 이미지 제거)
4) **Qwen3-VL 8B**로 이미지 내 텍스트/표 요약
5) 결과를 문서 메타데이터에 저장해 검색/답변에 활용

## 설정 위치
- 설정 파일: `src/rag/config.py`
- 주요 옵션
  - `qwen3_vl_model_path`: 로컬 모델 경로 (예: `models/qwen3-vl-8b`)
  - `qwen3_vl_enabled`: VLM 사용 여부 (True/False)
  - `qwen3_vl_max_tokens`: 이미지 1장당 생성 토큰 상한
  - `qwen3_vl_gpu_memory_utilization`: vLLM GPU 메모리 사용 비율
  - `qwen3_vl_max_model_len`: vLLM max_model_len
  - `qwen3_vl_dedupe_images`: 중복 이미지 제거
  - `qwen3_vl_min_image_pixels`: 너무 작은 이미지 제거
  - `qwen3_vl_min_nonwhite_ratio`: 거의 백지인 이미지 제거
  - `qwen3_vl_min_variance`: 단색/로고 이미지 제거
  - `qwen3_vl_min_edge_energy`: 도표/텍스트 없는 이미지 제거

## 모델 경로
- **로컬 경로 기준**: `models/qwen3-vl-8b`
- HuggingFace에서 다운로드를 요구하지 않도록 로컬 경로를 사용한다.

## 사용 시나리오
- `scripts/build_index.py` 실행 시 PDF 로딩 단계에서 자동 수행
- VLM 결과는 **메타데이터 필드**로 저장되어 추후 답변에 반영

## VLM 입력/출력 형태
- 입력: PDF 페이지 렌더링 이미지
- 출력: 페이지별 이미지 설명/표 요약 텍스트
- 메타데이터에 다음 필드로 저장됨
  - `vlm_image_text`: VLM이 생성한 이미지 설명/표 요약
  - `vlm_image_text_present`: VLM 결과가 비어있는지 여부

## 이미지 필터링 정책 (중요)
Qwen3-VL은 **로고, 배경, 아이콘 같은 무의미 이미지**에도 응답을 생성한다.
그래서 다음 기준으로 이미지 입력을 필터링한다.

- 크기 필터: `qwen3_vl_min_image_pixels`
- 백지 비율 필터: `qwen3_vl_min_nonwhite_ratio`
- 단색/로고 필터: `qwen3_vl_min_variance`
- 도표/텍스트 없는 이미지 필터: `qwen3_vl_min_edge_energy`

**의도**
- 의미 없는 로고/아이콘은 제외하고
- 표/그래프/텍스트가 있는 이미지에만 집중

## 성능 및 한계
- VLM 호출은 **페이지 수에 비례해 느려질 수 있음**
- 표/그래프 해석은 OCR보다 낫지만, 완전한 정합성 보장은 어려움
- 결과는 **RAG에 보조 신호**로 사용해야 함

## 실행 로그 예시
```
[PDF] parser=pdfplumber path=data/files/서울특별시_2024년 지도정보 플랫폼.pdf
[VLM] PDF images -> data/files/서울특별시_2024년 지도정보 플랫폼.pdf (pages=75)
[VLM] pages:  13%|██             | 10/75 [20:34<3:30:36, 194.41s/page]
[VLM] image_text_len=1234 path=data/files/서울특별시_2024년 지도정보 플랫폼.pdf
```

## 모델 로딩 방식 (vLLM)
- vLLM 엔진으로 로컬 모델을 로딩
- FP8 환경을 유지하면서도 `max_model_len`을 낮춰 VRAM 폭주를 방지
- VRAM 16GB 환경에서는 `qwen3_vl_max_model_len`을 낮춰야 안정적

## 실패/오류 대응
### 1) MuPDF 오류
- `MuPDF error: syntax error: invalid key in dict`
- PDF 렌더링 과정에서 깨진 페이지가 있을 수 있음
- 해당 페이지는 **건너뛰고 계속 처리**해야 함

### 2) 이미지 토큰 불일치
- `Image features and image tokens do not match`
- 이미지 전처리/토큰화 불일치 문제
- 페이지 렌더링 크기 축소나 VLM 입력 포맷 점검 필요

### 3) vLLM KV 캐시 부족
- `No available memory for the cache blocks`
- `qwen3_vl_max_model_len`과 `qwen3_vl_gpu_memory_utilization` 조정 필요

## 운영 팁
- 문서가 많고 페이지가 긴 경우:
  - VLM 사용을 끄고 (`qwen3_vl_enabled=False`)
  - 필요한 문서만 VLM 재인덱싱
- 의미 없는 이미지가 너무 많으면:
  - 필터 기준을 강화 (`min_variance`, `min_edge_energy` 증가)

## 참고
- 자세한 모델 정보/사용법은 HuggingFace 모델 페이지 참고
- 추가 설명/최신 내용은 hobi2k.github.io 참고
