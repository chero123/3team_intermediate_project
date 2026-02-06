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

## 코드 위치/구성
Qwen3-VL 로직은 `src/rag/data.py`에 집중되어 있다.

주요 함수
- `_get_qwen3_vl(config)`  
  - 로컬 경로(`config.qwen3_vl_model_path`)에서만 로딩  
  - `AutoProcessor` + vLLM `LLM` 인스턴스를 전역 캐시에 보관
- `_prepare_vllm_inputs(messages, processor)`  
  - Qwen3-VL의 **chat template**로 텍스트 프롬프트 구성  
  - `process_vision_info()`로 이미지 입력을 **multi_modal_data**로 변환
- `_vlm_extract_from_image(image, config)`  
  - 이미지 1장을 Qwen3-VL에 전달해 수치/표/텍스트를 요약  
  - `SamplingParams(temperature=0.0)`로 결정적 출력 유지
- `_extract_pdf_images_with_vlm(path, config)`  
  - `fitz`로 PDF 내 이미지를 추출/렌더링  
  - 중복 이미지 제거 + 의미 없는 이미지 필터링  
  - 페이지/이미지별로 VLM 추론
- `_is_meaningful_image(image, config)`  
  - 해상도/분산/비백색 비율/엣지 에너지로 무의미 이미지 제거

관련 코드 흐름
- `extract_text_from_pdf_with_vlm()`  
  - PDF 본문 텍스트 + 이미지 텍스트를 동시에 추출
- `load_documents()`  
  - PDF 처리 시 `extract_text_from_pdf_with_vlm()` 호출  
  - `vlm_image_text`, `vlm_image_text_present` 메타데이터 기록

## vLLM 호출 형식(코드 기준)
`_prepare_vllm_inputs()`가 만든 입력 구조는 아래와 같다.
```
{
  "prompt": "<chat template text>",
  "multi_modal_data": {"image": <image tensor>},
  "mm_processor_kwargs": <video/image metadata>
}
```

## Qwen3-VL 챗 템플릿 적용
`_prepare_vllm_inputs()`에서 `processor.apply_chat_template()`을 호출해 Qwen3-VL의
**공식 chat template**을 사용한다.

동작
- `messages` 리스트를 받아 **텍스트 프롬프트**로 변환
- `add_generation_prompt=True`로 **생성 시작 토큰**을 자동 추가

코드 위치
- `src/rag/data.py` -> `_prepare_vllm_inputs()`

## Qwen3-VL 프롬프트(이미지 전용)
- 이미지 입력과 함께 **표/수치 중심 요약**을 요청하는 프롬프트가 사용된다.
- 목표는 **의미 없는 해설**이 아니라 **수치/표 요약**이다.
- 프롬프트 문자열은 `QWEN3_VL_PROMPT` 상수로 관리된다.

## 모델 경로
- **로컬 경로 기준**: `models/qwen3-vl-8b`
- HuggingFace에서 다운로드를 요구하지 않도록 로컬 경로를 사용한다.

## 모델 초기화(로딩) 세부
모델 초기화는 `_get_qwen3_vl(config)`에서 수행된다.

1) 로컬 모델 경로 확인  
   - `config.qwen3_vl_model_path` 존재 여부를 검사  
   - 없으면 즉시 예외 발생

2) Processor 로딩  
   - `AutoProcessor.from_pretrained(..., trust_remote_code=True, local_files_only=True)`  
   - 이미지 전처리 + 텍스트 토큰화 포함

3) vLLM 엔진 로딩  
   - `vllm.LLM(model=..., gpu_memory_utilization=..., max_model_len=...)`  
   - `limit_mm_per_prompt={\"image\": 1, \"video\": 0}`  
   - 멀티모달 입력은 **이미지 1장만 허용**

4) 전역 캐시 재사용  
   - `_QWEN3_VL_CACHE`에 processor/llm 저장  
   - 동일 프로세스 내 재사용

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

## 주의사항
- `fitz`가 없으면 이미지 렌더링 자체가 불가능하다.
- `qwen3_vl_enabled=False`면 이미지 추출은 즉시 스킵된다.
- vLLM 엔진은 초기 로딩이 느리며, **첫 실행**이 가장 오래 걸린다.
- PDF가 깨진 경우 `MuPDF error` 로그가 발생할 수 있다. 해당 페이지는 스킵된다.

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
