## 📁 프로젝트 구조

### 1) `baseline/` — RAG 베이스라인 코드 모음
기존에 사용하던 RAG 파이프라인 코드를 한곳에 모아둔 “기준(베이스라인)” 구현입니다.  
각 파일은 파이프라인 단계 순서대로 번호를 붙여 정리했습니다.

- `1. text_parsing.py` : 문서 텍스트 추출/정리(파싱)
- `2. build_vector_db.py` : 문서 임베딩 & 벡터 DB(예: FAISS) 구축
- `3. retriever_hy+rerank.py` : 하이브리드 검색 + 리랭크 조합 실험
- `3+. retriever_mmr.py` : MMR 기반 검색(다양성 반영) 실험
- `4. context_builder.py` : 검색된 문서 조각을 컨텍스트로 구성
- `5. generate_answer.py` : LLM 답변 생성 단계
- `hwp_parsing.ipynb` : HWP 파싱/추출 실험 노트북

---

### 2) `hwp_to_pdf_to_md/` — HWP → PDF → MD 변환 파이프라인
HWP 문서를 **PDF로 변환한 뒤**, PDF를 **Markdown(MD)로 변환**하는 흐름입니다.

**MD를 쓰는 이유:**  
VLM(비전-언어 모델)을 활용해 **표(table) 형식까지 최대한 유지**하면서 텍스트를 정리하기 위함입니다.

- `1. hwp_to_pdf.py` : HWP → PDF 변환
- `2. pdf_to_md.py` : PDF → MD 변환(표 형식 반영 목적)
- `pdf_to_md_test.ipynb` : PDF **단일 파일** 대상으로 변환 테스트/검증

---

### 3) `extract_tables_to_md.py` — (전체 문서 대상) 표 추출/MD 변환 작업
단일 테스트가 아니라, **여러 문서(전체 파일)** 를 대상으로  
VLM을 이용해 **표를 추출하고 Markdown으로 반영**하는 “실제 처리용” 스크립트입니다.
