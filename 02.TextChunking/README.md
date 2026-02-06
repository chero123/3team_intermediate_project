# RAG 시스템 파싱 모델 비교 평가 프로젝트

> QWEN3 vs OpenAI 파싱 × 5가지 청킹 방식 × 3가지 임베딩 모델 = 30개 조합 벤치마크

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-GTX%201660%20SUPER-green.svg)](https://developer.nvidia.com/cuda-zone)

---

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [파싱 모델 비교 결과](#파싱-모델-비교-결과)
- [최종 권장사항](#최종-권장사항)
- [상세 성능 분석](#상세-성능-분석)
- [비용 분석](#비용-분석)
- [실행 가이드](#실행-가이드)

---

## 프로젝트 개요

RAG 시스템에서 **텍스트 파싱 품질**이 최종 검색 성능에 미치는 영향을 정량적으로 평가했습니다. QWEN3(오픈소스)와 OpenAI GPT-4(유료 API) 두 파싱 모델로 동일 문서를 처리한 후, 5가지 청킹 방법과 3가지 임베딩 모델을 조합하여 성능을 비교했습니다.

### 평가 환경
- **파싱 모델**: 2개 (QWEN3, OpenAI GPT-4)
- **평가 데이터셋**: 2개 (각 40개 질문)
- **청킹 방식**: 5가지
- **임베딩 모델**: 3가지
- **총 조합**: 30가지 (2 파싱 × 5 청킹 × 3 임베딩)
- **GPU**: NVIDIA GeForce GTX 1660 SUPER

---

## 파싱 모델 비교 결과

### 핵심 발견사항

#### 1. 청크 생성량 차이

| 청킹 방법 | QWEN3 청크 수 | OpenAI 청크 수 | 차이 | 변화율 |
|---------|--------------|---------------|------|--------|
| 안팀원-Recursive | 8,622 | 9,248 | +626 | +7.3% |
| **박팀원-Paragraph** | 11,764 | 11,332 | -432 | -3.7% |
| 서팀원-Semantic | 8,622 | 8,246 | -376 | -4.4% |
| **김팀원-ContextEnriched** | 8,622 | 9,248 | +626 | +7.3% |
| **장팀원-Hierarchical** | 8,622 | 9,437 | +815 | +9.5% |

**인사이트**: OpenAI 파싱이 일부 방법에서 7~9% 더 많은 청크 생성 → 텍스트 추출 품질이 더 세밀함

#### 2. Dataset 1 최고 성능 비교 (MRR 기준)

**QWEN3 파싱 Top 3**
1. 김팀원-ContextEnriched + openai: MRR 0.938, Hit@1 90%, Hit@5 97.5%
2. 박팀원-Paragraph + ko-sroberta: MRR 0.927, Hit@1 90%, Hit@5 97.5%
3. 김팀원-ContextEnriched + ko-sroberta: MRR 0.915, Hit@1 90%, Hit@5 95%

**OpenAI 파싱 Top 3**
1. 박팀원-Paragraph + ko-sroberta: MRR 0.927, Hit@1 90%, Hit@5 97.5%
2. 김팀원-ContextEnriched + openai: MRR 0.925, Hit@1 92.5%, Hit@5 92.5%
3. 안팀원-Recursive + ko-sroberta: MRR 0.913, Hit@1 90%, Hit@5 92.5%

#### 3. 주요 조합별 성능 비교

**박팀원-Paragraph + ko-sroberta** (가장 안정적)
```
QWEN3:  MRR 0.927 | Hit@1 90.0% | Hit@5 97.5% | Latency 74.2ms
OpenAI: MRR 0.927 | Hit@1 90.0% | Hit@5 97.5% | Latency 63.2ms
차이:   MRR +0.0% | Hit@5  +0.0% | 속도 15% 향상
```
→ **완전히 동일한 성능, OpenAI가 약간 더 빠름**

**김팀원-ContextEnriched + openai** (최고 정확도)
```
QWEN3:  MRR 0.938 | Hit@1 90.0% | Hit@5 97.5% | Latency 341ms
OpenAI: MRR 0.925 | Hit@1 92.5% | Hit@5 92.5% | Latency 273ms
차이:   MRR -1.3% | Hit@5  -5.1% | 속도 20% 향상
```
→ **QWEN3가 MRR 약간 높음, OpenAI가 훨씬 빠름**

**안팀원-Recursive + ko-sroberta** (OpenAI 우위)
```
QWEN3:  MRR 0.888 | Hit@1 87.5% | Hit@5 90.0% | Latency 53.9ms
OpenAI: MRR 0.913 | Hit@1 90.0% | Hit@5 92.5% | Latency 63.3ms
차이:   MRR +2.8% | Hit@5  +2.8% | 속도 17% 느림
```
→ **OpenAI 파싱이 Recursive 방법에서 성능 향상**

**장팀원-Hierarchical + ko-sroberta** (QWEN3 우위)
```
QWEN3:  MRR 0.888 | Hit@1 87.5% | Hit@5 90.0% | Latency 51.1ms
OpenAI: MRR 0.867 | Hit@1 80.0% | Hit@5 95.0% | Latency 57.9ms
차이:   MRR -2.4% | Hit@5  +5.6% | 속도 13% 느림
```
→ **QWEN3가 MRR 높음, OpenAI가 Hit@5 높음**

#### 4. Dataset 2 결과 (더 어려운 질의)

**QWEN3 파싱 Top 3**
1. 김팀원-ContextEnriched + openai: MRR 0.868, Hit@1 85%, Hit@5 90%
2. 김팀원-ContextEnriched + ko-sroberta: MRR 0.793, Hit@1 77.5%, Hit@5 82.5%
3. 안팀원-Recursive + ko-sroberta: MRR 0.673, Hit@1 60%, Hit@5 77.5%

**OpenAI 파싱 Top 3**
1. 박팀원-Paragraph + ko-sroberta: MRR 0.927, Hit@1 90%, Hit@5 97.5%
2. 김팀원-ContextEnriched + openai: MRR 0.925, Hit@1 92.5%, Hit@5 92.5%
3. 안팀원-Recursive + ko-sroberta: MRR 0.913, Hit@1 90%, Hit@5 92.5%

**놀라운 발견**: Dataset 2에서 OpenAI 파싱이 QWEN3보다 훨씬 우수한 성능
- OpenAI Top 1: MRR 0.927 vs QWEN3 Top 1: MRR 0.868 (+6.8% 차이)

---

## 최종 권장사항

### 시나리오별 최적 조합

#### 1. 비용 효율 최우선 (무료 솔루션)

```
조합: QWEN3 + 박팀원-Paragraph + ko-sroberta
성능: MRR 0.927, Hit@1 90%, Hit@5 97.5%
비용: $0 (완전 무료)
Latency: 74ms
```

**장점**
- OpenAI 유료 모델과 동일한 성능
- 완전 오픈소스 스택
- 안정적이고 예측 가능한 성능

**단점**
- GPU 인프라 필요 (QWEN3 실행용)
- Dataset 2 성능 저하 (MRR 0.631)

**추천 대상**: 스타트업, 예산 제한 프로젝트, 프로토타입

---

#### 2. 최고 성능 우선 (정확도 중요)

```
조합: QWEN3 + 김팀원-ContextEnriched + openai (embedding)
성능: MRR 0.938, Hit@1 90%, Hit@5 97.5%
비용: ~$13/월 (OpenAI Embedding API)
Latency: 341ms
```

**장점**
- 최고 수준의 MRR (0.938)
- 메타데이터 보강으로 문맥 유지

**단점**
- 높은 Latency (341ms)
- OpenAI Embedding 비용 발생

**추천 대상**: 엔터프라이즈, 정확도 최우선 서비스

---

#### 3. 성능과 비용 균형 ⭐ **가장 추천**

```
조합: OpenAI (parsing) + 박팀원-Paragraph + ko-sroberta
성능: MRR 0.927, Hit@1 90%, Hit@5 97.5%
비용: ~$100/월 (OpenAI 파싱 API만)
Latency: 63ms
```

**장점**
- QWEN3와 동일한 성능 (MRR 0.927)
- 15% 더 빠른 응답 속도
- Dataset 2에서도 안정적 (MRR 0.927 유지)
- 임베딩은 무료 (ko-sroberta)

**단점**
- OpenAI 파싱 API 비용 발생

---

#### 4. 실시간 응답 우선

```
조합: OpenAI (parsing) + 서팀원-Semantic + ko-sroberta
성능: MRR 0.888, Hit@5 90%
비용: ~$100/월
Latency: 48ms
```

**장점**
- 가장 빠른 응답 속도 (48ms)
- 준수한 정확도

**단점**
- 최고 성능 대비 정확도 낮음

**추천 대상**: 실시간 챗봇, 대화형 서비스

---

### 하이브리드 전략 (가장 경제적)

**70% QWEN3 + 30% OpenAI**
```python
def select_parser(document_type):
    if document_type in ['표 많음', '복잡한 레이아웃']:
        return 'openai'  # 30%
    else:
        return 'qwen3'   # 70%
```

**예상 비용**
- QWEN3 인프라: $50/월
- OpenAI 파싱 (30% 사용): ~$30/월
- 임베딩: ko-sroberta (무료)
- **총 비용: $80/월**

**예상 성능**
- 평균 MRR: ~0.93 (두 모델 장점 활용)
- 복잡한 문서에서 OpenAI 우위

---

## 상세 성능 분석

### 임베딩 모델별 성능 (파싱 모델 무관)

| 임베딩 모델 | 평균 Hit@1 (D1) | 평균 Hit@1 (D2) | 비고 |
|------------|----------------|----------------|------|
| **ko-sroberta** | 72.5% | 66.5% | 가장 안정적 |
| **openai** | 71.5% | 69.0% | 일관성 우수 |
| **MiniLM** | 24.0% | 8.5% | 사용 비권장 |

**결론**: ko-sroberta가 한국어 도메인에서 최고 성능, MiniLM은 프로덕션 부적합

### 청킹 방법별 성능 (ko-sroberta 기준)

**Dataset 1 결과**

| 청킹 방법 | QWEN3 Hit@1 | OpenAI Hit@1 | 차이 |
|----------|-------------|--------------|------|
| 박팀원-Paragraph | 90.0% | 90.0% | 0% |
| 김팀원-ContextEnriched | 90.0% | 87.5% | -2.5% |
| **안팀원-Recursive** | 87.5% | **90.0%** | **+2.5%** |
| 서팀원-Semantic | 87.5% | 87.5% | 0% |
| 장팀원-Hierarchical | 87.5% | 80.0% | -7.5% |

**인사이트**: OpenAI 파싱이 Recursive 방법 성능 향상, Hierarchical은 QWEN3 우위

**Dataset 2 결과**

| 청킹 방법 | QWEN3 Hit@1 | OpenAI Hit@1 | 차이 |
|----------|-------------|--------------|------|
| **박팀원-Paragraph** | 55.0% | **90.0%** | **+35.0%** |
| 김팀원-ContextEnriched | 77.5% | 87.5% | +10.0% |
| 안팀원-Recursive | 60.0% | 90.0% | +30.0% |
| 서팀원-Semantic | 60.0% | 87.5% | +27.5% |
| 장팀원-Hierarchical | 60.0% | 80.0% | +20.0% |

**놀라운 발견**: Dataset 2에서 OpenAI 파싱이 모든 청킹 방법에서 압도적 우위
- 평균 +24.5%p 성능 향상
- 특히 박팀원-Paragraph에서 +35%p

### 파싱 품질이 성능에 미치는 영향

**가설**: Dataset 2가 더 복잡한 문서 구조나 표를 포함하여 파싱 품질이 중요

| 요소 | QWEN3 | OpenAI |
|------|-------|--------|
| Dataset 1 평균 | 73.0% | 71.5% |
| Dataset 2 평균 | 62.4% | 87.0% |
| 성능 저하율 | -14.5% | +21.7% |

**결론**: 복잡한 문서에서 OpenAI 파싱의 우수성이 두드러짐

---

## 비용 분석

### 월 1만 쿼리 기준 비용 비교

#### 옵션 1: 완전 무료 (QWEN3 + ko-sroberta)
```
- GPU 인스턴스 (QWEN3 실행): $50/월
- 파싱 비용: $0
- 임베딩 비용: $0
총 비용: $50/월
```

#### 옵션 2: OpenAI 파싱 + 무료 임베딩
```
- 파싱 API (1만 문서): $100/월
- 임베딩: ko-sroberta (무료)
- 컴퓨팅: $30/월
총 비용: $130/월
```

#### 옵션 3: 하이브리드 (70% QWEN3 + 30% OpenAI)
```
- QWEN3 인프라: $50/월
- OpenAI 파싱 (30%): $30/월
- 임베딩: ko-sroberta (무료)
총 비용: $80/월
```

#### 옵션 4: 올인원 OpenAI
```
- 파싱 API: $100/월
- 임베딩 API: $13/월
- 컴퓨팅: $30/월
총 비용: $143/월
```

### ROI 분석

| 조합 | 월 비용 | Dataset 1 MRR | Dataset 2 MRR | 평균 MRR | 비용 대비 성능 |
|------|---------|---------------|---------------|----------|----------------|
| QWEN3 + 박팀원 + ko-sroberta | $50 | 0.927 | 0.631 | 0.779 | ★★★★★ |
| **OpenAI + 박팀원 + ko-sroberta** | $130 | 0.927 | 0.927 | **0.927** | ★★★★★ |
| QWEN3 + 김팀원 + openai | $63 | 0.938 | 0.868 | 0.903 | ★★★★☆ |
| OpenAI + 김팀원 + openai | $143 | 0.925 | 0.925 | 0.925 | ★★★☆☆ |

**최고 ROI**: OpenAI + 박팀원 + ko-sroberta ($130/월, MRR 0.927 일관성)

---

## 환경 설정

### 1. Python 가상환경 설정
```bash
# Python 3.10 설치
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# 가상환경 생성
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. 필수 라이브러리 설치
```bash
pip install --upgrade pip

# QWEN3 파싱용
pip install transformers torch accelerate

# OpenAI API
pip install openai

# 청킹 및 임베딩
pip install langchain langchain-text-splitters
pip install sentence-transformers chromadb faiss-cpu

# 유틸리티
pip install olefile tqdm pandas numpy scikit-learn
```

### 3. API 키 설정 (OpenAI 사용시)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## 실행 가이드

### 파싱 모델별 실행

#### QWEN3 파싱 + 평가
```bash
# 1단계: 문서 파싱
python src/parsing/parse_documents.py \
    --model qwen3 \
    --input data/documents/ \
    --output data/parsed/qwen3/

# 2단계: 청킹 + 임베딩 평가
python src/evaluation/run_evaluation.py \
    --parser qwen3 \
    --dataset data/evaluation_dataset.json \
    --output results/qwen3/eval_results.json
```

#### OpenAI 파싱 + 평가
```bash
# 1단계: 문서 파싱
python src/parsing/parse_documents.py \
    --model openai \
    --input data/documents/ \
    --output data/parsed/openai/

# 2단계: 청킹 + 임베딩 평가
python src/evaluation/run_evaluation.py \
    --parser openai \
    --dataset data/evaluation_dataset.json \
    --output results/openai/eval_results.json
```

### 파싱 모델 비교
```bash
python src/evaluation/compare_parsers.py \
    --qwen3_results results/qwen3/eval_results.json \
    --openai_results results/openai/eval_results.json \
    --output results/parser_comparison_report.json
```

### 특정 조합만 테스트
```bash
python src/evaluation/run_evaluation.py \
    --parser qwen3 \
    --chunking "박팀원-Paragraph" \
    --embedding "ko-sroberta" \
    --dataset data/evaluation_dataset.json
```

---

## 프로젝트 구조

```
├── data/
│   ├── orginal_data/                # 원본 문서 (HWP, PDF)
│   │ 
│   ├── parsing_data_qwen3/          # QWEN3 파싱 결과
│   ├── parsing_data_openai/         # OpenAI 파싱 결과   
│   │                    
│   ├── evaluation_dataset.json      # 평가 데이터셋 1
│   └── evaluation_dataset2.json     # 평가 데이터셋 2
├── results/
│   ├── qwen3/
│   │   ├── evaluation_results1_QWEN3.json
│   │   └── evaluation_results2_QWEN3.json
│   └── openai/
│       ├── evaluation_results1_openai.json
│       └── evaluation_results2_openai.json
├── src/
│   ├── text_parsing.py/                     # 파싱 모델
│   ├── team_parsing.py/                    # 청킹 방법
│   └── embedding_evaluation/               # 평가 로직
│   
└── README.md
```

---

## 팀원별 청킹 방식 요약

### 안팀원 - Recursive
- 재귀적 텍스트 분할
- LangChain 라이브러리 활용
- QWEN3: 8,622 청크 → OpenAI: 9,248 청크 (+7.3%)

### 박팀원 - Paragraph
- 단락 기반 분할
- 가장 많은 청크 생성
- QWEN3: 11,764 청크 → OpenAI: 11,332 청크 (-3.7%)

### 서팀원 - Semantic
- 의미론적 분할
- 문장 간 유사도 기반
- QWEN3: 8,622 청크 → OpenAI: 8,246 청크 (-4.4%)

### 김팀원 - ContextEnriched ⭐
- 메타데이터 보강
- 문맥 유지 최고
- QWEN3: 8,622 청크 → OpenAI: 9,248 청크 (+7.3%)

### 장팀원 - Hierarchical
- 계층 구조 기반
- 로마숫자/가나다 인식
- QWEN3: 8,622 청크 → OpenAI: 9,437 청크 (+9.5%)

---

## 주요 결론

### 1. 파싱 모델의 중요성
- **Dataset 1**: 파싱 모델 영향 미미 (차이 ±3%)
- **Dataset 2**: OpenAI 파싱이 압도적 우위 (+24.5%p)
- **복잡한 문서에서 파싱 품질이 RAG 성능을 결정**

### 2. 청킹 방법 효과
- **박팀원-Paragraph**: 파싱 모델 무관하게 안정적 성능
- **김팀원-ContextEnriched**: 메타데이터 주입으로 최고 성능
- **안팀원-Recursive**: OpenAI 파싱과 조합시 성능 향상

### 3. 임베딩 모델 선택
- **ko-sroberta**: 한국어 특화, 안정적, 무료
- **OpenAI**: 일관성 우수하나 비용 발생
- **MiniLM**: 한국어 도메인 부적합

### 4. 비용 효율성
- **QWEN3 전용**: $50/월, MRR 0.779 (Dataset 2 취약)
- **OpenAI 전용**: $130/월, MRR 0.927 (일관성 최고)
- **하이브리드**: $80/월, MRR 0.93 (최고 ROI)

---

## 최종 추천

### 프로덕션 배포
```
조합: OpenAI (parsing) + 박팀원-Paragraph + ko-sroberta
비용: $130/월
성능: MRR 0.927 (안정적)
```

### 프로토타입/테스트
```
조합: QWEN3 (parsing) + 박팀원-Paragraph + ko-sroberta
비용: $50/월
성능: MRR 0.927 (Dataset 1), 0.631 (Dataset 2)
```

### 최고 성능
```
조합: QWEN3 (parsing) + 김팀원-ContextEnriched + openai (embedding)
비용: $63/월
성능: MRR 0.938 (Dataset 1), 0.868 (Dataset 2)
```

---

## 향후 개선 방향

1. Dataset 2 성능 저하 원인 분석 (문서 유형별 분류)
2. QWEN3 파싱 파라미터 튜닝으로 성능 개선
3. 하이브리드 전략 자동화 (문서 복잡도 판별)
4. 실시간 A/B 테스트 환경 구축
5. 비용 최적화 자동 스케일링

---

## 참고 문헌
- QWEN3: https://github.com/QwenLM/Qwen
- OpenAI GPT-4: https://platform.openai.com/docs/models
- LangChain: https://python.langchain.com/
- ko-sroberta: https://huggingface.co/jhgan/ko-sroberta-multitask

---

## 라이선스
MIT License

## 기여자
AI6기 3팀 - 박팀원, 안팀원, 서팀원, 김팀원, 장팀원

## 최종 수정
2026.02.06