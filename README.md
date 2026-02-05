유종 님이 요청하신 대로 모든 이모티콘을 제거하고, 기술 문서의 격식에 맞춘 담백한 톤으로 최종 README를 재작성했습니다.

# 3팀 중기 프로젝트: RAG 기반 AI 문서 비서

본 프로젝트는 대량의 사업 요청서(PDF, TXT)를 분석하여 사용자의 질문에 정확한 답변을 제공하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 핵심 기능

* **하이브리드 세그멘테이션**: SemanticChunker와 RecursiveCharacterSplitter를 결합하여 문맥의 의미를 보존하는 동시에 검색에 최적화된 500자 단위의 청크를 생성합니다.
* **고정밀 시맨틱 검색**: 질문의 의도를 분석하여 벡터 데이터베이스에서 관련성이 높은 상위 10개(k=10)의 문서 조각을 추출합니다.
* **출처 기반 답변 생성**: GPT-5 모델을 활용하여 검색된 문맥 기반의 답변을 생성하며, 각 정보의 출처(파일명)를 명시하여 신뢰성을 확보합니다.
* **웹 인터페이스**: Gradio를 활용하여 실시간 질의응답이 가능한 채팅 화면을 제공합니다.

## 프로젝트 구조

```text
.
├── src/                # 핵심 로직 (database.py, chain.py)
├── data/
│   └── original_data/  # 원본 PDF 및 파싱된 TXT 파일 보관
├── db/                 # 생성된 벡터 데이터베이스 (ChromaDB)
├── main.py             # 터미널 실행용 스크립트
├── main_web.py         # Gradio 웹 실행용 스크립트 (환경변수 포함 권장)
├── rebuild_db.py       # 데이터베이스 재생성 스크립트 (청킹 전략 포함)
├── requirements.txt    # 필요 라이브러리 목록
├── .env                # API 키 설정 (OPENAI_API_KEY)
└── tmp/                # 시스템 용량 제한 대응용 임시 디렉토리

```

## 실행 가이드

### 1. 가상환경 설정 및 라이브러리 설치

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 필수 패키지 설치 (서버 용량 부족 시 임시 디렉토리 지정)
mkdir -p tmp_install
TMPDIR=./tmp_install pip install --no-cache-dir -r requirements.txt

```

### 2. 데이터베이스 생성 및 재구축

문서가 추가되거나 청킹 전략(500자 세분화 및 메타데이터 주입)을 반영해야 할 경우 실행합니다.

```bash
python rebuild_db.py

```

### 3. 서비스 실행

**터미널 모드:**

```bash
python main.py

```

**웹 인터페이스 모드 (Gradio):**
서버의 임시 디렉토리 용량 제한(Errno 2)을 방지하기 위해 프로젝트 내부 경로를 지정하여 실행합니다.

```bash
mkdir -p tmp
TMPDIR=./tmp python main_web.py

```

* 실행 후 출력되는 public URL을 통해 외부에서 접속이 가능합니다.

---

## 기술적 특이사항

* **검색 최적화**: 수치 데이터(사업비, 일정 등)의 정확도를 높이기 위해 리트리버의 k값을 10으로 설정하였습니다.
* **환경 대응**: 시스템 디스크 사용률이 높은 환경을 고려하여 모든 임시 파일 생성 경로를 프로젝트 내부 폴더로 우회 설정하였습니다.

