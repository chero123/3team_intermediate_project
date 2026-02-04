# 🏢 3팀 중기 프로젝트: RAG 기반 AI 문서 비서

이 프로젝트는 대량의 사업 요청서(PDF, HWP)를 분석하여 사용자의 질문에 정확하게 답변하는 **RAG(Retrieval-Augmented Generation)** 시스템입니다.

## 🛠 주요 기능
* **문서 지식화**: PDF 및 HWP 문서에서 텍스트를 추출하고 의미 단위로 분할(Chunking).
* **시맨틱 검색**: 질문의 의도를 파악하여 관련성이 높은 문서 조각을 검색.
* **AI 답변 생성**: 최신 LLM 모델을 활용하여 검색된 문맥 기반의 정확한 답변 제공.
* **웹 인터페이스**: Gradio를 활용한 사용자 친화적인 채팅 화면 제공.

## 📂 프로젝트 구조
```text
.
├── src/                # 핵심 로직 (DB 로드, RAG 체인)
├── data/               # 데이터 관리 (raw, processed)
├── db/                 # 벡터 데이터베이스 (ChromaDB)
├── main.py             # 터미널 실행용 스크립트
├── main_web.py         # Gradio 웹 실행용 스크립트
├── requirements.txt    # 필요 라이브러리 목록
└── .env                # API 키 설정 (별도 생성 필요)
```

## 🚀 시작하기 (팀원용 가이드)

### 1. 가상환경 설정 및 라이브러리 설치
```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 최상위 폴더에 `.env` 파일을 생성하고 본인의 OpenAI API 키를 입력합니다.
```text
OPENAI_API_KEY="본인의_API_키_입력"
```

### 3. 실행
**터미널에서 결과 확인:**
```bash
python main.py
```

**웹 인터페이스 실행 (Gradio):**
```bash
python main_web.py
```
* 실행 후 터미널에 출력되는 `public URL: https://xxxx.gradio.live` 링크로 접속하세요.

---

### 💡 PM 메모 (팀 협업 수칙)
1. **보안**: `.env` 파일과 `venv/` 폴더는 절대 Git에 Push하지 마세요. (이미 `.gitignore`에 반영됨)
2. **데이터**: 새로운 파싱 결과물은 `data/processed/` 폴더에 공유해 주세요.
3. **업데이트**: 새로운 라이브러리를 설치했다면 `pip freeze > requirements.txt`를 실행해 목록을 갱신해 주세요.
