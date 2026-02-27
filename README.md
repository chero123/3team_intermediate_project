## 설치 및 실행 가이드
**환경 설정 및 데이터 전처리 부분은 우정님이 작성하신 README.md 참고하시면 됩니다.**

### 1. 환경 설정
프로젝트를 클론하고 필수 라이브러리를 설치합니다. (Python 3.10+ 권장)

> 🔗 **상세 설정 가이드**: [GitHub README 참고](https://github.com/chero123/3team_intermediate_project/blob/jang-woojung/README.md)

```bash
# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정
프로젝트 루트 경로에 `.env` 파일을 생성하고 OpenAI API 키를 입력합니다.

```ini
# .env 파일 생성
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 데이터 전처리 (순서 중요!)
데이터를 검색 가능한 형태로 만들기 위해 **반드시 아래 순서대로 실행**해야 합니다.

> 🔗 **텍스트 파싱 상세 가이드**: [GitHub TextParsing README 참고](https://github.com/chero123/3team_intermediate_project/blob/jang-woojung/TextParsing/README.md)

**Step 1: 텍스트 추출 (`text_parsing.py`)**
`data/original_data` 폴더에 있는 HWP, PDF 파일을 읽어 텍스트로 변환합니다.
```bash
python TextParsing/text_parsing.py
```

**Step 2: 벡터 DB 구축 (`create_vectordb.py`)**
추출된 텍스트에 CSV 메타데이터를 결합하여 헤더를 주입하고, ChromaDB에 저장합니다.
```bash
python TextParsing/create_vectordb.py
```

### 4. 애플리케이션 실행 (`app.py`)
DB 구축이 완료되면 웹 인터페이스를 실행합니다.

```bash
streamlit run TextParsing/app.py 
```

### 5. 터미널 모드 실행 (`rag_system.py`)
웹 브라우저 없이 터미널에서 바로 질의응답을 테스트하고 싶다면 아래 명령어를 실행하세요.

```bash
python run TextParsing/rag_system.py
```