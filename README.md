# Hwp 문서 파싱 프로젝트 환경 설정 가이드 (Python 3.10 + venv)

이 문서는 Google Cloud VM 서버 환경에서 Python 3.10 가상환경을 구축하고 VS Code와 연결하는 방법을 설명합니다.

---

## 1. 개인 작업 폴더 생성 및 이동
먼저 서버에 접속한 후 본인의 개인 폴더를 생성하고 해당 디렉토리로 이동합니다. (명칭 변경 가능)

```bash
mkdir -p ~/jwj_folder
cd ~/jwj_folder
```

## 2. Python 3.10 설치 및 venv 도구 준비
서버에 Python 3.10이 설치되어 있지 않거나 venv 모듈이 없는 경우 아래 명령어를 실행합니다.

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

## 3. 가상환경(venv) 생성 및 활성화
3.1 가상환경 생성<br/>
프로젝트 폴더 내에서 Python 3.10 버전을 명시하여 가상환경을 생성합니다.

```bash
python3.10 -m venv .venv
```

3.2 가상환경 활성화<br/>
터미널에서 가상환경을 활성화합니다.

```bash
source .venv/bin/activate
```
- 활성화 성공 시 터미널 프롬프트 맨 앞에 (.venv) 문구가 나타나는지 확인하십시오.

## 4. 필수 라이브러리 설치
가상환경이 활성화된 상태에서 필요한 패키지를 설치합니다.

```bash
pip install --upgrade pip
pip install olefile
```

## 5. VS Code 인터프리터(커널) 설정
VS Code 에디터가 방금 만든 가상환경의 Python을 사용하도록 설정해야 합니다.<br/>

VS Code 실행 후 프로젝트 폴더(jwj_folder)를 엽니다.<br/>

명령 팔레트를 실행합니다. (단축키: Ctrl + Shift + P)<br/>

검색창에 "Python: Select Interpreter"를 입력하고 선택합니다.<br/>

리스트에 .venv 경로가 바로 뜨지 않을 경우, "+ Enter interpreter path..."를 클릭합니다.<br/>

"Find..."를 눌러 직접 탐색하거나 아래의 절대 경로를 직접 입력창에 붙여넣습니다.<br/>

경로: /home/spai0601/jwj_folder/.venv/bin/python<br/>

VS Code 하단 상태 바에 "Python 3.10.x ('.venv': venv)"라고 표시되는지 확인합니다.<br/>

## 6. 실행 및 확인
모든 설정이 완료되면 Python 파일을 생성하여 아래 코드로 정상 작동 여부를 확인합니다.

`data` 폴더를 만들어서 .hwp 파일을 불러들어와 실행을 확인합니다.