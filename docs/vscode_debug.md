# VSCode에서 모듈 디버깅 + 노트북처럼 셀 단위 작업하기

이 문서는 **Python 모듈 모드 디버깅**을 유지하면서, **ipynb처럼 셀 단위로 테스트**하는 실전 워크플로우를 정리한다.

## 1. 전제: 올바른 인터프리터 선택

VSCode가 **uv 가상환경**(혹은 그 외)을 사용해야 패키지/상대 import가 정상 작동한다.

- `Ctrl+Shift+P` → `Python: Select Interpreter`
- `3team_intermediate_project/.venv/bin/python` 선택

## 2. 모듈 디버깅 설정 (상대 import 문제 해결)

`.vscode/launch.json`에 아래 설정을 사용한다.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "rag indexing (debug)",
      "type": "python",
      "request": "launch",
      "module": "rag.indexing",
      "cwd": "${workspaceFolder}/3team_intermediate_project/src",
      "console": "integratedTerminal",
      "python": "${workspaceFolder}/3team_intermediate_project/.venv/bin/python"
    }
  ]
}
```

이렇게 실행하면 `from .config import ...` 같은 **상대 import**가 깨지지 않는다.

## 3. .py에서 ipynb처럼 셀 단위 실행하기

VSCode에서는 `.py`에서도 셀 단위 실행이 가능하다.

### 방법 A: 셀 구분 후 실행

1. 원하는 위치에 아래 셀 구분자 추가:
   ```python
   # %%
   ```
2. 셀 위에 표시되는 **Run Cell** 버튼 클릭

VSCode는 해당 셀만 실행하고, 결과는 **Python Interactive Window**에 표시된다.

### 방법 B: 선택 실행

- 코드 블록 선택 > 우클릭 > **Run Selection/Line in Python Interactive Window**

## 4. 디버깅 + 셀 실행을 같이 쓰는 팁

- 셀 실행은 **실험/검증**에 쓰고
- 디버깅은 `launch.json`의 **모듈 디버그 설정**으로 실행한다.
- 상대 import가 필요한 경우, **셀 실행 전에 아래 환경으로 REPL 실행**:

```bash
cd 3team_intermediate_project
PYTHONPATH=src python
```

그 다음:
```python
from rag import data
# data.load_documents(...) 같은 테스트 수행
```

## 5. 자주 겪는 문제

### Q. `attempted relative import with no known parent package`

- 파일을 직접 실행했기 때문임 (`python file.py`)
- 반드시 **모듈 모드**로 실행할 것

```bash
PYTHONPATH=src python -m rag.indexing
```

### Q. docx/hwp5 등이 안 잡힘

- 디버거가 다른 인터프리터를 쓰는 경우가 대부분
- 반드시 `.venv`를 선택하고 설치 확인

```bash
python -c "import docx; print(docx.__version__)"
```

## 요약

- **모듈 디버깅**: 상대 import 문제 해결
- **셀 실행**: ipynb처럼 빠른 검증
- 두 방식을 섞어 쓰면 `.py`에서도 노트북 수준의 테스트 경험을 얻을 수 있다.
