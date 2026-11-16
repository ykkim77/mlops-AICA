# 5주차 실습 — CI/CD 파이프라인 설계

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 실습 환경: WSL2 Ubuntu 22.04 / Python 3.10+ / Git / GitHub  
> 클라우드 계정: 필요 없음  
> 실습 기준 ZenML: 0.96.4

---

# 0. 실습 개요

5주차 실습: 총 2개 차시(2·3차시)로 구성

```text
2차시
Git Push
   ↓
GitHub Actions
   ↓
Checkout
   ↓
Python 환경 구성
   ↓
pytest
   ↓
PASS / FAIL
```

```text
3차시
Git Push
   ↓
GitHub Actions
   ↓
Test
   ↓
ZenML Pipeline 실행
   ↓
Preprocess
   ↓
Train
```

역할 구분:

```text
GitHub Actions
= 언제 실행할 것인가?
= Repository Event 기반 자동화

ZenML
= 무엇을 어떤 순서로 실행할 것인가?
= ML Step / Pipeline 구성
```

---

# 1. 공통 프로젝트 생성

```bash
mkdir -p ~/mlops-week05
cd ~/mlops-week05

git init -b main

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install "zenml[local]==0.96.4" scikit-learn pandas pytest

mkdir -p src tests .github/workflows
touch src/__init__.py

cat > pytest.ini <<'EOF'
[pytest]
pythonpath = .
EOF

cat > .gitignore <<'EOF'
.venv/
__pycache__/
.pytest_cache/
EOF
```

- `zenml[local]`: ZenML 기본 패키지만 설치하면 로컬 DB를 사용할 수 없어 `zenml init`, Pipeline 실행이 실패함 → 로컬 실행용 의존성 포함 설치 필요
- `pytest.ini`: 프로젝트 루트를 import 경로에 추가 → 없으면 `pytest` 실행 시 `ModuleNotFoundError: No module named 'src'` 발생
- `.gitignore`: 이 저장소는 GitHub에 Push하므로 `.venv` 업로드 방지가 필수임

> Python 3.12 이하의 `venv`는 자체 `.gitignore`를 만들지 않음 → `git add .` 실행 시 `.venv` 전체가 커밋되므로 반드시 먼저 작성

프로젝트 구조:

```text
mlops-week05/
├── .github/
│   └── workflows/
│       └── mlops-ci.yml
├── src/
│   ├── __init__.py
│   ├── functions.py
│   └── pipeline.py
├── tests/
│   └── test_functions.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── run_pipeline.py
```

---

# 2차시 실습 — GitHub Actions 워크플로우 작성

## 실습 목표

- GitHub Actions Workflow의 기본 구조를 이해함
- `push` 이벤트로 CI를 자동 실행함
- GitHub-hosted Runner에서 Python 테스트를 자동 수행함
- PASS/FAIL을 통해 CI의 Gate 역할을 확인함

## 2. ML 함수 작성

`src/functions.py`

```python
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def preprocess_data(test_size: float = 0.2, random_state: int = 42):
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Tuple[RandomForestClassifier, float]:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy
```

## 3. 테스트 코드 작성

`tests/test_functions.py`

```python
from src.functions import preprocess_data, train_model

def test_preprocess_data():
    X_train, X_test, y_train, y_test = preprocess_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == 150
    assert X_train.shape[1] == 4

def test_train_model():
    X_train, X_test, y_train, y_test = preprocess_data()

    model, accuracy = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=50,
        max_depth=3,
    )

    assert model is not None
    assert 0.0 <= accuracy <= 1.0
    assert accuracy >= 0.8
```

로컬 테스트:

```bash
pytest -v
```

## 4. requirements.txt

```text
zenml[local]==0.96.4
scikit-learn
pandas
pytest
```

## 5. GitHub Actions Workflow

`.github/workflows/mlops-ci.yml`

```yaml
name: Week05 MLOps CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest -v
```

Workflow 흐름:

```text
Push / Pull Request
       ↓
Checkout
       ↓
Setup Python
       ↓
Install dependencies
       ↓
pytest
       ↓
PASS / FAIL
```

## 6. GitHub 저장소 연결 및 Push

- GitHub에서 빈 Repository 생성 (README, .gitignore 추가 없이 생성 → 첫 Push 충돌 방지)
- HTTPS Push 시 비밀번호 대신 Personal Access Token 입력 필요 (또는 SSH 키 등록 후 SSH URL 사용)

```bash
git remote add origin <자신의-GitHub-Repository-URL>

git add .
git commit -m "week05: add CI workflow"
git push -u origin main
```

GitHub Repository의 `Actions` 메뉴에서 Workflow 결과를 확인함

## 7. 실패하는 CI 체험

테스트 조건을 일부러 잘못 바꿈

```python
assert accuracy >= 1.1
```

Commit / Push:

```bash
git add .
git commit -m "week05: intentionally fail CI"
git push
```

결과:

```text
Run tests
   ↓
FAILED
```

다시 원래대로 복원함

```python
assert accuracy >= 0.8
```

---

# 3차시 실습 — ZenML 파이프라인 구동 및 GitHub Actions 연계

## 실습 목표

- Python 함수를 ZenML Step으로 정의함
- 여러 Step을 Pipeline으로 연결함
- 로컬에서 ZenML Pipeline을 실행함
- GitHub Actions가 테스트 후 ZenML Pipeline을 자동 실행하도록 구성함

## 8. ZenML 초기화

```bash
cd ~/mlops-week05
source .venv/bin/activate

zenml init
zenml status
zenml stack list
```

이번 실습은 Local Stack을 사용함

## 9. ZenML Pipeline 작성

`src/pipeline.py`

```python
from typing import Annotated, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import pipeline, step

from src.functions import preprocess_data, train_model

@step
def preprocess_step(
    test_size: float = 0.2,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    X_train, X_test, y_train, y_test = preprocess_data(
        test_size=test_size,
        random_state=42,
    )

    # return 문에 4개 값을 직접 나열해야 다중 출력으로 인식됨
    return X_train, X_test, y_train, y_test

@step
def train_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Tuple[
    Annotated[RandomForestClassifier, "model"],
    Annotated[float, "accuracy"],
]:

    model, accuracy = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    print(f"Accuracy: {accuracy:.4f}")
    return model, accuracy

@pipeline
def training_pipeline(
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
):

    X_train, X_test, y_train, y_test = preprocess_step(
        test_size=test_size
    )

    train_step(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
```

> ZenML은 `return` 문에 여러 값을 직접 나열한 경우에만 Tuple 출력을 여러 Artifact로 분리함  
> `return preprocess_data(...)`처럼 함수 호출 결과를 그대로 반환하면 단일 Artifact로 처리되어 Pipeline에서 `Unable to unpack step artifact` 오류 발생

## 10. Pipeline 실행 파일

`run_pipeline.py`

```python
from src.pipeline import training_pipeline

if __name__ == "__main__":
    training_pipeline(
        test_size=0.2,
        n_estimators=100,
        max_depth=5,
    )
```

실행:

```bash
python run_pipeline.py
```

실행 흐름:

```text
preprocess_step
      ↓
train_step
```

## 11. GitHub Actions와 ZenML 연결

Workflow를 다음처럼 수정함

```yaml
name: Week05 MLOps CI Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test-and-pipeline:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest -v

      - name: Initialize ZenML
        run: |
          zenml init

      - name: Run ZenML pipeline
        run: |
          python run_pipeline.py
```

Commit / Push:

```bash
git add .
git commit -m "week05: connect GitHub Actions and ZenML"
git push
```

GitHub Actions에서 다음 순서를 확인함

```text
Checkout
   ↓
Setup Python
   ↓
Install
   ↓
Test
   ↓
ZenML Init
   ↓
ZenML Pipeline
```

## 12. Gate 확인

테스트가 실패하면:

```text
pytest
  ↓
FAIL
  ↓
STOP
```

따라서 이후의 ZenML Pipeline은 실행되지 않음

정상일 경우:

```text
pytest
  ↓
PASS
  ↓
ZenML Pipeline
```

## 13. 역할 정리

```text
GitHub Actions
= WHEN
= 언제 자동화를 시작할 것인가?

ZenML
= WHAT + ORDER
= 무엇을 어떤 순서로 실행할 것인가?
```

최종 구조:

```text
Git Push
   ↓
GitHub Actions
   ↓
Automated Test
   ↓
PASS
   ↓
ZenML
   ↓
Preprocess
   ↓
Train
```

---

# 14. 실습 정리

2차시:

```text
Code Change
   ↓
Git Push
   ↓
GitHub Actions
   ↓
Test
```

3차시:

```text
Git Push
   ↓
GitHub Actions
   ↓
Test
   ↓
ZenML
   ↓
Preprocess
   ↓
Train
```

이번 주에는 코드 변경을 Trigger로 Pipeline을 자동 실행하는 기본 구조를 구성함

6주차에서는:

```text
Data Drift
   ↓
Retraining Trigger
```

를 연결하여 Continuous Training 개념으로 확장함

---

# 15. 선택 과제

1. `workflow_dispatch`를 추가하여 GitHub UI에서 수동 실행함
2. `max_depth` 값을 변경한 뒤 Push하여 Pipeline 재실행을 확인함
3. `evaluate_step`을 추가해 `Preprocess → Train → Evaluate` 구조로 확장함
4. Accuracy가 기준 미만이면 Pipeline을 실패시키는 Gate를 추가함
