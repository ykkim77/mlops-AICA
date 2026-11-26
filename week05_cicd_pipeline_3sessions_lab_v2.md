# 5단원 실습 — CI/CD 파이프라인 설계

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 실습 환경: WSL2 Ubuntu 22.04 / Python 3.10+ / Git / GitHub  
> 클라우드 계정: 필요 없음  
> 실습 기준 ZenML: 0.96.4

---

# 0. 실습 개요

5단원 실습: 총 2개 차시(2·3차시)로 구성

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

# 2차시 실습 — GitHub Actions를 이용한 CI 자동화

## 실습 목표

- GitHub Actions Workflow의 기본 구조를 이해함
- `push` 이벤트를 이용하여 CI를 자동 실행함
- GitHub-hosted Runner에서 Python 테스트를 자동 수행함
- Actions 실행 로그를 직접 확인함
- 의도적으로 Test를 실패시켜 CI Gate의 동작을 확인함
- 오류 수정 후 다시 Push하여 PASS로 복구되는 과정을 확인함

이번 차시의 전체 흐름:

```text
Code 작성
   ↓
Local pytest
   ↓
Git Push
   ↓
GitHub Actions
   ↓
Checkout
   ↓
Python 환경 구성
   ↓
Dependencies 설치
   ↓
pytest
   ↓
PASS / FAIL
```

핵심 질문:

> “개발자가 직접 실행하던 테스트를 GitHub가 자동으로 대신 실행하게 만들 수 있는가?”

---

## 2. ML 함수 작성

`src/functions.py`

```python
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def preprocess_data(
    test_size: float = 0.2,
    random_state: int = 42,
):
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

---

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

---

## 4. 로컬에서 pytest 실행

```bash
cd ~/mlops-week05
source .venv/bin/activate

pytest -v
```

정상 결과 예:

```text
tests/test_functions.py::test_preprocess_data PASSED
tests/test_functions.py::test_train_model PASSED
```

핵심:

```text
Local Test
→ GitHub에 Push하기 전에 개발자 환경에서 먼저 검증
```

---

## 5. requirements.txt 작성

```bash
cat > requirements.txt <<'REQ'
zenml[local]==0.96.4
scikit-learn
pandas
pytest
REQ
```

역할:

```text
requirements.txt
→ GitHub Runner가 설치할 Python Package 목록
```

---

## 6. GitHub Actions Workflow 작성

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

---

## 7. Workflow 구조 이해

```text
push / pull_request
        ↓
Workflow Trigger
        ↓
GitHub-hosted Runner
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

---

## 8. GitHub Repository 연결 및 첫 Push

```bash
git remote add origin <자신의-GitHub-Repository-URL>

git add .

git commit -m "week05: add CI workflow"

git push -u origin main
```

---

## 9. GitHub Actions 실행 로그 확인

GitHub Repository에서:

```text
Actions
→ Week05 MLOps CI
→ 실행된 Workflow
→ test Job
```

다음 Step을 하나씩 펼쳐서 확인함.

```text
Checkout repository
Set up Python
Install dependencies
Run tests
```

`Run tests` 로그에서 다음을 확인:

```text
test_preprocess_data PASSED
test_train_model PASSED
```

핵심:

> 로컬에서 직접 실행했던 `pytest`를 GitHub Runner가 자동으로 실행함

---

## 10. 의도적으로 코드 테스트 실패시키기

`tests/test_functions.py`에서 다음 조건을 변경:

```python
assert len(X_train) == 999
```

Commit / Push:

```bash
git add .

git commit -m "week05: intentionally fail code test"

git push
```

GitHub Actions:

```text
Run tests
   ↓
FAILED
```

로그에서 Assertion Error를 확인함.

---

## 11. CI Gate 이해

```text
Code
 ↓
pytest
 ↓
CI Gate
 ├─ PASS → 다음 단계 진행 가능
 └─ FAIL → Workflow 실패
```

즉 pytest는 코드와 함수가 정상적으로 동작하는지 확인하는 CI Gate 역할을 함.

---

## 12. 오류 수정 후 다시 PASS

원래 코드로 복구:

```python
assert len(X_train) > 0
```

Push:

```bash
git add .

git commit -m "week05: fix failed test"

git push
```

GitHub Actions에서 다시:

```text
Run tests
   ↓
PASSED
```

를 확인함.

---

## 13. 2차시 핵심 정리

```text
Code Change
   ↓
Local pytest
   ↓
Git Push
   ↓
GitHub Actions
   ↓
GitHub-hosted Runner
   ↓
Automated Test
   ↓
PASS / FAIL
```

핵심 개념:

```text
GitHub Actions
= Repository Event 기반 자동화

pytest
= 코드와 함수 동작 검증

CI Gate
= Test 실패 시 자동화 흐름 중단
```

---

# 3차시 실습 — ZenML Pipeline과 GitHub Actions 연계

## 실습 목표

- Python 함수를 ZenML Step으로 정의함
- 여러 Step을 Pipeline으로 연결함
- `Preprocess → Train → Evaluate` 구조를 구성함
- Accuracy 기준을 이용한 Model Quality Gate를 구현함
- Pipeline Parameter를 변경하여 반복 실행함
- GitHub Actions가 테스트 성공 후 ZenML Pipeline을 자동 실행하도록 구성함
- CI Gate와 Model Quality Gate의 차이를 이해함

전체 흐름:

```text
Git Push
   ↓
GitHub Actions
   ↓
pytest
   ↓
PASS
   ↓
ZenML Pipeline
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Model Quality Gate
   ↓
PASS / FAIL
```

핵심 질문:

> “코드 테스트 이후의 ML 학습 과정도 Pipeline으로 정의하고 자동 실행할 수 있는가?”

---

## 14. ZenML 초기화

```bash
cd ~/mlops-week05
source .venv/bin/activate

zenml init
zenml status
zenml stack list
```

이번 실습은 Local Stack을 사용함.

---

## 15. ZenML Step 개념 확인

```text
Preprocess
Train
Evaluate
```

각 작업을 독립된 Step으로 정의하고 연결함.

```text
Preprocess Step
      ↓
Train Step
      ↓
Evaluate Step
```

---

## 16. Preprocess / Train Step 작성

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

    print("Train rows:", len(X_train))
    print("Test rows :", len(X_test))

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

    print("n_estimators:", n_estimators)
    print("max_depth    :", max_depth)
    print(f"Accuracy     : {accuracy:.4f}")

    return model, accuracy
```

> ZenML은 `return` 문에 여러 값을 직접 나열한 경우 Tuple 출력을 여러 Artifact로 분리함

---

## 17. Evaluate Step 추가

`src/pipeline.py`에 추가:

```python
@step
def evaluate_step(
    accuracy: float,
    min_accuracy: float = 0.8,
) -> bool:

    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Min Accuracy : {min_accuracy:.4f}")

    if accuracy < min_accuracy:
        raise RuntimeError(
            "Model quality gate failed: "
            f"{accuracy:.4f} < {min_accuracy:.4f}"
        )

    print("Model quality gate: PASS")

    return True
```

역할:

```text
Accuracy
   ↓
Evaluate Step
   ↓
min_accuracy와 비교
   ↓
PASS / FAIL
```

---

## 18. Pipeline 작성

`src/pipeline.py`에 추가:

```python
@pipeline
def training_pipeline(
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_accuracy: float = 0.8,
):

    X_train, X_test, y_train, y_test = preprocess_step(
        test_size=test_size,
    )

    model, accuracy = train_step(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    evaluate_step(
        accuracy=accuracy,
        min_accuracy=min_accuracy,
    )
```

전체 구조:

```text
training_pipeline
        ↓
preprocess_step
        ↓
train_step
        ↓
evaluate_step
```

---

## 19. Pipeline 실행 파일 작성

`run_pipeline.py`

```python
from src.pipeline import training_pipeline


if __name__ == "__main__":

    training_pipeline(
        test_size=0.2,
        n_estimators=100,
        max_depth=5,
        min_accuracy=0.8,
    )
```

실행:

```bash
python run_pipeline.py
```

---

## 20. 정상 Pipeline 실행 확인

```text
preprocess_step
      ↓
train_step
      ↓
Accuracy 출력
      ↓
evaluate_step
      ↓
Model quality gate: PASS
```

---

## 21. Model Quality Gate 실패 체험

`run_pipeline.py`를 수정:

```python
training_pipeline(
    test_size=0.2,
    n_estimators=100,
    max_depth=5,
    min_accuracy=1.1,
)
```

실행:

```bash
python run_pipeline.py
```

결과:

```text
Evaluate
   ↓
Accuracy < 1.1
   ↓
RuntimeError
   ↓
Pipeline FAILED
```

---

## 22. Quality Gate 정상 복구

다시:

```python
min_accuracy=0.8
```

로 복구하고:

```bash
python run_pipeline.py
```

정상 PASS를 확인함.

---

## 23. Pipeline Parameter 변경 실험

### 실행 1

```python
training_pipeline(
    test_size=0.2,
    n_estimators=100,
    max_depth=5,
    min_accuracy=0.8,
)
```

Accuracy를 기록함.

### 실행 2

```python
training_pipeline(
    test_size=0.2,
    n_estimators=10,
    max_depth=1,
    min_accuracy=0.8,
)
```

다시 실행:

```bash
python run_pipeline.py
```

확인:

```text
Pipeline 구조
→ 동일

Parameter
→ 변경

Model
→ 다시 학습

Accuracy
→ 달라질 수 있음
```

핵심:

> Pipeline 구조를 유지한 채 Parameter만 바꾸어 반복 실행할 수 있음

---

## 24. CI Gate와 Model Quality Gate 비교

| Gate | 확인 대상 | 도구 |
|---|---|---|
| CI Gate | 코드와 함수가 정상 동작하는가 | pytest |
| Model Quality Gate | 모델 성능이 기준을 만족하는가 | ZenML Evaluate Step |

흐름:

```text
pytest
  ↓
CI Gate
  ↓ PASS
ZenML
  ↓
Evaluate
  ↓
Model Quality Gate
```

---

## 25. GitHub Actions와 ZenML 연결

`.github/workflows/mlops-ci.yml`

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

---

## 26. GitHub Actions와 ZenML 역할 구분

```text
GitHub Actions
= WHEN
= 언제 자동화를 시작할 것인가?
```

```text
ZenML
= WHAT + ORDER
= 무엇을 어떤 순서로 실행할 것인가?
```

---

## 27. GitHub에 Push

```bash
pytest -v
python run_pipeline.py
```

정상 동작 확인 후:

```bash
git add .

git commit -m "week05: add ZenML evaluate pipeline"

git push
```

---

## 28. GitHub Actions에서 전체 Pipeline 확인

GitHub:

```text
Actions
→ Week05 MLOps CI Pipeline
→ 실행된 Workflow
→ test-and-pipeline
```

확인:

```text
Checkout repository
      ↓
Set up Python
      ↓
Install dependencies
      ↓
Run tests
      ↓
Initialize ZenML
      ↓
Run ZenML pipeline
```

---

## 29. CI Gate 실패 시 ZenML이 실행되지 않는지 확인

테스트 하나를 다시 고의로 실패시킴:

```python
assert len(X_train) == 999
```

Push:

```bash
git add .

git commit -m "week05: test CI gate stop"

git push
```

GitHub Actions:

```text
Run tests
   ↓
FAIL
   ↓
Initialize ZenML
→ 실행되지 않음

Run ZenML pipeline
→ 실행되지 않음
```

---

## 30. 정상 상태로 최종 복구

테스트 복구:

```python
assert len(X_train) > 0
```

Push:

```bash
git add .

git commit -m "week05: restore passing CI"

git push
```

최종 흐름:

```text
Git Push
   ↓
GitHub Actions
   ↓
pytest
   ↓
PASS
   ↓
ZenML
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
PASS
```

---

## 31. 3차시 핵심 정리

```text
GitHub Actions
= 자동화 Trigger와 실행환경 제공

pytest
= 코드 수준 CI Gate

ZenML
= ML Pipeline의 Step과 실행 순서 관리

Evaluate Step
= Model Quality Gate
```

최종 구조:

```text
Code Change
   ↓
Git Push
   ↓
GitHub Actions
   ↓
pytest
   ↓
CI Gate
   ↓ PASS
ZenML Pipeline
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Model Quality Gate
```

핵심:

> GitHub Actions는 “언제 실행할 것인가”를 자동화하고, ZenML은 “무엇을 어떤 순서로 실행할 것인가”를 관리함

> 코드 검증과 모델 성능 검증은 서로 다른 Gate로 분리하여 구성할 수 있음

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
pytest
   ↓
ZenML
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Model Quality Gate
```

이번 주에는 코드 변경을 Trigger로 Pipeline을 자동 실행하는 기본 구조를 구성함

6단원에서는:

```text
Data Drift
   ↓
Retraining Trigger
```

를 연결하여 Continuous Training 개념으로 확장함

---

# 15. 선택 과제

1. `workflow_dispatch`를 추가하여 GitHub UI에서 수동 실행함
2. `max_depth`, `n_estimators` 값을 변경한 뒤 Push하여 Pipeline 재실행을 확인함
3. `min_accuracy` 값을 여러 수준으로 변경하여 Quality Gate 동작을 비교함
4. 새로운 `data_check_step`을 추가하여 `Preprocess → Data Check → Train → Evaluate` 구조로 확장함
