# 7단원 3차시 실습 — Mock 설계에서 실제 End-to-End MLOps Pipeline으로 확장

## 1. 실습 목표

2차시에서는 다음을 수행함.

```text
Pipeline 설계
   ↓
pipeline_design.yaml
   ↓
ZenML Skeleton
   ↓
Mock Artifact
   ↓
Pipeline 구조 검증
```

이번 3차시에서는 2차시에서 만든 구조를 그대로 이어서 **Mock 부분을 실제 구현으로 교체하고 GitHub Actions까지 연결**함.

최종 흐름:

```text
2차시 설계
   ↓
실제 Preprocess / Train / Evaluate 구현
   ↓
pytest
   ↓
Git Push
   ↓
GitHub Actions
   ↓
Test
   ↓
ZenML Pipeline
   ↓
Quality Gate
   ↓
MLflow Tracking
   ↓
Model Registry
```

핵심:

> 2차시에서는 Pipeline 구조를 검증했고, 3차시에서는 같은 구조에 실제 데이터, 모델, Tracking, CI를 연결함

---

## 2. 2차시 산출물 확인

프로젝트 폴더:

```bash
cd ~/mlops-week07
source .venv/bin/activate
```

파일 확인:

```bash
ls
ls src
```

2차시 산출물:

```text
pipeline_design.yaml
run_design.py
run_design_error.py
src/pipeline_design.py
```

설계 구조:

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

Mock 값:

```text
mock_X_train
mock_X_test
mock_y_train
mock_y_test
mock_model
accuracy = 0.90
```

이번 차시에서는 이 Mock 값을 실제 객체로 교체함.

```text
mock_X_train
→ Pandas DataFrame

mock_model
→ RandomForestClassifier

mock accuracy
→ accuracy_score()
```

---

## 3. 실제 데이터 생성

`src/make_data.py` 생성:

```bash
cat > src/make_data.py <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n = 300

    study_hours = rng.normal(5, 2, n).clip(0, 12)
    attendance = rng.normal(85, 10, n).clip(40, 100)
    assignment_score = rng.normal(78, 12, n).clip(0, 100)
    midterm_score = rng.normal(75, 15, n).clip(0, 100)

    risk_score = (
        (study_hours < 3).astype(int)
        + (attendance < 75).astype(int)
        + (assignment_score < 65).astype(int)
        + (midterm_score < 60).astype(int)
    )

    risk_flag = (risk_score >= 2).astype(int)

    df = pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "assignment_score": assignment_score.round(2),
        "midterm_score": midterm_score.round(2),
        "risk_flag": risk_flag,
    })

    path = Path("data/students.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(df.head())
    print()
    print("shape:", df.shape)
    print("risk distribution:")
    print(df["risk_flag"].value_counts())


if __name__ == "__main__":
    main()
PY
```

실행:

```bash
python src/make_data.py
```

2차시와의 연결:

```text
2차시
students.csv라는 입력을 설계

3차시
실제 students.csv 생성
```

---

## 4. 핵심 ML 로직을 함수로 분리

GitHub Actions에서 Test하려면 Pipeline 내부 코드와 Test 대상이 동일해야 함.

따라서 핵심 ML 로직을 `src/functions.py`에 구현하고, ZenML Step은 이 함수를 호출하는 얇은 Wrapper로 구성함.

```text
                ┌─ pytest
src/functions.py
                ┤
                └─ ZenML Steps
```

`src/functions.py` 생성:

```bash
cat > src/functions.py <<'PY'
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "study_hours",
    "attendance",
    "assignment_score",
    "midterm_score",
]

TARGET = "risk_flag"


def preprocess_data(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )


def train_model(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=5,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
PY
```

핵심:

> Test와 Pipeline이 서로 다른 코드를 검사하지 않고 같은 `functions.py`를 공유함

---

## 5. 실제 ZenML Pipeline 작성

2차시의 `src/pipeline_design.py`에서 설계한 구조를 그대로 유지함.

```text
preprocess_step
train_step
evaluate_step
```

이번에는 내부 Mock 값을 실제 함수 호출로 교체함.

`src/pipeline.py` 생성:

```bash
cat > src/pipeline.py <<'PY'
from typing import Annotated, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from zenml import pipeline, step

from src.functions import (
    preprocess_data,
    train_model,
    evaluate_model,
)


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "week07-capstone"
MODEL_NAME = "student-risk-model"


@step
def preprocess_step(
    csv_path: str,
    test_size: float = 0.2,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    X_train, X_test, y_train, y_test = preprocess_data(
        csv_path=csv_path,
        test_size=test_size,
    )

    print("Preprocess completed")

    return X_train, X_test, y_train, y_test


@step
def train_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Annotated[RandomForestClassifier, "model"]:
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    print("Training completed")
    return model


@step(enable_cache=False)
def evaluate_and_register_step(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_accuracy: float = 0.85,
) -> Annotated[float, "accuracy"]:
    accuracy = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_accuracy", min_accuracy)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Accuracy: {accuracy:.4f}")

        # Quality Gate
        if accuracy >= min_accuracy:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=MODEL_NAME,
                input_example=X_test.iloc[:3],
            )

            print("Quality Gate: PASS")
            print("Model registered")
            print("Model URI:", model_info.model_uri)

        else:
            print("Quality Gate: FAIL")
            print("Model not registered")

    return accuracy


@pipeline
def mlops_pipeline(
    csv_path: str = "data/students.csv",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_accuracy: float = 0.85,
):
    X_train, X_test, y_train, y_test = preprocess_step(
        csv_path=csv_path,
        test_size=test_size,
    )

    model = train_step(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    evaluate_and_register_step(
        model=model,
        X_test=X_test,
        y_test=y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_accuracy=min_accuracy,
    )
PY
```

2차시와 비교:

```text
2차시 Mock
mock_X_train
mock_model
accuracy = 0.90

        ↓

3차시 실제 구현
DataFrame
RandomForestClassifier
accuracy_score()
```

Pipeline의 구조는 동일하고 내부 구현만 실제 코드로 변경됨.

---

## 6. 실행 파일 작성

`run_pipeline.py` 생성:

```bash
cat > run_pipeline.py <<'PY'
from src.pipeline import mlops_pipeline


if __name__ == "__main__":
    mlops_pipeline(
        csv_path="data/students.csv",
        test_size=0.2,
        n_estimators=100,
        max_depth=5,
        min_accuracy=0.85,
    )
PY
```

---

## 7. 로컬 MLflow Server 실행

별도 터미널:

```bash
cd ~/mlops-week07
source .venv/bin/activate
```

MLflow Server:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000
```

브라우저:

```text
http://127.0.0.1:5000
```

---

## 8. 로컬에서 실제 Pipeline 실행

다른 터미널:

```bash
cd ~/mlops-week07
source .venv/bin/activate
```

ZenML 초기화가 안 되어 있다면:

```bash
zenml init
```

Pipeline 실행:

```bash
python run_pipeline.py
```

확인:

```text
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Accuracy
   ↓
Quality Gate
```

성능 기준:

```text
accuracy >= 0.85
```

이면:

```text
Quality Gate: PASS
Model registered
```

기준 미달이면:

```text
Quality Gate: FAIL
Model not registered
```

---

## 9. MLflow UI 확인

브라우저:

```text
http://127.0.0.1:5000
```

Experiment:

```text
week07-capstone
```

확인 항목:

```text
Parameters
- n_estimators
- max_depth
- min_accuracy

Metric
- accuracy

Model
- student-risk-model
```

2차시 설계와 연결:

```text
2차시 YAML

parameters:
  n_estimators
  max_depth

metrics:
  accuracy

registry condition:
  accuracy >= 0.85

        ↓

3차시 MLflow

log_param()
log_metric()
Model Registry
```

---

## 10. pytest 작성

이제 GitHub Actions에서 코드가 정상인지 확인할 Test를 작성함.

```bash
mkdir -p tests
```

`tests/test_functions.py`:

```bash
cat > tests/test_functions.py <<'PY'
from src.functions import (
    preprocess_data,
    train_model,
    evaluate_model,
)


def test_preprocess():
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/students.csv"
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_and_evaluate():
    X_train, X_test, y_train, y_test = preprocess_data(
        "data/students.csv"
    )

    model = train_model(
        X_train,
        y_train,
        n_estimators=10,
        max_depth=3,
    )

    accuracy = evaluate_model(
        model,
        X_test,
        y_test,
    )

    assert model is not None
    assert 0.0 <= accuracy <= 1.0
PY
```

실행:

```bash
pytest -q
```

중요한 구분:

```text
pytest
→ 코드가 정상적으로 동작하는가?

Quality Gate
→ 모델 성능이 기준 이상인가?
```

따라서 Test 코드에는 다음을 넣지 않음.

```python
assert accuracy >= 0.85
```

모델 성능 기준은 ZenML Pipeline의 Quality Gate에서 판단함.

---

## 11. requirements.txt 작성

GitHub Actions Runner는 새로운 환경이므로 필요한 패키지를 다시 설치해야 함.

```bash
cat > requirements.txt <<'EOF2'
zenml[local]==0.96.4
mlflow==3.18.0
scikit-learn
pandas
numpy
pytest
EOF2
```

---

## 12. GitHub Actions Workflow 작성

```bash
mkdir -p .github/workflows
```

`.github/workflows/mlops.yml` 생성:

```bash
cat > .github/workflows/mlops.yml <<'YAML'
name: MLOps CI Pipeline

on:
  push:
    branches:
      - main

jobs:
  mlops-ci:
    runs-on: ubuntu-24.04

    steps:
      - name: Checkout
        uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Generate sample data
        run: |
          python src/make_data.py

      - name: Test
        run: |
          pytest -q

      - name: Initialize ZenML
        run: |
          zenml init

      - name: Start MLflow server
        run: |
          nohup mlflow server \
            --host 127.0.0.1 \
            --port 5000 \
            > mlflow.log 2>&1 &

          sleep 5

      - name: Run ZenML Pipeline
        run: |
          python run_pipeline.py
YAML
```

GitHub Actions의 역할:

```text
Git Push
   ↓
새로운 Runner 생성
   ↓
Python 환경 구성
   ↓
Dependency 설치
   ↓
pytest
   ↓
ZenML Pipeline 자동 실행
```

---

## 13. Git Push

Git 상태 확인:

```bash
git status
```

추가:

```bash
git add .
```

Commit:

```bash
git commit -m "Add end-to-end MLOps pipeline"
```

Push:

```bash
git push origin main
```

GitHub Actions가 자동 실행됨.

---

## 14. GitHub Actions 실행 확인

GitHub Repository에서:

```text
Actions
```

Workflow:

```text
MLOps CI Pipeline
```

실행 순서:

```text
Checkout
   ↓
Set up Python
   ↓
Install dependencies
   ↓
Generate sample data
   ↓
pytest
   ↓
ZenML init
   ↓
MLflow Server
   ↓
ZenML Pipeline
   ↓
Quality Gate
   ↓
MLflow Tracking / Registry
```

주의:

> GitHub Actions에서 실행한 MLflow Server는 GitHub-hosted Runner 내부의 임시 Server임.

따라서 Workflow가 끝난 뒤 학생 PC의 `http://127.0.0.1:5000`에서 GitHub Runner의 MLflow 기록이 보이는 것은 아님.

이번 실습에서 GitHub Actions의 핵심 목적은:

```text
Push
→ Test
→ Pipeline 자동 실행
```

이라는 CI 자동화 구조를 확인하는 것임.

MLflow UI에서 Run을 직접 비교하는 실습은 앞에서 수행한 로컬 Pipeline 결과를 사용함.

---

## 15. 전체 구조 확인

```text
Git Push
   ↓
GitHub Actions
   ↓
pytest
   ↓
ZenML Pipeline
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Quality Gate
   ↓
MLflow Tracking
   ↓
Model Registry
```

각 도구의 역할:

```text
Git
→ 코드 변경 관리

GitHub Actions
→ CI 자동 실행

pytest
→ 코드 검증

ZenML
→ ML Pipeline 단계 연결 및 실행

MLflow Tracking
→ Parameter / Metric / Run 기록

Quality Gate
→ 모델 성능 기준 판단

Model Registry
→ 통과한 모델 버전 관리
```

---

## 16. 2차시와 3차시 연결 정리

2차시:

```text
Pipeline 설계
↓
YAML
↓
Skeleton
↓
Mock Artifact
↓
구조 검증
```

3차시:

```text
Mock 제거
↓
실제 데이터
↓
실제 모델
↓
pytest
↓
Git Push
↓
GitHub Actions
↓
ZenML Pipeline
↓
MLflow
↓
Model Registry
```

즉 두 차시는 별개의 실습이 아니라:

```text
설계
   ↓
Mock 검증
   ↓
실제 구현
   ↓
자동화
```

로 이어지는 하나의 End-to-End 실습임.

---

## 17. 7단원 실습 최종 정리

이번 7단원에서는 새로운 도구를 배우는 것보다 지금까지 배운 요소를 하나의 흐름으로 연결함.

```text
2단원
Data Management

3단원
MLflow

4단원
Serving

5단원
GitHub Actions / ZenML

6단원
Monitoring

7단원
Integration
```

이번 실습에서 완성한 핵심 구조:

```text
Code Change
   ↓
Git Push
   ↓
CI Test
   ↓
ML Pipeline
   ↓
Model Evaluation
   ↓
Quality Gate
   ↓
Experiment Tracking
   ↓
Model Registry
```

핵심:

> MLOps는 하나의 도구가 아니라 데이터, 코드, Pipeline, Test, 모델 평가, Tracking, Registry를 하나의 운영 흐름으로 연결하는 방법론임
