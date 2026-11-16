# 7주차 실습 — MLOps 종합 실습 및 사례 연구

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 주제: 사례 기반 MLOps 파이프라인 설계 → ZenML + MLflow 기반 구현 → 결과 발표  
> 실습 환경: WSL2 Ubuntu 22.04 기준 / Python 3.10 이상 / Git / MLflow / ZenML  
> 클라우드 계정: 필요 없음

---

# 0. 이번 주 목표

7주차는 새로운 도구를 배우는 주차가 아니라,  
2~6주차에 배운 요소를 하나의 MLOps 흐름으로 연결하는 종합 실습입니다.

전체 흐름:

```text
문제 정의
   ↓
데이터 준비
   ↓
전처리
   ↓
모델 학습
   ↓
평가
   ↓
MLflow Tracking
   ↓
Model Registry
   ↓
Champion 선택
   ↓
로컬 Serving
   ↓
Monitoring
   ↓
Retraining 판단
```

이번 실습에서는 전체 흐름 가운데 다음 핵심 구간을 직접 구현합니다.

```text
ZenML Pipeline
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
MLflow Tracking
   ↓
Model Registry
```

그리고 마지막에는 각 팀 또는 개인이  
자신의 파이프라인 구조와 실행 결과를 설명합니다.

---

# 1. 실습 시나리오

이번 실습에서는 **학습위험 학생 예측**을 예시 사례로 사용합니다.

목표:

```text
학습활동 데이터
   ↓
전처리
   ↓
학습위험 여부 예측
```

예시 Feature:

```text
study_hours
attendance
assignment_score
midterm_score
```

Target:

```text
risk_flag
```

예:

```text
risk_flag = 0
정상

risk_flag = 1
학습위험
```

> 본 실습의 목적은 실제 교육적 판단 모델을 만드는 것이 아니라  
> MLOps 파이프라인의 구조를 학습하는 것입니다.

---

# 2. 프로젝트 생성

```bash
mkdir -p ~/mlops-week07
cd ~/mlops-week07

git init -b main

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install \
  "zenml==0.96.4" \
  mlflow \
  scikit-learn \
  pandas \
  numpy
```

폴더 생성:

```bash
mkdir -p data src
touch src/__init__.py
```

구조:

```text
mlops-week07/
├── data/
├── src/
│   ├── __init__.py
│   ├── make_data.py
│   ├── pipeline.py
│   └── register_model.py
└── run_pipeline.py
```

---

# 2차시 실습 — 사례 기반 MLOps 파이프라인 설계

# 3. 실습 목표

- 주어진 문제를 MLOps 단계로 분해함
- 각 단계의 입력과 출력을 정의함
- 어떤 도구를 어느 단계에 사용할지 결정함
- ZenML Step과 Pipeline 구조를 설계함
- MLflow Tracking 및 Registry 연결 위치를 설계함

---

# 4. Step 1 — 문제 정의

다음 질문에 답합니다.

```text
무엇을 예측할 것인가?
```

이번 사례:

```text
학생의 학습위험 여부
```

Target:

```text
risk_flag
```

---

# 5. Step 2 — 입력 데이터 정의

이번 실습의 Feature:

```text
study_hours
attendance
assignment_score
midterm_score
```

Target:

```text
risk_flag
```

구조:

```text
Raw Data
   ↓
Features + Target
```

---

# 6. Step 3 — Pipeline 단계 설계

최소 3단계로 구성합니다.

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

각 Step의 책임:

## Preprocess

```text
데이터 로딩
결측치 처리
Feature / Target 분리
Train / Test Split
```

## Train

```text
RandomForestClassifier 학습
Parameter 기록
Model 생성
```

## Evaluate

```text
Accuracy 계산
Metric 기록
등록 여부 판단
```

---

# 7. Step 4 — 도구 배치

이번 실습에서의 역할:

```text
ZenML
```

Pipeline의 단계와 순서 관리

```text
MLflow Tracking
```

Parameter / Metric / Run 기록

```text
MLflow Model Registry
```

학습된 모델 버전 관리

---

# 8. 설계안 작성

실습 전 다음 표를 작성합니다.

| 단계 | 입력 | 수행 작업 | 출력 | 도구 |
|---|---|---|---|---|
| Preprocess | Raw CSV | 결측치 처리, Split | Train/Test | ZenML |
| Train | Train Data | 모델 학습 | Model | ZenML + MLflow |
| Evaluate | Model + Test Data | Accuracy 계산 | Metric | ZenML + MLflow |
| Register | Model | Registry 등록 | Model Version | MLflow |

---

# 9. 설계 시 확인할 질문

1. 각 Step의 책임이 명확한가?
2. 하나의 Step이 너무 많은 일을 하고 있지 않은가?
3. Step 간 데이터 흐름이 명확한가?
4. 어떤 Parameter를 기록할 것인가?
5. 어떤 Metric을 기록할 것인가?
6. 모델을 언제 Registry에 등록할 것인가?
7. 어떤 조건에서 candidate로 지정할 것인가?

---

# 10. 2차시 산출물

2차시 종료 시 다음을 제출합니다.

```text
① Pipeline 구조도
② Step별 입력/출력 표
③ MLflow 기록 항목
④ 모델 등록 조건
```

예:

```text
Raw CSV
   ↓
[Preprocess]
   ↓
X_train / X_test / y_train / y_test
   ↓
[Train]
   ↓
Model
   ↓
[Evaluate]
   ↓
Accuracy
   ↓
if accuracy >= 0.85
   ↓
Model Registry
```

---

# 3차시 실습 — ZenML + MLflow 종합 파이프라인 구현

# 11. 실습 목표

- 2차시에서 설계한 Pipeline을 실제 구현함
- ZenML Step을 연결함
- MLflow에 Parameter와 Metric을 기록함
- 기준을 통과한 모델을 Registry에 등록함
- 실행 결과를 발표함

---

# 12. 실습 데이터 생성

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

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance": attendance.round(2),
            "assignment_score": assignment_score.round(2),
            "midterm_score": midterm_score.round(2),
            "risk_flag": risk_flag,
        }
    )

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

---

# 13. MLflow Server 실행

별도 터미널:

```bash
cd ~/mlops-week07
source .venv/bin/activate

mlflow server \
  --host 127.0.0.1 \
  --port 5000
```

브라우저:

```text
http://127.0.0.1:5000
```

---

# 14. ZenML 초기화

다른 터미널:

```bash
cd ~/mlops-week07
source .venv/bin/activate

zenml init
```

확인:

```bash
zenml status
```

---

# 15. Pipeline 작성

`src/pipeline.py`:

```bash
cat > src/pipeline.py <<'PY'
from typing import Annotated, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from zenml import pipeline, step


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "week07-capstone"
MODEL_NAME = "student-risk-model"


@step
def preprocess_step(
    csv_path: str = "data/students.csv",
    test_size: float = 0.2,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    df = pd.read_csv(csv_path)

    feature_cols = [
        "study_hours",
        "attendance",
        "assignment_score",
        "midterm_score",
    ]

    target_col = "risk_flag"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    print("Preprocess completed")
    print("train size:", len(X_train))
    print("test size :", len(X_test))

    return X_train, X_test, y_train, y_test


@step
def train_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Annotated[RandomForestClassifier, "model"]:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Training completed")

    return model


@step
def evaluate_and_register_step(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_accuracy: float = 0.85,
) -> Annotated[float, "accuracy"]:

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        mlflow.log_param(
            "min_accuracy",
            min_accuracy,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        if accuracy >= min_accuracy:

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                input_example=X_test.iloc[:3],
            )

            print(
                "Model registered"
            )

            print(
                "Model URI:",
                model_info.model_uri,
            )

        else:

            print(
                "Model not registered"
            )

            print(
                f"accuracy {accuracy:.4f} "
                f"< threshold {min_accuracy:.4f}"
            )

    return accuracy


@pipeline
def mlops_pipeline(
    csv_path: str = "data/students.csv",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_accuracy: float = 0.85,
):

    X_train, X_test, y_train, y_test = (
        preprocess_step(
            csv_path=csv_path,
            test_size=test_size,
        )
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

---

# 16. Pipeline 실행 파일 작성

`run_pipeline.py`:

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

# 17. Pipeline 실행

```bash
python run_pipeline.py
```

실행 순서:

```text
preprocess_step
      ↓
train_step
      ↓
evaluate_and_register_step
```

---

# 18. MLflow UI 확인

브라우저:

```text
http://127.0.0.1:5000
```

Experiment:

```text
week07-capstone
```

확인:

```text
Parameters

n_estimators
max_depth
min_accuracy
```

Metric:

```text
accuracy
```

Model:

```text
student-risk-model
```

---

# 19. 다른 Parameter로 재실행

`run_pipeline.py` 변경:

```python
mlops_pipeline(
    csv_path="data/students.csv",
    test_size=0.2,
    n_estimators=50,
    max_depth=3,
    min_accuracy=0.85,
)
```

실행:

```bash
python run_pipeline.py
```

다시 변경:

```python
mlops_pipeline(
    csv_path="data/students.csv",
    test_size=0.2,
    n_estimators=200,
    max_depth=8,
    min_accuracy=0.85,
)
```

실행:

```bash
python run_pipeline.py
```

MLflow UI에서 Run을 비교합니다.

---

# 20. Model Registry 확인

MLflow UI:

```text
Models
```

다음 모델을 확인:

```text
student-risk-model
```

실행 조건을 통과한 경우:

```text
Version 1
Version 2
...
```

이 생성됩니다.

---

# 21. candidate Alias 지정

가장 적절한 Model Version을 확인한 뒤:

```bash
python - <<'PY'
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()

MODEL_NAME = "student-risk-model"

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

latest = max(
    versions,
    key=lambda v: int(v.version),
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="candidate",
    version=latest.version,
)

print(
    "candidate version:",
    latest.version,
)
PY
```

---

# 22. 결과 발표 준비

각 팀 또는 개인은 다음 내용을 정리합니다.

## 1. 문제 정의

```text
무엇을 예측했는가?
```

## 2. Pipeline 구조

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

## 3. 기록 항목

```text
Parameter
Metric
Model
```

## 4. 실행 결과

```text
Accuracy
Run 비교
Model Version
```

## 5. 개선 방향

예:

```text
DVC 연동
GitHub Actions 자동 실행
Evidently Monitoring
Champion 자동 승격 Gate
Docker Serving
```

---

# 23. 발표 예시

```text
우리 파이프라인은
학습위험 학생 예측을 목적으로 설계했습니다.

전처리 단계에서는
Feature와 Target을 분리하고
Train/Test 데이터로 나눴습니다.

학습 단계에서는
Random Forest를 사용했습니다.

n_estimators와 max_depth를
Parameter로 관리했습니다.

평가 단계에서는
Accuracy를 MLflow에 기록했고,

Accuracy가 0.85 이상일 경우에만
Model Registry에 등록하도록 구성했습니다.

추후에는
DVC와 Evidently를 연결하여
데이터 버전과 Drift까지
관리할 수 있도록 확장할 수 있습니다.
```

---

# 24. 7주차 전체 정리

```text
Business Problem
      ↓
Data
      ↓
Preprocess
      ↓
Train
      ↓
Evaluate
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
Candidate
```

이번 실습에서 가장 중요한 것은  
특정 도구의 명령어를 외우는 것이 아니라:

```text
어떤 문제를
어떤 단계로 분해하고

각 단계에
어떤 도구를 배치할 것인가?
```

를 판단하는 것입니다.

---

# 25. 2~7주차 전체 연결

```text
2주차
Data Management
DVC

↓

3주차
Experiment Tracking
MLflow

↓

4주차
Model Serving
Docker / Nginx

↓

5주차
Automation
GitHub Actions / ZenML

↓

6주차
Monitoring
Evidently / Retraining

↓

7주차
Integration
End-to-End MLOps Design
```

---

# 26. 선택 확장 과제

1. `evaluate_step`을 별도 ZenML Step으로 분리합니다.
2. 2주차 DVC를 연결하여 데이터 버전을 함께 관리합니다.
3. GitHub Actions에서 Pipeline을 자동 실행합니다.
4. Accuracy 외에 Precision, Recall, F1-score를 기록합니다.
5. Model Registry에 `candidate`와 `champion` Alias를 적용합니다.
6. 4주차 방식으로 `candidate` 모델을 REST API로 Serving합니다.
7. Evidently를 이용하여 입력 데이터 Drift를 추가로 모니터링합니다.
8. 자신의 전공 또는 업무 사례를 하나 선정하여 동일한 MLOps 구조로 재설계합니다.
