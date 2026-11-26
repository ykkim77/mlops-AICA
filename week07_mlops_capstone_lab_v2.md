# 7단원 실습 — MLOps 종합 실습 및 사례 연구

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 주제: 사례 기반 MLOps 파이프라인 설계 → ZenML + MLflow 기반 구현 → 결과 발표  
> 실습 환경: WSL2 Ubuntu 22.04 기준 / Python 3.10 이상 / Git / MLflow / ZenML  
> 클라우드 계정: 필요 없음

---

# 0. 이번 주 목표

7단원는 새로운 도구를 배우는 단원가 아니라,  
2~6단원에 배운 요소를 하나의 MLOps 흐름으로 연결하는 종합 실습임

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

이번 실습에서는 전체 흐름 가운데 다음 핵심 구간을 직접 구현함

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
자신의 파이프라인 구조와 실행 결과를 설명함

---

# 1. 실습 시나리오

이번 실습에서는 학습위험 학생 예측을 예시 사례로 사용함

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
> MLOps 파이프라인의 구조를 학습하는 것임

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
  "zenml[local]==0.96.4" \
  "mlflow==3.18.0" \
  scikit-learn \
  pandas \
  numpy
```

> `zenml[local]`: 로컬 DB 기반 실행에 필요한 의존성 포함 (기본 `zenml`만 설치하면 `zenml init` 실패)  
> MLflow 버전은 3단원와 동일하게 고정

폴더 생성:

```bash
mkdir -p data src
touch src/__init__.py

cat > .gitignore <<'EOF'
.venv/
__pycache__/
mlflow.db
mlartifacts/
mlruns/
EOF
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

# 2차시 실습 — 사례 기반 End-to-End MLOps 파이프라인 설계

# 3. 실습 목표

- 주어진 문제를 MLOps 단계로 분해함
- 각 단계의 입력과 출력을 정의함
- 각 단계에 적절한 MLOps 도구를 배치함
- ZenML Step과 Pipeline 구조를 설계함
- MLflow Tracking 및 Model Registry 연결 위치를 설계함
- 모델 등록 기준과 운영 이후의 Monitoring / Retraining 흐름을 설계함
- 2~6단원에서 학습한 요소를 하나의 구조로 연결함

이번 차시의 전체 흐름:

```text
문제 정의
   ↓
데이터 정의
   ↓
Pipeline 단계 분해
   ↓
Step별 입력 / 출력 정의
   ↓
도구 배치
   ↓
Tracking 항목 정의
   ↓
Model Gate 정의
   ↓
Serving / Monitoring 위치 설계
   ↓
최종 MLOps 구조도 작성
```

핵심 질문:

> “하나의 ML 문제를 실제 운영 가능한 MLOps 구조로 만들려면 어떤 단계와 도구가 필요한가?”

---

# 4. Step 1 — 문제 정의

이번 사례:

```text
학생의 학습활동 데이터
        ↓
학습위험 여부 예측
```

예측 대상:

```text
risk_flag
```

의미:

```text
risk_flag = 0
→ 정상

risk_flag = 1
→ 학습위험
```

문제 유형:

```text
Binary Classification
```

즉:

```text
학생 데이터
   ↓
Random Forest
   ↓
0 또는 1
```

을 예측하는 문제임.

---

# 5. Step 2 — 입력 데이터 정의

Feature:

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
Raw CSV
   ↓
Feature Columns
   +
Target Column
```

각 항목의 역할:

| 변수 | 의미 | 역할 |
|---|---|---|
| study_hours | 학습시간 | Feature |
| attendance | 출석률 | Feature |
| assignment_score | 과제점수 | Feature |
| midterm_score | 중간고사 점수 | Feature |
| risk_flag | 학습위험 여부 | Target |

---

# 6. Step 3 — Data Quality 관점 추가

2단원에서 학습한 데이터 품질 관점을 적용함.

예:

```text
study_hours
→ 0 ~ 12

attendance
→ 0 ~ 100

assignment_score
→ 0 ~ 100

midterm_score
→ 0 ~ 100
```

설계 질문:

```text
결측치가 있으면?
범위를 벗어나면?
중복 데이터가 있으면?
```

Pipeline 앞단에 다음 단계를 둘 수 있음.

```text
Raw Data
   ↓
Data Validation
   ↓
Preprocess
```

이번 실습에서는 구현을 단순화하기 위해 Validation은 설계에만 포함하고,
실제 구현은 Preprocess부터 시작함.

---

# 7. Step 4 — Pipeline 단계 설계

기본 ML Pipeline:

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

각 Step의 책임을 명확하게 나눔.

## Preprocess

```text
CSV Load
↓
Feature / Target 분리
↓
Train / Test Split
```

입력:

```text
students.csv
```

출력:

```text
X_train
X_test
y_train
y_test
```

---

## Train

입력:

```text
X_train
y_train
```

수행:

```text
RandomForestClassifier
```

Parameter:

```text
n_estimators
max_depth
```

출력:

```text
trained model
```

---

## Evaluate

입력:

```text
model
X_test
y_test
```

수행:

```text
Prediction
Accuracy 계산
```

출력:

```text
accuracy
```

---

# 8. Step 5 — Model Quality Gate 설계

모든 모델을 Registry에 등록할 필요는 없음.

예:

```text
Accuracy
   ↓
0.85 이상인가?
   ↓
YES → Registry 등록
NO  → 등록하지 않음
```

설계:

```text
if accuracy >= 0.85
    register model
else
    stop
```

즉:

```text
Evaluate
   ↓
Model Quality Gate
   ├─ PASS → Registry
   └─ FAIL → 미등록
```

---

# 9. Step 6 — 도구 배치

이번 실습에서 각 도구의 역할을 다시 배치함.

```text
Git
→ Code Version

DVC
→ Data Version

ZenML
→ Pipeline / Step / 실행 순서

MLflow Tracking
→ Parameter / Metric / Run 기록

MLflow Registry
→ Model Version 관리

Docker
→ Serving 환경 패키징

Evidently
→ Data Drift Monitoring

GitHub Actions
→ Event 기반 자동화
```

이번 7단원 구현 범위:

```text
ZenML
+
MLflow Tracking
+
MLflow Registry
```

나머지는 전체 구조에 위치만 표시함.

---

# 10. Step 7 — Tracking 항목 설계

MLflow에 무엇을 기록할 것인지 미리 정의함.

## Parameter

```text
n_estimators
max_depth
min_accuracy
```

## Metric

```text
accuracy
```

## Model

```text
RandomForestClassifier
```

## Run

```text
한 번의 Pipeline 실행
```

구조:

```text
Run 1
├─ n_estimators = 100
├─ max_depth = 5
└─ accuracy = ...

Run 2
├─ n_estimators = 50
├─ max_depth = 3
└─ accuracy = ...

Run 3
├─ n_estimators = 200
├─ max_depth = 8
└─ accuracy = ...
```

---

# 11. Step 8 — Registry 정책 설계

Registered Model:

```text
student-risk-model
```

등록 기준:

```text
accuracy >= 0.85
```

등록 후:

```text
Version 1
Version 2
Version 3
...
```

후보 모델:

```text
candidate
```

운영 모델:

```text
champion
```

설계 흐름:

```text
Run
 ↓
Metric 확인
 ↓
Quality Gate
 ↓
Registry Version
 ↓
candidate
 ↓
검증
 ↓
champion
```

---

# 12. Step 9 — 운영 단계까지 확장 설계

이번 7단원 실습의 구현 범위를 넘어,
4단원과 6단원의 내용을 연결함.

```text
Registry
   ↓
Champion
   ↓
Docker Serving
   ↓
REST API
   ↓
Operational Data
   ↓
Evidently
   ↓
Data Drift
   ↓
Retraining 판단
```

즉 MLOps는 모델 학습으로 끝나지 않음.

---

# 13. Step 10 — 전체 End-to-End 구조 작성

최종 설계:

```text
Raw Data
   ↓
Data Validation
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Model Quality Gate
   ↓
MLflow Tracking
   ↓
Model Registry
   ↓
Candidate
   ↓
Champion
   ↓
Docker Serving
   ↓
REST API
   ↓
Monitoring
   ↓
Drift Detection
   ↓
Retraining 판단
```

---

# 14. Step별 입력 / 출력 표 작성

| 단계 | 입력 | 수행 작업 | 출력 | 도구 |
|---|---|---|---|---|
| Data Validation | Raw CSV | 결측·범위 확인 | Valid Data | Pandas |
| Preprocess | Valid Data | Feature/Target 분리, Split | Train/Test | ZenML |
| Train | Train Data | Random Forest 학습 | Model | ZenML |
| Evaluate | Model + Test | Accuracy 계산 | Metric | ZenML + MLflow |
| Gate | Accuracy | 기준 비교 | PASS/FAIL | ZenML |
| Register | Model | Registry 등록 | Version | MLflow |
| Serve | Champion | REST API | Prediction | Docker |
| Monitor | Current Data | Drift 비교 | Drift Signal | Evidently |

---

# 15. 도구 연결 구조 작성

```text
Git / DVC
   ↓
Code + Data
   ↓
ZenML
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
Docker Serving
   ↓
Evidently Monitoring
```

자동화까지 포함하면:

```text
Git Push
   ↓
GitHub Actions
   ↓
ZenML Pipeline
   ↓
MLflow
```

---

# 16. 설계 검토 체크리스트

다음 질문을 하나씩 확인함.

```text
1. 예측 Target이 명확한가?

2. Feature와 Target이 분리되어 있는가?

3. Step별 책임이 명확한가?

4. Step 간 입력 / 출력이 명확한가?

5. 어떤 Parameter를 기록할 것인가?

6. 어떤 Metric을 기록할 것인가?

7. 모델을 언제 Registry에 등록할 것인가?

8. candidate와 champion을 어떻게 구분할 것인가?

9. Serving은 어느 단계 이후에 이루어지는가?

10. Monitoring 결과가 Retraining 판단으로 어떻게 이어지는가?
```

---

# 17. 2차시 실습 산출물

2차시 종료 시 다음 결과를 완성함.

```text
① End-to-End Pipeline 구조도

② Step별 입력 / 출력 표

③ 각 단계별 도구 배치

④ MLflow 기록 항목

⑤ Model Quality Gate 기준

⑥ Registry / Serving / Monitoring 위치
```

최종 예:

```text
students.csv
   ↓
[Validation]
   ↓
[Preprocess]
   ↓
X_train / X_test / y_train / y_test
   ↓
[Train]
   ↓
Random Forest
   ↓
[Evaluate]
   ↓
Accuracy
   ↓
[Quality Gate]
   ↓
accuracy >= 0.85
   ↓
[MLflow Registry]
   ↓
candidate
   ↓
champion
   ↓
Serving
   ↓
Monitoring
```

---

# 18. 2차시 핵심 정리

```text
Business Problem
      ↓
ML Problem
      ↓
Pipeline Step
      ↓
Tool Mapping
      ↓
Quality Gate
      ↓
Registry
      ↓
Serving
      ↓
Monitoring
```

핵심:

> MLOps 설계에서 중요한 것은 도구 자체가 아니라 문제를 단계로 분해하고 각 단계의 책임과 연결 관계를 명확히 정의하는 것임

> 2~6단원에서 학습한 도구는 각각 독립적으로 사용하는 것이 아니라 전체 MLOps 흐름 안에서 역할을 가짐

---

# 3차시 실습 — ZenML + MLflow End-to-End 파이프라인 구현

# 19. 실습 목표

- 2차시에서 설계한 Pipeline을 실제 구현함
- ZenML Step을 연결함
- MLflow에 Parameter와 Metric을 기록함
- Model Quality Gate를 적용함
- 기준을 통과한 모델을 Registry에 등록함
- 여러 Parameter 조합으로 Run을 생성하고 비교함
- Registry Version과 candidate Alias를 확인함
- 종합 실행 결과를 설명함

이번 차시 전체 흐름:

```text
Data 생성
   ↓
MLflow Server
   ↓
ZenML 초기화
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
MLflow Tracking
   ↓
Quality Gate
   ↓
Model Registry
   ↓
Run 비교
   ↓
candidate 지정
```

핵심 질문:

> “지금까지 배운 Pipeline, Tracking, Registry를 하나의 실행 흐름으로 연결할 수 있는가?”

---

# 20. 실습 데이터 생성

`src/make_data.py`

```bash
cat > src/make_data.py <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd


def main():

    rng = np.random.default_rng(42)

    n = 300

    study_hours = (
        rng.normal(5, 2, n)
        .clip(0, 12)
    )

    attendance = (
        rng.normal(85, 10, n)
        .clip(40, 100)
    )

    assignment_score = (
        rng.normal(78, 12, n)
        .clip(0, 100)
    )

    midterm_score = (
        rng.normal(75, 15, n)
        .clip(0, 100)
    )

    risk_score = (
        (study_hours < 3).astype(int)
        + (attendance < 75).astype(int)
        + (assignment_score < 65).astype(int)
        + (midterm_score < 60).astype(int)
    )

    risk_flag = (
        risk_score >= 2
    ).astype(int)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance": attendance.round(2),
            "assignment_score": assignment_score.round(2),
            "midterm_score": midterm_score.round(2),
            "risk_flag": risk_flag,
        }
    )

    path = Path(
        "data/students.csv"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(df.head())
    print()
    print("shape:", df.shape)
    print()
    print("risk distribution:")
    print(
        df["risk_flag"]
        .value_counts()
    )


if __name__ == "__main__":
    main()
PY
```

실행:

```bash
python src/make_data.py
```

확인:

```bash
head data/students.csv
```

---

# 21. 데이터 기본 확인

Python에서 데이터 구조 확인:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv(
    "data/students.csv"
)

print(df.describe())
print()
print(df.isna().sum())
print()
print(
    df["risk_flag"]
    .value_counts(normalize=True)
)
PY
```

확인할 내용:

```text
행 수
Feature 범위
결측치 여부
Target 비율
```

---

# 22. MLflow Server 실행

별도 터미널 1:

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

# 23. ZenML 초기화

터미널 2:

```bash
cd ~/mlops-week07

source .venv/bin/activate

zenml init
```

확인:

```bash
zenml status
zenml stack list
```

---

# 24. Pipeline 작성

`src/pipeline.py`

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

    df = pd.read_csv(
        csv_path
    )

    feature_cols = [
        "study_hours",
        "attendance",
        "assignment_score",
        "midterm_score",
    ]

    target_col = "risk_flag"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
    )

    print(
        "Preprocess completed"
    )

    print(
        "train size:",
        len(X_train),
    )

    print(
        "test size :",
        len(X_test),
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


@step
def train_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Annotated[
    RandomForestClassifier,
    "model",
]:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed"
    )

    print(
        "n_estimators:",
        n_estimators,
    )

    print(
        "max_depth:",
        max_depth,
    )

    return model


@step(enable_cache=False)
def evaluate_and_register_step(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_accuracy: float = 0.85,
) -> Annotated[
    float,
    "accuracy",
]:

    predictions = model.predict(
        X_test
    )

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
            "Run ID:",
            run.info.run_id,
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        if accuracy >= min_accuracy:

            model_info = (
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format="cloudpickle",
                    registered_model_name=MODEL_NAME,
                    input_example=X_test.iloc[:3],
                )
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

# 25. Pipeline 실행 파일 작성

`run_pipeline.py`

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

# 26. 첫 번째 Pipeline 실행

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

확인할 내용:

```text
Train / Test Size

n_estimators

max_depth

Run ID

Accuracy

Registry 등록 여부
```

---

# 27. MLflow UI에서 첫 Run 확인

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
├─ n_estimators
├─ max_depth
└─ min_accuracy

Metric
└─ accuracy
```

Run을 클릭하여 상세 정보를 확인함.

---

# 28. Model Quality Gate 실패 실험

이번에는 등록되지 않는 경우를 확인함.

`run_pipeline.py`:

```python
mlops_pipeline(
    csv_path="data/students.csv",
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
MLflow Run
→ 생성됨

Parameter / Metric
→ 기록됨

Model Registry
→ 등록되지 않음
```

핵심:

```text
Tracking
≠
Registry
```

즉 모델이 Registry에 등록되지 않아도
실험 Run 자체는 Tracking에 남음.

---

# 29. Quality Gate 정상 복구

다시:

```python
min_accuracy=0.85
```

로 수정함.

---

# 30. 여러 Parameter로 Pipeline 실행

## Run A

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

---

## Run B

```python
mlops_pipeline(
    csv_path="data/students.csv",
    test_size=0.2,
    n_estimators=100,
    max_depth=5,
    min_accuracy=0.85,
)
```

실행:

```bash
python run_pipeline.py
```

---

## Run C

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

---

# 31. MLflow Run 비교

MLflow UI:

```text
Experiments
→ week07-capstone
```

Run A / B / C를 선택하여 비교함.

비교할 항목:

```text
n_estimators
max_depth
accuracy
```

표:

| Run | n_estimators | max_depth | accuracy |
|---|---:|---:|---:|
| A | 50 | 3 | 확인 |
| B | 100 | 5 | 확인 |
| C | 200 | 8 | 확인 |

핵심:

> 같은 Pipeline이라도 Parameter가 달라지면 서로 다른 Run으로 기록되고 결과를 비교할 수 있음

---

# 32. Model Registry 확인

MLflow UI:

```text
Models
→ student-risk-model
```

Quality Gate를 통과한 Run이 등록되었으면:

```text
Version 1
Version 2
Version 3
...
```

가 생성됨.

확인할 내용:

```text
Version
Run ID
Source
등록 시점
```

---

# 33. Registry와 Tracking 연결 확인

구조:

```text
Run A
   ↓
Accuracy 기록
   ↓
Quality Gate PASS
   ↓
Model Version 생성
```

즉:

```text
Experiment Run
↔
Model Version
```

이 연결됨.

Model Version 화면에서 어떤 Run에서 생성된 모델인지 확인함.

---

# 34. candidate Alias 지정

가장 최근에 등록된 Version을 candidate로 지정함.

```bash
python - <<'PY'
import mlflow

from mlflow import MlflowClient


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()

MODEL_NAME = (
    "student-risk-model"
)

versions = (
    client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
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

MLflow UI에서:

```text
student-risk-model
↓
candidate
```

Alias가 지정되어 있는지 확인함.

---

# 35. candidate 의미 확인

Version 번호만 사용하면:

```text
Version 1
Version 2
Version 3
```

처럼 기술적인 버전만 존재함.

Alias를 사용하면:

```text
candidate
→ 현재 검토 중인 모델

champion
→ 현재 운영 대표 모델
```

처럼 역할을 부여할 수 있음.

이번 실습에서는:

```text
candidate
```

까지만 지정함.

---

# 36. 가장 좋은 Run 선택 토의

MLflow에서 Run A / B / C를 비교하고 다음 질문을 확인함.

```text
1. Accuracy가 가장 높은 Run은 무엇인가?

2. Parameter가 가장 큰 모델이 항상 가장 좋은가?

3. Accuracy 차이가 거의 없다면
   더 단순한 모델을 선택할 수 있는가?

4. Registry에 등록되었다고 해서
   바로 champion으로 사용해야 하는가?
```

핵심:

> Registry 등록은 배포 확정이 아니라 관리 대상으로 등록한 것임

---

# 37. 전체 실행 구조 다시 확인

이번 3차시에서 실제로 구현한 흐름:

```text
students.csv
   ↓
ZenML
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
MLflow Tracking
   ↓
Model Quality Gate
   ↓
Model Registry
   ↓
candidate
```

---

# 38. 2~6단원과 연결

```text
2단원
Data Management / DVC
       ↓
7단원 Data 입력

3단원
MLflow Tracking / Registry
       ↓
7단원 실험 기록 / 모델 등록

4단원
Docker Serving
       ↓
candidate / champion 이후

5단원
GitHub Actions / ZenML
       ↓
7단원 Pipeline 실행

6단원
Evidently Monitoring
       ↓
Serving 이후 Drift 감지
```

---

# 39. 운영 단계까지 확장하면

이번 실습 이후 실제 운영 구조는:

```text
candidate
   ↓
검증
   ↓
champion
   ↓
Docker Image
   ↓
Container
   ↓
REST API
   ↓
Current Data
   ↓
Evidently Monitoring
   ↓
Drift Signal
   ↓
Retraining 판단
```

이 됨.

---

# 40. 결과 발표 준비

각 팀 또는 개인은 다음 내용을 정리함.

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

## 3. MLflow 기록 항목

```text
Parameter
Metric
Run
Model
```

## 4. Quality Gate

```text
accuracy >= 0.85
```

## 5. 실행 결과

```text
Run 비교
Accuracy
Model Version
candidate
```

## 6. 다음 확장

```text
DVC
GitHub Actions
Docker Serving
Evidently Monitoring
Champion Promotion
```

---

# 41. 발표 예시

```text
우리 파이프라인은
학생의 학습위험 여부를 예측하는 문제를 대상으로 설계했습니다.

전처리 단계에서는
학습활동 데이터를 읽고
Feature와 Target을 분리한 뒤
Train/Test 데이터로 나누었습니다.

학습 단계에서는
Random Forest 모델을 사용하고
n_estimators와 max_depth를 Parameter로 관리했습니다.

평가 단계에서는
Accuracy를 계산하고
MLflow에 Parameter와 Metric을 기록했습니다.

Accuracy가 0.85 이상일 경우에만
Model Registry에 등록되도록 구성했습니다.

여러 Parameter 조합을 실행하여
MLflow에서 Run을 비교했고,
등록된 최신 모델을 candidate로 지정했습니다.

실제 운영 환경에서는
candidate를 검증한 뒤 champion으로 승격하고,
Docker를 이용해 Serving한 후
Evidently로 Data Drift를 Monitoring할 수 있습니다.
```

---

# 42. 3차시 핵심 정리

```text
ZenML
→ Pipeline 실행 흐름 관리

MLflow Tracking
→ Run / Parameter / Metric 기록

Model Quality Gate
→ 모델 등록 여부 판단

Model Registry
→ Model Version 관리

Alias
→ candidate / champion 역할 지정
```

최종 구현:

```text
Data
 ↓
Preprocess
 ↓
Train
 ↓
Evaluate
 ↓
Tracking
 ↓
Gate
 ↓
Registry
 ↓
candidate
```

핵심:

> 7단원은 새로운 도구를 추가하는 것이 아니라 앞 단원에서 학습한 요소를 하나의 End-to-End MLOps 흐름으로 연결하는 종합 실습임

---


# 24. 7단원 전체 정리

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

를 판단하는 것임

---

# 25. 2~7단원 전체 연결

```text
2단원
Data Management
DVC

↓

3단원
Experiment Tracking
MLflow

↓

4단원
Model Serving
Docker / Nginx

↓

5단원
Automation
GitHub Actions / ZenML

↓

6단원
Monitoring
Evidently / Retraining

↓

7단원
Integration
End-to-End MLOps Design
```

---

# 26. 선택 확장 과제

1. `evaluate_and_register_step`을 `evaluate_step`과 `register_step`으로 분리함
2. 2단원 DVC를 연결하여 데이터 버전을 함께 관리함
3. GitHub Actions에서 Pipeline을 자동 실행함
4. Accuracy 외에 Precision, Recall, F1-score를 기록함
5. MLflow Run 비교 결과를 기준으로 직접 `champion` Alias를 지정함
6. 4단원 방식으로 `candidate` 또는 `champion` 모델을 REST API로 Serving함
7. Evidently를 이용하여 입력 데이터 Drift를 추가로 모니터링함
8. 자신의 전공 또는 업무 사례를 하나 선정하여 동일한 MLOps 구조로 재설계함
