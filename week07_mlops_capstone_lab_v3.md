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

# 2차시 실습 — MLOps Pipeline 설계와 Mock 실행 검증

# 3. 실습 목표

이번 차시는 문서로 Pipeline을 설명하는 데서 끝내지 않고,
설계안을 실제 ZenML 코드 구조로 옮기고 Mock Pipeline을 실행하여 설계를 검증함.

- 문제를 MLOps 단계로 분해함
- 각 Step의 입력과 출력을 직접 정의함
- Pipeline 설계 명세를 YAML로 작성함
- ZenML Step Skeleton을 코드로 작성함
- 실제 ML 학습 대신 Mock Data를 사용하여 Pipeline 연결을 실행함
- 의도적으로 잘못된 Step 연결을 수정함
- Step 간 의존성과 실행 순서를 확인함
- 3차시에서 구현할 MLflow 기록 위치와 Registry 조건을 코드에 표시함

이번 차시의 핵심:

```text
설계 문서
   ↓
실행 가능한 Pipeline Skeleton
   ↓
Mock 실행
   ↓
설계 오류 확인
   ↓
수정
   ↓
3차시 실제 구현
```

> 2차시에서는 "모델을 잘 만드는 것"이 목적이 아니라  
> "MLOps Pipeline의 구조를 실행 가능한 형태로 설계하는 것"이 목적임

---

# 4. Step 1 — 문제와 데이터 흐름 확인

이번 사례:

```text
학생의 학습위험 여부 예측
```

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

먼저 전체 문제를 Pipeline 관점에서 분해함.

```text
Raw CSV
   ↓
Preprocess
   ↓
Train
   ↓
Evaluate
   ↓
Model Registry
```

각 단계에서 다음 질문에 답해야 함.

```text
무엇을 입력받는가?
무엇을 수행하는가?
무엇을 출력하는가?
다음 Step은 어떤 출력을 필요로 하는가?
```

---

# 5. Step 2 — Pipeline 설계 명세 작성

실제 코드를 작성하기 전에
Pipeline의 입력/출력 관계를 YAML 파일로 명시함.

터미널:

```bash
cd ~/mlops-week07
source .venv/bin/activate
```

`pipeline_design.yaml` 생성:

```bash
cat > pipeline_design.yaml <<'YAML'
pipeline:
  name: student-risk-pipeline

steps:

  preprocess:
    input:
      - students.csv

    output:
      - X_train
      - X_test
      - y_train
      - y_test

    responsibility:
      - load_data
      - split_feature_target
      - train_test_split

  train:
    input:
      - X_train
      - y_train

    output:
      - model

    parameters:
      - n_estimators
      - max_depth

  evaluate:
    input:
      - model
      - X_test
      - y_test

    output:
      - accuracy

    metrics:
      - accuracy

registry:
  model_name: student-risk-model

  condition:
    metric: accuracy
    operator: ">="
    threshold: 0.85
YAML
```

확인:

```bash
cat pipeline_design.yaml
```

설계 결과:

```text
Preprocess
Input
- students.csv

Output
- X_train
- X_test
- y_train
- y_test

        ↓

Train
Input
- X_train
- y_train

Output
- model

        ↓

Evaluate
Input
- model
- X_test
- y_test

Output
- accuracy
```

핵심:

> Pipeline 설계의 핵심은 Step 이름보다  
> 각 Step의 Output이 다음 Step의 어떤 Input으로 연결되는지 정의하는 것임

---

# 6. Step 3 — ZenML 초기화

2차시에서 Mock Pipeline을 실제 실행하기 위해 ZenML을 초기화함.

```bash
zenml init
```

확인:

```bash
zenml status
```

> 이미 동일 프로젝트에서 초기화되어 있으면 기존 설정을 그대로 사용함

---

# 7. Step 4 — Mock Pipeline Skeleton 작성

이번 단계에서는 실제 CSV 처리나 RandomForest 학습을 하지 않음.

대신:

```text
Preprocess
Train
Evaluate
```

Step의 입력/출력 연결 구조만 실제 ZenML 코드로 구현함.

`src/pipeline_design.py` 생성:

```bash
cat > src/pipeline_design.py <<'PY'
from typing import Annotated, Tuple

from zenml import pipeline, step


@step
def preprocess_step(
    csv_path: str,
) -> Tuple[
    Annotated[str, "X_train"],
    Annotated[str, "X_test"],
    Annotated[str, "y_train"],
    Annotated[str, "y_test"],
]:

    print()
    print("[PREPROCESS]")
    print("input :", csv_path)

    # 실제 데이터 대신 Mock Artifact 반환
    X_train = "mock_X_train"
    X_test = "mock_X_test"
    y_train = "mock_y_train"
    y_test = "mock_y_test"

    print(
        "output:",
        X_train,
        X_test,
        y_train,
        y_test,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


@step
def train_step(
    X_train: str,
    y_train: str,
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Annotated[str, "model"]:

    print()
    print("[TRAIN]")
    print("input :", X_train, y_train)
    print(
        "params:",
        n_estimators,
        max_depth,
    )

    # 실제 모델 대신 Mock Model 반환
    model = "mock_model"

    print("output:", model)

    return model


@step
def evaluate_step(
    model: str,
    X_test: str,
    y_test: str,
    min_accuracy: float = 0.85,
) -> Annotated[float, "accuracy"]:

    print()
    print("[EVALUATE]")
    print(
        "input :",
        model,
        X_test,
        y_test,
    )

    # 실제 평가 대신 Mock Accuracy 사용
    accuracy = 0.90

    print("output: accuracy =", accuracy)

    if accuracy >= min_accuracy:
        print(
            "registry decision:",
            "REGISTER",
        )
    else:
        print(
            "registry decision:",
            "DO NOT REGISTER",
        )

    return accuracy


@pipeline
def design_pipeline(
    csv_path: str = "data/students.csv",
    n_estimators: int = 100,
    max_depth: int = 5,
    min_accuracy: float = 0.85,
):

    X_train, X_test, y_train, y_test = (
        preprocess_step(
            csv_path=csv_path,
        )
    )

    model = train_step(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    evaluate_step(
        model=model,
        X_test=X_test,
        y_test=y_test,
        min_accuracy=min_accuracy,
    )
PY
```

---

# 8. Step 5 — Mock Pipeline 실행 파일 작성

`run_design.py` 생성:

```bash
cat > run_design.py <<'PY'
from src.pipeline_design import design_pipeline


if __name__ == "__main__":

    design_pipeline(
        csv_path="data/students.csv",
        n_estimators=100,
        max_depth=5,
        min_accuracy=0.85,
    )
PY
```

실행:

```bash
python run_design.py
```

실행 구조:

```text
preprocess_step
      ↓
train_step
      ↓
evaluate_step
```

예상 의미:

```text
Preprocess
→ X_train / X_test / y_train / y_test 생성

Train
→ X_train / y_train 사용
→ model 생성

Evaluate
→ model / X_test / y_test 사용
→ accuracy 생성
```

여기서 실제 데이터와 모델은 사용하지 않았지만
Pipeline의 입출력 연결과 실행 순서는 실제로 동작함.

---

# 9. Step 6 — Output → Input 연결 확인

Pipeline 코드에서 다음 부분을 확인함.

```python
X_train, X_test, y_train, y_test = (
    preprocess_step(...)
)
```

Preprocess Output:

```text
X_train
X_test
y_train
y_test
```

그중:

```python
model = train_step(
    X_train=X_train,
    y_train=y_train,
)
```

으로 연결됨.

즉:

```text
preprocess_step
      │
      ├─ X_train ──────┐
      ├─ y_train ──────┤
      │                ↓
      │            train_step
      │                ↓
      │              model
      │                │
      ├─ X_test ───────┤
      └─ y_test ───────┤
                       ↓
                 evaluate_step
```

ZenML은 Step 이름만 보고 연결하는 것이 아니라
앞 Step의 Output 객체가 뒤 Step의 Input으로 전달되는 관계를 보고 DAG를 구성함.

---

# 10. Step 7 — 의도적으로 잘못된 설계 만들기

이번에는 Pipeline 설계를 일부러 잘못 구성함.

먼저 현재 파일 백업:

```bash
cp src/pipeline_design.py src/pipeline_design_ok.py
```

`train_step` 호출에서 `y_train` 연결을 제거하면:

```python
model = train_step(
    X_train=X_train,
    n_estimators=n_estimators,
    max_depth=max_depth,
)
```

이 구조는 잘못된 설계임.

이유:

```text
모델 학습
=
X_train
+
y_train
```

이 필요한데 Target이 빠졌기 때문임.

또한 Evaluate를 다음처럼 설계하면:

```python
evaluate_step(
    model=model,
    min_accuracy=min_accuracy,
)
```

잘못된 구조임.

왜냐하면 평가에는:

```text
model
X_test
y_test
```

가 필요하기 때문임.

설계 검토 질문:

```text
Train에 Target이 있는가?
Evaluate에 Test Feature가 있는가?
Evaluate에 실제 정답 y_test가 있는가?
```

---

# 11. Step 8 — 실행 가능한 오류를 직접 확인

이번에는 Python 함수의 필수 Input을 빠뜨려
실제로 오류가 발생하도록 해봄.

`run_design_error.py` 생성:

```bash
cat > run_design_error.py <<'PY'
from src.pipeline_design import train_step


if __name__ == "__main__":

    # y_train을 의도적으로 전달하지 않음
    train_step(
        X_train="mock_X_train",
        n_estimators=100,
        max_depth=5,
    )
PY
```

실행:

```bash
python run_design_error.py
```

오류를 통해 확인:

```text
train_step
필수 입력:
X_train
y_train
```

즉 설계 단계에서 Input을 명확히 정의하면
필수 데이터 누락을 코드 수준에서도 확인할 수 있음.

---

# 12. Step 9 — 설계 복구 및 재실행

정상 Pipeline으로 다시 실행:

```bash
python run_design.py
```

확인:

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

Pipeline이 정상 실행되면:

```text
Step 책임
Input
Output
의존성
실행 순서
```

가 기본적으로 설계된 상태임.

---

# 13. Step 10 — MLflow 기록 위치를 Skeleton에 표시

3차시에서는 Mock Step을 실제 코드로 교체하고 MLflow를 연결함.

2차시에서는 어디에서 무엇을 기록할지 먼저 설계함.

Train Step:

```text
Parameter

n_estimators
max_depth
```

Evaluate Step:

```text
Metric

accuracy
```

Model Registry:

```text
조건

accuracy >= 0.85
```

현재 Mock 코드와 실제 구현의 대응 관계:

```text
2차시 Mock 설계

train_step
    ↓
Parameter 위치 결정

evaluate_step
    ↓
Metric 위치 결정

accuracy >= 0.85
    ↓
Registry 조건 결정
```

3차시 실제 구현:

```text
mlflow.log_param()
mlflow.log_metric()
mlflow.sklearn.log_model()
```

---

# 14. Step 11 — Pipeline 설계표 최종 확인

2차시에서 실제 작성한 코드와 설계 명세를 기준으로
다음 표를 완성함.

| Step | Input | 수행 작업 | Output | 3차시 연결 |
|---|---|---|---|---|
| Preprocess | Raw CSV | 데이터 로딩, 분리, Split | X_train, X_test, y_train, y_test | Pandas + train_test_split |
| Train | X_train, y_train | 모델 학습 | model | RandomForest + MLflow Parameter |
| Evaluate | model, X_test, y_test | 성능 평가 | accuracy | MLflow Metric |
| Registry Gate | accuracy | 기준 비교 | Register / Reject | MLflow Model Registry |

---

# 15. 2차시 산출물

이번 차시 종료 시 실제 PC에 다음 파일이 존재함.

```text
mlops-week07/
├── pipeline_design.yaml
├── run_design.py
├── run_design_error.py
└── src/
    └── pipeline_design.py
```

2차시에서 완성한 것:

```text
① Pipeline 설계 명세
② ZenML Step Skeleton
③ Mock Pipeline 실행
④ 잘못된 Input 연결 확인
⑤ Step 간 Output → Input 흐름 확인
⑥ MLflow Parameter / Metric 기록 위치 설계
⑦ Registry Gate 조건 설계
```

---

# 16. 2차시 핵심 정리

이번 차시에서는 실제 RandomForest를 학습하지 않았음.

대신:

```text
설계
   ↓
코드 Skeleton
   ↓
Mock Artifact
   ↓
Pipeline 실제 실행
   ↓
Input / Output 검증
```

을 수행함.

핵심:

> MLOps Pipeline 설계는 단순히 박스와 화살표를 그리는 것이 아니라  
> 각 Step의 책임과 Input/Output을 실행 가능한 구조로 정의하는 과정임

3차시에서는 이번에 만든 Skeleton을 실제 구현으로 교체함.

```text
Mock X_train
→ 실제 Pandas DataFrame

Mock Model
→ RandomForestClassifier

Mock Accuracy
→ accuracy_score()

주석으로 설계한 Parameter
→ mlflow.log_param()

주석으로 설계한 Metric
→ mlflow.log_metric()

Registry 조건
→ MLflow Model Registry 등록
```

즉:

```text
2차시
설계 + Mock 검증

        ↓

3차시
실제 구현 + 실행 결과 관리
```

의 흐름으로 연결됨.

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

> 3·4단원 MLflow Server가 아직 실행 중이면 `5000` 포트 충돌 발생 → 먼저 종료 (`Ctrl + C`)

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

# MLflow 기록·Registry 등록은 외부 시스템에 영향을 주는 작업이므로 캐시 비활성화
@step(enable_cache=False)
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

            # MLflow 3.x: name 사용, RandomForest는 cloudpickle로 저장 (3단원와 동일)
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
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

> `evaluate_and_register_step`에 `enable_cache=False`를 지정한 이유:  
> ZenML은 입력과 Parameter가 같으면 이전 결과를 재사용(Cache)함 → 같은 설정으로 다시 실행하면 Step 자체가 건너뛰어져 MLflow Run 기록과 모델 등록이 일어나지 않음

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

MLflow UI에서 Run을 비교함

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

이 생성됨

---

# 21. candidate Alias 지정

가장 최근에 등록된 Model Version을 `candidate`로 지정:

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

> 위 코드는 성능 비교 없이 최신 Version을 선택함  
> MLflow UI에서 비교 후 다른 Version을 지정하려면 `version=latest.version` 대신 Version 번호(예: `"2"`)를 직접 입력

---

# 22. 결과 발표 준비

각 팀 또는 개인은 다음 내용을 정리함

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

1. `evaluate_step`을 별도 ZenML Step으로 분리함
2. 2단원 DVC를 연결하여 데이터 버전을 함께 관리함
3. GitHub Actions에서 Pipeline을 자동 실행함
4. Accuracy 외에 Precision, Recall, F1-score를 기록함
5. Model Registry에 `candidate`와 `champion` Alias를 적용함
6. 4단원 방식으로 `candidate` 모델을 REST API로 Serving함
7. Evidently를 이용하여 입력 데이터 Drift를 추가로 모니터링함
8. 자신의 전공 또는 업무 사례를 하나 선정하여 동일한 MLOps 구조로 재설계함
