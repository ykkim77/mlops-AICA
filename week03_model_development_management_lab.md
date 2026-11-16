# 3주차 실습 — 모델 개발 및 관리

> MLOps 방법론  
> 주제: MLflow Tracking → MLflow Projects → Model Registry  
> 실습 환경: WSL2 Ubuntu 22.04 기준 / Python 3.10 이상 / Git  
> 클라우드 계정: 필요 없음

---

# 0. 실습 개요

이번 주 실습에서는 **하나의 Iris 분류 프로젝트를 2~4차시 동안 계속 발전**시킴

```text
[2차시]
모델 학습
   ↓
MLflow Tracking
   ↓
Parameter / Metric / Artifact 기록
   ↓
여러 Run 비교

[3차시]
학습 코드를 MLflow Project로 패키징
   ↓
동일한 명령 구조로 반복 실행
   ↓
재현 가능한 실행 단위 구성

[4차시]
학습된 모델 등록
   ↓
Model Version 관리
   ↓
Alias / Tag 관리
   ↓
Candidate → Champion
```

이번 주의 핵심 질문은 다음과 같음

> **“어떤 설정으로 어떤 모델을 만들었고, 어떤 모델을 사용할 것인가?”**

---

# 1. 공통 실습 환경 구성

## 1.1 프로젝트 디렉터리 생성

```bash
mkdir -p ~/mlops-week03
cd ~/mlops-week03
```

## 1.2 Git 저장소 초기화

```bash
git init -b main
```

## 1.3 Python 가상환경 생성

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 1.4 패키지 설치

```bash
python -m pip install --upgrade pip
pip install "mlflow==3.18.0" scikit-learn pandas matplotlib
```

> 실습 검증 기준 버전: MLflow 3.18.0 — 버전을 고정하지 않으면 학생마다 다른 버전이 설치되어 동작이 달라질 수 있음

설치 확인:

```bash
python --version
mlflow --version
```

## 1.5 폴더 구조 생성

```bash
mkdir -p src artifacts
```

## 1.6 `.gitignore` 작성

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
mlflow.db
mlartifacts/
mlruns/
artifacts/
EOF
```

> Python 3.12 이하의 `venv`는 자체 `.gitignore`를 만들지 않음 → `git add .` 실행 시 `.venv` 전체가 커밋되므로 반드시 먼저 작성
> `mlflow.db`, `mlartifacts/`는 MLflow 서버가 생성하는 실행 기록 저장소이므로 Git 관리 대상에서 제외

---

# 2. MLflow Tracking Server 실행

실습에서는 학생 PC에서 로컬 MLflow 서버를 실행함

## 2.1 터미널 1에서 MLflow Server 실행

프로젝트 루트에서:

```bash
cd ~/mlops-week03
source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

정상 실행 시 브라우저에서 다음 주소 접속

```text
http://127.0.0.1:5000
```

> 이 터미널은 MLflow 서버 실행용이므로 그대로 둠

## 2.2 터미널 2 열기

새 WSL 터미널을 열고:

```bash
cd ~/mlops-week03
source .venv/bin/activate
```

이후 실습 명령은 터미널 2에서 수행함

---

# 2차시 실습 — MLflow Tracking

## 3. 실습 목표

- Experiment와 Run의 차이를 이해함
- Parameter, Metric, Artifact를 기록함
- 여러 번의 모델 학습을 하나의 Experiment에 모음
- MLflow UI에서 Run을 비교함
- 어떤 설정이 더 좋은 결과를 만들었는지 확인함

---

# 4. 첫 번째 학습 프로그램 작성

`src/train.py` 파일을 생성함

```bash
cat > src/train.py <<'PY'
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. MLflow Tracking Server 설정
    # ---------------------------------------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # 여러 번의 실행을 하나의 Experiment 아래에 모음
    mlflow.set_experiment("week03-iris")

    # ---------------------------------------------------------
    # 2. 데이터 준비
    # ---------------------------------------------------------
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=iris.target,
    )

    # ---------------------------------------------------------
    # 3. 하나의 학습 실행 = 하나의 MLflow Run
    # ---------------------------------------------------------
    with mlflow.start_run() as run:

        # Parameter 기록
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # -----------------------------------------------------
        # 4. 모델 학습
        # -----------------------------------------------------
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        model.fit(X_train, y_train)

        # -----------------------------------------------------
        # 5. 평가
        # -----------------------------------------------------
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Metric 기록
        mlflow.log_metric("accuracy", accuracy)

        # -----------------------------------------------------
        # 6. Artifact 생성
        # -----------------------------------------------------
        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(exist_ok=True)

        # Confusion Matrix 이미지 생성
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=iris.target_names,
            ax=ax,
        )
        fig.tight_layout()

        confusion_path = artifact_dir / "confusion_matrix.png"
        fig.savefig(confusion_path)
        plt.close(fig)

        # 학습 결과를 JSON 파일로 저장
        result = {
            "run_id": run.info.run_id,
            "accuracy": accuracy,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        }

        result_path = artifact_dir / "result.json"

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Artifact 기록
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(result_path))

        # -----------------------------------------------------
        # 7. 모델 자체도 MLflow Artifact로 기록
        # -----------------------------------------------------
        # MLflow 3.x: artifact_path 대신 name 사용
        # 최신 MLflow는 sklearn 모델을 기본적으로 skops 형식으로 저장하며,
        # RandomForest는 이 형식에서 저장 오류가 발생하므로 cloudpickle 지정
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            input_example=X_test[:3],
        )

        print("======================================")
        print(f"Run ID       : {run.info.run_id}")
        print(f"n_estimators : {args.n_estimators}")
        print(f"max_depth    : {args.max_depth}")
        print(f"accuracy     : {accuracy:.4f}")
        print("======================================")

if __name__ == "__main__":
    main()
PY
```

---

# 5. 첫 번째 Run 실행

```bash
python src/train.py \
  --n_estimators 20 \
  --max_depth 2
```

> 실행 시 `pickle or cloudpickle format requires exercising caution` 경고가 출력됨 — 신뢰할 수 없는 모델 파일을 불러올 때의 보안 주의 문구이며 실습 진행에는 문제 없음

브라우저의 MLflow UI에서 다음 Experiment를 확인함

```text
week03-iris
```

확인할 항목:

- Parameters
- Metrics
- Artifacts
- Run ID
- 실행 시간

---

# 6. 서로 다른 파라미터로 여러 번 실행

## Run 1

```bash
python src/train.py \
  --n_estimators 20 \
  --max_depth 2
```

## Run 2

```bash
python src/train.py \
  --n_estimators 50 \
  --max_depth 3
```

## Run 3

```bash
python src/train.py \
  --n_estimators 100 \
  --max_depth 5
```

## Run 4

```bash
python src/train.py \
  --n_estimators 150 \
  --max_depth 8
```

---

# 7. MLflow UI에서 Run 비교

브라우저에서:

```text
http://127.0.0.1:5000
```

다음 순서로 확인함

1. `week03-iris` Experiment 선택
2. 여러 Run 선택
3. Compare 기능 실행
4. `n_estimators`, `max_depth`, `accuracy` 비교
5. 각 Run의 Artifact 확인

### 생각해보기

정확도가 가장 높은 Run이 항상 가장 좋은 모델인가?

고려해야 할 요소:

- 정확도
- 모델 크기
- 추론 속도
- 과적합 가능성
- 재현성
- 운영 비용

---

# 8. Parameter / Metric / Artifact 정리

| 구분 | 예시 | 의미 |
|---|---|---|
| Parameter | `max_depth=3` | 학습 전에 결정한 설정 |
| Metric | `accuracy=0.9667` | 학습/평가 결과 |
| Artifact | `confusion_matrix.png` | 실행 결과로 생성된 파일 |
| Model | RandomForest 모델 | 학습된 모델 객체 |
| Run | 한 번의 `train.py` 실행 | 하나의 실험 실행 기록 |
| Experiment | `week03-iris` | 여러 Run을 묶는 논리적 그룹 |

---

# 9. 선택 실습 — Autologging 확인

MLflow는 일부 ML 프레임워크에서 자동 기록 기능을 제공함

`src/train_autolog.py` 생성:

```bash
cat > src/train_autolog.py <<'PY'
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("week03-autolog")

mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("test accuracy:", model.score(X_test, y_test))
PY
```

실행:

```bash
python src/train_autolog.py
```

MLflow UI에서 수동 기록 방식과 Autologging 방식의 차이를 확인함

---

# 3차시 실습 — MLflow Projects

## 10. 실습 목표

- 학습 코드를 실행 가능한 프로젝트 단위로 패키징함
- `MLproject` 파일의 역할을 이해함
- 파라미터를 프로젝트 외부에서 전달함
- 동일한 실행 방법으로 여러 번 모델을 학습함
- 재현 가능한 실행 구조의 의미를 이해함

---

# 11. MLflow Project 구조

이번 실습의 프로젝트 구조:

```text
mlops-week03/
├── MLproject
├── requirements.txt
├── src/
│   └── train.py
└── artifacts/
```

---

# 12. requirements.txt 작성

```bash
cat > requirements.txt <<'EOF'
mlflow==3.18.0
scikit-learn
pandas
matplotlib
EOF
```

---

# 13. MLproject 파일 작성

프로젝트 루트에 `MLproject` 파일을 생성함

```bash
cat > MLproject <<'YAML'
name: week03-iris-project

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 50}
      max_depth: {type: int, default: 3}
      test_size: {type: float, default: 0.2}
      random_state: {type: int, default: 42}

    command: >
      python src/train.py
      --n_estimators {n_estimators}
      --max_depth {max_depth}
      --test_size {test_size}
      --random_state {random_state}
YAML
```

---

# 14. MLflow Project 실행

현재 Python 환경을 그대로 사용하여 실행함

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow run . \
  --env-manager local \
  --experiment-name week03-iris \
  -P n_estimators=50 \
  -P max_depth=3
```

> `mlflow run`은 `train.py`보다 먼저 Run을 생성함 → Tracking URI와 Experiment를 `train.py`와 동일하게 지정해야 함  
> 지정하지 않으면 `active experiment ID does not match environment run ID` 오류로 실패

다른 파라미터로 다시 실행:

```bash
mlflow run . \
  --env-manager local \
  --experiment-name week03-iris \
  -P n_estimators=100 \
  -P max_depth=5
```

MLflow UI에서 새로운 Run이 추가되는 것을 확인함

---

# 15. `python train.py`와 `mlflow run` 비교

일반 실행:

```bash
python src/train.py \
  --n_estimators 100 \
  --max_depth 5
```

MLflow Project 실행:

```bash
mlflow run . \
  --env-manager local \
  --experiment-name week03-iris \
  -P n_estimators=100 \
  -P max_depth=5
```

차이:

```text
일반 Python 실행
      ↓
개별 스크립트 실행

MLflow Project
      ↓
프로젝트 정의 + Entry Point + Parameter
      ↓
표준화된 실행 단위
```

---

# 16. Git에 프로젝트 저장

```bash
git add .
git commit -m "week03: mlflow tracking and project"
```

확인:

```bash
git log --oneline
```

MLflow Project는 로컬 디렉터리뿐 아니라 Git 저장소를 실행 단위로 사용할 수도 있음

---

# 17. 재현 가능한 실행의 의미

재현 가능한 모델 학습은 단순히 모델 파일을 저장하는 것이 아님

```text
Code
+
Parameter
+
Environment
+
Data
      ↓
Reproducible Training
```

이번 차시에서는 이 중:

```text
Code
+
Parameter
+
실행 방법
```

을 `MLproject`로 표준화함

> 데이터 버전은 2주차 DVC에서 다룬 개념과 연결됨

---

# 4차시 실습 — MLflow Model Registry

## 18. 실습 목표

- 학습된 모델을 Model Registry에 등록함
- 모델의 Version을 확인함
- 새로운 모델을 등록하여 Version이 증가하는 것을 확인함
- Alias와 Tag를 이용하여 모델의 역할과 상태를 표현함
- `candidate`와 `champion` 모델을 구분함

---

# 19. Registry용 학습 프로그램 작성

`src/train_register.py` 생성:

```bash
cat > src/train_register.py <<'PY'
import argparse

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "iris-random-forest"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("week03-registry")

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_metric("accuracy", accuracy)

        # 모델을 기록하면서 Registry에 등록
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_NAME,
            input_example=X_test[:3],
        )

        print("======================================")
        print(f"Run ID      : {run.info.run_id}")
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Model Name  : {MODEL_NAME}")
        print(f"Model URI   : {model_info.model_uri}")
        print("======================================")

if __name__ == "__main__":
    main()
PY
```

---

# 20. Model Version 1 생성

```bash
python src/train_register.py \
  --n_estimators 50 \
  --max_depth 3
```

MLflow UI에서:

```text
Models
→ iris-random-forest
```

를 선택함

확인:

```text
Version 1
```

---

# 21. Model Version 2 생성

다른 설정으로 다시 학습함

```bash
python src/train_register.py \
  --n_estimators 150 \
  --max_depth 6
```

MLflow UI에서 다시 확인함

```text
iris-random-forest

Version 1
Version 2
```

동일한 Registered Model에 새로운 모델을 등록하면 Version이 증가함

---

# 22. Version별 성능 비교

MLflow UI에서 각 Model Version이 생성된 Run으로 이동하여 Accuracy를 확인함

예:

```text
iris-random-forest

Version 1
  └─ accuracy = ...

Version 2
  └─ accuracy = ...
```

> Model Version 자체뿐 아니라 어떤 Run에서 생성되었는지 추적할 수 있는 것이 중요함

---

# 23. Alias 설정

최신 MLflow에서는 모델의 배포 역할을 표현할 때 **Alias**를 사용할 수 있음

이번 실습에서는 다음 Alias를 사용함

```text
candidate
   ↓
배포 후보 모델

champion
   ↓
현재 선택된 대표 모델
```

---

# 24. Python으로 Alias 지정

`src/set_alias.py` 생성:

```bash
cat > src/set_alias.py <<'PY'
import argparse

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "iris-random-forest"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=args.alias,
        version=args.version,
    )

    print(
        f"{MODEL_NAME} version {args.version} "
        f"→ alias '{args.alias}' 지정 완료"
    )

if __name__ == "__main__":
    main()
PY
```

Version 2를 candidate로 지정:

```bash
python src/set_alias.py \
  --version 2 \
  --alias candidate
```

MLflow UI에서 Alias가 표시되는지 확인함

---

# 25. Champion 지정

성능을 비교한 뒤 사용할 모델을 선택함

예를 들어 Version 2를 대표 모델로 선택한다면:

```bash
python src/set_alias.py \
  --version 2 \
  --alias champion
```

구조:

```text
iris-random-forest

Version 1

Version 2
 ├─ candidate
 └─ champion
```

Alias는 나중에 다른 Version으로 옮길 수 있음

---

# 26. Tag 설정

Tag는 모델 버전에 추가적인 상태 정보를 기록할 때 사용할 수 있음

`src/set_tag.py` 생성:

```bash
cat > src/set_tag.py <<'PY'
import argparse

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "iris-random-forest"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--value", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=args.version,
        key=args.key,
        value=args.value,
    )

    print(
        f"version {args.version}: "
        f"{args.key}={args.value}"
    )

if __name__ == "__main__":
    main()
PY
```

예:

```bash
python src/set_tag.py \
  --version 2 \
  --key validation_status \
  --value passed
```

---

# 27. Alias로 모델 불러오기

`champion` Alias가 붙은 모델을 코드에서 직접 참조할 수 있음

`src/load_champion.py` 생성:

```bash
cat > src/load_champion.py <<'PY'
import mlflow
import mlflow.pyfunc

from sklearn.datasets import load_iris

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/iris-random-forest@champion"

model = mlflow.pyfunc.load_model(model_uri)

X, _ = load_iris(return_X_y=True)

predictions = model.predict(X[:5])

print("Model URI:", model_uri)
print("Predictions:", predictions)
PY
```

실행:

```bash
python src/load_champion.py
```

중요한 점은 코드가 특정 Version 번호를 직접 사용하지 않는다는 것임

```text
models:/iris-random-forest/2
```

대신:

```text
models:/iris-random-forest@champion
```

을 사용함

따라서 나중에 champion Alias를 Version 3으로 변경해도 호출 코드 자체는 바꿀 필요가 없음

---

# 28. Stage와 Alias

과거 MLflow에서는 다음과 같은 고정 Stage를 사용했음

```text
None
  ↓
Staging
  ↓
Production
  ↓
Archived
```

현재 MLflow에서는 고정 Stage보다 **Alias와 Tag를 이용한 유연한 상태 관리 방식**을 사용함

이번 실습에서는:

```text
candidate
   ↓
검증 대상

champion
   ↓
현재 사용할 대표 모델
```

을 사용했음

개념적으로는 다음과 같이 연결해서 이해할 수 있음

```text
전통적 개념              이번 실습

Staging          →       candidate
Production       →       champion
```

단, Alias는 고정된 Stage가 아니라 사용자가 정의할 수 있는 이름임

---

# 29. 전체 실습 결과 확인

프로젝트 구조:

```text
mlops-week03/
├── .git/
├── .venv/
├── MLproject
├── requirements.txt
├── mlflow.db
├── mlartifacts/
├── artifacts/
│   ├── confusion_matrix.png
│   └── result.json
└── src/
    ├── train.py
    ├── train_autolog.py
    ├── train_register.py
    ├── set_alias.py
    ├── set_tag.py
    └── load_champion.py
```

환경에 따라 MLflow가 생성하는 저장 디렉터리 이름은 다를 수 있음

---

# 30. 3주차 전체 정리

이번 주 실습의 전체 흐름:

```text
모델 학습
   ↓
MLflow Tracking
   ↓
Parameter
Metric
Artifact
Model
   ↓
여러 Run 비교
   ↓
MLflow Project
   ↓
재현 가능한 실행 구조
   ↓
Model Registry
   ↓
Model Version
   ↓
Alias / Tag
   ↓
candidate
   ↓
champion
```

---

# 31. 핵심 개념

## 1. Experiment와 Run

```text
Experiment
 ├─ Run 1
 ├─ Run 2
 ├─ Run 3
 └─ Run 4
```

Experiment는 여러 Run을 묶는 그룹이고,  
Run은 한 번의 모델 학습 실행임

---

## 2. Parameter와 Metric

```text
Parameter
모델을 실행하기 전에 설정하는 값

예:
n_estimators
max_depth
```

```text
Metric
실행 결과로 얻어지는 성능 값

예:
accuracy
loss
F1-score
```

---

## 3. Artifact

학습 과정에서 생성되는 파일:

```text
모델
그래프
리포트
이미지
JSON
CSV
```

등을 의미함

---

## 4. MLflow Project

```text
Code
+
Parameter
+
실행 방법
+
Environment 정의
```

를 하나의 재현 가능한 프로젝트 단위로 구성함

---

## 5. Model Registry

```text
Registered Model
   │
   ├─ Version 1
   ├─ Version 2
   └─ Version 3
```

학습된 여러 모델 버전을 중앙에서 관리함

---

## 6. Alias

모델 Version에 사람이 이해하기 쉬운 역할 이름을 부여함

```text
Version 2
   ↓
champion
```

배포 코드는:

```text
models:/iris-random-forest@champion
```

과 같이 특정 Version 번호가 아니라 Alias를 참조할 수 있음

---

# 32. 2주차와 3주차 연결

```text
2주차

어떤 데이터로 학습했는가?
        ↓
DVC
        ↓
Data Version

3주차

어떤 설정으로 학습했는가?
        ↓
MLflow Tracking
        ↓
Parameter / Metric / Artifact

어떤 모델을 사용할 것인가?
        ↓
Model Registry
        ↓
Version / Alias
```

---

# 33. 다음 주차 연결

4주차에서는 이번 주에 Registry에 등록한 모델을 실제 서비스로 연결함

```text
Model Registry
      ↓
champion Model
      ↓
MLflow Model Serving
      ↓
REST API
      ↓
Docker
      ↓
다중 인스턴스
      ↓
Load Balancing
```

즉,

```text
3주차
좋은 모델을 기록하고 선택

↓

4주차
선택한 모델을 서비스로 배포
```

로 이어짐

---

# 34. 선택 과제

1. `n_estimators`, `max_depth` 값을 바꾸어 5개 이상의 Run을 생성함
2. MLflow UI에서 Accuracy 기준으로 Run을 비교함
3. 가장 성능이 좋은 Run과 가장 단순한 모델의 차이를 설명함
4. Model Version을 하나 더 생성하여 Version 3까지 만듦
5. `candidate` Alias를 Version 3으로 옮김
6. 검증 후 `champion` Alias도 Version 3으로 변경함
7. `load_champion.py`를 다시 실행하여 코드 변경 없이 새로운 Champion 모델이 로드되는지 확인함
