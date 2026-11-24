# 6단원 실습 — 모델 모니터링 및 유지보수

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 주제: Drift 시뮬레이션 → Evidently Drift Report 생성·해석  
> 실습 환경: WSL2 Ubuntu 22.04 기준 / Python 3.10 이상  
> 클라우드 계정: 필요 없음  
> 실습 기준 Evidently: 0.7.23

---

# 0. 실습 개요

6단원 실습: 총 2개 차시(2·3차시)로 구성

```text
2차시
Reference Data
      +
Current Data
      ↓
Drift가 없는 데이터
      +
Drift가 있는 데이터 생성
      ↓
분포 변화 확인
```

```text
3차시
Reference
    vs
Current
      ↓
Evidently
      ↓
Data Drift Report
      ↓
HTML / JSON
      ↓
Drift Share 해석
```

이번 주의 핵심은 다음과 같음

```text
Monitoring
=
운영 이후의 데이터와 모델 상태를 지속적으로 관찰

Data Drift
=
입력 데이터 분포 P(X)의 변화

Concept Drift
=
입력과 정답의 관계 P(Y|X)의 변화
```

> Data Drift가 발생했다고 해서 모델 성능이 반드시 저하되었다는 뜻은 아님  
> 이번 실습은 입력 데이터의 분포 변화를 감지하는 데 초점을 둠

---

# 1. 공통 실습 환경 구성

```bash
mkdir -p ~/mlops-week06
cd ~/mlops-week06

git init -b main

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install \
  "evidently==0.7.23" \
  scikit-learn \
  "pandas<3" \
  numpy
```

> Evidently 0.7.23은 pandas 3.x에서 `import evidently` 단계부터 실패함 → `pandas<3`으로 고정  
> (Python 3.10은 pandas 3을 설치하지 않으나 Python 3.11 이상 환경에서는 문제 발생)

설치 확인:

```bash
python -c "import evidently; print(evidently.__version__)"
```

폴더 생성:

```bash
mkdir -p data/reference
mkdir -p data/current
mkdir -p reports
mkdir -p src
```

프로젝트 구조:

```text
mlops-week06/
├── data/
│   ├── reference/
│   └── current/
├── reports/
└── src/
```

---

# 2차시 실습 — Drift 시뮬레이션

## 실습 목표

- Reference Data와 Current Data의 차이를 이해함
- 정상 Current Data와 Drift Current Data를 각각 생성함
- 일부 Feature의 분포를 의도적으로 변화시킴
- 평균값과 통계량을 비교하여 Drift의 개념을 확인함

---

# 2. 실습 데이터 생성

`src/make_data.py`

```python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

REFERENCE_PATH = Path("data/reference/reference.csv")
CURRENT_NORMAL_PATH = Path("data/current/current_normal.csv")
CURRENT_DRIFT_PATH = Path("data/current/current_drift.csv")

def main():
    rng = np.random.default_rng(42)

    iris = load_iris(as_frame=True)

    df = iris.frame.copy()

    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    # 1. Reference Data
    reference = df.copy()

    # 2. Current Normal
    current_normal = df.copy()

    for col in feature_columns:
        current_normal[col] = (
            current_normal[col]
            + rng.normal(0, 0.02, size=len(current_normal))
        )

    # 3. Current Drift
    current_drift = df.copy()

    current_drift["sepal_length"] += 1.5
    current_drift["petal_length"] += 2.0
    current_drift["petal_width"] *= 1.6

    for col in feature_columns:
        current_drift[col] = (
            current_drift[col]
            + rng.normal(0, 0.03, size=len(current_drift))
        )

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_NORMAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_DRIFT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference.to_csv(REFERENCE_PATH, index=False)
    current_normal.to_csv(CURRENT_NORMAL_PATH, index=False)
    current_drift.to_csv(CURRENT_DRIFT_PATH, index=False)

    print("데이터 생성 완료")
    print(f"- Reference     : {REFERENCE_PATH}")
    print(f"- Current Normal: {CURRENT_NORMAL_PATH}")
    print(f"- Current Drift : {CURRENT_DRIFT_PATH}")

if __name__ == "__main__":
    main()
```

실행:

```bash
python src/make_data.py
```

확인:

```bash
ls -R data
```

---

# 3. 통계량 비교

`src/compare_stats.py`

```python
import pandas as pd

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

reference = pd.read_csv(
    "data/reference/reference.csv"
)

normal = pd.read_csv(
    "data/current/current_normal.csv"
)

drift = pd.read_csv(
    "data/current/current_drift.csv"
)

def print_means(name, df):
    print(f"\n=== {name} ===")
    print(
        df[FEATURES]
        .mean()
        .round(3)
    )

print_means(
    "REFERENCE",
    reference,
)

print_means(
    "CURRENT NORMAL",
    normal,
)

print_means(
    "CURRENT DRIFT",
    drift,
)
```

실행:

```bash
python src/compare_stats.py
```

확인할 내용:

```text
Reference
vs
Current Normal
```

은 평균과 분포가 비슷함

반면:

```text
Reference
vs
Current Drift
```

에서는 일부 Feature의 평균이 크게 달라짐

---

# 4. 2차시 핵심

```text
Reference Data
=
비교 기준

Current Data
=
현재 운영 데이터
```

모니터링의 기본 구조:

```text
Reference
    vs
Current
```

이번 차시에서는 Drift가 없는 데이터와 Drift가 있는 데이터를 직접 만들어  
분포 변화가 무엇인지 눈으로 확인함

---

# 3차시 실습 — Evidently Drift Report 생성 및 해석

## 실습 목표

- Evidently의 기본 구조를 이해함
- `DataDriftPreset`을 이용해 Drift Report를 생성함
- 정상 Current와 Drift Current를 각각 비교함
- HTML Report에서 Feature별 분포 변화를 확인함
- Drift된 Feature 수와 Drift Share를 해석함

---

# 5. Evidently 기본 구조

```text
Pandas DataFrame
      ↓
DataDefinition
      ↓
Evidently Dataset
      ↓
Report
      ↓
DataDriftPreset
      ↓
Snapshot
      ↓
HTML / JSON
```

---

# 6. Drift Report 코드 작성

`src/drift_report.py`

```python
import argparse
from pathlib import Path

import pandas as pd

from evidently import (
    DataDefinition,
    Dataset,
    Report,
)
from evidently.presets import DataDriftPreset

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

def make_dataset(df):
    definition = DataDefinition(
        numerical_columns=FEATURES,
    )

    return Dataset.from_pandas(
        df[FEATURES],
        data_definition=definition,
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--current",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    reference_df = pd.read_csv(
        "data/reference/reference.csv"
    )

    current_df = pd.read_csv(
        args.current
    )

    reference_dataset = make_dataset(
        reference_df
    )

    current_dataset = make_dataset(
        current_df
    )

    report = Report(
        [
            DataDriftPreset(
                columns=FEATURES,
                drift_share=0.5,
            )
        ]
    )

    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(
        str(output_path)
    )

    json_path = output_path.with_suffix(
        ".json"
    )

    json_path.write_text(
        snapshot.json(),
        encoding="utf-8",
    )

    print(
        f"HTML report: {output_path}"
    )

    print(
        f"JSON report: {json_path}"
    )

    report_dict = snapshot.dict()

    for metric in report_dict.get(
        "metrics",
        [],
    ):
        metric_name = metric.get(
            "metric_name",
            "",
        )

        if metric_name.startswith(
            "DriftedColumnsCount"
        ):
            value = metric.get(
                "value",
                {},
            )

            print()
            print(
                "Drifted Columns:",
                value.get("count"),
            )

            print(
                "Drift Share:",
                value.get("share"),
            )

if __name__ == "__main__":
    main()
```

---

# 7. 정상 Current Data 평가

```bash
python src/drift_report.py \
  --current data/current/current_normal.csv \
  --output reports/normal_drift_report.html
```

생성:

```text
reports/
├── normal_drift_report.html
└── normal_drift_report.json
```

예상 출력:

```text
Drifted Columns: 0.0
Drift Share: 0.0
```

---

# 8. Drift Current Data 평가

```bash
python src/drift_report.py \
  --current data/current/current_drift.csv \
  --output reports/drift_report.html
```

생성:

```text
reports/
├── drift_report.html
└── drift_report.json
```

예상 출력:

```text
Drifted Columns: 3.0
Drift Share: 0.75
```

- `sepal_length`, `petal_length`, `petal_width`: Drift 감지
- `sepal_width`: 잡음만 추가했으므로 Drift 미감지

---

# 9. HTML Report 확인

WSL에서:

```bash
explorer.exe .
```

`reports` 폴더의 다음 파일을 브라우저로 열어 확인

```text
normal_drift_report.html
drift_report.html
```

확인할 내용:

```text
1. Reference Distribution
2. Current Distribution
3. Drifted Column 수
4. Drift Share
5. Feature별 Drift 여부
```

---

# 10. Drift Share 해석

예:

```text
전체 Feature = 4
Drift Feature = 3
```

이면:

```text
Drift Share
=
3 / 4
=
0.75
=
75%
```

실습에서는:

```python
drift_share=0.5
```

를 사용함

즉 전체 Feature 중 50% 이상에서 Drift가 발견되면  
Dataset 수준에서 Drift가 있다고 판단하는 기준으로 사용함

Feature별 Drift 판정 방식:

- 이번 실습 데이터(150행 수치형)는 Evidently 기본값에 따라 K-S 검정 사용
- p-value < 0.05이면 해당 Feature를 Drift로 판정

> 이 값은 교육 실습용 기준이며, 실제 운영에서는 Feature 중요도, 표본 크기, 오탐 비용 등을 고려해 별도의 정책을 설계해야 함

---

# 11. 결과 해석 시 주의점

```text
Data Drift
```

가 의미하는 것:

```text
입력 데이터의 분포가 과거와 달라졌음
```

의미하지 않는 것:

```text
Accuracy가 반드시 떨어졌음
```

또는:

```text
모델이 반드시 실패했음
```

따라서 Drift는:

```text
추가 확인이 필요한 Monitoring Signal
```

로 해석해야 함

---

# 12. 3차시 핵심

```text
Reference
   vs
Current
   ↓
Evidently
   ↓
Statistical Comparison
   ↓
Drift Report
```

Evidently는 Drift를 자동으로 평가하고  
사람이 해석할 수 있는 Report를 제공함

이 결과는 이후:

```text
Alert
Retraining 검토
Monitoring 정책
```

등의 입력으로 사용할 수 있음

---

# 13. 실습 정리

2차시:

```text
Reference
   ↓
Current Normal
   ↓
Current Drift
```

3차시:

```text
Reference
   vs
Current
   ↓
Evidently
   ↓
Drift Report
   ↓
Drift Share
```

이번 주의 핵심:

```text
모델 배포 이후에도
데이터의 상태를 지속적으로 관찰해야 함
```

---

# 14. 선택 과제

1. `sepal_width`에도 의도적으로 Drift를 추가함
2. `drift_share` 값을 0.25, 0.5, 0.75로 바꾸어 결과를 비교함
3. 정상 Report와 Drift Report의 Feature별 분포를 비교함
4. Drift가 발생한 Feature와 발생하지 않은 Feature를 구분함
5. “Data Drift가 있어도 재학습하지 않아도 되는 경우”를 한 가지 작성함
