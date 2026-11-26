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

# 2차시 실습 — Data Drift 시뮬레이션과 분포 변화 관찰

## 실습 목표

- Reference Data와 Current Data의 차이를 이해함
- 정상 Current Data와 Drift Current Data를 각각 생성함
- 일부 Feature의 분포를 의도적으로 변화시킴
- 평균, 표준편차, 분위수를 비교하여 Drift를 수치로 확인함
- Histogram과 Boxplot으로 분포 변화를 시각적으로 확인함
- Drift 강도를 변경하면서 분포 변화가 어떻게 달라지는지 관찰함

이번 차시의 전체 흐름:

```text
Reference Data
      ↓
Current Normal 생성
      ↓
Current Drift 생성
      ↓
통계량 비교
      ↓
Histogram 비교
      ↓
Boxplot 비교
      ↓
Drift 강도 변경
      ↓
분포 변화 재확인
```

핵심 질문:

> “운영 데이터의 분포가 학습 당시 데이터와 달라졌다는 것을 어떻게 확인할 수 있는가?”

---

# 2. 실습 데이터 생성

`src/make_data.py`

```python
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris


REFERENCE_PATH = Path(
    "data/reference/reference.csv"
)

CURRENT_NORMAL_PATH = Path(
    "data/current/current_normal.csv"
)

CURRENT_DRIFT_PATH = Path(
    "data/current/current_drift.csv"
)


def main():

    rng = np.random.default_rng(42)

    iris = load_iris(
        as_frame=True
    )

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
            + rng.normal(
                0,
                0.02,
                size=len(current_normal),
            )
        )

    # 3. Current Drift
    current_drift = df.copy()

    current_drift["sepal_length"] += 1.5
    current_drift["petal_length"] += 2.0
    current_drift["petal_width"] *= 1.6

    for col in feature_columns:
        current_drift[col] = (
            current_drift[col]
            + rng.normal(
                0,
                0.03,
                size=len(current_drift),
            )
        )

    REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CURRENT_NORMAL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CURRENT_DRIFT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    current_normal.to_csv(
        CURRENT_NORMAL_PATH,
        index=False,
    )

    current_drift.to_csv(
        CURRENT_DRIFT_PATH,
        index=False,
    )

    print("데이터 생성 완료")
    print("- Reference     :", REFERENCE_PATH)
    print("- Current Normal:", CURRENT_NORMAL_PATH)
    print("- Current Drift :", CURRENT_DRIFT_PATH)


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

# 3. 생성된 데이터 일부 확인

Reference:

```bash
head data/reference/reference.csv
```

Current Normal:

```bash
head data/current/current_normal.csv
```

Current Drift:

```bash
head data/current/current_drift.csv
```

확인할 내용:

```text
Reference
→ 원본 데이터

Current Normal
→ 아주 작은 Noise만 추가

Current Drift
→ 일부 Feature에 의도적인 분포 변화 추가
```

---

# 4. 평균값 비교

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

확인:

```text
Reference
vs
Current Normal
→ 평균이 거의 비슷함

Reference
vs
Current Drift
→ 일부 Feature의 평균이 크게 달라짐
```

---

# 5. 평균뿐 아니라 표준편차와 분위수 비교

평균만 비교하면 분포 전체의 변화를 놓칠 수 있음.

`src/compare_detail.py`

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


def summarize(name, df):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    summary = (
        df[FEATURES]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "25%",
                "50%",
                "75%",
            ]
        ]
        .round(3)
    )

    print(summary)


summarize(
    "REFERENCE",
    reference,
)

summarize(
    "CURRENT NORMAL",
    normal,
)

summarize(
    "CURRENT DRIFT",
    drift,
)
```

실행:

```bash
python src/compare_detail.py
```

확인할 내용:

```text
mean
→ 중심 위치 변화

std
→ 데이터 퍼짐 정도 변화

25%, 50%, 75%
→ 분포 위치 변화
```

핵심:

> Data Drift는 평균 하나의 변화만 보는 것이 아니라 분포 전체가 달라졌는지 확인해야 함

---

# 6. 평균 변화량 계산

`src/mean_shift.py`

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


reference_mean = (
    reference[FEATURES]
    .mean()
)

normal_shift = (
    normal[FEATURES].mean()
    - reference_mean
)

drift_shift = (
    drift[FEATURES].mean()
    - reference_mean
)


result = pd.DataFrame(
    {
        "normal_mean_shift": normal_shift,
        "drift_mean_shift": drift_shift,
    }
)

print(
    result.round(3)
)
```

실행:

```bash
python src/mean_shift.py
```

확인:

```text
normal_mean_shift
→ 대부분 0에 가까움

drift_mean_shift
→ 일부 Feature에서 크게 나타남
```

---

# 7. Histogram으로 분포 비교

시각화를 위해 matplotlib 설치:

```bash
pip install matplotlib
```

`src/plot_hist.py`

```python
from pathlib import Path

import matplotlib.pyplot as plt
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


Path(
    "reports/plots"
).mkdir(
    parents=True,
    exist_ok=True,
)


for feature in FEATURES:

    plt.figure()

    plt.hist(
        reference[feature],
        bins=20,
        alpha=0.5,
        label="Reference",
    )

    plt.hist(
        normal[feature],
        bins=20,
        alpha=0.5,
        label="Current Normal",
    )

    plt.title(
        f"{feature}: Reference vs Normal"
    )

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"reports/plots/{feature}_normal.png"
    )

    plt.close()


for feature in FEATURES:

    plt.figure()

    plt.hist(
        reference[feature],
        bins=20,
        alpha=0.5,
        label="Reference",
    )

    plt.hist(
        drift[feature],
        bins=20,
        alpha=0.5,
        label="Current Drift",
    )

    plt.title(
        f"{feature}: Reference vs Drift"
    )

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"reports/plots/{feature}_drift.png"
    )

    plt.close()


print(
    "Histogram 생성 완료"
)
```

실행:

```bash
python src/plot_hist.py
```

확인:

```bash
ls reports/plots
```

Windows에서 폴더 열기:

```bash
explorer.exe reports/plots
```

확인할 내용:

```text
Reference vs Normal
→ Histogram이 거의 겹침

Reference vs Drift
→ 일부 Feature의 분포가 오른쪽 또는 넓은 방향으로 이동
```

---

# 8. Boxplot으로 분포 변화 확인

`src/plot_box.py`

```python
from pathlib import Path

import matplotlib.pyplot as plt
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

drift = pd.read_csv(
    "data/current/current_drift.csv"
)


Path(
    "reports/plots"
).mkdir(
    parents=True,
    exist_ok=True,
)


for feature in FEATURES:

    plt.figure()

    plt.boxplot(
        [
            reference[feature],
            drift[feature],
        ],
        labels=[
            "Reference",
            "Current Drift",
        ],
    )

    plt.title(
        f"{feature} distribution"
    )

    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        f"reports/plots/{feature}_box.png"
    )

    plt.close()


print(
    "Boxplot 생성 완료"
)
```

실행:

```bash
python src/plot_box.py
```

확인:

```text
Reference
vs
Current Drift
```

의 중앙값, 범위, 분포 위치가 어떻게 달라졌는지 관찰함.

---

# 9. Drift 강도 변경 실험

이제 Drift 강도를 직접 바꾸어 봄.

`src/make_data.py`의 다음 부분:

```python
current_drift["sepal_length"] += 1.5
current_drift["petal_length"] += 2.0
current_drift["petal_width"] *= 1.6
```

을 약한 Drift로 변경:

```python
current_drift["sepal_length"] += 0.2
current_drift["petal_length"] += 0.3
current_drift["petal_width"] *= 1.1
```

다시 실행:

```bash
python src/make_data.py
python src/compare_detail.py
python src/plot_hist.py
```

확인:

```text
강한 Drift
→ 육안으로 분포 차이가 큼

약한 Drift
→ 분포 차이가 작아짐
```

핵심:

> Drift는 항상 명확하게 보이는 큰 변화만 발생하는 것이 아니며, 실제 운영에서는 작은 변화도 통계적으로 감지해야 함

실습 종료 후 원래 강한 Drift 값으로 복구:

```python
current_drift["sepal_length"] += 1.5
current_drift["petal_length"] += 2.0
current_drift["petal_width"] *= 1.6
```

그리고 다시:

```bash
python src/make_data.py
```

---

# 10. 2차시 핵심 정리

```text
Reference Data
=
비교 기준

Current Data
=
현재 운영 데이터
```

이번 실습에서는:

```text
Reference
   ↓
Current Normal
   ↓
Current Drift
   ↓
통계량 비교
   ↓
Histogram
   ↓
Boxplot
   ↓
Drift 강도 변경
```

을 통해 Data Drift를 직접 확인함.

핵심:

> Data Drift는 입력 데이터 분포 P(X)가 과거와 달라지는 현상임

> 평균뿐 아니라 표준편차, 분위수, Histogram 등 분포 전체를 함께 확인해야 함

---

# 3차시 실습 — Evidently Drift Report 생성·해석과 Monitoring Gate

## 실습 목표

- Evidently의 기본 구조를 이해함
- `DataDriftPreset`으로 Drift Report를 생성함
- 정상 Current와 Drift Current를 각각 비교함
- HTML Report에서 Feature별 분포 변화를 확인함
- Drift된 Feature 수와 Drift Share를 해석함
- `drift_share` 기준을 변경하여 Dataset Drift 판단이 달라지는지 확인함
- JSON 결과에서 Feature별 Drift 정보를 확인함
- Drift 결과를 이용한 간단한 Monitoring Gate를 구성함

이번 차시의 전체 흐름:

```text
Reference
    vs
Current
   ↓
Evidently
   ↓
Feature Drift 판정
   ↓
Drifted Columns
   ↓
Drift Share
   ↓
Dataset Drift
   ↓
Monitoring Gate
```

핵심 질문:

> “분포 변화가 어느 정도 발생했을 때 운영상 경고를 발생시킬 것인가?”

---

# 11. Evidently 기본 구조

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

# 12. Drift Report 코드 작성

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

from evidently.presets import (
    DataDriftPreset,
)


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

    parser.add_argument(
        "--drift-share",
        type=float,
        default=0.5,
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
                drift_share=args.drift_share,
            )
        ]
    )

    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(
        str(output_path)
    )

    json_path = (
        output_path
        .with_suffix(".json")
    )

    json_path.write_text(
        snapshot.json(),
        encoding="utf-8",
    )

    print(
        "HTML report:",
        output_path,
    )

    print(
        "JSON report:",
        json_path,
    )

    print(
        "Dataset drift threshold:",
        args.drift_share,
    )

    report_dict = (
        snapshot.dict()
    )

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

# 13. 정상 Current Data 평가

```bash
python src/drift_report.py \
  --current data/current/current_normal.csv \
  --output reports/normal_drift_report.html \
  --drift-share 0.5
```

예상:

```text
Drifted Columns: 0.0
Drift Share: 0.0
```

생성:

```text
reports/
├── normal_drift_report.html
└── normal_drift_report.json
```

---

# 14. Drift Current Data 평가

```bash
python src/drift_report.py \
  --current data/current/current_drift.csv \
  --output reports/drift_report.html \
  --drift-share 0.5
```

예상:

```text
Drifted Columns: 3.0
Drift Share: 0.75
```

이번 실습 데이터:

```text
sepal_length
→ Drift

sepal_width
→ Drift 없음

petal_length
→ Drift

petal_width
→ Drift
```

---

# 15. 정상 Report와 Drift Report 비교

Windows에서:

```bash
explorer.exe reports
```

다음 두 파일을 브라우저에서 각각 열어 비교:

```text
normal_drift_report.html

drift_report.html
```

확인할 내용:

```text
1. Reference Distribution
2. Current Distribution
3. Feature별 Drift 여부
4. Drifted Column 수
5. Drift Share
```

비교:

```text
Normal Current
→ Reference와 거의 비슷

Drift Current
→ 일부 Feature 분포 차이가 명확
```

---

# 16. Drift Share 계산 이해

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

즉 Feature의 75%에서 Drift가 감지되었다는 의미임.

---

# 17. Dataset Drift 기준 변경 실험

현재:

```python
drift_share=0.5
```

의 의미:

```text
전체 Feature 중
50% 이상에서 Drift 발생
→ Dataset Drift로 판단
```

이번에는 기준을 변경하여 비교함.

## 기준 1: 0.25

```bash
python src/drift_report.py \
  --current data/current/current_drift.csv \
  --output reports/drift_share_025.html \
  --drift-share 0.25
```

## 기준 2: 0.50

```bash
python src/drift_report.py \
  --current data/current/current_drift.csv \
  --output reports/drift_share_050.html \
  --drift-share 0.50
```

## 기준 3: 0.75

```bash
python src/drift_report.py \
  --current data/current/current_drift.csv \
  --output reports/drift_share_075.html \
  --drift-share 0.75
```

확인:

```text
같은 Feature Drift 결과라도
Dataset Drift 판정 기준은 정책에 따라 달라질 수 있음
```

핵심:

> drift_share는 Feature별 Drift 검정 기준이 아니라 Dataset 전체를 Drift 상태로 볼 것인지 결정하는 정책 기준임

---

# 18. JSON Report 확인

HTML은 사람이 보기 좋지만 자동화에서는 JSON 결과가 중요함.

확인:

```bash
python -m json.tool \
  reports/drift_report.json \
  | head -100
```

또는:

```bash
less reports/drift_report.json
```

확인할 내용:

```text
metrics
drift
count
share
```

즉:

```text
HTML
→ 사람이 분석

JSON
→ 프로그램이 결과 사용
```

---

# 19. Drift Summary 추출 코드 작성

`src/drift_summary.py`

```python
import json


REPORT_PATH = (
    "reports/drift_report.json"
)


with open(
    REPORT_PATH,
    encoding="utf-8",
) as f:

    report = json.load(f)


for metric in report.get(
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

        print(
            "Drifted Columns:",
            value.get("count"),
        )

        print(
            "Drift Share:",
            value.get("share"),
        )
```

실행:

```bash
python src/drift_summary.py
```

핵심:

```text
Evidently JSON
     ↓
Python 프로그램
     ↓
Monitoring 결과 자동 활용 가능
```

---

# 20. Monitoring Gate 작성

Drift Share가 일정 기준 이상이면 경고를 발생시키는 간단한 Gate를 작성함.

`src/drift_gate.py`

```python
import json
import sys


REPORT_PATH = (
    "reports/drift_report.json"
)

WARNING_THRESHOLD = 0.5


with open(
    REPORT_PATH,
    encoding="utf-8",
) as f:

    report = json.load(f)


drift_share = None


for metric in report.get(
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

        drift_share = value.get(
            "share"
        )

        break


if drift_share is None:

    print(
        "ERROR: Drift Share를 찾을 수 없음"
    )

    sys.exit(1)


print(
    f"Drift Share: {drift_share:.2f}"
)

print(
    f"Warning Threshold: {WARNING_THRESHOLD:.2f}"
)


if drift_share >= WARNING_THRESHOLD:

    print(
        "WARNING: Data Drift 감지"
    )

else:

    print(
        "PASS: Drift 기준 미만"
    )
```

실행:

```bash
python src/drift_gate.py
```

예상:

```text
Drift Share: 0.75
Warning Threshold: 0.50
WARNING: Data Drift 감지
```

중요:

> 이번 실습의 Monitoring Gate는 경고를 발생시키는 예시이며, Data Drift가 발생했다고 해서 바로 모델 재학습을 실행하는 것은 아님

---

# 21. 정상 데이터에 대한 Gate 확인

정상 Current Data에 대한 Report를 사용하도록 경로를 변경:

```python
REPORT_PATH = (
    "reports/normal_drift_report.json"
)
```

다시 실행:

```bash
python src/drift_gate.py
```

예상:

```text
Drift Share: 0.00
Warning Threshold: 0.50
PASS: Drift 기준 미만
```

비교:

```text
Normal Current
→ PASS

Drift Current
→ WARNING
```

---

# 22. Data Drift와 모델 성능 저하 구분

이번 실습에서 확인한 것:

```text
Data Drift
=
입력 데이터의 분포 변화
```

하지만 이것만으로는:

```text
Accuracy가 떨어졌는가?
```

를 알 수 없음.

따라서 운영에서는:

```text
Data Drift
   ↓
Monitoring Signal
   ↓
추가 성능 평가
   ↓
재학습 필요 여부 판단
```

흐름이 필요함.

---

# 23. Feature별 Drift와 Dataset Drift 구분

```text
Feature Drift
→ 개별 Feature의 분포 변화 여부
```

```text
Dataset Drift
→ 전체 Feature 중 Drift된 Feature 비율이
   정책 기준을 넘었는지 판단
```

예:

```text
4개 Feature 중
3개 Drift
→ Drift Share = 0.75
```

`drift_share=0.5`이면:

```text
0.75 ≥ 0.5
→ Dataset Drift
```

---

# 24. 3차시 핵심 정리

```text
Reference
   vs
Current
   ↓
Evidently
   ↓
Feature별 통계 검정
   ↓
Drift 여부
   ↓
Drifted Columns
   ↓
Drift Share
   ↓
Dataset Drift
   ↓
Monitoring Gate
```

역할 정리:

```text
Evidently HTML
→ 사람이 결과 확인

Evidently JSON
→ 프로그램이 결과 활용

Drift Share
→ Drift된 Feature 비율

Monitoring Gate
→ 운영 정책 기준에 따라 경고
```

핵심:

> Data Drift는 모델 실패의 확정 신호가 아니라 추가 확인이 필요한 Monitoring Signal임

> Monitoring 결과를 JSON으로 저장하면 이후 Alert, Pipeline, Retraining 판단 등에 연결할 수 있음

---


# 13. 실습 정리

2차시:

```text
Reference
   ↓
Current Normal / Current Drift
   ↓
통계량 비교
   ↓
Histogram / Boxplot
   ↓
Drift 강도 변화 관찰
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
   ↓
Threshold 비교
   ↓
Monitoring Gate
```

이번 주의 핵심:

```text
모델 배포 이후에도
데이터의 상태를 지속적으로 관찰해야 함
```

---

# 14. 선택 과제

1. `sepal_width`에도 의도적으로 Drift를 추가하고 Drift Share 변화를 확인함
2. `WARNING_THRESHOLD`를 0.25, 0.5, 0.75로 바꾸어 Monitoring Gate 결과를 비교함
3. Drift 강도를 약하게 조정한 뒤 Evidently가 어떤 Feature를 Drift로 판단하는지 확인함
4. 정상 Report와 Drift Report의 Feature별 분포를 비교함
5. “Data Drift가 있어도 즉시 재학습하지 않아도 되는 경우”를 한 가지 작성함
