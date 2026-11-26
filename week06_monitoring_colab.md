# 6단원 실습 — 모델 모니터링 및 유지보수 (Google Colab 버전)

> MLOps 방법론  
> 구성: 1차시 이론 + 2차시 실습 + 3차시 실습  
> 주제: Drift 시뮬레이션 → Evidently Drift Report 생성·해석 → Monitoring Gate  
> 실습 환경: Google Colab / Python 3.x  
> 실습 기준 Evidently: 0.7.23

---

# 0. 실습 개요

```text
2차시
Reference / Current 생성
→ 통계량 비교
→ Histogram / Boxplot
→ Strong / Weak Drift 비교

3차시
Reference vs Current
→ Evidently
→ Drift Report
→ Drift Share
→ Threshold 실험
→ Monitoring Gate
```

> Data Drift는 입력 분포 P(X)의 변화이며, Drift 발생이 곧 모델 성능 저하를 의미하지는 않음

---

# 1. 공통 환경 구성

## Cell 1 — 패키지 설치

```python
!pip -q install "evidently==0.7.23" "pandas<3" scikit-learn matplotlib
```

## Cell 2 — 공통 import 및 버전 확인

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evidently
from sklearn.datasets import load_iris

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

print("pandas    :", pd.__version__)
print("evidently:", evidently.__version__)
```

---

# 2차시 실습 — Drift 시뮬레이션과 분포 변화 관찰

## 실습 목표

- Reference Data와 Current Data의 차이 이해
- 정상 Current와 Drift Current 생성
- 평균·표준편차·사분위수 비교
- Histogram과 Boxplot으로 분포 변화 확인
- Strong / Weak Drift 비교
- Drift는 평균이 아니라 분포 전체의 변화라는 점 이해

## Cell 3 — Iris 데이터 준비

```python
iris = load_iris(as_frame=True)
df = iris.frame.copy()

df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
})

df.head()
```

## Cell 4 — Reference / Current Normal / Current Drift 생성

```python
rng = np.random.default_rng(42)

reference = df.copy()

current_normal = df.copy()
for col in FEATURES:
    current_normal[col] += rng.normal(0, 0.02, size=len(current_normal))

current_drift = df.copy()
current_drift["sepal_length"] += 1.5
current_drift["petal_length"] += 2.0
current_drift["petal_width"] *= 1.6

for col in FEATURES:
    current_drift[col] += rng.normal(0, 0.03, size=len(current_drift))

print("Reference     :", reference.shape)
print("Current Normal:", current_normal.shape)
print("Current Drift :", current_drift.shape)
```

의도적 Drift:

```text
sepal_length → +1.5
petal_length → +2.0
petal_width  → ×1.6
sepal_width  → 큰 변화 없음
```

## Cell 5 — 데이터 일부 비교

```python
display(reference[FEATURES].head())
display(current_normal[FEATURES].head())
display(current_drift[FEATURES].head())
```

## Cell 6 — 평균 비교

```python
mean_compare = pd.DataFrame({
    "Reference": reference[FEATURES].mean(),
    "Current Normal": current_normal[FEATURES].mean(),
    "Current Drift": current_drift[FEATURES].mean(),
}).round(3)

mean_compare
```

## Cell 7 — Mean Shift 계산

```python
mean_shift = pd.DataFrame({
    "Normal Shift": current_normal[FEATURES].mean() - reference[FEATURES].mean(),
    "Drift Shift": current_drift[FEATURES].mean() - reference[FEATURES].mean(),
}).round(3)

mean_shift
```

## Cell 8 — 상세 통계량 비교

```python
def summary_stats(data):
    return data[FEATURES].describe().loc[
        ["mean", "std", "25%", "50%", "75%"]
    ].round(3)

print("=== Reference ===")
display(summary_stats(reference))

print("=== Current Normal ===")
display(summary_stats(current_normal))

print("=== Current Drift ===")
display(summary_stats(current_drift))
```

> Drift는 평균 하나가 아니라 분포 전체의 변화로 이해해야 함

## Cell 9 — Reference vs Current Drift Histogram

```python
for feature in FEATURES:
    plt.figure(figsize=(7, 4))
    plt.hist(reference[feature], bins=20, alpha=0.6, label="Reference")
    plt.hist(current_drift[feature], bins=20, alpha=0.6, label="Current Drift")
    plt.title(f"Reference vs Current Drift: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.show()
```

## Cell 10 — Reference vs Current Normal Histogram

```python
for feature in FEATURES:
    plt.figure(figsize=(7, 4))
    plt.hist(reference[feature], bins=20, alpha=0.6, label="Reference")
    plt.hist(current_normal[feature], bins=20, alpha=0.6, label="Current Normal")
    plt.title(f"Reference vs Current Normal: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.show()
```

## Cell 11 — Boxplot

```python
for feature in FEATURES:
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [reference[feature], current_drift[feature]],
        tick_labels=["Reference", "Current Drift"],
    )
    plt.title(f"Boxplot: {feature}")
    plt.ylabel(feature)
    plt.show()
```

## Cell 12 — Weak Drift 생성

```python
rng_weak = np.random.default_rng(42)
current_weak_drift = df.copy()

current_weak_drift["sepal_length"] += 0.2
current_weak_drift["petal_length"] += 0.3
current_weak_drift["petal_width"] *= 1.1

for col in FEATURES:
    current_weak_drift[col] += rng_weak.normal(0, 0.03, size=len(current_weak_drift))

pd.DataFrame({
    "Strong Drift": current_drift[FEATURES].mean() - reference[FEATURES].mean(),
    "Weak Drift": current_weak_drift[FEATURES].mean() - reference[FEATURES].mean(),
}).round(3)
```

## Cell 13 — Strong / Weak Drift 시각화

```python
feature = "petal_length"

plt.figure(figsize=(7, 4))
plt.hist(reference[feature], bins=20, alpha=0.5, label="Reference")
plt.hist(current_weak_drift[feature], bins=20, alpha=0.5, label="Weak Drift")
plt.hist(current_drift[feature], bins=20, alpha=0.5, label="Strong Drift")
plt.title("Reference vs Weak / Strong Drift")
plt.xlabel(feature)
plt.ylabel("Count")
plt.legend()
plt.show()
```

## 2차시 핵심 정리

```text
Reference = 비교 기준
Current = 현재 운영 데이터
Data Drift = P(X)의 변화
```

> 눈으로 변화는 볼 수 있지만 실제 Monitoring에서는 자동 판단이 필요함

---

# 3차시 실습 — Evidently Drift Report와 Monitoring Gate

## 실습 목표

- Evidently Dataset 생성
- Normal / Drift Report 생성
- Colab 셀에서 Report 직접 확인
- HTML / JSON 결과 확인
- Drifted Feature 수와 Drift Share 추출
- Dataset-level 기준 변경 실험
- Monitoring Gate 구현

## Cell 14 — Evidently import

```python
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
```

## Cell 15 — Evidently Dataset 생성

```python
definition = DataDefinition(numerical_columns=FEATURES)

reference_dataset = Dataset.from_pandas(
    reference[FEATURES],
    data_definition=definition,
)

normal_dataset = Dataset.from_pandas(
    current_normal[FEATURES],
    data_definition=definition,
)

drift_dataset = Dataset.from_pandas(
    current_drift[FEATURES],
    data_definition=definition,
)

print("Evidently Dataset 생성 완료")
```

## Cell 16 — Current Normal 평가

```python
normal_report = Report([
    DataDriftPreset(
        columns=FEATURES,
        drift_share=0.5,
    )
])

normal_snapshot = normal_report.run(
    current_data=normal_dataset,
    reference_data=reference_dataset,
)

normal_snapshot
```

> Colab에서는 Snapshot 객체를 셀의 마지막 줄에 두면 Report를 Notebook 안에서 바로 확인할 수 있음

## Cell 17 — Current Drift 평가

```python
drift_report = Report([
    DataDriftPreset(
        columns=FEATURES,
        drift_share=0.5,
    )
])

drift_snapshot = drift_report.run(
    current_data=drift_dataset,
    reference_data=reference_dataset,
)

drift_snapshot
```

예상:

```text
sepal_length → Drift
petal_length → Drift
petal_width  → Drift
sepal_width  → 상대적으로 안정
```

## Cell 18 — HTML 저장

```python
normal_snapshot.save_html("normal_drift_report.html")
drift_snapshot.save_html("drift_report.html")

print("HTML Report 저장 완료")
```

선택 다운로드:

```python
from google.colab import files
# files.download("drift_report.html")
```

## Cell 19 — JSON 결과 확인

```python
drift_json = drift_snapshot.json()
print(drift_json[:2000])
```

```text
HTML / Notebook Report → 사람이 해석
JSON / Dictionary → 자동화 로직에서 사용
```

## Cell 20 — Drift Summary 추출 함수

```python
def get_drift_summary(snapshot):
    report_dict = snapshot.dict()

    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")

        if metric_name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            return {
                "count": value.get("count"),
                "share": value.get("share"),
            }

    return {"count": None, "share": None}


normal_summary = get_drift_summary(normal_snapshot)
drift_summary = get_drift_summary(drift_snapshot)

print("Normal:", normal_summary)
print("Drift :", drift_summary)
```

## Drift Share 해석

```text
전체 Feature = 4
Drift Feature = 3
Drift Share = 3/4 = 0.75
```

중요:

```text
drift_share=0.5
```

은 개별 Feature의 통계검정 threshold가 아니라,
전체 Feature 중 어느 정도가 Drift일 때 Dataset 수준에서 Drift로 볼지 정하는 정책 기준임.

## Cell 21 — drift_share 기준 실험

```python
threshold_results = []

for threshold in [0.25, 0.50, 0.75]:
    report = Report([
        DataDriftPreset(
            columns=FEATURES,
            drift_share=threshold,
        )
    ])

    snapshot = report.run(
        current_data=drift_dataset,
        reference_data=reference_dataset,
    )

    summary = get_drift_summary(snapshot)

    threshold_results.append({
        "drift_share_threshold": threshold,
        "drifted_columns": summary["count"],
        "actual_drift_share": summary["share"],
    })

pd.DataFrame(threshold_results)
```

## Cell 22 — Weak Drift Evidently 평가

```python
weak_drift_dataset = Dataset.from_pandas(
    current_weak_drift[FEATURES],
    data_definition=definition,
)

weak_report = Report([
    DataDriftPreset(
        columns=FEATURES,
        drift_share=0.5,
    )
])

weak_snapshot = weak_report.run(
    current_data=weak_drift_dataset,
    reference_data=reference_dataset,
)

weak_snapshot
```

## Cell 23 — Weak Drift Summary

```python
weak_summary = get_drift_summary(weak_snapshot)
weak_summary
```

## Cell 24 — Monitoring Gate

```python
WARNING_THRESHOLD = 0.5
current_share = drift_summary["share"]

print("Current Drift Share:", current_share)

if current_share is None:
    print("ERROR: Drift metric을 찾지 못함")
elif current_share >= WARNING_THRESHOLD:
    print("WARNING: Data Drift 감지")
    print("추가 분석 또는 Retraining 검토 필요")
else:
    print("PASS: Drift 수준이 허용 범위")
```

## Cell 25 — Normal Data와 비교

```python
normal_share = normal_summary["share"]

print("Normal Drift Share:", normal_share)

if normal_share is not None and normal_share >= WARNING_THRESHOLD:
    print("WARNING: Drift 감지")
else:
    print("PASS: Drift 수준이 허용 범위")
```

## 3차시 핵심 정리

```text
Reference vs Current
      ↓
Evidently
      ↓
Drift Report
      ↓
Drift Share
      ↓
Monitoring Gate
```

> Drift 감지 ≠ 모델 성능 저하 확정  
> Drift 감지 ≠ 즉시 재학습  
> Drift는 추가 분석이 필요한 Monitoring Signal

---

# 선택 확장 과제

1. `sepal_width`에도 Drift를 추가하고 결과 비교
2. Strong / Weak Drift의 Evidently 결과 비교
3. `drift_share`를 0.25 / 0.50 / 0.75로 변경
4. `WARNING_THRESHOLD`를 변경하여 Monitoring 정책 비교
5. Drift가 발생했지만 재학습하지 않아도 되는 사례 작성
