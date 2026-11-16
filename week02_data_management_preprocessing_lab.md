# 2주차 실습 — 데이터 관리 및 전처리

> MLOps 방법론  
> 주제: Pandas 기반 데이터 전처리 → DVC 데이터 버전관리 → 데이터 품질 검증  
> 실습 환경: Windows 10/11 + WSL2 Ubuntu 22.04 / Python 3.10 이상 / Git  
> 클라우드 계정: 필요 없음

---

## 0. 실습 목표

이번 주 실습: 하나의 작은 데이터셋을 2~4차시 동안 계속 사용

실습 후 직접 수행할 흐름:

```text
원본 데이터
    ↓
Pandas 전처리
    ↓
학습 가능한 데이터셋
    ↓
DVC 버전관리
    ↓
데이터 품질 검증
```

핵심 목표: Pandas 문법 학습을 넘어 다음 작업 수행

- 원본 데이터와 가공 데이터를 구분함
- 같은 전처리를 반복 실행할 수 있도록 코드화함
- 데이터도 코드처럼 버전을 관리함
- 모델 학습 전에 데이터 품질을 자동으로 검사함

---

# 1. 공통 실습 환경 구성

## 1.0 WSL2 Ubuntu 설치 (Windows 최초 1회)

2~7주차 실습 명령은 모두 Linux 셸(bash) 기준임 → Windows에서는 WSL2(Windows Subsystem for Linux)로 Ubuntu를 설치하여 사용

- 이미 WSL2 Ubuntu가 설치되어 있으면 1.0.5부터 진행
- 4주차 Docker Desktop도 WSL2 기반으로 동작하므로 이번에 함께 준비

### 1.0.1 설치 조건 확인

- Windows 10 버전 2004(빌드 19041) 이상 또는 Windows 11
- BIOS/UEFI에서 CPU 가상화 기능 활성화 (Intel VT-x 또는 AMD-V/SVM)
- 확인 방법: `작업 관리자` → `성능` → `CPU` → `가상화: 사용`

### 1.0.2 Ubuntu 설치

`시작` 메뉴에서 **PowerShell**을 마우스 오른쪽 버튼으로 클릭 → **관리자 권한으로 실행**

```powershell
wsl --install -d Ubuntu-22.04
```

- WSL2 기능과 Ubuntu 22.04를 함께 설치함
- `-d Ubuntu-22.04` 없이 `wsl --install`만 실행하면 최신 Ubuntu가 설치됨 → 실습 자료 기준과 맞추기 위해 버전을 지정함
- 설치 완료 후 **Windows 재부팅** 필요

### 1.0.3 Ubuntu 사용자 계정 생성

재부팅 후 Ubuntu 창이 자동으로 열림 (열리지 않으면 `시작` 메뉴에서 `Ubuntu 22.04` 실행)

- Linux 사용자 이름 입력 (영문 소문자 권장, 예: `student`)
- 비밀번호 입력 및 확인
- Windows 계정과 별개의 계정임
- 비밀번호 입력 시 화면에 아무것도 표시되지 않는 것이 정상
- 이 비밀번호는 이후 `sudo` 명령 실행 시 사용하므로 반드시 기억

### 1.0.4 설치 확인

PowerShell에서:

```powershell
wsl -l -v
```

출력 예:

```text
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

`VERSION`이 `2`인지 확인

### 1.0.5 Ubuntu 기본 패키지 설치

Ubuntu 터미널에서:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git
```

- Ubuntu 22.04는 기본으로 `python3-venv`가 설치되어 있지 않음 → 설치하지 않으면 1.3의 `python3 -m venv` 실행 시 `ensurepip is not available` 오류 발생

확인:

```bash
python3 --version
git --version
```

### 1.0.6 Git 사용자 정보 설정

3차시에서 `git commit`을 실행하므로 미리 설정함 → 설정하지 않으면 commit 단계에서 오류 발생

```bash
git config --global user.name "홍길동"
git config --global user.email "student@example.com"
```

### 1.0.7 작업 위치 주의

- 실습 폴더는 Ubuntu 홈 디렉터리(`~`) 아래에 생성함
- `/mnt/c/...` (Windows C 드라이브)에서 작업하면 파일 입출력이 크게 느려지고 권한 문제가 생길 수 있음
- Windows 탐색기에서 Ubuntu 폴더 열기: Ubuntu 터미널에서 `explorer.exe .` 실행
- 또는 탐색기 주소창에 `\\wsl$\Ubuntu-22.04\home\<사용자이름>` 입력

### 1.0.8 자주 발생하는 오류

| 증상 | 원인 및 해결 |
|---|---|
| `0x80370102` 오류 | CPU 가상화 비활성화 → BIOS/UEFI에서 VT-x 또는 AMD-V(SVM) 활성화 |
| `wsl -l -v`의 VERSION이 `1` | PowerShell에서 `wsl --set-version Ubuntu-22.04 2` 실행 |
| WSL 업데이트 필요 안내 | PowerShell에서 `wsl --update` 실행 |
| `python3 -m venv` 실패 | 1.0.5의 `python3-venv` 설치 여부 확인 |

> 선택: VS Code를 사용할 경우 **WSL** 확장을 설치하면 Ubuntu 터미널에서 `code .`으로 실습 폴더를 바로 열 수 있음

## 1.1 프로젝트 디렉터리 생성

WSL Ubuntu 터미널에서 다음 명령 실행:

```bash
mkdir -p ~/mlops-week02
cd ~/mlops-week02
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

터미널 앞에 `(.venv)` 표시 여부 확인

`.gitignore` 작성:

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
EOF
```

> Python 3.12 이하의 `venv`는 자체 `.gitignore`를 만들지 않음 → `git add .` 실행 시 `.venv` 전체가 커밋되므로 반드시 먼저 작성

## 1.4 필요한 패키지 설치

```bash
python -m pip install --upgrade pip
pip install pandas numpy dvc
```

설치 확인:

```bash
python --version
git --version
dvc --version
```

## 1.5 폴더 구조 생성

```bash
mkdir -p data/raw data/processed src reports
```

현재 구조:

```text
mlops-week02/
├── data/
│   ├── raw/
│   └── processed/
├── src/
└── reports/
```

---

# 2차시 실습 — Pandas 기반 데이터 수집·정제·변환

## 2.1 실습 목표

- CSV 파일을 Pandas DataFrame으로 불러옴
- 결측치, 중복값, 비정상 범위 값을 확인함
- 데이터를 정제하고 새로운 파생변수를 생성함
- 전처리 결과를 별도의 파일로 저장함
- 전처리 작업을 Python 코드로 재현 가능하게 만듦

---

## 2.2 실습용 원본 데이터 생성

다음 명령으로 실습용 CSV 파일 생성:

```bash
cat > data/raw/students_raw.csv <<'EOF'
student_id,study_hours,attendance,assignment_score,midterm_score
S001,3.5,92,85,78
S002,5.0,88,90,84
S003,2.0,75,72,68
S004,7.0,95,96,91
S005,,83,80,76
S006,4.0,,88,82
S007,18.0,90,93,89
S008,1.5,65,60,58
S009,6.0,102,95,92
S010,4.5,87,,81
S010,4.5,87,,81
S011,8.0,97,98,95
EOF
```

파일 확인:

```bash
cat data/raw/students_raw.csv
```

데이터에 의도적으로 포함한 문제:

- `study_hours` 결측치
- `attendance` 결측치
- `assignment_score` 결측치
- `student_id=S010` 중복
- `study_hours=18` 비정상적으로 큰 값
- `attendance=102` 허용 범위를 벗어난 값

---

## 2.3 Pandas로 원본 데이터 확인

`src/inspect_data.py` 파일 생성:

```bash
cat > src/inspect_data.py <<'PY'
import pandas as pd

DATA_PATH = "data/raw/students_raw.csv"

df = pd.read_csv(DATA_PATH)

print("=== 데이터 앞부분 ===")
print(df.head())

print("\n=== 데이터 크기 ===")
print(df.shape)

print("\n=== 컬럼과 자료형 ===")
print(df.dtypes)

print("\n=== 결측치 개수 ===")
print(df.isna().sum())

print("\n=== 중복 student_id 개수 ===")
print(df.duplicated(subset=["student_id"]).sum())

print("\n=== 기초 통계량 ===")
print(df.describe())
PY
```

실행:

```bash
python src/inspect_data.py
```

### 확인할 내용

출력 결과에서 직접 확인할 내용:

1. 전체 행과 열의 수는 몇 개인가?
2. 어떤 컬럼에 결측치가 있는가?
3. 중복된 학생 ID가 있는가?
4. `attendance`의 최대값이 정상 범위인가?
5. `study_hours`의 최대값은 다른 값들과 비교해 지나치게 크지 않은가?

---

## 2.4 데이터 전처리 코드 작성

`src/preprocess.py` 생성:

```bash
cat > src/preprocess.py <<'PY'
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/students_raw.csv")
OUTPUT_PATH = Path("data/processed/students_clean.csv")

# 1. 데이터 수집
df = pd.read_csv(RAW_PATH)

print(f"[1] 원본 데이터: {df.shape[0]}행 × {df.shape[1]}열")

# 2. 중복 제거
before = len(df)
df = df.drop_duplicates(subset=["student_id"], keep="first")
print(f"[2] 중복 제거: {before - len(df)}건 제거")

# 3. 숫자형 컬럼 정의
numeric_cols = [
    "study_hours",
    "attendance",
    "assignment_score",
    "midterm_score",
]

# 숫자로 변환할 수 없는 값은 NaN으로 처리
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. 결측치 처리
# 실습에서는 단순화를 위해 각 컬럼의 중앙값으로 대체
for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# 5. 허용 범위를 벗어난 값 처리
# 실제 업무에서는 도메인 규칙과 데이터 담당자의 확인이 필요함
df["study_hours"] = df["study_hours"].clip(lower=0, upper=12)
df["attendance"] = df["attendance"].clip(lower=0, upper=100)
df["assignment_score"] = df["assignment_score"].clip(lower=0, upper=100)
df["midterm_score"] = df["midterm_score"].clip(lower=0, upper=100)

# 6. 파생변수 생성
df["avg_score"] = (
    df["assignment_score"] * 0.4
    + df["midterm_score"] * 0.6
).round(2)

# 간단한 학습위험 예시
df["risk_flag"] = (
    (df["attendance"] < 80) | (df["avg_score"] < 70)
).astype(int)

# 7. 정렬 및 저장
df = df.sort_values("student_id").reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[3] 전처리 완료: {df.shape[0]}행 × {df.shape[1]}열")
print(f"[4] 저장 위치: {OUTPUT_PATH}")
print("\n=== 전처리 결과 ===")
print(df)
PY
```

실행:

```bash
python src/preprocess.py
```

결과 파일 확인:

```bash
cat data/processed/students_clean.csv
```

---

## 2.5 전처리 전·후 비교

다음 명령 실행:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("=== RAW ===")
print("shape:", raw.shape)
print("missing:", raw.isna().sum().sum())
print("duplicates:", raw.duplicated(subset=["student_id"]).sum())

print("\n=== CLEAN ===")
print("shape:", clean.shape)
print("missing:", clean.isna().sum().sum())
print("duplicates:", clean.duplicated(subset=["student_id"]).sum())
PY
```

### 핵심 확인

```text
원본 데이터 → 문제 발견 → 전처리 규칙 적용 → 가공 데이터 생성
```

MLOps에서 중요한 것은 데이터 자체를 한 번 수정하는 것이 아니라  
**같은 입력에 같은 전처리 규칙을 반복 적용할 수 있도록 코드로 관리**

---

# 3차시 실습 — DVC 기반 데이터 버전관리

## 3.1 실습 목표

- Git과 DVC의 역할 차이를 이해함
- 대용량 데이터 파일을 Git 저장소와 분리하여 관리함
- 데이터 버전을 코드 버전과 연결함
- 과거 데이터 버전으로 되돌아가는 과정을 확인함

---

## 3.2 DVC 초기화

프로젝트 루트에서 실행:

```bash
dvc init
```

확인:

```bash
ls -a
```

`.dvc/` 디렉터리 생성 여부 확인

---

## 3.3 원본 데이터 DVC로 관리

```bash
dvc add data/raw/students_raw.csv
```

확인:

```bash
ls data/raw
cat data/raw/students_raw.csv.dvc
```

DVC는 원본 CSV 대신 `students_raw.csv.dvc` 메타데이터 파일을 생성

Git 상태 확인:

```bash
git status
```

---

## 3.4 로컬 DVC Remote 구성

이번 실습의 DVC 저장소: 클라우드 스토리지 대신 프로젝트 밖의 로컬 폴더 사용

```bash
mkdir -p ../dvc-storage
dvc remote add -d localstorage ../dvc-storage
```

설정 확인:

```bash
dvc remote list
```

데이터 저장:

```bash
dvc push
```

### 구조 이해

```text
Git
 ├─ Python 코드
 ├─ .dvc 파일
 └─ DVC 설정

DVC Remote
 └─ 실제 데이터 파일 내용
```

- Git: **코드와 데이터의 위치·버전 정보** 관리
- DVC: **실제 데이터 파일**을 별도 저장소에 보관

---

## 3.5 첫 번째 데이터 버전 Git에 기록

```bash
git add .
git commit -m "week02: data version 1"
git tag data-v1
```

확인:

```bash
git log --oneline --decorate
```

---

## 3.6 데이터 변경

새로운 학생 데이터가 수집된 상황 가정

```bash
cat >> data/raw/students_raw.csv <<'EOF'
S012,5.5,94,91,88
EOF
```

원본 데이터 마지막 부분 확인:

```bash
tail data/raw/students_raw.csv
```

DVC 상태 확인:

```bash
dvc status
```

---

## 3.7 변경된 데이터 다시 버전관리

```bash
dvc add data/raw/students_raw.csv
dvc push
```

전처리 재실행:

```bash
python src/preprocess.py
```

두 번째 버전을 Git에 기록:

```bash
git add .
git commit -m "week02: data version 2"
git tag data-v2
```

확인:

```bash
git log --oneline --decorate
```

---

## 3.8 과거 데이터 버전으로 이동

현재 행 수 확인:

```bash
wc -l data/raw/students_raw.csv
```

첫 번째 버전으로 이동:

```bash
git checkout data-v1
dvc checkout
```

다시 확인:

```bash
tail data/raw/students_raw.csv
```

`S012`가 없는지 확인

두 번째 버전으로 이동:

```bash
git checkout data-v2
dvc checkout
```

다시 확인:

```bash
tail data/raw/students_raw.csv
```

`S012`가 다시 나타나는지 확인

실습을 계속하기 위해 main 브랜치로 복귀

```bash
git checkout main
dvc checkout
```

---

## 3.9 Git과 DVC의 역할 정리

| 관리 대상 | Git | DVC |
|---|---:|---:|
| Python 코드 | O | - |
| 설정 파일 | O | - |
| `.dvc` 메타데이터 | O | - |
| 대용량 데이터 | 직접 관리 비권장 | O |
| 데이터 버전과 코드 버전 연결 | O | O |
| 로컬/원격 데이터 저장소 | - | O |

핵심 정리:

```text
Git commit
   ↕ 연결
DVC data version
```

코드만 과거 버전으로 돌아가는 것이 아니라  
**코드가 사용한 데이터 버전까지 함께 복원할 수 있어야 재현성 확보**

---

# 4차시 실습 — 데이터 품질 검증

## 4.1 실습 목표

- 모델 학습 전에 데이터 품질을 검사해야 하는 이유를 이해함
- 스키마, 결측치, 중복, 허용 범위, 이상치를 자동으로 검사함
- 품질 검증 실패 시 후속 파이프라인을 중단할 수 있는 형태를 구현함

---

## 4.2 품질 검증 규칙

이번 실습의 적용 규칙:

| 항목 | 검증 규칙 |
|---|---|
| 필수 컬럼 | 정해진 컬럼이 모두 존재해야 함 |
| 학생 ID | 결측치와 중복이 없어야 함 |
| 숫자형 데이터 | 숫자로 해석 가능해야 함 |
| 출석률 | 0~100 |
| 과제/중간 점수 | 0~100 |
| 학습시간 | 0~12 |
| 전체 결측치 | 없어야 함 |
| 이상치 | IQR 기준으로 확인 |

---

## 4.3 품질 검증 프로그램 작성

`src/quality_check.py` 생성:

```bash
cat > src/quality_check.py <<'PY'
import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "student_id",
    "study_hours",
    "attendance",
    "assignment_score",
    "midterm_score",
    "avg_score",
    "risk_flag",
]

RANGE_RULES = {
    "study_hours": (0, 12),
    "attendance": (0, 100),
    "assignment_score": (0, 100),
    "midterm_score": (0, 100),
    "avg_score": (0, 100),
    "risk_flag": (0, 1),
}

def check_iqr_outliers(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]
    return outliers, lower, upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/processed/students_clean.csv",
    )
    args = parser.parse_args()

    path = Path(args.csv_path)

    print(f"=== DATA QUALITY CHECK: {path} ===")

    if not path.exists():
        print("[FAIL] 파일이 존재하지 않습니다.")
        sys.exit(1)

    df = pd.read_csv(path)
    errors = []

    # 1. 스키마 확인
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        errors.append(f"필수 컬럼 누락: {missing_columns}")
    else:
        print("[PASS] 필수 컬럼")

    # 이후 검사는 필요한 컬럼이 있을 때만 수행
    if "student_id" in df.columns:
        if df["student_id"].isna().any():
            errors.append("student_id 결측치 존재")

        duplicated = df["student_id"].duplicated().sum()
        if duplicated > 0:
            errors.append(f"student_id 중복 {duplicated}건")
        else:
            print("[PASS] student_id 중복 없음")

    # 2. 전체 결측치
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        errors.append(f"결측치 총 {total_missing}건")
    else:
        print("[PASS] 결측치 없음")

    # 3. 숫자형 및 허용 범위 검사
    for col, (min_value, max_value) in RANGE_RULES.items():
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        # 값이 있는데 숫자로 변환되지 않은 경우만 형식 오류로 판정
        # (빈 값은 위의 결측치 검사에서 이미 처리)
        non_numeric = numeric.isna() & df[col].notna()

        if non_numeric.any():
            errors.append(f"{col}: 숫자가 아닌 값 {int(non_numeric.sum())}건")
            continue

        invalid = df[(numeric < min_value) | (numeric > max_value)]

        if len(invalid) > 0:
            errors.append(
                f"{col}: 허용범위 {min_value}~{max_value} 위반 {len(invalid)}건"
            )
        else:
            print(f"[PASS] {col} 범위")

    # 4. IQR 기반 통계적 이상치 확인
    # 이상치는 반드시 오류라는 뜻이 아니므로 경고(WARN)로 처리
    for col in ["study_hours", "avg_score"]:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        outliers, lower, upper = check_iqr_outliers(series)

        if len(outliers) > 0:
            print(
                f"[WARN] {col}: IQR 기준 이상치 {len(outliers)}건 "
                f"(범위 {lower:.2f} ~ {upper:.2f})"
            )
        else:
            print(f"[PASS] {col}: IQR 기준 특이값 없음")

    # 5. 최종 판정
    if errors:
        print("\n=== QUALITY CHECK FAILED ===")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("\n=== QUALITY CHECK PASSED ===")
    sys.exit(0)

if __name__ == "__main__":
    main()
PY
```

---

## 4.4 정상 데이터 검증

최신 전처리 데이터 생성:

```bash
python src/preprocess.py
```

품질 검사 실행:

```bash
python src/quality_check.py
```

정상 실행 시 마지막 출력:

```text
=== QUALITY CHECK PASSED ===
```

종료코드 확인:

```bash
echo $?
```

정상 종료 시 표시:

```text
0
```

---

## 4.5 오류 데이터 생성

품질검사가 실제로 문제를 잡는지 확인하기 위해  
정상 데이터 복사본에 의도적으로 오류 삽입

```bash
python - <<'PY'
import pandas as pd

src = "data/processed/students_clean.csv"
dst = "data/processed/students_invalid.csv"

df = pd.read_csv(src)

# 비정상 출석률
df.loc[0, "attendance"] = 150

# 비정상 학습시간
df.loc[1, "study_hours"] = 99

# 중복 ID 생성
df.loc[2, "student_id"] = df.loc[1, "student_id"]

# 결측치 생성
df.loc[3, "assignment_score"] = None

df.to_csv(dst, index=False)

print(f"오류 데이터 생성: {dst}")
PY
```

검증:

```bash
python src/quality_check.py data/processed/students_invalid.csv
```

예상 결과:

```text
=== QUALITY CHECK FAILED ===
- student_id 중복 ...
- 결측치 ...
- study_hours 허용범위 위반 ...
- attendance 허용범위 위반 ...
```

종료코드 확인:

```bash
echo $?
```

실패한 경우:

```text
1
```

---

## 4.6 왜 종료코드가 중요한가?

화면을 보고 사람이 판단하는 방식만으로는 MLOps 파이프라인 자동화 불가

품질검사 프로그램 동작:

```text
품질 정상 → exit code 0
품질 실패 → exit code 1
```

GitHub Actions, ZenML, Airflow 등의 자동화 도구: 종료코드에 따라 다음 단계 실행 또는 중단 결정

```text
데이터 입력
   ↓
품질 검증
   ↓
┌───────────────┐
│ 통과          │ 실패
↓               ↓
모델 학습       파이프라인 중단
```

이후 CI/CD 및 파이프라인 자동화 실습으로 연결

---

# 5. 전체 실습 결과 확인

프로젝트 구조 확인:

```bash
find . -maxdepth 3 -type f | sort
```

예상 프로젝트 구조:

```text
mlops-week02/
├── .dvc/
├── .dvcignore
├── .git/
├── data/
│   ├── raw/
│   │   ├── students_raw.csv
│   │   └── students_raw.csv.dvc
│   └── processed/
│       ├── students_clean.csv
│       └── students_invalid.csv
├── reports/
└── src/
    ├── inspect_data.py
    ├── preprocess.py
    └── quality_check.py
```

---

# 6. 실습 정리

이번 주 실습의 전체 수행 과정:

```text
[2차시]
원본 CSV
  ↓
Pandas
  ↓
결측치 처리
중복 제거
범위 보정
파생변수 생성
  ↓
전처리 데이터

[3차시]
데이터 파일
  ↓
DVC add
  ↓
DVC Remote
  ↕
Git commit
  ↓
코드 + 데이터 버전 연결

[4차시]
전처리 데이터
  ↓
Schema Check
Missing Value Check
Range Check
Duplicate Check
Outlier Check
  ↓
PASS / FAIL
```

### 핵심 개념

**1. 데이터 전처리는 일회성 작업이 아니라 재현 가능한 코드여야 함**

**2. 모델뿐 아니라 데이터에도 버전이 존재해야 함**

**3. 어떤 모델을 어떤 데이터로 학습했는지 추적할 수 있어야 함**

**4. 모델 학습 전에 데이터 품질을 자동으로 검증해야 함**

**5. 데이터 품질 검증 결과는 이후 자동화 파이프라인의 실행 조건으로 활용할 수 있음**

---

# 7. 선택 과제

다음 과제 중 하나 이상 수행:

1. `students_raw.csv`에 새로운 학생 3명을 추가하고 DVC 새 버전 생성
2. `preprocess.py`의 결측치 처리 방법을 중앙값에서 평균값으로 바꾸고 결과 차이 비교
3. 품질검사 규칙에 `attendance < 50`인 학생 수를 경고하는 항목 추가
4. `avg_score` 이외의 새로운 파생변수 설계
5. `data-v1`과 `data-v2`를 각각 체크아웃하여 행 수 차이 확인

---

# 8. 다음 주차 연결

이번 주 학습 주제: **데이터의 재현성**

다음 주 학습 주제: 같은 원리를 모델 개발 단계로 확장하고 MLflow로 다음 항목 관리

```text
데이터 버전
    ↓
모델 학습
    ↓
파라미터
메트릭
아티팩트
    ↓
실험 추적
    ↓
모델 버전관리
```

즉,

```text
2주차: 어떤 데이터로 학습했는가?
3주차: 어떤 설정으로 어떤 모델을 만들었는가?
```

각 단계에서 해당 항목을 추적하는 것이 핵심
