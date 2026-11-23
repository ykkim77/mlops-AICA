# 2주차 실습 — 데이터 관리 및 전처리

> MLOps 방법론  
> 주제: Pandas 기반 데이터 전처리 → DVC 데이터 버전관리 → 데이터 품질 검증  
> 실습 환경: Google Colab(2차시 체험) → Windows 10/11 + WSL2 Ubuntu 22.04 + VS Code / Python 3.10 이상 / Git  
> 클라우드 계정: 필요 없음 (Colab 실습에는 Google 계정 필요)

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

### 실습 진행 방식

이번 주에는 같은 전처리 실습을 두 가지 환경에서 진행함

| 구분 | A. Colab 기반 | B. VS Code 기반 |
|---|---|---|
| 실행 위치 | Google 클라우드의 노트북 | 내 PC의 WSL Ubuntu |
| 코드 작성 | 노트북 셀에 직접 작성 | `.py` 파일로 작성 |
| 실행 방법 | 셀 실행 (`Shift + Enter`) | 터미널에서 `python 파일명.py` |
| 설치 | 필요 없음 | WSL, VS Code, Python 가상환경 |
| 파일 보존 | 런타임 종료 시 삭제 | 내 PC에 계속 남음 |
| 진행 범위 | 2차시 전처리 체험 | 2~4차시 전체, 3주차 이후 모든 실습 |

- 먼저 Colab에서 설치 없이 전처리 흐름을 체험함
- 이후 VS Code 환경을 구성하여 같은 작업을 파일 기반으로 다시 수행함
- 3주차부터는 VS Code 기반으로만 진행함

왜 VS Code(파일 기반)로 옮기는가:

- 노트북 셀은 실행 순서에 따라 결과가 달라질 수 있음 → 같은 결과를 반복해서 얻기 어려움
- `.py` 파일은 항상 처음부터 같은 순서로 실행됨 → 재현 가능한 전처리
- 파일이어야 Git·DVC로 버전관리하고, 이후 GitHub Actions·ZenML 같은 자동화 도구로 실행할 수 있음
- MLflow 서버, Docker 등 3주차 이후 실습은 로컬 환경이 필요함

---

# A. Colab 기반 실습 (2차시 먼저 체험)

## A.1 Colab 노트북 열기

- 브라우저에서 `https://colab.research.google.com` 접속
- Google 계정으로 로그인
- `파일` → `새 노트북` 선택
- 노트북 이름을 `week02_preprocessing.ipynb`로 변경

## A.2 Colab 셀 명령 규칙

| 표기 | 의미 | 예 |
|---|---|---|
| `!명령` | Linux 셸 명령 실행 | `!mkdir`, `!cat` |
| `%cd 경로` | 노트북의 작업 위치 변경 | `%cd /content` |
| `%%writefile 파일` | 셀 내용을 파일로 저장 (반드시 셀의 첫 줄) | `%%writefile data/raw/a.csv` |
| 표기 없음 | 일반 Python 코드 실행 | `import pandas as pd` |

- `!cd 폴더`는 다음 셀에 적용되지 않음 → 작업 위치 변경은 반드시 `%cd` 사용
- Colab에는 pandas가 이미 설치되어 있음 → 가상환경 생성·패키지 설치 불필요

## A.3 디렉터리 생성

셀 1:

```python
%cd /content
!mkdir -p /content/data/raw /content/data/processed /content/src /content/reports
```

- 왼쪽 `파일`(폴더 아이콘) 패널에서 `data`, `src`, `reports` 폴더 생성 확인
- B. VS Code 실습과 같은 폴더 구조를 사용함

## A.4 원본 데이터 파일 생성

셀 2:

```python
%%writefile data/raw/students_raw.csv
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
```

- `%%writefile` 앞에 빈 줄이나 주석이 있으면 동작하지 않음 → 반드시 셀의 첫 줄에 작성
- 실행 시 `Writing data/raw/students_raw.csv` 출력

## A.5 원본 데이터 파일 확인

셀 3:

```python
!cat data/raw/students_raw.csv
```

데이터에 의도적으로 포함한 문제:

- `study_hours`, `attendance`, `assignment_score` 결측치
- `student_id=S010` 중복
- `study_hours=18` 비정상적으로 큰 값
- `attendance=102` 허용 범위를 벗어난 값

## A.6 Pandas로 원본 데이터 확인

셀 4:

```python
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
```

확인할 내용:

1. 전체 행과 열의 수는 몇 개인가?
2. 어떤 컬럼에 결측치가 있는가?
3. 중복된 학생 ID가 있는가?
4. `attendance`의 최대값이 정상 범위인가?
5. `study_hours`의 최대값은 다른 값들과 비교해 지나치게 크지 않은가?

## A.7 데이터 전처리

셀 5:

```python
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
```

예상 출력 (앞부분):

```text
[1] 원본 데이터: 12행 × 5열
[2] 중복 제거: 1건 제거
[3] 전처리 완료: 11행 × 7열
[4] 저장 위치: data/processed/students_clean.csv
```

## A.8 결과 파일 확인

셀 6:

```python
!cat data/processed/students_clean.csv
```

확인할 내용:

- 중복된 `S010`이 1건으로 줄었는가?
- 결측치가 중앙값으로 채워졌는가?
- `S007`의 `study_hours`가 18 → 12로, `S009`의 `attendance`가 102 → 100으로 보정되었는가?
- `avg_score`, `risk_flag` 컬럼이 추가되었는가?

## A.9 Colab 실습의 한계

- 런타임 연결이 끊기면 `/content` 아래 파일이 모두 삭제됨
- 셀을 순서와 다르게 실행하면 결과가 달라질 수 있음
- 3차시의 Git·DVC 버전관리, 3주차 이후의 MLflow 서버·Docker 실습에는 적합하지 않음

→ 이후 같은 내용을 B. VS Code 기반 실습에서 `.py` 파일로 다시 작성하고, 이상치 탐지(2.4)와 3·4차시까지 이어서 진행함

---

# B. VS Code 기반 실습

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

### 1.0.8 VS Code 설치 및 WSL 연결

- Windows에 VS Code 설치: `https://code.visualstudio.com` → 다운로드 후 설치
- VS Code 실행 → 왼쪽 `확장(Extensions)` 아이콘(`Ctrl + Shift + X`)에서 다음 두 확장 설치
  - **WSL** (Microsoft)
  - **Python** (Microsoft)
- VS Code는 Windows에 설치하고, 코드와 Python은 WSL Ubuntu 안에서 실행하는 구조임

연결 확인은 1.1에서 실습 폴더를 만든 뒤 진행함

### 1.0.9 자주 발생하는 오류

| 증상 | 원인 및 해결 |
|---|---|
| `0x80370102` 오류 | CPU 가상화 비활성화 → BIOS/UEFI에서 VT-x 또는 AMD-V(SVM) 활성화 |
| `wsl -l -v`의 VERSION이 `1` | PowerShell에서 `wsl --set-version Ubuntu-22.04 2` 실행 |
| WSL 업데이트 필요 안내 | PowerShell에서 `wsl --update` 실행 |
| `python3 -m venv` 실패 | 1.0.5의 `python3-venv` 설치 여부 확인 |
| Ubuntu에서 `code: command not found` | VS Code에 WSL 확장 설치 후 Ubuntu 터미널을 닫았다 다시 열기 |

## 1.1 프로젝트 디렉터리 생성

`시작` 메뉴에서 `Ubuntu 22.04`를 열고 다음 명령 실행:

```bash
mkdir -p ~/mlops-week02
cd ~/mlops-week02
code .
```

- `code .` = 현재 폴더를 VS Code로 열기
- 처음 실행 시 WSL용 VS Code Server가 자동 설치됨 (1~2분 소요)
- VS Code 왼쪽 아래에 `WSL: Ubuntu-22.04`가 표시되면 연결 성공
- `신뢰할 수 있는 작성자` 확인 창이 나타나면 `예, 작성자를 신뢰합니다` 선택

이후 모든 명령은 VS Code 터미널에서 실행함

- VS Code 메뉴 `터미널` → `새 터미널` (단축키: ``Ctrl + ` ``)
- 터미널이 `~/mlops-week02` 위치의 bash로 열리는지 확인
- 파일 생성은 `cat > 파일 <<'PY'` 명령 대신 VS Code 왼쪽 탐색기에서 새 파일을 만들고 코드를 붙여넣어도 됨

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

VS Code에서 Python 인터프리터 선택:

- `Ctrl + Shift + P` → `Python: Select Interpreter` 입력
- `./.venv/bin/python` 선택
- 이후 새 터미널을 열면 가상환경이 자동으로 활성화됨

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
- IQR 기준으로 이상치를 탐지하고 처리 방법을 비교함
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

## 2.4 이상치 탐지 및 처리 방법 비교

### 이상치와 오류의 구분

| 구분 | 의미 | 예 | 기본 처리 방향 |
|---|---|---|---|
| 범위 위반 | 도메인 규칙상 존재할 수 없는 값 | 출석률 102% | 오류로 보고 수정 |
| 이상치 | 다른 값들과 비교해 드물게 크거나 작은 값 | 출석률 65% | 오류인지 실제 값인지 확인 후 결정 |

- 이상치 = 반드시 오류는 아님
- 통계적으로 드문 값이 실제로는 가장 중요한 데이터일 수 있음

### IQR 기준 이상치 탐지

```text
Q1  = 하위 25% 지점의 값
Q3  = 하위 75% 지점의 값
IQR = Q3 − Q1

정상 범위 = Q1 − 1.5 × IQR  ~  Q3 + 1.5 × IQR
→ 정상 범위를 벗어난 값을 이상치로 판단
```

- 가운데 50% 데이터의 퍼짐(IQR)을 기준으로 판단함
- 평균·표준편차를 사용하지 않음 → 극단값 자신이 기준을 왜곡하지 않음
- 데이터 분포가 정규분포가 아니어도, 표본이 작아도 사용 가능
- 1.5는 관례적인 배수이며, 더 극단적인 값만 찾으려면 3.0 사용

### 탐지 및 비교 코드 작성

`src/detect_outliers.py` 생성:

```bash
cat > src/detect_outliers.py <<'PY'
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/students_raw.csv")
REPORT_PATH = Path("reports/outlier_report.csv")

NUMERIC_COLS = [
    "study_hours",
    "attendance",
    "assignment_score",
    "midterm_score",
]

IQR_K = 1.5  # 이상치 판단 배수 (일반적으로 1.5 사용)


def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - IQR_K * iqr, q3 + IQR_K * iqr


# 1. 데이터 준비: 중복 제거 후 숫자형 변환 (결측치는 그대로 둠)
df = pd.read_csv(RAW_PATH)
df = df.drop_duplicates(subset=["student_id"], keep="first")
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. IQR 기준 이상치 탐지
records = []
for col in NUMERIC_COLS:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    lower, upper = iqr_bounds(s)

    print(f"\n=== {col} ===")
    print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={q3 - q1:.2f}")
    print(f"정상 범위: {lower:.2f} ~ {upper:.2f}")

    outliers = s[(s < lower) | (s > upper)]

    if outliers.empty:
        print("- 이상치 없음")

    for idx, value in outliers.items():
        sid = df.loc[idx, "student_id"]
        print(f"- {sid}: {value}")
        records.append(
            {
                "student_id": sid,
                "column": col,
                "value": value,
                "lower": round(lower, 2),
                "upper": round(upper, 2),
            }
        )

# 3. 탐지 결과 저장
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
report = pd.DataFrame(records)
report.to_csv(REPORT_PATH, index=False)
print(f"\n이상치 리포트 저장: {REPORT_PATH} ({len(report)}건)")

# 4. 처리 방법 비교: study_hours의 S007(18시간)을 예로 비교
col = "study_hours"
s = df[col]
lower, upper = iqr_bounds(s.dropna())

compare = {
    "원본": s,
    "제거(drop)": s[(s >= lower) & (s <= upper) | s.isna()],
    "IQR 경계로 보정": s.clip(lower=lower, upper=upper),
    "도메인 상한으로 보정": s.clip(lower=0, upper=12),
    "중앙값으로 대체": s.mask(s > upper, s.median()),
}

summary = pd.DataFrame(
    [
        {
            "방법": name,
            "행 수": len(values),
            "평균": round(values.mean(), 2),
            "최대": round(values.max(), 2),
        }
        for name, values in compare.items()
    ]
).set_index("방법")

print(f"\n=== 처리 방법별 {col} 비교 ===")
print(summary)
PY
```

실행:

```bash
python src/detect_outliers.py
```

탐지 결과:

```text
=== study_hours ===
Q1=3.62, Q3=6.75, IQR=3.12
정상 범위: -1.06 ~ 11.44
- S007: 18.0

=== attendance ===
Q1=84.00, Q3=94.25, IQR=10.25
정상 범위: 68.62 ~ 109.62
- S008: 65.0

=== assignment_score ===
Q1=81.25, Q3=94.50, IQR=13.25
정상 범위: 61.38 ~ 114.38
- S008: 60.0

=== midterm_score ===
Q1=77.00, Q3=90.00, IQR=13.00
정상 범위: 57.50 ~ 109.50
- 이상치 없음

이상치 리포트 저장: reports/outlier_report.csv (3건)
```

결과 해석:

| 학생 | 값 | IQR 탐지 | 해석 |
|---|---|---|---|
| S007 | 학습시간 18 | 이상치 | 하루 학습시간으로 비현실적 → 입력 오류 가능성 높음 |
| S008 | 출석 65, 과제 60 | 이상치 | 모든 지표가 낮은 학습위험 학생 → 오류가 아닌 실제 데이터 |
| S009 | 출석률 102 | 탐지 안 됨 | 정상 범위(68.62 ~ 109.62) 안의 값 → 통계 기준만으로는 오류를 놓침 |

- S008: 이상치라고 삭제하면 모델이 찾아야 할 학습위험 학생 데이터가 사라짐
- S009: IQR은 "드문 값"을 찾을 뿐, "불가능한 값"은 찾지 못함 → 허용 범위 검사가 별도로 필요함

### 처리 방법 비교

같은 스크립트의 마지막 출력:

```text
=== 처리 방법별 study_hours 비교 ===
             행 수    평균     최대
방법
원본            11  5.95  18.00
제거(drop)      10  4.61   8.00
IQR 경계로 보정    11  5.29  11.44
도메인 상한으로 보정   11  5.35  12.00
중앙값으로 대체      11  4.62   8.00
```

| 처리 방법 | 방식 | 장점 | 단점 |
|---|---|---|---|
| 제거 | 해당 행 삭제 | 단순함 | 다른 컬럼의 정상 정보까지 함께 사라짐, 소규모 데이터에서 손실 큼 |
| IQR 경계로 보정 | 경계값으로 잘라냄 (Capping) | 행 유지 | 데이터가 바뀔 때마다 경계값이 달라짐 |
| 도메인 상한으로 보정 | 허용 범위 경계로 잘라냄 | 기준이 고정되어 재현성 높음 | 실제 값이 무엇이었는지는 알 수 없음 |
| 중앙값으로 대체 | 이상치를 결측치처럼 취급 | 분포 왜곡 최소화 | 원래 값과의 관계가 완전히 사라짐 |
| 유지 + 표시 | 값은 그대로 두고 플래그 컬럼 추가 | 정보 손실 없음, 나중에 판단 가능 | 모델이 이상치의 영향을 그대로 받음 |

- 원본 평균 5.95 → 처리 방법에 따라 4.61 ~ 5.35로 달라짐
- 처리 방법 선택 = 모델 학습 데이터를 바꾸는 결정 → 반드시 코드와 문서로 기록해야 함

### 이번 실습의 처리 정책

- IQR 이상치(S007, S008): `outlier_flag = 1`로 표시하여 추적 가능하게 함
- 허용 범위 위반(S007, S009): 범위의 경계값으로 보정 (18 → 12, 102 → 100)
- 허용 범위 안의 이상치(S008): 값을 바꾸지 않고 유지

> 실제 업무에서는 이상치를 발견하면 먼저 원천 데이터 담당자에게 확인하고, 처리 기준을 팀의 데이터 정책으로 문서화함

다음 단계에서 이 정책을 `preprocess.py`에 반영함

---

## 2.5 데이터 전처리 코드 작성

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

# 5. 이상치 탐지 및 처리
RANGE_RULES = {
    "study_hours": (0, 12),
    "attendance": (0, 100),
    "assignment_score": (0, 100),
    "midterm_score": (0, 100),
}

# 5-1. 이상치 표시 (IQR 기준)
# 하나의 컬럼이라도 IQR 정상 범위를 벗어나면 outlier_flag = 1
outlier_mask = pd.Series(False, index=df.index)

for col in RANGE_RULES:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    iqr_outlier = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    outlier_mask = outlier_mask | iqr_outlier

df["outlier_flag"] = outlier_mask.astype(int)
print(f"[3] 이상치 표시: {df['outlier_flag'].sum()}명")

# 5-2. 처리
# 허용 범위를 벗어난 값(명백한 오류)만 범위의 경계값으로 보정
# 범위 안의 이상치는 삭제·변경하지 않고 outlier_flag로 표시만 유지
# 실제 업무에서는 도메인 규칙과 데이터 담당자의 확인이 필요함
for col, (min_value, max_value) in RANGE_RULES.items():
    df[col] = df[col].clip(lower=min_value, upper=max_value)

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

print(f"[4] 전처리 완료: {df.shape[0]}행 × {df.shape[1]}열")
print(f"[5] 저장 위치: {OUTPUT_PATH}")
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

확인할 내용:

- 실행 로그에 `[3] 이상치 표시: 2명` 출력
- `S007`: `study_hours` 18 → 12로 보정, `outlier_flag = 1`
- `S008`: 값은 그대로 유지, `outlier_flag = 1`
- `S009`: `attendance` 102 → 100으로 보정, `outlier_flag = 0` (IQR 이상치가 아니라 범위 위반)
- 전체 컬럼 수: 8개 (`outlier_flag`, `avg_score`, `risk_flag` 추가)

---

## 2.6 전처리 전·후 비교

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
- 원본 데이터와 전처리 데이터를 함께 버전관리함
- 대량의 신규 데이터가 들어온 상황을 재현함
- 데이터가 변경될 때마다 전처리를 다시 수행하고 새 데이터 버전을 생성함
- `data-v0 → data-v1 → data-v2`로 이동하며 데이터 버전을 직접 복원함

이번 실습의 핵심 흐름:

```text
2차시에서 만든 초기 데이터
        ↓
      data-v0
        ↓
신규 데이터 50건 추가
        ↓
전처리 다시 실행
        ↓
      data-v1
        ↓
신규 데이터 100건 추가
        ↓
전처리 다시 실행
        ↓
      data-v2
        ↓
과거 버전으로 이동·복원
```

> 핵심: 데이터가 바뀌면 원본 CSV만 바뀌는 것이 아니라, 같은 전처리 코드를 다시 실행하여 가공 데이터도 함께 새 버전으로 관리해야 함

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

Git 상태 확인:

```bash
git status
```

---

## 3.3 원본 데이터와 전처리 데이터 DVC로 관리

2차시까지 실습을 완료했으므로 다음 두 파일이 이미 존재함:

```text
data/raw/students_raw.csv
data/processed/students_clean.csv
```

두 파일을 DVC 관리 대상으로 등록:

```bash
dvc add data/raw/students_raw.csv
dvc add data/processed/students_clean.csv
```

확인:

```bash
ls data/raw
ls data/processed
```

다음 메타데이터 파일이 생성되어야 함:

```text
data/raw/students_raw.csv.dvc
data/processed/students_clean.csv.dvc
```

내용 확인:

```bash
cat data/raw/students_raw.csv.dvc
cat data/processed/students_clean.csv.dvc
```

DVC는 실제 CSV의 내용 대신 `.dvc` 메타데이터 파일에 데이터 버전을 식별할 수 있는 정보를 기록함.

---

## 3.4 로컬 DVC Remote 구성

이번 실습에서는 별도의 클라우드 계정 없이 프로젝트 밖의 로컬 폴더를 DVC Remote로 사용함.

```bash
mkdir -p ../dvc-storage
dvc remote add -d localstorage ../dvc-storage
```

설정 확인:

```bash
dvc remote list
```

초기 데이터 저장:

```bash
dvc push
```

구조:

```text
Git
 ├─ Python 코드
 ├─ .dvc 파일
 └─ DVC 설정

DVC Remote
 ├─ 원본 데이터 실제 내용
 └─ 전처리 데이터 실제 내용
```

---

## 3.5 초기 데이터 버전 생성 — data-v0

2차시에서 사용한 초기 데이터 상태를 기준 버전으로 저장함.

현재 원본/전처리 데이터 행 수 확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("RAW   :", raw.shape)
print("CLEAN :", clean.shape)
PY
```

초기 상태를 Git에 기록:

```bash
git add .
git commit -m "week02: baseline data version"
git tag data-v0
```

확인:

```bash
git log --oneline --decorate
```

현재 상태:

```text
data-v0
- 2차시에서 사용한 초기 원본 데이터
- 2차시 전처리 결과
```

---

## 3.6 첫 번째 대량 데이터 추가

실제 운영환경에서 새로운 학생 데이터가 한꺼번에 수집된 상황을 가정함.

이번에는 **신규 학생 50명(S012~S061)** 을 추가함.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

path = Path("data/raw/students_raw.csv")

old = pd.read_csv(path)

# 항상 같은 실습 결과가 나오도록 난수 고정
rng = np.random.default_rng(42)

n = 50
new = pd.DataFrame({
    "student_id": [f"S{i:03d}" for i in range(12, 62)],
    "study_hours": np.round(rng.normal(5.0, 2.0, n), 1),
    "attendance": np.round(rng.normal(87, 8, n), 1),
    "assignment_score": np.round(rng.normal(82, 10, n), 1),
    "midterm_score": np.round(rng.normal(80, 11, n), 1),
})

# 현실적인 데이터 문제를 일부 의도적으로 삽입
new.loc[3, "study_hours"] = np.nan
new.loc[8, "attendance"] = np.nan
new.loc[15, "assignment_score"] = np.nan

# 허용 범위를 벗어난 값
new.loc[20, "study_hours"] = 17
new.loc[31, "attendance"] = 105

updated = pd.concat([old, new], ignore_index=True)
updated.to_csv(path, index=False)

print("기존 행 수   :", len(old))
print("추가 행 수   :", len(new))
print("변경 후 행 수:", len(updated))
PY
```

변경 확인:

```bash
tail -n 15 data/raw/students_raw.csv
wc -l data/raw/students_raw.csv
dvc status
```

원본 데이터가 변경되었다는 메시지가 나타나는지 확인.

---

## 3.7 첫 번째 변경 데이터 전처리

데이터가 변경되었으므로 2차시에서 작성한 동일한 전처리 코드를 다시 실행함.

```bash
python src/preprocess.py
```

전처리 후 원본/가공 데이터 확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("=== VERSION 1 후보 데이터 ===")
print("RAW   :", raw.shape)
print("CLEAN :", clean.shape)
print("RAW 결측치  :", raw.isna().sum().sum())
print("CLEAN 결측치:", clean.isna().sum().sum())
print("CLEAN study_hours max:", clean["study_hours"].max())
print("CLEAN attendance max :", clean["attendance"].max())
PY
```

확인할 내용:

- 원본 데이터에 신규 학생 50명이 추가되었는가?
- 신규 데이터의 결측치가 전처리되었는가?
- `study_hours=17`이 12 이하로 보정되었는가?
- `attendance=105`가 100 이하로 보정되었는가?
- 2차시에서 만든 동일한 `src/preprocess.py`가 다시 사용되었는가?

---

## 3.8 첫 번째 변경 데이터 저장 — data-v1

변경된 원본 데이터와 전처리 데이터를 DVC에 다시 등록:

```bash
dvc add data/raw/students_raw.csv
dvc add data/processed/students_clean.csv
```

DVC Remote에 저장:

```bash
dvc push
```

Git에 새 데이터 버전의 메타데이터 기록:

```bash
git add .
git commit -m "week02: data version 1 - add 50 students"
git tag data-v1
```

확인:

```bash
git log --oneline --decorate
```

흐름:

```text
data-v0
  ↓
신규 데이터 50건 추가
  ↓
전처리 재실행
  ↓
data-v1
```

---

## 3.9 두 번째 대량 데이터 추가

다시 새로운 데이터가 수집된 상황을 가정함.

이번에는 **신규 학생 100명(S062~S161)** 을 추가하여 데이터 규모를 한 번 더 증가시킴.

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

path = Path("data/raw/students_raw.csv")

old = pd.read_csv(path)

rng = np.random.default_rng(2026)

n = 100
new = pd.DataFrame({
    "student_id": [f"S{i:03d}" for i in range(62, 162)],
    "study_hours": np.round(rng.normal(4.5, 2.3, n), 1),
    "attendance": np.round(rng.normal(84, 10, n), 1),
    "assignment_score": np.round(rng.normal(79, 12, n), 1),
    "midterm_score": np.round(rng.normal(78, 13, n), 1),
})

# 결측치 삽입
new.loc[5, "study_hours"] = np.nan
new.loc[18, "attendance"] = np.nan
new.loc[27, "assignment_score"] = np.nan
new.loc[44, "midterm_score"] = np.nan

# 허용 범위를 벗어난 값 삽입
new.loc[52, "study_hours"] = 20
new.loc[71, "attendance"] = 108
new.loc[83, "assignment_score"] = 115

updated = pd.concat([old, new], ignore_index=True)
updated.to_csv(path, index=False)

print("기존 행 수   :", len(old))
print("추가 행 수   :", len(new))
print("변경 후 행 수:", len(updated))
PY
```

변경 확인:

```bash
tail -n 15 data/raw/students_raw.csv
wc -l data/raw/students_raw.csv
dvc status
```

---

## 3.10 두 번째 변경 데이터 전처리

데이터가 다시 변경되었으므로 같은 전처리 코드를 한 번 더 실행함.

```bash
python src/preprocess.py
```

전처리 결과 확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("=== VERSION 2 후보 데이터 ===")
print("RAW   :", raw.shape)
print("CLEAN :", clean.shape)

print("\n=== 결측치 ===")
print("RAW   :", raw.isna().sum().sum())
print("CLEAN :", clean.isna().sum().sum())

print("\n=== 전처리 데이터 허용범위 ===")
print("study_hours max      :", clean["study_hours"].max())
print("attendance max       :", clean["attendance"].max())
print("assignment_score max :", clean["assignment_score"].max())
print("midterm_score max    :", clean["midterm_score"].max())
PY
```

확인할 내용:

- 두 번째 신규 데이터 100명이 추가되었는가?
- 원본 데이터의 결측치가 전처리 후 제거되었는가?
- 허용범위를 벗어난 값이 보정되었는가?
- 데이터가 바뀌어도 같은 전처리 코드를 반복 실행할 수 있는가?

---

## 3.11 두 번째 변경 데이터 저장 — data-v2

변경된 원본 데이터와 전처리 데이터를 다시 DVC에 등록:

```bash
dvc add data/raw/students_raw.csv
dvc add data/processed/students_clean.csv
```

DVC Remote에 저장:

```bash
dvc push
```

Git에 새 데이터 버전 기록:

```bash
git add .
git commit -m "week02: data version 2 - add 100 students"
git tag data-v2
```

확인:

```bash
git log --oneline --decorate
```

최종 흐름:

```text
data-v0
초기 데이터
   ↓
+ 신규 50건
+ 전처리
   ↓
data-v1
   ↓
+ 신규 100건
+ 전처리
   ↓
data-v2
```

---

## 3.12 데이터 버전별 실제 내용 비교

현재 `data-v2` 상태 확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("RAW   :", len(raw))
print("CLEAN :", len(clean))
PY
```

이제 과거 버전으로 이동하면서 실제 CSV 내용이 바뀌는지 확인함.

---

## 3.13 data-v0으로 되돌리기

```bash
git checkout data-v0
dvc checkout
```

확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("현재 버전: data-v0")
print("RAW   :", len(raw))
print("CLEAN :", len(clean))
print("마지막 student_id:", raw.iloc[-1]["student_id"])
PY
```

초기 데이터로 돌아왔는지 확인.

---

## 3.14 data-v1으로 이동

```bash
git checkout data-v1
dvc checkout
```

확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("현재 버전: data-v1")
print("RAW   :", len(raw))
print("CLEAN :", len(clean))
print("마지막 student_id:", raw.iloc[-1]["student_id"])
PY
```

`S061`까지 존재하는지 확인.

---

## 3.15 data-v2로 이동

```bash
git checkout data-v2
dvc checkout
```

확인:

```bash
python - <<'PY'
import pandas as pd

raw = pd.read_csv("data/raw/students_raw.csv")
clean = pd.read_csv("data/processed/students_clean.csv")

print("현재 버전: data-v2")
print("RAW   :", len(raw))
print("CLEAN :", len(clean))
print("마지막 student_id:", raw.iloc[-1]["student_id"])
PY
```

`S161`까지 존재하는지 확인.

실습을 계속하기 위해 main 브랜치로 복귀:

```bash
git checkout main
dvc checkout
```

---

## 3.16 데이터 버전이 실제로 무엇을 의미하는가?

이번 실습에서는 각 버전마다 원본과 전처리 데이터가 함께 바뀜.

```text
data-v0
├─ raw v0
└─ processed v0

data-v1
├─ raw v1
└─ processed v1

data-v2
├─ raw v2
└─ processed v2
```

즉 데이터 버전은 CSV 파일 이름을 바꾸는 것이 아님.

```text
같은 파일명
students_raw.csv

하지만

data-v0에서의 내용
≠
data-v1에서의 내용
≠
data-v2에서의 내용
```

DVC는 각 시점의 파일 내용을 Hash 기반으로 식별하고, Git commit/tag와 연결하여 어떤 데이터 버전이 사용되었는지 추적함.

---

## 3.17 Git과 DVC의 역할 정리

| 관리 대상 | Git | DVC |
|---|---:|---:|
| Python 코드 | O | - |
| 설정 파일 | O | - |
| `.dvc` 메타데이터 | O | - |
| 원본 데이터 실제 내용 | 직접 관리 비권장 | O |
| 전처리 데이터 실제 내용 | 직접 관리 비권장 | O |
| 데이터 버전과 코드 버전 연결 | O | O |
| 로컬/원격 데이터 저장소 | - | O |

핵심 구조:

```text
Git commit / tag
        ↕
     .dvc 파일
        ↕
DVC 데이터 버전
        ↓
raw + processed 데이터
```


---

## 3.18 3차시 핵심 정리

이번 실습에서 수행한 작업:

```text
초기 데이터
   ↓
DVC 등록
   ↓
data-v0
   ↓
50건 추가
   ↓
전처리 재실행
   ↓
DVC 갱신
   ↓
data-v1
   ↓
100건 추가
   ↓
전처리 재실행
   ↓
DVC 갱신
   ↓
data-v2
   ↓
과거 버전 복원
```

핵심 개념:

- 데이터가 변경되면 전처리 결과도 다시 생성해야 함
- DVC는 파일 이름이 아니라 **파일 내용의 버전**을 관리함
- Git은 `.dvc` 메타데이터와 코드 버전을 관리함
- `data-v0`, `data-v1`, `data-v2` 태그를 이용해 특정 시점의 코드와 데이터를 함께 복원할 수 있음
- MLOps에서는 “현재 데이터가 무엇인가?”뿐 아니라 **“이 모델이 어느 데이터 버전으로 학습되었는가?”**를 재현할 수 있어야 함


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
| 이상치 표시 | `outlier_flag`는 0 또는 1 |
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
    "outlier_flag",
]

RANGE_RULES = {
    "study_hours": (0, 12),
    "attendance": (0, 100),
    "assignment_score": (0, 100),
    "midterm_score": (0, 100),
    "avg_score": (0, 100),
    "risk_flag": (0, 1),
    "outlier_flag": (0, 1),
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

출력 중 다음 경고가 함께 나타남:

```text
[WARN] study_hours: IQR 기준 이상치 1건 (범위 -0.38 ~ 10.62)
[WARN] avg_score: IQR 기준 이상치 1건 (범위 60.30 ~ 110.70)
```

- `study_hours`: S007을 12로 보정했지만 다른 학생과 비교하면 여전히 큰 값
- `avg_score`: S008의 낮은 점수
- 2차시 처리 정책대로 유지한 값이므로 실패가 아닌 경고(WARN)로만 표시

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
│   └── outlier_report.csv
└── src/
    ├── inspect_data.py
    ├── detect_outliers.py
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
이상치 탐지·처리
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

**2. 이상치는 반드시 오류가 아님 → 탐지와 처리를 분리하고 처리 기준을 코드로 기록해야 함**

**3. 모델뿐 아니라 데이터에도 버전이 존재해야 함**

**4. 어떤 모델을 어떤 데이터로 학습했는지 추적할 수 있어야 함**

**5. 모델 학습 전에 데이터 품질을 자동으로 검증해야 함**

**6. 데이터 품질 검증 결과는 이후 자동화 파이프라인의 실행 조건으로 활용할 수 있음**

---

# 7. 선택 과제

다음 과제 중 하나 이상 수행:

1. `students_raw.csv`에 새로운 학생 3명을 추가하고 DVC 새 버전 생성
2. `preprocess.py`의 결측치 처리 방법을 중앙값에서 평균값으로 바꾸고 결과 차이 비교
3. 품질검사 규칙에 `attendance < 50`인 학생 수를 경고하는 항목 추가
4. `avg_score` 이외의 새로운 파생변수 설계
5. `data-v1`과 `data-v2`를 각각 체크아웃하여 행 수 차이 확인
6. `detect_outliers.py`의 `IQR_K`를 1.5와 3.0으로 바꾸어 탐지 결과 비교
7. `S008`을 제거했을 때와 유지했을 때 `risk_flag = 1`인 학생 수 비교 후 어느 쪽이 적절한지 설명

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
