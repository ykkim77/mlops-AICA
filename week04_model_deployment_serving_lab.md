# 4단원 실습 — 모델 배포 및 서비스화

> MLOps 방법론  
> 주제: 로컬 REST API 서빙 → Docker 컨테이너 패키징 → 다중 인스턴스·부하분산  
> 실습 환경: WSL2 Ubuntu 22.04 기준 / Python 3.10 이상 / Docker Desktop  
> 클라우드 계정: 필요 없음

---

# 0. 실습 개요

이번 주에는 3단원에서 MLflow Model Registry에 등록하고 `champion` Alias를 지정한 Iris Random Forest 모델을 실제 서비스 형태로 발전시킴

전체 흐름은 다음과 같음

```text
[3단원 결과]
MLflow Model Registry
        ↓
iris-random-forest@champion

[2차시]
Champion Model
        ↓
mlflow models serve
        ↓
Local REST API
        ↓
HTTP 요청으로 추론

[3차시]
Champion Model
        ↓
mlflow models build-docker
        ↓
Docker Image
        ↓
Model Container

[4차시]
Model Container × 3
        ↓
Nginx Load Balancer
        ↓
요청 분산
```

이번 주 핵심 질문은 다음과 같음

> “선택한 모델을 다른 프로그램이 사용할 수 있는 서비스로 어떻게 만들 것인가?”

---

# 1. 사전 준비

## 1.1 3단원 MLflow Tracking Server 실행

3단원 프로젝트에서 MLflow Server를 실행함

터미널 1:

```bash
cd ~/mlops-week03
source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

브라우저에서 확인:

```text
http://127.0.0.1:5000
```

MLflow UI의 Models 메뉴에서 다음 모델이 존재하는지 확인함

```text
iris-random-forest
```

`champion` Alias가 특정 Version에 연결되어 있는지 확인함

---

## 1.2 4단원 작업 디렉터리 생성

터미널 2:

```bash
mkdir -p ~/mlops-week04
cd ~/mlops-week04
```

---

## 1.3 Python 가상환경

3단원 환경을 그대로 사용해도 됨

> 새로 만들 경우에도 MLflow 버전을 3단원 Tracking Server와 동일하게 맞춤

새로 만들 경우:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "mlflow==3.18.0" scikit-learn requests
```

---

## 1.4 MLflow Tracking URI 설정

CLI가 3단원 Tracking Server와 Model Registry를 바라보도록 설정함

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

확인:

```bash
echo $MLFLOW_TRACKING_URI
```

출력:

```text
http://127.0.0.1:5000
```

> 새 터미널을 열면 이 환경변수를 다시 설정해야 함

---

## 1.5 Champion 모델 확인

Python으로 `champion` Alias가 가리키는 모델을 확인함

```bash
python - <<'PY'
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model = client.get_model_version_by_alias(
    "iris-random-forest",
    "champion",
)

print("Model Name :", model.name)
print("Version    :", model.version)
print("Run ID     :", model.run_id)
PY
```

정상적으로 Version과 Run ID가 출력되는지 확인함

---

# 2차시 실습 — MLflow 모델을 로컬 REST API로 서빙

# 2. 실습 목표

- 모델 파일과 모델 서비스의 차이를 이해함
- MLflow Model Registry의 `champion` 모델을 로컬 API 서버로 실행함
- `/invocations` 엔드포인트로 HTTP 요청을 전송함
- JSON 요청과 예측 응답의 구조를 확인함
- 모델 서빙의 가장 기본적인 형태를 이해함

---

# 3. 모델과 모델 서비스의 차이

학습이 끝난 모델은 아직 다른 프로그램이 호출할 수 있는 서비스가 아님

```text
Model File
```

을 다음과 같이 바꾸어야 함

```text
Client
   ↓
HTTP Request
   ↓
Model Server
   ↓
Model
   ↓
Prediction
   ↓
HTTP Response
```

이번 차시에서는 MLflow가 제공하는 Model Server를 사용함

---

# 4. MLflow Model Serving 실행

## 4.1 모델 URI

3단원에서 등록한 모델:

```text
iris-random-forest
```

Alias:

```text
champion
```

따라서 모델 URI는 다음과 같음

```text
models:/iris-random-forest@champion
```

---

## 4.2 모델 서버 실행

터미널 2에서:

```bash
cd ~/mlops-week04
source .venv/bin/activate

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models serve \
  --model-uri "models:/iris-random-forest@champion" \
  --port 5001 \
  --host 127.0.0.1 \
  --env-manager local
```

> `5000` 포트는 Tracking Server가 사용 중이므로 Model Serving은 `5001`을 사용함

정상적으로 시작되면 Model Server가 다음 주소에서 동작함

```text
http://127.0.0.1:5001
```

이 터미널은 서버 실행용으로 그대로 둠

---

# 5. 서버 상태 확인

새 터미널 3을 열고 다음 명령을 실행함

```bash
curl http://127.0.0.1:5001/ping
```

또는:

```bash
curl http://127.0.0.1:5001/health
```

버전 확인:

```bash
curl http://127.0.0.1:5001/version
```

---

# 6. REST API로 예측 요청

Iris 데이터의 입력 특성은 4개임

```text
sepal length
sepal width
petal length
petal width
```

예제 입력:

```text
5.1, 3.5, 1.4, 0.2
```

요청:

```bash
curl -X POST \
  http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2]
        ]
      }'
```

예측 결과가 JSON으로 반환되는지 확인함

---

# 7. 여러 데이터 한 번에 요청

```bash
curl -X POST \
  http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2],
          [6.0, 2.9, 4.5, 1.5],
          [6.7, 3.1, 5.6, 2.4]
        ]
      }'
```

하나의 요청으로 여러 건의 추론을 수행할 수 있음

---

# 8. Python 프로그램에서 REST API 호출

`predict_client.py` 생성:

```bash
cat > predict_client.py <<'PY'
import requests

URL = "http://127.0.0.1:5001/invocations"

payload = {
    "inputs": [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.7, 3.1, 5.6, 2.4],
    ]
}

response = requests.post(
    URL,
    json=payload,
    timeout=10,
)

print("status code:", response.status_code)
print("response   :", response.json())
PY
```

실행:

```bash
python predict_client.py
```

---

# 9. REST API 구조 정리

```text
Client
   │
   │ POST /invocations
   │ JSON
   ↓
MLflow Model Server
   ↓
Champion Model
   ↓
Prediction
   │
   │ JSON
   ↓
Client
```

모델을 호출하는 프로그램은 모델의 내부 구현을 알 필요가 없음

클라이언트가 알아야 할 항목:

```text
URL
HTTP Method
Input Format
Output Format
```

---

# 10. 2차시 핵심

```text
Model
```

은 학습 결과물이고,

```text
Model Serving
```

은 그 모델을 다른 시스템이 호출할 수 있도록 실행하는 것임

이번 실습에서는:

```text
MLflow Registry
      ↓
champion
      ↓
mlflow models serve
      ↓
REST API
```

를 구성했음

---

# 3차시 실습 — Dockerfile로 모델 서빙 환경 패키징

# 11. 실습 목표

- Docker Desktop을 설치하고 WSL2 Ubuntu와 연동함
- Dockerfile의 역할과 기본 구조를 이해함
- 모델 서빙 애플리케이션과 Python Runtime, 의존성을 하나의 Docker Image로 패키징함
- Docker Image를 직접 Build함
- Docker Container를 실행하고 Host Port와 Container Port를 연결함
- REST API로 요청을 전송하여 Prediction 결과를 확인함

이번 차시의 핵심 흐름:

```text
Model Serving Application
        +
Python Runtime
        +
Dependencies
        ↓
Dockerfile
        ↓
Docker Image
        ↓
Docker Container
        ↓
REST API
```

핵심 질문:

> “다른 컴퓨터에서도 동일한 Image를 이용하면 같은 모델 서빙 환경을 실행할 수 있는가?”

---

# 12. Docker Desktop 설치 및 WSL2 연동

이번 실습은 다음 환경을 기준으로 함.

```text
Windows
  ↓
Docker Desktop
  ↓
WSL2
  ↓
Ubuntu 22.04
  ↓
VS Code / Ubuntu Terminal
```

> Docker Desktop 버전은 특정 버전으로 고정하지 않고 실습 시점의 안정 버전을 사용함

---

## 12.1 WSL2 상태 확인

Windows PowerShell에서:

```powershell
wsl -l -v
```

예:

```text
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

확인:

- `Ubuntu-22.04`가 존재함
- `VERSION`이 `2`임

VERSION이 1이면:

```powershell
wsl --set-version Ubuntu-22.04 2
```

WSL 업데이트:

```powershell
wsl --update
```

---

## 12.2 Docker Desktop 설치

Windows에 Docker Desktop을 설치하고 실행함.

Docker Desktop은 Windows에서 Docker Engine을 제공하며,
이번 실습에서는 WSL2 Ubuntu에서 그 Docker Engine을 사용함.

구조:

```text
Windows
└─ Docker Desktop
      └─ Docker Engine
             ↑
             │ WSL Integration
             │
      WSL2 Ubuntu
          └─ docker CLI
```

> 이번 실습에서는 Ubuntu 내부에 `docker.io` 또는 Docker Engine을 별도로 설치하지 않음

---

## 12.3 WSL2 Backend 확인

Docker Desktop에서:

```text
Settings
→ General
```

WSL2 기반 Engine을 사용하도록 설정함.

환경에 따라 WSL2가 기본 Backend로 적용되어 설정 항목이 별도로 보이지 않을 수도 있음.

---

## 12.4 Ubuntu WSL Integration 활성화

Docker Desktop:

```text
Settings
→ Resources
→ WSL Integration
```

다음 항목을 활성화함.

```text
Ubuntu-22.04
```

설정을 변경한 경우:

```text
Apply
```

또는

```text
Apply & Restart
```

---

## 12.5 WSL 재시작

필요한 경우 Windows PowerShell에서:

```powershell
wsl --shutdown
```

그 후:

1. Docker Desktop 실행
2. Ubuntu 22.04 실행
3. VS Code에서 WSL 환경 다시 연결

---

## 12.6 Docker 동작 확인

WSL Ubuntu 터미널에서:

```bash
docker --version
```

Docker Compose 확인:

```bash
docker compose version
```

Docker Engine 연결 확인:

```bash
docker version
```

`Client`와 `Server` 정보가 모두 출력되는지 확인함.

테스트:

```bash
docker run --rm hello-world
```

정상 메시지가 출력되면 Docker 실습 환경 준비 완료.

---

# 13. 실습 프로젝트 구성

작업 디렉터리:

```bash
cd ~/mlops-week04

mkdir -p docker-serving
cd docker-serving
```

이번 실습의 최종 파일 구조:

```text
docker-serving/
├── Dockerfile
├── requirements.txt
├── app.py
└── model/
    ├── MLmodel
    ├── python_model.pkl 또는 model.pkl
    └── ...
```

각 파일의 역할:

```text
Dockerfile
→ Image를 어떻게 만들 것인지 정의

requirements.txt
→ 필요한 Python Package 정의

app.py
→ REST API 기반 Model Serving Application

model/
→ MLflow Registry에서 내려받은 champion 모델
```

---

# 14. Champion 모델을 로컬로 내려받기

3단원에서 등록한 모델:

```text
iris-random-forest
```

사용할 Alias:

```text
champion
```

모델 URI:

```text
models:/iris-random-forest@champion
```

먼저 MLflow Tracking Server가 실행 중인지 확인함.

터미널 1:

```bash
cd ~/mlops-week03
source .venv/bin/activate

mlflow server \
  --host 127.0.0.1 \
  --port 5000
```

새 터미널에서:

```bash
cd ~/mlops-week04
source .venv/bin/activate

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

cd ~/mlops-week04/docker-serving
```

기존 model 디렉터리가 있으면 삭제:

```bash
rm -rf model
```

Champion 모델을 내려받음:

```bash
python - <<'PY'
import os
import shutil
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

uri = "models:/iris-random-forest@champion"

downloaded = mlflow.artifacts.download_artifacts(
    artifact_uri=uri
)

if os.path.exists("model"):
    shutil.rmtree("model")

shutil.copytree(downloaded, "model")

print("Downloaded model:", downloaded)
print("Copied to       : ./model")
PY
```

확인:

```bash
find model -maxdepth 2 -type f | head -20
```

`MLmodel` 파일이 존재하는지 확인:

```bash
ls model/MLmodel
```

핵심:

```text
MLflow Registry
      ↓
champion Model
      ↓
로컬 model/ 디렉터리
      ↓
Docker Image 안에 COPY
```

> Registry에 있는 모델을 Container 실행 시마다 원격으로 읽는 것이 아니라,
> 이번 실습에서는 모델 자체를 Image 안에 포함하여 하나의 배포 단위로 만듦

---

# 15. requirements.txt 작성

Docker Image 안에서 필요한 Python Package를 정의함.

```bash
cat > requirements.txt <<'REQ'
mlflow==3.18.0
scikit-learn
numpy
flask
gunicorn
REQ
```

확인:

```bash
cat requirements.txt
```

각 Package 역할:

```text
mlflow
→ 저장된 MLflow Model 로드

scikit-learn
→ RandomForest 모델 실행

numpy
→ 입력 데이터를 배열 형태로 처리

flask
→ REST API Application 작성

gunicorn
→ Container 내부에서 Flask Application 실행
```

---

# 16. Model Serving Application 작성

`app.py` 생성:

```bash
cat > app.py <<'PY'
import os

import mlflow.pyfunc
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")

model = mlflow.pyfunc.load_model(MODEL_PATH)


@app.get("/ping")
def ping():
    return "OK", 200


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/invocations")
def invocations():
    body = request.get_json()

    if body is None or "inputs" not in body:
        return jsonify(
            {
                "error": 'JSON body must contain "inputs".'
            }
        ), 400

    inputs = body["inputs"]

    predictions = model.predict(inputs)

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    return jsonify(
        {
            "predictions": predictions
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
    )
PY
```

확인:

```bash
cat app.py
```

구조:

```text
Client
  ↓
POST /invocations
  ↓
Flask Model Server
  ↓
MLflow Model
  ↓
Prediction
  ↓
JSON Response
```

---

# 17. Dockerfile 작성

Dockerfile은 Docker Image를 만드는 절차를 정의함.

생성:

```bash
cat > Dockerfile <<'DOCKER'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY model ./model

ENV MODEL_PATH=/app/model

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "app:app"]
DOCKER
```

확인:

```bash
cat Dockerfile
```

---

## 17.1 Dockerfile 구성 이해

### 1. Python Runtime

```dockerfile
FROM python:3.10-slim
```

의미:

```text
Python 3.10 실행환경을 Base Image로 사용
```

---

### 2. 작업 디렉터리

```dockerfile
WORKDIR /app
```

Container 내부의 기본 작업 위치:

```text
/app
```

---

### 3. Dependencies 설치

```dockerfile
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
```

구조:

```text
requirements.txt
      ↓
pip install
      ↓
MLflow / scikit-learn / Flask / Gunicorn 설치
```

---

### 4. Model Serving Application 포함

```dockerfile
COPY app.py .
```

---

### 5. ML Model 포함

```dockerfile
COPY model ./model
```

Container 내부:

```text
/app/model
```

에 MLflow Model이 저장됨.

---

### 6. Port 정의

```dockerfile
EXPOSE 8080
```

Model Server가 Container 내부의 `8080` Port를 사용함.

---

### 7. Container 시작 명령

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "app:app"]
```

Container가 실행되면 자동으로 Model Server가 시작됨.

---

# 18. Docker Image Build

현재 위치 확인:

```bash
pwd
```

예:

```text
/home/<사용자>/mlops-week04/docker-serving
```

파일 확인:

```bash
ls
```

예:

```text
Dockerfile
app.py
requirements.txt
model
```

Docker Image Build:

```bash
docker build \
  -t week04-iris-model:v1 \
  .
```

의미:

```text
docker build
→ Dockerfile을 읽어 Image 생성

-t week04-iris-model:v1
→ Image 이름과 Tag 지정

.
→ 현재 디렉터리를 Build Context로 사용
```

전체 흐름:

```text
Dockerfile
   +
app.py
   +
requirements.txt
   +
model/
   ↓
docker build
   ↓
week04-iris-model:v1
```

---

# 19. Docker Image 확인

```bash
docker images
```

다음 Image가 나타나는지 확인:

```text
week04-iris-model   v1
```

상세 확인:

```bash
docker image inspect week04-iris-model:v1
```

---

# 20. Docker Container 실행

Image를 이용하여 Container 실행:

```bash
docker run --rm \
  --name iris-model \
  -p 5002:8080 \
  week04-iris-model:v1
```

구조:

```text
Host PC                      Docker Container

localhost:5002      ─────→   8080
                                 ↓
                           Gunicorn
                                 ↓
                           Flask App
                                 ↓
                           MLflow Model
```

이 터미널은 Container 실행 상태로 유지함.

---

# 21. Port Mapping 이해

실행 명령:

```bash
-p 5002:8080
```

형식:

```text
-p HostPort:ContainerPort
```

이번 실습:

```text
Host 5002
     ↓
Container 8080
```

따라서 사용자는:

```text
http://127.0.0.1:5002
```

로 접속하지만,

Container 내부 Model Server는:

```text
0.0.0.0:8080
```

에서 실행됨.

전체 흐름:

```text
Host
  ↓
localhost:5002
  ↓
Port Mapping
  ↓
Docker Container:8080
  ↓
Model Server
  ↓
ML Model
```

---

# 22. Container 상태 확인

새 Ubuntu 터미널에서:

```bash
docker ps
```

다음 Container를 확인:

```text
iris-model
```

상태 확인:

```bash
curl http://127.0.0.1:5002/ping
```

예:

```text
OK
```

Health Check:

```bash
curl http://127.0.0.1:5002/health
```

예:

```json
{"status":"healthy"}
```

---

# 23. REST API 요청

Iris 입력 데이터:

```text
sepal length
sepal width
petal length
petal width
```

예제:

```text
5.1, 3.5, 1.4, 0.2
```

요청:

```bash
curl -X POST \
  http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2]
        ]
      }'
```

예측 결과가 JSON으로 반환되는지 확인함.

예:

```json
{
  "predictions": [0]
}
```

---

# 24. 여러 데이터를 한 번에 요청

```bash
curl -X POST \
  http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2],
          [6.0, 2.9, 4.5, 1.5],
          [6.7, 3.1, 5.6, 2.4]
        ]
      }'
```

여러 Prediction이 반환되는지 확인함.

---

# 25. Prediction 결과 확인

Iris Target:

```text
0 = setosa
1 = versicolor
2 = virginica
```

따라서:

```json
{
  "predictions": [0, 1, 2]
}
```

처럼 반환된다면 각 입력 데이터를 모델이 해당 Class로 분류한 것임.

---

# 26. Docker Image 내부 구성 정리

이번에 생성한 Image의 논리적 구조:

```text
week04-iris-model:v1
│
├─ Python Runtime
│    └─ Python 3.10
│
├─ Dependencies
│    ├─ MLflow
│    ├─ scikit-learn
│    ├─ NumPy
│    ├─ Flask
│    └─ Gunicorn
│
├─ Model Serving Application
│    └─ app.py
│
└─ ML Model
     └─ model/
```

즉:

```text
Model Server
+
Python Runtime
+
Dependencies
+
ML Model
      ↓
Docker Image
```

로 패키징한 것임.

---

# 27. 로컬 서빙과 Docker 서빙 비교

2차시:

```text
Host Python 환경
      ↓
MLflow Model Server
      ↓
REST API
```

3차시:

```text
Docker Image
      ↓
Docker Container
      ↓
Model Server
      ↓
REST API
```

비교:

| 구분 | 2차시 로컬 서빙 | 3차시 Docker 서빙 |
|---|---|---|
| Python Runtime | Host 환경 사용 | Image 내부 포함 |
| Package | Host에 설치 | Image에 설치 |
| Model Server | Host Process | Container 내부 |
| Model | Host에서 로드 | Image 내부 포함 |
| 실행 단위 | Python Process | Docker Container |
| 이동성 | 낮음 | 높음 |
| 재현성 | Host 환경 영향 | Image 기준으로 일관성 높음 |

---

# 28. 동일 Image로 다시 실행

실행 중인 Container 중지:

```bash
docker stop iris-model
```

동일한 Image로 다시 실행:

```bash
docker run --rm \
  --name iris-model \
  -p 5002:8080 \
  week04-iris-model:v1
```

같은 REST API 요청을 다시 보냄.

```bash
curl -X POST \
  http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2]
        ]
      }'
```

동일한 형태의 Prediction 결과가 나오는지 확인함.

핵심:

```text
동일 Docker Image
      ↓
동일 Runtime
+
동일 Dependencies
+
동일 Model
+
동일 Application
```

따라서 다른 Docker 실행환경에서도 동일한 서빙 구성을 재현하기 쉬움.

---

# 29. Container 종료

실행 터미널에서:

```text
Ctrl + C
```

또는 다른 터미널에서:

```bash
docker stop iris-model
```

확인:

```bash
docker ps
```

Image는 그대로 남아 있음.

```bash
docker images
```

---

# 30. 3차시 핵심 정리

이번 실습 전체 흐름:

```text
MLflow Registry
      ↓
champion Model
      ↓
model/ 다운로드
      ↓
Model Serving Application 작성
      ↓
requirements.txt 작성
      ↓
Dockerfile 작성
      ↓
docker build
      ↓
Docker Image
      ↓
docker run
      ↓
Docker Container
      ↓
Port Mapping
      ↓
REST API
      ↓
Prediction
```

슬라이드의 흐름과 연결:

```text
1. Dockerfile 작성
2. 필요 Package 정의 및 설치
3. Model Serving 환경 구성
4. Docker Image Build
5. Container 실행
6. Port Mapping
7. REST API 요청
8. Prediction 결과 확인
```

핵심 개념:

> Dockerfile은 모델 서빙 환경을 어떤 순서와 구성으로 Image에 담을지 정의하는 설계도임

> Docker Image는 Model Server, Python Runtime, Dependencies, Model을 포함한 실행 가능한 배포 패키지임

> Docker Container는 그 Image를 실제로 실행한 인스턴스임

---


# 4차시 실습 — 다중 인스턴스와 부하분산

# 21. 실습 목표

- 하나의 모델 서버만 사용할 때의 한계를 이해함
- 동일한 모델 컨테이너를 여러 개 실행함
- Nginx를 Reverse Proxy / Load Balancer로 구성함
- 요청이 여러 모델 인스턴스로 분산되는 것을 확인함
- Scale-out과 Load Balancing의 기본 원리를 이해함

> 이번 실습은 자동 확장(Auto Scaling) 실습이 아님  
> 동일한 컨테이너를 여러 개 미리 실행하고 요청을 분산하는 수평 확장(Scale-out)과 Load Balancing의 기본 구조를 실습함

---

# 22. 단일 인스턴스의 한계

```text
Clients
   ↓
Model Server 1
```

Model Server 하나에 장애가 발생하면:

```text
Service Down
```

요청이 많아지면:

```text
Model Server 1
      ↓
처리 지연
```

이를 개선하는 기본 구조:

```text
               ┌→ Model 1
Client → LB ───┼→ Model 2
               └→ Model 3
```

---

# 23. Docker Compose 프로젝트 구성

작업 디렉터리:

```bash
cd ~/mlops-week04
mkdir -p loadbalancer
cd loadbalancer
```

구조:

```text
loadbalancer/
├── compose.yaml
└── nginx.conf
```

---

# 24. Nginx 설정

`nginx.conf` 생성:

```bash
cat > nginx.conf <<'NGINX'
events {}

http {
    upstream model_backend {
        server model1:8080;
        server model2:8080;
        server model3:8080;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://model_backend;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            # 어떤 Backend가 응답했는지 확인하기 위한 실습용 Header
            add_header X-Upstream-Addr $upstream_addr always;
        }
    }
}
NGINX
```

---

# 25. Docker Compose 작성

`compose.yaml` 생성:

```bash
cat > compose.yaml <<'YAML'
services:

  model1:
    image: week04-iris-model
    restart: unless-stopped

  model2:
    image: week04-iris-model
    restart: unless-stopped

  model3:
    image: week04-iris-model
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - model1
      - model2
      - model3
YAML
```

---

# 26. 다중 모델 서버 실행

```bash
docker compose up -d
```

상태 확인:

```bash
docker compose ps
```

다음 4개 서비스가 실행되어야 함

```text
model1
model2
model3
nginx
```

---

# 27. Nginx를 통해 예측 요청

이제 사용자는 개별 모델 서버의 주소를 알 필요가 없음

요청 주소:

```text
http://127.0.0.1:8080
```

예측 요청:

```bash
curl -i -X POST \
  http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2]
        ]
      }'
```

HTTP Header에서 다음 값을 확인함

```text
X-Upstream-Addr:
```

이 값은 실제 요청을 처리한 모델 컨테이너의 주소임

---

# 28. 여러 번 요청하여 부하분산 확인

다음 명령을 여러 번 실행함

```bash
for i in {1..9}
do
  echo "===== request $i ====="

  curl -s -D - \
    -o /dev/null \
    -X POST \
    http://127.0.0.1:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{
          "inputs": [
            [5.1, 3.5, 1.4, 0.2]
          ]
        }' \
    | grep -i X-Upstream-Addr

done
```

요청을 처리한 Backend 주소가 바뀌는지 확인함

Nginx의 기본 Round Robin 방식으로 여러 Backend에 요청이 분산됨

---

# 29. 전체 응답도 함께 확인

```bash
for i in {1..6}
do
  echo
  echo "===== request $i ====="

  curl -i -X POST \
    http://127.0.0.1:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{
          "inputs": [
            [6.0, 2.9, 4.5, 1.5]
          ]
        }'

  echo

done
```

확인:

```text
Prediction은 동일
X-Upstream-Addr는 달라질 수 있음
```

즉 동일한 Docker Image를 사용하므로 각 인스턴스의 모델은 동일하지만, 요청은 서로 다른 인스턴스가 처리함

---

# 30. 컨테이너 로그 확인

```bash
docker compose logs model1
```

```bash
docker compose logs model2
```

```bash
docker compose logs model3
```

Nginx 로그:

```bash
docker compose logs nginx
```

실시간 로그:

```bash
docker compose logs -f
```

종료:

```text
Ctrl + C
```

---

# 31. 하나의 Model Instance 중지

고가용성 개념을 간단히 확인함

```bash
docker compose stop model2
```

상태:

```bash
docker compose ps
```

다시 요청:

```bash
curl -i -X POST \
  http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": [
          [5.1, 3.5, 1.4, 0.2]
        ]
      }'
```

나머지 인스턴스에서 응답하는지 확인함

> Nginx는 실패한 Backend를 일시적으로 제외하고 나머지 Backend로 요청을 전달할 수 있음  
> 이 실습은 고가용성의 기본 개념을 체감하기 위한 간단한 예시임

중지했던 인스턴스 다시 실행:

```bash
docker compose start model2
```

---

# 32. Scale-up과 Scale-out

## Scale-up

한 서버의 자원을 크게 만듦

```text
Model Server

CPU 2 → CPU 8
RAM 4GB → RAM 32GB
```

## Scale-out

동일한 서버의 개수를 늘림

```text
Model 1
Model 2
Model 3
```

이번 실습은:

```text
Scale-out
+
Load Balancing
```

을 직접 구현한 것임

---

# 33. Load Balancer의 역할

```text
                ┌→ Model 1
                │
Client → Nginx ─┼→ Model 2
                │
                └→ Model 3
```

Nginx의 역할:

- 사용자의 단일 진입점 제공
- 여러 Backend에 요청 분산
- 개별 Backend 주소 숨김
- 장애 인스턴스에 대한 기본적인 우회 가능

---

# 34. 이번 실습과 Kubernetes의 관계

이번 실습:

```text
Docker Container × 3
        ↓
Nginx
```

Kubernetes 환경에서는 유사한 문제를 다음 구성요소가 담당함

```text
Docker Container 개념
        ↓
Pod

여러 Model Instance
        ↓
Replica

Replica 수 관리
        ↓
Deployment

내부 부하분산
        ↓
Service

외부 HTTP 진입
        ↓
Ingress / Gateway
```

이번 실습에서는 컨테이너 수를 사람이 직접 정했음

Kubernetes의 HPA 같은 기능을 사용하면 CPU 등의 지표를 기반으로 Replica 수를 자동으로 증감할 수 있음

즉:

```text
이번 실습
수동 Scale-out + Load Balancing

Kubernetes / Cloud
자동화된 Scale-out + Load Balancing 가능
```

---

# 35. 실습 종료 및 정리

전체 컨테이너 종료:

```bash
docker compose down
```

확인:

```bash
docker ps
```

Docker Image는 그대로 남아 있음

```bash
docker images
```

---

# 36. 4단원 전체 실습 흐름

```text
MLflow Model Registry
        ↓
champion
        ↓
[2차시]
mlflow models serve
        ↓
Local REST API
        ↓
HTTP Inference
        ↓
[3차시]
mlflow models build-docker
        ↓
Docker Image
        ↓
Container
        ↓
[4차시]
Container × 3
        ↓
Nginx
        ↓
Load Balancing
```

---

# 37. 핵심 개념 정리

## 1. Model Serving

학습된 모델을 다른 시스템이 호출할 수 있는 서비스로 실행함

```text
Model
 ↓
Model Server
 ↓
API
```

---

## 2. REST API

클라이언트와 모델 서버가 HTTP를 통해 요청과 응답을 주고받음

```text
POST /invocations
```

입력:

```text
JSON
```

출력:

```text
Prediction
```

---

## 3. Containerization

모델과 실행 환경을 Docker Image로 패키징함

```text
Model
+
Runtime
+
Dependencies
      ↓
Docker Image
```

---

## 4. Instance

동일한 Docker Image로 여러 Container를 실행할 수 있음

```text
Image
 ├→ Container 1
 ├→ Container 2
 └→ Container 3
```

---

## 5. Load Balancing

사용자의 요청을 여러 Model Instance에 분산함

```text
Client
  ↓
Load Balancer
  ↓
Model Instances
```

---

## 6. Scale-out

서버 한 대를 크게 만드는 것이 아니라 동일한 서비스 인스턴스의 수를 늘림

```text
1 Instance
   ↓
3 Instances
```

---

# 38. 3단원와 4단원 연결

```text
3단원

Experiment
   ↓
Run
   ↓
Model Registry
   ↓
champion

4단원

champion
   ↓
Model Serving
   ↓
REST API
   ↓
Docker
   ↓
Multi Instance
   ↓
Load Balancing
```

---

# 39. 다음 단원 연결

5단원에서는 지금까지 사람이 직접 실행하던 과정을 자동화함

현재:

```text
개발자
 ↓
명령 실행
 ↓
학습
 ↓
테스트
 ↓
배포 준비
```

5단원:

```text
Git Push
   ↓
GitHub Actions
   ↓
Test
   ↓
Pipeline Trigger
   ↓
자동 실행
```

그리고 ZenML을 이용하여:

```text
Preprocess
   ↓
Train
   ↓
Evaluate
```

과정을 하나의 ML Pipeline으로 연결함

---

# 40. 선택 과제

1. `model1`, `model2`, `model3` 중 하나를 중지한 뒤 요청이 계속 처리되는지 확인함
2. 모델 인스턴스를 2개만 사용하도록 `nginx.conf`와 `compose.yaml`을 수정함
3. 모델 인스턴스를 4개로 늘려봄
4. 서로 다른 Iris 입력값을 10번 이상 전송함
5. Nginx의 `X-Upstream-Addr` Header를 이용하여 어느 Backend가 요청을 처리했는지 기록함
6. Scale-up과 Scale-out의 차이를 자신의 말로 설명함
7. Docker Compose에서 구현한 구조가 Kubernetes의 Pod / Deployment / Service 중 무엇에 대응되는지 정리함
