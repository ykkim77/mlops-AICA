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

# 3차시 실습 — 모델을 Docker 이미지로 패키징

# 11. 실습 목표

- 로컬 프로세스 기반 서빙의 한계를 이해함
- MLflow 모델을 Docker Image로 패키징함
- Docker Container로 모델 서버를 실행함
- 동일한 REST API를 컨테이너 환경에서도 호출함
- 모델 + 런타임 + 의존성을 함께 배포하는 개념을 이해함

---

# 11.1 Docker Desktop 설치 및 WSL2 연동

이번 실습은 **Windows + WSL2 Ubuntu + VS Code** 환경을 기준으로 진행함.

```text
Windows
  ↓
Docker Desktop
  ↓
WSL2 Backend
  ↓
Ubuntu 22.04 WSL Integration
  ↓
Ubuntu 터미널에서 docker / docker compose 사용
```

> 이번 실습에서는 WSL Ubuntu 내부에 Docker Engine을 별도로 설치하지 않음  
> Docker Desktop의 Docker Engine을 WSL2 Ubuntu에서 사용하는 구조임

---

## 11.1.1 WSL2 상태 확인

Windows PowerShell에서 실행:

```powershell
wsl -l -v
```

예:

```text
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

확인할 내용:

- `Ubuntu-22.04`가 존재함
- `VERSION`이 `2`임

VERSION이 1이면:

```powershell
wsl --set-version Ubuntu-22.04 2
```

WSL 버전 확인:

```powershell
wsl --version
```

업데이트가 필요하면:

```powershell
wsl --update
```

---

## 11.1.2 Docker Desktop 설치

Windows에 **Docker Desktop**을 설치함.

설치 시 가능한 경우 **WSL 2 backend**를 사용하도록 선택함.

설치 완료 후 Windows 시작 메뉴에서:

```text
Docker Desktop
```

을 실행함.

> Docker Desktop은 Windows에 설치하는 프로그램임  
> 이번 실습에서는 Ubuntu WSL 내부에서 `apt install docker...` 방식으로 Docker Engine을 따로 설치하지 않음

---

## 11.1.3 WSL2 Backend 확인

Docker Desktop에서:

```text
Settings
→ General
```

로 이동하여 WSL2 기반 Engine을 사용하는지 확인함.

환경에 따라 WSL2 backend가 기본값으로 적용되어 별도 선택 항목이 보이지 않을 수도 있음.

```text
Docker Desktop
      ↓
WSL2 Backend
      ↓
Linux Container 실행
```

---

## 11.1.4 Ubuntu 22.04 WSL Integration 활성화

Docker Desktop에서:

```text
Settings
→ Resources
→ WSL Integration
```

으로 이동함.

다음을 활성화:

```text
Enable integration with my default WSL distro
```

배포판 목록에서:

```text
Ubuntu-22.04
```

를 활성화함.

설정 변경 후:

```text
Apply
```

또는

```text
Apply & Restart
```

를 선택함.

---

## 11.1.5 WSL 재시작

Docker Desktop 설치 전에 Ubuntu WSL이 실행 중이었다면
새 Docker 연동 설정이 반영되지 않을 수 있음.

Windows PowerShell에서:

```powershell
wsl --shutdown
```

그 후:

1. Docker Desktop 실행 상태 확인
2. Ubuntu 22.04 다시 실행
3. VS Code에서 WSL 프로젝트 다시 열기

Ubuntu 터미널에서:

```bash
cd ~/mlops-week04
```

---

## 11.1.6 Ubuntu에서 Docker 확인

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

정상이라면 `Client`와 `Server` 정보가 모두 나타남.

```text
Client
→ Docker 명령을 실행하는 CLI

Server
→ 실제 Container를 관리하는 Docker Engine
```

`Server`까지 출력되면 WSL Ubuntu가 Docker Desktop Engine과 정상 연결된 것임.

---

## 11.1.7 hello-world 실행

```bash
docker run --rm hello-world
```

처음 실행하면 로컬에 Image가 없으므로 Docker Hub에서 자동으로 내려받음.

```text
docker run hello-world
        ↓
로컬 Image 확인
        ↓
Image가 없으면 Pull
        ↓
Container 생성
        ↓
프로그램 실행
        ↓
Container 종료
```

정상 메시지가 출력되면 Docker 실행 환경 준비 완료.

---

## 11.1.8 프로젝트 위치 확인

이번 실습은 WSL Linux 파일 시스템에서 진행함.

```bash
pwd
```

예:

```text
/home/student/mlops-week04
```

가능하면 다음과 같은 Windows 마운트 경로에서 실습하지 않음:

```text
/mnt/c/Users/...
```

이번 실습 권장 위치:

```text
~/mlops-week04
```

---

## 11.1.9 Docker Engine 중복 설치 주의

이번 실습에서는:

```text
Windows Docker Desktop
        ↓
Docker Engine
        ↓
WSL Integration
        ↓
Ubuntu에서 Docker CLI 사용
```

구조를 사용함.

따라서 WSL Ubuntu 내부에 다음과 같이 Docker Engine을 별도로 설치하지 않음:

```bash
sudo apt install docker.io
```

또는 Docker CE를 별도로 설치하는 방식.

Docker Desktop과 WSL 내부 Docker Engine을 동시에 사용하면:

- 어떤 Docker Engine을 사용하는지 혼동
- Docker socket 충돌
- 권한 문제
- 서로 다른 Container / Image 목록 표시

등의 문제가 발생할 수 있음.

> 이번 강의에서는 **Docker Desktop + WSL2 Integration 방식으로 통일**

---

## 11.1.10 자주 발생하는 오류

### 오류 1. `docker: command not found`

다음을 확인:

1. Docker Desktop 설치 여부
2. Docker Desktop 실행 여부
3. `Settings → Resources → WSL Integration`에서 Ubuntu-22.04 활성화 여부

설정 후 PowerShell에서:

```powershell
wsl --shutdown
```

Ubuntu를 다시 실행함.

### 오류 2. `Cannot connect to the Docker daemon`

Docker Desktop이 실행 중인지 확인하고:

```bash
docker version
```

을 다시 실행함.

Client만 나오고 Server가 나오지 않으면 Docker Engine 연결 문제임.

### 오류 3. WSL Integration 메뉴가 보이지 않음

WSL 상태 확인:

```powershell
wsl -l -v
```

Ubuntu가 `VERSION 2`인지 확인함.

### 오류 4. `docker compose` 명령이 없음

확인:

```bash
docker compose version
```

이번 실습에서는 과거 방식인:

```bash
docker-compose
```

보다 현재 Compose Plugin 방식인:

```bash
docker compose
```

를 사용함.

---

## 11.1.11 설치 완료 체크리스트

아래 명령이 모두 정상 실행되면 Docker 실습 준비 완료:

```bash
docker --version
docker compose version
docker version
docker run --rm hello-world
```

최종 환경:

```text
Windows
├─ Docker Desktop
│    └─ Docker Engine
│
└─ WSL2 Ubuntu 22.04
     ├─ VS Code
     ├─ Python / MLflow
     ├─ docker CLI
     └─ docker compose
              ↓
        Docker Desktop Engine
```

---

# 12. Docker 동작 최종 확인

앞 단계에서 Docker Desktop 설치와 WSL2 연동을 완료했으므로,
3차시 본 실습을 시작하기 전에 Ubuntu 터미널에서 다시 확인함.

```bash
docker version
docker compose version
```

간단한 Container 실행 테스트:

```bash
docker run --rm hello-world
```

정상 출력이 나타나면 Docker 환경 준비 완료.

---

# 13. 왜 컨테이너가 필요한가?

2차시의 실행 방식:

```text
내 PC
 ├─ Python
 ├─ MLflow
 ├─ scikit-learn
 └─ Model Server
```

다른 컴퓨터에서 실행하려면 동일한 환경을 다시 맞춰야 함

Docker를 사용하면:

```text
Docker Image
 ├─ OS Runtime
 ├─ Python
 ├─ MLflow
 ├─ Model Dependencies
 └─ Model
```

을 하나의 실행 가능한 패키지로 만들 수 있음

---

# 14. MLflow 모델 Docker Image 빌드

3단원 Model Registry의 `champion` 모델을 이미지로 만듦

```bash
cd ~/mlops-week04

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models build-docker \
  --model-uri "models:/iris-random-forest@champion" \
  --name week04-iris-model
```

빌드에는 시간이 걸릴 수 있음

---

# 15. Docker Image 확인

```bash
docker images
```

다음 이미지가 나타나는지 확인함

```text
week04-iris-model
```

필요하면:

```bash
docker image inspect week04-iris-model
```

---

# 16. Docker Container 실행

MLflow가 생성한 이미지의 Model Server는 컨테이너 내부 `8080` 포트를 사용함

실행:

```bash
docker run --rm \
  --name iris-model \
  -p 5002:8080 \
  week04-iris-model
```

구조:

```text
Host PC                 Docker Container

localhost:5002   →      8080
                         ↓
                    MLflow Server
                         ↓
                    Champion Model
```

이 터미널은 컨테이너 실행용으로 그대로 둠

---

# 17. 컨테이너 모델 상태 확인

다른 터미널에서:

```bash
curl http://127.0.0.1:5002/ping
```

예측 요청:

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

2차시의 결과와 동일한 형태의 예측이 반환되는지 확인함

---

# 18. 로컬 서빙과 Docker 서빙 비교

## 로컬 프로세스

```text
mlflow models serve
```

실행 환경:

```text
학생 PC의 Python 환경
```

## Docker

```text
docker run
```

실행 환경:

```text
Docker Image 안에 패키징된 독립 환경
```

핵심 차이:

| 구분 | 로컬 서빙 | Docker 서빙 |
|---|---|---|
| 실행 환경 | Host 환경 의존 | Image에 포함 |
| 이동성 | 낮음 | 높음 |
| 배포 일관성 | 환경에 따라 달라질 수 있음 | 동일 Image 사용 |
| 확장 | 추가 프로세스 필요 | Container 복제 용이 |

---

# 19. 컨테이너 중지

`docker run`을 실행한 터미널에서:

```text
Ctrl + C
```

또는 다른 터미널에서:

```bash
docker stop iris-model
```

---

# 20. 3차시 핵심

2차시:

```text
Model
 ↓
Local Process
 ↓
REST API
```

3차시:

```text
Model
 ↓
Docker Image
 ↓
Container
 ↓
REST API
```

컨테이너화를 통해:

```text
Model
+
Runtime
+
Dependencies
```

를 하나의 배포 단위로 만들 수 있음

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
