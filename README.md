# Deep-Guardian: AI 기반 포트홀 탐지 및 관리 시스템

> YOLOv8 + Intel NPU/OpenVINO 기반의 도로 포트홀(도로 파임) 자동 탐지, 깊이 검증, 위험도 분석 및 웹 대시보드 시스템

## 목차
- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택](#기술-스택)
- [사전 요구사항](#사전-요구사항)
- [설치 및 실행](#설치-및-실행)
- [API 키 발급](#api-키-발급)
- [환경 변수 설정](#환경-변수-설정)
- [프로젝트 구조](#프로젝트-구조)
- [사용 방법](#사용-방법)

---

## 프로젝트 소개

**Deep-Guardian**은 도로 영상에서 포트홀(도로 파임)을 자동으로 탐지하고, 깊이 추정을 통해 위험도를 분류한 뒤, 웹 대시보드를 통해 시각화·관리하는 종합 AI 시스템입니다.

- 비디오 파일 또는 실시간 스트림에서 포트홀 자동 탐지
- Intel NPU를 활용한 깊이 추정으로 위험도 검증
- 카카오 맵 API를 이용한 좌표 → 주소 변환 및 지도 표시
- Gemini AI / Phi-3 (OpenVINO) 챗봇 기반 리포트 요약 및 Q&A
- 자동 파인튜닝 스케줄러로 모델 지속 개선
- ngrok / Cloudflare Tunnel을 통한 외부 접근 지원

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **포트홀 탐지** | YOLOv8 커스텀 모델로 비디오/이미지에서 포트홀 탐지 |
| **깊이 검증** | Intel NPU + Depth Anything V2 (OpenVINO IR)로 깊이 비율 계산 및 위험 포트홀 필터링 |
| **위험도 분류** | depth_ratio 기반 위험 등급 분류 및 위험 구역(risk_zones) 관리 |
| **위치 변환** | GPS 좌표 → 카카오 맵 API로 실제 도로 주소 변환 |
| **웹 대시보드** | Streamlit 기반 통계, 지도 시각화, 결과 영상 재생 |
| **AI 요약/챗봇** | Gemini API / Phi-3 (OpenVINO) 기반 탐지 결과 요약 및 챗봇 |
| **자동 파인튜닝** | 누적 탐지 데이터로 매일 자정 자동 모델 파인튜닝 (APScheduler) |
| **합성 데이터 생성** | 포트홀 합성 이미지 자동 생성으로 학습 데이터 증강 |
| **외부 터널링** | ngrok 또는 Cloudflare Tunnel로 외부 공개 URL 생성 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   사용자 브라우저                          │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│  Docker Compose (5 Container)                           │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐                   │
│  │  web-server  │    │  dashboard   │                   │
│  │  (Apache)   │───▶│  (Streamlit) │                   │
│  │  Port: 80   │    │  Port: 8501  │                   │
│  └─────────────┘    └──────┬───────┘                   │
│                            │                            │
│  ┌─────────────┐    ┌──────▼───────┐                   │
│  │  cloudflared │    │   ai-core    │                   │
│  │ (ngrok/CF)  │    │  (YOLOv8)   │                   │
│  └─────────────┘    └──────┬───────┘                   │
│                            │                            │
│                     ┌──────▼───────┐                   │
│                     │      db      │                    │
│                     │ (PostgreSQL) │                    │
│                     └─────────────┘                    │
└─────────────────────────────────────────────────────────┘
                           │ HTTP (host.docker.internal:9001)
┌──────────────────────────▼──────────────────────────────┐
│  Windows Host                                           │
│  ┌─────────────────────────────────────────────┐       │
│  │  NPU Worker (npu_worker.py)                 │       │
│  │  - Intel OpenVINO                           │       │
│  │  - Depth Anything V2 (IR 형식)              │       │
│  │  - Port: 9001                               │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 기술 스택

**AI / ML**
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 포트홀 객체 탐지
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) - 단안 깊이 추정 (OpenVINO IR)
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) - NPU 추론 가속
- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) - 경량 LLM 챗봇 (OpenVINO GenAI)

**Backend**
- Python, Flask, Django ORM
- PostgreSQL
- APScheduler (자동 파인튜닝)
- Kakao Map API, Gemini API

**Frontend / 인프라**
- Streamlit (대시보드)
- Apache (리버스 프록시)
- Docker / Docker Compose
- ngrok / Cloudflare Tunnel

---

## 사전 요구사항

- **OS**: Windows 10/11 (NPU Worker는 Windows 전용)
- **Docker Desktop** (WSL2 백엔드 권장)
- **Python 3.10+** (NPU Worker 실행용 Windows 가상환경)
- **Intel OpenVINO** 설치 (NPU Worker용)
- **Intel NPU** 또는 CPU (Depth Anything V2 추론)
- **Git**

---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/your_github_username/pothole-detection-system.git
cd pothole-detection-system
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 API 키를 입력하세요
```

### 3. 모델 파일 준비

`ai-core/models/` 디렉토리에 YOLOv8 커스텀 모델을 배치합니다:
```
ai-core/models/best2.pt
```

NPU Worker용 Depth Anything V2 OpenVINO 모델을 준비합니다:
```
<your_path>/depth_npu/openvino_model.xml
<your_path>/depth_npu/openvino_model.bin
```

### 4. Docker 컨테이너 실행

```powershell
# PowerShell (Windows)
docker compose up -d --build

# 상태 확인
docker compose ps
```

### 5. NPU Worker 실행 (Windows Host)

```powershell
# Python 가상환경 설치 (최초 1회)
python -m venv venv-deep-guardian
.\venv-deep-guardian\Scripts\Activate.ps1
pip install -r requirements_slm_npu.txt

# NPU Worker 시작
python npu_worker.py --model "C:\your\path\to\openvino_model.xml" --device AUTO:NPU,CPU --port 9001

# 또는 스크립트 사용 (start_npu_worker.ps1 내 경로 수정 후)
.\start_npu_worker.ps1
```

### 6. 접속

| 서비스 | URL |
|--------|-----|
| 메인 대시보드 | http://localhost |
| Streamlit 직접 접속 | http://localhost:8501 |
| NPU Worker 헬스 체크 | http://localhost:9001/health |
| PostgreSQL | localhost:5432 |

---

## API 키 발급

| API | 발급 URL | 용도 |
|-----|----------|------|
| Gemini API | https://aistudio.google.com/apikey | 탐지 결과 AI 요약, 챗봇 |
| Kakao Map API | https://developers.kakao.com/ | GPS 좌표 → 도로 주소 변환 |
| ngrok | https://dashboard.ngrok.com/get-started/your-authtoken | 외부 터널링 |

---

## 환경 변수 설정

`.env.example`을 복사하여 `.env`를 생성하고 값을 입력합니다:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
NGROK_AUTHTOKEN=your_ngrok_authtoken_here
KAKAO_MAP_APP_KEY=your_kakao_map_app_key_here
```

> **주의**: `.env` 파일은 절대 git에 커밋하지 마세요. `.gitignore`에 이미 포함되어 있습니다.

---

## 프로젝트 구조

```
pothole-detection-system/
├── ai-core/                   # AI 엔진 (YOLOv8 탐지, 파인튜닝)
│   ├── main.py                # AI Core 메인 서버
│   ├── synthetic_pothole_generator.py  # 합성 데이터 생성
│   ├── models/                # YOLOv8 모델 파일 (*.pt)
│   ├── videos/                # 처리할 입력 비디오
│   └── Dockerfile
├── apache/                    # Apache 리버스 프록시
│   ├── httpd.conf
│   └── Dockerfile
├── dashboard/                 # Streamlit 대시보드
│   ├── app.py                 # 메인 대시보드
│   ├── auth.py                # 인증 모듈
│   ├── gemini_summary.py      # Gemini AI 요약
│   ├── road_chatbot.py        # 도로 상태 챗봇
│   └── Dockerfile
├── database/
│   └── init.sql               # PostgreSQL 초기화 스크립트
├── django_app/                # Django ORM 모델
├── cloudflared/               # ngrok/Cloudflare 터널
├── inference-container/       # 추론 컨테이너 (2컨테이너 구조)
├── npu_worker.py              # Intel NPU 깊이 추정 워커
├── slm_npu_worker_phi3.py     # Phi-3 LLM 워커 (OpenVINO)
├── docker-compose.yml         # 5컨테이너 구성
├── docker-compose-optimized.yml  # 최적화 구성
├── .env.example               # 환경 변수 예시
├── requirements.txt           # 공통 Python 의존성
└── requirements_slm_npu.txt   # NPU Worker 의존성
```

---

## 사용 방법

### 비디오 처리

```bash
# ai-core 컨테이너에 비디오 복사 후 처리 요청
docker cp your_video.mp4 deep-guardian-ai:/app/videos/
```

대시보드(http://localhost:8501)에서 비디오를 선택하고 처리를 시작합니다.

### 포트홀 합성 데이터 생성

```powershell
.\generate_synthetic_potholes.ps1
```

### NPU Worker API

```bash
# 헬스 체크
curl http://localhost:9001/health

# 깊이 추정
curl -X POST http://localhost:9001/depth -F "image=@pothole.jpg"
```

### 자동 파인튜닝

매일 자정(UTC)에 누적된 탐지 이미지로 자동 파인튜닝이 실행됩니다.
`ai-core/main.py`에서 `finetune_threshold`(기본값: 100장)를 조정할 수 있습니다.

---

## 문서

| 문서 | 설명 |
|------|------|
| [QUICK_START.md](QUICK_START.md) | 빠른 시작 가이드 |
| [ARCHITECTURE_UPDATES.md](ARCHITECTURE_UPDATES.md) | 아키텍처 상세 설명 |
| [AUTHENTICATION_GUIDE.md](AUTHENTICATION_GUIDE.md) | 인증 시스템 가이드 |
| [SYNTHETIC_POTHOLES_GUIDE.md](SYNTHETIC_POTHOLES_GUIDE.md) | 합성 데이터 생성 가이드 |
| [docs/](docs/) | 전체 문서 모음 |

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
