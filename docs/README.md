# Deep-Guardian NPU Worker (Windows Host)

OpenVINO를 사용하여 Depth Anything V2 모델로 깊이 추정을 수행하는 HTTP 서비스입니다.

## 역할

- **위치**: Windows Host에서 실행
- **기능**: 포트홀 이미지의 깊이 검증
- **호출**: Docker 컨테이너(ai-core)에서 `http://host.docker.internal:9001/depth`로 호출
- **출력**: depth_ratio, validation_result 등 검증 결과

## 요구사항

- Windows 환경
- OpenVINO 설치 (venv-atomman-win 가상환경)
- Depth Anything V2 OpenVINO IR 모델 파일 (openvino_model.xml, openvino_model.bin)

## 설치

1. 가상환경 활성화:
```powershell
& $env:USERPROFILE\venv-atomman-win\Scripts\Activate.ps1
```

2. 필요한 패키지 설치:
```powershell
pip install -r requirements.txt
```

## 사용 방법

### 서버 시작

```powershell
python npu_worker.py --model openvino_model.xml --device AUTO:NPU,CPU --port 9001
```

또는 스크립트 사용:
```powershell
.\start_npu_worker.ps1
```

### 엔드포인트

#### 1. 헬스 체크
```
GET http://localhost:9001/health
```

응답 예시:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "openvino_available": true,
  "model_path": "C:\\path\\to\\openvino_model.xml"
}
```

#### 2. 모델 로드 (런타임)
```
POST http://localhost:9001/load_model
Content-Type: application/json
Body: {
  "model_path": "C:\\path\\to\\openvino_model.xml",
  "device": "AUTO:NPU,CPU"
}
```

응답 예시:
```json
{
  "success": true,
  "message": "모델 로드 완료",
  "model_path": "C:\\path\\to\\openvino_model.xml",
  "device": "AUTO:NPU,CPU"
}
```

#### 3. 깊이 추정
```
POST http://localhost:9001/depth
Content-Type: multipart/form-data
Body: image=<이미지 파일>
```

응답 예시:
```json
{
  "success": true,
  "depth_ratio": 0.2345,
  "validation_result": true,
  "depth_map_shape": [518, 518],
  "depth_min": 0.0,
  "depth_max": 1.0,
  "message": "추론 완료"
}
```

## 테스트

### 서버 테스트
```powershell
python test_npu_worker.py --image sample.jpg
```

### 모델 로드 (서버 실행 중)
서버가 모델 없이 시작된 경우, 런타임에 모델을 로드할 수 있습니다:

```powershell
python load_model_example.py "C:\path\to\openvino_model.xml"
```

또는 직접 API 호출:
```powershell
curl -X POST http://localhost:9001/load_model -H "Content-Type: application/json" -d "{\"model_path\": \"C:\\path\\to\\openvino_model.xml\"}"
```

## 파라미터

### npu_worker.py

- `--model`: OpenVINO IR 모델 XML 파일 경로 (기본값: openvino_model.xml)
- `--device`: 사용할 디바이스 (기본값: AUTO:NPU,CPU)
- `--port`: 서버 포트 (기본값: 9001)
- `--host`: 서버 호스트 (기본값: 0.0.0.0)

## Docker 컨테이너에서 호출

Docker 컨테이너(ai-core)에서 다음과 같이 호출할 수 있습니다:

```python
import requests

url = "http://host.docker.internal:9001/depth"
with open("pothole_crop.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    result = response.json()
    
    if result["success"]:
        depth_ratio = result["depth_ratio"]
        is_valid = result["validation_result"]
        # 검증 통과 시 DB 저장
```

## 주의사항

- 모델 파일(openvino_model.xml, openvino_model.bin)이 같은 디렉토리에 있어야 합니다.
- NPU는 AUTO:NPU,CPU 모드로만 동작합니다 (NPU 단독 컴파일 불가).
- 서버는 0.0.0.0으로 바인딩되어 Docker 컨테이너에서 접근 가능합니다.
