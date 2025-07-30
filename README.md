# AI OCR Server

금융권 내부망에서 안전하게 동작하는 웹 기반 AI OCR 솔루션입니다. 문서·이미지를 텍스트로 변환하고, 지속적 딥러닝 학습을 통해 인식률을 높여 업무 효율과 정확성을 극대화합니다.

## 🚀 주요 기능

### ✅ 완료된 기능 (Phase 1 - MVP)

- **FastAPI 기반 OCR 추론 API** - 고성능 비동기 REST API
- **이미지 업로드 및 처리 파이프라인** - Redis 큐 기반 비동기 처리
- **고급 이미지 전처리 모듈** - OpenCV & Albumentations 활용
  - 이진화, 노이즈 제거, 회전 보정, 왜곡 보정
  - YAML 기반 설정으로 유연한 파이프라인 구성
- **PyTorch OCR 모델 연동** - GPU/CPU 자동 감지 및 추론
- **포괄적인 테스트 커버리지** - 단위 테스트 및 통합 테스트

### 🔄 개발 예정 (Phase 2 & 3)

- 웹 관리 콘솔 (모델 학습, 버전 관리, 대시보드)
- 모델 학습 파이프라인 (증분 학습, GPU 스케줄링)
- 웹 뷰어 (이미지-텍스트 동시 표시, 편집)
- Auto-Label 지원 (반자동 바운딩 박스)

## 🏗️ 기술 스택

- **Backend**: Python 3.10, FastAPI, PyTorch
- **Queue**: Redis (비동기 작업 처리)
- **Image Processing**: OpenCV, Albumentations, scikit-image
- **AI/ML**: PyTorch, torchvision, transformers
- **Testing**: pytest, pytest-asyncio
- **Deploy**: Docker, Kubernetes (예정)

## 📋 시스템 요구사항

- Python 3.10+
- Redis Server
- CUDA (GPU 사용 시, 선택사항)
- 최소 8GB RAM (GPU 미사용 시)

## 🚀 빠른 시작

### 1. 프로젝트 클론 및 의존성 설치

```bash
git clone https://github.com/answhdgkr95/AI-OCR.git
cd AI-OCR/backend
pip install -r requirements.txt
```

### 2. Redis 서버 실행

```bash
# Windows (Redis 설치 후)
redis-server

# Docker 사용
docker run -d -p 6379:6379 redis:alpine
```

### 3. 애플리케이션 실행

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. API 테스트

```bash
# 헬스 체크
curl http://localhost:8000/health

# OCR 워커 상태 확인
curl http://localhost:8000/workers/status

# 이미지 업로드 및 OCR 처리
curl -X POST "http://localhost:8000/api/v1/ocr/upload" \
     -F "file=@sample_image.jpg"
```

## 📖 API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 설정

### 환경 변수 (.env 파일)

```env
# 애플리케이션 설정
APP_NAME="AI OCR 서버"
DEBUG=false

# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 파일 업로드 설정
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB

# 작업 큐 설정
MAX_WORKERS=4
TASK_TIMEOUT=300
```

### 전처리 설정 (configs/preprocessing.yaml)

```yaml
enabled: true
pipeline:
  resize:
    enabled: true
    max_width: 2048
    max_height: 2048
  grayscale:
    enabled: true
  denoise:
    enabled: true
    method: "gaussian_blur"
  rotation_correction:
    enabled: true
    method: "hough_lines"
  binarization:
    enabled: true
    method: "adaptive_threshold"
```

### 모델 설정 (configs/model.yaml)

```yaml
model:
  name: "ocr_model_v1"
  model_path: "models/ocr_model.pt"
  
device:
  auto_detect: true
  preferred: "cuda"  # or "cpu"
  
inference:
  batch_size: 1
  confidence_threshold: 0.5
```

## 🧪 테스트 실행

```bash
cd backend

# 전체 테스트 실행
python -m pytest

# 특정 모듈 테스트
python -m pytest tests/test_pytorch_model_server.py -v

# 커버리지 포함 테스트
python -m pytest --cov=services tests/
```

## 🏗️ 프로젝트 구조

```
AI-OCR/
├── backend/
│   ├── api/                    # FastAPI 라우터 및 엔드포인트
│   │   └── v1/
│   │       ├── endpoints/      # API 엔드포인트
│   │       └── schemas/        # Pydantic 스키마
│   ├── app/                    # 애플리케이션 진입점
│   ├── configs/                # 설정 파일 (YAML)
│   ├── core/                   # 코어 설정 및 유틸리티
│   ├── services/               # 비즈니스 로직
│   │   ├── file_service.py     # 파일 처리
│   │   ├── queue_service.py    # Redis 큐 관리
│   │   ├── ocr_worker.py       # OCR 작업 워커
│   │   ├── preprocessing_service.py     # 전처리 기본 기능
│   │   ├── image_processors.py          # 개별 전처리 단계
│   │   ├── image_preprocessing_pipeline.py  # 통합 전처리
│   │   └── pytorch_model_server.py      # PyTorch 모델 서버
│   ├── tests/                  # 테스트 코드
│   └── requirements.txt        # Python 의존성
├── models/                     # PyTorch 모델 파일 (*.pt, *.pth)
├── uploads/                    # 업로드된 파일 저장소
└── README.md
```

## 🎯 성능 목표

- **정확도**: 95% 이상
- **응답 시간**: A4 1페이지 1초 이하
- **동시 처리**: 초당 10건
- **가용성**: 99.9%

## 🔐 보안 고려사항

- 내부망 전용 설계 (외부 API 호출 없음)
- 파일 업로드 검증 및 크기 제한
- 접근 로그 기록 및 추적
- ISMS-P 기준 보안 개발 가이드라인 준수

## 📝 라이선스

이 프로젝트는 금융권 내부 사용을 위한 프로젝트입니다.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 지원

문제가 있거나 문의사항이 있으시면 이슈를 생성해 주세요.

---

**Built with ❤️ for 금융권 디지털 혁신** 