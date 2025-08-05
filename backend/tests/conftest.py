"""
T-006: pytest 설정 및 공통 픽스처
테스트 환경 구성과 재사용 가능한 테스트 유틸리티 제공
"""

import pytest
import asyncio
import tempfile
import os
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient

# 테스트용 이미지 데이터
TEST_IMAGE_DATA = {
    "small_jpeg": b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x02\x01\x01\x02\x02\x02\x02\x02\x02\x02\x02\x03\x05\x03\x03\x03\x03\x03\x06\x04\x04\x03\x05\x07\x06\x07\x07\x07\x06\x07\x07\x08\t\x0b\t\x08\x08\n\x08\x07\x07\n\r\n\n\x0b\x0c\x0c\x0c\x0c\x07\t\x0e\x0f\r\x0c\x0e\x0b\x0c\x0c\x0c\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9',
    "width": 1,
    "height": 1,
    "channels": 3
}


@pytest.fixture(scope="session")
def event_loop():
    """세션 범위의 이벤트 루프 생성"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """임시 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image_data() -> bytes:
    """테스트용 이미지 데이터"""
    return TEST_IMAGE_DATA["small_jpeg"]


@pytest.fixture
def sample_cv_image() -> np.ndarray:
    """테스트용 OpenCV 이미지 (numpy array)"""
    # 100x100 RGB 이미지 생성
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_file_service():
    """FileService 모의 객체"""
    mock = MagicMock()
    mock.save_uploaded_file = AsyncMock()
    mock.get_file_content = AsyncMock()
    mock.delete_file = AsyncMock()
    return mock


@pytest.fixture
def mock_queue_service():
    """QueueService 모의 객체"""
    mock = MagicMock()
    mock.enqueue_task = AsyncMock()
    mock.get_task_status = AsyncMock()
    mock.get_task_result = AsyncMock()
    mock.get_task_error = AsyncMock()
    mock.get_queue_size = AsyncMock(return_value=0)
    return mock


@pytest.fixture
def mock_preprocessing_service():
    """PreprocessingService 모의 객체"""
    mock = MagicMock()
    mock.is_preprocessing_enabled = MagicMock(return_value=True)
    mock.preprocess_image = AsyncMock()
    mock.convert_result_to_bytes = MagicMock()
    mock.get_pipeline_status = MagicMock(return_value={"enabled": True})
    return mock


@pytest.fixture
def mock_model_server():
    """PyTorchModelServer 모의 객체"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.infer_async = AsyncMock()
    mock.get_health_status = MagicMock(return_value={
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu"
    })
    mock.get_model_info = MagicMock(return_value={
        "model_name": "test_model",
        "model_version": "1.0.0"
    })
    return mock


@pytest.fixture
def mock_inference_result():
    """모의 추론 결과"""
    from services.pytorch_model_server import OCRResult, InferenceResult
    
    mock_ocr_result = OCRResult(
        text="테스트 텍스트",
        confidence=0.95,
        bbox=(10, 10, 100, 30)
    )
    
    return InferenceResult(
        results=[mock_ocr_result],
        total_text="테스트 텍스트",
        processing_time=0.150,
        model_version="test_model_v1",
        device_used="cpu",
        memory_usage={"cpu_percent": 45.0}
    )


@pytest.fixture
def test_upload_file():
    """테스트용 UploadFile 모의 객체"""
    from fastapi import UploadFile
    import io
    
    file_content = TEST_IMAGE_DATA["small_jpeg"]
    file_obj = io.BytesIO(file_content)
    
    upload_file = UploadFile(
        filename="test_image.jpg",
        file=file_obj,
        size=len(file_content),
        headers={"content-type": "image/jpeg"}
    )
    upload_file.content_type = "image/jpeg"
    
    return upload_file


@pytest.fixture
def app():
    """FastAPI 애플리케이션 인스턴스"""
    from app.main import app
    return app


@pytest.fixture
def client(app) -> TestClient:
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """비동기 테스트 클라이언트"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def mock_settings():
    """테스트용 설정 모의"""
    from core.config import settings
    
    # 테스트용 설정 오버라이드
    original_upload_dir = settings.upload_dir
    original_debug = settings.debug
    
    settings.upload_dir = "test_uploads"
    settings.debug = True
    
    yield settings
    
    # 원래 설정 복원
    settings.upload_dir = original_upload_dir
    settings.debug = original_debug


@pytest.fixture
def redis_mock():
    """Redis 모의 객체"""
    import fakeredis
    return fakeredis.FakeRedis()


@pytest.fixture(scope="function")
def clean_test_files():
    """테스트 후 생성된 파일 정리"""
    created_files = []
    
    def track_file(filepath: str):
        created_files.append(filepath)
    
    yield track_file
    
    # 테스트 후 파일 정리
    for filepath in created_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass


@pytest.fixture
def mock_torch():
    """PyTorch 모의 객체 (GPU 관련 테스트용)"""
    import sys
    from unittest.mock import MagicMock
    
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.OutOfMemoryError = Exception
    torch_mock.device.return_value = "cpu"
    
    sys.modules['torch'] = torch_mock
    yield torch_mock
    
    # 정리
    if 'torch' in sys.modules:
        del sys.modules['torch']


# 마커별 설정
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "integration: 통합 테스트 (외부 의존성 포함)"
    )
    config.addinivalue_line(
        "markers", "unit: 단위 테스트 (격리된 테스트)"
    )
    config.addinivalue_line(
        "markers", "slow: 실행이 오래 걸리는 테스트"
    )
    config.addinivalue_line(
        "markers", "gpu: GPU가 필요한 테스트"
    )


def pytest_collection_modifyitems(config, items):
    """테스트 수집 후 자동 마킹"""
    for item in items:
        # 통합 테스트 자동 마킹
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # 단위 테스트 자동 마킹  
        if "unit" in item.nodeid or "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # GPU 테스트 자동 마킹
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)


# 환경 변수 설정 (테스트용)
@pytest.fixture(autouse=True)
def set_test_env():
    """테스트 환경 변수 설정"""
    os.environ.setdefault("TESTING", "1")
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("REDIS_DB", "1")  # 테스트용 DB
    yield
    # 환경 변수 정리는 필요시 추가 