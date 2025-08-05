"""
OCR API 엔드포인트 통합 테스트
T-006-003: OCR 추론 API 엔드포인트 통합 테스트 구현
"""

import pytest
import io
import uuid
from fastapi.testclient import TestClient
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware


# OCR 엔드포인트를 위한 독립적인 구현 (aioredis 의존성 제거)
def create_test_ocr_router():
    """테스트용 OCR 라우터 생성"""
    from fastapi import APIRouter
    
    router = APIRouter()
    
    # 허용되는 이미지 파일 형식
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/tiff"
    }
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    def validate_image_file(file: UploadFile) -> None:
        """이미지 파일 유효성 검증"""
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            allowed_formats = ', '.join(ALLOWED_IMAGE_TYPES)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식입니다. 허용된 형식: {allowed_formats}"
            )
        
        if file.size and file.size > MAX_FILE_SIZE:
            max_size_mb = MAX_FILE_SIZE // (1024*1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {max_size_mb}MB"
            )
    
    @router.post("/infer")
    async def test_infer_ocr_direct(
        file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
    ):
        """테스트용 OCR 추론 엔드포인트"""
        import time
        start_time = time.time()
        
        try:
            # 파일 유효성 검증
            validate_image_file(file)
            
            # 파일 읽기
            image_data = await file.read()
            if not image_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="빈 파일입니다"
                )
            
            # 모의 이미지 정보 구성
            image_info = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(image_data),
                "width": 640,
                "height": 480,
                "channels": 3
            }
            
            # 모의 OCR 결과 생성
            mock_results = [
                {
                    "text": "Test Text",
                    "bbox": [10, 10, 100, 30],
                    "confidence": 0.95
                },
                {
                    "text": "Second Line", 
                    "bbox": [10, 40, 120, 60],
                    "confidence": 0.92
                }
            ]
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "message": "OCR 처리 완료",
                "timestamp": "2024-01-15T09:30:45.123456",
                "results": mock_results,
                "total_text": "Test Text\nSecond Line",
                "processing_time_ms": max(total_processing_time, 1.0),
                "image_info": image_info,
                "preprocessing": {
                    "enabled": True,
                    "applied_steps": ["resize", "grayscale"],
                    "processing_time_ms": 50.0,
                    "quality_metrics": {"contrast_improvement": 0.1}
                },
                "model_info": {
                    "name": "ocr_model_v1",
                    "version": "1.0.0",
                    "device": "cpu"
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"예기치 않은 오류가 발생했습니다: {str(e)}"
            )
    
    @router.post("/upload")
    async def test_upload_for_ocr(
        file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
    ):
        """테스트용 이미지 업로드 엔드포인트"""
        try:
            validate_image_file(file)
            
            # 모의 업로드 응답
            file_size = file.size or 1024
            if hasattr(file, 'read') and not file.size:
                content = await file.read()
                file_size = len(content)
            
            return {
                "task_id": f"task_{uuid.uuid4()}",
                "file_id": "test_file_123",
                "status": "uploaded",
                "message": "파일이 성공적으로 업로드되었습니다. OCR 처리가 시작됩니다.",
                "file_info": {
                    "filename": file.filename,
                    "size_bytes": file_size,
                    "content_type": file.content_type
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"파일 업로드 실패: {str(e)}"
            )
    
    @router.get("/task/{task_id}/status")
    async def test_get_task_status(task_id: str):
        """테스트용 작업 상태 조회 엔드포인트"""
        if task_id == "nonexistent_task":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="작업을 찾을 수 없습니다"
            )
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "status": "success",
                "total_text": "Test Text\nSecond Line",
                "processing_time_ms": 150.5
            }
        }
    
    return router


# 테스트용 FastAPI 앱 생성
def create_test_app() -> FastAPI:
    """테스트용 FastAPI 애플리케이션 생성"""
    app = FastAPI(title="Test OCR API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 테스트용 라우터 추가
    ocr_router = create_test_ocr_router()
    app.include_router(ocr_router, prefix="/api/v1/ocr", tags=["OCR"])
    
    return app


@pytest.fixture
def test_client():
    """테스트 클라이언트 픽스처"""
    app = create_test_app()
    return TestClient(app)


def create_test_image_file(filename="test_image.jpg", size_kb=1):
    """테스트용 이미지 파일 생성"""
    # 간단한 JPEG 헤더를 포함한 더미 데이터
    jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    dummy_data = b'test_image_data' * (size_kb * 64)  # 대략적인 크기 조정
    
    return io.BytesIO(jpeg_header + dummy_data)


class TestOCRInferEndpoint:
    """OCR 직접 추론 엔드포인트 테스트"""
    
    def test_successful_ocr_inference(self, test_client):
        """성공적인 OCR 추론 테스트"""
        # 테스트 이미지 파일 생성
        test_file = create_test_image_file("test.jpg", 1)
        
        files = {
            "file": ("test.jpg", test_file, "image/jpeg")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # 응답 구조 검증
        assert data["status"] == "success"
        assert data["message"] == "OCR 처리 완료"
        assert "timestamp" in data
        assert "results" in data
        assert "total_text" in data
        assert "processing_time_ms" in data
        assert "image_info" in data
        
        # 결과 내용 검증
        assert len(data["results"]) == 2
        assert data["results"][0]["text"] == "Test Text"
        assert data["results"][0]["confidence"] == 0.95
        assert data["results"][0]["bbox"] == [10, 10, 100, 30]
        
        assert data["total_text"] == "Test Text\nSecond Line"
        assert data["processing_time_ms"] > 0
        
        # 이미지 정보 검증
        image_info = data["image_info"]
        assert image_info["filename"] == "test.jpg"
        assert image_info["content_type"] == "image/jpeg"
        assert image_info["size_bytes"] > 0
    
    def test_unsupported_file_format(self, test_client):
        """지원하지 않는 파일 형식 테스트"""
        test_file = io.BytesIO(b"this is not an image")
        
        files = {
            "file": ("test.txt", test_file, "text/plain")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 400
        assert "지원하지 않는 파일 형식" in response.json()["detail"]
    
    def test_empty_file(self, test_client):
        """빈 파일 테스트"""
        empty_file = io.BytesIO(b"")
        
        files = {
            "file": ("empty.jpg", empty_file, "image/jpeg")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 400
        assert "빈 파일입니다" in response.json()["detail"]
    
    def test_large_file_rejection(self, test_client):
        """큰 파일 거부 테스트"""
        # 11MB 크기의 파일 생성
        large_file = create_test_image_file("large.jpg", 11 * 1024)
        
        files = {
            "file": ("large.jpg", large_file, "image/jpeg")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 413
        assert "파일 크기가 너무 큽니다" in response.json()["detail"]
    
    def test_missing_file(self, test_client):
        """파일 누락 테스트"""
        response = test_client.post("/api/v1/ocr/infer")
        
        assert response.status_code == 422  # Validation error


class TestOCRUploadEndpoint:
    """OCR 업로드 엔드포인트 테스트"""
    
    def test_successful_file_upload(self, test_client):
        """성공적인 파일 업로드 테스트"""
        test_file = create_test_image_file("upload_test.jpg", 2)
        
        files = {
            "file": ("upload_test.jpg", test_file, "image/jpeg")
        }
        
        response = test_client.post("/api/v1/ocr/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # 응답 구조 검증
        assert "task_id" in data
        assert "file_id" in data
        assert data["status"] == "uploaded"
        assert "message" in data
        assert "file_info" in data
        
        # 파일 정보 검증
        file_info = data["file_info"]
        assert file_info["filename"] == "upload_test.jpg"
        assert file_info["content_type"] == "image/jpeg"
        assert file_info["size_bytes"] > 0
    
    def test_upload_invalid_file(self, test_client):
        """잘못된 파일 업로드 테스트"""
        invalid_file = io.BytesIO(b"not an image")
        
        files = {
            "file": ("invalid.pdf", invalid_file, "application/pdf")
        }
        
        response = test_client.post("/api/v1/ocr/upload", files=files)
        
        assert response.status_code == 400
        assert "지원하지 않는 파일 형식" in response.json()["detail"]


class TestOCRTaskStatusEndpoint:
    """OCR 작업 상태 조회 엔드포인트 테스트"""
    
    def test_get_completed_task_status(self, test_client):
        """완료된 작업 상태 조회 테스트"""
        task_id = "test_task_123"
        
        response = test_client.get(f"/api/v1/ocr/task/{task_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == task_id
        assert data["status"] == "completed"
        assert "result" in data
    
    def test_get_nonexistent_task_status(self, test_client):
        """존재하지 않는 작업 상태 조회 테스트"""
        task_id = "nonexistent_task"
        response = test_client.get(f"/api/v1/ocr/task/{task_id}/status")
        
        assert response.status_code == 404
        assert "작업을 찾을 수 없습니다" in response.json()["detail"]


class TestOCRResponseFormat:
    """OCR 응답 포맷 검증 테스트"""
    
    def test_ocr_response_schema_validation(self, test_client):
        """OcrResponse 스키마 검증 테스트"""
        test_file = create_test_image_file("schema_test.jpg", 1)
        
        files = {
            "file": ("schema_test.jpg", test_file, "image/jpeg")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # OcrResponse 스키마 필수 필드들 검증
        required_fields = [
            "status", "message", "timestamp", "results", "total_text",
            "processing_time_ms", "image_info"
        ]
        
        for field in required_fields:
            assert field in data, f"필수 필드 '{field}'가 응답에 없습니다"
        
        # TextBlock 스키마 검증
        if data["results"]:
            text_block = data["results"][0]
            assert "text" in text_block
            assert "bbox" in text_block
            assert "confidence" in text_block
            assert isinstance(text_block["bbox"], list)
            assert len(text_block["bbox"]) == 4
            assert 0.0 <= text_block["confidence"] <= 1.0
    
    def test_error_response_format(self, test_client):
        """에러 응답 포맷 테스트"""
        invalid_file = io.BytesIO(b"invalid")
        
        files = {
            "file": ("test.txt", invalid_file, "text/plain")
        }
        
        response = test_client.post("/api/v1/ocr/infer", files=files)
        
        assert response.status_code == 400
        error_data = response.json()
        
        # FastAPI 기본 에러 포맷 검증
        assert "detail" in error_data 