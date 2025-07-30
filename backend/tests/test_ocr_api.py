"""
OCR API 엔드포인트 테스트
"""

import io
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestOCRAPI:
    """OCR API 테스트 클래스"""
    
    def test_root_endpoint(self):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        message = response.json()["message"]
        assert "AI OCR 서버가 정상 동작 중입니다" in message
    
    def test_health_check(self):
        """헬스 체크 엔드포인트 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_ocr_health_check(self):
        """OCR 서비스 헬스 체크 테스트"""
        response = client.get("/api/v1/ocr/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "OCR API"
        assert "supported_formats" in data
    
    def test_valid_image_upload(self):
        """유효한 이미지 파일 업로드 테스트"""
        # 가짜 이미지 데이터 생성
        fake_image_data = b"fake_jpeg_data"
        files = {
            "file": ("test_image.jpg", io.BytesIO(fake_image_data), "image/jpeg")
        }
        
        response = client.post("/api/v1/ocr/predict", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "extracted_text" in data
        assert "items" in data
        assert "processing_time_ms" in data
        assert "image_info" in data
    
    def test_invalid_file_format(self):
        """지원하지 않는 파일 형식 테스트"""
        fake_text_data = b"this is not an image"
        files = {
            "file": ("test.txt", io.BytesIO(fake_text_data), "text/plain")
        }
        
        response = client.post("/api/v1/ocr/predict", files=files)
        assert response.status_code == 400
        assert "지원하지 않는 파일 형식" in response.json()["error"]
    
    def test_empty_file(self):
        """빈 파일 업로드 테스트"""
        files = {
            "file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")
        }
        
        response = client.post("/api/v1/ocr/predict", files=files)
        assert response.status_code == 400
        assert "빈 파일입니다" in response.json()["error"]
    
    def test_large_file_rejection(self):
        """큰 파일 거부 테스트"""
        # 11MB 크기의 가짜 데이터 생성
        large_data = b"x" * (11 * 1024 * 1024)
        files = {
            "file": ("large_image.jpg", io.BytesIO(large_data), "image/jpeg")
        }
        
        response = client.post("/api/v1/ocr/predict", files=files)
        assert response.status_code == 413
        assert "파일 크기가 너무 큽니다" in response.json()["error"]
    
    def test_no_file_provided(self):
        """파일이 제공되지 않은 경우 테스트"""
        response = client.post("/api/v1/ocr/predict")
        assert response.status_code == 422  # Validation error 