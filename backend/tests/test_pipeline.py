"""
이미지 업로드 및 처리 파이프라인 테스트
"""

import asyncio
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from services.file_service import file_service
from services.queue_service import queue_service, TaskMessage


client = TestClient(app)


class TestUploadPipeline:
    """업로드 파이프라인 테스트 클래스"""
    
    def test_file_upload_success(self):
        """파일 업로드 성공 테스트"""
        fake_image_data = b"fake_jpeg_data"
        files = {
            "file": ("test_image.jpg", io.BytesIO(fake_image_data), "image/jpeg")
        }
        
        with patch.object(queue_service, 'enqueue_task', new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = True
            
            response = client.post("/api/v1/ocr/upload", files=files)
            
            assert response.status_code == 202
            data = response.json()
            assert "task_id" in data
            assert "file_id" in data
            assert data["status"] == "uploaded"
            assert "성공적으로 업로드" in data["message"]
    
    def test_file_upload_invalid_format(self):
        """잘못된 파일 형식 업로드 테스트"""
        fake_text_data = b"this is not an image"
        files = {
            "file": ("test.txt", io.BytesIO(fake_text_data), "text/plain")
        }
        
        response = client.post("/api/v1/ocr/upload", files=files)
        assert response.status_code == 400
        assert "지원하지 않는 파일 형식" in response.json()["error"]
    
    def test_task_status_not_found(self):
        """존재하지 않는 작업 상태 조회 테스트"""
        with patch.object(queue_service, 'get_task_status', new_callable=AsyncMock) as mock_status:
            mock_status.return_value = None
            
            response = client.get("/api/v1/ocr/task/non-existent-task/status")
            assert response.status_code == 404
            assert "작업을 찾을 수 없습니다" in response.json()["error"]
    
    def test_task_status_pending(self):
        """대기 중인 작업 상태 조회 테스트"""
        task_id = "test-task-123"
        
        with patch.object(queue_service, 'get_task_status', new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "status": "pending",
                "updated_at": "2024-01-01T00:00:00Z"
            }
            
            response = client.get(f"/api/v1/ocr/task/{task_id}/status")
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == task_id
            assert data["status"] == "pending"
            assert data["updated_at"] == "2024-01-01T00:00:00Z"
    
    def test_task_status_completed(self):
        """완료된 작업 상태 조회 테스트"""
        task_id = "test-task-456"
        
        mock_result = {
            "result": {
                "status": "success",
                "extracted_text": "테스트 텍스트",
                "items": [],
                "processing_time_ms": 100.0
            }
        }
        
        with patch.object(queue_service, 'get_task_status', new_callable=AsyncMock) as mock_status, \
             patch.object(queue_service, 'get_task_result', new_callable=AsyncMock) as mock_result_get:
            
            mock_status.return_value = {
                "status": "completed",
                "updated_at": "2024-01-01T00:01:00Z"
            }
            mock_result_get.return_value = mock_result
            
            response = client.get(f"/api/v1/ocr/task/{task_id}/status")
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == task_id
            assert data["status"] == "completed"
            assert data["result"] is not None
            assert data["result"]["extracted_text"] == "테스트 텍스트"
    
    def test_ocr_health_with_queue_info(self):
        """큐 정보를 포함한 OCR 헬스 체크 테스트"""
        with patch.object(queue_service, 'get_queue_size', new_callable=AsyncMock) as mock_queue_size:
            mock_queue_size.return_value = 5
            
            response = client.get("/api/v1/ocr/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["queue_size"] == 5
    
    def test_worker_status_endpoint(self):
        """워커 상태 조회 엔드포인트 테스트"""
        response = client.get("/workers/status")
        assert response.status_code == 200
        data = response.json()
        assert "worker_count" in data
        assert "running_workers" in data
        assert "total_processed" in data
        assert "workers" in data


class TestConcurrency:
    """동시성 테스트 클래스"""
    
    @pytest.mark.asyncio
    async def test_concurrent_file_uploads(self):
        """동시 파일 업로드 테스트"""
        async def upload_file(file_data, filename):
            """비동기 파일 업로드 함수"""
            files = {
                "file": (filename, io.BytesIO(file_data), "image/jpeg")
            }
            
            with patch.object(queue_service, 'enqueue_task', new_callable=AsyncMock) as mock_enqueue:
                mock_enqueue.return_value = True
                response = client.post("/api/v1/ocr/upload", files=files)
                return response.status_code == 202
        
        # 10개 파일 동시 업로드
        tasks = []
        for i in range(10):
            file_data = f"fake_image_data_{i}".encode()
            filename = f"test_image_{i}.jpg"
            tasks.append(upload_file(file_data, filename))
        
        results = await asyncio.gather(*tasks)
        
        # 모든 업로드가 성공해야 함
        assert all(results), "일부 파일 업로드가 실패했습니다"
    
    def test_file_service_uuid_uniqueness(self):
        """파일 서비스 UUID 유일성 테스트"""
        file_ids = set()
        
        # 100개의 UUID 생성하여 중복 확인
        for _ in range(100):
            file_id = file_service._generate_file_id()
            assert file_id not in file_ids, f"중복된 UUID 발견: {file_id}"
            file_ids.add(file_id)
        
        assert len(file_ids) == 100, "생성된 UUID 수가 예상과 다릅니다"


class TestQueueService:
    """큐 서비스 테스트 클래스"""
    
    @pytest.mark.asyncio
    async def test_task_message_serialization(self):
        """작업 메시지 직렬화/역직렬화 테스트"""
        original_task = TaskMessage(
            task_id="test-123",
            task_type="ocr_processing",
            file_id="file-456",
            file_path="/path/to/file.jpg",
            metadata={"original_filename": "test.jpg", "size": 1024},
            priority=5
        )
        
        # 딕셔너리로 변환 후 다시 객체로 변환
        task_dict = original_task.to_dict()
        restored_task = TaskMessage.from_dict(task_dict)
        
        assert restored_task.task_id == original_task.task_id
        assert restored_task.task_type == original_task.task_type
        assert restored_task.file_id == original_task.file_id
        assert restored_task.file_path == original_task.file_path
        assert restored_task.metadata == original_task.metadata
        assert restored_task.priority == original_task.priority


class TestErrorHandling:
    """오류 처리 테스트 클래스"""
    
    def test_upload_queue_failure(self):
        """큐 추가 실패 시 오류 처리 테스트"""
        fake_image_data = b"fake_jpeg_data"
        files = {
            "file": ("test_image.jpg", io.BytesIO(fake_image_data), "image/jpeg")
        }
        
        with patch.object(queue_service, 'enqueue_task', new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = False  # 큐 추가 실패
            
            response = client.post("/api/v1/ocr/upload", files=files)
            assert response.status_code == 500
            assert "작업 큐 추가에 실패" in response.json()["error"]
    
    def test_large_file_rejection(self):
        """큰 파일 거부 테스트"""
        # 11MB 크기의 가짜 데이터 생성
        large_data = b"x" * (11 * 1024 * 1024)
        files = {
            "file": ("large_image.jpg", io.BytesIO(large_data), "image/jpeg")
        }
        
        response = client.post("/api/v1/ocr/upload", files=files)
        assert response.status_code == 413
        assert "파일 크기가 너무 큽니다" in response.json()["error"]
    
    def test_empty_file_upload(self):
        """빈 파일 업로드 테스트"""
        files = {
            "file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")
        }
        
        response = client.post("/api/v1/ocr/upload", files=files)
        assert response.status_code == 400
        assert "빈 파일입니다" in response.json()["error"] 