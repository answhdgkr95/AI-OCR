"""
T-005: OCR 결과 JSON 응답 포맷 테스트
표준화된 응답 스키마 검증 및 데이터 변환 로직 테스트
"""

import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from api.v1.schemas.ocr import (
    TextBlock, OcrResponse, BoundingBox, OCRResultItem,
    PreprocessingInfo, ModelInfo, ErrorResponse
)
from api.v1.endpoints.ocr import convert_inference_result_to_ocr_response


class TestTextBlockModel:
    """TextBlock Pydantic 모델 테스트"""
    
    def test_text_block_creation(self):
        """TextBlock 생성 테스트"""
        text_block = TextBlock(
            text="금융권 디지털 혁신",
            bbox=[10, 10, 300, 35],
            confidence=0.98
        )
        
        assert text_block.text == "금융권 디지털 혁신"
        assert text_block.bbox == [10, 10, 300, 35]
        assert text_block.confidence == 0.98
        
    def test_text_block_validation(self):
        """TextBlock 유효성 검증 테스트"""
        # 신뢰도 범위 검증
        with pytest.raises(ValueError):
            TextBlock(
                text="test",
                bbox=[0, 0, 10, 10],
                confidence=1.5  # 1.0 초과
            )
            
        with pytest.raises(ValueError):
            TextBlock(
                text="test", 
                bbox=[0, 0, 10, 10],
                confidence=-0.1  # 0.0 미만
            )
    
    def test_text_block_json_serialization(self):
        """TextBlock JSON 직렬화 테스트"""
        text_block = TextBlock(
            text="AI OCR 서버",
            bbox=[50, 100, 300, 130],
            confidence=0.96
        )
        
        json_data = text_block.dict()
        
        assert json_data["text"] == "AI OCR 서버"
        assert json_data["bbox"] == [50, 100, 300, 130]
        assert json_data["confidence"] == 0.96


class TestOcrResponseModel:
    """OcrResponse Pydantic 모델 테스트"""
    
    def test_ocr_response_creation(self):
        """OcrResponse 생성 테스트"""
        text_blocks = [
            TextBlock(text="첫 번째 텍스트", bbox=[10, 10, 200, 30], confidence=0.98),
            TextBlock(text="두 번째 텍스트", bbox=[10, 40, 250, 60], confidence=0.95)
        ]
        
        preprocessing_info = PreprocessingInfo(
            enabled=True,
            applied_steps=["resize", "grayscale", "binarization"],
            processing_time_ms=45.2,
            quality_metrics={"contrast": 0.85, "sharpness": 0.92}
        )
        
        model_info = ModelInfo(
            name="ocr_model_v1",
            version="1.0.0",
            device="cpu"
        )
        
        response = OcrResponse(
            status="success",
            message="OCR 처리 완료",
            results=text_blocks,
            total_text="첫 번째 텍스트\n두 번째 텍스트",
            processing_time_ms=150.5,
            image_info={"width": 1024, "height": 768},
            preprocessing=preprocessing_info,
            model_info=model_info
        )
        
        assert response.status == "success"
        assert response.message == "OCR 처리 완료"
        assert len(response.results) == 2
        assert response.total_text == "첫 번째 텍스트\n두 번째 텍스트"
        assert response.processing_time_ms == 150.5
        assert response.preprocessing.enabled is True
        assert response.model_info.name == "ocr_model_v1"
    
    def test_ocr_response_empty_results(self):
        """빈 결과에 대한 OcrResponse 테스트"""
        response = OcrResponse(
            status="success",
            message="OCR 처리 완료 (텍스트 없음)",
            results=[],
            total_text="",
            processing_time_ms=120.0,
            image_info={"width": 500, "height": 300}
        )
        
        assert response.results == []
        assert response.total_text == ""
        assert response.status == "success"
    
    def test_ocr_response_json_schema(self):
        """OcrResponse JSON 스키마 검증"""
        response = OcrResponse(
            status="success",
            results=[TextBlock(text="테스트", bbox=[0, 0, 100, 20], confidence=0.9)],
            total_text="테스트",
            processing_time_ms=100.0
        )
        
        json_data = response.dict()
        
        # 필수 필드 존재 확인
        required_fields = ["status", "message", "timestamp", "results", 
                          "total_text", "processing_time_ms"]
        
        for field in required_fields:
            assert field in json_data
        
        # 타임스탬프 형식 확인
        timestamp_str = json_data["timestamp"]
        datetime.fromisoformat(timestamp_str)  # 유효한 ISO 형식인지 확인


class TestDataConversionLogic:
    """데이터 변환 로직 테스트"""
    
    def test_convert_inference_result_to_ocr_response(self):
        """추론 결과를 OcrResponse로 변환 테스트"""
        # 모의 추론 결과 생성
        mock_result_item = MagicMock()
        mock_result_item.text = "변환 테스트 텍스트"
        mock_result_item.bbox = (10, 20, 200, 40)
        mock_result_item.confidence = 0.95
        
        mock_inference_result = MagicMock()
        mock_inference_result.results = [mock_result_item]
        mock_inference_result.total_text = "변환 테스트 텍스트"
        mock_inference_result.model_version = "test_model_v1"
        mock_inference_result.device_used = "cpu"
        
        # 모의 전처리 결과
        mock_preprocessing_result = MagicMock()
        mock_preprocessing_result.applied_steps = ["resize", "grayscale"]
        mock_preprocessing_result.processing_time = 0.045
        mock_preprocessing_result.quality_metrics = {
            'improvement': {'contrast': 0.1, 'sharpness': 0.05}
        }
        
        image_info = {"width": 800, "height": 600, "format": "JPEG"}
        processing_time = 180.5
        
        # 변환 실행
        response = convert_inference_result_to_ocr_response(
            mock_inference_result,
            processing_time,
            image_info,
            mock_preprocessing_result
        )
        
        # 검증
        assert response.status == "success"
        assert response.message == "OCR 처리 완료"
        assert len(response.results) == 1
        assert response.results[0].text == "변환 테스트 텍스트"
        assert response.results[0].bbox == [10, 20, 200, 40]
        assert response.results[0].confidence == 0.95
        assert response.total_text == "변환 테스트 텍스트"
        assert response.processing_time_ms == 180.5
        assert response.image_info == image_info
        assert response.preprocessing.enabled is True
        assert response.preprocessing.applied_steps == ["resize", "grayscale"]
        assert response.model_info.name == "test_model_v1"
        assert response.model_info.device == "cpu"
    
    def test_convert_empty_inference_result(self):
        """빈 추론 결과 변환 테스트"""
        mock_inference_result = MagicMock()
        mock_inference_result.results = []
        mock_inference_result.total_text = ""
        
        image_info = {"width": 400, "height": 300}
        processing_time = 95.0
        
        response = convert_inference_result_to_ocr_response(
            mock_inference_result,
            processing_time,
            image_info
        )
        
        assert response.status == "success"
        assert response.results == []
        assert response.total_text == ""
        assert response.processing_time_ms == 95.0
    
    def test_convert_without_preprocessing_result(self):
        """전처리 없이 변환 테스트"""
        mock_result_item = MagicMock()
        mock_result_item.text = "전처리 없음"
        mock_result_item.bbox = (0, 0, 100, 20)
        mock_result_item.confidence = 0.9
        
        mock_inference_result = MagicMock()
        mock_inference_result.results = [mock_result_item]
        mock_inference_result.total_text = "전처리 없음"
        
        response = convert_inference_result_to_ocr_response(
            mock_inference_result,
            120.0,
            {"width": 300, "height": 200}
        )
        
        assert response.preprocessing is None
        assert response.model_info.name == "ocr_model_v1"  # 기본값


class TestErrorResponseModel:
    """ErrorResponse 모델 테스트"""
    
    def test_error_response_creation(self):
        """ErrorResponse 생성 테스트"""
        error_response = ErrorResponse(
            error="지원하지 않는 파일 형식입니다",
            error_code="INVALID_FILE_FORMAT",
            detail="허용된 형식: jpg, jpeg, png, tiff, pdf"
        )
        
        assert error_response.status == "error"
        assert error_response.error == "지원하지 않는 파일 형식입니다"
        assert error_response.error_code == "INVALID_FILE_FORMAT"
        assert error_response.detail == "허용된 형식: jpg, jpeg, png, tiff, pdf"
        assert error_response.timestamp is not None
    
    def test_error_response_json_schema(self):
        """ErrorResponse JSON 스키마 검증"""
        error_response = ErrorResponse(
            error="서버 오류",
            error_code="INTERNAL_SERVER_ERROR"
        )
        
        json_data = error_response.dict()
        
        required_fields = ["status", "error", "error_code", "timestamp"]
        for field in required_fields:
            assert field in json_data
        
        assert json_data["status"] == "error"


class TestBackwardCompatibility:
    """기존 스키마와의 호환성 테스트"""
    
    def test_old_ocr_response_still_works(self):
        """기존 OCRResponse 모델이 여전히 작동하는지 테스트"""
        # 기존 응답 모델 생성
        bbox = BoundingBox(x1=10, y1=10, x2=100, y2=30)
        item = OCRResultItem(
            text="호환성 테스트",
            confidence=0.9,
            bbox=bbox
        )
        
        # 문제없이 생성되어야 함
        assert item.text == "호환성 테스트"
        assert item.confidence == 0.9
        assert item.bbox.x1 == 10


class TestJSONSchemaCompliance:
    """JSON 스키마 준수 테스트"""
    
    def test_response_json_serializable(self):
        """응답이 JSON으로 직렬화 가능한지 테스트"""
        response = OcrResponse(
            results=[
                TextBlock(text="JSON 테스트", bbox=[0, 0, 100, 20], confidence=0.95)
            ],
            total_text="JSON 테스트",
            processing_time_ms=150.0,
            image_info={"test": "value"}
        )
        
        # JSON 직렬화가 성공해야 함
        json_str = json.dumps(response.dict(), ensure_ascii=False)
        
        # 다시 파싱이 가능해야 함
        parsed_data = json.loads(json_str)
        
        assert parsed_data["status"] == "success"
        assert len(parsed_data["results"]) == 1
        assert parsed_data["results"][0]["text"] == "JSON 테스트"
    
    def test_coordinate_integer_conversion(self):
        """좌표값이 정수로 변환되는지 테스트"""
        text_block = TextBlock(
            text="좌표 테스트",
            bbox=[10.7, 20.3, 100.9, 40.1],  # 실수 입력
            confidence=0.9
        )
        
        # bbox는 정수 리스트여야 함
        assert all(isinstance(coord, int) for coord in text_block.bbox)
        assert text_block.bbox == [10, 20, 100, 40]


class TestOpenAPIDocumentation:
    """OpenAPI 문서화 테스트"""
    
    def test_schema_has_examples(self):
        """스키마에 예시가 포함되어 있는지 테스트"""
        # TextBlock 스키마 예시 확인
        text_block_schema = TextBlock.schema()
        assert "example" in text_block_schema.get("properties", {}).get("text", {})
        assert "example" in text_block_schema.get("properties", {}).get("bbox", {})
        
        # OcrResponse 스키마 예시 확인  
        ocr_response_schema = OcrResponse.schema()
        config_example = getattr(OcrResponse.Config, 'schema_extra', {})
        assert 'example' in config_example
    
    def test_field_descriptions(self):
        """필드 설명이 포함되어 있는지 테스트"""
        text_block_schema = TextBlock.schema()
        properties = text_block_schema.get("properties", {})
        
        assert "description" in properties.get("text", {})
        assert "description" in properties.get("bbox", {})
        assert "description" in properties.get("confidence", {}) 