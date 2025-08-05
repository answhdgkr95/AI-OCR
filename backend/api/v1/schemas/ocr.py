"""
OCR API 요청/응답 스키마 정의
T-005: OCR 결과 JSON 응답 포맷 정의 및 처리
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BoundingBox(BaseModel):
    """텍스트 경계 상자 좌표"""
    x1: int = Field(..., description="좌상단 X 좌표", example=10)
    y1: int = Field(..., description="좌상단 Y 좌표", example=10) 
    x2: int = Field(..., description="우하단 X 좌표", example=250)
    y2: int = Field(..., description="우하단 Y 좌표", example=30)
    
    class Config:
        schema_extra = {
            "example": {
                "x1": 10,
                "y1": 10,
                "x2": 250,
                "y2": 30
            }
        }


class TextBlock(BaseModel):
    """T-005 요구사항: 텍스트 블록 (Pydantic 모델)"""
    text: str = Field(..., description="추출된 텍스트", example="금융권 디지털 혁신")
    bbox: List[int] = Field(..., description="경계 상자 [x1, y1, x2, y2]", example=[10, 10, 250, 30])
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0-1.0)", example=0.95)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "금융권 디지털 혁신",
                "bbox": [10, 10, 250, 30],
                "confidence": 0.95
            }
        }


class OCRResultItem(BaseModel):
    """개별 OCR 결과 아이템 (호환성 유지)"""
    text: str = Field(..., description="추출된 텍스트", example="AI OCR 서버")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0-1.0)", example=0.98)
    bbox: BoundingBox = Field(..., description="텍스트 경계 상자")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "AI OCR 서버",
                "confidence": 0.98,
                "bbox": {
                    "x1": 50,
                    "y1": 100,
                    "x2": 300,
                    "y2": 130
                }
            }
        }


class PreprocessingInfo(BaseModel):
    """전처리 정보"""
    enabled: bool = Field(..., description="전처리 활성화 여부")
    applied_steps: List[str] = Field(default=[], description="적용된 전처리 단계")
    processing_time_ms: float = Field(default=0.0, description="전처리 시간(밀리초)")
    quality_metrics: Dict[str, float] = Field(default={}, description="품질 지표")


class ModelInfo(BaseModel):
    """모델 정보"""
    name: str = Field(..., description="모델 이름", example="ocr_model_v1")
    version: str = Field(..., description="모델 버전", example="1.0.0")
    device: str = Field(..., description="사용된 디바이스", example="cpu")


class OcrResponse(BaseModel):
    """T-005 요구사항: 표준화된 OCR 응답 모델"""
    status: str = Field(default="success", description="처리 상태", example="success")
    message: str = Field(default="OCR 처리 완료", description="응답 메시지", example="OCR 처리 완료")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="처리 시간")
    
    # 핵심 결과 데이터
    results: List[TextBlock] = Field(default=[], description="텍스트 블록 목록")
    total_text: str = Field(default="", description="전체 추출된 텍스트")
    
    # 메타데이터
    processing_time_ms: float = Field(..., description="총 처리 시간(밀리초)", example=150.5)
    image_info: Dict[str, Any] = Field(default={}, description="이미지 정보")
    preprocessing: Optional[PreprocessingInfo] = Field(None, description="전처리 정보")
    model_info: Optional[ModelInfo] = Field(None, description="모델 정보")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "OCR 처리 완료",
                "timestamp": "2024-01-15T09:30:45.123456",
                "results": [
                    {
                        "text": "금융권 디지털 혁신을 위한",
                        "bbox": [10, 10, 300, 35],
                        "confidence": 0.98
                    },
                    {
                        "text": "AI OCR 서버 구축",
                        "bbox": [10, 45, 250, 70],
                        "confidence": 0.96
                    }
                ],
                "total_text": "금융권 디지털 혁신을 위한\nAI OCR 서버 구축",
                "processing_time_ms": 150.5,
                "image_info": {
                    "width": 1024,
                    "height": 768,
                    "format": "JPEG",
                    "size_bytes": 245760
                },
                "preprocessing": {
                    "enabled": True,
                    "applied_steps": ["resize", "grayscale", "denoise", "binarization"],
                    "processing_time_ms": 45.2,
                    "quality_metrics": {
                        "contrast": 0.85,
                        "sharpness": 0.92
                    }
                },
                "model_info": {
                    "name": "ocr_model_v1",
                    "version": "1.0.0",
                    "device": "cpu"
                }
            }
        }


class OCRResponse(BaseModel):
    """기존 OCR 처리 결과 응답 (호환성 유지)"""
    status: str = Field("success", description="처리 상태")
    extracted_text: str = Field(..., description="전체 추출된 텍스트")
    items: List[OCRResultItem] = Field(..., description="개별 텍스트 항목 목록")
    processing_time_ms: float = Field(..., description="처리 시간(밀리초)")
    image_info: dict = Field(..., description="이미지 정보")


class UploadResponse(BaseModel):
    """파일 업로드 응답"""
    task_id: str = Field(..., description="작업 ID", example="task_123e4567-e89b-12d3-a456-426614174000")
    file_id: str = Field(..., description="파일 ID", example="file_987f6543-210a-bcde-f012-345678901234")
    status: str = Field("uploaded", description="업로드 상태", example="uploaded")
    message: str = Field(..., description="응답 메시지", example="파일이 성공적으로 업로드되었습니다")
    file_info: Dict[str, Any] = Field(default={}, description="파일 정보")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
                "file_id": "file_987f6543-210a-bcde-f012-345678901234",
                "status": "uploaded",
                "message": "파일이 성공적으로 업로드되었습니다",
                "file_info": {
                    "filename": "document.jpg",
                    "size_bytes": 1048576,
                    "content_type": "image/jpeg"
                }
            }
        }


class TaskStatusResponse(BaseModel):
    """작업 상태 응답"""
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태", example="completed")
    progress: int = Field(default=0, ge=0, le=100, description="진행률 (%)")
    updated_at: Optional[str] = Field(None, description="상태 업데이트 시간")
    result: Optional[OcrResponse] = Field(None, description="OCR 결과 (완료 시)")
    error: Optional[str] = Field(None, description="오류 메시지 (실패 시)")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "progress": 100,
                "updated_at": "2024-01-15T09:30:50.123456",
                "result": {
                    "status": "success",
                    "message": "OCR 처리 완료",
                    "results": [],
                    "total_text": "",
                    "processing_time_ms": 150.5
                }
            }
        }


class ErrorResponse(BaseModel):
    """표준화된 오류 응답"""
    status: str = Field(default="error", description="응답 상태")
    error: str = Field(..., description="오류 메시지")
    error_code: str = Field(..., description="오류 코드", example="INVALID_FILE_FORMAT")
    detail: Optional[str] = Field(None, description="상세 오류 정보")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="오류 발생 시간")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": "지원하지 않는 파일 형식입니다",
                "error_code": "INVALID_FILE_FORMAT",
                "detail": "허용된 형식: jpg, jpeg, png, tiff, pdf",
                "timestamp": "2024-01-15T09:30:45.123456"
            }
        } 