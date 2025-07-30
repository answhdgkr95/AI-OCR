"""
OCR API 요청/응답 스키마 정의
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """텍스트 경계 상자 좌표"""
    x1: int = Field(..., description="좌상단 X 좌표")
    y1: int = Field(..., description="좌상단 Y 좌표") 
    x2: int = Field(..., description="우하단 X 좌표")
    y2: int = Field(..., description="우하단 Y 좌표")


class OCRResultItem(BaseModel):
    """개별 OCR 결과 아이템"""
    text: str = Field(..., description="추출된 텍스트")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0-1.0)")
    bbox: BoundingBox = Field(..., description="텍스트 경계 상자")


class OCRResponse(BaseModel):
    """OCR 처리 결과 응답"""
    status: str = Field("success", description="처리 상태")
    extracted_text: str = Field(..., description="전체 추출된 텍스트")
    items: List[OCRResultItem] = Field(..., description="개별 텍스트 항목 목록")
    processing_time_ms: float = Field(..., description="처리 시간(밀리초)")
    image_info: dict = Field(..., description="이미지 정보")


class UploadResponse(BaseModel):
    """파일 업로드 응답"""
    task_id: str = Field(..., description="작업 ID")
    file_id: str = Field(..., description="파일 ID")
    status: str = Field("uploaded", description="업로드 상태")
    message: str = Field(..., description="응답 메시지")


class TaskStatusResponse(BaseModel):
    """작업 상태 응답"""
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태")
    updated_at: Optional[str] = Field(None, description="상태 업데이트 시간")
    result: Optional[dict] = Field(None, description="작업 결과 (완료 시)")
    error: Optional[str] = Field(None, description="오류 메시지 (실패 시)")


class ErrorResponse(BaseModel):
    """오류 응답"""
    error: str = Field(..., description="오류 메시지")
    detail: Optional[str] = Field(None, description="상세 오류 정보")
    status_code: int = Field(..., description="HTTP 상태 코드") 