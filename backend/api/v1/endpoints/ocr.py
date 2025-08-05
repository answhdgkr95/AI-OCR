"""
OCR 추론 API 엔드포인트
이미지 파일을 받아 텍스트를 추출하는 RESTful API
PyTorch 모델 서버 통합 및 T-005 표준화된 응답 적용
"""

import time
import uuid
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Path
from typing import List

from ..schemas.ocr import (
    OcrResponse, OCRResponse, OCRResultItem, BoundingBox, 
    UploadResponse, TaskStatusResponse, ErrorResponse,
    TextBlock, PreprocessingInfo, ModelInfo
)
from services.file_service import file_service
from services.queue_service import queue_service, TaskMessage
from services.image_preprocessing_pipeline import preprocessing_service
from services.pytorch_model_server import model_server


router = APIRouter()

# 허용되는 이미지 파일 형식
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff"}

# 최대 파일 크기 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_image_file(file: UploadFile) -> None:
    """이미지 파일 유효성 검증"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"파일 크기가 너무 큽니다. 최대 크기: {MAX_FILE_SIZE // (1024*1024)}MB"
        )


def convert_inference_result_to_ocr_response(
    inference_result, 
    processing_time: float,
    image_info: dict,
    preprocessing_result=None
) -> OcrResponse:
    """T-005: PyTorch 추론 결과를 표준화된 OcrResponse로 변환"""
    
    # TextBlock 목록 생성
    text_blocks = []
    if hasattr(inference_result, 'results') and inference_result.results:
        for item in inference_result.results:
            text_block = TextBlock(
                text=item.text,
                bbox=[item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]],
                confidence=item.confidence
            )
            text_blocks.append(text_block)
    
    # 전체 텍스트 구성
    total_text = inference_result.total_text if hasattr(inference_result, 'total_text') else ""
    if not total_text and text_blocks:
        total_text = "\n".join([block.text for block in text_blocks])
    
    # 전처리 정보 구성
    preprocessing_info = None
    if preprocessing_result:
        preprocessing_info = PreprocessingInfo(
            enabled=True,
            applied_steps=preprocessing_result.applied_steps,
            processing_time_ms=preprocessing_result.processing_time * 1000,
            quality_metrics=preprocessing_result.quality_metrics.get('improvement', {})
        )
    
    # 모델 정보 구성
    model_info = ModelInfo(
        name=getattr(inference_result, 'model_version', 'ocr_model_v1'),
        version="1.0.0",
        device=getattr(inference_result, 'device_used', 'cpu')
    )
    
    return OcrResponse(
        status="success",
        message="OCR 처리 완료",
        results=text_blocks,
        total_text=total_text,
        processing_time_ms=processing_time,
        image_info=image_info,
        preprocessing=preprocessing_info,
        model_info=model_info
    )


@router.post(
    "/infer",
    response_model=OcrResponse,
    summary="직접 OCR 추론",
    description="이미지 파일을 업로드하여 즉시 OCR 처리를 수행합니다. 전처리 파이프라인과 PyTorch 모델을 활용한 고정밀 텍스트 추출을 제공합니다.",
    responses={
        200: {
            "description": "OCR 처리 성공",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "OCR 처리 완료",
                        "timestamp": "2024-01-15T09:30:45.123456",
                        "results": [
                            {
                                "text": "금융권 디지털 혁신",
                                "bbox": [10, 10, 300, 35],
                                "confidence": 0.98
                            }
                        ],
                        "total_text": "금융권 디지털 혁신",
                        "processing_time_ms": 150.5,
                        "image_info": {"width": 1024, "height": 768},
                        "model_info": {"name": "ocr_model_v1", "device": "cpu"}
                    }
                }
            }
        },
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        413: {"model": ErrorResponse, "description": "파일 크기 초과"},
        500: {"model": ErrorResponse, "description": "서버 오류"}
    }
)
async def infer_ocr_direct(
    file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
):
    """T-005: 표준화된 응답 포맷을 사용한 직접 OCR 추론"""
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
        
        # 이미지 정보 구성
        image_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_data)
        }
        
        # 전처리 실행
        preprocessing_result = None
        processed_image_data = image_data
        
        if preprocessing_service.is_preprocessing_enabled():
            try:
                task_id = f"direct_{uuid.uuid4().hex[:8]}"
                preprocessing_result = await preprocessing_service.preprocess_image(
                    image_data, task_id
                )
                processed_image_data = preprocessing_service.convert_result_to_bytes(
                    preprocessing_result
                )
            except Exception as e:
                # 전처리 실패 시 원본 이미지 사용
                processed_image_data = image_data
        
        # PyTorch 모델 추론
        try:
            import numpy as np
            import cv2
            
            # 바이트 데이터를 OpenCV 이미지로 변환
            nparr = np.frombuffer(processed_image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                raise ValueError("이미지 디코딩 실패")
            
            # 이미지 정보 업데이트
            image_info.update({
                "width": cv_image.shape[1],
                "height": cv_image.shape[0],
                "channels": cv_image.shape[2] if len(cv_image.shape) == 3 else 1
            })
            
            # PyTorch 모델 추론 실행
            inference_result = await model_server.infer_async(cv_image)
            
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU 메모리 부족으로 처리할 수 없습니다. 잠시 후 다시 시도해주세요."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR 처리 중 오류가 발생했습니다: {str(e)}"
            )
        
        # 총 처리 시간 계산
        total_processing_time = (time.time() - start_time) * 1000
        
        # 표준화된 응답 생성
        response = convert_inference_result_to_ocr_response(
            inference_result,
            total_processing_time,
            image_info,
            preprocessing_result
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예기치 않은 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="이미지 업로드 및 비동기 OCR 처리",
    description="이미지 파일을 업로드하고 Redis 큐를 통한 비동기 OCR 처리를 시작합니다."
)
async def upload_for_ocr(
    file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
):
    """이미지 업로드 및 비동기 OCR 처리 시작"""
    try:
        # 파일 유효성 검증
        validate_image_file(file)
        
        # 파일 저장
        file_metadata = await file_service.save_uploaded_file(file)
        
        # 큐에 작업 추가
        task = TaskMessage(
            task_id=f"task_{uuid.uuid4()}",
            file_id=file_metadata.file_id,
            file_path=file_metadata.file_path,
            priority=1
        )
        
        await queue_service.enqueue_task(task)
        
        return UploadResponse(
            task_id=task.task_id,
            file_id=file_metadata.file_id,
            status="uploaded",
            message="파일이 성공적으로 업로드되었습니다. OCR 처리가 시작됩니다.",
            file_info={
                "filename": file_metadata.filename,
                "size_bytes": file_metadata.size,
                "content_type": file_metadata.content_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 업로드 실패: {str(e)}"
        )


@router.get(
    "/task/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="OCR 작업 상태 조회",
    description="업로드된 작업의 처리 상태와 결과를 조회합니다."
)
async def get_task_status(
    task_id: str = Path(..., description="작업 ID")
):
    """OCR 작업 상태 조회"""
    try:
        # 작업 상태 조회
        status_info = await queue_service.get_task_status(task_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="작업을 찾을 수 없습니다"
            )
        
        # 결과가 있는 경우 OcrResponse 형식으로 변환
        result = None
        if status_info.get("status") == "completed":
            raw_result = await queue_service.get_task_result(task_id)
            if raw_result:
                # 기존 결과를 새로운 형식으로 변환
                result = OcrResponse(
                    status="success",
                    message="OCR 처리 완료",
                    results=[],  # 실제 구현에서는 변환 로직 적용
                    total_text=raw_result.get("extracted_text", ""),
                    processing_time_ms=raw_result.get("processing_time", 0.0)
                )
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status_info.get("status", "unknown"),
            progress=100 if status_info.get("status") == "completed" else 50,
            updated_at=status_info.get("updated_at"),
            result=result,
            error=await queue_service.get_task_error(task_id) if status_info.get("status") == "failed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"작업 상태 조회 실패: {str(e)}"
        )


@router.get(
    "/health",
    summary="OCR 서비스 헬스 체크",
    description="OCR 모델 서버와 전처리 파이프라인의 상태를 확인합니다."
)
async def ocr_health_check():
    """OCR 서비스 헬스 체크"""
    try:
        # 모델 서버 상태 확인
        model_health = model_server.get_health_status()
        
        # 전처리 서비스 상태 확인
        preprocessing_status = preprocessing_service.get_pipeline_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "model_server": model_health,
            "preprocessing": {
                "enabled": preprocessing_service.is_preprocessing_enabled(),
                "pipeline_info": preprocessing_status
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"서비스 상태 확인 실패: {str(e)}"
        ) 