"""
OCR 추론 API 엔드포인트
이미지 파일을 받아 텍스트를 추출하는 RESTful API
PyTorch 모델 서버 통합
"""

import time
import uuid
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Path
from typing import Union
import asyncio

from ..schemas.ocr import (
    OCRResponse, OCRResultItem, BoundingBox, 
    UploadResponse, TaskStatusResponse
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
    # 파일 형식 검증
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. "
                   f"허용 형식: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # 파일 크기 검증 (실제 파일 크기는 업로드 후 확인)
    if file.size and file.size > MAX_FILE_SIZE:
        max_size_mb = MAX_FILE_SIZE // (1024*1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"파일 크기가 너무 큽니다. 최대 크기: {max_size_mb}MB"
        )


async def process_ocr_mock(image_data: bytes, filename: str) -> OCRResponse:
    """
    OCR 처리 모의 함수 (실제 구현에서는 PyTorch 모델 서버 호출)
    향후 실제 OCR 모델로 대체될 예정
    """
    start_time = time.time()
    
    # 모의 OCR 결과 생성
    mock_items = [
        OCRResultItem(
            text="샘플 텍스트 1",
            confidence=0.95,
            bbox=BoundingBox(x1=10, y1=10, x2=100, y2=30)
        ),
        OCRResultItem(
            text="샘플 텍스트 2", 
            confidence=0.92,
            bbox=BoundingBox(x1=10, y1=40, x2=120, y2=60)
        )
    ]
    
    processing_time = (time.time() - start_time) * 1000
    
    return OCRResponse(
        status="success",
        extracted_text="샘플 텍스트 1\n샘플 텍스트 2",
        items=mock_items,
        processing_time_ms=processing_time,
        image_info={
            "filename": filename,
            "size_bytes": len(image_data),
            "format": "detected_format"
        }
    )


@router.post(
    "/ocr/infer",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="PyTorch OCR 직접 추론",
    description="PyTorch 모델 서버를 통한 직접 OCR 추론 (전처리 포함)"
)
async def infer_ocr_direct(
    file: UploadFile = File(..., description="OCR 처리할 이미지 파일"),
    enable_preprocessing: bool = True
) -> OCRResponse:
    """
    PyTorch 모델을 이용한 직접 OCR 추론
    
    전처리 파이프라인과 PyTorch 모델 서버를 통한 고성능 OCR 처리
    
    Args:
        file: 업로드된 이미지 파일
        enable_preprocessing: 전처리 파이프라인 사용 여부
    
    Returns:
        OCRResponse: OCR 추론 결과
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # 파일 유효성 검증
        validate_image_file(file)
        
        # 파일 데이터 읽기
        image_data = await file.read()
        
        # 실제 파일 크기 검증
        if len(image_data) > MAX_FILE_SIZE:
            max_size_mb = MAX_FILE_SIZE // (1024*1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {max_size_mb}MB"
            )
        
        # 빈 파일 검증
        if len(image_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="빈 파일입니다. 유효한 이미지 파일을 업로드해주세요."
            )
        
        # 이미지 전처리 (선택적)
        preprocessed_image = None
        preprocessing_info = {}
        
        if enable_preprocessing and preprocessing_service.is_preprocessing_enabled():
            try:
                preprocessing_result = await preprocessing_service.preprocess_image(
                    image_data, request_id
                )
                preprocessed_image = preprocessing_result.processed_image
                preprocessing_info = {
                    "enabled": True,
                    "applied_steps": preprocessing_result.applied_steps,
                    "processing_time_ms": preprocessing_result.processing_time * 1000,
                    "quality_improvement": preprocessing_result.quality_metrics.get("improvement", {})
                }
            except Exception as e:
                # 전처리 실패 시 원본 이미지 사용
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_data, np.uint8)
                preprocessed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                preprocessing_info = {
                    "enabled": False,
                    "error": f"전처리 실패: {str(e)}"
                }
        else:
            # 전처리 비활성화 시 원본 이미지 사용
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_data, np.uint8)
            preprocessed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            preprocessing_info = {
                "enabled": False,
                "reason": "전처리 비활성화 또는 사용자 설정"
            }
        
        # PyTorch 모델 추론
        if not model_server.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PyTorch 모델이 로드되지 않았습니다. 서버 관리자에게 문의하세요."
            )
        
        try:
            # 비동기 추론 실행
            inference_result = await model_server.infer_async(preprocessed_image)
            
            # OCR 결과를 API 스키마 형태로 변환
            ocr_items = [
                OCRResultItem(
                    text=result.text,
                    confidence=result.confidence,
                    bbox=BoundingBox(
                        x1=result.bbox[0],
                        y1=result.bbox[1],
                        x2=result.bbox[2],
                        y2=result.bbox[3]
                    )
                )
                for result in inference_result.results
            ]
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return OCRResponse(
                status="success",
                extracted_text=inference_result.total_text,
                items=ocr_items,
                processing_time_ms=total_processing_time,
                image_info={
                    "filename": file.filename or "unknown",
                    "size_bytes": len(image_data),
                    "processed_shape": preprocessed_image.shape[:2] if preprocessed_image is not None else None,
                    "model_version": inference_result.model_version,
                    "device_used": inference_result.device_used,
                    "inference_time_ms": inference_result.processing_time * 1000,
                    "memory_usage": inference_result.memory_usage,
                    "preprocessing": preprocessing_info,
                    "request_id": request_id
                }
            )
            
        except torch.cuda.OutOfMemoryError:
            # CUDA 메모리 부족 오류
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GPU 메모리가 부족합니다. 잠시 후 다시 시도해주세요."
            )
        except Exception as e:
            # 기타 추론 오류
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"모델 추론 중 오류가 발생했습니다: {str(e)}"
            )
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/ocr/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="OCR 처리용 파일 업로드 (비동기)",
    description="이미지 파일을 업로드하고 비동기 OCR 처리를 시작합니다."
)
async def upload_for_ocr(
    file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
) -> UploadResponse:
    """
    이미지 파일을 업로드하고 비동기 OCR 처리를 시작

    지원 형식: JPEG, PNG, TIFF
    최대 파일 크기: 10MB

    Returns:
        UploadResponse: 업로드 성공 응답과 작업 ID
    """
    try:
        # 파일 유효성 검증
        validate_image_file(file)

        # 파일 저장
        file_metadata = await file_service.save_uploaded_file(file)

        # 작업 ID 생성
        task_id = str(uuid.uuid4())

        # 작업 메시지 생성
        task = TaskMessage(
            task_id=task_id,
            task_type="ocr_processing",
            file_id=file_metadata.file_id,
            file_path=file_metadata.file_path,
            metadata=file_metadata.to_dict()
        )

        # 큐에 작업 추가
        success = await queue_service.enqueue_task(task)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="작업 큐 추가에 실패했습니다."
            )

        return UploadResponse(
            task_id=task_id,
            file_id=file_metadata.file_id,
            status="uploaded",
            message="파일이 성공적으로 업로드되었습니다. OCR 처리가 시작됩니다."
        )

    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/ocr/task/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="OCR 작업 상태 조회",
    description="작업 ID로 OCR 처리 상태를 조회합니다."
)
async def get_task_status(
    task_id: str = Path(..., description="조회할 작업 ID")
) -> TaskStatusResponse:
    """OCR 작업 상태 조회"""
    try:
        # 작업 상태 조회
        status_data = await queue_service.get_task_status(task_id)

        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="작업을 찾을 수 없습니다."
            )

        response = TaskStatusResponse(
            task_id=task_id,
            status=status_data.get("status", "unknown"),
            updated_at=status_data.get("updated_at")
        )

        # 완료된 작업인 경우 결과 조회
        if status_data.get("status") == "completed":
            result_data = await queue_service.get_task_result(task_id)
            if result_data:
                response.result = result_data.get("result")

        # 실패한 작업인 경우 오류 조회
        elif status_data.get("status") == "failed":
            # 간단화를 위해 상태 데이터에서 오류 정보 조회
            response.error = "OCR 처리 중 오류가 발생했습니다."

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"작업 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/ocr/predict",
    response_model=OCRResponse,
    status_code=status.HTTP_200_OK,
    summary="OCR 텍스트 추출 (동기 처리)",
    description="업로드된 이미지에서 텍스트를 즉시 추출합니다."
)
async def predict_ocr(
    file: UploadFile = File(..., description="OCR 처리할 이미지 파일")
) -> OCRResponse:
    """
    이미지 파일에서 텍스트를 즉시 추출하는 OCR API (동기 처리)
    
    지원 형식: JPEG, PNG, TIFF
    최대 파일 크기: 10MB
    """
    try:
        # 파일 유효성 검증
        validate_image_file(file)
        
        # 파일 데이터 읽기
        image_data = await file.read()
        
        # 실제 파일 크기 검증
        if len(image_data) > MAX_FILE_SIZE:
            max_size_mb = MAX_FILE_SIZE // (1024*1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {max_size_mb}MB"
            )
        
        # 빈 파일 검증
        if len(image_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="빈 파일입니다. 유효한 이미지 파일을 업로드해주세요."
            )
        
        # OCR 처리 (현재는 모의 처리)
        result = await process_ocr_mock(image_data, file.filename or "unknown")
        
        return result
        
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/ocr/health",
    summary="OCR 서비스 상태 확인",
    description="OCR 서비스의 상태를 확인합니다."
)
async def ocr_health_check():
    """OCR 서비스 헬스 체크"""
    # 큐 상태 확인
    queue_size = await queue_service.get_queue_size()
    
    # PyTorch 모델 서버 상태 확인
    model_health = model_server.get_health_status()
    model_info = model_server.get_model_info()
    
    return {
        "status": "healthy",
        "service": "OCR API",
        "version": "1.0.0",
        "supported_formats": list(ALLOWED_IMAGE_TYPES),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "queue_size": queue_size,
        "pytorch_model": {
            "status": model_health.get("status", "unknown"),
            "loaded": model_health.get("model_loaded", False),
            "device": model_health.get("device", "unknown"),
            "model_info": model_info
        },
        "preprocessing": {
            "enabled": preprocessing_service.is_preprocessing_enabled(),
            "pipeline_info": preprocessing_service.get_pipeline_status()
        }
    } 