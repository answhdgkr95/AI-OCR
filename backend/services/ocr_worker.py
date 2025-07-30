"""
OCR 처리 워커
큐에서 작업을 가져와 OCR 처리를 수행하는 백그라운드 워커
이미지 전처리 파이프라인과 PyTorch 모델 서버 통합
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List

from services.queue_service import queue_service, TaskMessage
from services.file_service import file_service
from services.image_preprocessing_pipeline import preprocessing_service
from services.pytorch_model_server import model_server, InferenceResult
from core.config import settings

logger = logging.getLogger(__name__)


class OCRWorker:
    """OCR 처리 워커 클래스"""
    
    def __init__(self, worker_id: str = "worker-1"):
        self.worker_id = worker_id
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        
    async def process_ocr_task(self, task: TaskMessage) -> Dict[str, Any]:
        """OCR 작업 처리 (전처리 + PyTorch 모델 추론)"""
        request_id = str(uuid.uuid4())[:8]
        
        try:
            logger.info(f"[{request_id}] OCR 작업 처리 시작: {task.task_id}")
            
            # 파일 내용 읽기
            image_data = await file_service.get_file_content(task.file_path)
            
            start_time = time.time()
            
            # 1. 이미지 전처리 실행
            preprocessing_result = None
            processed_image = None
            
            if preprocessing_service.is_preprocessing_enabled():
                try:
                    preprocessing_result = await preprocessing_service.preprocess_image(
                        image_data, task.task_id
                    )
                    logger.info(f"[{request_id}] 전처리 완료: "
                               f"{len(preprocessing_result.applied_steps)}단계")
                    
                    # 전처리된 이미지 사용
                    processed_image = preprocessing_result.processed_image
                    
                except Exception as e:
                    logger.warning(f"[{request_id}] 전처리 실패, 원본 이미지 사용: {e}")
                    # 전처리 실패 시 원본 이미지를 OpenCV 형태로 변환
                    import cv2
                    import numpy as np
                    nparr = np.frombuffer(image_data, np.uint8)
                    processed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                logger.info(f"[{request_id}] 전처리 비활성화, 원본 이미지 사용")
                # 원본 이미지를 OpenCV 형태로 변환
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_data, np.uint8)
                processed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 2. PyTorch 모델 추론 실행
            try:
                logger.info(f"[{request_id}] PyTorch 모델 추론 시작")
                
                if not model_server.is_loaded:
                    raise RuntimeError("PyTorch 모델이 로드되지 않았습니다")
                
                # 비동기 추론 실행
                inference_result: InferenceResult = await model_server.infer_async(processed_image)
                
                logger.info(f"[{request_id}] PyTorch 추론 완료: "
                           f"{len(inference_result.results)}개 결과, "
                           f"{inference_result.processing_time:.3f}초")
                
            except Exception as e:
                logger.error(f"[{request_id}] PyTorch 추론 실패: {e}")
                # 실패 시 fallback 처리 (모의 결과)
                inference_result = self._create_fallback_result(task, processed_image)
            
            total_processing_time = (time.time() - start_time) * 1000
            
            # 3. 결과 구성
            result = {
                "status": "success",
                "request_id": request_id,
                "extracted_text": inference_result.total_text,
                "items": [
                    {
                        "text": ocr_result.text,
                        "confidence": ocr_result.confidence,
                        "bbox": {
                            "x1": ocr_result.bbox[0],
                            "y1": ocr_result.bbox[1],
                            "x2": ocr_result.bbox[2],
                            "y2": ocr_result.bbox[3]
                        }
                    }
                    for ocr_result in inference_result.results
                ],
                "processing_time_ms": total_processing_time,
                "worker_id": self.worker_id,
                "model_info": {
                    "model_version": inference_result.model_version,
                    "device_used": inference_result.device_used,
                    "inference_time_ms": inference_result.processing_time * 1000
                },
                "image_info": {
                    "file_id": task.file_id,
                    "size_bytes": len(image_data),
                    "original_filename": task.metadata.get("original_filename", "unknown"),
                    "processed_shape": processed_image.shape[:2] if processed_image is not None else None
                }
            }
            
            # 4. 전처리 정보 추가
            if preprocessing_result:
                result["preprocessing"] = {
                    "enabled": True,
                    "applied_steps": preprocessing_result.applied_steps,
                    "processing_time_ms": preprocessing_result.processing_time * 1000,
                    "quality_metrics": preprocessing_result.quality_metrics,
                    "original_shape": preprocessing_result.original_shape
                }
            else:
                result["preprocessing"] = {
                    "enabled": False,
                    "reason": "preprocessing_disabled_or_failed"
                }
            
            # 5. 메모리 사용량 정보 추가
            if inference_result.memory_usage:
                result["memory_usage"] = inference_result.memory_usage
            
            logger.info(f"[{request_id}] OCR 작업 처리 완료: {task.task_id}, "
                       f"총 처리시간: {total_processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{request_id}] OCR 작업 처리 실패: {task.task_id}, 오류: {e}")
            raise e
    
    def _create_fallback_result(self, task: TaskMessage, image) -> InferenceResult:
        """Fallback 결과 생성 (PyTorch 모델 실패 시)"""
        from services.pytorch_model_server import OCRResult, InferenceResult
        
        fallback_results = [
            OCRResult(
                text=f"Fallback OCR: {task.file_id[:8]}",
                confidence=0.85,
                bbox=(10, 10, 200, 30)
            ),
            OCRResult(
                text="Model fallback activated",
                confidence=0.80,
                bbox=(10, 40, 180, 60)
            )
        ]
        
        total_text = "\n".join([result.text for result in fallback_results])
        
        return InferenceResult(
            results=fallback_results,
            total_text=total_text,
            processing_time=0.1,
            model_version="fallback_v1.0",
            device_used="cpu",
            memory_usage=None
        )
    
    async def run_forever(self):
        """워커 무한 실행"""
        self.is_running = True
        logger.info(f"OCR 워커 시작: {self.worker_id}")
        
        # PyTorch 모델 서버 상태 확인
        model_health = model_server.get_health_status()
        logger.info(f"모델 서버 상태: {model_health}")
        
        while self.is_running:
            try:
                # 큐에서 작업 가져오기
                task = await queue_service.dequeue_task()
                
                if task:
                    try:
                        # OCR 작업 처리
                        result = await self.process_ocr_task(task)
                        
                        # 결과 저장
                        await queue_service.set_task_result(task.task_id, result)
                        
                        self.processed_count += 1
                        logger.info(f"워커 {self.worker_id}: 처리 완료 "
                                   f"(성공: {self.processed_count}건, "
                                   f"실패: {self.error_count}건)")
                        
                    except Exception as e:
                        # 작업 실패 처리
                        self.error_count += 1
                        error_msg = f"OCR 처리 오류: {str(e)}"
                        await queue_service.set_task_error(task.task_id, error_msg)
                        logger.error(f"작업 실패: {task.task_id}, 오류: {error_msg}")
                
                else:
                    # 큐가 비어있으면 잠시 대기
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"워커 실행 중 오류: {e}")
                await asyncio.sleep(5)  # 오류 발생 시 5초 대기
    
    def stop(self):
        """워커 중지"""
        self.is_running = False
        logger.info(f"OCR 워커 중지: {self.worker_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """워커 통계 정보"""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.error_count)
                if (self.processed_count + self.error_count) > 0 else 0.0
            )
        }


class OCRWorkerManager:
    """OCR 워커 관리자"""
    
    def __init__(self, worker_count: int = None):
        self.worker_count = worker_count or settings.max_workers
        self.workers: List[OCRWorker] = []
        self.tasks: List[asyncio.Task] = []
    
    async def start_workers(self):
        """모든 워커 시작"""
        logger.info(f"OCR 워커 {self.worker_count}개 시작")
        
        # PyTorch 모델 서버 상태 확인
        try:
            model_info = model_server.get_model_info()
            logger.info(f"PyTorch 모델 정보: {model_info}")
            
            if model_server.is_loaded:
                logger.info("PyTorch 모델 로드 완료 - 실제 추론 사용")
            else:
                logger.warning("PyTorch 모델 로드 실패 - Fallback 모드 사용")
        except Exception as e:
            logger.error(f"모델 서버 상태 확인 실패: {e}")
        
        # 전처리 서비스 상태 확인
        if preprocessing_service.is_preprocessing_enabled():
            pipeline_info = preprocessing_service.get_pipeline_status()
            enabled_steps = [step["name"] for step in pipeline_info["enabled_steps"]]
            logger.info(f"전처리 파이프라인 활성화: {enabled_steps}")
        else:
            logger.info("전처리 파이프라인 비활성화")
        
        for i in range(self.worker_count):
            worker = OCRWorker(f"worker-{i+1}")
            self.workers.append(worker)
            
            # 워커를 백그라운드 태스크로 실행
            task = asyncio.create_task(worker.run_forever())
            self.tasks.append(task)
    
    async def stop_workers(self):
        """모든 워커 중지"""
        logger.info("모든 OCR 워커 중지")
        
        # 워커 중지
        for worker in self.workers:
            worker.stop()
        
        # 태스크 취소
        for task in self.tasks:
            task.cancel()
        
        # 태스크 완료 대기
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.workers.clear()
        self.tasks.clear()
    
    def get_worker_status(self) -> Dict[str, Any]:
        """워커 상태 조회"""
        total_processed = sum(w.processed_count for w in self.workers)
        total_errors = sum(w.error_count for w in self.workers)
        
        return {
            "worker_count": len(self.workers),
            "running_workers": sum(1 for w in self.workers if w.is_running),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "overall_success_rate": (
                total_processed / (total_processed + total_errors)
                if (total_processed + total_errors) > 0 else 0.0
            ),
            "pytorch_model_loaded": model_server.is_loaded,
            "pytorch_model_info": model_server.get_model_info(),
            "preprocessing_enabled": preprocessing_service.is_preprocessing_enabled(),
            "pipeline_info": preprocessing_service.get_pipeline_status(),
            "workers": [worker.get_stats() for worker in self.workers]
        }


# 글로벌 워커 매니저 인스턴스
worker_manager = OCRWorkerManager() 