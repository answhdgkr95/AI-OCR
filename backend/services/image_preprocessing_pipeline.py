"""
통합 이미지 전처리 파이프라인
모든 전처리 단계를 순차적으로 적용하는 메인 서비스
"""

import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from .preprocessing_service import (
    PreprocessingConfig, BasePreprocessor, ProcessingResult
)
from .image_processors import (
    ResizeProcessor, GrayscaleProcessor, DenoiseProcessor,
    RotationCorrectionProcessor, BinarizationProcessor,
    MorphologyProcessor, DistortionCorrectionProcessor
)

logger = logging.getLogger(__name__)


class ImagePreprocessingPipeline:
    """이미지 전처리 파이프라인"""
    
    def __init__(self, config: PreprocessingConfig = None):
        if config is None:
            config = PreprocessingConfig()
        
        self.config = config
        self.base_processor = BasePreprocessor(config)
        
        # 전처리 단계 순서 정의
        self.processing_steps = [
            ("resize", ResizeProcessor),
            ("grayscale", GrayscaleProcessor),
            ("denoise", DenoiseProcessor),
            ("rotation_correction", RotationCorrectionProcessor),
            ("binarization", BinarizationProcessor),
            ("morphology", MorphologyProcessor),
            ("distortion_correction", DistortionCorrectionProcessor)
        ]
        
        # 프로세서 인스턴스 생성
        self.processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """프로세서 인스턴스들을 초기화"""
        for step_name, processor_class in self.processing_steps:
            if self.config.is_step_enabled(step_name):
                step_config = self.config.get_step_config(step_name)
                self.processors[step_name] = processor_class(step_config)
                logger.debug(f"{step_name} 프로세서 초기화 완료")
    
    async def process_image(self, image_data: bytes, 
                          task_id: str = "unknown") -> ProcessingResult:
        """이미지 전처리 실행"""
        start_time = time.time()
        applied_steps = []
        
        try:
            # 바이트 데이터를 OpenCV 이미지로 변환
            image = self.base_processor.convert_to_opencv(image_data)
            original_shape = image.shape[:2]
            
            # 이미지 유효성 검증
            if not self.base_processor.validate_image(image):
                raise ValueError("유효하지 않은 이미지입니다")
            
            # 전처리가 비활성화된 경우
            if not self.config.is_enabled():
                logger.info("전처리가 비활성화되어 있습니다")
                return ProcessingResult(
                    processed_image=image,
                    original_shape=original_shape,
                    processing_time=0.0,
                    applied_steps=[],
                    quality_metrics={},
                    metadata={"preprocessing_enabled": False}
                )
            
            # 초기 품질 측정
            initial_quality = self.base_processor.quality_checker.assess_quality(image)
            
            # 디버그 이미지 저장 (원본)
            self.base_processor.save_debug_image(image, "00_original", task_id)
            
            # 각 전처리 단계 실행
            current_image = image.copy()
            step_index = 1
            
            for step_name, _ in self.processing_steps:
                if step_name in self.processors:
                    try:
                        step_start = time.time()
                        
                        # 전처리 단계 실행
                        processed = self.processors[step_name].process(current_image)
                        
                        if processed is not None:
                            current_image = processed
                            applied_steps.append(step_name)
                            
                            step_time = (time.time() - step_start) * 1000
                            logger.debug(f"{step_name} 완료: {step_time:.2f}ms")
                            
                            # 디버그 이미지 저장
                            self.base_processor.save_debug_image(
                                current_image, f"{step_index:02d}_{step_name}", task_id
                            )
                            step_index += 1
                        
                    except Exception as e:
                        logger.error(f"{step_name} 처리 중 오류: {e}")
                        # 단계별 오류는 건너뛰고 계속 진행
                        continue
            
            # 최종 품질 측정
            final_quality = self.base_processor.quality_checker.assess_quality(current_image)
            
            # 품질 향상 계산
            quality_improvement = {
                "contrast_improvement": final_quality["contrast"] - initial_quality["contrast"],
                "sharpness_improvement": final_quality["sharpness"] - initial_quality["sharpness"],
                "overall_improvement": final_quality["overall_quality"] - initial_quality["overall_quality"]
            }
            
            processing_time = time.time() - start_time
            
            logger.info(f"전처리 완료: {len(applied_steps)}단계, {processing_time:.3f}초")
            
            return ProcessingResult(
                processed_image=current_image,
                original_shape=original_shape,
                processing_time=processing_time,
                applied_steps=applied_steps,
                quality_metrics={
                    "initial_quality": initial_quality,
                    "final_quality": final_quality,
                    "improvement": quality_improvement
                },
                metadata={
                    "task_id": task_id,
                    "preprocessing_enabled": True,
                    "total_steps": len(applied_steps),
                    "pipeline_version": "1.0.0"
                }
            )
            
        except Exception as e:
            logger.error(f"전처리 파이프라인 실행 실패: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        enabled_steps = []
        disabled_steps = []
        
        for step_name, _ in self.processing_steps:
            if self.config.is_step_enabled(step_name):
                enabled_steps.append({
                    "name": step_name,
                    "config": self.config.get_step_config(step_name)
                })
            else:
                disabled_steps.append(step_name)
        
        return {
            "pipeline_enabled": self.config.is_enabled(),
            "total_steps": len(self.processing_steps),
            "enabled_steps": enabled_steps,
            "disabled_steps": disabled_steps,
            "config_path": self.config.config_path
        }
    
    def reload_config(self):
        """설정 파일 다시 로드"""
        self.config = PreprocessingConfig(self.config.config_path)
        self.processors.clear()
        self._initialize_processors()
        logger.info("전처리 설정 다시 로드 완료")


class PreprocessingService:
    """전처리 서비스 파사드"""
    
    def __init__(self):
        self.pipeline = ImagePreprocessingPipeline()
    
    async def preprocess_image(self, image_data: bytes, 
                             task_id: str = "unknown") -> ProcessingResult:
        """이미지 전처리 (메인 인터페이스)"""
        return await self.pipeline.process_image(image_data, task_id)
    
    def convert_result_to_bytes(self, result: ProcessingResult) -> bytes:
        """처리 결과를 바이트로 변환"""
        import cv2
        
        # JPEG로 인코딩
        _, encoded = cv2.imencode('.jpg', result.processed_image, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded.tobytes()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 반환"""
        return self.pipeline.get_pipeline_info()
    
    def reload_configuration(self):
        """설정 다시 로드"""
        self.pipeline.reload_config()
    
    def is_preprocessing_enabled(self) -> bool:
        """전처리 활성화 여부 확인"""
        return self.pipeline.config.is_enabled()


# 전역 서비스 인스턴스
preprocessing_service = PreprocessingService() 