"""
이미지 전처리 서비스
OpenCV와 Albumentations를 활용한 고급 이미지 전처리
"""

import cv2
import numpy as np
import yaml
import os
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import time
from PIL import Image
import albumentations as A
from skimage import morphology, measure
from dataclasses import dataclass

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """전처리 결과 클래스"""
    processed_image: np.ndarray
    original_shape: Tuple[int, int]
    processing_time: float
    applied_steps: List[str]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class PreprocessingConfig:
    """전처리 설정 관리 클래스"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join("configs", "preprocessing.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"전처리 설정 로드 완료: {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "enabled": True,
            "pipeline": {
                "resize": {"enabled": True, "max_width": 2048, "max_height": 2048},
                "grayscale": {"enabled": True},
                "denoise": {"enabled": True, "method": "gaussian_blur", "kernel_size": 3},
                "rotation_correction": {"enabled": True, "method": "hough_lines"},
                "binarization": {"enabled": True, "method": "adaptive_threshold"},
                "morphology": {"enabled": True, "operation": "opening"}
            },
            "quality_check": {"enabled": True, "min_contrast": 0.3},
            "performance": {"use_gpu": False, "max_image_size": 10485760},
            "debug": {"save_intermediate_steps": False}
        }
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """파이프라인 설정 반환"""
        return self.config.get("pipeline", {})
    
    def is_enabled(self) -> bool:
        """전처리 활성화 여부"""
        return self.config.get("enabled", True)
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """특정 단계 설정 반환"""
        pipeline = self.get_pipeline_config()
        return pipeline.get(step_name, {})
    
    def is_step_enabled(self, step_name: str) -> bool:
        """특정 단계 활성화 여부"""
        step_config = self.get_step_config(step_name)
        return step_config.get("enabled", False)


class ImageQualityChecker:
    """이미지 품질 검사 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """이미지 대비 계산"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.std(gray) / 255.0)
    
    def calculate_blur_metric(self, image: np.ndarray) -> float:
        """이미지 블러 정도 계산 (라플라시안 분산)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """이미지 선명도 계산"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sobel 연산자를 이용한 선명도 계산
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """종합적인 이미지 품질 평가"""
        metrics = {
            "contrast": self.calculate_contrast(image),
            "blur_metric": self.calculate_blur_metric(image),
            "sharpness": self.calculate_sharpness(image)
        }
        
        # 전체 품질 점수 계산 (0-1)
        contrast_score = min(metrics["contrast"] / 0.3, 1.0)
        blur_score = min(metrics["blur_metric"] / 100.0, 1.0)
        sharpness_score = min(metrics["sharpness"] / 50.0, 1.0)
        
        metrics["overall_quality"] = (contrast_score + blur_score + sharpness_score) / 3.0
        
        return metrics


class BasePreprocessor:
    """기본 전처리 클래스"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.quality_checker = ImageQualityChecker(config.config.get("quality_check", {}))
    
    def validate_image(self, image: np.ndarray) -> bool:
        """이미지 유효성 검증"""
        if image is None or image.size == 0:
            return False
        
        # 이미지 크기 검증
        max_size = self.config.config.get("performance", {}).get("max_image_size", 10485760)
        if image.nbytes > max_size:
            logger.warning(f"이미지 크기가 너무 큽니다: {image.nbytes} > {max_size}")
            return False
        
        return True
    
    def convert_to_opencv(self, image_data: bytes) -> np.ndarray:
        """바이트 데이터를 OpenCV 이미지로 변환"""
        try:
            # NumPy 배열로 변환
            nparr = np.frombuffer(image_data, np.uint8)
            
            # OpenCV로 디코드
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지 디코딩에 실패했습니다")
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 변환 실패: {e}")
            raise
    
    def save_debug_image(self, image: np.ndarray, step_name: str, 
                        task_id: str = "unknown") -> None:
        """디버그용 이미지 저장"""
        debug_config = self.config.config.get("debug", {})
        
        if not debug_config.get("save_intermediate_steps", False):
            return
        
        try:
            output_dir = debug_config.get("output_dir", "debug_preprocessing")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{task_id}_{step_name}_{int(time.time())}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, image)
            logger.debug(f"디버그 이미지 저장: {filepath}")
            
        except Exception as e:
            logger.warning(f"디버그 이미지 저장 실패: {e}")


# 전역 설정 및 인스턴스
preprocessing_config = PreprocessingConfig()
base_preprocessor = BasePreprocessor(preprocessing_config) 