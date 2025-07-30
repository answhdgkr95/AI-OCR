"""
PyTorch OCR 모델 서버
모델 로딩, 추론, GPU 관리를 담당하는 핵심 모듈
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
import logging
import os
import gc
import psutil
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR 추론 결과 클래스"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class InferenceResult:
    """추론 결과 종합 클래스"""
    results: List[OCRResult]
    total_text: str
    processing_time: float
    model_version: str
    device_used: str
    memory_usage: Optional[Dict[str, float]] = None


class ModelConfig:
    """모델 설정 관리 클래스"""
    
    def __init__(self, config_path: str = "configs/model.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"모델 설정 로드 완료: {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"모델 설정 파일을 찾을 수 없습니다: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"모델 설정 로드 실패: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "model": {
                "name": "default_ocr_model",
                "model_path": "models/ocr_model.pt",
                "input": {"image_size": [640, 640], "channels": 3}
            },
            "device": {"auto_detect": True, "preferred": "cuda"},
            "inference": {"timeout_seconds": 30, "warmup_iterations": 5},
            "fallback": {"enabled": True, "method": "mock"}
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.config.get("model", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return self.config.get("device", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """추론 설정 반환"""
        return self.config.get("inference", {})


class DeviceManager:
    """GPU/CPU 디바이스 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.device_info = self._get_device_info()
    
    def _setup_device(self) -> torch.device:
        """최적 디바이스 설정"""
        auto_detect = self.config.get("auto_detect", True)
        preferred = self.config.get("preferred", "cuda")
        gpu_id = self.config.get("gpu_id", 0)
        
        if not auto_detect:
            device = torch.device(preferred)
            logger.info(f"수동 디바이스 설정: {device}")
            return device
        
        # 자동 감지
        if preferred == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"CUDA 디바이스 사용: {device}")
            return device
        elif preferred == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon MPS 디바이스 사용")
            return device
        else:
            device = torch.device("cpu")
            logger.info("CPU 디바이스 사용")
            return device
    
    def _get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 수집"""
        info = {
            "device_type": self.device.type,
            "device_index": getattr(self.device, "index", None)
        }
        
        if self.device.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(self.device),
                "gpu_memory_total": torch.cuda.get_device_properties(self.device).total_memory,
                "cuda_version": torch.version.cuda
            })
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        usage = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            usage.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved(self.device) / 1024**3,
                "gpu_memory_total": torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            })
        
        return usage
    
    def clear_memory(self):
        """메모리 정리"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ImagePreprocessor:
    """이미지 전처리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = tuple(config.get("image_size", [640, 640]))
        self.normalize = config.get("normalize", True)
        self.mean = config.get("mean", [0.485, 0.456, 0.406])
        self.std = config.get("std", [0.229, 0.224, 0.225])
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """전처리 파이프라인 구성"""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리 실행"""
        try:
            # OpenCV BGR을 RGB로 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 전처리 적용
            tensor = self.transform(image)
            
            # 배치 차원 추가
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {e}")
            raise


class MockOCRModel(nn.Module):
    """개발용 모의 OCR 모델"""
    
    def __init__(self):
        super().__init__()
        self.dummy_layer = nn.Linear(1, 1)
    
    def forward(self, x):
        # 모의 출력 생성
        batch_size = x.shape[0]
        
        # 가짜 텍스트 감지 결과
        mock_results = {
            "texts": ["Sample OCR Text", "Another Line", "Document Content"],
            "boxes": torch.tensor([[10, 10, 200, 30], [10, 40, 180, 60], [10, 70, 250, 90]]),
            "scores": torch.tensor([0.95, 0.92, 0.88])
        }
        
        return mock_results


class PyTorchModelServer:
    """PyTorch OCR 모델 서버"""
    
    def __init__(self, config_path: str = "configs/model.yaml"):
        self.config = ModelConfig(config_path)
        self.device_manager = DeviceManager(self.config.get_device_config())
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 모델 초기화
        self._initialize_model()
    
    def _initialize_model(self):
        """모델 초기화"""
        try:
            model_config = self.config.get_model_config()
            model_path = model_config.get("model_path", "models/ocr_model.pt")
            
            # 전처리기 설정
            input_config = model_config.get("input", {})
            self.preprocessor = ImagePreprocessor(input_config)
            
            # 모델 로드
            self._load_model(model_path)
            
            # 워밍업 실행
            self._warmup_model()
            
            self.is_loaded = True
            logger.info("PyTorch 모델 서버 초기화 완료")
            
        except Exception as e:
            logger.error(f"모델 서버 초기화 실패: {e}")
            self.is_loaded = False
            raise
    
    def _load_model(self, model_path: str):
        """모델 파일 로드"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"모델 파일을 찾을 수 없음: {model_path}")
                logger.info("개발용 모의 모델을 사용합니다")
                self.model = MockOCRModel()
            else:
                # 실제 모델 로드
                logger.info(f"모델 로드 중: {model_path}")
                self.model = torch.load(model_path, map_location="cpu")
            
            # 모델을 지정된 디바이스로 이동
            self.model = self.model.to(self.device_manager.device)
            self.model.eval()
            
            logger.info(f"모델 로드 완료: {self.device_manager.device}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def _warmup_model(self):
        """모델 워밍업"""
        try:
            warmup_iterations = self.config.get_inference_config().get("warmup_iterations", 5)
            logger.info(f"모델 워밍업 시작 ({warmup_iterations}회)")
            
            # 더미 입력 생성
            input_config = self.config.get_model_config().get("input", {})
            image_size = input_config.get("image_size", [640, 640])
            channels = input_config.get("channels", 3)
            
            dummy_input = torch.randn(1, channels, *image_size).to(self.device_manager.device)
            
            with torch.no_grad():
                for i in range(warmup_iterations):
                    _ = self.model(dummy_input)
                    if i == 0:
                        # 첫 번째 추론 후 메모리 정리
                        self.device_manager.clear_memory()
            
            logger.info("모델 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"모델 워밍업 실패 (계속 진행): {e}")
    
    def _postprocess_results(self, model_output: Dict[str, Any]) -> List[OCRResult]:
        """모델 출력 후처리"""
        try:
            texts = model_output.get("texts", [])
            boxes = model_output.get("boxes", torch.tensor([]))
            scores = model_output.get("scores", torch.tensor([]))
            
            results = []
            
            for i, text in enumerate(texts):
                if i < len(boxes) and i < len(scores):
                    box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]
                    score = float(scores[i]) if torch.is_tensor(scores[i]) else float(scores[i])
                    
                    result = OCRResult(
                        text=text,
                        confidence=score,
                        bbox=tuple(box.astype(int))
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"후처리 실패: {e}")
            return []
    
    def infer(self, image: np.ndarray) -> InferenceResult:
        """OCR 추론 실행"""
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        start_time = time.time()
        
        try:
            # 이미지 전처리
            input_tensor = self.preprocessor.preprocess(image)
            input_tensor = input_tensor.to(self.device_manager.device)
            
            # 추론 실행
            with torch.no_grad():
                model_output = self.model(input_tensor)
            
            # 후처리
            ocr_results = self._postprocess_results(model_output)
            
            # 전체 텍스트 결합
            total_text = "\n".join([result.text for result in ocr_results])
            
            processing_time = time.time() - start_time
            
            # 메모리 사용량 조회
            memory_usage = self.device_manager.get_memory_usage()
            
            return InferenceResult(
                results=ocr_results,
                total_text=total_text,
                processing_time=processing_time,
                model_version=self.config.get_model_config().get("version", "unknown"),
                device_used=str(self.device_manager.device),
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"추론 실행 실패: {e}")
            raise
        finally:
            # 메모리 정리
            if self.device_manager.device.type == "cuda":
                torch.cuda.empty_cache()
    
    async def infer_async(self, image: np.ndarray) -> InferenceResult:
        """비동기 OCR 추론"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.infer, image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        model_config = self.config.get_model_config()
        
        return {
            "model_name": model_config.get("name", "unknown"),
            "model_version": model_config.get("version", "unknown"),
            "model_type": model_config.get("type", "pytorch"),
            "device_info": self.device_manager.device_info,
            "is_loaded": self.is_loaded,
            "input_size": model_config.get("input", {}).get("image_size", [640, 640])
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """모델 서버 헬스 상태"""
        return {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "model_loaded": self.is_loaded,
            "device": str(self.device_manager.device),
            "memory_usage": self.device_manager.get_memory_usage()
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model is not None:
                del self.model
            
            self.device_manager.clear_memory()
            self.executor.shutdown(wait=True)
            
            logger.info("모델 서버 정리 완료")
            
        except Exception as e:
            logger.error(f"모델 서버 정리 실패: {e}")


# 전역 모델 서버 인스턴스
model_server = PyTorchModelServer() 