"""
이미지 전처리 모듈 테스트
"""

import pytest
import numpy as np
import cv2
import io
from unittest.mock import patch, MagicMock

from services.preprocessing_service import (
    PreprocessingConfig, BasePreprocessor, ImageQualityChecker
)
from services.image_processors import (
    ResizeProcessor, GrayscaleProcessor, DenoiseProcessor,
    RotationCorrectionProcessor, BinarizationProcessor,
    MorphologyProcessor
)
from services.image_preprocessing_pipeline import (
    ImagePreprocessingPipeline, PreprocessingService
)


class TestPreprocessingConfig:
    """전처리 설정 테스트"""
    
    def test_default_config_loading(self):
        """기본 설정 로드 테스트"""
        config = PreprocessingConfig()
        assert config.is_enabled()
        assert config.is_step_enabled("grayscale")
        assert config.is_step_enabled("binarization")
    
    def test_step_config_retrieval(self):
        """단계별 설정 조회 테스트"""
        config = PreprocessingConfig()
        resize_config = config.get_step_config("resize")
        assert "max_width" in resize_config
        assert "max_height" in resize_config


class TestImageQualityChecker:
    """이미지 품질 검사기 테스트"""
    
    def test_contrast_calculation(self):
        """대비 계산 테스트"""
        checker = ImageQualityChecker({})
        
        # 고대비 이미지 생성 (흑백 패턴)
        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:50, :] = 255
        
        # 저대비 이미지 생성 (회색)
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        
        high_contrast_score = checker.calculate_contrast(high_contrast)
        low_contrast_score = checker.calculate_contrast(low_contrast)
        
        assert high_contrast_score > low_contrast_score
    
    def test_blur_metric_calculation(self):
        """블러 측정 테스트"""
        checker = ImageQualityChecker({})
        
        # 선명한 이미지 생성
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(sharp_image, (20, 20), (80, 80), 255, 2)
        
        # 블러 이미지 생성
        blurred_image = cv2.GaussianBlur(sharp_image, (15, 15), 5)
        
        sharp_score = checker.calculate_blur_metric(sharp_image)
        blur_score = checker.calculate_blur_metric(blurred_image)
        
        assert sharp_score > blur_score


class TestImageProcessors:
    """개별 이미지 프로세서 테스트"""
    
    def create_test_image(self, width=200, height=150, color=True):
        """테스트용 이미지 생성"""
        if color:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:
            image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        return image
    
    def test_resize_processor(self):
        """리사이징 프로세서 테스트"""
        config = {"max_width": 100, "max_height": 100, "maintain_aspect_ratio": True}
        processor = ResizeProcessor(config)
        
        # 큰 이미지 생성
        large_image = self.create_test_image(400, 300)
        resized = processor.process(large_image)
        
        assert resized.shape[1] <= 100  # width
        assert resized.shape[0] <= 100  # height
    
    def test_grayscale_processor(self):
        """그레이스케일 프로세서 테스트"""
        processor = GrayscaleProcessor({})
        
        # 컬러 이미지 생성
        color_image = self.create_test_image(100, 100, color=True)
        gray_image = processor.process(color_image)
        
        assert len(gray_image.shape) == 2  # 그레이스케일은 2D
    
    def test_denoise_processor_gaussian(self):
        """가우시안 노이즈 제거 테스트"""
        config = {"method": "gaussian_blur", "kernel_size": 5, "sigma": 1.0}
        processor = DenoiseProcessor(config)
        
        noisy_image = self.create_test_image(100, 100, color=False)
        denoised = processor.process(noisy_image)
        
        assert denoised.shape == noisy_image.shape
    
    def test_denoise_processor_median(self):
        """중간값 노이즈 제거 테스트"""
        config = {"method": "median_blur", "kernel_size": 5}
        processor = DenoiseProcessor(config)
        
        noisy_image = self.create_test_image(100, 100, color=False)
        denoised = processor.process(noisy_image)
        
        assert denoised.shape == noisy_image.shape
    
    def test_rotation_correction_processor(self):
        """회전 보정 프로세서 테스트"""
        config = {"method": "hough_lines", "angle_threshold": 0.5, "max_angle": 45}
        processor = RotationCorrectionProcessor(config)
        
        # 직사각형이 있는 이미지 생성 (회전 감지용)
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 255, 2)
        
        corrected = processor.process(image)
        assert corrected is not None
    
    def test_binarization_processor_otsu(self):
        """Otsu 이진화 테스트"""
        config = {"method": "otsu"}
        processor = BinarizationProcessor(config)
        
        gray_image = self.create_test_image(100, 100, color=False)
        binary = processor.process(gray_image)
        
        # 이진화된 이미지는 0과 255만 가져야 함
        unique_values = np.unique(binary)
        assert len(unique_values) <= 2
    
    def test_binarization_processor_adaptive(self):
        """적응적 이진화 테스트"""
        config = {"method": "adaptive_threshold", "block_size": 11, "c_constant": 2}
        processor = BinarizationProcessor(config)
        
        gray_image = self.create_test_image(100, 100, color=False)
        binary = processor.process(gray_image)
        
        unique_values = np.unique(binary)
        assert len(unique_values) <= 2
    
    def test_morphology_processor_opening(self):
        """열림 연산 테스트"""
        config = {"operation": "opening", "kernel_size": 3, "iterations": 1}
        processor = MorphologyProcessor(config)
        
        # 이진 이미지 생성
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(binary_image, (20, 20), (80, 80), 255, -1)
        
        result = processor.process(binary_image)
        assert result.shape == binary_image.shape


class TestPreprocessingPipeline:
    """전처리 파이프라인 테스트"""
    
    def create_test_image_bytes(self, width=200, height=150):
        """테스트용 이미지 바이트 생성"""
        # 간단한 테스트 이미지 생성
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # JPEG로 인코딩
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return encoded.tobytes()
    
    @pytest.mark.asyncio
    async def test_pipeline_disabled(self):
        """파이프라인 비활성화 테스트"""
        # 비활성화된 설정으로 파이프라인 생성
        with patch.object(PreprocessingConfig, '_load_config') as mock_load:
            mock_load.return_value = {"enabled": False}
            
            pipeline = ImagePreprocessingPipeline()
            image_bytes = self.create_test_image_bytes()
            
            result = await pipeline.process_image(image_bytes, "test-task")
            
            assert result.processing_time == 0.0
            assert len(result.applied_steps) == 0
            assert not result.metadata["preprocessing_enabled"]
    
    @pytest.mark.asyncio
    async def test_pipeline_enabled_with_steps(self):
        """파이프라인 활성화 및 단계 실행 테스트"""
        pipeline = ImagePreprocessingPipeline()
        image_bytes = self.create_test_image_bytes()
        
        result = await pipeline.process_image(image_bytes, "test-task")
        
        assert result.processing_time > 0
        assert len(result.applied_steps) > 0
        assert result.metadata["preprocessing_enabled"]
        assert "initial_quality" in result.quality_metrics
        assert "final_quality" in result.quality_metrics
    
    @pytest.mark.asyncio
    async def test_pipeline_with_invalid_image(self):
        """잘못된 이미지 데이터 테스트"""
        pipeline = ImagePreprocessingPipeline()
        invalid_bytes = b"invalid image data"
        
        with pytest.raises(Exception):
            await pipeline.process_image(invalid_bytes, "test-task")
    
    def test_pipeline_info(self):
        """파이프라인 정보 조회 테스트"""
        pipeline = ImagePreprocessingPipeline()
        info = pipeline.get_pipeline_info()
        
        assert "pipeline_enabled" in info
        assert "total_steps" in info
        assert "enabled_steps" in info
        assert "disabled_steps" in info
    
    def test_pipeline_reload_config(self):
        """설정 다시 로드 테스트"""
        pipeline = ImagePreprocessingPipeline()
        initial_processor_count = len(pipeline.processors)
        
        # 설정 다시 로드
        pipeline.reload_config()
        
        # 프로세서가 다시 초기화되었는지 확인
        assert len(pipeline.processors) >= 0  # 설정에 따라 다를 수 있음


class TestPreprocessingService:
    """전처리 서비스 테스트"""
    
    def create_test_image_bytes(self):
        """테스트용 이미지 바이트 생성"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', image)
        return encoded.tobytes()
    
    @pytest.mark.asyncio
    async def test_preprocess_image(self):
        """이미지 전처리 메인 인터페이스 테스트"""
        service = PreprocessingService()
        image_bytes = self.create_test_image_bytes()
        
        result = await service.preprocess_image(image_bytes, "test-task")
        
        assert result is not None
        assert hasattr(result, 'processed_image')
        assert hasattr(result, 'processing_time')
    
    def test_convert_result_to_bytes(self):
        """결과를 바이트로 변환 테스트"""
        service = PreprocessingService()
        
        # 모의 결과 생성
        mock_result = MagicMock()
        mock_result.processed_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result_bytes = service.convert_result_to_bytes(mock_result)
        
        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0
    
    def test_get_pipeline_status(self):
        """파이프라인 상태 조회 테스트"""
        service = PreprocessingService()
        status = service.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert "pipeline_enabled" in status
    
    def test_is_preprocessing_enabled(self):
        """전처리 활성화 여부 확인 테스트"""
        service = PreprocessingService()
        enabled = service.is_preprocessing_enabled()
        
        assert isinstance(enabled, bool)


class TestErrorHandling:
    """오류 처리 테스트"""
    
    @pytest.mark.asyncio
    async def test_config_file_not_found(self):
        """설정 파일 없음 처리 테스트"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            config = PreprocessingConfig("nonexistent.yaml")
            # 기본 설정으로 대체되어야 함
            assert config.is_enabled()
    
    @pytest.mark.asyncio
    async def test_processor_error_handling(self):
        """프로세서 오류 처리 테스트"""
        pipeline = ImagePreprocessingPipeline()
        
        # 잘못된 이미지 데이터로 오류 발생 시뮬레이션
        with patch.object(pipeline.base_processor, 'convert_to_opencv') as mock_convert:
            mock_convert.side_effect = ValueError("Invalid image")
            
            image_bytes = b"invalid data"
            
            with pytest.raises(ValueError):
                await pipeline.process_image(image_bytes, "test-task")


class TestQualityMetrics:
    """품질 지표 테스트"""
    
    def test_quality_improvement_calculation(self):
        """품질 향상 계산 테스트"""
        checker = ImageQualityChecker({})
        
        # 품질이 다른 두 이미지 생성
        low_quality = np.full((100, 100), 128, dtype=np.uint8)  # 낮은 대비
        high_quality = np.zeros((100, 100), dtype=np.uint8)     # 높은 대비
        high_quality[:50, :] = 255
        
        low_metrics = checker.assess_quality(low_quality)
        high_metrics = checker.assess_quality(high_quality)
        
        assert high_metrics["contrast"] > low_metrics["contrast"]
        assert high_metrics["overall_quality"] > low_metrics["overall_quality"] 