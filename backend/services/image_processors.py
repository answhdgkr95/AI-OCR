"""
개별 이미지 처리 기능들
각각의 전처리 단계를 담당하는 클래스들
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import albumentations as A
from skimage import morphology, filters
from scipy import ndimage

logger = logging.getLogger(__name__)


class ResizeProcessor:
    """이미지 리사이징 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_width = config.get("max_width", 2048)
        self.max_height = config.get("max_height", 2048)
        self.maintain_aspect_ratio = config.get("maintain_aspect_ratio", True)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """이미지 리사이징"""
        h, w = image.shape[:2]
        
        # 리사이징이 필요한지 확인
        if w <= self.max_width and h <= self.max_height:
            return image
        
        if self.maintain_aspect_ratio:
            # 종횡비 유지하며 리사이징
            scale = min(self.max_width / w, self.max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = self.max_width
            new_h = self.max_height
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"이미지 리사이징: {w}x{h} -> {new_w}x{new_h}")
        
        return resized


class GrayscaleProcessor:
    """그레이스케일 변환 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """그레이스케일 변환"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug("이미지를 그레이스케일로 변환")
            return gray
        return image


class DenoiseProcessor:
    """노이즈 제거 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "gaussian_blur")
        self.kernel_size = config.get("kernel_size", 3)
        self.sigma = config.get("sigma", 1.0)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        if self.method == "gaussian_blur":
            return self._gaussian_blur(image)
        elif self.method == "median_blur":
            return self._median_blur(image)
        elif self.method == "bilateral":
            return self._bilateral_filter(image)
        else:
            logger.warning(f"알 수 없는 노이즈 제거 방법: {self.method}")
            return image
    
    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """가우시안 블러"""
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
    
    def _median_blur(self, image: np.ndarray) -> np.ndarray:
        """중간값 필터"""
        return cv2.medianBlur(image, self.kernel_size)
    
    def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """양방향 필터"""
        return cv2.bilateralFilter(image, -1, 80, 80)


class RotationCorrectionProcessor:
    """회전 보정 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "hough_lines")
        self.angle_threshold = config.get("angle_threshold", 0.5)
        self.max_angle = config.get("max_angle", 45)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """회전 보정"""
        if self.method == "hough_lines":
            angle = self._detect_angle_hough(image)
        elif self.method == "contour_based":
            angle = self._detect_angle_contour(image)
        else:
            logger.warning(f"알 수 없는 회전 감지 방법: {self.method}")
            return image
        
        if abs(angle) < self.angle_threshold:
            logger.debug(f"회전 각도가 임계값 미만: {angle:.2f}도")
            return image
        
        if abs(angle) > self.max_angle:
            logger.warning(f"회전 각도가 너무 큼: {angle:.2f}도")
            return image
        
        return self._rotate_image(image, angle)
    
    def _detect_angle_hough(self, image: np.ndarray) -> float:
        """Hough 변환을 이용한 각도 감지"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough 직선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        # 각도 계산
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < self.max_angle:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # 중간값 사용 (노이즈에 강함)
        median_angle = float(np.median(angles))
        logger.debug(f"Hough 변환으로 감지된 회전 각도: {median_angle:.2f}도")
        
        return median_angle
    
    def _detect_angle_contour(self, image: np.ndarray) -> float:
        """윤곽선 기반 각도 감지"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 가장 큰 윤곽선 사용
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 최소 면적 사각형 찾기
        rect = cv2.minAreaRect(largest_contour)
        angle = float(rect[2])
        
        # 각도 정규화
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        logger.debug(f"윤곽선 기반 감지된 회전 각도: {angle:.2f}도")
        return angle
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 새로운 이미지 크기 계산 (잘림 방지)
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # 회전 중심 조정
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # 이미지 회전
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        logger.debug(f"이미지 회전 완료: {angle:.2f}도")
        return rotated


class BinarizationProcessor:
    """이진화 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "adaptive_threshold")
        self.block_size = config.get("block_size", 11)
        self.c_constant = config.get("c_constant", 2)
        self.threshold_value = config.get("threshold_value", 127)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """이진화"""
        # 그레이스케일 확인
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self.method == "otsu":
            return self._otsu_threshold(gray)
        elif self.method == "adaptive_threshold":
            return self._adaptive_threshold(gray)
        elif self.method == "manual":
            return self._manual_threshold(gray)
        else:
            logger.warning(f"알 수 없는 이진화 방법: {self.method}")
            return gray
    
    def _otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Otsu 이진화"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug("Otsu 이진화 적용")
        return binary
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """적응적 이진화"""
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, self.block_size, self.c_constant
        )
        logger.debug(f"적응적 이진화 적용: block_size={self.block_size}, C={self.c_constant}")
        return binary
    
    def _manual_threshold(self, image: np.ndarray) -> np.ndarray:
        """수동 이진화"""
        _, binary = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        logger.debug(f"수동 이진화 적용: threshold={self.threshold_value}")
        return binary


class MorphologyProcessor:
    """모폴로지 연산 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.operation = config.get("operation", "opening")
        self.kernel_size = config.get("kernel_size", 2)
        self.iterations = config.get("iterations", 1)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """모폴로지 연산"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (self.kernel_size, self.kernel_size))
        
        if self.operation == "opening":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, 
                                    iterations=self.iterations)
        elif self.operation == "closing":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, 
                                    iterations=self.iterations)
        elif self.operation == "gradient":
            result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, 
                                    iterations=self.iterations)
        else:
            logger.warning(f"알 수 없는 모폴로지 연산: {self.operation}")
            return image
        
        logger.debug(f"모폴로지 연산 적용: {self.operation}")
        return result


class DistortionCorrectionProcessor:
    """왜곡 보정 처리기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "perspective_transform")
        self.auto_detect = config.get("auto_detect", True)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """왜곡 보정"""
        if not self.auto_detect:
            return image
        
        if self.method == "perspective_transform":
            return self._perspective_correction(image)
        else:
            logger.warning(f"알 수 없는 왜곡 보정 방법: {self.method}")
            return image
    
    def _perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """원근 변환 보정"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 윤곽선 찾기
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # 가장 큰 사각형 윤곽선 찾기
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # 사각형 발견
                # 원근 변환 적용
                return self._apply_perspective_transform(image, approx)
        
        return image
    
    def _apply_perspective_transform(self, image: np.ndarray, 
                                   corners: np.ndarray) -> np.ndarray:
        """원근 변환 적용"""
        # 모서리 정렬
        corners = corners.reshape(4, 2)
        ordered_corners = self._order_corners(corners)
        
        # 목표 크기 계산
        h, w = image.shape[:2]
        dst_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), 
                                           dst_corners)
        
        # 원근 변환 적용
        corrected = cv2.warpPerspective(image, matrix, (w, h))
        logger.debug("원근 변환 보정 적용")
        
        return corrected
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """모서리를 시계방향으로 정렬"""
        # 합계와 차이를 이용한 정렬
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]      # 좌상단
        ordered[1] = corners[np.argmin(diff)]   # 우상단
        ordered[2] = corners[np.argmax(s)]      # 우하단
        ordered[3] = corners[np.argmax(diff)]   # 좌하단
        
        return ordered 