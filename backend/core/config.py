"""
애플리케이션 설정 관리
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    app_name: str = Field(default="AI OCR 서버", description="애플리케이션 이름")
    debug: bool = Field(default=False, description="디버그 모드")
    
    # 파일 업로드 설정
    upload_dir: str = Field(default="uploads", description="업로드 파일 저장 디렉토리")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="최대 파일 크기 (bytes)")
    allowed_extensions: set = Field(
        default={"jpg", "jpeg", "png", "tiff", "pdf"}, 
        description="허용되는 파일 확장자"
    )
    
    # Redis 설정
    redis_host: str = Field(default="localhost", description="Redis 호스트")
    redis_port: int = Field(default=6379, description="Redis 포트")
    redis_db: int = Field(default=0, description="Redis 데이터베이스 번호")
    redis_password: Optional[str] = Field(default=None, description="Redis 비밀번호")
    
    # 작업 큐 설정
    ocr_queue_name: str = Field(default="ocr_processing_queue", description="OCR 처리 큐 이름")
    max_workers: int = Field(default=4, description="최대 워커 수")
    task_timeout: int = Field(default=300, description="작업 타임아웃 (초)")
    
    # 로깅 설정
    log_level: str = Field(default="INFO", description="로그 레벨")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 글로벌 설정 인스턴스
settings = Settings()


def get_upload_path() -> str:
    """업로드 디렉토리 경로 반환"""
    upload_path = os.path.abspath(settings.upload_dir)
    os.makedirs(upload_path, exist_ok=True)
    return upload_path


def get_redis_url() -> str:
    """Redis 연결 URL 생성"""
    if settings.redis_password:
        return f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    else:
        return f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}" 