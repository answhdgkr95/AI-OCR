"""
파일 업로드 및 저장 서비스
UUID 기반 파일명 생성과 메타데이터 관리
"""

import uuid
import aiofiles
import os
from typing import Dict, Any
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
import time
import hashlib

from core.config import settings, get_upload_path


class FileMetadata:
    """파일 메타데이터 클래스"""
    
    def __init__(
        self,
        file_id: str,
        original_filename: str,
        content_type: str,
        size: int,
        file_path: str,
        file_hash: str,
        upload_time: float
    ):
        self.file_id = file_id
        self.original_filename = original_filename
        self.content_type = content_type
        self.size = size
        self.file_path = file_path
        self.file_hash = file_hash
        self.upload_time = upload_time
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "file_id": self.file_id,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "size": self.size,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "upload_time": self.upload_time
        }


class FileService:
    """파일 업로드 및 저장 서비스"""
    
    def __init__(self):
        self.upload_path = get_upload_path()
    
    def _generate_file_id(self) -> str:
        """유니크한 파일 ID 생성"""
        return str(uuid.uuid4())
    
    def _get_file_extension(self, filename: str) -> str:
        """파일 확장자 추출"""
        return Path(filename).suffix.lower()
    
    def _generate_file_path(self, file_id: str, extension: str) -> str:
        """파일 저장 경로 생성"""
        # 날짜별 서브디렉토리 생성 (YYYY/MM/DD)
        today = time.strftime("%Y/%m/%d")
        subdir = os.path.join(self.upload_path, today)
        os.makedirs(subdir, exist_ok=True)
        
        filename = f"{file_id}{extension}"
        return os.path.join(subdir, filename)
    
    def _calculate_file_hash(self, content: bytes) -> str:
        """파일 해시 계산 (SHA-256)"""
        return hashlib.sha256(content).hexdigest()
    
    def _validate_file(self, file: UploadFile, content: bytes) -> None:
        """파일 유효성 검증"""
        # 파일 크기 검증
        if len(content) > settings.max_file_size:
            max_mb = settings.max_file_size // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"파일 크기가 너무 큽니다. 최대 크기: {max_mb}MB"
            )
        
        # 빈 파일 검증
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="빈 파일입니다. 유효한 이미지 파일을 업로드해주세요."
            )
        
        # 파일 확장자 검증
        if file.filename:
            extension = self._get_file_extension(file.filename)[1:]  # '.' 제거
            if extension not in settings.allowed_extensions:
                allowed_ext_str = ", ".join(settings.allowed_extensions)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"지원하지 않는 파일 형식입니다. "
                           f"허용 형식: {allowed_ext_str}"
                )
    
    async def save_uploaded_file(self, file: UploadFile) -> FileMetadata:
        """업로드된 파일을 저장하고 메타데이터 반환"""
        try:
            # 파일 내용 읽기
            content = await file.read()
            
            # 파일 유효성 검증
            self._validate_file(file, content)
            
            # 파일 메타데이터 생성
            file_id = self._generate_file_id()
            extension = self._get_file_extension(file.filename or "")
            file_path = self._generate_file_path(file_id, extension)
            file_hash = self._calculate_file_hash(content)
            
            # 파일 저장
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # 메타데이터 객체 생성
            metadata = FileMetadata(
                file_id=file_id,
                original_filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                size=len(content),
                file_path=file_path,
                file_hash=file_hash,
                upload_time=time.time()
            )
            
            return metadata
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"파일 저장 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def get_file_content(self, file_path: str) -> bytes:
        """저장된 파일 내용 읽기"""
        try:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="파일을 찾을 수 없습니다."
                )
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"파일 읽기 중 오류가 발생했습니다: {str(e)}"
            )
    
    def delete_file(self, file_path: str) -> bool:
        """파일 삭제"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False


# 싱글톤 인스턴스
file_service = FileService() 