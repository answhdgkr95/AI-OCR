"""
Redis 기반 작업 큐 서비스
비동기 OCR 처리를 위한 큐 관리
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import aioredis
from aioredis import Redis

from core.config import settings, get_redis_url

logger = logging.getLogger(__name__)


class TaskMessage:
    """작업 메시지 클래스"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        file_id: str,
        file_path: str,
        metadata: Dict[str, Any],
        priority: int = 5,
        retry_count: int = 0
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.file_id = file_id
        self.file_path = file_path
        self.metadata = metadata
        self.priority = priority
        self.retry_count = retry_count
        self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "file_id": self.file_id,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMessage":
        """딕셔너리에서 객체 생성"""
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            file_id=data["file_id"],
            file_path=data["file_path"],
            metadata=data["metadata"],
            priority=data.get("priority", 5),
            retry_count=data.get("retry_count", 0)
        )
        task.created_at = data.get("created_at", task.created_at)
        return task


class QueueService:
    """Redis 기반 큐 서비스"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.queue_name = settings.ocr_queue_name
        self.status_prefix = "task_status:"
        self.result_prefix = "task_result:"
        self.error_prefix = "task_error:"
    
    async def connect(self):
        """Redis 연결"""
        try:
            self.redis = await aioredis.from_url(
                get_redis_url(),
                encoding="utf-8",
                decode_responses=True
            )
            # 연결 테스트
            await self.redis.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise
    
    async def disconnect(self):
        """Redis 연결 해제"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis 연결 해제")
    
    async def enqueue_task(self, task: TaskMessage) -> bool:
        """작업을 큐에 추가"""
        try:
            if not self.redis:
                await self.connect()
            
            # 작업 메시지를 JSON으로 직렬화
            task_data = json.dumps(task.to_dict())
            
            # 우선순위 큐에 추가 (priority가 낮을수록 높은 우선순위)
            score = task.priority
            await self.redis.zadd(self.queue_name, {task_data: score})
            
            # 작업 상태를 'pending'으로 설정
            await self.set_task_status(task.task_id, "pending")
            
            logger.info(f"작업 큐에 추가됨: {task.task_id}")
            return True
            
        except Exception as e:
            logger.error(f"작업 큐 추가 실패: {e}")
            return False
    
    async def dequeue_task(self) -> Optional[TaskMessage]:
        """큐에서 작업 가져오기"""
        try:
            if not self.redis:
                await self.connect()
            
            # 우선순위가 가장 높은 작업 가져오기
            result = await self.redis.bzpopmin(self.queue_name, timeout=1)
            
            if result:
                queue_name, task_data, score = result
                task_dict = json.loads(task_data)
                task = TaskMessage.from_dict(task_dict)
                
                # 작업 상태를 'processing'으로 변경
                await self.set_task_status(task.task_id, "processing")
                
                logger.info(f"작업 큐에서 가져옴: {task.task_id}")
                return task
            
            return None
            
        except Exception as e:
            logger.error(f"작업 큐 가져오기 실패: {e}")
            return None
    
    async def set_task_status(self, task_id: str, status: str) -> bool:
        """작업 상태 설정"""
        try:
            if not self.redis:
                await self.connect()
            
            key = f"{self.status_prefix}{task_id}"
            status_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await self.redis.hset(key, mapping=status_data)
            await self.redis.expire(key, 3600 * 24)  # 24시간 후 만료
            
            return True
            
        except Exception as e:
            logger.error(f"작업 상태 설정 실패: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        try:
            if not self.redis:
                await self.connect()
            
            key = f"{self.status_prefix}{task_id}"
            status_data = await self.redis.hgetall(key)
            
            return status_data if status_data else None
            
        except Exception as e:
            logger.error(f"작업 상태 조회 실패: {e}")
            return None
    
    async def set_task_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """작업 결과 저장"""
        try:
            if not self.redis:
                await self.connect()
            
            key = f"{self.result_prefix}{task_id}"
            result_data = {
                "result": json.dumps(result),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis.hset(key, mapping=result_data)
            await self.redis.expire(key, 3600 * 24)  # 24시간 후 만료
            
            # 상태를 'completed'로 변경
            await self.set_task_status(task_id, "completed")
            
            return True
            
        except Exception as e:
            logger.error(f"작업 결과 저장 실패: {e}")
            return False
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 결과 조회"""
        try:
            if not self.redis:
                await self.connect()
            
            key = f"{self.result_prefix}{task_id}"
            result_data = await self.redis.hgetall(key)
            
            if result_data and "result" in result_data:
                return {
                    "result": json.loads(result_data["result"]),
                    "completed_at": result_data.get("completed_at")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"작업 결과 조회 실패: {e}")
            return None
    
    async def set_task_error(self, task_id: str, error: str) -> bool:
        """작업 오류 저장"""
        try:
            if not self.redis:
                await self.connect()
            
            key = f"{self.error_prefix}{task_id}"
            error_data = {
                "error": error,
                "failed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis.hset(key, mapping=error_data)
            await self.redis.expire(key, 3600 * 24)  # 24시간 후 만료
            
            # 상태를 'failed'로 변경
            await self.set_task_status(task_id, "failed")
            
            return True
            
        except Exception as e:
            logger.error(f"작업 오류 저장 실패: {e}")
            return False
    
    async def get_queue_size(self) -> int:
        """큐 크기 조회"""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.zcard(self.queue_name)
            
        except Exception as e:
            logger.error(f"큐 크기 조회 실패: {e}")
            return 0


# 싱글톤 인스턴스
queue_service = QueueService() 