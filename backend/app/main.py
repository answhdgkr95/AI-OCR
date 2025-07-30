"""
AI OCR 서버 메인 애플리케이션
금융권 내부망 환경에서 안전하게 동작하는 OCR API 서버
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.v1.endpoints import ocr
from services.ocr_worker import worker_manager


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    logger.info("AI OCR 서버 시작 중...")
    
    # OCR 워커 시작
    try:
        await worker_manager.start_workers()
        logger.info("OCR 워커 시작 완료")
    except Exception as e:
        logger.error(f"OCR 워커 시작 실패: {e}")
    
    yield
    
    # OCR 워커 중지
    try:
        await worker_manager.stop_workers()
        logger.info("OCR 워커 중지 완료")
    except Exception as e:
        logger.error(f"OCR 워커 중지 실패: {e}")
    
    logger.info("AI OCR 서버 종료 중...")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI OCR 서버",
    description="금융권 내부망용 고정밀 OCR 추론 API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS 미들웨어 설정 (내부망 환경)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리기"""
    logger.error(f"예상치 못한 오류 발생: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "서버 내부 오류가 발생했습니다.",
            "detail": "시스템 관리자에게 문의하세요."
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리기"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


# API 라우터 등록
app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])


@app.get("/")
async def root():
    """서버 상태 확인"""
    worker_status = worker_manager.get_worker_status()
    
    return {
        "message": "AI OCR 서버가 정상 동작 중입니다.",
        "version": "1.0.0",
        "status": "healthy",
        "workers": {
            "running": worker_status["running_workers"],
            "total": worker_status["worker_count"],
            "processed": worker_status["total_processed"]
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "서버가 정상 상태입니다."}


@app.get("/workers/status")
async def get_worker_status():
    """워커 상태 조회"""
    return worker_manager.get_worker_status()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 