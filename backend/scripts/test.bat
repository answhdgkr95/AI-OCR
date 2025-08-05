@echo off
REM Windows용 테스트 실행 배치 스크립트

echo ========================================
echo AI OCR Server - 테스트 실행
echo ========================================

if "%1"=="unit" (
    echo 단위 테스트만 실행...
    pytest tests/test_preprocessing.py tests/test_pytorch_model_server.py -v
) else if "%1"=="integration" (
    echo 통합 테스트만 실행...
    pytest tests/test_ocr_integration.py -v
) else if "%1"=="coverage" (
    echo 커버리지 포함 전체 테스트 실행...
    python scripts/run_tests_with_coverage.py
) else if "%1"=="fast" (
    echo 빠른 테스트 실행 (커버리지 없음)...
    pytest tests/test_preprocessing.py tests/test_pytorch_model_server.py tests/test_ocr_integration.py -v --tb=short
) else (
    echo 사용법: test.bat [unit^|integration^|coverage^|fast]
    echo.
    echo   unit        - 단위 테스트만 실행
    echo   integration - 통합 테스트만 실행  
    echo   coverage    - 전체 테스트 + 커버리지 측정
    echo   fast        - 전체 테스트 (커버리지 없음)
    echo.
    echo 기본값으로 fast 모드 실행...
    pytest tests/test_preprocessing.py tests/test_pytorch_model_server.py tests/test_ocr_integration.py -v --tb=short
) 