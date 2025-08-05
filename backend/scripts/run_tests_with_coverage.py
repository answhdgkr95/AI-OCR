#!/usr/bin/env python3
"""
CI/CD용 테스트 및 커버리지 실행 스크립트
"""

import sys
import subprocess
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def run_command(cmd, check=True):
    """명령어 실행"""
    print(f"실행 중: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if check and result.returncode != 0:
        print(f"명령어 실행 실패: {' '.join(cmd)}")
        sys.exit(result.returncode)
    
    return result


def extract_coverage_from_xml(xml_path="coverage.xml"):
    """XML 파일에서 커버리지 정보 추출"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 전체 커버리지 정보
        coverage_elem = root.find(".")
        lines_covered = int(coverage_elem.get("lines-covered", 0))
        lines_valid = int(coverage_elem.get("lines-valid", 1))
        line_rate = float(coverage_elem.get("line-rate", 0))
        
        return {
            "lines_covered": lines_covered,
            "lines_valid": lines_valid,
            "line_rate": line_rate,
            "percentage": round(line_rate * 100, 2)
        }
    except Exception as e:
        print(f"XML 파서 에러: {e}")
        return {"percentage": 0.0}


def generate_coverage_badge_data(coverage_percentage):
    """커버리지 뱃지 데이터 생성"""
    if coverage_percentage >= 80:
        color = "brightgreen"
    elif coverage_percentage >= 60:
        color = "yellow"
    elif coverage_percentage >= 40:
        color = "orange"
    else:
        color = "red"
    
    badge_data = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{coverage_percentage}%",
        "color": color
    }
    
    return badge_data


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("AI OCR Server - 테스트 및 커버리지 실행")
    print("=" * 60)
    
    # 1. 테스트 실행
    print("\n1. 테스트 실행 중...")
    test_result = run_command([
        sys.executable, "-m", "pytest",
        "tests/test_preprocessing.py",
        "tests/test_pytorch_model_server.py", 
        "tests/test_ocr_integration.py",
        "-v",
        "--tb=short"
    ], check=False)
    
    if test_result.returncode != 0:
        print("❌ 테스트 실패!")
        return test_result.returncode
    
    print("✅ 모든 테스트 통과!")
    
    # 2. 커버리지 정보 추출
    print("\n2. 커버리지 정보 추출 중...")
    if Path("coverage.xml").exists():
        coverage_info = extract_coverage_from_xml()
        print(f"📊 커버리지: {coverage_info['percentage']}%")
        print(f"   - 커버된 라인: {coverage_info.get('lines_covered', 'N/A')}")
        print(f"   - 전체 라인: {coverage_info.get('lines_valid', 'N/A')}")
        
        # 3. 뱃지 데이터 생성
        badge_data = generate_coverage_badge_data(coverage_info['percentage'])
        with open("coverage-badge.json", "w") as f:
            json.dump(badge_data, f, indent=2)
        print("✅ 커버리지 뱃지 데이터 생성: coverage-badge.json")
        
        # 4. 커버리지 임계값 확인
        if coverage_info['percentage'] < 48:
            print(f"⚠️  경고: 커버리지가 임계값(48%) 미달: {coverage_info['percentage']}%")
            return 1
        else:
            print(f"✅ 커버리지 임계값 통과: {coverage_info['percentage']}% >= 48%")
    else:
        print("⚠️  coverage.xml 파일을 찾을 수 없습니다.")
        return 1
    
    print("\n🎉 테스트 및 커버리지 검사 완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 