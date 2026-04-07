"""
프로젝트 전반에서 사용하는 전역 상수 및 설정값 모음.
실험 조건(드론 수, 문자열, 크기)과 시뮬레이션 파라미터를 정의한다.
"""

MIN_SAFE_DIST = 2.0          # 드론 간 최소 안전 거리
MAX_STEPS = 200              # 최대 타임스텝
# run_collision_resolution: 타임라인 0→T-1 스캔을 최대 몇 번 반복할지 (잔여 충돌 감소)
COLLISION_RESOLUTION_MAX_PASSES = 25
DRONE_COUNTS = [20, 50, 100]
STRINGS = ["AL", "LOVE", "KENTECH"]
SIZES = ["small", "medium", "large"]
IMAGE_SIZE = (200, 100)      # 렌더링 이미지 크기

CONTOUR_METHODS = ["contour", "poisson", "grid"]   # 좌표 샘플링 방식
DEFAULT_METHOD = "contour"                         # 기본 샘플링 방식
