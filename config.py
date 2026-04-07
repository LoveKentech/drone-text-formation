"""
프로젝트 전반에서 사용하는 전역 상수 및 설정값 모음.
실험 조건(드론 수, 문자열, 크기)과 시뮬레이션 파라미터를 정의한다.
"""

MIN_SAFE_DIST = 2.0          # 드론 간 최소 안전 거리
MAX_STEPS = 200              # 최대 타임스텝
DRONE_COUNTS = [20, 50, 100]
STRINGS = ["AL", "LOVE", "KENTECH"]
SIZES = ["small", "medium", "large"]
IMAGE_SIZE = (200, 100)      # 렌더링 이미지 크기

# PIL 렌더링 시 글자 사이에 더 넣을 간격(px). 0이면 폰트 기본 자간만 사용.
LETTER_SPACING_PX = 10.0

CONTOUR_METHODS = ["contour", "poisson", "grid"]   # 좌표 샘플링 방식
DEFAULT_METHOD = "contour"                         # 기본 샘플링 방식
