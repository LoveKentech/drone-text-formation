"""
프로젝트 전반에서 사용하는 전역 상수 및 설정값 모음.
실험 조건(드론 수, 문자열, 크기)과 시뮬레이션 파라미터를 정의한다.
"""

# 데이터 좌표(픽셀) 기준 드론 중심–중심 최소 거리. morph 등에서 scatter 마커보다 크게 두면 겹쳐 보임이 줄어든다.
MIN_SAFE_DIST = 6
# ``animate_morph`` 기본 마커 면적(s). 화면상 반경은 대략 √s pt 수준이라 MIN_SAFE_DIST와 함께 조절한다.
MORPH_DRONE_SCATTER_S = 36.0
MAX_STEPS = 200              # 최대 타임스텝
# run_collision_resolution: 타임라인 0→T-1 스캔을 최대 몇 번 반복할지 (잔여 충돌 감소)
COLLISION_RESOLUTION_MAX_PASSES = 25
DRONE_COUNTS = [50, 100, 200]
STRINGS = ["AL", "LOVE", "KENTECH"]
SIZES = ["small", "medium", "large"]

# main.py 배치 실험: 출발·도착 포메이션 (무작위 초기화 없음)
EXPERIMENT_START_TEXT = "LOVE"
EXPERIMENT_END_TEXT = "KENTECH"
# True면 hungarian + cbs trial, False면 hungarian 만 (CBS 탐색 생략)
EXPERIMENT_INCLUDE_CBS = False
IMAGE_SIZE = (200, 100)      # 렌더링 이미지 크기

CONTOUR_METHODS = ["contour", "poisson", "grid"]   # 좌표 샘플링 방식
DEFAULT_METHOD = "contour"                         # 기본 샘플링 방식

# 충돌 회피 시 목표까지 거리가 비슷한 후보들 중 이전 속도와 맞는 방향을 고를 때 허용 오차
DETOUR_TIE_EPS = 0.1
# greedy_timeline_then_resolve 이후 궤적 스무딩(시각용): 시간축 블렌드 + 분리 반복
TRAJECTORY_SMOOTH_BLEND = 0.28
TRAJECTORY_SMOOTH_OUTER_ITERS = 12
TRAJECTORY_SEPARATION_PASSES_PER_ITER = 6
