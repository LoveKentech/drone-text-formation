"""
프로젝트 전반에서 사용하는 전역 상수 및 설정값 모음.
실험 조건(드론 수, 문자열, 크기)과 시뮬레이션 파라미터를 정의한다.
"""

# 데이터 좌표(픽셀) 기준 드론 중심–중심 최소 거리. morph 등에서 scatter 마커보다 크게 두면 겹쳐 보임이 줄어든다.
MIN_SAFE_DIST = 0.3
# ``animate_morph`` 기본 마커 면적(s). 화면상 반경은 대략 √s pt 수준이라 MIN_SAFE_DIST와 함께 조절한다.
MORPH_DRONE_SCATTER_S = 36.0
MAX_STEPS = 500         # 최대 타임스텝
# heuristic 충돌 복구: 무제한 루프 대신 충분히 큰 상한과 종료 조건을 함께 둔다.
COLLISION_RESOLUTION_MAX_PASSES = 100000
COLLISION_RESOLUTION_TIMEOUT_SEC = 200.0
COLLISION_RESOLUTION_STAGNATION_PASSES = 20
DRONE_COUNTS = [50, 100, 150, 200, 250, 300]
STRINGS = ["AL", "LOVE", "KENTECH"]
SIZES = ["small", "medium", "large"]

# main.py 배치 실험: 출발·도착 포메이션 (무작위 초기화 없음)
EXPERIMENT_START_TEXT = "LOVE"
EXPERIMENT_END_TEXT = "KENTECH"
# 전체 비교 실험용 전환 목록. 순서쌍이므로 A→B와 B→A를 별도 조건으로 본다.
EXPERIMENT_TRANSITIONS = [
    ("AL", "LOVE"),
    ("LOVE", "AL"),
    ("LOVE", "KENTECH"),
    ("KENTECH", "LOVE"),
    ("AL", "KENTECH"),
    ("KENTECH", "AL"),
]
# True면 hungarian + cbs trial, False면 hungarian 만 (CBS 탐색 생략)
EXPERIMENT_INCLUDE_CBS = True
# True면 대표 케이스만 실행 (빠른 GIF 확인용), False면 EXPERIMENT_TRANSITIONS × DRONE_COUNTS × SIZES 전체 실행
EXPERIMENT_REPRESENTATIVE_ONLY = False
EXPERIMENT_REPRESENTATIVE_N = 200
EXPERIMENT_REPRESENTATIVE_SIZE = "large"
# 전체 실험 후 GIF로 저장할 대표 케이스. 모든 trial GIF 저장은 시간/용량이 커서 하지 않는다.
EXPERIMENT_GIF_CASES = [
    ("LOVE", "KENTECH", 200, "large"),
    # AL 관련 size 효과 확인용: 같은 n=100에서 small/medium/large 비교
    ("AL", "KENTECH", 100, "small"),
    ("AL", "KENTECH", 100, "medium"),
    ("AL", "KENTECH", 100, "large"),
    ("KENTECH", "AL", 100, "small"),
    ("KENTECH", "AL", 100, "medium"),
    ("KENTECH", "AL", 100, "large"),
]
# CBS 내부 정수 격자 해상도 배율 (1=기존, 2 이상이면 더 촘촘한 격자)
CBS_GRID_SCALE = 2
# 200대 이상에서는 CBS 탐색 트리가 급격히 커질 수 있어 상한 내 실패로 기록한다.
CBS_MAX_ITERATIONS = 100000
# CBS가 이 시간 안에 해를 못 찾으면 실패로 기록한다. Hungarian 폴백은 사용하지 않는다.
CBS_TIMEOUT_SEC = 200.0
IMAGE_SIZE = (200, 100)      # 렌더링 이미지 크기

CONTOUR_METHODS = ["contour", "poisson", "grid"]   # 좌표 샘플링 방식
DEFAULT_METHOD = "contour"                         # 기본 샘플링 방식

# 충돌 회피 시 목표까지 거리가 비슷한 후보들 중 이전 속도와 맞는 방향을 고를 때 허용 오차
DETOUR_TIE_EPS = 0.1
# greedy_timeline_then_resolve 이후 궤적 스무딩(시각용): 시간축 블렌드 + 분리 반복
TRAJECTORY_SMOOTH_BLEND = 0.28
TRAJECTORY_SMOOTH_OUTER_ITERS = 12
TRAJECTORY_SEPARATION_PASSES_PER_ITER = 6
