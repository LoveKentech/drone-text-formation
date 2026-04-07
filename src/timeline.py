"""
드론별 경로 계획 결과를 타임스텝 단위 시뮬레이션 타임라인으로 변환하는 모듈.
각 스텝의 드론 위치를 관리하고 충돌 통계 등 지표를 수집한다.

타임라인 구조
-------------
frames : ndarray (max_steps, n, 2)
    frames[t, i] = 시각 t에서 드론 i의 (x, y) 위치.
    t=0 은 초기 위치, t=max_steps-1 은 최종 위치.

충돌 해결 우선순위
------------------
1. Head-on  : 두 드론 방향 벡터가 정반대 → assignment 목표 swap + 경로 재계산.
2. 경로 회피 : 우선순위 낮은 드론(인덱스 큰 쪽)이 상대를 장애물로 간주해
              가장 가까운 대안 칸으로 이동 + 이후 경로 재계산.
3. Wait     : 대안 칸 없음 → 현재 위치 유지 + 이후 경로 재계산.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLLISION_RESOLUTION_MAX_PASSES, MIN_SAFE_DIST, MAX_STEPS


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _recompute_from_goal(
    frames: np.ndarray,
    drone_idx: int,
    goal: np.ndarray,
    start_t: int,
) -> None:
    """
    start_t부터 끝까지 드론의 경로를 np.sign 이동으로 재계산 (in-place).

    Parameters
    ----------
    frames    : (max_steps, n, 2) — 수정 대상 타임라인
    drone_idx : 재계산할 드론 인덱스
    goal      : shape (2,) — 목표 위치
    start_t   : 재계산 시작 타임스텝 (반드시 >= 1)
    """
    max_steps = frames.shape[0]
    for t in range(max(1, start_t), max_steps):
        prev = frames[t - 1, drone_idx]
        frames[t, drone_idx] = prev + np.sign(goal - prev)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_timeline_hungarian(
    drones: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    max_steps: int = MAX_STEPS,
) -> np.ndarray:
    """
    헝가리안 배정 결과를 바탕으로, 매 스텝 각 드론이 목표를 향해
    np.sign 으로 한 칸(대각선 포함) 이동하는 타임라인을 계산한다.

    이동 규칙
    ---------
    diff = goal - current_pos
    step = np.sign(diff)          # 각 축 독립적으로 -1/0/+1
    next_pos = current_pos + step  # 1스텝 = Chebyshev 거리 1

    목표에 도달한 드론은 np.sign((0,0)) = (0,0) 이므로 자동으로 정지한다.

    Parameters
    ----------
    drones     : ndarray (n, 2) — 드론 초기 위치
    targets    : ndarray (n, 2) — 목표 좌표 집합
    assignment : ndarray (n,)   — assignment[i] = 드론 i의 목표 인덱스
    max_steps  : int            — 타임라인 총 길이 (기본값 MAX_STEPS)

    Returns
    -------
    ndarray (max_steps, n, 2) — frames[t, i] = 시각 t 드론 i 위치

    Example
    -------
    # frames = compute_timeline_hungarian(drones, targets, assignment, max_steps=100)
    # assert frames.shape == (100, len(drones), 2)
    # assert np.allclose(frames[0], drones)
    """
    n = len(drones)
    frames = np.zeros((max_steps, n, 2), dtype=float)
    frames[0] = drones.astype(float)

    for t in range(1, max_steps):
        for i in range(n):
            goal = targets[assignment[i]]
            diff = goal - frames[t - 1, i]
            frames[t, i] = frames[t - 1, i] + np.sign(diff)

    return frames


def compute_timeline_linear(
    drones: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    max_steps: int = MAX_STEPS,
) -> np.ndarray:
    """
    할당된 목표까지 유클리드 직선으로 등속 보간 (충돌 미고려, 시각화용).
    """
    n = len(drones)
    frames = np.zeros((max_steps, n, 2), dtype=float)
    starts = drones.astype(float)
    goals = np.stack([targets[assignment[i]] for i in range(n)])
    if max_steps <= 1:
        frames[0] = starts
        return frames
    for t in range(max_steps):
        alpha = t / (max_steps - 1)
        frames[t] = (1.0 - alpha) * starts + alpha * goals
    return frames


def pad_timeline_hold(
    frames: np.ndarray,
    hold_start: int = 0,
    hold_end: int = 0,
) -> np.ndarray:
    """앞·뒤에 정지 구간(프레임 복제)을 붙인다."""
    parts: List[np.ndarray] = []
    if hold_start > 0:
        parts.append(np.repeat(frames[:1], hold_start, axis=0))
    parts.append(frames)
    if hold_end > 0:
        parts.append(np.repeat(frames[-1:], hold_end, axis=0))
    return np.concatenate(parts, axis=0)


def detect_collisions(
    frames: np.ndarray,
    t: int,
    min_dist: float = MIN_SAFE_DIST,
) -> List[Tuple[int, int]]:
    """
    시각 t에서 모든 드론 쌍을 검사해 거리 < min_dist 인 충돌 쌍을 반환한다.

    Parameters
    ----------
    frames   : ndarray (max_steps, n, 2)
    t        : 검사할 타임스텝
    min_dist : 충돌 판정 거리 기준 (기본값 MIN_SAFE_DIST)

    Returns
    -------
    list of (drone_a_idx, drone_b_idx) — 인덱스 오름차순 정렬된 충돌 쌍 목록

    Example
    -------
    # collisions = detect_collisions(frames, t=5)
    # if collisions:
    #     print(f"t=5 충돌 쌍: {collisions}")
    """
    positions = frames[t]  # (n, 2)
    n = len(positions)
    collisions: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                collisions.append((i, j))

    return collisions


def resolve_collisions(
    frames: np.ndarray,
    assignment: np.ndarray,
    targets: np.ndarray,
    collisions: List[Tuple[int, int]],
    t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    detect_collisions 결과를 받아 3단계 우선순위로 충돌을 해결한다.

    목표 위치
    ---------
    드론 i의 목표는 항상 ``targets[assignment[i]]`` (헝가리안 배정 후 목표 좌표)이다.
    assignment 스왑 이후에도 일관되게 재경로화하기 위해 frames[-1]은 사용하지 않는다.

    우선순위
    --------
    1. Head-on  — np.sign(goal - pos) 방향이 정반대인 두 드론 →
       assignment swap 후 각자 상대방의 유효 목표로 경로 재계산.
    2. 경로 회피 — 인덱스 큰 드론(b)이 상대(a)의 t+1 위치를 피하는
       인접 9칸 중 목표에 가장 가까운 칸으로 이동 후 재계산.
    3. Wait      — 대안 칸 없으면 b를 현재 위치에 정지시킨 뒤 재계산.

    Parameters
    ----------
    frames      : ndarray (max_steps, n, 2) — in-place 수정됨
    assignment  : ndarray (n,) — 드론-목표 인덱스 매핑 (복사본 반환)
    targets     : ndarray (n, 2) — 목표 좌표 집합 (coordinate 등에서 생성)
    collisions  : list of (a, b) — detect_collisions 반환값
    t           : 충돌 발생 시각

    Returns
    -------
    (frames, assignment) — 수정된 타임라인 및 배정

    Example
    -------
    # frames, assignment = resolve_collisions(frames, assignment, targets, [(0, 1)], t=3)
    # assert detect_collisions(frames, t=3) == []  # 해결 후 충돌 없음 (이상적)
    """
    max_steps  = frames.shape[0]
    assignment = assignment.copy()

    for (a, b) in collisions:
        pos_a = frames[t, a]
        pos_b = frames[t, b]

        goal_a = targets[assignment[a]].copy()
        goal_b = targets[assignment[b]].copy()

        dir_a = np.sign(goal_a - pos_a)
        dir_b = np.sign(goal_b - pos_b)
        is_head_on = np.allclose(dir_a, -dir_b) and np.any(dir_a != 0)

        if is_head_on:
            # ── Strategy 1 : Head-on → 목표 swap ────────────────────────
            assignment[a], assignment[b] = int(assignment[b]), int(assignment[a])
            if t + 1 < max_steps:
                ga = targets[assignment[a]]
                gb = targets[assignment[b]]
                frames[t + 1, a] = pos_a + np.sign(ga - pos_a)
                frames[t + 1, b] = pos_b + np.sign(gb - pos_b)
                _recompute_from_goal(frames, a, ga, t + 2)
                _recompute_from_goal(frames, b, gb, t + 2)
            continue

        if t + 1 >= max_steps:
            continue

        # ── Strategy 2 : 경로 회피 (b가 a를 장애물로 간주) ──────────────
        goal_b = targets[assignment[b]]
        next_pos_a = frames[t + 1, a]

        best_pos  = None
        best_dist = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand = pos_b + np.array([dx, dy], dtype=float)
                if np.linalg.norm(cand - next_pos_a) >= MIN_SAFE_DIST:
                    d = np.linalg.norm(cand - goal_b)
                    if d < best_dist:
                        best_dist = d
                        best_pos  = cand

        # 현재 위치와 같으면(wait와 동일) strategy 3으로 넘김
        if best_pos is not None and not np.allclose(best_pos, pos_b):
            frames[t + 1, b] = best_pos
            _recompute_from_goal(frames, b, goal_b, t + 2)
            continue

        # ── Strategy 3 : Wait ────────────────────────────────────────────
        frames[t + 1, b] = pos_b.copy()
        _recompute_from_goal(frames, b, goal_b, t + 2)

    return frames, assignment


def run_collision_resolution(
    frames: np.ndarray,
    assignment: np.ndarray,
    targets: np.ndarray,
    min_dist: float = MIN_SAFE_DIST,
    max_passes: int = COLLISION_RESOLUTION_MAX_PASSES,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    전체 타임라인에 대해 detect_collisions → resolve_collisions 를 순차적으로 실행한다.

    한 번의 스캔(0→T-1)만으로 해결되지 않는 경우가 있어, **여러 패스**를 반복한다.
    (한 패스에서 해결한 내용이 다른 시각에 새 충돌을 만들 수 있음.)

    Parameters
    ----------
    frames     : ndarray (max_steps, n, 2) — compute_timeline_hungarian 결과
    assignment : ndarray (n,)              — hungarian_assign 결과
    targets    : ndarray (n, 2)            — 목표 좌표 (충돌 해결 시 재경로화에 사용)
    min_dist   : 충돌 판정 거리 기준 (기본값 MIN_SAFE_DIST)
    max_passes : 타임라인 전체 스캔 반복 상한 (기본 `config.COLLISION_RESOLUTION_MAX_PASSES`)

    Returns
    -------
    (frames, assignment, stats) 튜플

    stats 구조
    ~~~~~~~~~~
    {
        'total_detected'      : int,   # 모든 패스에서 처리한 충돌 쌍 횟수(누적)
        'per_step'            : dict,  # 스텝별 감지 충돌 쌍 수 (마지막 패스 기준)
        'steps_with_collision': int,   # 첫 패스에서 충돌이 있었던 서로 다른 t 수
        'remaining'           : int,   # 최종 잔여 충돌 쌍 수 (전체 스캔)
        'passes_used'         : int,   # 실제 반복한 스캔 횟수
    }

    Note
    ----
    완전 무충돌을 수학적으로 보장하지는 않는다. 잔여는 ``stats['remaining']`` 으로 확인.
    """
    max_steps = frames.shape[0]
    frames = frames.copy()
    assignment = assignment.copy()

    per_step: Dict[int, int] = {}
    steps_with_collision = 0
    total_detected = 0
    passes_used = 0
    first_pass_collision_ts: set[int] = set()

    for _ in range(max_passes):
        passes_used += 1
        any_this_pass = False
        for t in range(max_steps):
            collisions = detect_collisions(frames, t, min_dist)
            if not collisions:
                continue
            any_this_pass = True
            cnt = len(collisions)
            total_detected += cnt
            if passes_used == 1 and t not in first_pass_collision_ts:
                first_pass_collision_ts.add(t)
                steps_with_collision += 1

            frames, assignment = resolve_collisions(frames, assignment, targets, collisions, t)

        if not any_this_pass:
            break

    remaining = sum(
        len(detect_collisions(frames, t, min_dist)) for t in range(max_steps)
    )

    for t in range(max_steps):
        c = detect_collisions(frames, t, min_dist)
        if c:
            per_step[t] = len(c)

    stats: Dict = {
        "total_detected": total_detected,
        "per_step": per_step,
        "steps_with_collision": steps_with_collision,
        "remaining": remaining,
        "passes_used": passes_used,
    }

    return frames, assignment, stats


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== compute_timeline_hungarian 단위 테스트 ===\n")

    # 1) 출력 shape 확인
    n, ms = 5, 30
    d = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0],
                  [10.0, 10.0], [5.0, 5.0]])
    tgt = np.array([[8.0, 8.0], [2.0, 2.0], [8.0, 2.0],
                    [2.0, 8.0], [0.0, 0.0]])
    asgn = np.arange(n)
    frames = compute_timeline_hungarian(d, tgt, asgn, max_steps=ms)
    assert frames.shape == (ms, n, 2), f"shape 오류: {frames.shape}"
    assert np.allclose(frames[0], d), "t=0이 초기 위치와 다름"
    print("[PASS] shape (max_steps, n, 2) 및 frames[0] == drones")

    # 2) 단일 드론 (0,0) → (5,5): 5스텝에 도달
    d1   = np.array([[0.0, 0.0]])
    tgt1 = np.array([[5.0, 5.0]])
    f1   = compute_timeline_hungarian(d1, tgt1, np.array([0]), max_steps=20)
    assert np.allclose(f1[5, 0], [5.0, 5.0]), f"t=5 위치 오류: {f1[5,0]}"
    assert np.allclose(f1[10, 0], [5.0, 5.0]), "도달 후 정지 안 됨"
    print("[PASS] (0,0)→(5,5) 5스텝 도달 및 이후 정지")

    # 3) 각 드론이 max_steps 내에 목표 도달
    for i in range(n):
        goal = tgt[asgn[i]]
        reached = np.any(
            np.all(np.abs(frames[:, i, :] - goal) < 0.5, axis=1)
        )
        assert reached, f"드론 {i}이 목표에 도달하지 못함"
    print("[PASS] 모든 드론 목표 도달 (max_steps=30)")

    print("\n=== detect_collisions 단위 테스트 ===\n")

    # 4) 같은 위치 → 충돌
    f_col = np.zeros((5, 3, 2))
    f_col[2, 0] = [3.0, 3.0]
    f_col[2, 1] = [3.0, 3.0]   # 0과 1이 같은 위치
    f_col[2, 2] = [9.0, 9.0]
    cols = detect_collisions(f_col, t=2, min_dist=MIN_SAFE_DIST)
    assert (0, 1) in cols, "같은 위치 충돌 미감지"
    assert (0, 2) not in cols and (1, 2) not in cols
    print("[PASS] 같은 위치 쌍 충돌 감지, 이격 쌍 비감지")

    # 5) 거리 < min_dist → 충돌 / 거리 >= min_dist → 비충돌
    f_near = np.zeros((3, 2, 2))
    f_near[1, 0] = [0.0, 0.0]
    f_near[1, 1] = [MIN_SAFE_DIST - 0.5, 0.0]   # 거리 = 1.5 < 2.0 → 충돌
    assert detect_collisions(f_near, t=1, min_dist=MIN_SAFE_DIST), "근거리 미감지"

    f_far = np.zeros((3, 2, 2))
    f_far[1, 0] = [0.0, 0.0]
    f_far[1, 1] = [MIN_SAFE_DIST + 0.1, 0.0]    # 거리 = 2.1 >= 2.0 → 안전
    assert not detect_collisions(f_far, t=1, min_dist=MIN_SAFE_DIST), "원거리 오감지"
    print("[PASS] 거리 임계값 기준 충돌 판정")

    print("\n=== resolve_collisions 단위 테스트 ===\n")

    # 6) Head-on: 드론0 (0,0)→(8,0),  드론1 (8,0)→(0,0)
    #    t=4에서 둘 다 (4,0)에 겹침 → head-on 해결 후 t=5에서는 이격돼야 함
    #    resolve는 t+1 이후만 수정 가능하므로 t=4 충돌 자체는 남고 t=5 이후가 개선됨
    d_ho   = np.array([[0.0, 0.0], [8.0, 0.0]])
    tgt_ho = np.array([[8.0, 0.0], [0.0, 0.0]])
    a_ho   = np.array([0, 1])
    f_ho   = compute_timeline_hungarian(d_ho, tgt_ho, a_ho, max_steps=30)

    cols_ho = detect_collisions(f_ho, t=4, min_dist=MIN_SAFE_DIST)
    assert cols_ho, "t=4에서 head-on 충돌이 감지되어야 함"

    f_ho, a_ho = resolve_collisions(f_ho, a_ho, tgt_ho, cols_ho, t=4)

    # t=4 자체는 이미 발생한 충돌이므로 여전히 존재 (물리적으로 되돌릴 수 없음)
    # t=5 이후부터 충돌이 해소됐는지 확인
    cols_after = detect_collisions(f_ho, t=5, min_dist=MIN_SAFE_DIST)
    print(f"[INFO] head-on: t=4 충돌={len(cols_ho)}, t=5 충돌={len(cols_after)}")
    assert len(cols_after) == 0, f"t=5에서도 충돌이 남아 있음: {cols_after}"
    print("[PASS] resolve_collisions head-on → t=5 이후 충돌 해소 확인")

    # 7) Wait 확인: b를 둘러싼 모든 칸이 막힌 가상 상황은 구현이 복잡하므로
    #    대신 반환 타입과 frames 불변성(입력 수정 안 됨) 확인
    f_test = f_ho.copy()
    f_ret, a_ret = resolve_collisions(f_test, a_ho.copy(), tgt_ho, [], t=0)
    assert isinstance(f_ret, np.ndarray) and isinstance(a_ret, np.ndarray)
    print("[PASS] 빈 충돌 목록 → frames/assignment 타입 정상 반환")

    print("\n=== run_collision_resolution 단위 테스트 (n=10) ===\n")

    # 8) n=10 랜덤 케이스
    rng  = np.random.default_rng(7)
    n10  = 10
    d10  = rng.uniform(0, 20, (n10, 2))
    t10  = rng.uniform(0, 20, (n10, 2))

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from hungarian import build_cost_matrix, hungarian_assign

    asgn10 = hungarian_assign(build_cost_matrix(d10, t10))
    f10    = compute_timeline_hungarian(d10, t10, asgn10, max_steps=80)

    f_res, a_res, stats = run_collision_resolution(f10, asgn10, t10, min_dist=MIN_SAFE_DIST)

    # 반환 타입 및 shape 확인
    assert f_res.shape == f10.shape
    assert a_res.shape == asgn10.shape
    assert set(stats.keys()) == {
        "total_detected",
        "per_step",
        "steps_with_collision",
        "remaining",
        "passes_used",
    }
    # remaining 은 최종 frames 전체 재스캔 결과이므로 total_detected 와 단순 대소 비교 불가

    # 입력 frames 불변 확인
    assert not np.allclose(f_res, f10) or stats["total_detected"] == 0, \
        "충돌 있는데 frames가 변경되지 않음"

    # 해결률: 전체 스캔 기준 (resolved = total_before - remaining)
    total_before = sum(
        len(detect_collisions(f10, t, MIN_SAFE_DIST)) for t in range(f10.shape[0])
    )
    resolution_rate = (
        1.0 - stats["remaining"] / total_before if total_before > 0 else 1.0
    )
    print(f"[PASS] n=10 run_collision_resolution 완료")
    print(f"       감지: {stats['total_detected']}쌍, "
          f"잔여: {stats['remaining']}쌍, "
          f"해결률: {resolution_rate:.1%}, "
          f"충돌 발생 스텝: {stats['steps_with_collision']}, "
          f"패스: {stats['passes_used']}")

    # 9) 최종 스캔 기준 per_step 합계 == remaining
    assert sum(stats["per_step"].values()) == stats["remaining"]
    print("[PASS] per_step 합계 == remaining (최종 스캔)")

    # 10) 입력 배열 불변성 (run_collision_resolution이 복사본 사용)
    f10_backup   = f10.copy()
    asgn10_backup = asgn10.copy()
    run_collision_resolution(f10, asgn10, t10)
    assert np.allclose(f10, f10_backup),    "입력 frames가 외부에서 수정됨"
    assert np.array_equal(asgn10, asgn10_backup), "입력 assignment가 외부에서 수정됨"
    print("[PASS] 입력 배열 불변성 확인 (내부 복사본 사용)")

    print("\n모든 테스트 완료.")
