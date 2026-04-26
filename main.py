"""
드론 텍스트 포메이션 시뮬레이션의 진입점.
config.py에 정의된 실험 조건을 순회하며 전체 파이프라인을 실행한다.
"""

import hashlib
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DRONE_COUNTS, STRINGS, SIZES, MAX_STEPS, MIN_SAFE_DIST
from src.coordinate import generate_coordinates
from src.hungarian_collision import (
    greedy_timeline_then_resolve,
    hungarian_assignment,
)
from src.timeline import compute_timeline_hungarian, detect_collisions
from src.cbs import cbs_assign, detect_conflict
from src.visualize import animate, compare_animate


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _paths_to_frames(paths: Dict[int, list], n: int, max_steps: int) -> np.ndarray:
    """CBS PathsDict → ndarray (max_steps, n, 2)."""
    frames = np.zeros((max_steps, n, 2), dtype=float)
    for i in range(n):
        path = paths.get(i, [])
        length = min(len(path), max_steps)
        for t in range(length):
            frames[t, i] = path[t]
        if path:
            last = path[min(len(path) - 1, max_steps - 1)]
            for t in range(length, max_steps):
                frames[t, i] = last
    return frames


def _trim_frames(frames: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """드론이 모두 정지한 이후 프레임을 잘라낸다 (애니메이션 길이 단축)."""
    if frames.shape[0] <= 2:
        return frames
    for t in range(frames.shape[0] - 1, 0, -1):
        if np.any(np.linalg.norm(frames[t] - frames[t - 1], axis=1) > tol):
            return frames[: t + 2]
    return frames[:2]


def _count_all_collisions(frames: np.ndarray, min_dist: float = MIN_SAFE_DIST) -> int:
    """타임라인 전체에서 충돌 쌍 수를 합산한다."""
    return sum(
        len(detect_collisions(frames, t, min_dist)) for t in range(frames.shape[0])
    )


def _compute_dist_metrics(frames: np.ndarray):
    """(total_dist, max_dist) — 전 드론 이동 거리 합, 단일 드론 최대 이동 거리."""
    diffs       = np.diff(frames, axis=0)           # (T-1, n, 2)
    step_dists  = np.linalg.norm(diffs, axis=2)     # (T-1, n)
    drone_dists = step_dists.sum(axis=0)            # (n,)
    return float(drone_dists.sum()), float(drone_dists.max())


def _all_reached_goal(
    frames: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """
    마지막 프레임에서 모든 드론이 할당 목표에 도달했는지 검사한다.

    CBS는 목표를 정수 격자로 반올림해 계획하므로, 도달 판정도 동일한 정수 목표 기준으로
    수행해 정합성을 맞춘다.
    """
    n = frames.shape[1]
    final_pos = frames[-1]
    goal_pos = np.stack(
        [
            np.array(
                [
                    float(int(round(targets[assignment[i]][0]))),
                    float(int(round(targets[assignment[i]][1]))),
                ]
            )
            for i in range(n)
        ],
        axis=0,
    )
    return bool(np.all(np.linalg.norm(final_pos - goal_pos, axis=1) <= tol))


def _make_drones(targets: np.ndarray, n: int, text: str, size: str) -> np.ndarray:
    """실험 조건별 고정 랜덤 시드로 드론 초기 위치를 생성한다."""
    # hash()는 프로세스마다 달라지므로 재현 가능한 시드로 고정한다.
    digest = hashlib.sha256(f"{n}|{text}|{size}".encode()).digest()
    seed   = int.from_bytes(digest[:4], "little") % (2**31)
    rng    = np.random.default_rng(seed)
    span   = targets.max() + 20.0
    return rng.uniform(0, span, (n, 2))


# ---------------------------------------------------------------------------
# 핵심 실험 함수
# ---------------------------------------------------------------------------

def run_experiment(text: str, n: int, size: str, algorithm: str) -> Dict:
    """
    단일 실험 조건에 대해 전체 파이프라인을 실행하고 성능 지표를 반환한다.

    Parameters
    ----------
    text      : 렌더링할 문자열
    n         : 드론 수
    size      : 이미지 배율 ("small" / "medium" / "large")
    algorithm : "hungarian" | "cbs"

    Returns
    -------
    dict — 성능 지표
        algorithm, n, text, size,
        total_dist, max_dist,
        assign_time, collision_before, collision_after, total_time
    """
    total_start = time.perf_counter()

    # ── 목표·초기 위치 ────────────────────────────────────────────────────
    targets = generate_coordinates(text, n, size)
    drones  = _make_drones(targets, n, text, size)

    # ── 공통: Hungarian 배정 ──────────────────────────────────────────────
    t0 = time.perf_counter()
    _, assignment = hungarian_assignment(drones, targets)
    hungarian_time = time.perf_counter() - t0

    # ── 알고리즘별 경로 계획 ──────────────────────────────────────────────
    # collision_before = greedy 타임라인(해결 전)의 충돌 수 (두 알고리즘 공통 기준)
    frames_raw = compute_timeline_hungarian(drones, targets, assignment)
    collision_before = _count_all_collisions(frames_raw)

    if algorithm == "hungarian":
        assign_time = hungarian_time
        frames, assignment, stats, _ = greedy_timeline_then_resolve(
            drones, targets, assignment, frames_raw=frames_raw
        )
        collision_after = stats["remaining"]

    else:  # cbs
        t0 = time.perf_counter()
        paths = cbs_assign(drones, targets, assignment, max_steps=MAX_STEPS)
        assign_time = time.perf_counter() - t0

        if paths is not None:
            cbs_horizon = max((len(p) for p in paths.values()), default=MAX_STEPS)
            frames = _paths_to_frames(paths, n, cbs_horizon)
        else:
            print(
                f"  [WARN] CBS 탐색 실패 — Hungarian+resolve 폴백 "
                f"(n={n}, text={text}, size={size})"
            )
            frames, assignment, _, _ = greedy_timeline_then_resolve(
                drones, targets, assignment, frames_raw=frames_raw
            )

        collision_after = _count_all_collisions(frames)

    # ── 거리 지표 ─────────────────────────────────────────────────────────
    total_dist, max_dist = _compute_dist_metrics(frames)
    total_time = time.perf_counter() - total_start

    return {
        "algorithm":        algorithm,
        "n":                n,
        "text":             text,
        "size":             size,
        "total_dist":       round(total_dist, 3),
        "max_dist":         round(max_dist, 3),
        "assign_time":      round(assign_time, 4),
        "collision_before": collision_before,
        "collision_after":  collision_after,
        "total_time":       round(total_time, 4),
    }


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def save_results(
    results: List[Dict],
    filename: str = "results/experiment.csv",
) -> None:
    """실험 결과를 pandas DataFrame으로 변환 후 CSV에 저장한다."""
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    pd.DataFrame(results).to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[저장] {filename}")


# ---------------------------------------------------------------------------
# 애니메이션 저장
# ---------------------------------------------------------------------------

def _save_animations(text: str = "KENTECH", n: int = 200, size: str = "medium") -> None:
    """대표 케이스 애니메이션 3종을 output/에 저장하고 충돌 통계를 출력한다."""
    os.makedirs("output", exist_ok=True)

    targets    = generate_coordinates(text, n, size)
    # drones     = _make_drones(targets, n, text, size)
    drones    = generate_coordinates("LOVE", n, size)

    _, assignment = hungarian_assignment(drones, targets)

    # Hungarian 프레임
    frames_h_raw = compute_timeline_hungarian(drones, targets, assignment)
    frames_h, _, _, _ = greedy_timeline_then_resolve(
        drones, targets, assignment, frames_raw=frames_h_raw
    )
    frames_h = _trim_frames(frames_h)
    collision_before = _count_all_collisions(frames_h_raw)
    h_collision_after = _count_all_collisions(frames_h)

    # CBS 프레임 + 상태 판정
    paths = cbs_assign(drones, targets, assignment, max_steps=MAX_STEPS)
    if paths is not None:
        cbs_horizon = max((len(p) for p in paths.values()), default=MAX_STEPS)
        frames_c_raw = _paths_to_frames(paths, n, cbs_horizon)
        reached = _all_reached_goal(frames_c_raw, targets, assignment)
        has_mapf_conflict = detect_conflict(paths) is not None

        if reached and not has_mapf_conflict:
            cbs_status = "SUCCESS"
            frames_c = _trim_frames(frames_c_raw)
        elif reached and has_mapf_conflict:
            cbs_status = "PARTIAL_CONFLICT"
            frames_c = frames_c_raw
        else:
            cbs_status = "PARTIAL_UNREACHED"
            frames_c = frames_c_raw
    else:
        print(f"[WARN] 대표 케이스 CBS 실패 — Hungarian 결과로 대체")
        cbs_status = "FALLBACK_HUNGARIAN"
        frames_c = frames_h
    c_collision_after = _count_all_collisions(frames_c)

    print(
        f"[충돌] Hungarian {collision_before}->{h_collision_after} | "
        f"CBS {collision_before}->{c_collision_after} ({cbs_status})"
    )

    h_path = f"output/hungarian_{text}_{n}_{size}.gif"
    c_path = f"output/cbs_{text}_{n}_{size}_{cbs_status}.gif"
    cmp_path = f"output/compare_{text}_{n}_{size}_{cbs_status}.gif"

    print(f"[GIF] {h_path}")
    animate(frames_h, targets,
            title=f"Hungarian | {text} | n={n} | {size}",
            save_path=h_path)

    print(f"[GIF] {c_path}")
    animate(frames_c, targets,
            title=f"CBS | {text} | n={n} | {size} | {cbs_status}",
            save_path=c_path)

    print(f"[GIF] {cmp_path}")
    compare_animate(frames_h, frames_c, targets, save_path=cmp_path)

    print(f"[GIF] 저장 완료 (CBS 상태: {cbs_status})")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # print("=" * 60)
    # print("드론 텍스트 포메이션 실험 시작")
    # print("=" * 60)

    # results: List[Dict] = []
    # algorithms = ["hungarian", "cbs"]
    # total = len(algorithms) * len(DRONE_COUNTS) * len(STRINGS) * len(SIZES)
    # idx = 0

    # for algorithm in algorithms:
    #     for n in DRONE_COUNTS:
    #         for text in STRINGS:
    #             for size in SIZES:
    #                 idx += 1
    #                 print(
    #                     f"[{idx:>3}/{total}] {algorithm:10s} | "
    #                     f"n={n:>3} | {text:>7} | {size}"
    #                 )
    #                 result = run_experiment(text, n, size, algorithm)
    #                 results.append(result)
    #                 print(
    #                     f"         총거리={result['total_dist']:.1f}  "
    #                     f"충돌 {result['collision_before']}→{result['collision_after']}  "
    #                     f"배정={result['assign_time']:.3f}s  "
    #                     f"전체={result['total_time']:.2f}s"
    #                 )

    # save_results(results, "results/experiment.csv")

    print("\n[애니메이션] 대표 케이스 (n=200, KENTECH, medium) 생성 중 ...")
    _save_animations(text="KENTECH", n=200, size="medium")

    print("\n실험 완료.")
