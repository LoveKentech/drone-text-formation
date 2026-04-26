"""
드론 텍스트 포메이션 시뮬레이션의 진입점.
LOVE 포메이션에서 KENTECH 포메이션으로 충돌 회피 경로를 실험한다.
"""

import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

from config import (
    DRONE_COUNTS,
    EXPERIMENT_END_TEXT,
    EXPERIMENT_INCLUDE_CBS,
    EXPERIMENT_START_TEXT,
    MAX_STEPS,
    MIN_SAFE_DIST,
    SIZES,
)
from src.coordinate import generate_coordinates
from src.experiment_record import (
    compute_evaluation_metrics,
    new_experiment_run_id,
    output_run_dir,
    save_trials_csv,
    trial_id,
    write_manifest,
)
from src.hungarian_collision import compute_assignment, greedy_timeline_then_resolve
from src.timeline import compute_timeline_greedy, detect_collisions
from src.cbs import cbs_assign
from src.visualize import animate, compare_animate


FORMATION_LABEL = f"{EXPERIMENT_START_TEXT}_to_{EXPERIMENT_END_TEXT}"


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


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
    diffs       = np.diff(frames, axis=0)
    step_dists  = np.linalg.norm(diffs, axis=2)
    drone_dists = step_dists.sum(axis=0)
    return float(drone_dists.sum()), float(drone_dists.max())
# ---------------------------------------------------------------------------
# 핵심 실험 함수
# ---------------------------------------------------------------------------


def run_experiment(
    n: int,
    size: str,
    algorithm: str,
    *,
    experiment_run_id: str,
    trial_index: int,
    min_safe_dist: float = MIN_SAFE_DIST,
    max_steps: int = MAX_STEPS,
    formation_label: str = FORMATION_LABEL,
) -> Dict:
    """
    LOVE → KENTECH: 출발·목표 좌표는 ``generate_coordinates``로 고정하고,
    헝가리안 배정 후 **greedy + collision resolve** 타임라인으로 지표를 계산한다.

    CBS 분기에서는 CBS 탐색 시간·성공 여부를 기록하되, 기본 실험은 config에서 꺼져 있다.
    """
    total_start = time.perf_counter()

    drones  = generate_coordinates(EXPERIMENT_START_TEXT, n, size)
    targets = generate_coordinates(EXPERIMENT_END_TEXT, n, size)

    t0 = time.perf_counter()
    _, assignment = compute_assignment(drones, targets)
    hungarian_time = time.perf_counter() - t0

    frames_raw = compute_timeline_greedy(
        drones, targets, assignment, max_steps=max_steps
    )
    collision_before_raw = _count_all_collisions(frames_raw, min_dist=min_safe_dist)

    stats_resolve: Optional[Dict] = None
    cbs_fallback = False
    cbs_success: Optional[bool] = None

    if algorithm == "hungarian":
        assign_time = hungarian_time
        frames, assignment, stats, _ = greedy_timeline_then_resolve(
            drones,
            targets,
            assignment,
            max_steps=max_steps,
            min_dist=min_safe_dist,
            frames_raw=frames_raw,
            smooth_visual=True,
        )
        collision_before = collision_before_raw
        collision_after = stats["remaining"]
        stats_resolve = {
            "passes_used": stats.get("passes_used"),
            "total_detected": stats.get("total_detected"),
            "steps_with_collision": stats.get("steps_with_collision"),
            "remaining": stats.get("remaining"),
        }
    else:
        t0 = time.perf_counter()
        paths = cbs_assign(drones, targets, assignment, max_steps=max_steps)
        assign_time = time.perf_counter() - t0

        if paths is not None:
            cbs_success = True
        else:
            print(
                f"  [WARN] CBS 탐색 실패 (선형 궤적은 그대로 사용) "
                f"(n={n}, {formation_label}, size={size})"
            )
            cbs_fallback = True
            cbs_success = False

        frames, assignment, stats, _ = greedy_timeline_then_resolve(
            drones,
            targets,
            assignment,
            max_steps=max_steps,
            min_dist=min_safe_dist,
            frames_raw=frames_raw,
            smooth_visual=True,
        )
        collision_before = collision_before_raw
        collision_after = stats["remaining"]
        stats_resolve = {
            "passes_used": stats.get("passes_used"),
            "total_detected": stats.get("total_detected"),
            "steps_with_collision": stats.get("steps_with_collision"),
            "remaining": stats.get("remaining"),
        }

    total_dist, max_dist = _compute_dist_metrics(frames)
    pipeline_time = time.perf_counter() - total_start

    base = {
        "experiment_run_id": experiment_run_id,
        "trial_id": trial_id(experiment_run_id, algorithm, n, formation_label, size),
        "trial_index": trial_index,
        "algorithm": algorithm,
        "n": n,
        "text": formation_label,
        "size": size,
        "min_safe_dist": min_safe_dist,
        "max_steps": max_steps,
        "total_dist": round(total_dist, 6),
        "max_dist": round(max_dist, 6),
        "assign_time_sec": round(assign_time, 6),
        "pipeline_time_sec": round(pipeline_time, 6),
    }
    metrics = compute_evaluation_metrics(
        frames,
        min_safe_dist,
        n,
        total_dist,
        collision_before,
        collision_after,
        stats_resolve=stats_resolve,
        cbs_fallback=cbs_fallback,
        cbs_success=cbs_success,
    )
    base.update(metrics)
    return base


# ---------------------------------------------------------------------------
# 애니메이션 저장
# ---------------------------------------------------------------------------

def _save_animations(
    n: int = 200,
    size: str = "medium",
    experiment_run_id: str = "",
    include_cbs: bool = EXPERIMENT_INCLUDE_CBS,
) -> None:
    """LOVE→KENTECH 충돌회피 모핑 GIF (타겟 X 표시 없음)."""
    out_dir = output_run_dir(experiment_run_id) if experiment_run_id else "output"
    os.makedirs(out_dir, exist_ok=True)

    drones  = generate_coordinates(EXPERIMENT_START_TEXT, n, size)
    targets = generate_coordinates(EXPERIMENT_END_TEXT, n, size)
    _, assignment = compute_assignment(drones, targets)

    frames_raw = compute_timeline_greedy(
        drones, targets, assignment, max_steps=MAX_STEPS
    )
    frames_resolved, assignment, stats, _ = greedy_timeline_then_resolve(
        drones,
        targets,
        assignment,
        max_steps=MAX_STEPS,
        min_dist=MIN_SAFE_DIST,
        frames_raw=frames_raw,
        smooth_visual=True,
    )
    frames_resolved = _trim_frames(frames_resolved)
    print(
        f"[충돌회피] 대표 GIF 잔여={stats['remaining']}  "
        f"패스={stats['passes_used']}  감지누적={stats['total_detected']}"
    )

    base = f"{FORMATION_LABEL}_{n}_{size}"
    h_path = os.path.join(out_dir, f"collision_resolved_hungarian_{base}.gif")
    title_h = f"Collision-resolved | Hungarian assign | {FORMATION_LABEL} | n={n} | {size}"

    print(f"[GIF] {h_path}")
    animate(
        frames_resolved,
        targets,
        title=title_h,
        save_path=h_path,
        show_targets=False,
    )

    if include_cbs:
        t0 = time.perf_counter()
        paths = cbs_assign(drones, targets, assignment, max_steps=MAX_STEPS)
        cbs_elapsed = time.perf_counter() - t0
        if paths is None:
            print(
                f"[WARN] 대표 케이스 CBS 실패 ({cbs_elapsed:.2f}s) — 비교는 동일 충돌회피 프레임"
            )
        frames_c = frames_resolved

        c_path = os.path.join(out_dir, f"collision_resolved_cbs_view_{base}.gif")
        cmp_path = os.path.join(out_dir, f"compare_collision_resolved_{base}.gif")
        title_c = f"Collision-resolved | CBS checked | {FORMATION_LABEL} | n={n} | {size}"

        print(f"[GIF] {c_path}")
        animate(
            frames_c,
            targets,
            title=title_c,
            save_path=c_path,
            show_targets=False,
        )

        print(f"[GIF] {cmp_path}")
        compare_animate(
            frames_resolved,
            frames_c,
            targets,
            save_path=cmp_path,
            show_targets=False,
        )

    print(f"[GIF] 저장 완료 (include_cbs={include_cbs})")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    experiment_run_id = new_experiment_run_id()
    manifest_path = write_manifest(experiment_run_id)
    gif_dir = output_run_dir(experiment_run_id)
    os.makedirs(gif_dir, exist_ok=True)
    shutil.copy2(manifest_path, os.path.join(gif_dir, "manifest.json"))

    print("=" * 60)
    print("드론 텍스트 포메이션 실험 시작")
    print(f"포메이션: {FORMATION_LABEL} (충돌회피 이동)")
    print(f"CBS 실험: {'on' if EXPERIMENT_INCLUDE_CBS else 'off'}")
    print(f"experiment_run_id = {experiment_run_id}")
    print(f"매니페스트: {manifest_path}")
    print(f"GIF 출력 폴더: {gif_dir}")
    print("=" * 60)

    results: List[Dict] = []
    algorithms = (
        ["hungarian", "cbs"] if EXPERIMENT_INCLUDE_CBS else ["hungarian"]
    )
    total = len(algorithms) * len(DRONE_COUNTS) * len(SIZES)
    idx = 0

    for algorithm in algorithms:
        for n in DRONE_COUNTS:
            for size in SIZES:
                idx += 1
                print(
                    f"[{idx:>3}/{total}] {algorithm:10s} | "
                    f"n={n:>3} | {FORMATION_LABEL:16s} | {size}"
                )
                result = run_experiment(
                    n,
                    size,
                    algorithm,
                    experiment_run_id=experiment_run_id,
                    trial_index=idx,
                )
                results.append(result)
                print(
                    f"         trial_id={result['trial_id']}  "
                    f"총거리={result['total_dist']:.1f}  "
                    f"충돌이벤트 {result['collision_pair_events_before']}→"
                    f"{result['collision_pair_events_after']}  "
                    f"배정={result['assign_time_sec']:.3f}s  "
                    f"전체={result['pipeline_time_sec']:.2f}s"
                )

    trials_csv = save_trials_csv(results, experiment_run_id)
    print(f"\n[저장] {trials_csv}")
    print(f"[누적] results/history.csv (append)")

    print("\n[애니메이션] 대표 케이스 (n=200, medium) 생성 중 ...")
    _save_animations(n=200, size="medium", experiment_run_id=experiment_run_id)

    print("\n실험 완료.")
