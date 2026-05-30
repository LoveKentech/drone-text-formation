"""
드론 텍스트 포메이션 시뮬레이션의 진입점.
LOVE 포메이션에서 KENTECH 포메이션으로 충돌 회피 경로를 실험한다.
"""

import os
import shutil
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from config import (
    CBS_MAX_ITERATIONS,
    CBS_GRID_SCALE,
    CBS_TIMEOUT_SEC,
    DRONE_COUNTS,
    EXPERIMENT_END_TEXT,
    EXPERIMENT_GIF_CASES,
    EXPERIMENT_INCLUDE_CBS,
    EXPERIMENT_REPRESENTATIVE_N,
    EXPERIMENT_REPRESENTATIVE_ONLY,
    EXPERIMENT_REPRESENTATIVE_SIZE,
    EXPERIMENT_START_TEXT,
    EXPERIMENT_TRANSITIONS,
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


def _formation_label(start_text: str, end_text: str) -> str:
    return f"{start_text}_to_{end_text}"


def _trim_frames(frames: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """드론이 모두 정지한 이후 프레임을 잘라낸다 (애니메이션 길이 단축)."""
    if frames.shape[0] <= 2:
        return frames
    for t in range(frames.shape[0] - 1, 0, -1):
        if np.any(np.linalg.norm(frames[t] - frames[t - 1], axis=1) > tol):
            return frames[: t + 2]
    return frames[:2]


def _paths_to_frames(paths: Dict[int, list], n: int, max_steps: int) -> np.ndarray:
    """CBS PathsDict를 (max_steps, n, 2) 프레임으로 변환한다."""
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


def _to_cbs_grid_cells(points: np.ndarray, grid_scale: int) -> List[tuple[int, int]]:
    """연속 좌표를 CBS가 쓰는 정수 격자 셀로 변환한다."""
    return [
        (int(round(p[0] * grid_scale)), int(round(p[1] * grid_scale)))
        for p in points
    ]


def _duplicate_cell_stats(cells: List[tuple[int, int]]) -> Dict[str, int]:
    counts = Counter(cells)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return {
        "duplicate_cells": len(duplicate_counts),
        "duplicate_agents": sum(count - 1 for count in duplicate_counts),
        "max_cell_occupancy": max(counts.values(), default=0),
    }


def _cell_is_clear(
    cell: tuple[int, int],
    occupied: List[tuple[int, int]],
    occupied_set: set[tuple[int, int]],
    min_grid_dist_sq: float,
) -> bool:
    if cell in occupied_set:
        return False
    if min_grid_dist_sq <= 0:
        return True
    cx, cy = cell
    for ox, oy in occupied:
        dx = cx - ox
        dy = cy - oy
        if dx * dx + dy * dy < min_grid_dist_sq:
            return False
    return True


def _nearest_clear_cell(
    base_cell: tuple[int, int],
    occupied: List[tuple[int, int]],
    occupied_set: set[tuple[int, int]],
    min_grid_dist_sq: float,
    max_radius: int = 80,
) -> tuple[int, int]:
    if _cell_is_clear(base_cell, occupied, occupied_set, min_grid_dist_sq):
        return base_cell

    bx, by = base_cell
    best_cell: Optional[tuple[int, int]] = None
    best_key: Optional[tuple[int, int, int, int]] = None
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cand = (bx + dx, by + dy)
                if not _cell_is_clear(cand, occupied, occupied_set, min_grid_dist_sq):
                    continue
                key = (dx * dx + dy * dy, abs(dy), abs(dx), dx + dy)
                if best_key is None or key < best_key:
                    best_key = key
                    best_cell = cand
        if best_cell is not None:
            return best_cell

    raise ValueError(
        f"CBS-ready 좌표 보정 실패: {base_cell} 주변 radius={max_radius} 안에 빈 격자 없음"
    )


def _make_cbs_ready_points(
    points: np.ndarray,
    *,
    grid_scale: int,
    min_safe_dist: float,
    label: str,
) -> tuple[np.ndarray, Dict[str, object]]:
    """좌표를 CBS 격자 셀 중심으로 스냅하고 중복/초기거리 위반을 제거한다."""
    original_cells = _to_cbs_grid_cells(points, grid_scale)
    min_grid_dist_sq = (min_safe_dist * grid_scale) ** 2
    occupied: List[tuple[int, int]] = []
    occupied_set: set[tuple[int, int]] = set()
    chosen_cells: List[tuple[int, int]] = []
    reassigned = 0

    for cell in original_cells:
        chosen = _nearest_clear_cell(
            cell,
            occupied,
            occupied_set,
            min_grid_dist_sq,
        )
        if chosen != cell:
            reassigned += 1
        chosen_cells.append(chosen)
        occupied.append(chosen)
        occupied_set.add(chosen)

    adjusted = np.array(chosen_cells, dtype=float) / float(grid_scale)
    displacement = np.linalg.norm(adjusted - points, axis=1)
    before_stats = _duplicate_cell_stats(original_cells)
    after_stats = _duplicate_cell_stats(chosen_cells)
    before_close = _first_close_pair(points, min_safe_dist)
    after_close = _first_close_pair(adjusted, min_safe_dist)

    stats: Dict[str, object] = {
        f"{label}_cbs_ready_reassigned_cells": reassigned,
        f"{label}_cbs_ready_duplicate_cells_before": before_stats["duplicate_cells"],
        f"{label}_cbs_ready_duplicate_cells_after": after_stats["duplicate_cells"],
        f"{label}_cbs_ready_close_pair_before": "",
        f"{label}_cbs_ready_close_dist_before": "",
        f"{label}_cbs_ready_close_pair_after": "",
        f"{label}_cbs_ready_close_dist_after": "",
        f"{label}_cbs_ready_mean_adjustment": round(float(displacement.mean()), 6)
        if len(displacement)
        else 0.0,
        f"{label}_cbs_ready_max_adjustment": round(float(displacement.max()), 6)
        if len(displacement)
        else 0.0,
    }
    if before_close is not None:
        i, j, dist = before_close
        stats[f"{label}_cbs_ready_close_pair_before"] = f"{i}-{j}"
        stats[f"{label}_cbs_ready_close_dist_before"] = round(dist, 6)
    if after_close is not None:
        i, j, dist = after_close
        stats[f"{label}_cbs_ready_close_pair_after"] = f"{i}-{j}"
        stats[f"{label}_cbs_ready_close_dist_after"] = round(dist, 6)
    return adjusted, stats


def _prepare_cbs_ready_coordinates(
    drones: np.ndarray,
    targets: np.ndarray,
    *,
    grid_scale: int,
    min_safe_dist: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """비교 실험 전체가 같은 CBS-feasible 시작/목표 좌표를 쓰도록 보정한다."""
    drones_ready, start_stats = _make_cbs_ready_points(
        drones,
        grid_scale=grid_scale,
        min_safe_dist=min_safe_dist,
        label="start",
    )
    targets_ready, goal_stats = _make_cbs_ready_points(
        targets,
        grid_scale=grid_scale,
        min_safe_dist=min_safe_dist,
        label="goal",
    )
    stats: Dict[str, object] = {
        "cbs_ready_grid_scale": grid_scale,
        "cbs_ready_coordinate_adjustment": 1,
    }
    stats.update(start_stats)
    stats.update(goal_stats)
    return drones_ready, targets_ready, stats


def _first_close_pair(
    points: np.ndarray,
    min_dist: float,
) -> Optional[tuple[int, int, float]]:
    """t=0/final처럼 움직여서 해결할 수 없는 최소거리 위반 한 쌍을 찾는다."""
    if min_dist <= 0:
        return None
    limit_sq = min_dist * min_dist
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist_sq = float(np.dot(diff, diff))
            if dist_sq < limit_sq:
                return i, j, float(np.sqrt(dist_sq))
    return None


def _cbs_precheck(
    drones: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    *,
    grid_scale: int,
    min_safe_dist: float,
) -> tuple[bool, Dict[str, object]]:
    """격자 MAPF CBS가 풀 수 없는 입력이면 긴 탐색 전에 스킵한다."""
    assigned_targets = targets[assignment]
    start_stats = _duplicate_cell_stats(_to_cbs_grid_cells(drones, grid_scale))
    goal_stats = _duplicate_cell_stats(_to_cbs_grid_cells(assigned_targets, grid_scale))
    initial_close = _first_close_pair(drones, min_safe_dist)
    final_close = _first_close_pair(assigned_targets, min_safe_dist)

    reasons: List[str] = []
    if start_stats["duplicate_cells"] > 0:
        reasons.append(f"start_grid_duplicates={start_stats['duplicate_cells']}")
    if goal_stats["duplicate_cells"] > 0:
        reasons.append(f"goal_grid_duplicates={goal_stats['duplicate_cells']}")
    if initial_close is not None:
        i, j, dist = initial_close
        reasons.append(f"initial_min_dist={i}-{j}:{dist:.3f}")
    if final_close is not None:
        i, j, dist = final_close
        reasons.append(f"final_min_dist={i}-{j}:{dist:.3f}")

    stats: Dict[str, object] = {
        "cbs_skipped_by_precheck": 1 if reasons else 0,
        "cbs_skip_reason": "; ".join(reasons),
        "cbs_precheck_start_duplicate_cells": start_stats["duplicate_cells"],
        "cbs_precheck_start_duplicate_agents": start_stats["duplicate_agents"],
        "cbs_precheck_start_max_cell_occupancy": start_stats["max_cell_occupancy"],
        "cbs_precheck_goal_duplicate_cells": goal_stats["duplicate_cells"],
        "cbs_precheck_goal_duplicate_agents": goal_stats["duplicate_agents"],
        "cbs_precheck_goal_max_cell_occupancy": goal_stats["max_cell_occupancy"],
        "cbs_precheck_initial_close_pair": "",
        "cbs_precheck_initial_close_dist": "",
        "cbs_precheck_final_close_pair": "",
        "cbs_precheck_final_close_dist": "",
    }
    if initial_close is not None:
        i, j, dist = initial_close
        stats["cbs_precheck_initial_close_pair"] = f"{i}-{j}"
        stats["cbs_precheck_initial_close_dist"] = round(dist, 6)
    if final_close is not None:
        i, j, dist = final_close
        stats["cbs_precheck_final_close_pair"] = f"{i}-{j}"
        stats["cbs_precheck_final_close_dist"] = round(dist, 6)

    return not reasons, stats


def _count_all_collisions(frames: np.ndarray, min_dist: float = MIN_SAFE_DIST) -> int:
    """타임라인 전체에서 충돌 쌍 수를 합산한다."""
    return sum(
        len(detect_collisions(frames, t, min_dist)) for t in range(frames.shape[0])
    )


def _count_cbs_grid_conflicts(frames: np.ndarray, grid_scale: int) -> Dict[str, object]:
    """CBS 격자 MAPF 기준 vertex/edge 충돌 이벤트를 센다."""
    cells = np.rint(np.asarray(frames, dtype=float) * grid_scale).astype(int)
    T, n, _ = cells.shape
    vertex_conflicts = 0
    edge_conflicts = 0
    conflict_timesteps: set[int] = set()
    first_type = ""
    first_t: object = ""
    first_agents = ""

    for t in range(T):
        pos_agents: Dict[tuple[int, int], List[int]] = {}
        for aid in range(n):
            pos = (int(cells[t, aid, 0]), int(cells[t, aid, 1]))
            pos_agents.setdefault(pos, []).append(aid)

        for agents in pos_agents.values():
            if len(agents) < 2:
                continue
            vertex_conflicts += len(agents) * (len(agents) - 1) // 2
            conflict_timesteps.add(t)
            if first_type == "":
                first_type = "vertex"
                first_t = t
                first_agents = f"{agents[0]}-{agents[1]}"

        if t + 1 >= T:
            continue

        edge_agents: Dict[tuple[tuple[int, int], tuple[int, int]], List[int]] = {}
        for aid in range(n):
            src = (int(cells[t, aid, 0]), int(cells[t, aid, 1]))
            dst = (int(cells[t + 1, aid, 0]), int(cells[t + 1, aid, 1]))
            if src == dst:
                continue
            edge_agents.setdefault((src, dst), []).append(aid)

        seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for edge, agents in edge_agents.items():
            if edge in seen_edges:
                continue
            src, dst = edge
            reverse = (dst, src)
            reverse_agents = edge_agents.get(reverse)
            if not reverse_agents:
                continue
            edge_conflicts += len(agents) * len(reverse_agents)
            conflict_timesteps.add(t)
            seen_edges.add(edge)
            seen_edges.add(reverse)
            if first_type == "":
                first_type = "edge"
                first_t = t
                first_agents = f"{agents[0]}-{reverse_agents[0]}"

    total = vertex_conflicts + edge_conflicts
    return {
        "vertex_conflicts": vertex_conflicts,
        "edge_conflicts": edge_conflicts,
        "total_conflicts": total,
        "timesteps_with_conflict": len(conflict_timesteps),
        "first_conflict_type": first_type,
        "first_conflict_t": first_t,
        "first_conflict_agents": first_agents,
        "conflict_free": 1 if total == 0 else 0,
    }


def _cbs_grid_stats_for_frames(
    frames: np.ndarray,
    *,
    suffix: str,
    grid_scale: int,
) -> Dict[str, object]:
    stats = _count_cbs_grid_conflicts(frames, grid_scale)
    return {
        f"cbs_grid_vertex_conflicts_{suffix}": stats["vertex_conflicts"],
        f"cbs_grid_edge_conflicts_{suffix}": stats["edge_conflicts"],
        f"cbs_grid_conflicts_{suffix}": stats["total_conflicts"],
        f"cbs_grid_timesteps_with_conflict_{suffix}": stats["timesteps_with_conflict"],
        f"cbs_grid_first_conflict_type_{suffix}": stats["first_conflict_type"],
        f"cbs_grid_first_conflict_t_{suffix}": stats["first_conflict_t"],
        f"cbs_grid_first_conflict_agents_{suffix}": stats["first_conflict_agents"],
        f"cbs_grid_conflict_free_{suffix}": stats["conflict_free"],
    }


def _blank_cbs_grid_after_stats() -> Dict[str, object]:
    return {
        "cbs_grid_vertex_conflicts_after": "",
        "cbs_grid_edge_conflicts_after": "",
        "cbs_grid_conflicts_after": "",
        "cbs_grid_timesteps_with_conflict_after": "",
        "cbs_grid_first_conflict_type_after": "",
        "cbs_grid_first_conflict_t_after": "",
        "cbs_grid_first_conflict_agents_after": "",
        "cbs_grid_conflict_free_after": "",
    }


def _compute_dist_metrics(frames: np.ndarray):
    """(total_dist, max_dist) — 전 드론 이동 거리 합, 단일 드론 최대 이동 거리."""
    diffs       = np.diff(frames, axis=0)
    step_dists  = np.linalg.norm(diffs, axis=2)
    drone_dists = step_dists.sum(axis=0)
    return float(drone_dists.sum()), float(drone_dists.max())


def _fmt_num(value: object, digits: int = 1) -> str:
    if value == "" or value is None:
        return "NA"
    try:
        number = float(value)
        if np.isnan(number):
            return "NA"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cbs_failed_metrics(
    *,
    collision_before: int,
    cbs_failure_reason: str,
    extra_stats: Dict[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "timesteps": "",
        "active_timesteps": "",
        "trajectory_type": "",
        "collision_avoidance_type": "cbs_failed",
        "timesteps_with_collision": "",
        "max_collision_pairs_at_one_timestep": "",
        "mean_dist_per_drone": "",
        "collision_pair_events_before": collision_before,
        "collision_pair_events_after": "",
        "collision_pair_events_reduced": "",
        "collision_reduction_ratio": "",
        "collision_residual_ratio_vs_before": "",
        "is_collision_free": "",
        "cbs_planner_success": 0,
        "cbs_used_hungarian_fallback": 0,
        "cbs_failure_reason": cbs_failure_reason,
        "resolve_passes_used": "",
        "resolve_total_detected": "",
        "resolve_steps_with_collision_first_pass": "",
        "resolve_remaining_reported": "",
    }
    row.update(extra_stats)
    return row
# ---------------------------------------------------------------------------
# 핵심 실험 함수
# ---------------------------------------------------------------------------


def run_experiment(
    n: int,
    size: str,
    algorithm: str,
    *,
    start_text: str = EXPERIMENT_START_TEXT,
    end_text: str = EXPERIMENT_END_TEXT,
    experiment_run_id: str,
    trial_index: int,
    min_safe_dist: float = MIN_SAFE_DIST,
    max_steps: int = MAX_STEPS,
    formation_label: Optional[str] = None,
) -> Dict:
    """
    start_text → end_text: 출발·목표 좌표는 ``generate_coordinates``로 고정하고,
    CBS-ready 좌표 보정 후 알고리즘별 지표를 계산한다.

    CBS 분기에서는 실패 시 Hungarian 폴백을 쓰지 않고 실패 row로 기록한다.
    """
    if formation_label is None:
        formation_label = _formation_label(start_text, end_text)
    total_start = time.perf_counter()

    raw_drones  = generate_coordinates(start_text, n, size)
    raw_targets = generate_coordinates(end_text, n, size)
    drones, targets, coordinate_stats = _prepare_cbs_ready_coordinates(
        raw_drones,
        raw_targets,
        grid_scale=CBS_GRID_SCALE,
        min_safe_dist=min_safe_dist,
    )

    t0 = time.perf_counter()
    _, assignment = compute_assignment(drones, targets)
    hungarian_time = time.perf_counter() - t0

    frames_raw = compute_timeline_greedy(
        drones, targets, assignment, max_steps=max_steps
    )
    collision_before_raw = _count_all_collisions(frames_raw, min_dist=min_safe_dist)
    cbs_grid_before_stats = _cbs_grid_stats_for_frames(
        frames_raw,
        suffix="before",
        grid_scale=CBS_GRID_SCALE,
    )

    stats_resolve: Optional[Dict] = None
    extra_stats: Dict[str, object] = {**coordinate_stats, **cbs_grid_before_stats}
    cbs_fallback = False
    cbs_success: Optional[bool] = None
    cbs_search_stats: Dict[str, object] = {}

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
        grid_after_stats = _cbs_grid_stats_for_frames(
            frames,
            suffix="after",
            grid_scale=CBS_GRID_SCALE,
        )
        extra_stats.update(grid_after_stats)
        extra_stats.update({
            "cbs_grid_conflict_free": grid_after_stats["cbs_grid_conflict_free_after"],
            "cbs_grid_first_conflict_t": grid_after_stats[
                "cbs_grid_first_conflict_t_after"
            ],
        })
    else:
        precheck_ok, precheck_stats = _cbs_precheck(
            drones,
            targets,
            assignment,
            grid_scale=CBS_GRID_SCALE,
            min_safe_dist=min_safe_dist,
        )
        extra_stats.update(precheck_stats)
        if precheck_ok:
            t0 = time.perf_counter()
            paths, cbs_search_stats = cbs_assign(
                drones,
                targets,
                assignment,
                max_steps=max_steps,
                grid_scale=CBS_GRID_SCALE,
                strict_cbs=False,
                max_iterations=CBS_MAX_ITERATIONS,
                timeout_sec=CBS_TIMEOUT_SEC,
                move_model="8n",
                return_stats=True,
            )
            assign_time = time.perf_counter() - t0
            extra_stats.update(cbs_search_stats)
        else:
            paths = None
            assign_time = 0.0
            extra_stats["cbs_search_stop_reason"] = "precheck_failed"
            print(
                f"  [WARN] CBS 입력 불가: {precheck_stats['cbs_skip_reason']}"
            )

        if paths is not None:
            cbs_success = True
            frames = _paths_to_frames(paths, n, max_steps)
            collision_before = collision_before_raw
            collision_after = _count_all_collisions(frames, min_dist=min_safe_dist)
            grid_after_stats = _cbs_grid_stats_for_frames(
                frames,
                suffix="after",
                grid_scale=CBS_GRID_SCALE,
            )
            extra_stats.update(grid_after_stats)
            extra_stats.update({
                "cbs_grid_conflict_free": grid_after_stats["cbs_grid_conflict_free_after"],
                "cbs_grid_first_conflict_t": grid_after_stats[
                    "cbs_grid_first_conflict_t_after"
                ],
            })
        else:
            if precheck_ok:
                stop_reason = cbs_search_stats.get("cbs_search_stop_reason", "")
                expanded_nodes = cbs_search_stats.get("cbs_search_expanded_nodes", "")
                max_open = cbs_search_stats.get("cbs_search_max_open_size", "")
                print(
                    f"  [WARN] CBS 탐색 실패 "
                    f"(n={n}, {formation_label}, size={size}, "
                    f"stop={stop_reason}, expanded={expanded_nodes}, "
                    f"max_open={max_open}, max_iterations={CBS_MAX_ITERATIONS}, "
                    f"timeout={CBS_TIMEOUT_SEC}s)"
                )
            cbs_success = False
            extra_stats.update({
                "cbs_grid_conflict_free": "",
                "cbs_grid_first_conflict_t": "",
            })
            extra_stats.update(_blank_cbs_grid_after_stats())
            pipeline_time = time.perf_counter() - total_start
            reason = (
                f"precheck_failed: {precheck_stats['cbs_skip_reason']}"
                if not precheck_ok
                else f"planner_failed:{cbs_search_stats.get('cbs_search_stop_reason', 'unknown')}"
                f":max_iterations:{CBS_MAX_ITERATIONS}:timeout:{CBS_TIMEOUT_SEC}s"
            )
            base = {
                "experiment_run_id": experiment_run_id,
                "trial_id": trial_id(experiment_run_id, algorithm, n, formation_label, size),
                "trial_index": trial_index,
                "algorithm": algorithm,
                "n": n,
                "start_text": start_text,
                "end_text": end_text,
                "text": formation_label,
                "size": size,
                "min_safe_dist": min_safe_dist,
                "max_steps": max_steps,
                "total_dist": "",
                "max_dist": "",
                "assign_time_sec": round(assign_time, 6),
                "pipeline_time_sec": round(pipeline_time, 6),
            }
            base.update(
                _cbs_failed_metrics(
                    collision_before=collision_before_raw,
                    cbs_failure_reason=reason,
                    extra_stats=extra_stats,
                )
            )
            return base

    total_dist, max_dist = _compute_dist_metrics(frames)
    pipeline_time = time.perf_counter() - total_start

    base = {
        "experiment_run_id": experiment_run_id,
        "trial_id": trial_id(experiment_run_id, algorithm, n, formation_label, size),
        "trial_index": trial_index,
        "algorithm": algorithm,
        "n": n,
        "start_text": start_text,
        "end_text": end_text,
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
        extra_stats=extra_stats,
    )
    base.update(metrics)
    return base


# ---------------------------------------------------------------------------
# 애니메이션 저장
# ---------------------------------------------------------------------------

def _save_animations(
    n: int = 200,
    size: str = "medium",
    start_text: str = EXPERIMENT_START_TEXT,
    end_text: str = EXPERIMENT_END_TEXT,
    experiment_run_id: str = "",
    include_cbs: bool = EXPERIMENT_INCLUDE_CBS,
) -> None:
    """대표 충돌회피 모핑 GIF (타겟 X 표시 없음)."""
    formation_label = _formation_label(start_text, end_text)
    out_dir = output_run_dir(experiment_run_id) if experiment_run_id else "output"
    os.makedirs(out_dir, exist_ok=True)

    raw_drones  = generate_coordinates(start_text, n, size)
    raw_targets = generate_coordinates(end_text, n, size)
    drones, targets, _ = _prepare_cbs_ready_coordinates(
        raw_drones,
        raw_targets,
        grid_scale=CBS_GRID_SCALE,
        min_safe_dist=MIN_SAFE_DIST,
    )
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

    base = f"{formation_label}_{n}_{size}"
    h_path = os.path.join(out_dir, f"collision_resolved_hungarian_{base}.gif")
    title_h = f"Collision-resolved | Hungarian assign | {formation_label} | n={n} | {size}"

    print(f"[GIF] {h_path}")
    animate(
        frames_resolved,
        targets,
        title=title_h,
        save_path=h_path,
        show_targets=False,
    )

    if include_cbs:
        precheck_ok, precheck_stats = _cbs_precheck(
            drones,
            targets,
            assignment,
            grid_scale=CBS_GRID_SCALE,
            min_safe_dist=MIN_SAFE_DIST,
        )
        if not precheck_ok:
            print(
                f"[WARN] 대표 케이스 CBS 입력 불가: "
                f"{precheck_stats['cbs_skip_reason']} — CBS GIF 생략"
            )
        else:
            t0 = time.perf_counter()
            paths = cbs_assign(
                drones,
                targets,
                assignment,
                max_steps=MAX_STEPS,
                grid_scale=CBS_GRID_SCALE,
                strict_cbs=False,
                max_iterations=CBS_MAX_ITERATIONS,
                timeout_sec=CBS_TIMEOUT_SEC,
                move_model="8n",
            )
            cbs_elapsed = time.perf_counter() - t0
            if paths is None:
                print(
                    f"[WARN] 대표 케이스 CBS 실패 ({cbs_elapsed:.2f}s, "
                    f"max_iterations={CBS_MAX_ITERATIONS}, timeout={CBS_TIMEOUT_SEC}s) "
                    f"— CBS GIF 생략"
                )
            else:
                frames_c = _trim_frames(_paths_to_frames(paths, n, MAX_STEPS))
                c_path = os.path.join(out_dir, f"collision_resolved_cbs_view_{base}.gif")
                cmp_path = os.path.join(out_dir, f"compare_collision_resolved_{base}.gif")
                title_c = f"Collision-resolved | CBS checked | {formation_label} | n={n} | {size}"

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
    if EXPERIMENT_REPRESENTATIVE_ONLY:
        print(f"포메이션: {FORMATION_LABEL} (대표 케이스)")
    else:
        print(f"포메이션 전환 수: {len(EXPERIMENT_TRANSITIONS)}")
    print(f"CBS 실험: {'on' if EXPERIMENT_INCLUDE_CBS else 'off'}")
    print(f"experiment_run_id = {experiment_run_id}")
    print(f"매니페스트: {manifest_path}")
    print(f"GIF 출력 폴더: {gif_dir}")
    print("=" * 60)

    results: List[Dict] = []
    algorithms = (
        ["cbs", "hungarian"] if EXPERIMENT_INCLUDE_CBS else ["hungarian"]
    )
    if EXPERIMENT_REPRESENTATIVE_ONLY:
        trial_cases = [
            (
                EXPERIMENT_START_TEXT,
                EXPERIMENT_END_TEXT,
                EXPERIMENT_REPRESENTATIVE_N,
                EXPERIMENT_REPRESENTATIVE_SIZE,
            )
        ]
    else:
        trial_cases = [
            (start_text, end_text, n, size)
            for start_text, end_text in EXPERIMENT_TRANSITIONS
            for n in DRONE_COUNTS
            for size in SIZES
        ]
    total = len(algorithms) * len(trial_cases)
    idx = 0

    for algorithm in algorithms:
        for start_text, end_text, n, size in trial_cases:
            formation_label = _formation_label(start_text, end_text)
            idx += 1
            print(
                f"[{idx:>3}/{total}] {algorithm:10s} | "
                f"n={n:>3} | {formation_label:18s} | {size}"
            )
            result = run_experiment(
                n,
                size,
                algorithm,
                start_text=start_text,
                end_text=end_text,
                formation_label=formation_label,
                experiment_run_id=experiment_run_id,
                trial_index=idx,
            )
            results.append(result)
            cbs_status = ""
            if result["algorithm"] == "cbs":
                if result.get("cbs_planner_success") == 1:
                    cbs_status = "  CBS=SUCCESS"
                elif result.get("cbs_planner_success") == 0:
                    cbs_status = "  CBS=FAILED"
            print(
                f"         trial_id={result['trial_id']}  "
                f"총거리={_fmt_num(result['total_dist'], 1)}  "
                f"거리충돌={_fmt_num(result.get('collision_pair_events_before'), 0)}→"
                f"{_fmt_num(result.get('collision_pair_events_after'), 0)}  "
                f"격자충돌={_fmt_num(result.get('cbs_grid_conflicts_before'), 0)}→"
                f"{_fmt_num(result.get('cbs_grid_conflicts_after'), 0)}  "
                f"배정={_fmt_num(result['assign_time_sec'], 3)}s  "
                f"전체={_fmt_num(result['pipeline_time_sec'], 2)}s"
                f"{cbs_status}"
            )

    trials_csv = save_trials_csv(results, experiment_run_id)
    print(f"\n[저장] {trials_csv}")
    print(f"[누적] results/history.csv (append)")

    if EXPERIMENT_GIF_CASES:
        print(f"\n[애니메이션] 대표 GIF {len(EXPERIMENT_GIF_CASES)}개 생성 중 ...")
    for start_text, end_text, n, size in EXPERIMENT_GIF_CASES:
        print(
            f"[애니메이션] {_formation_label(start_text, end_text)} "
            f"(n={n}, {size})"
        )
        _save_animations(
            n=n,
            size=size,
            start_text=start_text,
            end_text=end_text,
            experiment_run_id=experiment_run_id,
        )

    print("\n실험 완료.")
