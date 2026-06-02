"""
실험 실행 단위(run) ID, 매니페스트, CSV 행에 넣을 평가 지표 보조.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .timeline import detect_collisions


def new_experiment_run_id(results_dir: str = "results") -> str:
    """
    배치(run) 식별자: 로컬 시각 ``MMDDHHMM`` (예: 04241336).

    ``results/runs/{id}`` 가 이미 있으면 ``04241336_2``, ``04241336_3`` … 로만든다.
    """
    base = datetime.now().strftime("%m%d%H%M")
    candidate = base
    n = 2
    while os.path.isdir(results_run_dir(candidate, base=results_dir)):
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _slug_condition_text(text: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in text)
    s = "_".join(p for p in s.split("_") if p)
    return s or "text"


def trial_id(
    run_id: str,
    algorithm: str,
    n: int,
    text: str,
    size: str,
) -> str:
    """한 건의 실험: ``{run_id}_{algorithm}_{n}_{text}_{size}``."""
    return f"{run_id}_{algorithm}_{n}_{_slug_condition_text(text)}_{size}"


def results_run_dir(run_id: str, base: str = "results") -> str:
    return os.path.join(base, "runs", run_id)


def output_run_dir(run_id: str, base: str = "output") -> str:
    return os.path.join(base, "runs", run_id)


def config_snapshot() -> Dict[str, Any]:
    import config as cfg

    keys = [
        "MIN_SAFE_DIST",
        "MAX_STEPS",
        "COLLISION_RESOLUTION_MAX_PASSES",
        "COLLISION_RESOLUTION_TIMEOUT_SEC",
        "COLLISION_RESOLUTION_STAGNATION_PASSES",
        "DRONE_COUNTS",
        "STRINGS",
        "SIZES",
        "EXPERIMENT_START_TEXT",
        "EXPERIMENT_END_TEXT",
        "EXPERIMENT_TRANSITIONS",
        "EXPERIMENT_GIF_CASES",
        "EXPERIMENT_INCLUDE_CBS",
        "EXPERIMENT_REPRESENTATIVE_ONLY",
        "EXPERIMENT_REPRESENTATIVE_N",
        "EXPERIMENT_REPRESENTATIVE_SIZE",
        "CBS_GRID_SCALE",
        "CBS_MAX_ITERATIONS",
        "CBS_TIMEOUT_SEC",
        "IMAGE_SIZE",
        "CONTOUR_METHODS",
        "DEFAULT_METHOD",
        "DETOUR_TIE_EPS",
        "TRAJECTORY_SMOOTH_BLEND",
        "TRAJECTORY_SMOOTH_OUTER_ITERS",
        "TRAJECTORY_SEPARATION_PASSES_PER_ITER",
        "MORPH_DRONE_SCATTER_S",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            if isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
    return out


def write_manifest(
    run_id: str,
    results_dir: str = "results",
    notes: str = "",
) -> str:
    """results/runs/{run_id}/manifest.json 기록. 저장 경로 반환."""
    root = results_run_dir(run_id, base=results_dir)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "manifest.json")
    payload = {
        "experiment_run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_at_local": datetime.now().astimezone().isoformat(),
        "notes": notes,
        "config": config_snapshot(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def count_active_timesteps(frames: np.ndarray, tol: float = 1e-6) -> int:
    """마지막 의미 있는 이동이 있는 시각까지의 길이(1-based 스텝 수)."""
    if frames.shape[0] <= 1:
        return int(frames.shape[0])
    for t in range(frames.shape[0] - 1, 0, -1):
        if np.any(np.linalg.norm(frames[t] - frames[t - 1], axis=1) > tol):
            return t + 1
    return 1


def compute_evaluation_metrics(
    frames: np.ndarray,
    min_dist: float,
    n: int,
    total_dist: float,
    collision_before: int,
    collision_after: int,
    stats_resolve: Optional[Dict[str, Any]] = None,
    cbs_fallback: bool = False,
    cbs_success: Optional[bool] = None,
    trajectory_type: str = "",
    collision_avoidance_type: str = "",
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CSV에 넣을 평가 지표. collision_* 는 (t, 쌍) 이벤트 합과 동일한 기준.
    """
    T = int(frames.shape[0])
    ts_with_col = sum(
        1 for t in range(T) if len(detect_collisions(frames, t, min_dist)) > 0
    )
    pair_counts_per_t = [len(detect_collisions(frames, t, min_dist)) for t in range(T)]
    max_pairs_one_timestep = max(pair_counts_per_t) if pair_counts_per_t else 0

    mean_drone = float(total_dist / n) if n else 0.0
    reduction = int(collision_before - collision_after)

    residual_ratio = (
        float(collision_after) / float(collision_before)
        if collision_before > 0
        else 0.0
    )

    reduction_ratio = (
        float(collision_before - collision_after) / float(collision_before)
        if collision_before > 0
        else 0.0
    )

    row: Dict[str, Any] = {
        "timesteps": T,
        "active_timesteps": count_active_timesteps(frames),
        "trajectory_type": trajectory_type,
        "collision_avoidance_type": collision_avoidance_type,
        "timesteps_with_collision": ts_with_col,
        "max_collision_pairs_at_one_timestep": max_pairs_one_timestep,
        "mean_dist_per_drone": round(mean_drone, 6),
        "collision_pair_events_before": collision_before,
        "collision_pair_events_after": collision_after,
        "collision_pair_events_reduced": reduction,
        "collision_reduction_ratio": round(reduction_ratio, 6),
        "collision_residual_ratio_vs_before": round(residual_ratio, 6),
        "is_collision_free": 1 if collision_after == 0 else 0,
        "cbs_used_hungarian_fallback": 1 if cbs_fallback else 0,
    }

    if cbs_success is not None:
        row["cbs_planner_success"] = 1 if cbs_success else 0
    else:
        row["cbs_planner_success"] = ""

    if stats_resolve is not None:
        row["resolve_passes_used"] = stats_resolve.get("passes_used", "")
        row["resolve_total_detected"] = stats_resolve.get("total_detected", "")
        row["resolve_steps_with_collision_first_pass"] = stats_resolve.get(
            "steps_with_collision", ""
        )
        row["resolve_remaining_reported"] = stats_resolve.get("remaining", "")
        row["resolve_stop_reason"] = stats_resolve.get("stop_reason", "")
        row["resolve_runtime_sec"] = stats_resolve.get("runtime_sec", "")
        row["resolve_best_remaining"] = stats_resolve.get("best_remaining", "")
        row["resolve_stagnant_passes"] = stats_resolve.get("stagnant_passes", "")
    else:
        row["resolve_passes_used"] = ""
        row["resolve_total_detected"] = ""
        row["resolve_steps_with_collision_first_pass"] = ""
        row["resolve_remaining_reported"] = ""
        row["resolve_stop_reason"] = ""
        row["resolve_runtime_sec"] = ""
        row["resolve_best_remaining"] = ""
        row["resolve_stagnant_passes"] = ""

    if extra_stats is not None:
        row.update(extra_stats)

    return row


_TRIAL_COLUMN_ORDER = [
    "experiment_run_id",
    "trial_id",
    "trial_index",
    "algorithm",
    "n",
    "start_text",
    "end_text",
    "text",
    "size",
    "min_safe_dist",
    "max_steps",
    "total_dist",
    "max_dist",
    "distance_metric",
    "visual_total_dist",
    "visual_max_dist",
    "soc_cost",
    "soc_max_agent_cost",
    "soc_cost_metric",
    "mean_dist_per_drone",
    "assign_time_sec",
    "pipeline_time_sec",
    "timesteps",
    "active_timesteps",
    "timesteps_with_collision",
    "max_collision_pairs_at_one_timestep",
    "collision_pair_events_before",
    "collision_pair_events_after",
    "collision_pair_events_reduced",
    "collision_residual_ratio_vs_before",
    "is_collision_free",
    "heuristic_planner_success",
    "heuristic_solution_status",
    "heuristic_stop_reason",
    "cbs_planner_success",
    "cbs_used_hungarian_fallback",
    "cbs_failure_reason",
    "cbs_search_stop_reason",
    "cbs_search_expanded_nodes",
    "cbs_search_generated_nodes",
    "cbs_search_max_open_size",
    "cbs_search_conflicts_seen",
    "cbs_search_vertex_conflicts_seen",
    "cbs_search_edge_conflicts_seen",
    "cbs_search_first_conflict_type",
    "cbs_search_first_conflict_t",
    "cbs_search_first_conflict_agents",
    "cbs_search_unique_conflict_pairs",
    "cbs_search_top_conflict_pair",
    "cbs_search_top_conflict_pair_count",
    "cbs_search_low_level_astar_calls",
    "cbs_search_low_level_astar_failures",
    "cbs_search_low_level_astar_time_sec",
    "cbs_search_max_constraints",
    "cbs_search_initial_cost",
    "cbs_search_solution_cost",
    "cbs_search_required_steps",
    "cbs_search_effective_max_steps",
    "cbs_search_runtime_sec",
    "cbs_skipped_by_precheck",
    "cbs_skip_reason",
    "cbs_grid_conflict_free",
    "cbs_grid_first_conflict_t",
    "cbs_grid_vertex_conflicts_before",
    "cbs_grid_edge_conflicts_before",
    "cbs_grid_conflicts_before",
    "cbs_grid_timesteps_with_conflict_before",
    "cbs_grid_conflict_free_before",
    "cbs_grid_first_conflict_type_before",
    "cbs_grid_first_conflict_t_before",
    "cbs_grid_first_conflict_agents_before",
    "cbs_grid_vertex_conflicts_after",
    "cbs_grid_edge_conflicts_after",
    "cbs_grid_conflicts_after",
    "cbs_grid_timesteps_with_conflict_after",
    "cbs_grid_conflict_free_after",
    "cbs_grid_first_conflict_type_after",
    "cbs_grid_first_conflict_t_after",
    "cbs_grid_first_conflict_agents_after",
    "cbs_precheck_start_duplicate_cells",
    "cbs_precheck_start_duplicate_agents",
    "cbs_precheck_start_max_cell_occupancy",
    "cbs_precheck_goal_duplicate_cells",
    "cbs_precheck_goal_duplicate_agents",
    "cbs_precheck_goal_max_cell_occupancy",
    "cbs_precheck_initial_close_pair",
    "cbs_precheck_initial_close_dist",
    "cbs_precheck_final_close_pair",
    "cbs_precheck_final_close_dist",
    "cbs_ready_grid_scale",
    "cbs_ready_coordinate_adjustment",
    "heuristic_step_size",
    "start_cbs_ready_reassigned_cells",
    "start_cbs_ready_duplicate_cells_before",
    "start_cbs_ready_duplicate_cells_after",
    "start_cbs_ready_close_pair_before",
    "start_cbs_ready_close_dist_before",
    "start_cbs_ready_close_pair_after",
    "start_cbs_ready_close_dist_after",
    "start_cbs_ready_mean_adjustment",
    "start_cbs_ready_max_adjustment",
    "goal_cbs_ready_reassigned_cells",
    "goal_cbs_ready_duplicate_cells_before",
    "goal_cbs_ready_duplicate_cells_after",
    "goal_cbs_ready_close_pair_before",
    "goal_cbs_ready_close_dist_before",
    "goal_cbs_ready_close_pair_after",
    "goal_cbs_ready_close_dist_after",
    "goal_cbs_ready_mean_adjustment",
    "goal_cbs_ready_max_adjustment",
    "resolve_passes_used",
    "resolve_total_detected",
    "resolve_steps_with_collision_first_pass",
    "resolve_remaining_reported",
    "resolve_stop_reason",
    "resolve_runtime_sec",
    "resolve_best_remaining",
    "resolve_stagnant_passes",
    "trajectory_type",
    "collision_avoidance_type",
    "collision_reduction_ratio",
    "delay_step",
    "max_delay",
    "max_delay_used",
    "mean_delay_used",
    "delayed_drone_count",
]


def save_trials_csv(
    rows: List[Dict[str, Any]],
    run_id: str,
    results_dir: str = "results",
    history_filename: str = "history.csv",
) -> str:
    """
    results/runs/{run_id}/trials.csv 저장 후 results/history.csv 에 동일 스키마로 append.
    반환: trials.csv 상대 경로.
    """
    root = results_run_dir(run_id, base=results_dir)
    os.makedirs(root, exist_ok=True)
    trials_path = os.path.join(root, "trials.csv")
    df = pd.DataFrame(rows)
    front = [c for c in _TRIAL_COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    df.to_csv(trials_path, index=False, encoding="utf-8-sig")

    hist_path = os.path.join(results_dir, history_filename)
    os.makedirs(os.path.dirname(os.path.abspath(hist_path)) or ".", exist_ok=True)
    if os.path.isfile(hist_path):
        try:
            prev = pd.read_csv(hist_path)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            backup_path = f"{hist_path}.malformed.{run_id}.bak"
            shutil.copy2(hist_path, backup_path)
            df.to_csv(hist_path, index=False, encoding="utf-8-sig")
        else:
            merged = pd.concat([prev, df], ignore_index=True, sort=False)
            front = [c for c in _TRIAL_COLUMN_ORDER if c in merged.columns]
            rest = [c for c in merged.columns if c not in front]
            merged = merged[front + rest]
            merged.to_csv(hist_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    return trials_path
