"""
2. 헝가리안 배정 + 그리디 이동 + 충돌 회피 파이프라인.

coordinate(목표 좌표)와 독립적으로, 드론 초기 위치·목표·헝가리안 결과를 받아
타임라인을 만들고 run_collision_resolution으로 충돌을 줄인다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLLISION_RESOLUTION_MAX_PASSES, MAX_STEPS, MIN_SAFE_DIST

from .hungarian import build_cost_matrix, hungarian_assign
from .timeline import (
    compute_timeline_hungarian,
    detect_collisions,
    run_collision_resolution,
    smooth_timeline_separation,
)


def hungarian_assignment(
    drones: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    드론–목표 1:1 최소 총거리 배정.

    Returns
    -------
    (cost_matrix, assignment)
        assignment[i] = 드론 i에 할당된 목표 인덱스.
    """
    cost = build_cost_matrix(drones, targets)
    assignment = hungarian_assign(cost)
    return cost, assignment


def greedy_timeline_then_resolve(
    drones: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    max_steps: int = MAX_STEPS,
    min_dist: float = MIN_SAFE_DIST,
    frames_raw: Optional[np.ndarray] = None,
    max_passes: int = COLLISION_RESOLUTION_MAX_PASSES,
    smooth_visual: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    그리디(sign) 타임라인 생성 후 충돌 회피를 적용한다.

    Parameters
    ----------
    frames_raw
        이미 계산한 그리디 타임라인. 주면 재계산하지 않는다 (collision_before용 1회 생성).
    smooth_visual
        True면 회피 직후 시간축 스무딩+분리로 GIF용 떨림을 줄인다. 실험 CSV의 충돌 수는
        ``stats['remaining']``(스무딩 후 재계산)을 보면 된다.
    """
    if frames_raw is None:
        frames_raw = compute_timeline_hungarian(
            drones, targets, assignment, max_steps=max_steps
        )
    frames, assignment, stats = run_collision_resolution(
        frames_raw,
        assignment,
        targets,
        min_dist=min_dist,
        max_passes=max_passes,
    )
    if smooth_visual:
        frames = smooth_timeline_separation(frames, min_dist=min_dist)
        remaining = sum(
            len(detect_collisions(frames, t, min_dist)) for t in range(frames.shape[0])
        )
        stats = {**stats, "remaining": remaining}
    return frames, assignment, stats, frames_raw
