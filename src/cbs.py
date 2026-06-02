"""
CBS(Conflict-Based Search) 알고리즘을 구현하는 모듈.
다수 드론의 충돌을 감지하고 제약 조건을 추가해 충돌 없는 경로를 탐색한다.

구조
----
High-Level : 충돌 발견 시 제약 조건 트리를 분기하며 best-first(최소 비용 우선) 탐색.
Low-Level  : 각 드론마다 astar_with_constraints로 단일 경로를 재계획.

탐색 공간  : (x, y, t) — 2D 위치 + 시각의 3차원 격자.
이동 모델  : 기본 4방향 + 대기(논문 정합), 옵션으로 8방향 + 대기 지원.
"""

import heapq
import itertools
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_STEPS

# 4방향 + 대기 (논문 기본 MAPF 실험 정합)
_MOVES_4N_WAIT: List[Tuple[int, int]] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (0, 0),
]

# 8방향 + 대기 (기존 동작 호환 옵션)
_MOVES_8N_WAIT: List[Tuple[int, int]] = [
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
]

# 타입 별칭
Pos        = Tuple[int, int]
Constraint = Tuple[int, int, int]   # (x, y, t)
EdgeConstraint = Tuple[int, int, int, int, int]  # (x1, y1, x2, y2, t)
Path       = List[Pos]
PathsDict  = Dict[int, Path]

_AXIS_MOVE_COST = 1.0


def _move_cost(dx: int, dy: int) -> float:
    """논문 기본 SoC: 목표 도착 전 move/wait action은 모두 비용 1."""
    return _AXIS_MOVE_COST


def _path_cost(path: Path) -> float:
    """논문 기본 SoC 기준 단일 agent의 도착 전 action 수."""
    if len(path) < 2:
        return 0.0
    return sum(
        _move_cost(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:])
    )


def _direct_unconstrained_path(
    start: Pos,
    goal: Pos,
    moves: List[Tuple[int, int]],
) -> Path:
    """제약/장애물이 없을 때 열린 격자에서의 최단 초기 경로."""
    x, y = start
    gx, gy = goal
    path: Path = [(x, y)]
    allows_diagonal = any(abs(dx) == 1 and abs(dy) == 1 for dx, dy in moves)
    if allows_diagonal:
        while (x, y) != (gx, gy):
            x += int(math.copysign(1, gx - x)) if x != gx else 0
            y += int(math.copysign(1, gy - y)) if y != gy else 0
            path.append((x, y))
    else:
        while x != gx:
            x += int(math.copysign(1, gx - x))
            path.append((x, y))
        while y != gy:
            y += int(math.copysign(1, gy - y))
            path.append((x, y))
    return path


@dataclass(frozen=True)
class AgentConstraints:
    """개별 agent에 적용되는 CBS 제약."""
    vertex: FrozenSet[Constraint]
    edge: FrozenSet[EdgeConstraint]


# ---------------------------------------------------------------------------
# Low-Level : 제약 조건 A*
# ---------------------------------------------------------------------------

def astar_with_constraints(
    start: Pos,
    goal: Pos,
    vertex_constraints: Set[Constraint],
    edge_constraints: Set[EdgeConstraint],
    max_steps: int,
    grid_bounds: Tuple[int, int, int, int],
    blocked_cells: Set[Pos],
    moves: List[Tuple[int, int]],
    deadline: Optional[float] = None,
) -> Optional[Path]:
    """
    제약 조건을 피하며 start → goal 최단 경로를 (x, y, t) 3차원 A*로 탐색한다.

    핵심 규칙
    ---------
    - Vertex 제약 (x, y, t) : 시각 t에 위치 (x, y) 점유 불가.
    - Edge 제약 (x1, y1, x2, y2, t) : t→t+1 사이 directed edge 통과 금지.
    - 목표 점유 제약 처리 : 드론이 goal에 도착한 뒤 제자리 대기(패딩)가 충돌로
      이어질 수 있으므로, goal에 대한 제약 중 가장 늦은 시각(latest_goal_constraint)
      보다 '이후'에 도착해야만 경로를 확정한다.
    - 휴리스틱 : Chebyshev 거리 (8방향 이동에 대해 허용적, admissible).

    Parameters
    ----------
    start       : (x, y) 출발 위치 (정수 좌표)
    goal        : (x, y) 목표 위치 (정수 좌표)
    vertex_constraints : {(x, y, t), ...}
    edge_constraints   : {(x1, y1, x2, y2, t), ...}
    max_steps          : 탐색 깊이 한계 (이 시각 이후 노드는 확장하지 않음)

    Returns
    -------
    Path (start ~ goal 포함한 위치 리스트), 또는 None (경로 없음)

    Example
    -------
    # path = astar_with_constraints((0, 0), (5, 3), set(), max_steps=50)
    # assert path[0] == (0, 0) and path[-1] == (5, 3)
    """
    sx, sy = start
    gx, gy = goal

    # goal에 걸린 제약 중 가장 늦은 시각 → 그 이후에 도착해야 안전
    latest_goal_constraint: int = max(
        (t for (x, y, t) in vertex_constraints if x == gx and y == gy),
        default=-1,
    )

    min_x, max_x, min_y, max_y = grid_bounds
    if not (min_x <= sx <= max_x and min_y <= sy <= max_y):
        return None
    if not (min_x <= gx <= max_x and min_y <= gy <= max_y):
        return None
    if (sx, sy) in blocked_cells or (gx, gy) in blocked_cells:
        return None
    if (sx, sy, 0) in vertex_constraints:
        return None

    allows_diagonal = any(abs(dx) == 1 and abs(dy) == 1 for dx, dy in moves)

    def h(x: int, y: int) -> float:
        dx = abs(x - gx)
        dy = abs(y - gy)
        if allows_diagonal:
            return float(max(dx, dy))  # 8방향 unit-cost Chebyshev
        return float(dx + dy)  # 4방향 Manhattan

    # 힙: (f, g, x, y, t)
    open_heap: list = []
    heapq.heappush(open_heap, (h(sx, sy), 0, sx, sy, 0))

    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {(sx, sy, 0): 0.0}

    while open_heap:
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        f, g, x, y, t = heapq.heappop(open_heap)

        cur_state = (x, y, t)

        # 오래된(stale) 힙 항목 무시 (lazy deletion)
        if g > g_score.get(cur_state, float("inf")):
            continue

        # 목표 도달 확인 — goal 제약이 모두 지난 시각이어야 확정
        if (x, y) == (gx, gy) and t > latest_goal_constraint:
            path: Path = []
            cur = cur_state
            while cur in came_from:
                path.append((cur[0], cur[1]))
                cur = came_from[cur]
            path.append((cur[0], cur[1]))
            path.reverse()
            return path

        if t >= max_steps:
            continue

        # 인접 노드 확장
        for dx, dy in moves:
            nx, ny, nt = x + dx, y + dy, t + 1
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue
            if (nx, ny) in blocked_cells:
                continue
            if (nx, ny, nt) in vertex_constraints:
                continue
            if (x, y, nx, ny, t) in edge_constraints:
                continue
            new_g = g + _move_cost(dx, dy)
            new_state = (nx, ny, nt)
            if new_g < g_score.get(new_state, float("inf")):
                g_score[new_state] = new_g
                came_from[new_state] = cur_state
                heapq.heappush(open_heap, (new_g + h(nx, ny), new_g, nx, ny, nt))

    return None  # 경로 없음


# ---------------------------------------------------------------------------
# High-Level 보조 : 충돌 감지
# ---------------------------------------------------------------------------

def detect_conflict(paths: PathsDict) -> Optional[dict]:
    """
    모든 드론 경로를 검사해 첫 번째 충돌을 반환한다.

    경로 길이가 다를 경우 마지막 위치로 패딩하여 비교한다
    (드론이 목표에 도달한 뒤 제자리에 머문다고 가정).

    충돌 유형
    ---------
    vertex : 같은 시각 t에 두 드론이 동일 위치를 점유.
    edge   : t→t+1 사이에 두 드론이 위치를 맞바꿈 (교차 이동).

    Parameters
    ----------
    paths : {drone_id: [(x, y), ...]} — 드론별 경로

    Returns
    -------
    충돌 정보 dict, 또는 None (충돌 없음)

    반환 dict 구조
    ~~~~~~~~~~~~~~
    {
        'type'   : 'vertex' | 'edge',
        'agents' : (i, j),           # 충돌 드론 쌍
        'pos'    : (x, y),           # vertex 충돌 위치
        'move_i' : ((x1,y1),(x2,y2)) | None, # edge only: 드론 i의 이동 (t->t+1)
        'move_j' : ((x1,y1),(x2,y2)) | None, # edge only: 드론 j의 이동 (t->t+1)
        't'      : int,              # vertex: 충돌 시각 / edge: 교차 시작 시각
    }

    Example
    -------
    # vertex conflict
    # c = detect_conflict({0: [(0,0),(1,0)], 1: [(2,0),(1,0)]})
    # assert c['type'] == 'vertex' and c['t'] == 1
    #
    # edge conflict
    # c = detect_conflict({0: [(0,0),(1,0)], 1: [(1,0),(0,0)]})
    # assert c['type'] == 'edge' and c['t'] == 0
    """
    agents = list(paths.keys())
    if len(agents) < 2:
        return None

    max_len = max(len(p) for p in paths.values())

    def get_pos(aid: int, t: int) -> Pos:
        p = paths[aid]
        return p[t] if t < len(p) else p[-1]

    for t in range(max_len):
        # ── Vertex conflict ──────────────────────────────────────────────
        pos_owner: Dict[Pos, int] = {}
        for aid in agents:
            pos = get_pos(aid, t)
            if pos in pos_owner:
                return {
                    "type": "vertex",
                    "agents": (pos_owner[pos], aid),
                    "pos": pos,
                    "move_i": None,
                    "move_j": None,
                    "t": t,
                }
            pos_owner[pos] = aid

        # ── Edge conflict (swap) ─────────────────────────────────────────
        if t + 1 < max_len:
            n = len(agents)
            for ii in range(n):
                for jj in range(ii + 1, n):
                    ai, aj = agents[ii], agents[jj]
                    pi_t,  pj_t  = get_pos(ai, t),     get_pos(aj, t)
                    pi_t1, pj_t1 = get_pos(ai, t + 1), get_pos(aj, t + 1)
                    if pi_t == pj_t1 and pj_t == pi_t1:
                        return {
                            "type": "edge",
                            "agents": (ai, aj),
                            "pos": None,
                            "move_i": (pi_t, pi_t1),
                            "move_j": (pj_t, pj_t1),
                            "t": t,
                        }

    return None  # 충돌 없음


# ---------------------------------------------------------------------------
# High-Level : CBS 메인
# ---------------------------------------------------------------------------

def cbs_assign(
    drones: np.ndarray,
    targets: np.ndarray,
    assignment: np.ndarray,
    max_steps: int = MAX_STEPS,
    grid_margin: int = 5,
    grid_scale: int = 1,
    strict_cbs: bool = True,
    max_iterations: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    move_model: str = "4n",
    blocked_cells: Optional[Set[Pos]] = None,
    return_stats: bool = False,
):
    """
    CBS(Conflict-Based Search)로 드론 전체의 충돌 없는 경로를 탐색한다.

    알고리즘 흐름
    -------------
    1. 초기화 : 제약 없이 각 드론에 대해 A*로 최단 경로 계획.
    2. 루프    : 우선순위 큐(최소 비용 우선)에서 CBS 노드를 꺼낸다.
       a. 충돌 없음 → 해 반환.
       b. 충돌 발견 → 충돌 드론 쌍에 대해 각각 제약 추가 후 A* 재계획,
          두 자식 노드를 큐에 삽입.
    3. 반복 횟수 초과 or 큐 소진 → None 반환.

    제약 조건 형식 : agent별 {vertex, edge}를 독립 관리

    Parameters
    ----------
    drones     : ndarray (n, 2) — 드론 초기 위치
    targets    : ndarray (n, 2) — 목표 좌표 집합 (generate_coordinates 결과)
    assignment : ndarray (n,)   — hungarian_assign 결과. assignment[i] = 드론 i의 목표 인덱스
    max_steps  : A* 탐색 깊이 제한 (기본값 config.MAX_STEPS)

    Returns
    -------
    {drone_id: [(x, y), ...]} — 드론별 충돌 없는 경로, 또는 None.
    return_stats=True이면 (paths_or_none, stats) 튜플을 반환한다.

    주의
    ----
    - CBS 탐색 트리는 최악의 경우 지수적으로 성장한다.
      n이 클수록 (≥ 50) max_iterations 내에 해를 못 찾을 수 있음.
    - 해를 못 찾으면 None 반환. 호출부는 CBS 실패로 기록한다.

    Example
    -------
    # paths = cbs_assign(drones, targets, assignment, max_steps=100)
    # if paths is None:
    #     print("CBS 탐색 실패")
    # else:
    #     assert detect_conflict(paths) is None
    """
    n = len(drones)
    started_at = time.perf_counter()
    conflict_pair_counts: Counter[Tuple[int, int]] = Counter()
    stats: Dict[str, object] = {
        "cbs_search_stop_reason": "",
        "cbs_search_expanded_nodes": 0,
        "cbs_search_generated_nodes": 0,
        "cbs_search_max_open_size": 0,
        "cbs_search_conflicts_seen": 0,
        "cbs_search_vertex_conflicts_seen": 0,
        "cbs_search_edge_conflicts_seen": 0,
        "cbs_search_first_conflict_type": "",
        "cbs_search_first_conflict_t": "",
        "cbs_search_first_conflict_agents": "",
        "cbs_search_unique_conflict_pairs": 0,
        "cbs_search_top_conflict_pair": "",
        "cbs_search_top_conflict_pair_count": 0,
        "cbs_search_low_level_astar_calls": 0,
        "cbs_search_low_level_astar_failures": 0,
        "cbs_search_low_level_astar_time_sec": 0.0,
        "cbs_search_max_constraints": 0,
        "cbs_search_initial_cost": "",
        "cbs_search_solution_cost": "",
        "cbs_search_required_steps": "",
        "cbs_search_effective_max_steps": "",
        "cbs_search_runtime_sec": "",
    }

    def _finish(paths: Optional[PathsDict], reason: str):
        stats["cbs_search_stop_reason"] = reason
        stats["cbs_search_runtime_sec"] = round(
            time.perf_counter() - started_at, 6
        )
        if conflict_pair_counts:
            pair, count = conflict_pair_counts.most_common(1)[0]
            stats["cbs_search_unique_conflict_pairs"] = len(conflict_pair_counts)
            stats["cbs_search_top_conflict_pair"] = f"{pair[0]}-{pair[1]}"
            stats["cbs_search_top_conflict_pair_count"] = count
        stats["cbs_search_low_level_astar_time_sec"] = round(
            float(stats["cbs_search_low_level_astar_time_sec"]), 6
        )
        if return_stats:
            return paths, stats
        return paths

    def _constraint_count(constraints: Dict[int, AgentConstraints]) -> int:
        return sum(len(c.vertex) + len(c.edge) for c in constraints.values())

    def _run_low_level_astar(
        start: Pos,
        goal: Pos,
        vertex_constraints: Set[Constraint],
        edge_constraints: Set[EdgeConstraint],
        max_steps_for_agent: int,
        bounds: Tuple[int, int, int, int],
        blocked_for_agent: Set[Pos],
        moves_for_agent: List[Tuple[int, int]],
    ) -> Optional[Path]:
        stats["cbs_search_low_level_astar_calls"] = (
            int(stats["cbs_search_low_level_astar_calls"]) + 1
        )
        t_astar = time.perf_counter()
        path = astar_with_constraints(
            start,
            goal,
            vertex_constraints,
            edge_constraints,
            max_steps_for_agent,
            bounds,
            blocked_for_agent,
            moves_for_agent,
            deadline,
        )
        stats["cbs_search_low_level_astar_time_sec"] = (
            float(stats["cbs_search_low_level_astar_time_sec"])
            + time.perf_counter() - t_astar
        )
        if path is None:
            stats["cbs_search_low_level_astar_failures"] = (
                int(stats["cbs_search_low_level_astar_failures"]) + 1
            )
        return path

    if grid_scale < 1:
        raise ValueError("grid_scale must be >= 1")
    deadline = (
        time.perf_counter() + timeout_sec
        if timeout_sec is not None and timeout_sec > 0
        else None
    )

    scaled_margin = max(1, int(round(grid_margin * grid_scale)))

    # 실수 좌표 → (고해상도) 정수 격자 좌표 변환
    starts: List[Pos] = [
        (
            int(round(drones[i][0] * grid_scale)),
            int(round(drones[i][1] * grid_scale)),
        )
        for i in range(n)
    ]
    goals: List[Pos] = [
        (
            int(round(targets[assignment[i]][0] * grid_scale)),
            int(round(targets[assignment[i]][1] * grid_scale)),
        )
        for i in range(n)
    ]
    if len(set(starts)) < n or len(set(goals)) < n:
        return _finish(None, "duplicate_start_or_goal")

    # ── 유한 격자 경계 구성 (논문형 G(V,E)에서 V를 bounded grid로 근사) ─────
    all_x = [p[0] for p in starts] + [p[0] for p in goals]
    all_y = [p[1] for p in starts] + [p[1] for p in goals]
    min_x = min(all_x) - scaled_margin
    max_x = max(all_x) + scaled_margin
    min_y = min(all_y) - scaled_margin
    max_y = max(all_y) + scaled_margin
    grid_bounds = (min_x, max_x, min_y, max_y)
    blocked = (
        {
            (
                int(round(pos[0] * grid_scale)),
                int(round(pos[1] * grid_scale)),
            )
            for pos in blocked_cells
        }
        if blocked_cells
        else set()
    )

    if move_model == "4n":
        moves = _MOVES_4N_WAIT
    elif move_model == "8n":
        moves = _MOVES_8N_WAIT
    else:
        raise ValueError("move_model must be one of {'4n', '8n'}")

    # ── 동적 탐색 깊이 설정 ────────────────────────────────────────────────
    # 기본 max_steps가 너무 작아 해가 잘리는 경우를 줄이기 위해,
    # 시작-목표 하한 거리의 최댓값에 여유분(slack)을 더해 최소 필요 깊이를 보장한다.
    if move_model == "4n":
        lb_per_agent = [
            abs(starts[i][0] - goals[i][0]) + abs(starts[i][1] - goals[i][1])
            for i in range(n)
        ]  # Manhattan lower bound
    else:
        lb_per_agent = [
            max(abs(starts[i][0] - goals[i][0]), abs(starts[i][1] - goals[i][1]))
            for i in range(n)
        ]  # Chebyshev lower bound

    required_steps = max(lb_per_agent) if lb_per_agent else 0
    slack_steps = 40
    effective_max_steps = max(max_steps, required_steps + slack_steps)
    stats["cbs_search_required_steps"] = required_steps
    stats["cbs_search_effective_max_steps"] = effective_max_steps

    def _to_world(paths: PathsDict) -> PathsDict:
        if grid_scale == 1:
            return paths
        inv = 1.0 / float(grid_scale)
        return {
            aid: [(x * inv, y * inv) for (x, y) in path]
            for aid, path in paths.items()
        }

    # ── 초기 경로 계획 (제약 없음) ────────────────────────────────────────
    init_paths: PathsDict = {}
    for i in range(n):
        if blocked:
            path = _run_low_level_astar(
                starts[i],
                goals[i],
                set(),
                set(),
                effective_max_steps,
                grid_bounds,
                blocked,
                moves,
            )
        else:
            path = _direct_unconstrained_path(starts[i], goals[i], moves)
        if path is None:
            if deadline is not None and time.perf_counter() >= deadline:
                return _finish(None, "timeout")
            return _finish(None, "initial_path_failed")
        init_paths[i] = path

    def total_cost(paths: PathsDict) -> float:
        """논문 기본 Sum-of-Costs: 모든 드론의 최종 도착 전 action 수 합."""
        return sum(_path_cost(p) for p in paths.values())

    # ── CBS 우선순위 큐 초기화 ────────────────────────────────────────────
    # 힙 원소: (cost, tiebreak_id, constraints_dict, paths_dict)
    # constraints_dict : {drone_id: frozenset of (x, y, t)}
    counter = itertools.count()
    init_constraints: Dict[int, AgentConstraints] = {
        i: AgentConstraints(vertex=frozenset(), edge=frozenset()) for i in range(n)
    }
    open_list: list = [
        (total_cost(init_paths), next(counter), init_constraints, init_paths)
    ]
    stats["cbs_search_generated_nodes"] = 1
    stats["cbs_search_max_open_size"] = 1
    stats["cbs_search_initial_cost"] = round(total_cost(init_paths), 6)

    if strict_cbs:
        max_loops = None
    else:
        max_loops = 2_000 if max_iterations is None else max_iterations

    loops = 0
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            return _finish(None, "timeout")
        if max_loops is not None and loops >= max_loops:
            return _finish(None, "iteration_limit")
        if not open_list:
            return _finish(None, "open_empty")
        loops += 1

        cost, _, constraints, paths = heapq.heappop(open_list)
        stats["cbs_search_expanded_nodes"] = int(
            stats["cbs_search_expanded_nodes"]
        ) + 1
        stats["cbs_search_max_constraints"] = max(
            int(stats["cbs_search_max_constraints"]),
            _constraint_count(constraints),
        )

        conflict = detect_conflict(paths)
        if conflict is None:
            stats["cbs_search_solution_cost"] = round(cost, 6)
            return _finish(_to_world(paths), "success")  # ★ 충돌 없는 해 발견

        ai, aj = conflict["agents"]
        pair = tuple(sorted((ai, aj)))
        conflict_pair_counts[pair] += 1
        stats["cbs_search_conflicts_seen"] = int(
            stats["cbs_search_conflicts_seen"]
        ) + 1
        if conflict["type"] == "vertex":
            stats["cbs_search_vertex_conflicts_seen"] = int(
                stats["cbs_search_vertex_conflicts_seen"]
            ) + 1
        else:
            stats["cbs_search_edge_conflicts_seen"] = int(
                stats["cbs_search_edge_conflicts_seen"]
            ) + 1
        if not stats["cbs_search_first_conflict_type"]:
            stats["cbs_search_first_conflict_type"] = conflict["type"]
            stats["cbs_search_first_conflict_t"] = conflict["t"]
            stats["cbs_search_first_conflict_agents"] = f"{ai}-{aj}"

        # ── 두 자식 노드 생성 ─────────────────────────────────────────────
        for agent in (ai, aj):
            # 충돌 유형에 따라 추가할 제약 결정
            if conflict["type"] == "vertex":
                # vertex : 해당 위치·시각을 금지
                cx, cy = conflict["pos"]
                ct = conflict["t"]
                add_vertex = (cx, cy, ct)
                add_edge = None
            else:
                # edge : 해당 시각의 directed edge 통과 금지 (논문 정합)
                move = conflict["move_i"] if agent == ai else conflict["move_j"]
                (x1, y1), (x2, y2) = move
                ct = conflict["t"]
                add_vertex = None
                add_edge = (x1, y1, x2, y2, ct)

            new_constraints = dict(constraints)
            prev = constraints[agent]
            if add_vertex is not None:
                new_constraints[agent] = AgentConstraints(
                    vertex=prev.vertex | frozenset([add_vertex]),
                    edge=prev.edge,
                )
            else:
                new_constraints[agent] = AgentConstraints(
                    vertex=prev.vertex,
                    edge=prev.edge | frozenset([add_edge]),
                )

            # 해당 드론만 재계획
            new_path = _run_low_level_astar(
                starts[agent],
                goals[agent],
                set(new_constraints[agent].vertex),
                set(new_constraints[agent].edge),
                effective_max_steps,
                grid_bounds,
                blocked,
                moves,
            )
            if deadline is not None and time.perf_counter() >= deadline:
                return _finish(None, "timeout")
            if new_path is None:
                continue  # 이 브랜치는 해 없음 → 스킵

            new_paths = dict(paths)
            new_paths[agent] = new_path
            new_cost = total_cost(new_paths)

            heapq.heappush(
                open_list,
                (new_cost, next(counter), new_constraints, new_paths),
            )
            stats["cbs_search_generated_nodes"] = int(
                stats["cbs_search_generated_nodes"]
            ) + 1
            stats["cbs_search_max_open_size"] = max(
                int(stats["cbs_search_max_open_size"]),
                len(open_list),
            )
            stats["cbs_search_max_constraints"] = max(
                int(stats["cbs_search_max_constraints"]),
                _constraint_count(new_constraints),
            )

    return _finish(None, "unknown")  # strict=False에서 보통 도달하지 않음


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    sys.path.insert(0, os.path.dirname(__file__))  # src/ 경로 추가
    from hungarian import build_cost_matrix, hungarian_assign

    print("=== astar_with_constraints 단위 테스트 ===\n")

    # 1) 기본 경로 탐색
    path = astar_with_constraints(
        (0, 0), (5, 5), set(), set(), max_steps=50,
        grid_bounds=(-20, 20, -20, 20), blocked_cells=set(), moves=_MOVES_8N_WAIT
    )
    assert path is not None and path[0] == (0, 0) and path[-1] == (5, 5)
    assert len(path) == 6  # Chebyshev 거리 5 → 6 스텝(start 포함)
    print("[PASS] (0,0)→(5,5) 최단 경로 길이 = 6")

    # 2) start == goal
    path_trivial = astar_with_constraints(
        (3, 3), (3, 3), set(), set(), max_steps=20,
        grid_bounds=(-20, 20, -20, 20), blocked_cells=set(), moves=_MOVES_8N_WAIT
    )
    assert path_trivial == [(3, 3)]
    print("[PASS] start == goal → [(3,3)]")

    # 3) 제약 조건 회피
    #    (2, 0) ~ (2, 2) 를 t=1~3 에 막으면 우회해야 함
    blocked = {(2, y, t) for y in range(-1, 3) for t in range(1, 6)}
    path_detour = astar_with_constraints(
        (0, 0), (4, 0), blocked, set(), max_steps=30,
        grid_bounds=(-20, 20, -20, 20), blocked_cells=set(), moves=_MOVES_8N_WAIT
    )
    assert path_detour is not None and path_detour[-1] == (4, 0)
    for pos, t in zip(path_detour, range(len(path_detour))):
        assert (pos[0], pos[1], t) not in blocked, f"제약 위반: {pos} at t={t}"
    print("[PASS] 장애물 회피 경로 생성 및 제약 미위반 확인")

    # 4) 불가능한 경로 → None
    impossible = {(x, y, t) for x in range(-5, 10) for y in range(-5, 10) for t in range(1, 30)}
    path_none = astar_with_constraints(
        (0, 0), (5, 5), impossible, set(), max_steps=20,
        grid_bounds=(-20, 20, -20, 20), blocked_cells=set(), moves=_MOVES_8N_WAIT
    )
    assert path_none is None
    print("[PASS] 탈출 불가 제약 → None 반환")

    print("\n=== detect_conflict 단위 테스트 ===\n")

    # 5) Vertex conflict
    vc = detect_conflict({0: [(0, 0), (1, 0)], 1: [(2, 0), (1, 0)]})
    assert vc is not None and vc["type"] == "vertex" and vc["t"] == 1
    print("[PASS] vertex conflict : t=1, pos=(1,0)")

    # 6) Edge conflict (swap)
    ec = detect_conflict({0: [(0, 0), (1, 0)], 1: [(1, 0), (0, 0)]})
    assert ec is not None and ec["type"] == "edge" and ec["t"] == 0
    print("[PASS] edge conflict : t=0 교차 이동 감지")

    # 7) No conflict
    nc = detect_conflict({0: [(0, 0), (1, 0), (2, 0)], 1: [(0, 2), (1, 2), (2, 2)]})
    assert nc is None
    print("[PASS] 충돌 없는 경로 → None 반환")

    # 8) 경로 길이 불일치 (패딩 테스트)
    #    drone0 이 (1,0)에 먼저 도달 후 대기, drone1이 나중에 (1,0) 진입
    padded_conflict = detect_conflict({
        0: [(0, 0), (1, 0)],             # t=1부터 (1,0)에 정박
        1: [(3, 0), (2, 0), (1, 0)],     # t=2에 (1,0) 도달 → vertex conflict
    })
    assert padded_conflict is not None and padded_conflict["type"] == "vertex"
    print("[PASS] 경로 길이 불일치 패딩 후 vertex conflict 감지")

    print("\n=== cbs_assign 단위 테스트 (n=10) ===\n")

    # 9) Trivial : 모든 드론이 이미 목표에 위치
    n = 5
    pts = np.array([[float(i * 10), 0.0] for i in range(n)])
    asgn = np.arange(n)
    paths_trivial = cbs_assign(pts, pts, asgn, max_steps=30)
    assert paths_trivial is not None
    assert detect_conflict(paths_trivial) is None
    print("[PASS] trivial: 각 드론 이미 목표 위치, 충돌 없음")

    # 10) 2-드론 교차 충돌 해결
    d2 = np.array([[0.0, 0.0], [6.0, 0.0]])
    t2 = np.array([[6.0, 0.0], [0.0, 0.0]])
    a2 = np.array([0, 1])
    paths2 = cbs_assign(d2, t2, a2, max_steps=30)
    assert paths2 is not None and detect_conflict(paths2) is None
    print("[PASS] 2-드론 교차 충돌 해결 성공")

    # 11) n=10 랜덤 케이스
    rng = np.random.default_rng(42)
    n = 10
    d10 = rng.uniform(0, 20, (n, 2))
    t10 = rng.uniform(0, 20, (n, 2))
    a10 = hungarian_assign(build_cost_matrix(d10, t10))

    t_start = time.time()
    paths10 = cbs_assign(d10, t10, a10, max_steps=80)
    elapsed = time.time() - t_start

    if paths10 is not None:
        assert detect_conflict(paths10) is None
        avg_len = sum(len(p) for p in paths10.values()) / n
        print(f"[PASS] n=10 CBS 탐색 성공  ({elapsed:.2f}s, 평균 경로 길이 {avg_len:.1f})")
    else:
        print(f"[WARN] n=10 CBS 탐색 실패 — max_steps/max_iterations 조정 필요 ({elapsed:.2f}s)")

    print("\n모든 테스트 완료.")
