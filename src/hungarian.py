"""
헝가리안 알고리즘(최적 이분 매칭)을 이용해 드론과 목표 좌표를 1:1로 할당하는 모듈.
scipy.optimize.linear_sum_assignment를 사용하여 총 이동 비용(유클리드 거리 합)을 최소화한다.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MIN_SAFE_DIST


def build_cost_matrix(drones: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    드론 초기 위치와 목표 좌표 사이의 유클리드 거리로 비용 행렬을 생성한다.

    Parameters
    ----------
    drones : np.ndarray, shape (n, 2)
        드론 초기 위치 배열. 각 행은 [x, y].
    targets : np.ndarray, shape (n, 2)
        목표 좌표 배열. 각 행은 [x, y].

    Returns
    -------
    np.ndarray, shape (n, n)
        비용 행렬 C. C[i][j] = drones[i]에서 targets[j]까지의 유클리드 거리.

    Example
    -------
    # drones  = np.array([[0, 0], [1, 0], [0, 1]])
    # targets = np.array([[3, 4], [1, 1], [2, 2]])
    # C = build_cost_matrix(drones, targets)
    # print(C[0, 0])  # 5.0  (0,0) → (3,4)
    """
    # 브로드캐스팅으로 (n, n, 2) 차이 행렬 계산 후 L2 노름
    diff = drones[:, np.newaxis, :] - targets[np.newaxis, :, :]  # (n, n, 2)
    cost = np.linalg.norm(diff, axis=2)                          # (n, n)
    return cost


def hungarian_assign(cost_matrix: np.ndarray) -> np.ndarray:
    """
    비용 행렬에 헝가리안 알고리즘을 적용하여 최적 1:1 할당을 구한다.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n, n)
        build_cost_matrix()가 반환한 비용 행렬.

    Returns
    -------
    np.ndarray, shape (n,)
        assignment[i] = 드론 i에 할당된 목표 인덱스.
        즉, 드론 i는 targets[assignment[i]]로 이동한다.

    Example
    -------
    # C = build_cost_matrix(drones, targets)
    # assignment = hungarian_assign(C)
    # total_cost = C[np.arange(len(assignment)), assignment].sum()
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind는 항상 [0, 1, ..., n-1] 순서이므로 col_ind만 반환
    assignment = col_ind
    return assignment


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    print("=== build_cost_matrix 단위 테스트 ===\n")

    # 1) 거리 계산 정확성
    drones  = np.array([[0.0, 0.0], [3.0, 0.0]])
    targets = np.array([[3.0, 4.0], [0.0, 0.0]])
    C = build_cost_matrix(drones, targets)
    assert C.shape == (2, 2), f"shape 오류: {C.shape}"
    assert math.isclose(C[0, 0], 5.0), f"C[0,0] 오류: {C[0,0]}"  # (0,0)→(3,4) = 5
    assert math.isclose(C[0, 1], 0.0), f"C[0,1] 오류: {C[0,1]}"  # (0,0)→(0,0) = 0
    assert math.isclose(C[1, 0], 4.0), f"C[1,0] 오류: {C[1,0]}"  # (3,0)→(3,4) = 4
    assert math.isclose(C[1, 1], 3.0), f"C[1,1] 오류: {C[1,1]}"  # (3,0)→(0,0) = 3
    print("[PASS] 2×2 비용 행렬 거리 계산 정확")

    # 2) 대각선 = 0 (드론과 목표가 동일 위치)
    pts = np.random.rand(10, 2) * 100
    C2 = build_cost_matrix(pts, pts)
    assert np.allclose(np.diag(C2), 0.0), "동일 위치 거리가 0이 아님"
    print("[PASS] 동일 위치 드론-목표 거리 = 0")

    # 3) 대칭성 (drones ↔ targets 교환 시 전치 행렬)
    d = np.random.rand(5, 2) * 50
    t = np.random.rand(5, 2) * 50
    assert np.allclose(build_cost_matrix(d, t), build_cost_matrix(t, d).T)
    print("[PASS] 비용 행렬 전치 대칭성 확인")

    print("\n=== hungarian_assign 단위 테스트 ===\n")

    # 4) 자명한 최적 매칭 — 대각선이 0인 경우 assignment = [0,1,...,n-1]
    n = 5
    pts = np.random.rand(n, 2) * 100
    C3 = build_cost_matrix(pts, pts)
    asgn = hungarian_assign(C3)
    assert list(asgn) == list(range(n)), f"자명 매칭 실패: {asgn}"
    print("[PASS] 동일 위치 배열에서 항등 매칭 반환")

    # 5) 2×2 명시적 최적 검증
    #    drones=[(0,0),(3,0)], targets=[(3,4),(0,0)]
    #    최적: 드론0→target1(비용0), 드론1→target0(비용4) → 합 4
    #    vs.  드론0→target0(비용5), 드론1→target1(비용3) → 합 8
    C4 = build_cost_matrix(
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.array([[3.0, 4.0], [0.0, 0.0]])
    )
    asgn4 = hungarian_assign(C4)
    total = C4[np.arange(2), asgn4].sum()
    assert math.isclose(total, 4.0), f"최적 비용 오류: {total}"
    print("[PASS] 2×2 최적 매칭 비용 = 4.0")

    # 6) 반환 shape 및 유일성 (각 목표는 정확히 1번 할당)
    n = 50
    d50 = np.random.rand(n, 2) * 200
    t50 = np.random.rand(n, 2) * 200
    a50 = hungarian_assign(build_cost_matrix(d50, t50))
    assert a50.shape == (n,), f"shape 오류: {a50.shape}"
    assert len(set(a50)) == n, "중복 할당 존재"
    print("[PASS] n=50 할당 결과 shape 및 유일성 확인")

    # 7) MIN_SAFE_DIST 참조 확인 (현재 모듈에서 import되었는지)
    assert MIN_SAFE_DIST == 2.0
    print(f"[INFO] MIN_SAFE_DIST = {MIN_SAFE_DIST} (config.py에서 정상 import)")

    print("\n모든 테스트 통과.")
