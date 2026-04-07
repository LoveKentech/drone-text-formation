"""
시뮬레이션 결과를 시각화하는 모듈.
matplotlib을 사용해 드론 이동 궤적 애니메이션을 생성하고 output/에 저장한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _bounds_from_points(*point_arrays, margin: float = 5.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """여러 (n,2) 배열로부터 xlim, ylim (y 반전용) 튜플을 계산한다."""
    xs = np.concatenate([np.asarray(a)[:, 0].ravel() for a in point_arrays])
    ys = np.concatenate([np.asarray(a)[:, 1].ravel() for a in point_arrays])
    return (xs.min() - margin, xs.max() + margin), (ys.max() + margin, ys.min() - margin)


def _setup_ax(
    ax: plt.Axes,
    frames: np.ndarray,
    targets: np.ndarray,
    title: str,
    margin: float = 5.0,
) -> None:
    """axes 범위·비율·제목을 설정한다. y축은 픽셀 좌표계에 맞게 반전."""
    (x0, x1), (y0, y1) = _bounds_from_points(frames.reshape(-1, 2), targets, margin=margin)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate(
    frames: np.ndarray,
    targets: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    사전 계산된 frames를 순서대로 재생하는 애니메이션을 생성한다.

    blitting 모드로 성능을 최적화한다.

    Parameters
    ----------
    frames    : ndarray (max_steps, n, 2) — 타임라인
    targets   : ndarray (n, 2)            — 목표 좌표 (축 범위만; 마커는 안 그림)
    title     : str                        — 제목 ("Hungarian" / "CBS" 등)
    save_path : str | None                — GIF 저장 경로 (None 이면 plt.show())

    Returns
    -------
    FuncAnimation 객체

    Example
    -------
    # animate(frames, targets, title="Hungarian", save_path="out/hungarian.gif")
    """
    max_steps = frames.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    _setup_ax(ax, frames, targets, title)

    # 목표 좌표는 축 범위에만 반영하고, 빨간 × 등 목표 마커는 그리지 않는다.

    # 드론 위치 — 동적
    drone_sc = ax.scatter(
        frames[0, :, 0], frames[0, :, 1],
        c="steelblue", s=25, zorder=3, label="Drones",
    )
    time_text = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    def init():
        drone_sc.set_offsets(frames[0])
        time_text.set_text("")
        return drone_sc, time_text

    def update(t: int):
        drone_sc.set_offsets(frames[t])
        time_text.set_text(f"t = {t}")
        return drone_sc, time_text

    anim = animation.FuncAnimation(
        fig, update,
        frames=max_steps,
        init_func=init,
        blit=True,
        interval=50,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=20)
        plt.close(fig)
    else:
        plt.show()

    return anim


def animate_morph(
    frames: np.ndarray,
    targets_start: np.ndarray,
    targets_end: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    interval_ms: int = 35,
    label_start: str = "Start",
    draw_start_cloud: bool = False,
    drone_size: float = 38.0,
) -> animation.FuncAnimation:
    """
    글자 모핑 시각화. 드론(frames)만 애니메이션한다.

    ``draw_start_cloud=False``(기본)이면 시작 글자는 **드론 점**으로만 보이고
    주황색 겹침을 그리지 않는다 (t=0에서 LOVE 형태가 드론과 일치).
    ``draw_start_cloud=True``이면 ``targets_start`` 점군을 추가로 표시한다.
    목표 글자(``targets_end``)는 축 범위에만 반영하고 빨간 ×로는 그리지 않는다.
    """
    max_steps = frames.shape[0]
    fig, ax = plt.subplots(figsize=(10, 7))
    (x0, x1), (y0, y1) = _bounds_from_points(
        frames.reshape(-1, 2), targets_start, targets_end, margin=8.0
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if draw_start_cloud:
        ax.scatter(
            targets_start[:, 0], targets_start[:, 1],
            c="#ff7f50", s=18, alpha=0.55, linewidths=0, zorder=1,
            label=label_start,
        )
    # 목표(HELLO) 좌표는 축 범위 계산에만 쓰고, 빨간 ×는 표시하지 않는다.

    drone_sc = ax.scatter(
        frames[0, :, 0], frames[0, :, 1],
        c="steelblue", s=drone_size, zorder=4, edgecolors="white", linewidths=0.4,
        label="Drones",
    )
    time_text = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    def init():
        drone_sc.set_offsets(frames[0])
        time_text.set_text("")
        return drone_sc, time_text

    def update(t: int):
        drone_sc.set_offsets(frames[t])
        time_text.set_text(f"t = {t} / {max_steps - 1}")
        return drone_sc, time_text

    anim = animation.FuncAnimation(
        fig, update,
        frames=max_steps,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=max(1, int(round(1000 / interval_ms))))
        plt.close(fig)
    else:
        plt.show()

    return anim


def compare_animate(
    frames_hungarian: np.ndarray,
    frames_cbs: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    Hungarian과 CBS 두 알고리즘 결과를 나란히 subplot으로 비교한다.

    왼쪽 : Hungarian, 오른쪽 : CBS.
    두 애니메이션이 같은 타임스텝을 공유하며 동기화된다.

    Parameters
    ----------
    frames_hungarian : ndarray (T_h, n, 2)
    frames_cbs       : ndarray (T_c, n, 2)
    targets          : ndarray (n, 2)
    save_path        : str | None — GIF 저장 경로

    Returns
    -------
    FuncAnimation 객체

    Example
    -------
    # compare_animate(f_h, f_c, targets, save_path="out/compare.gif")
    """
    h_steps   = frames_hungarian.shape[0]
    c_steps   = frames_cbs.shape[0]
    max_steps = max(h_steps, c_steps)

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    _setup_ax(ax_h, frames_hungarian, targets, "Hungarian")
    _setup_ax(ax_c, frames_cbs,       targets, "CBS")

    # 목표 마커는 표시하지 않음 (축 범위는 targets로 이미 설정됨)

    sc_h = ax_h.scatter(
        frames_hungarian[0, :, 0], frames_hungarian[0, :, 1],
        c="steelblue", s=25, zorder=3,
    )
    sc_c = ax_c.scatter(
        frames_cbs[0, :, 0], frames_cbs[0, :, 1],
        c="steelblue", s=25, zorder=3,
    )
    tt_h = ax_h.text(0.02, 0.96, "", transform=ax_h.transAxes, fontsize=9, va="top")
    tt_c = ax_c.text(0.02, 0.96, "", transform=ax_c.transAxes, fontsize=9, va="top")

    fig.tight_layout()

    def init():
        sc_h.set_offsets(frames_hungarian[0])
        sc_c.set_offsets(frames_cbs[0])
        tt_h.set_text("")
        tt_c.set_text("")
        return sc_h, sc_c, tt_h, tt_c

    def update(t: int):
        sc_h.set_offsets(frames_hungarian[min(t, h_steps - 1)])
        sc_c.set_offsets(frames_cbs[min(t, c_steps - 1)])
        tt_h.set_text(f"t = {t}")
        tt_c.set_text(f"t = {t}")
        return sc_h, sc_c, tt_h, tt_c

    anim = animation.FuncAnimation(
        fig, update,
        frames=max_steps,
        init_func=init,
        blit=True,
        interval=50,
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=20)
        plt.close(fig)
    else:
        plt.show()

    return anim
