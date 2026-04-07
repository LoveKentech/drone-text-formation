"""
LOVE 포메이션(초기 위치)에서 HELLO 목표로 이동하는 애니메이션을 생성한다.

- 좌표: ``contour`` 는 윤곽선별 샘플(다중 윤곽 버그 수정). 글자를 더 두껍게 채우려면 ``--method poisson``.
- t=0: 드론이 LOVE 점에 정확히 올라가 있음(시작 주황 겹침은 끔).
- 앞/뒤 ``--pause-*`` 프레임만큼 정지(LOVE 유지 → 이동 → HELLO 유지).

  MPLBACKEND=Agg python scripts/visualize_love_to_hello.py --n 80 --size medium
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import CONTOUR_METHODS, DEFAULT_METHOD, MAX_STEPS
from src.coordinate import generate_coordinates
from src.hungarian_collision import greedy_timeline_then_resolve, hungarian_assignment
from src.timeline import (
    compute_timeline_hungarian,
    compute_timeline_linear,
    pad_timeline_hold,
)
from src.visualize import animate_morph


def main() -> None:
    p = argparse.ArgumentParser(description="LOVE → HELLO 드론 모핑 GIF")
    p.add_argument("--n", type=int, default=80, help="드론 수 (LOVE/HELLO 동일 샘플 수)")
    p.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="래스터 스케일",
    )
    p.add_argument(
        "--method",
        choices=CONTOUR_METHODS,
        default=DEFAULT_METHOD,
        help="좌표 샘플링(contour=윤곽, poisson=내부 채움, grid=격자)",
    )
    p.add_argument(
        "--mode",
        choices=("linear", "grid"),
        default="linear",
        help=(
            "linear: 직선 보간. "
            "grid: 격자 sign + 충돌 회피."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="이동 구간 프레임 수 (기본: linear=180, grid=max(400, MAX_STEPS))",
    )
    p.add_argument(
        "--pause-start",
        type=int,
        default=28,
        help="LOVE 형태로 정지할 초반 프레임 수",
    )
    p.add_argument(
        "--pause-end",
        type=int,
        default=28,
        help="HELLO 도착 후 정지할 말판 프레임 수",
    )
    p.add_argument(
        "--output",
        default="output/morph_LOVE_to_HELLO.gif",
        help="저장 GIF 경로",
    )
    args = p.parse_args()

    if args.max_steps is None:
        ms = 180 if args.mode == "linear" else max(400, MAX_STEPS)
    else:
        ms = args.max_steps

    pos_love = generate_coordinates("LOVE", args.n, args.size, method=args.method)
    targets_hello = generate_coordinates("HELLO", args.n, args.size, method=args.method)

    drones = pos_love.astype(float).copy()
    _, assignment = hungarian_assignment(drones, targets_hello)

    if args.mode == "linear":
        frames = compute_timeline_linear(drones, targets_hello, assignment, max_steps=ms)
    else:
        raw = compute_timeline_hungarian(drones, targets_hello, assignment, max_steps=ms)
        frames, _, _, _ = greedy_timeline_then_resolve(
            drones, targets_hello, assignment, max_steps=ms, frames_raw=raw
        )

    if args.mode == "linear":
        assert np.allclose(frames[0], drones, atol=1e-3)
        assert np.allclose(frames[-1], targets_hello[assignment], atol=1e-2)

    frames = pad_timeline_hold(
        frames,
        hold_start=max(0, args.pause_start),
        hold_end=max(0, args.pause_end),
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    title = (
        f"LOVE → HELLO | n={args.n} | {args.size} | {args.method} | {args.mode}"
    )
    animate_morph(
        frames,
        pos_love,
        targets_hello,
        title=title,
        save_path=args.output,
        label_start="LOVE (ref)",
        draw_start_cloud=False,
    )
    print(
        f"[저장] {os.path.abspath(args.output)}  "
        f"frames={frames.shape[0]}  mode={args.method}/{args.mode}"
    )


if __name__ == "__main__":
    main()
