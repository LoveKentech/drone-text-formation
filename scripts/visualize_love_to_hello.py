"""
LOVE → HELLO 모핑 GIF.

기본: 격자 이동 + 다중 패스 충돌 회피 (``greedy_timeline_then_resolve``).
선형 보간만 쓰려면 ``--mode linear`` (충돌 무시).

  MPLBACKEND=Agg python scripts/visualize_love_to_hello.py
  MPLBACKEND=Agg python scripts/visualize_love_to_hello.py --mode linear
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CONTOUR_METHODS, DEFAULT_METHOD
from src.coordinate import generate_coordinates
from src.hungarian_collision import greedy_timeline_then_resolve, hungarian_assignment
from src.timeline import compute_timeline_linear, pad_timeline_hold
from src.visualize import animate_morph


def main() -> None:
    p = argparse.ArgumentParser(description="LOVE → HELLO 모핑 GIF")
    p.add_argument("--n", type=int, default=80, help="드론 수")
    p.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    p.add_argument("--method", choices=CONTOUR_METHODS, default=DEFAULT_METHOD)
    p.add_argument(
        "--mode",
        choices=("resolve", "linear"),
        default="resolve",
        help="resolve: 격자+충돌회피 | linear: 직선보간(충돌 없음)",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--pause-start", type=int, default=28)
    p.add_argument("--pause-end", type=int, default=28)
    p.add_argument("--output", default="output/morph_LOVE_to_HELLO.gif")
    args = p.parse_args()

    ms = args.max_steps or (400 if args.mode == "resolve" else 180)

    pos_love = generate_coordinates("LOVE", args.n, args.size, method=args.method)
    hello = generate_coordinates("HELLO", args.n, args.size, method=args.method)
    drones = pos_love.astype(float)
    _, assignment = hungarian_assignment(drones, hello)

    if args.mode == "resolve":
        frames, _, stats, _ = greedy_timeline_then_resolve(
            drones, hello, assignment, max_steps=ms
        )
        print(
            f"[충돌회피] 잔여={stats['remaining']}  "
            f"패스={stats['passes_used']}  감지누적={stats['total_detected']}"
        )
    else:
        frames = compute_timeline_linear(drones, hello, assignment, max_steps=ms)

    frames = pad_timeline_hold(
        frames,
        hold_start=max(0, args.pause_start),
        hold_end=max(0, args.pause_end),
    )

    _od = os.path.dirname(os.path.abspath(args.output))
    if _od:
        os.makedirs(_od, exist_ok=True)
    title = f"LOVE→HELLO | {args.method} | {args.mode} | n={args.n}"
    animate_morph(frames, pos_love, hello, title=title, save_path=args.output)
    print(f"[저장] {os.path.abspath(args.output)}  frames={frames.shape[0]}")


if __name__ == "__main__":
    main()
