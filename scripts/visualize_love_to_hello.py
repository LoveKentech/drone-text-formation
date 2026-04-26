"""
텍스트 A → 텍스트 B 모핑 GIF (기본 LOVE → HELLO).

기본: 격자 이동 + 다중 패스 충돌 회피 (``greedy_timeline_then_resolve``).
선형 보간만 쓰려면 ``--mode linear`` (충돌 무시).

  MPLBACKEND=Agg python scripts/visualize_love_to_hello.py
  MPLBACKEND=Agg python scripts/visualize_love_to_hello.py --start LOVE --end KENTECH
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CONTOUR_METHODS, DEFAULT_METHOD
from src.coordinate import generate_coordinates
from src.hungarian_collision import compute_assignment, greedy_timeline_then_resolve
from src.timeline import compute_timeline_linear, pad_timeline_hold
from src.visualize import animate_morph


def main() -> None:
    p = argparse.ArgumentParser(description="텍스트 A → B 모핑 GIF")
    p.add_argument("--start", default="LOVE", help="출발 포메이션 문자열")
    p.add_argument("--end", default="HELLO", help="목표 포메이션 문자열")
    p.add_argument("--n", type=int, default=80, help="드론 수")
    p.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    p.add_argument("--method", choices=CONTOUR_METHODS, default=DEFAULT_METHOD)
    p.add_argument(
        "--mode",
        choices=("resolve", "linear"),
        default="resolve",
        help="resolve: 격자+충돌회피 | linear: 직선보간(충돌 없음)",
    )
    p.add_argument(
        "--no-smooth",
        action="store_true",
        help="resolve 모드에서 궤적 스무딩(떨림 완화) 끄기 — 이전 동작에 가깝게",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--pause-start", type=int, default=28)
    p.add_argument("--pause-end", type=int, default=28)
    p.add_argument(
        "--output",
        default=None,
        help="저장 경로 (미지정 시 output/morph_{start}_to_{end}.gif)",
    )
    args = p.parse_args()

    out_path = args.output or f"output/morph_{args.start}_to_{args.end}.gif"
    ms = args.max_steps or (400 if args.mode == "resolve" else 180)

    pos_start = generate_coordinates(args.start, args.n, args.size, method=args.method)
    pos_end = generate_coordinates(args.end, args.n, args.size, method=args.method)
    drones = pos_start.astype(float)
    _, assignment = compute_assignment(drones, pos_end)

    if args.mode == "resolve":
        frames, _, stats, _ = greedy_timeline_then_resolve(
            drones, pos_end, assignment, max_steps=ms, smooth_visual=not args.no_smooth
        )
        print(
            f"[충돌회피] 잔여={stats['remaining']}  "
            f"패스={stats['passes_used']}  감지누적={stats['total_detected']}"
        )
    else:
        frames = compute_timeline_linear(drones, pos_end, assignment, max_steps=ms)

    frames = pad_timeline_hold(
        frames,
        hold_start=max(0, args.pause_start),
        hold_end=max(0, args.pause_end),
    )

    _od = os.path.dirname(os.path.abspath(out_path))
    if _od:
        os.makedirs(_od, exist_ok=True)
    title = f"{args.start}→{args.end} | {args.method} | {args.mode} | n={args.n}"
    animate_morph(frames, pos_start, pos_end, title=title, save_path=out_path)
    print(f"[저장] {os.path.abspath(out_path)}  frames={frames.shape[0]}")


if __name__ == "__main__":
    main()
