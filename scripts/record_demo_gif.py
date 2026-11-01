#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录一局 GIF（指定权重 + 种子 + 对手），用于人工核验行为。

用法：
  python record_demo_gif.py --weights data/rl-v27.npz --seed 7003 --out out/demo-v27-s7003.gif
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena import Arena, draw, W, H  # noqa: E402
from eval_bc import NetBrain, TeacherBrain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3, help="每几步存一帧")
    ap.add_argument("--destructible", action="store_true", default=True)
    args = ap.parse_args()

    import pygame
    from PIL import Image
    pygame.init()
    surface = pygame.Surface((W, H))

    brain = NetBrain(args.weights)
    obs_v2 = brain.net.weights[0].shape[1] == 40
    foe = TeacherBrain()
    arena = Arena(seed=args.seed, destructible=args.destructible, obs_v2=obs_v2)
    frames = []
    while not arena.done:
        me, other = arena.tanks[0], arena.tanks[1]
        o1 = arena.observe(me, other)
        o2 = arena.observe(other, me)
        a = brain.act(o1, sample=False)[0]
        b = foe.act(o2)[0]
        arena.step([a, b])
        if arena.step_n % args.every == 0:
            draw(surface, arena, f"seed {args.seed} | step {arena.step_n} | 蓝 hp {arena.tanks[0].hp}")
            frames.append(Image.fromarray(np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
    print(f"{args.out} | {len(frames)} 帧 | {os.path.getsize(args.out)//1024}KB | "
          f"{'40 维' if obs_v2 else '37 维'} | winner={arena.winner.name if arena.winner else '平'} "
          f"| 步数 {arena.step_n}")


if __name__ == "__main__":
    main()
