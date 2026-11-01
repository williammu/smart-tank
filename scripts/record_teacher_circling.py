#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录「老师 vs 老师」的绕柱片段（两遍法：先扫出最佳绕柱段，再重跑只录那一段）。

用法：
  python record_teacher_circling.py --scan 20 --seed0 9000 --out out/demo-teacher-circle.gif
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena import Arena  # noqa: E402
from eval_bc import TeacherBrain  # noqa: E402


def nearest_wall(a, t):
    best = None
    for _, (ox, oy, ow, oh) in a.live_obstacles():
        cx, cy = ox + ow / 2, oy + oh / 2
        d = math.hypot(t.x - cx, t.y - cy)
        if best is None or d < best[2]:
            best = (cx, cy, d)
    return best


def step_once(a, b1, b2):
    me, ot = a.tanks[0], a.tanks[1]
    x1 = b1.act(a.observe(me, ot))[0]
    x2 = b2.act(a.observe(ot, me))[0]
    a.step([x1, x2])


def scan(seed, near=60.0):
    """返回该局所有贴墙段的 (起始步, 结束步, 累计角度, 掩体中心)。"""
    a = Arena(seed=seed, destructible=True)
    b1, b2 = TeacherBrain(), TeacherBrain()
    seg = None
    segs = []
    while not a.done:
        me = a.tanks[0]
        w = nearest_wall(a, me)
        if w and w[2] < near:
            th = math.atan2(me.y - w[1], me.x - w[0])
            if seg is None or math.hypot(w[0] - seg["cx"], w[1] - seg["cy"]) > 25:
                if seg:
                    segs.append(seg)
                seg = dict(cx=w[0], cy=w[1], th=th, acc=0.0, start=a.step_n, end=a.step_n)
            else:
                d = th - seg["th"]
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                seg["acc"] += abs(d)
                seg["th"] = th
                seg["end"] = a.step_n
        else:
            if seg:
                segs.append(seg); seg = None
        step_once(a, b1, b2)
    if seg:
        segs.append(seg)
    return segs


def record(seed, start, end, out, every=2, scale=(480, 360)):
    import pygame
    from PIL import Image
    pygame.init()
    from arena import draw, W, H
    surface = pygame.Surface((W, H))
    a = Arena(seed=seed, destructible=True)
    b1, b2 = TeacherBrain(), TeacherBrain()
    frames = []
    while not a.done:
        step_once(a, b1, b2)
        if start <= a.step_n <= end and (a.step_n - start) % every == 0:
            draw(surface, a, f"teacher(BLUE) vs teacher(RED) | step {a.step_n} | blue hp {a.tanks[0].hp} red hp {a.tanks[1].hp}")
            img = Image.fromarray(np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)))
            frames.append(img.resize(scale, Image.LANCZOS))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)
    print(f"{out} | {len(frames)} 帧 | {os.path.getsize(out)//1024}KB | seed {seed} 步 {start}~{end}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    best = None
    for g in range(args.scan):
        seed = args.seed0 + g
        segs = scan(seed)
        for s in segs:
            deg = math.degrees(s["acc"])
            if best is None or deg > best[0]:
                best = (deg, seed, s)
        if segs:
            mx = max(math.degrees(s["acc"]) for s in segs)
            print(f"seed {seed}: 最长段 {mx:.0f}°（{len(segs)} 段）")
        else:
            print(f"seed {seed}: 无贴墙段")

    deg, seed, s = best
    print(f"\n最佳：seed {seed} 段 {deg:.0f}° 步 {s['start']}~{s['end']}")
    pad = 40
    record(seed, max(1, s["start"] - pad), s["end"] + 15, args.out, every=args.every)


if __name__ == "__main__":
    main()
