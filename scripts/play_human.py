#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人机对战：你用键盘操控红方坦克，挑战神经网络（蓝方，默认 v22 冠军）。

操作（按住即持续生效）：
  ←/→ 或 A/D : 炮塔左转/右转
  ↑/↓ 或 W/S : 前进/后退
  空格        : 开火（有冷却，顶部条显示装填进度）
  R           : 本局重开  Q/ESC: 退出

用法：
  DISPLAY=:1 ./.venv/bin/python scripts/play_human.py --weights data/rl-v22.npz [--no-cover] [--fps 60]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from arena import Arena, draw, W, H, MAX_STEPS  # noqa: E402
from eval_bc import NetBrain  # noqa: E402

TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, FIRE = 0, 1, 2, 3, 4


class HumanBrain:
    """键盘大脑：act() 读取 pygame 键盘状态（需外部每帧 pump 事件）。"""

    def __init__(self):
        self.fire_queued = False

    def act(self, obs, sample=False, rng=None):
        import pygame
        keys = pygame.key.get_pressed()
        if self.fire_queued:          # 空格 KEYDOWN 抢占一帧：按下的瞬间就开炮
            self.fire_queued = False
            return FIRE, None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return TURN_LEFT, None
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return TURN_RIGHT, None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            return FORWARD, None
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return BACKWARD, None
        if keys[pygame.K_SPACE]:      # 长按空格：松开方向键时也能持续开火
            return FIRE, None
        return FORWARD, None   # 无输入时缓行（动作空间没有"原地不动"）


def _save_gif(gif_frames):
    """保存 GIF（每 3 帧抽 1 帧，防内存爆）。"""
    if not gif_frames:
        print("没有帧可保存")
        return
    from PIL import Image
    out_path = os.path.join(HERE, "..", "out", "demo-human-vs-ai.gif")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil = [Image.fromarray(f) for f in gif_frames[::3]]
    pil[0].save(out_path, save_all=True, append_images=pil[1:],
                duration=50, loop=0, optimize=True)
    print(f"GIF 已保存: {out_path} ({len(pil)}帧, {os.path.getsize(out_path)//1024}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "..", "data", "rl-v22.npz"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-cover", action="store_true", help="裸场地（v22 的原生战场）")
    ap.add_argument("--destructible", action="store_true")
    ap.add_argument("--obs-v2", action="store_true", help="40 维观测（敌方速度+冷却）")
    args = ap.parse_args()

    import pygame
    net = NetBrain(args.weights)
    human = HumanBrain()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("人机对战 · 你(红,键盘) vs 神经网络(蓝) | ←→转向 ↑↓移动 空格开火 R重开 Q退出")
    clock = pygame.time.Clock()
    arena = Arena(seed=args.seed, destructible=args.destructible, no_cover=args.no_cover,
                  obs_v2=args.obs_v2)

    wins = {"you": 0, "net": 0, "draw": 0}
    gif_frames = []

    running = True
    while running:
        arena.reset()
        human.fire_queued = False
        over = False
        while not over and running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    _save_gif(gif_frames)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        human.fire_queued = True
                    elif e.key == pygame.K_r:
                        over = True          # 重开本局
                        _save_gif(gif_frames)
                        gif_frames.clear()
                    elif e.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                        _save_gif(gif_frames)
            if not running:
                break
            # 一步
            actions = []
            for ti, (tank, foe, brain) in enumerate(zip(arena.tanks, arena.tanks[::-1],
                                                        (net, human))):
                if not tank.alive:
                    actions.append(0)
                    continue
                obs = arena.observe(tank, foe)
                a, _ = brain.act(obs, sample=(ti == 0), rng=arena.rng)
                actions.append(a)
            arena.step(actions)
            t0, t1 = arena.tanks
            draw(screen, arena, f"你(红) hp {t1.hp} | 神经网(蓝) hp {t0.hp} | "
                                f"step {arena.step_n}/{MAX_STEPS} | "
                                f"战绩 你{wins['you']}:平{wins['draw']}:网{wins['net']}")
            pygame.display.flip()
            # GIF 帧捕获
            import pygame as _pg
            if _pg.key.get_pressed()[_pg.K_g]:
                pass  # 按住 G 录制
            if arena.step_n % 3 == 0:
                frame_data = pygame.surfarray.array3d(screen)
                gif_frames.append(np.transpose(frame_data, (1, 0, 2)))
            clock.tick(args.fps)
            if arena.done:
                over = True
                w = arena.winner
                if w is None:
                    wins["draw"] += 1
                elif w.name == "蓝":
                    wins["net"] += 1
                else:
                    wins["you"] += 1
                label = {None: "平局", "蓝": "神经网络胜", "红": "你赢了！"}[w.name if w else None]
                print(f"结果: {label} | 步数 {arena.step_n} | "
                      f"你 {arena.stats['hits'][1]}中/{arena.stats['shots'][1]}射 "
                      f"| 网 {arena.stats['hits'][0]}中/{arena.stats['shots'][0]}射 | "
                      f"总战绩 你{wins['you']}:平{wins['draw']}:网{wins['net']}", flush=True)
    pygame.quit()
    print(f"最终战绩：你 {wins['you']} 胜 / 平 {wins['draw']} / 神经网 {wins['net']} 胜", flush=True)


if __name__ == "__main__":
    main()
