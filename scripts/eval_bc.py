#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行为克隆实战验收：克隆坦克 vs 老师 / vs 随机大脑。
验收线：val_acc 已过（94%+，懒人基线 86.1%）；这里是终验——行为像不像老师。

用法：
  eval_bc.py                # 各打 100 局无头对战，出战绩表
  eval_bc.py --record X.gif # 录一局 克隆vs老师 的 GIF
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena, MLP, record_gif  # noqa: E402
from rule_teacher import RuleTeacher  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bc-v1.npz")


class TeacherBrain:
    """把规则老师包成 arena 大脑接口（act 返回 (动作, None)）。"""
    def __init__(self):
        self.t = RuleTeacher()

    def act(self, obs, sample=False, rng=None):
        return self.t.decide(obs), None


class NetBrain:
    """自动识别存档类型：含 wl 键 → LSTM 脑（有记忆），否则 MLP 脑。"""
    def __init__(self, path):
        if "wl" in set(np.load(path).keys()):
            from lstm_net import LSTMBrain
            self._lstm = LSTMBrain(path)
            self.net = None
        else:
            self._lstm = None
            self.net = MLP.load(path)

    def reset(self):
        if self._lstm:
            self._lstm.reset()

    def act(self, obs, sample=False, rng=None):
        if self._lstm:
            return self._lstm.act(obs)
        return self.net.act(obs, sample=False)[0], None  # 实战一律 argmax


def match(brain_a, brain_b, games, seed, destructible=False, no_cover=False, obs_v2=False):
    """brain_a 固定蓝方。返回 (a胜, 平, b胜, a命中率, 平均步数)。"""
    wa = wb = dr = 0
    hits_a = shots_a = 0
    steps = []
    for g in range(games):
        arena = Arena(seed=seed + g, destructible=destructible, no_cover=no_cover,
                      obs_v2=obs_v2)
        for b in (brain_a, brain_b):            # 有记忆的脑每局清零（LSTMBrain）
            if hasattr(b, "reset"):
                b.reset()
        while not arena.done:
            acts = []
            for tank, foe, brain in zip(arena.tanks, arena.tanks[::-1], (brain_a, brain_b)):
                acts.append(brain.act(arena.observe(tank, foe))[0] if tank.alive else 0)
            arena.step(acts)
        if arena.winner is None:
            dr += 1
        elif arena.winner is arena.tanks[0]:
            wa += 1
        else:
            wb += 1
        hits_a += arena.stats["hits"][0]
        shots_a += arena.stats["shots"][0]
        steps.append(arena.step_n)
    return wa, dr, wb, hits_a / max(1, shots_a), float(np.mean(steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--destructible", action="store_true", help="掩体可摧毁（v15 几何手术）")
    ap.add_argument("--no-cover", action="store_true", help="裸场地（v19）")
    ap.add_argument("--obs-v2", action="store_true", help="40 维观测（敌方速度+冷却）")
    ap.add_argument("--record", default=None)
    ap.add_argument("--live", action="store_true", help="桌面窗口实时演示：克隆 vs 随机")
    ap.add_argument("--weights", default=WEIGHTS, help="大脑权重路径（默认 bc-v1，可指 rl-v1）")
    ap.add_argument("--vs", default="random", help="实机演示/录像的对手：random/teacher/莽夫/狙击手/游击手/绕后/拆迁队长")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    ap2_destructible = getattr(args, "destructible", False)
    net = NetBrain(args.weights)
    teacher = TeacherBrain()
    rngbrain = type("R", (), {"act": lambda self, o, sample=False, rng=None:
                              (int(np.random.default_rng().integers(0, 5)), None)})()

    # 剧本小队对手（bot_roster，v18）
    from bot_roster import BOTS
    _bot_map = {n: cls for n, cls in BOTS}
    if args.vs in _bot_map:
        class _BotBrain:
            def __init__(self, bot):
                self._b = bot

            def act(self, obs, sample=False, rng=None):
                return self._b.decide(obs), None
        foe, foe_name = _BotBrain(_bot_map[args.vs]()), args.vs
    else:
        foe = teacher if args.vs == "teacher" else rngbrain
        foe_name = "老师" if args.vs == "teacher" else "随机"

    if args.live:
        import pygame
        from arena import draw, W, H, run_episode
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(f"智能坦克 · 神经网络(蓝) vs {foe_name}(红)")
        clock = pygame.time.Clock()
        arena = Arena(seed=args.seed, destructible=args.destructible, no_cover=args.no_cover, obs_v2=args.obs_v2)
        running = True

        def hook(a, info):
            nonlocal running
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    a.done = True
            t0, t1 = a.tanks
            draw(screen, a, f"神经网络(蓝) hp {t0.hp} | {foe_name}(红) hp {t1.hp} | step {a.step_n}")
            pygame.display.flip()
            clock.tick(args.fps)

        while running:
            arena.reset()
            for b in (net, foe):                # 有记忆的脑每局清零
                if hasattr(b, "reset"):
                    b.reset()
            run_episode(arena, (net, foe), render_hook=hook)
            if not running:
                break
            pygame.time.wait(1500)
        pygame.quit()
        return

    if args.record:
        arena = Arena(seed=args.seed, destructible=args.destructible, no_cover=args.no_cover, obs_v2=args.obs_v2)
        record_gif(arena, (net, foe), args.record, fps=30)
        print(f"GIF 已录（克隆=蓝 vs 老师=红）: {args.record}")
        return

    print(f"各 {args.games} 局（克隆固定蓝方）：")
    wa, dr, wb, hr, st = match(net, teacher, args.games, 0)
    print(f"克隆 vs 老师 : 胜{wa} 平{dr} 负{wb} | 克隆命中率 {hr*100:.1f}% | 平均 {st:.0f} 步")
    wa, dr, wb, hr, st = match(net, rngbrain, args.games, 10000)
    print(f"克隆 vs 随机 : 胜{wa} 平{dr} 负{wb} | 克隆命中率 {hr*100:.1f}% | 平均 {st:.0f} 步")
    wa, dr, wb, hr, st = match(teacher, rngbrain, args.games, 20000)
    print(f"老师 vs 随机 : 胜{wa} 平{dr} 负{wb} | 老师命中率 {hr*100:.1f}% | 平均 {st:.0f} 步（参考）")


if __name__ == "__main__":
    main()
