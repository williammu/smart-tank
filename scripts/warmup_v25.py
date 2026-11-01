#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 热身：RuleTeacher 在 random_cover + destructible 场景录 400 局示范，
帧级监督把 256 宽 Net（37→256→256→5）热身到 val acc ≥70%，存 data/warm-v25.npz。

用法：./.venv/bin/python scripts/warmup_v25.py [--games 400] [--epochs 12]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena import Arena  # noqa: E402
from rule_teacher import RuleTeacher  # noqa: E402
from train_bc import Net, Adam, evaluate  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "warm-v25.npz")


def record(games):
    """老师(蓝) vs 六流派随机抽签对手(红)，逐帧记 (obs, 老师动作)。"""
    from bot_roster import BOTS
    teacher = RuleTeacher()
    X, Y = [], []
    for g in range(games):
        arena = Arena(seed=100000 + g, destructible=True, random_cover=True)
        foe = BOTS[int(np.random.default_rng(g).integers(0, len(BOTS)))][1]()
        while not arena.done:
            acts = []
            for tank, f_obj in zip(arena.tanks, arena.tanks[::-1]):
                if not tank.alive:
                    acts.append(0)
                    continue
                obs = arena.observe(tank, f_obj)
                acts.append(teacher.decide(obs) if tank is arena.tanks[0] else foe.decide(obs))
            arena.step(acts)
            if arena.tanks[0].alive:
                X.append(arena.observe(arena.tanks[0], arena.tanks[1]))
                Y.append(acts[0])
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--target-acc", type=float, default=0.70)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    t0 = time.time()
    X, Y = record(args.games)
    print(f"录制完成：{len(Y):,} 帧（{args.games} 局）| {time.time()-t0:.0f}s")
    # 分层打散后 95/5 切 train/val
    idx = np.random.default_rng(7).permutation(len(Y))
    n_va = max(1, len(Y) // 20)
    va, tr = idx[:n_va], idx[n_va:]
    Xtr, ytr, Xva, yva = X[tr], Y[tr], X[va], Y[va]
    counts = np.bincount(ytr, minlength=5)
    cw = len(ytr) / (5.0 * counts)
    print(f"train {len(ytr):,} / val {len(yva):,} | 动作分布 {counts} | 类别权重 {np.round(cw, 2)}")

    net = Net(seed=0, hidden1=256, hidden2=256)
    opt = Adam(net, lr=args.lr)
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        perm = np.random.default_rng(1000 + ep).permutation(len(ytr))
        tot = 0.0
        for b in range(0, len(ytr), args.batch):
            sl = perm[b:b + args.batch]
            P, cache = net.forward(Xtr[sl])
            tot += net.loss(P, ytr[sl], cw) * len(sl)
            dWs, dbs = net.backward(P, ytr[sl], cw, cache)
            opt.step(net, dWs, dbs)
        vl, va_acc, vr = evaluate(net, Xva, yva, cw)
        best_acc = max(best_acc, va_acc)
        print(f"epoch {ep:2d} | train_loss {tot/len(ytr):.4f} | val_loss {vl:.4f} "
              f"| val_acc {va_acc*100:.2f}% | 召回 " +
              " ".join(f"{r*100:.0f}" for r in vr) + f" | {time.time()-t0:.0f}s")
        if va_acc >= args.target_acc:
            break
    np.savez(OUT, **{f"w{i}": w for i, w in enumerate(net.Ws)},
             **{f"b{i}": b for i, b in enumerate(net.bs)})
    print(f"最佳 val_acc {best_acc*100:.2f}%（目标 ≥{args.target_acc*100:.0f}%）→ {OUT}")
    assert best_acc >= args.target_acc, "未达热身精度线"


if __name__ == "__main__":
    main()
