#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 冒烟①：RuleTeacher 在 random_cover + destructible 场景录 400 局示范，
帧级监督把 256 宽 Net 热身到 val_acc ≥70%，产物存 data/warm-v25.npz。
（参考 train_bc.main 流程的临时脚本；老师只读 obs，对任意布局有效）"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arena import Arena, MAX_STEPS  # noqa: E402
from rule_teacher import RuleTeacher  # noqa: E402
from train_bc import Net, Adam, evaluate, infer_hidden  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "warm-v25.npz")
GAMES = 400
MAX_GAME_STEPS = 600          # 单局封顶（老师内战可能拖到 1800 步，没必要全录）
WIDTH = 256


def collect():
    X, y = [], []
    t0 = time.time()
    for g in range(GAMES):
        arena = Arena(seed=10_000 + g, destructible=True, random_cover=True)
        teachers = (RuleTeacher(), RuleTeacher())
        steps = 0
        while not arena.done and steps < MAX_GAME_STEPS:
            acts = []
            for i, (tank, foe) in enumerate(zip(arena.tanks, arena.tanks[::-1])):
                obs = arena.observe(tank, foe)
                if tank.alive:
                    a = teachers[i].decide(obs)
                    X.append(obs)
                    y.append(a)
                acts.append(a if tank.alive else 0)
            arena.step(acts)
            steps += 1
    print(f"录制完成：{GAMES} 局 → {len(y):,} 帧（{time.time()-t0:.0f}s）")
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def main():
    X, y = collect()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_val = len(y) // 20
    va, tr = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    counts = np.bincount(ytr, minlength=5)
    cw = len(ytr) / (5.0 * counts)
    print(f"train {len(ytr):,} / val {len(yva):,} | 类别分布 {counts} | 类别权重 {np.round(cw,2)}")

    net = Net(seed=0, hidden1=WIDTH, hidden2=WIDTH)
    assert net.sizes == (37, WIDTH, WIDTH, 5)
    opt = Adam(net, lr=1e-3)
    batch = 512
    t0 = time.time()
    for ep in range(1, 9):
        perm = np.random.default_rng(1000 + ep).permutation(len(ytr))
        tot = 0.0
        for b in range(0, len(ytr), batch):
            sl = perm[b:b + batch]
            P, cache = net.forward(Xtr[sl])
            tot += net.loss(P, ytr[sl], cw) * len(sl)
            dWs, dbs = net.backward(P, ytr[sl], cw, cache)
            opt.step(net, dWs, dbs)
        vl, acc, _ = evaluate(net, Xva, yva, cw)
        print(f"epoch {ep:2d} | train_loss {tot/len(ytr):.4f} | val_loss {vl:.4f} | val_acc {acc*100:.2f}% | {time.time()-t0:.0f}s")

    # 回读校验：npz 宽度可被 infer_hidden 还原
    np.savez(OUT, **{f"w{i}": w for i, w in enumerate(net.Ws)},
             **{f"b{i}": b for i, b in enumerate(net.bs)})
    d = np.load(OUT)
    print(f"回读 infer_hidden={infer_hidden(d)}（期望 (256, 256)），已存 {OUT}")
    assert infer_hidden(d) == (WIDTH, WIDTH)
    _, acc, _ = evaluate(net, Xva, yva, cw)
    assert acc >= 0.70, f"val_acc {acc*100:.2f}% 未达 70% 门禁"
    print(f"WARM-UP OK: val_acc {acc*100:.2f}% ≥ 70%")


if __name__ == "__main__":
    main()
