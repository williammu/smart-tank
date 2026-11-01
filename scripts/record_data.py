#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据录制器——路线 C 第①步下半场：让规则老师打比赛，把决策录成标签数据集。

产出：data/dataset-v1.npz
  X_train/X_val: (N,37) float32  局面（观测向量）
  y_train/y_val: (N,)   int64    标签（动作编号 0-4）

数据卫生三条军规（notes/004）：
  ① train/val 按【局】切分，不按行随机切（同局相邻帧近乎重复，随机切 = 考题漏进练习册）
  ② 打印类别分布（防"永远喊前进"的假聪明）
  ③ 只录活坦克的决策

用法：python record_data.py --games 100 --seed 0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena
from rule_teacher import RuleTeacher

ACTION_NAMES = ["左转", "右转", "前进", "后退", "开火"]


def record(games, seed):
    """混合对手：一半 老师vs老师（双视角都录），一半 老师vs随机策略（只录老师）。

    军规④：随机对手的决策绝不能录成标签（垃圾进垃圾出），它只是陪练。
    混合对手是为了破"镜像僵局"——两个同策略老师对打会陷入逼近→后退的循环舞步，
    数据被一种行为模式垄断（第一版实测 后退 占 40%）。
    """
    teacher = RuleTeacher()
    episodes = []  # 每局一个 (X, y) 列表对，切分要按局来
    total_hits, total_shots, lengths = 0, 0, []

    for g in range(games):
        arena = Arena(seed=seed + g)
        vs_teacher = g % 2 == 0
        rng_act = np.random.default_rng(seed * 100000 + g)
        xs, ys = [], []
        while not arena.done:
            acts = []
            for ti, (tank, foe) in enumerate(zip(arena.tanks, arena.tanks[::-1])):
                if not tank.alive:
                    acts.append(0)
                    continue
                obs = arena.observe(tank, foe)
                if vs_teacher or ti == 0:
                    a = teacher.decide(obs)  # 老师的决策 → 录为标签
                    xs.append(obs)
                    ys.append(a)
                else:
                    a = int(rng_act.integers(0, 5))  # 随机陪练：不产标签
                acts.append(a)
            arena.step(acts)
        episodes.append((np.stack(xs), np.array(ys, dtype=np.int64)))
        total_hits += sum(arena.stats["hits"])
        total_shots += sum(arena.stats["shots"])
        lengths.append(arena.step_n)
    return episodes, total_hits, total_shots, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset-v1.npz")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    episodes, hits, shots, lengths = record(args.games, args.seed)

    # 军规①：按局切分（打乱局序后 90/10）
    rng = np.random.default_rng(123)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * args.val_ratio))
    val_idx, train_idx = order[:n_val], order[n_val:]

    def pack(indices):
        X = np.concatenate([episodes[i][0] for i in indices])
        y = np.concatenate([episodes[i][1] for i in indices])
        return X, y

    X_tr, y_tr = pack(train_idx)
    X_va, y_va = pack(val_idx)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)

    # 军规②：类别分布
    print(f"\n===== 数据集 {args.out} =====")
    print(f"样本: train {len(y_tr):,} / val {len(y_va):,}（{args.games} 局，按局切分）")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]}: {np.mean(y_tr == a) * 100:5.1f}%", end="")
    print(f"\n老师战绩: 命中率 {hits}/{shots} = {hits / max(1, shots) * 100:.1f}%"
          f" | 平均每局 {np.mean(lengths):.0f} 步")
    size = os.path.getsize(args.out) / 1e6
    print(f"文件: {size:.1f} MB（npz 压缩）")

    # 抽 3 行实物看看
    print("\n===== 样本实物（前 3 行）=====")
    for i in range(3):
        print(f"y={y_tr[i]}({ACTION_NAMES[y_tr[i]]})  X[:6]={X_tr[i, :6].round(2).tolist()} ...")


if __name__ == "__main__":
    main()
