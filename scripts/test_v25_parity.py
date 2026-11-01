#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v25 parity + 随机掩体验证。

① parity：同一 seed 下，git HEAD 的旧 arena（random_cover 语义不存在）
   与新 arena（random_cover=False）各跑 200 步确定性对局（argmax MLP 脑），
   逐步对比双方 tank 状态，必须完全一致。
② random_cover=True：验证 1~4 块（实现目标 2~4，兜底允许 2）、宽高 40~140、
   互不重叠且间距 ≥30、不压四角出生禁区、obs 恒 37 维；
   destructible 下同一掩体吃 3 发炮弹被拆除。
"""
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NEW = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("arena_new", os.path.join(HERE, "arena.py")))
NEW.__loader__.exec_module(NEW)

# 从 git HEAD 取旧版 arena.py（parity 基准）
_blob = subprocess.run(["git", "-C", os.path.join(HERE, "..", "..", "..", ".."), "show",
                        "HEAD:work/dnn-learning/subprojects/smart-tank/scripts/arena.py"],
                       capture_output=True, check=True).stdout
_old_path = os.path.join(HERE, "_arena_old_head.py")
with open(_old_path, "wb") as f:
    f.write(_blob)
OLD = importlib.util.module_from_spec(importlib.util.spec_from_file_location("arena_old", _old_path))
OLD.__loader__.exec_module(OLD)


def run_steps(ArenaCls, seed, steps):
    """确定性对局：argmax MLP 脑（sample=False），逐步快照双方 tank 状态。"""
    arena = ArenaCls(seed=seed)
    brains = [NEW.MLP(seed=seed + 1), NEW.MLP(seed=seed + 1000)]
    snap = []
    for _ in range(steps):
        if arena.done:
            break
        acts = []
        for tank, foe, brain in zip(arena.tanks, arena.tanks[::-1], brains):
            if not tank.alive:
                acts.append(0)
                continue
            obs = arena.observe(tank, foe)
            assert obs.shape == (37,), f"obs 维度 {obs.shape}"
            acts.append(brain.act(obs, sample=False)[0])
        arena.step(acts)
        snap.append([(t.x, t.y, t.angle, t.hp, t.alive) for t in arena.tanks]
                    + [(b.x, b.y, b.alive) for b in arena.bullets]
                    + [(arena.step_n, arena.done, tuple(arena.obs_hp))])
    return snap


def test_parity():
    for seed in (0, 1, 7, 42, 12345):
        a = run_steps(OLD.Arena, seed, 200)
        b = run_steps(NEW.Arena, seed, 200)
        assert len(a) == len(b), f"seed {seed}: 步数不同 {len(a)} vs {len(b)}"
        for i, (sa, sb) in enumerate(zip(a, b)):
            assert sa == sb, f"seed {seed} step {i + 1} 不一致:\n旧 {sa}\n新 {sb}"
        print(f"parity seed={seed}: {len(a)} 步逐步一致 ✅")


def test_random_cover():
    import math
    for seed in range(50):
        ar = NEW.Arena(seed=seed, random_cover=True)
        obs_list = ar.obstacles
        assert 2 <= len(obs_list) <= 4, f"seed {seed}: 块数 {len(obs_list)}"
        for i, (x, y, w, h) in enumerate(obs_list):
            assert 40 <= w <= 140 and 40 <= h <= 140, f"seed {seed} 掩体{i} 尺寸 {w}x{h}"
            assert 0 <= x and x + w <= NEW.W and 0 <= y and y + h <= NEW.H
            assert not NEW._rect_hits_spawn_zone((x, y, w, h)), f"seed {seed} 掩体{i} 压出生禁区"
            for j, o in enumerate(obs_list):
                if j <= i:
                    continue
                assert NEW._rects_clear((x, y, w, h), o), f"seed {seed}: 掩体 {i},{j} 重叠/间距<30"
        # obs 恒 37 维
        o = ar.observe(ar.tanks[0], ar.tanks[1])
        assert o.shape == (37,), f"seed {seed}: obs {o.shape}"
        # 随机布局本身不含空槽掩体特征（destructible=False 时 hp=1 全存活）
        assert len(ar.live_obstacles()) == len(obs_list)
    print("random_cover 50 seeds: 块数/尺寸/间距/出生禁区/obs 37 维 全部通过 ✅")


def test_destructible_3_shots():
    seed = 3
    ar = NEW.Arena(seed=seed, destructible=True, random_cover=True)
    (x, y, w, h), = [ob for ob in ar.obstacles][:1]
    target = ar.obstacles[0]
    cx, cy = target[0] + target[2] / 2, target[1] + target[3] / 2
    assert ar.obs_hp[0] == 3
    for shot in range(3):
        ar.bullets = [NEW.Bullet(cx - 10, cy, 5.0, 0.0, ar.tanks[0])]
        ar.step([0, 0])
        assert ar.bullets == [], f"第 {shot + 1} 炮未命中掩体"
        assert ar.obs_hp[0] == 2 - shot, f"命中后 hp={ar.obs_hp[0]}"
    assert ar.live_obstacles()[0][0] != 0 or all(i != 0 for i, _ in ar.live_obstacles()), \
        "3 炮后掩体 0 仍存活"
    assert ar.obs_hp[0] == 0
    # 被拆掩体的特征槽位归零
    o = ar.observe(ar.tanks[0], ar.tanks[1])
    assert list(o[20:24]) == [0.0, 0.0, 0.0, 0.0], f"拆除后槽位特征 {o[20:24]}"
    print("destructible 随机掩体: 3 炮拆除 + 槽位归零 ✅")


if __name__ == "__main__":
    test_parity()
    test_random_cover()
    test_destructible_3_shots()
    os.remove(_old_path)
    print("ALL PASS")
