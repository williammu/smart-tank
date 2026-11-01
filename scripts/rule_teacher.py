#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则机器人老师（入门级）——路线 C 第①步：人造标签的生产者。

设计要点（对应 notes/004）：
  - 只看 37 维观测向量做决策，不碰游戏内部状态
    → 顺便实证观测完备性：观测够不够用，老师说了算；
  - 决策树刻意保持"入门级"：会瞄准开火、会调整距离，
    但不躲弹、不算提前量——给 RL 阶段留出超越空间。
"""
import math

# 观测向量索引（与 arena.Arena.observe 的布局对齐）
ME_X, ME_Y, FX, FY, ME_HP = 0, 1, 2, 3, 4
FOE_X, FOE_Y, EFX, EFY, FOE_HP = 5, 6, 7, 8, 9
DX, DY, DIST = 10, 11, 12
BDX, BDY, BVX, BVY = 13, 14, 15, 16
READY, TIME_LEFT, CONST = 17, 18, 19
OBS_LOS = 36  # 掩体感 20-35，视线遮挡 bit 在最后

# 动作编号（与 arena.ACTIONS 对齐）
TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, FIRE = 0, 1, 2, 3, 4

ARENA_W, ARENA_H = 640, 480  # 与 arena.py 一致（dy 按 H、dx 按 W 归一化，算角度前要还原）


class RuleTeacher:
    """入门级规则老师。

    决策树：
      没对准 → 转向（朝减小角差的方向）
      对准 + 视线畅通 + 冷却就绪 → 开火
      对准但开不了火 → 调整距离（太远前进 / 贴脸后退）
    """

    AIM_TOL_MAX = 0.05    # 瞄准容差上限（弧度，≈2.9°）
    PREFER_DIST = 0.30    # 偏好交战距离（归一化，≈240px）
    CLOSE_DIST = 0.10     # 贴脸阈值（≈80px，再近才后退——0.15 太怂，后退样本占 30%）

    def decide(self, obs):
        """obs: (37,) numpy 数组 → 动作编号 0-4"""
        # 当前朝向（cos/sin 编码，atan2 还原成角度）
        cur = math.atan2(obs[FY], obs[FX])
        # 期望朝向：敌人在哪个方向（dx/dy 分别按 W/H 归一化过，还原纵横比再算角度）
        want = math.atan2(obs[DY] * ARENA_H, obs[DX] * ARENA_W)
        # 最短角差，折回 [-π, π]——正是 cos/sin 编码讨论里的"圆周接缝"，规则代码也要处理
        diff = (want - cur + math.pi) % (2 * math.pi) - math.pi

        # 容差随距离缩放：子弹命中要求 垂直偏移 < 坦克半径+弹半径≈15px，
        # 远了就要瞄更准（notes/002 的几何账），否则"开火"标签里混进大量必miss的坏示范
        dist_px = obs[DIST] * math.hypot(ARENA_W, ARENA_H)
        aim_tol = min(self.AIM_TOL_MAX, math.atan2(18, max(60.0, dist_px)))

        aimed = abs(diff) < aim_tol
        if aimed and obs[OBS_LOS] < 0.5 and obs[READY] > 0.5:
            return FIRE
        if not aimed:
            return TURN_RIGHT if diff > 0 else TURN_LEFT
        # 已瞄准但开不了火（冷却中或视线被挡）→ 调整距离
        if obs[DIST] > self.PREFER_DIST:
            return FORWARD
        if obs[DIST] < self.CLOSE_DIST:
            return BACKWARD
        return FORWARD  # 默认缓慢逼近（入门级不站桩等冷却）


if __name__ == "__main__":
    # 自检：构造几个"明知道该干什么"的局面，验证决策树
    import numpy as np

    t = RuleTeacher()

    def make_obs(diff_angle=0.0, dist=0.5, ready=1.0, los=0.0):
        obs = np.zeros(37, dtype=np.float32)
        obs[FX], obs[FY] = 1.0, 0.0                      # 朝右
        obs[DX] = math.cos(diff_angle) * dist * 800 / ARENA_W
        obs[DY] = math.sin(diff_angle) * dist * 800 / ARENA_H
        obs[DIST], obs[READY], obs[OBS_LOS] = dist, ready, los
        return obs

    names = ["左转", "右转", "前进", "后退", "开火"]
    cases = [
        ("对准+畅通+就绪 → 开火", make_obs(0.0), FIRE),
        ("敌人在上方 → 转向", make_obs(0.5), None),       # 转向即可，方向由符号定
        ("对准但冷却中+太远 → 前进", make_obs(0.0, dist=0.6, ready=0.0), FORWARD),
        ("对准但视线被挡 → 前进逼近", make_obs(0.0, los=1.0), FORWARD),
        ("贴脸 → 后退", make_obs(0.0, dist=0.08, ready=0.0), BACKWARD),
    ]
    for name, obs, expect in cases:
        a = t.decide(obs)
        ok = "✓" if expect is None or a == expect else f"✗ 期望{names[expect]}"
        print(f"{ok} {name} → {names[a]}")
