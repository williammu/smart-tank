#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剧本机器人花名册（v18：对手多样性，联赛的"剧本版"）。

每个机器人一个鲜明战术流派，全部只看 37 维观测、实现 decide(obs)→动作 接口：
  RuleTeacher 原版·中距缠斗（接近+瞄准+调距）
  Rusher     莽夫·贴脸冲锋（永远全速逼近，畅通就轰）
  Sniper     狙击手·原地蹲守（永不位移，炮塔扫射——蹲坑流本人，学生必须学会拔钉子）
  Kiter      游击手·远程风筝（敌近我退、敌远我瞄，交战距离比老师远一半）
  Flanker    绕后·侧翼包抄（中远距离开垂直机动，近了才转正对敌）
  Demolisher 拆迁队长·先拆墙再打人（视线被挡就瞄准掩体开炮）
"""
import math

from rule_teacher import (RuleTeacher, ME_X, ME_Y, FX, FY, FOE_X, FOE_Y,  # noqa: F401
                          DX, DY, DIST, READY, OBS_LOS,
                          TURN_LEFT, TURN_RIGHT, FORWARD, BACKWARD, FIRE,
                          ARENA_W, ARENA_H)


def _diff_to_foe(obs):
    """当前朝向与敌人方向的最短角差（rad，正=敌在左）。"""
    cur = math.atan2(obs[FY], obs[FX])
    want = math.atan2(obs[DY] * ARENA_H, obs[DX] * ARENA_W)
    return (want - cur + math.pi) % (2 * math.pi) - math.pi


def _aim_tol(obs):
    dist_px = obs[DIST] * math.hypot(ARENA_W, ARENA_H)
    return min(0.05, math.atan2(18, max(60.0, dist_px)))


class Rusher:
    """莽夫：永不后退，畅通就轰。"""
    name = "莽夫"

    def decide(self, obs):
        diff = _diff_to_foe(obs)
        if abs(diff) < _aim_tol(obs) and obs[OBS_LOS] < 0.5 and obs[READY] > 0.5:
            return FIRE
        if abs(diff) >= _aim_tol(obs):
            return TURN_RIGHT if diff > 0 else TURN_LEFT
        return FORWARD


class Sniper:
    """狙击手：永不位移，炮塔左右扫射，锁定即开火（蹲坑流本人）。"""
    name = "狙击手"

    def __init__(self):
        self._flip = 0

    def decide(self, obs):
        diff = _diff_to_foe(obs)
        if abs(diff) < _aim_tol(obs) and obs[OBS_LOS] < 0.5 and obs[READY] > 0.5:
            return FIRE
        self._flip ^= 1
        return TURN_RIGHT if (diff > 0) == bool(self._flip) else TURN_LEFT   # 原地扫射（零位移）


class Kiter:
    """游击手：交战距离拉远一半，敌进我退。"""
    name = "游击手"
    PREFER, CLOSE = 0.45, 0.32

    def decide(self, obs):
        diff = _diff_to_foe(obs)
        aimed = abs(diff) < _aim_tol(obs)
        if aimed and obs[OBS_LOS] < 0.5 and obs[READY] > 0.5:
            return FIRE
        if not aimed:
            return TURN_RIGHT if diff > 0 else TURN_LEFT
        if obs[DIST] < self.CLOSE:
            return BACKWARD      # 保持面对敌人后撤（风筝）
        if obs[DIST] > self.PREFER:
            return FORWARD
        return BACKWARD if obs[READY] < 0.5 else FORWARD   # 冷却中拉开，装填好压迫


class Flanker:
    """绕后：中远距离走垂直切线，近了转正对敌。"""
    name = "绕后"
    ENGAGE = 0.32

    def decide(self, obs):
        if obs[DIST] < self.ENGAGE:
            return RuleTeacher().decide(obs)   # 近身战同老师
        cur = math.atan2(obs[FY], obs[FX])
        foe_ang = math.atan2(obs[DY] * ARENA_H, obs[DX] * ARENA_W)
        want = foe_ang + math.pi / 2           # 固定逆时针绕
        diff = (want - cur + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > 0.10:
            return TURN_RIGHT if diff > 0 else TURN_LEFT
        return FORWARD


class Demolisher:
    """拆迁队长：视线被挡就瞄准最近的存活掩体开炮（v15 物理），畅通则同老师。"""
    name = "拆迁队长"

    def decide(self, obs):
        if obs[OBS_LOS] < 0.5:
            return RuleTeacher().decide(obs)
        # 找最近的存活掩体（特征组 20..35：dx,dy,hw,hh ×4）
        best, best_d = None, 1e9
        for k in range(4):
            dx, dy, hw, hh = obs[20 + 4 * k:24 + 4 * k]
            if hw <= 0:
                continue
            d = math.hypot(dx * ARENA_W, dy * ARENA_H)
            if d < best_d:
                best_d, best = d, (dx, dy)
        if best is None:
            return RuleTeacher().decide(obs)
        cur = math.atan2(obs[FY], obs[FX])
        want = math.atan2(best[1] * ARENA_H, best[0] * ARENA_W)
        diff = (want - cur + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) > 0.05:
            return TURN_RIGHT if diff > 0 else TURN_LEFT
        return FIRE if obs[READY] > 0.5 else FORWARD


# 花名册：(名字, 工厂)——原版老师领衔
BOTS = [("老师", RuleTeacher), ("莽夫", Rusher), ("狙击手", Sniper),
        ("游击手", Kiter), ("绕后", Flanker), ("拆迁队长", Demolisher)]


if __name__ == "__main__":
    import numpy as np
    for name, cls in BOTS:
        bot = cls()
        obs = np.zeros(37, dtype=np.float32)
        obs[FX], obs[FY], obs[DX], obs[DIST], obs[READY] = 1.0, 0.0, 0.3, 0.5, 1.0
        print(f"{name:5s} decide→{bot.decide(obs)}")
