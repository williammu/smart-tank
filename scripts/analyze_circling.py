#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绕柱行为量化检测器（v27 实验配套）。

定义（可复现、可对照）：
  对学习方（蓝方）逐步找最近的存活掩体；仅统计「距该掩体 < --near」的连续段；
  角度 θ = atan2(tank.y - wall.cy, tank.x - wall.cx)，累计 |Δθ|；
  一段内累计 ≥ --min-deg（默认 270°，即 3/4 圈）记一次「绕柱事件」；
  同时统计该段内「视线被掩体挡住」的比例（LoS bit=1），用来区分
  「绕柱找射击窗口」和「贴着墙乱走」。

输出：每局的圈数/总角度/遮挡占比 + 汇总 + 可选 GIF。

用法：
  python analyze_circling.py --weights data/rl-v27.npz --games 30
  python analyze_circling.py --weights data/rl-v23.npz --games 30   # 对照组
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena  # noqa: E402
from eval_bc import NetBrain, TeacherBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def nearest_wall(arena, tank):
    """返回最近存活掩体的 (cx, cy, dist)，无掩体则 None。"""
    best = None
    for _, (ox, oy, ow, oh) in arena.live_obstacles():
        cx, cy = ox + ow / 2, oy + oh / 2
        d = math.hypot(tank.x - cx, tank.y - cy)
        if best is None or d < best[2]:
            best = (cx, cy, d)
    return best


def run_episode(arena, brain, foe, near=60.0, min_deg=270.0):
    """打一局，返回该局的绕柱统计。"""
    seg = None            # 当前贴墙段：{cx,cy,theta_prev,acc_deg,steps,blocked}
    events, total_deg, near_steps, blocked_steps = 0, 0.0, 0, 0
    max_seg_deg = 0.0
    feat_used = feat_n = 0            # 清零新增3维后动作变化的次数（特征使用率）
    shots = shots_on_cd = 0          # 开火次数 / 其中发生在对手冷却中的次数
    cd_sum = 0.0
    cd_all_sum = 0.0                 # 全程对手冷却均值（基准线）
    popouts = popout_cd = 0          # 遮挡→通视 的次数 / 当时的对手冷却和
    prev_blocked = None

    while not arena.done:
        me, other = arena.tanks[0], arena.tanks[1]
        obs = arena.observe(me, other)
        blocked = float(obs[36])          # LoS bit（37/40 维都在 36）
        w = nearest_wall(arena, me)

        if w is not None and w[2] < near:
            theta = math.atan2(me.y - w[1], me.x - w[0])
            near_steps += 1
            blocked_steps += blocked
            if seg is None or math.hypot(w[0] - seg["cx"], w[1] - seg["cy"]) > 25:
                seg = dict(cx=w[0], cy=w[1], theta_prev=theta, cum=0.0,
                           lo=0.0, hi=0.0, steps=0, blocked=0.0)
            else:
                d = theta - seg["theta_prev"]
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                # 关键：累加「有符号」增量并取净跨度（max−min），
                # 否则原地左右摆动会被 |d| 累加成假的「绕圈」
                seg["cum"] += d
                seg["lo"] = min(seg["lo"], seg["cum"])
                seg["hi"] = max(seg["hi"], seg["cum"])
                seg["theta_prev"] = theta
            seg["steps"] += 1
            seg["blocked"] += blocked
            span = seg["hi"] - seg["lo"]
            max_seg_deg = max(max_seg_deg, math.degrees(span))
            if math.degrees(span) >= min_deg:
                events += 1
                total_deg += math.degrees(span)
                seg = None                # 结算一段，防止重复计数
        else:
            if seg is not None and math.degrees(seg["hi"] - seg["lo"]) >= 120:
                max_seg_deg = max(max_seg_deg, math.degrees(seg["hi"] - seg["lo"]))
            seg = None

        cd_all_sum += other.cooldown
        if prev_blocked == 1 and blocked == 0:
            popouts += 1
            popout_cd += other.cooldown
        prev_blocked = blocked
        a = brain.act(obs, sample=False)[0]
        if len(obs) == 40:
            o0 = list(obs); o0[37] = o0[38] = o0[39] = 0.0
            feat_n += 1
            if brain.act(o0, sample=False)[0] != a:
                feat_used += 1
        if a == 4 and me.cooldown == 0:          # 这一步真的开火
            shots += 1
            if other.cooldown > 0:
                shots_on_cd += 1
            cd_sum += other.cooldown
        b = foe.act(arena.observe(other, me), sample=False)[0]
        arena.step([a, b])

    return dict(events=events, total_deg=total_deg, max_seg_deg=max_seg_deg,
                near_steps=near_steps, blocked_ratio=(blocked_steps / near_steps) if near_steps else 0.0,
                steps=arena.step_n, winner=(arena.winner.name if arena.winner else "平"),
                hits=arena.stats["hits"][0],
                shots=shots, shots_on_cd=shots_on_cd,
                cd_at_fire=(cd_sum / shots) if shots else 0.0,
                cd_base=(cd_all_sum / max(1, arena.step_n)),
                popouts=popouts, feat_used=feat_used, feat_n=feat_n,
                cd_at_popout=(popout_cd / popouts) if popouts else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--destructible", action="store_true")
    ap.add_argument("--near", type=float, default=60.0, help="距掩体多少像素内算贴墙")
    ap.add_argument("--min-deg", type=float, default=270.0, help="累计转多少度算绕柱")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    brain = NetBrain(args.weights)
    foe = TeacherBrain()
    obs_v2 = brain.net.weights[0].shape[1] == 40

    rows = []
    for g in range(args.games):
        arena = Arena(seed=args.seed + g, destructible=args.destructible, obs_v2=obs_v2)
        for b in (brain, foe):
            if hasattr(b, "reset"):
                b.reset()
        rows.append(run_episode(arena, brain, foe, near=args.near, min_deg=args.min_deg))

    ev = sum(r["events"] for r in rows)
    games_with = sum(1 for r in rows if r["events"] > 0)
    tot_near = sum(r["near_steps"] for r in rows)
    blk = sum(r["blocked_ratio"] * r["near_steps"] for r in rows) / max(1, tot_near)
    print(f"\n=== {os.path.basename(args.weights)}（{'40 维' if obs_v2 else '37 维'}）{args.games} 局 ===")
    print(f"绕柱事件（≥{args.min_deg:.0f}°）: {ev} 次 | 出现绕柱的对局: {games_with}/{args.games}")
    print(f"最长单段转角: {max(r['max_seg_deg'] for r in rows):.0f}°")
    print(f"贴墙步数占比: {tot_near/max(1,sum(r['steps'] for r in rows))*100:.1f}%")
    print(f"贴墙时视线被挡占比: {blk*100:.1f}%")
    sh = sum(r["shots"] for r in rows); shc = sum(r["shots_on_cd"] for r in rows)
    print(f"开火 {sh} 次 | 其中踩在对手冷却上 {shc} 次（{shc/max(1,sh)*100:.1f}%）"
          f" | 开火时对手冷却均值 {sum(r['cd_at_fire']*r['shots'] for r in rows)/max(1,sh):.1f} 步")
    tot_steps = sum(r["steps"] for r in rows)
    print(f"对手冷却基准均值 {sum(r['cd_base']*r['steps'] for r in rows)/max(1,tot_steps):.1f} 步"
          f" | 通视瞬间 {sum(r['cd_at_popout']*r['popouts'] for r in rows)/max(1,sum(r['popouts'] for r in rows)):.1f} 步"
          f"（{sum(r['popouts'] for r in rows)} 次）")
    fu = sum(r["feat_used"] for r in rows); fn = sum(r["feat_n"] for r in rows)
    if fn:
        print(f"特征使用率（清零新增3维→动作改变）: {fu}/{fn} = {fu/fn*100:.1f}%")
    print(f"命中: {sum(r['hits'] for r in rows)} | 胜: {sum(1 for r in rows if r['winner']=='蓝')}"
          f" | 平: {sum(1 for r in rows if r['winner']=='平')}")
    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)
        print("明细已写", args.json)


if __name__ == "__main__":
    main()
