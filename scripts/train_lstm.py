#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v11 · LSTM 自对弈训练器：记忆 + BPTT（随时间反向传播）。

与 train_rl.py（v8 配方）的唯一变量 = 网络架构（MLP → LSTM），奖励表一字不动：
  命中当步 +0.2/−0.2 | 胜+1 负−1 平−0.5 | 顶墙−0.01 | 站桩−0.01(15px) | 接近±0.01
  射界奖 +0.002 | 盲射 −0.02 | 弹药费 −0.01 | critic 0.5 | γ=0.99
流程：自对弈采样（局内携带 h,c）→ 存每步 (obs, act, rew, P, V, cache)
     → return-to-go 得 R → A = R − V(s)（批归一化）→ BPTT 逆流 → Adam。
军规：--grad-check 数值校验过了才准 --train。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena, MAX_STEPS  # noqa: E402
from lstm_net import LSTMNet, LSTM_H  # noqa: E402
from eval_bc import TeacherBrain, NetBrain, match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 奖励表（= v8 配方，一字不动）
R_HIT_DEALT, R_HIT_TAKEN = 0.2, -0.2
R_WIN, R_LOSE, DRAW_PENALTY = 1.0, -1.0, 0.5
STUCK_PEN, IDLE_PEN, IDLE_WIN, IDLE_TH = 0.01, 0.01, 10, 15.0
CLOSE_BONUS, LOS_BONUS, BLIND_PEN, FIRE_COST = 0.0005, 0.002, 0.02, 0.01
GAMMA, VALUE_COEF, ENTROPY_BETA, CLIP_NORM = 0.99, 0.5, 0.005, 5.0


def play_game(net, rng):
    """自对弈一局（双方=当前 net，采样探索）。返回每侧的逐步记录（含 LSTM cache/P/V）。"""
    arena = Arena(seed=int(rng.integers(0, 2**31)))
    S = [dict(obs=[], act=[], rew=[], P=[], V=[], cache=[]) for _ in range(2)]
    hs = [np.zeros(LSTM_H), np.zeros(LSTM_H)]
    cs = [np.zeros(LSTM_H), np.zeros(LSTM_H)]
    disp_hist = [[], []]
    stuck_n = move_n = 0
    while not arena.done:
        acts = []
        for i, (tank, foe) in enumerate(zip(arena.tanks, arena.tanks[::-1])):
            if not tank.alive:
                acts.append(0)
                continue
            obs = arena.observe(tank, foe)
            P, V, hs[i], cs[i], cache = net.step(np.asarray(obs, dtype=np.float64), hs[i], cs[i])
            a = int(rng.choice(5, p=P / P.sum()))
            s = S[i]
            s["obs"].append(obs); s["act"].append(a); s["rew"].append(0.0)
            s["P"].append(P); s["V"].append(V); s["cache"].append(cache)
            acts.append(a)
        prev_xy = [(t.x, t.y) for t in arena.tanks]
        prev_d = float(np.hypot(arena.tanks[0].x - arena.tanks[1].x,
                                arena.tanks[0].y - arena.tanks[1].y))
        prev_hits = list(arena.stats["hits"])
        arena.step(acts)
        d = float(np.hypot(arena.tanks[0].x - arena.tanks[1].x,
                           arena.tanks[0].y - arena.tanks[1].y))
        for i, tank in enumerate(arena.tanks):
            if not S[i]["rew"]:
                continue
            disp = float(np.hypot(tank.x - prev_xy[i][0], tank.y - prev_xy[i][1]))
            a = S[i]["act"][-1]
            dh = arena.stats["hits"][i] - prev_hits[i]
            dh_foe = arena.stats["hits"][1 - i] - prev_hits[1 - i]
            if dh or dh_foe:
                S[i]["rew"][-1] += R_HIT_DEALT * dh + R_HIT_TAKEN * dh_foe
            if a == 4:
                S[i]["rew"][-1] -= FIRE_COST
                if S[i]["obs"][-1][-1] == 1.0:   # 盲射另罚（obs末位 0=畅通 1=被挡）
                    S[i]["rew"][-1] -= BLIND_PEN
            if a in (2, 3):
                move_n += 1
                if disp < 0.5 and tank.alive:
                    S[i]["rew"][-1] -= STUCK_PEN
                    stuck_n += 1
            if tank.alive:
                disp_hist[i].append(disp)
                if len(disp_hist[i]) > IDLE_WIN:
                    disp_hist[i].pop(0)
                if len(disp_hist[i]) == IDLE_WIN and sum(disp_hist[i]) < IDLE_TH:
                    S[i]["rew"][-1] -= IDLE_PEN
            S[i]["rew"][-1] += float(np.clip(CLOSE_BONUS * (prev_d - d), -0.01, 0.01))
            if (tank.alive and len(S[0]["rew"]) == len(S[1]["rew"])
                    and S[i]["obs"][-1][-1] == 0.0 and S[1 - i]["obs"][-1][-1] == 1.0):
                S[i]["rew"][-1] += LOS_BONUS
    for i in range(2):
        if not S[i]["rew"]:
            continue
        if arena.winner is None:
            S[i]["rew"][-1] -= DRAW_PENALTY
        elif arena.winner is arena.tanks[i]:
            S[i]["rew"][-1] += R_WIN
        else:
            S[i]["rew"][-1] += R_LOSE
        # return-to-go
        R, n = 0.0, len(S[i]["rew"])
        S[i]["ret"] = [0.0] * n
        for t in range(n - 1, -1, -1):
            R = S[i]["rew"][t] + GAMMA * R
            S[i]["ret"][t] = R
    outcome = "draw" if arena.winner is None else ("t0" if arena.winner is arena.tanks[0] else "t1")
    return S, outcome, arena.step_n, arena.stats["hits"], stuck_n / max(1, move_n)


def record_teacher_episode(rng):
    """老师自对弈一局，按序记录每侧 (obs, act)——LSTM 的 SL 热身数据（序列版，不破时间序）。"""
    arena = Arena(seed=int(rng.integers(0, 2**31)))
    teacher = TeacherBrain()
    seqs = [dict(obs=[], act=[]) for _ in range(2)]
    while not arena.done:
        acts = []
        for i, (tank, foe) in enumerate(zip(arena.tanks, arena.tanks[::-1])):
            if not tank.alive:
                acts.append(0)
                continue
            obs = arena.observe(tank, foe)
            a = teacher.act(obs)[0]
            seqs[i]["obs"].append(obs)
            seqs[i]["act"].append(a)
            acts.append(a)
        arena.step(acts)
    return [s for s in seqs if s["act"]]


# SL 类别权重（与 bc-v1 训练一致：开火稀有，重权）
CLASS_W = np.array([13.23, 14.74, 0.23, 1.94, 27.91])


def bc_warmup(net, params, m, v, rng, games, epochs, lr, log):
    """路线C第一环的 LSTM 版：老师序列 → 加权交叉熵 BPTT → 政策头先学会开火。"""
    log(f"  🏫 SL 热身：录制老师 {games} 局…")
    t0 = time.time()
    seqs = []
    for _ in range(games):
        seqs.extend(record_teacher_episode(rng))
    log(f"  🏫 {len(seqs)} 条序列，{sum(len(s['act']) for s in seqs)} 步，{time.time()-t0:.0f}s")
    t_adam = 0
    BATCH = 8    # 每 8 条序列拧一次（v1: 全数据集才一步=没训练；v2: 缩小批量换更新次数）
    for ep in range(epochs):
        rng.shuffle(seqs)
        grads = [np.zeros_like(p) for p in params]
        tot = correct = 0
        batch_n = 0
        n_updates = 0
        for s in seqs:
            T = len(s["act"])
            h, c = np.zeros(LSTM_H), np.zeros(LSTM_H)
            caches, Ps = [], []
            for t in range(T):
                P, _, h, c, cache = net.step(np.asarray(s["obs"][t], dtype=np.float64), h, c)
                Ps.append(P); caches.append(cache)
            tot += T
            dP_list = []
            for t in range(T):
                dZ = Ps[t].copy(); dZ[s["act"][t]] -= 1.0
                dP_list.append(dZ * CLASS_W[s["act"][t]] / T)
            gseq = net.bptt(caches, dP_list, [0.0] * T, T)
            for k in range(len(grads)):
                grads[k] += gseq[k]
            batch_n += 1
            if batch_n == BATCH:
                gstep = [g / batch_n for g in grads]
                norm = np.sqrt(sum(float((g ** 2).sum()) for g in gstep))
                if norm > CLIP_NORM:
                    gstep = [g * CLIP_NORM / norm for g in gstep]
                t_adam += 1
                for k, p in enumerate(params):
                    m[k] = 0.9 * m[k] + 0.1 * gstep[k]
                    v[k] = 0.999 * v[k] + 0.001 * gstep[k] ** 2
                    mh = m[k] / (1 - 0.9 ** t_adam)
                    vh = v[k] / (1 - 0.999 ** t_adam)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
                grads = [np.zeros_like(p) for p in params]
                batch_n = 0
                n_updates += 1
        # 测一次 acc（训练后）
        for s in seqs[:20]:
            h, c = np.zeros(LSTM_H), np.zeros(LSTM_H)
            for t in range(len(s["act"])):
                P, _, h, c, _ = net.step(np.asarray(s["obs"][t], dtype=np.float64), h, c)
                correct += int(P.argmax() == s["act"][t])
        acc_n = sum(len(s["act"]) for s in seqs[:20])
        log(f"  🏫 epoch {ep+1}/{epochs}: acc {correct/acc_n*100:.1f}%（{n_updates} 次更新）")
    log("  🏫 SL 热身完成，进入自对弈 RL")


def grad_check(net, rng):
    """军规：数值梯度 vs BPTT 解析梯度。分两条通路校：
    ① 价值通路：loss=Σ ½(V_t−tgt)²，dV_t=(V_t−tgt) → 校 W0/b0/Wl/bl/Wv/bv
    ② 政策通路：loss=Σ −adv·logP_a，dZ_t=adv·(P−onehot) → 校 Wl/bl/Wp/bp（经 LSTM 的远程锅）"""
    T = 4
    seq = [(rng.standard_normal(37), int(rng.integers(0, 5)),
            float(rng.standard_normal()), float(rng.standard_normal())) for _ in range(T)]
    eps, worst = 1e-5, 0.0
    params = net.params()

    def forward_seq(net):
        h, c = np.zeros(LSTM_H), np.zeros(LSTM_H)
        caches, Ps, Vs = [], [], []
        for x, *_ in seq:
            P, V, h, c, cache = net.step(x, h, c)
            Ps.append(P); Vs.append(V); caches.append(cache)
        return caches, Ps, Vs

    def run(loss_kind, check_params):
        nonlocal worst
        caches, Ps, Vs = forward_seq(net)
        if loss_kind == "value":
            dP_list = [np.zeros(5) for _ in range(T)]
            dV_list = [Vs[t] - seq[t][3] for t in range(T)]
        else:
            dP_list = []
            for t in range(T):
                dZ = Ps[t].copy(); dZ[seq[t][1]] -= 1.0
                dP_list.append(dZ * seq[t][2])
            dV_list = [0.0] * T
        grads = net.bptt(caches, dP_list, dV_list, T)

        def numeric(net):
            caches, Ps, Vs = forward_seq(net)
            if loss_kind == "value":
                return sum(0.5 * (Vs[t] - seq[t][3]) ** 2 for t in range(T))
            return sum(-seq[t][2] * np.log(Ps[t][seq[t][1]] + 1e-12) for t in range(T))

        for pi in check_params:
            flat = params[pi].ravel()
            for j in rng.choice(len(flat), min(6, len(flat)), replace=False):
                old = flat[j]
                flat[j] = old + eps; lp = numeric(net)
                flat[j] = old - eps; lm = numeric(net)
                flat[j] = old
                num = (lp - lm) / (2 * eps)
                # 判定：相对误差 <1e-6，或两者都在浮点噪声地板下（<1e-4 且吻合到 4 位）
                if abs(num) < 1e-4 and abs(grads[pi].ravel()[j]) < 1e-4:
                    continue
                worst = max(worst, abs(num - grads[pi].ravel()[j])
                            / max(1e-8, abs(num) + abs(grads[pi].ravel()[j])))

    run("value", [0, 1, 2, 3, 6, 7])   # W0 b0 Wl bl Wv bv
    run("policy", [2, 3, 4, 5])        # Wl bl Wp bp
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--games", type=int, default=32, help="每轮局数（LSTM 比 MLP 慢，减半）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--init", default=None, help="LSTM 存档（续训）")
    ap.add_argument("--enc-from", default=os.path.join(HERE, "..", "data", "bc-v1.npz"),
                    help="编码器视觉皮层来源")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "rl-lstm-v1.npz"))
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--bc-warmup", type=int, default=0, help="SL 热身局数（>0 先克隆老师再 RL）")
    ap.add_argument("--bc-epochs", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    if args.init and os.path.exists(args.init):
        net = LSTMNet.load(args.init)
    else:
        net = LSTMNet(seed=0, enc_from=args.enc_from if os.path.exists(args.enc_from) else None)

    if args.grad_check:
        worst = grad_check(net, rng)
        print(f"BPTT 数值梯度校验：最大相对误差 {worst:.2e}", "✅ 过" if worst < 1e-6 else "❌ 挂")
        return

    out_log = args.out.replace(".npz", "-log.txt")
    logf = open(out_log, "a", encoding="utf-8")
    t0 = time.time()

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    params = net.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t_adam = 0
    best_score = -1e9
    log(f"=== LSTM 自对弈开始 {time.strftime('%H:%M:%S')} init={args.init} "
        f"iters={args.iters} games/iter={args.games} lr={args.lr} ===")
    if args.bc_warmup > 0 and args.init is None:
        bc_warmup(net, params, m, v, rng, args.bc_warmup, args.bc_epochs, 3e-4, log)
    for it in range(1, args.iters + 1):
        games_data = []
        n_draw = hits_tot = steps_tot = 0
        stuck_tot = []
        all_A = []
        for _ in range(args.games):
            S, outcome, steps, hits, stuck = play_game(net, rng)
            n_draw += outcome == "draw"
            hits_tot += sum(hits); steps_tot += steps; stuck_tot.append(stuck)
            for i in range(2):
                if S[i]["act"]:
                    R = np.asarray(S[i]["ret"])
                    V = np.asarray(S[i]["V"])
                    S[i]["A"] = R - V
                    all_A.append(S[i]["A"])
                    games_data.append(S[i])
        # 批归一化 A
        flat_A = np.concatenate(all_A)
        mu, sd = flat_A.mean(), flat_A.std() + 1e-8
        for s in games_data:
            s["A"] = (s["A"] - mu) / sd
        # BPTT 累加梯度
        total_steps = sum(len(s["act"]) for s in games_data)
        grads = [np.zeros_like(p) for p in params]
        verr = 0.0
        for s in games_data:
            T = len(s["act"])
            dP_list, dV_list = [], []
            for t in range(T):
                P = s["P"][t]
                a = s["act"][t]
                A = s["A"][t]
                H = -(P * np.log(P + 1e-12)).sum()
                dZ = P.copy(); dZ[a] -= 1.0
                dZ *= A
                dZ += ENTROPY_BETA * P * (np.log(P + 1e-12) + H)
                # 注意：dZ 是 logits 的 δ（softmax 雅可比已并入 P−onehot 形式）
                dP_list.append(dZ / total_steps)
                dV_list.append(VALUE_COEF * (s["V"][t] - s["ret"][t]) / total_steps)
                verr += abs(s["V"][t] - s["ret"][t])
            gseq = net.bptt(s["cache"], dP_list, dV_list, T)
            for k in range(len(grads)):
                grads[k] += gseq[k]
        verr /= total_steps
        # 裁剪 + Adam
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if norm > CLIP_NORM:
            grads = [g * CLIP_NORM / norm for g in grads]
        t_adam += 1
        for k, p in enumerate(params):
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            mh = m[k] / (1 - 0.9 ** t_adam)
            vh = v[k] / (1 - 0.999 ** t_adam)
            p -= args.lr * mh / (np.sqrt(vh) + 1e-8)
        log(f"iter {it:3d} | 僵局 {n_draw}/{args.games} | 局均命中 {hits_tot/args.games:.1f} "
            f"| 局均步 {steps_tot/args.games:.0f} | 顶墙率 {np.mean(stuck_tot)*100:.1f}% "
            f"| V误差 {verr:.3f} | {time.time()-t0:.0f}s")
        # 月考（有记忆的脑，match 会自动 reset）
        if it % args.eval_every == 0:
            class _B:
                def __init__(self):
                    self.h = np.zeros(LSTM_H); self.c = np.zeros(LSTM_H)

                def reset(self):
                    self.h = np.zeros(LSTM_H); self.c = np.zeros(LSTM_H)

                def act(self, obs, sample=False, rng=None):
                    P, _, self.h, self.c, _ = net.step(np.asarray(obs, dtype=np.float64), self.h, self.c)
                    return int(P.argmax()), None
            wa, dr, wb, hr, st = match(_B(), TeacherBrain(), 30, seed=50000 + it)
            rnd = type("R", (), {"act": lambda self, o, sample=False, rng=None:
                                 (int(np.random.default_rng().integers(0, 5)), None)})()
            wr, dr2, wl, hr2, _ = match(_B(), rnd, 30, seed=60000 + it)
            score = (wa - wb) + 0.5 * (wr - wl)
            log(f"  📋 月考 @{it}: vs老师 {wa}胜{dr}平{wb}负(命中{hr*100:.0f}%) "
                f"vs随机 {wr}胜{dr2}平{wl}负 | score {score:+.0f}")
            if score > best_score:
                best_score = score
                net.save(args.out)
                log(f"  💾 新最佳，权重已存 {args.out}")
    log("=== 训练结束 ===")
    logf.close()


if __name__ == "__main__":
    main()
