#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 2.0 · 工人-学徒并行 PPO 工厂（train_parallel.py 的 LSTM 版）。

与 MLP 版的关键差异：LSTM 有状态——一条对局是一个序列，训练沿时间反向传播（BPTT）。
  工人：领当前权重 → 有状态推理打 n 局（vs 六流派剧本 BotOneHot / 自对弈第二份权重快照）
        → 逐步记录 (obs, act, logp, V, reward)，整局 float32 紧凑回传
  学徒：每局 GAE（γ/λ 从 CLI）→ 优势批归一化 → K epoch PPO：
        每局切 ≤--bptt-len 步窗口，窗口携带入场 (h,c)（上一窗口前向终态，detached），
        窗口内顺序 step 前向 → PPO clip loss + value loss → lstm_net.bptt 全窗口反传
        → 梯度全局范数裁剪 → Adam 步进。BC 拴绳帧级交叉熵可掺。
  月考：vs TeacherBrain 30 局 + vs 随机 30 局（LSTMBrain.reset 每局清零），best 只升不降。

军规：--grad-check 对"单窗口 BPTT+PPO 损失"做有限差分对账，过了才准开工。

用法：
  ./.venv/bin/python scripts/train_lstm_parallel.py --workers 28 --games 252 --iters 4000 \
      --init data/rl-lstm-v2.npz --out data/rl-lstm-v3.npz --vs-teacher --destructible
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena, MAX_STEPS  # noqa: E402
from lstm_net import LSTMNet, LSTMBrain, LSTM_H, ENC_OUT  # noqa: E402
from eval_bc import TeacherBrain, match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 奖励 shaping（= train_rl.play_game 全表，global 变量式，main 里被 CLI 覆写）----
R_HIT_DEALT, R_HIT_TAKEN = 1.0, -1.0
R_WIN, R_LOSE = 1.0, -1.0
ENTROPY_BETA = 0.005
CLIP_NORM = 5.0

P_KEYS = ["w0", "b0", "wl", "bl", "wp", "bp", "wv", "bv"]


def net_from_dict(d):
    net = LSTMNet(seed=0)
    net.W0, net.b0 = d["w0"], d["b0"]
    net.Wl, net.bl = d["wl"], d["bl"]
    net.Wp, net.bp = d["wp"], d["bp"]
    net.Wv, net.bv = d["wv"], d["bv"]
    return net


def weights_dict(net):
    return {"w0": net.W0, "b0": net.b0, "wl": net.Wl, "bl": net.bl,
            "wp": net.Wp, "bp": net.bp, "wv": net.Wv, "bv": net.bv}


# ---------------- 工人进程（模块级，可 pickle） ----------------
class BotOneHot:
    """剧本机器人 → play 的 foe 接口：确定性 one-hot 分布。"""
    def __init__(self, bot):
        self._b = bot

    def forward(self, x):
        P = np.zeros((1, 5)); P[0, self._b.decide(x[0])] = 1.0
        return P, None


class RandomBrain:
    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, sample=False, rng=None):
        return int(self.rng.integers(0, 5)), None


def play_one(bnet, foe, rng, cfg):
    """一局。bnet/foe 都是 LSTMNet（或 BotOneHot/RandomBrain 伪脑，有 forward）。
    奖励表照抄 train_rl.play_game。返回 (蓝pack, 红pack|None, stats)。"""
    arena = Arena(seed=int(rng.integers(0, 2**31)), destructible=cfg["destructible"],
                  no_cover=cfg.get("no_cover", False))
    nets = [bnet, foe if foe is not None else bnet]
    S = [dict(obs=[], act=[], logp=[], V=[], rew=[]) for _ in range(2)]
    H = [np.zeros(LSTM_H), np.zeros(LSTM_H)]
    C = [np.zeros(LSTM_H), np.zeros(LSTM_H)]
    disp_hist = [[], []]
    stuck_n = move_n = 0
    while not arena.done:
        acts = []
        for i, (tank, fn) in enumerate(zip(arena.tanks, nets)):
            if not tank.alive:
                acts.append(0)
                continue
            obs = arena.observe(tank, arena.tanks[1 - i])
            if isinstance(fn, LSTMNet):                     # LSTM 有状态推理
                P, V, H[i], C[i], _ = fn.step(np.asarray(obs, dtype=np.float64), H[i], C[i])
            else:                                           # BotOneHot / RandomBrain
                p1, _ = fn.forward(np.asarray(obs, dtype=np.float64)[None])
                P, V = p1[0], 0.0
            p = P / P.sum()
            a = int(rng.choice(5, p=p))
            s = S[i]
            s["obs"].append(obs); s["act"].append(a); s["rew"].append(0.0)
            if isinstance(fn, LSTMNet):
                s["logp"].append(float(np.log(P[a] + 1e-12))); s["V"].append(float(V))
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
            if cfg["stall_coef"]:
                S[i]["rew"][-1] -= cfg["stall_coef"] * arena.step_n / MAX_STEPS
            disp = float(np.hypot(tank.x - prev_xy[i][0], tank.y - prev_xy[i][1]))
            a = S[i]["act"][-1]
            dh = arena.stats["hits"][i] - prev_hits[i]
            dh_foe = arena.stats["hits"][1 - i] - prev_hits[1 - i]
            if dh or dh_foe:
                S[i]["rew"][-1] += R_HIT_DEALT * dh + R_HIT_TAKEN * dh_foe
            if a == 4:
                if cfg["fire_cost"]:
                    S[i]["rew"][-1] -= cfg["fire_cost"]
                if S[i]["obs"][-1][-1] == 0.0:              # 射界畅通=正当瞄准；否则盲射罚
                    pass
                elif cfg["blind_penalty"]:
                    S[i]["rew"][-1] -= cfg["blind_penalty"]
            if a in (2, 3):
                move_n += 1
                if disp < 0.5 and tank.alive:
                    S[i]["rew"][-1] -= cfg["stuck_penalty"]
                    stuck_n += 1
            if tank.alive:
                disp_hist[i].append(disp)
                if len(disp_hist[i]) > 10:
                    disp_hist[i].pop(0)
                if len(disp_hist[i]) == 10 and sum(disp_hist[i]) < cfg["idle_thresh"]:
                    S[i]["rew"][-1] -= cfg["idle_penalty"]
            S[i]["rew"][-1] += float(np.clip(cfg["close_bonus"] * (prev_d - d), -0.01, 0.01))
            if (cfg["los_bonus"] and tank.alive and len(S[0]["rew"]) == len(S[1]["rew"])
                    and S[i]["obs"][-1][-1] == 0.0 and S[1 - i]["obs"][-1][-1] == 1.0):
                S[i]["rew"][-1] += cfg["los_bonus"]
    for i in range(2):
        if not S[i]["rew"]:
            continue
        if arena.winner is None:
            r = -cfg["draw_penalty"]
        elif arena.winner is arena.tanks[i]:
            r = R_WIN + cfg["speed_bonus"] * (1.0 - arena.step_n / MAX_STEPS)
        else:
            r = R_LOSE
        S[i]["rew"][-1] += r
    sides = (0,) if foe is not None and not isinstance(foe, LSTMNet) else (0, 1)
    packs = {}
    for i in sides:
        if S[i]["act"]:
            packs[i] = (np.asarray(S[i]["obs"], dtype=np.float32),
                        np.asarray(S[i]["act"], dtype=np.int8),
                        np.asarray(S[i]["logp"], dtype=np.float32),
                        np.asarray(S[i]["V"], dtype=np.float32),
                        np.asarray(S[i]["rew"], dtype=np.float32))
    stats = dict(draw=int(arena.winner is None), hits=int(sum(arena.stats["hits"])),
                 steps=int(arena.step_n), stuck=stuck_n / max(1, move_n))
    return packs, stats


def worker_games(payload):
    """一个工人打 n 局，回传整局序列包。"""
    wsd, foe_wsd, seed, n_games, cfg = payload
    net = net_from_dict(wsd)
    foe_net = net_from_dict(foe_wsd) if foe_wsd is not None else None
    rng = np.random.default_rng(seed)
    use_bots = cfg.get("vs_teacher")
    packs, stats = [], dict(draw=0, hits=0, steps=0, games=0, stuck=[])
    for _ in range(n_games):
        foe = None
        if use_bots:
            from bot_roster import BOTS
            foe = BotOneHot(BOTS[int(rng.integers(0, len(BOTS)))][1]())
        elif foe_net is not None:
            foe = foe_net                     # 自对弈：第二份权重快照（各自 h,c）
        pks, st = play_one(net, foe, rng, cfg)
        for i in sorted(pks):
            packs.append(pks[i])
        stats["draw"] += st["draw"]; stats["hits"] += st["hits"]; stats["steps"] += st["steps"]
        stats["games"] += 1; stats["stuck"].append(st["stuck"])
    return packs, stats


# ---------------- 学徒：窗口 BPTT + PPO ----------------
def window_forward(net, X, h0, c0):
    """窗口顺序 step 前向。返回 caches, Ps, Vs, h_end, c_end。"""
    h, c = h0.copy(), c0.copy()
    caches, Ps, Vs = [], [], []
    for t in range(len(X)):
        P, V, h, c, cache = net.step(np.asarray(X[t], dtype=np.float64), h, c)
        Ps.append(P); Vs.append(V); caches.append(cache)
    return caches, Ps, Vs, h, c


def ppo_window_loss(net, X, acts, adv, logp_old, ret, h0, c0, value_coef, eps_clip):
    """单窗口前向 + PPO clip loss + value loss（标量，均值口径）。"""
    caches, Ps, Vs, _, _ = window_forward(net, X, h0, c0)
    Ps = np.asarray(Ps); Vs = np.asarray(Vs)
    T = len(acts)
    logp = np.array([np.log(Ps[t][acts[t]] + 1e-12) for t in range(T)])
    ratio = np.exp(logp - logp_old)
    s1 = ratio * adv
    s2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv
    H = -(Ps * np.log(Ps + 1e-12)).sum(axis=1)
    loss_pg = float(-np.minimum(s1, s2).mean()) - float(ENTROPY_BETA * H.mean())
    loss_v = float(0.5 * ((Vs - ret) ** 2).mean())
    return loss_pg + value_coef * loss_v


def ppo_window_grad(net, X, acts, adv, logp_old, ret, h0, c0, value_coef, eps_clip):
    """单窗口解析梯度（不裁剪）。返回 (loss, 8 个梯度, h_end, c_end)。"""
    caches, Ps, Vs, h_end, c_end = window_forward(net, X, h0, c0)
    Ps = np.asarray(Ps); Vs = np.asarray(Vs)
    T = len(acts)
    logp = np.array([np.log(Ps[t][acts[t]] + 1e-12) for t in range(T)])
    ratio = np.exp(logp - logp_old)
    s1 = ratio * adv
    s2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv
    mask = (s1 <= s2).astype(np.float64)
    w = ratio * adv * mask
    H = -(Ps * np.log(Ps + 1e-12)).sum(axis=1)
    dP_list, dV_list = [], []
    for t in range(T):
        dZ = Ps[t].copy(); dZ[acts[t]] -= 1.0
        dZ *= w[t]
        dZ += ENTROPY_BETA * Ps[t] * (np.log(Ps[t] + 1e-12) + H[t])
        dP_list.append(dZ / T)
        dV_list.append(value_coef * (Vs[t] - ret[t]) / T)
    grads = net.bptt(caches, dP_list, dV_list, T)
    loss_pg = float(-np.minimum(s1, s2).mean()) - float(ENTROPY_BETA * H.mean())
    loss_v = float(0.5 * ((np.array(Vs) - ret) ** 2).mean())
    clipfrac = float((1 - mask).mean())
    verr_w = float(np.abs(np.array(Vs) - ret).mean())
    return loss_pg + value_coef * loss_v, grads, h_end, c_end, clipfrac, verr_w


def grad_check(rng, T=20):
    """军规：对"单窗口 BPTT+PPO 损失"有限差分对账（少参数抽样）。"""
    net = LSTMNet(seed=0)
    X = rng.standard_normal((T, 37))
    acts = rng.integers(0, 5, T)
    adv = rng.standard_normal(T)
    ret = rng.standard_normal(T)
    logp_old = np.log(np.full(T, 0.2) + 1e-12)     # 旧策略≈均匀 → 初始 ratio=1，不踩裁剪边界
    h0, c0 = rng.standard_normal(LSTM_H) * 0.1, rng.standard_normal(LSTM_H) * 0.1
    vc, eps = 0.5, 0.2
    _, grads, _, _, _, _ = ppo_window_grad(net, X, acts, adv, logp_old, ret, h0, c0, vc, eps)
    params = net.params()
    eps_fd, worst = 1e-5, 0.0
    n_checked = 0
    for pi in range(len(params)):
        flat = params[pi].ravel()
        for j in rng.choice(len(flat), min(4, len(flat)), replace=False):
            old = flat[j]
            flat[j] = old + eps_fd
            lp = ppo_window_loss(net, X, acts, adv, logp_old, ret, h0, c0, vc, eps)
            flat[j] = old - eps_fd
            lm = ppo_window_loss(net, X, acts, adv, logp_old, ret, h0, c0, vc, eps)
            flat[j] = old
            num = (lp - lm) / (2 * eps_fd)
            ana = grads[pi].ravel()[j]
            n_checked += 1
            if abs(num) < 1e-4 and abs(ana) < 1e-4:
                continue                            # 浮点噪声地板豁免
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst, n_checked


class AdamLSTM:
    """AdamAC 风格：8 个张量的 Adam（m/v 可从 resume 恢复）。"""
    def __init__(self, net, lr=1e-4):
        self.lr, self.b1, self.b2, self.eps = lr, 0.9, 0.999, 1e-8
        self.t = 0
        self.m = [np.zeros_like(p) for p in net.params()]
        self.v = [np.zeros_like(p) for p in net.params()]

    def step(self, net, grads):
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if norm > CLIP_NORM:
            grads = [g * CLIP_NORM / norm for g in grads]
        self.t += 1
        for k, p in enumerate(net.params()):
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def bc_frame_grads(net, X, y, class_w):
    """帧级 BC 交叉熵梯度（h=c=0 独立帧，复用 bptt 的已校验逆流）。返回 8 个梯度。"""
    grads = [np.zeros_like(p) for p in net.params()]
    n = len(y)
    for i in range(n):
        P, _, _, _, cache = net.step(np.asarray(X[i], dtype=np.float64),
                                     np.zeros(LSTM_H), np.zeros(LSTM_H))
        dZ = P.copy(); dZ[y[i]] -= 1.0
        g = net.bptt([cache], [dZ * class_w[y[i]] / n], [0.0], 1)
        for k in range(len(grads)):
            grads[k] += g[k]
    return grads


def save_resume(path, net, opt, it, best_score):
    d = weights_dict(net)
    for k in range(8):
        d[f"adam_m{k}"] = opt.m[k]
        d[f"adam_v{k}"] = opt.v[k]
    d.update({"t_adam": np.array(opt.t), "iter": np.array(it), "best": np.array(best_score)})
    np.savez(path, **d)


class EvalBrain:
    """月考脑：包当前 net，有记忆，match 每局开头自动 reset。"""
    def __init__(self, net):
        self.net = net
        self.h = np.zeros(LSTM_H)
        self.c = np.zeros(LSTM_H)

    def reset(self):
        self.h = np.zeros(LSTM_H)
        self.c = np.zeros(LSTM_H)

    def act(self, obs, sample=False, rng=None):
        P, _, self.h, self.c, _ = self.net.step(np.asarray(obs, dtype=np.float64), self.h, self.c)
        return int(P.argmax()), None


def gae(rew, V, gamma, lam):
    """单局 GAE。返回 (adv, ret)。"""
    n = len(rew)
    adv = np.zeros(n)
    g = 0.0
    for t in range(n - 1, -1, -1):
        v_next = V[t + 1] if t + 1 < n else 0.0
        delta = rew[t] + gamma * v_next - V[t]
        g = delta + gamma * lam * g
        adv[t] = g
    return adv, adv + np.asarray(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--games", type=int, default=252, help="每轮全局局数（分摊给工人）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--init", default=os.path.join(HERE, "..", "data", "rl-lstm-v2.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "rl-lstm-v3.npz"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bptt-len", type=int, default=64, help="BPTT 窗口步长")
    ap.add_argument("--grad-check", action="store_true")
    # 奖励表（默认 = v14 配方 + v15 物理；global 可被 CLI 覆写）
    ap.add_argument("--stuck-penalty", type=float, default=0.01)
    ap.add_argument("--idle-penalty", type=float, default=0.01)
    ap.add_argument("--idle-thresh", type=float, default=15.0)
    ap.add_argument("--close-bonus", type=float, default=0.0005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--draw-penalty", type=float, default=0.5)
    ap.add_argument("--los-bonus", type=float, default=0.002)
    ap.add_argument("--blind-penalty", type=float, default=0.02)
    ap.add_argument("--fire-cost", type=float, default=0.0)
    ap.add_argument("--hit-reward", type=float, default=1.0)
    ap.add_argument("--speed-bonus", type=float, default=0.0)
    ap.add_argument("--destructible", action="store_true")
    ap.add_argument("--no-cover", action="store_true")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-eps", type=float, default=0.2)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--stall-coef", type=float, default=0.0)
    ap.add_argument("--vs-teacher", action="store_true", help="红方=六流派剧本对手（非自对弈）")
    ap.add_argument("--bc-lambda", type=float, default=0.0, help="BC 拴绳强度")
    ap.add_argument("--bc-data", default=os.path.join(HERE, "..", "data", "bc-v23-cover.npz"))
    ap.add_argument("--bc-anneal", default=None, help='退火 "start,end,iters" 例 0.5,0.05,2000')
    args = ap.parse_args()

    global R_HIT_DEALT, R_HIT_TAKEN
    R_HIT_DEALT, R_HIT_TAKEN = args.hit_reward, -args.hit_reward

    out_log = args.out.replace(".npz", "-log.txt")
    resume_path = args.out.replace(".npz", "-resume.npz")
    logf = open(out_log, "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    if args.grad_check:
        worst, n = grad_check(np.random.default_rng(7))
        log(f"BPTT+PPO 窗口数值梯度校验（{n} 个采样点）：最大相对误差 {worst:.2e} "
            + ("✅ 过" if worst < 1e-6 else "❌ 挂"))
        logf.close()
        return

    # ---- 初始化 / 断点续训 ----
    start_it, best_score = 0, -1e9
    if args.resume and os.path.exists(resume_path):
        net = net_from_dict(np.load(resume_path))
        opt = AdamLSTM(net, lr=args.lr)
        d = np.load(resume_path)
        opt.m = [d[f"adam_m{k}"] for k in range(8)]
        opt.v = [d[f"adam_v{k}"] for k in range(8)]
        opt.t = int(d["t_adam"]); start_it = int(d["iter"]); best_score = float(d["best"])
        log(f"=== 断点续训 @{start_it}（best {best_score:+.0f}）===")
    else:
        if args.init and os.path.exists(args.init):
            net = LSTMNet.load(args.init)
            log(f"init={os.path.basename(args.init)}")
        else:
            net = LSTMNet(seed=0)
            log("init=随机（无 enc_from）")
        opt = AdamLSTM(net, lr=args.lr)

    ANNEAL = None
    if args.bc_anneal:
        ANNEAL = tuple(float(x) for x in args.bc_anneal.split(","))
    if args.bc_lambda > 0:
        _d = np.load(args.bc_data)
        DX = _d["X"].astype(np.float64); DY = _d["y"].astype(np.int64)
        _cnt = np.bincount(DY, minlength=5).astype(np.float64)
        DCW = np.clip(len(DY) / (5 * _cnt + 1e-9), 0.5, 5.0)
        log(f"拴绳拳谱 {len(DY)} 张 λ={args.bc_lambda} 类权重 {np.round(DCW,2)}")

    cfg = dict(stuck_penalty=args.stuck_penalty, close_bonus=args.close_bonus,
               gamma=args.gamma, idle_penalty=args.idle_penalty,
               idle_thresh=args.idle_thresh, draw_penalty=args.draw_penalty,
               los_bonus=args.los_bonus, blind_penalty=args.blind_penalty,
               fire_cost=args.fire_cost, speed_bonus=args.speed_bonus,
               gae_lambda=args.gae_lambda, destructible=args.destructible,
               stall_coef=args.stall_coef, vs_teacher=args.vs_teacher,
               no_cover=args.no_cover)

    per_worker = max(1, args.games // args.workers)
    t0 = time.time()
    log(f"=== LSTM 数据工厂开工 {time.strftime('%H:%M:%S')} 工人={args.workers} "
        f"局/轮={per_worker * args.workers} 目标={args.iters}轮 lr={args.lr} "
        f"bptt={args.bptt_len} vs_teacher={args.vs_teacher} ===")

    with mp.Pool(args.workers) as pool:
        for it in range(start_it + 1, args.iters + 1):
            # ① 分发权重（自对弈时红方拿第二份快照），并行打局
            wsd = {k: v.copy() for k, v in weights_dict(net).items()}
            foe_wsd = None if args.vs_teacher else {k: v.copy() for k, v in wsd.items()}
            payloads = [(wsd, foe_wsd, int(1e9) + it * 100 + w, per_worker, cfg)
                        for w in range(args.workers)]
            results = pool.map(worker_games, payloads)
            # ② 汇总整局 → 每局 GAE → 优势批归一化
            episodes = []            # (obs, acts, adv, logp, ret)
            n_draw = hits_tot = steps_tot = games = 0
            stuck_tot = []
            for packs, st in results:
                for X, Y, LP, V, R in packs:
                    adv, ret = gae(R.astype(np.float64), V.astype(np.float64),
                                   args.gamma, args.gae_lambda)
                    episodes.append((X.astype(np.float64), Y.astype(np.int64),
                                     adv, LP.astype(np.float64), ret))
                n_draw += st["draw"]; hits_tot += st["hits"]; steps_tot += st["steps"]
                games += st["games"]; stuck_tot.extend(st["stuck"])
            flat_A = np.concatenate([e[2] for e in episodes])
            mu, sd = flat_A.mean(), flat_A.std() + 1e-8
            for e in episodes:
                e[2][:] = (e[2] - mu) / sd
            # ③ PPO K epoch：切窗口 → 顺序前向 → BPTT 全窗口反传 → Adam
            total_loss = verr = 0.0
            clip_num = clip_den = 0.0
            total_steps = sum(len(e[1]) for e in episodes)
            for ep in range(args.ppo_epochs):
                grads = [np.zeros_like(p) for p in net.params()]
                w_steps = 0
                for X, Y, adv, LP, ret in episodes:
                    T = len(Y)
                    h_e = np.zeros(LSTM_H); c_e = np.zeros(LSTM_H)
                    for k0 in range(0, T, args.bptt_len):
                        k1 = min(k0 + args.bptt_len, T)
                        loss, g, h_e, c_e, cf, vwe = ppo_window_grad(
                            net, X[k0:k1], Y[k0:k1], adv[k0:k1], LP[k0:k1],
                            ret[k0:k1], h_e, c_e, args.value_coef, args.ppo_eps)
                        total_loss += loss
                        clip_num += cf * (k1 - k0); clip_den += k1 - k0
                        verr += vwe * (k1 - k0)
                        for kk in range(len(grads)):
                            grads[kk] += g[kk]
                        w_steps += k1 - k0
                # 归一到"每步均值"口径（跨窗口/局累加后按步数摊）
                grads = [g * (total_steps / max(1, w_steps)) / args.ppo_epochs for g in grads]
                # BC 拴绳
                if args.bc_lambda > 0:
                    lam = args.bc_lambda
                    if ANNEAL:
                        s_, e_, n_ = ANNEAL
                        lam = max(e_, s_ + (e_ - s_) * it / n_)
                    bi = np.random.default_rng(it * 100 + ep).integers(0, len(DX), 2048)
                    gbc = bc_frame_grads(net, DX[bi], DY[bi], DCW)
                    for k in range(8):
                        grads[k] = grads[k] + lam * gbc[k]
                opt.step(net, grads)
            # ④ 断点存档（原子：.tmp.npz → os.replace）
            tmp = resume_path + ".tmp.npz"
            save_resume(tmp, net, opt, it, best_score)
            os.replace(tmp, resume_path)
            if it % 10 == 0:
                n_eps = max(1, args.ppo_epochs * max(1, len(episodes)))
                cf = clip_num / max(1, clip_den)
                log(f"iter {it:5d} | loss {total_loss / n_eps:+.4f} "
                    f"| 僵局 {n_draw}/{games} | 局均命中 {hits_tot/games:.1f} "
                    f"| 局均步 {steps_tot/games:.0f} | 顶墙率 {np.mean(stuck_tot)*100:.1f}% "
                    f"| V误差 {verr / max(1, args.ppo_epochs * max(1, total_steps)):.3f} "
                    f"| 裁剪率 {cf*100:.0f}% | {time.time()-t0:.0f}s")
            # ⑤ 月考（best-checkpoint 军规）
            if it % args.eval_every == 0:
                wa, dr, wb, hr, _ = match(EvalBrain(net), TeacherBrain(), 30,
                                          seed=50000 + it, destructible=args.destructible)
                rnd = RandomBrain(np.random.default_rng(1))
                wr, dr2, wl, _, _ = match(EvalBrain(net), rnd, 30,
                                          seed=60000 + it, destructible=args.destructible)
                score = (wa - wb) + 0.5 * (wr - wl)
                log(f"  📋 月考 @{it}: vs老师 {wa}胜{dr}平{wb}负(命中{hr*100:.0f}%) "
                    f"vs随机 {wr}胜{dr2}平{wl}负 | score {score:+.0f}")
                if score > best_score:
                    best_score = score
                    tmp_best = args.out + ".tmp.npz"
                    np.savez(tmp_best, **weights_dict(net))
                    os.replace(tmp_best, args.out)
                    log(f"  💾 新最佳，权重已存 {args.out}")
    log("=== LSTM 数据工厂收工 ===")
    logf.close()


if __name__ == "__main__":
    main()
