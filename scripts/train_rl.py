#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线C 第③步：自对弈强化学习（REINFORCE 策略梯度）。

剧本（对话里聊的那套）：
  1. 左右互搏：克隆体 vs 克隆体，全程按 P 抽签（探索），记下每步 (obs, 动作)
  2. 局终结算：命中+0.2 / 挨打-0.2 / 胜+1 / 负-1 / 平-0.05（反僵死）→ 这局的 A
  3. loss = -logP(自己抽的动作) × A（A>0 提概率，A<0 压概率）+ 熵奖励（防过早锁死）
  4. 反向传播（train_bc 同一套），梯度裁剪，Adam 拧旋钮
  5. 定期月考：argmax 打老师 & 打随机，战绩变好就存档

用法：train_rl.py --iters 200 --games 64
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena import Arena, MAX_STEPS  # noqa: E402
from train_bc import Net, Adam  # noqa: E402
from eval_bc import TeacherBrain, match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BC_W = os.path.join(HERE, "..", "data", "bc-v1.npz")
OUT_W = os.path.join(HERE, "..", "data", "rl-v1.npz")
OUT_LOG = os.path.join(HERE, "..", "data", "rl-v1-log.txt")

# 奖励 shaping（v4：平局重罚走 --draw-penalty；命中已改当步记账；v14：命中奖 --hit-reward 可调）
R_HIT_DEALT, R_HIT_TAKEN = 0.2, -0.2  # 默认值，main 里可被 --hit-reward 覆盖
R_WIN, R_LOSE = 1.0, -1.0
ENTROPY_BETA = 0.005      # 熵奖励：鼓励保持探索，防策略过早锁死
CLIP_NORM = 5.0           # 梯度裁剪：排坑表第4条（防爆炸）


def play_game(net, rng, stuck_penalty=0.01, close_bonus=0.0005, gamma=0.99,
              idle_penalty=0.01, idle_window=10, idle_thresh=3.0,
              draw_penalty=0.05, los_bonus=0.0, fire_exempt=10, blind_penalty=0.0,
              fire_cost=0.0, speed_bonus=0.0, foe_net=None, gae_lambda=None,
              destructible=False, stall_coef=0.0, no_cover=False, random_cover=False,
              fire_idle_penalty=0.0, fire_idle_thresh=150, obs_v2=False):
    """自对弈一局（双方都从当前 net 抽签）；v10：foe_net 非 None 时红方用冻结的历史对手（联赛），
    只有蓝方(0号)的经验参与训练——打的是"不送上门的历代自己"，蹲坑生态位拆除。
    v2：逐步 shaping——顶墙流血 + 站桩流血 + 接近有奖 + return-to-go 精准分锅。
    v4：命中当步记账（不再押送终局）+ 射界奖 + 平局重罚可调 + 开火豁免站桩罚。
    v12：gae_lambda 非 None 时记录 V(s)/logp，用 GAE 算优势（PPO 用）。"""
    arena = Arena(seed=int(rng.integers(0, 2**31)), destructible=destructible, no_cover=no_cover,
                  random_cover=random_cover, obs_v2=obs_v2)
    obs_buf, act_buf, rew_buf = [[], []], [[], []], [[], []]
    v_buf, logp_buf = [[], []], [[], []]
    disp_hist = [[], []]
    last_fire = [-10**9, -10**9]
    stuck_n = move_n = 0
    while not arena.done:
        acts = []
        for i, (tank, foe) in enumerate(zip(arena.tanks, arena.tanks[::-1])):
            if not tank.alive:
                acts.append(0)
                continue
            obs = arena.observe(tank, foe)
            brain = net if (i == 0 or foe_net is None) else foe_net   # 联赛：红方=冻结历史脑
            P, cache = brain.forward(np.asarray(obs, dtype=np.float64)[None])
            p = P[0]
            a = int(rng.choice(5, p=p / p.sum()))
            obs_buf[i].append(obs)
            act_buf[i].append(a)
            rew_buf[i].append(0.0)
            if gae_lambda is not None and brain is net:
                v_buf[i].append(float(cache[2][0][0] @ net.Wv[:, 0] + net.bv[0]))
                logp_buf[i].append(float(np.log(p[a] + 1e-12)))
            acts.append(a)
        prev_xy = [(t.x, t.y) for t in arena.tanks]
        prev_d = float(np.hypot(arena.tanks[0].x - arena.tanks[1].x,
                                arena.tanks[0].y - arena.tanks[1].y))
        prev_hits = list(arena.stats["hits"])
        arena.step(acts)
        d = float(np.hypot(arena.tanks[0].x - arena.tanks[1].x,
                           arena.tanks[0].y - arena.tanks[1].y))
        step_i = arena.step_n
        for i, tank in enumerate(arena.tanks):
            if not rew_buf[i]:
                continue
            # v17: 僵局流血——时间税线性加重，全蹲局期望严格低于战败（拆吸引子）
            if stall_coef:
                rew_buf[i][-1] -= stall_coef * arena.step_n / MAX_STEPS
            disp = float(np.hypot(tank.x - prev_xy[i][0], tank.y - prev_xy[i][1]))
            a = act_buf[i][-1]
            if a == 4:
                # v6: 弹药费——每炮必收（spam经济学死刑）；盲射另罚
                if fire_cost:
                    rew_buf[i][-1] -= fire_cost
                # v4.1: 只有射界畅通时开火才记为"正当瞄准"；盲射挨罚
                if obs_buf[i][-1][-1] == 0.0:
                    last_fire[i] = step_i
                elif blind_penalty:
                    rew_buf[i][-1] -= blind_penalty
            # v26: 哑火罚——连续 thresh 步无"正当开炮"流血（直击枪怯；开局豁免前 thresh 步）
            if fire_idle_penalty and step_i > fire_idle_thresh \
                    and step_i - last_fire[i] > fire_idle_thresh:
                rew_buf[i][-1] -= fire_idle_penalty
            # v4: 命中当步记账——开炮轮只隔炮弹飞行 ~15 步，功劳 86% 到账
            dh = arena.stats["hits"][i] - prev_hits[i]
            dh_foe = arena.stats["hits"][1 - i] - prev_hits[1 - i]
            if dh or dh_foe:
                rew_buf[i][-1] += R_HIT_DEALT * dh + R_HIT_TAKEN * dh_foe
            if a in (2, 3):  # 想前进/后退
                move_n += 1
                if disp < 0.5 and tank.alive:  # 轮子空转：顶墙/顶人
                    rew_buf[i][-1] -= stuck_penalty
                    stuck_n += 1
            # 站桩流血：近 idle_window 步几乎没挪窝（活着就罚；v6 拆除开火豁免——瞄准该付的小税）
            if tank.alive:
                disp_hist[i].append(disp)
                if len(disp_hist[i]) > idle_window:
                    disp_hist[i].pop(0)
                if len(disp_hist[i]) == idle_window and sum(disp_hist[i]) < idle_thresh:
                    rew_buf[i][-1] -= idle_penalty
            # 接近敌人有奖（远离扣分），单步限幅
            rew_buf[i][-1] += float(np.clip(close_bonus * (prev_d - d), -0.01, 0.01))
            # v4: 射界奖——我射界畅通且敌人被挡（我能打它/它打不到我），obs 末位 0=畅通 1=被挡
            if (los_bonus and tank.alive and len(obs_buf[0]) == len(obs_buf[1])
                    and obs_buf[i][-1][-1] == 0.0 and obs_buf[1 - i][-1][-1] == 1.0):
                rew_buf[i][-1] += los_bonus
    rets = []
    advs = []
    for i in range(2):
        # v4: 命中已在当步记账，终局只剩胜负平；v9: 速胜奖——给时间定价
        if arena.winner is None:
            r = -draw_penalty
        elif arena.winner is arena.tanks[i]:
            r = R_WIN + speed_bonus * (1.0 - arena.step_n / MAX_STEPS)
        else:
            r = R_LOSE
        if rew_buf[i]:
            rew_buf[i][-1] += r  # 终局奖罚记在最后一步
        # v12: GAE 优势（δ_t = r + γV' − V，A_t = δ + γλ·A_{t+1}）；否则 MC return-to-go
        n = len(rew_buf[i])
        if gae_lambda is not None and v_buf[i]:
            adv = [0.0] * n
            gae = 0.0
            for t in range(n - 1, -1, -1):
                v_next = v_buf[i][t + 1] if t + 1 < n else 0.0
                delta = rew_buf[i][t] + gamma * v_next - v_buf[i][t]
                gae = delta + gamma * gae_lambda * gae
                adv[t] = gae
            rets.append([adv[t] + v_buf[i][t] for t in range(n)])   # 价值目标
            advs.append(adv)
        else:
            R = 0.0
            adv = [0.0] * n
            for t in range(n - 1, -1, -1):
                R = rew_buf[i][t] + gamma * R
                adv[t] = R
            rets.append(adv)
            advs.append(None)
    outcome = "draw" if arena.winner is None else ("t0" if arena.winner is arena.tanks[0] else "t1")
    return (obs_buf, act_buf, rets, advs, logp_buf, outcome,
            arena.step_n, arena.stats["hits"], stuck_n / max(1, move_n))


def batch_loss_grad(net, X, Y, A):
    """loss = -logP(a)*A - β*H(P)（均值）。返回 (标量loss, dWs, dbs)。"""
    P, cache = net.forward(X)
    B = len(Y)
    logp = np.log(P[np.arange(B), Y] + 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(axis=1)
    loss = float(-(logp * A).mean() - ENTROPY_BETA * H.mean())
    # d/dz [-A·logp_a] = A·(P − onehot)；d/dz [-βH] = β·P·(logP + H)
    dZ = P.copy()
    dZ[np.arange(B), Y] -= 1.0
    dZ *= A[:, None]
    dZ += ENTROPY_BETA * P * (np.log(P + 1e-12) + H[:, None])
    dZ /= B
    dWs, dbs = [0] * 3, [0] * 3
    for i in (2, 1, 0):
        h_in, _z = cache[i]
        dWs[i] = h_in.T @ dZ
        dbs[i] = dZ.sum(axis=0)
        if i > 0:
            dh = dZ @ net.Ws[i].T
            dZ = dh * (cache[i - 1][1] > 0)
    # 梯度裁剪（全局范数）
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in dWs + dbs))
    if norm > CLIP_NORM:
        scale = CLIP_NORM / norm
        dWs = [g * scale for g in dWs]
        dbs = [g * scale for g in dbs]
    return loss, dWs, dbs


# ---------------- v8: Actor-Critic（一个躯干两个头） ----------------
def attach_critic(net, d=None, seed=1):
    """给共享躯干挂价值头 V(s)=h2@Wv+bv。有存档则加载，否则小随机初始化。"""
    if d is not None and "wv" in d:
        net.Wv, net.bv = d["wv"], d["bv"]
    else:
        rng = np.random.default_rng(seed)
        # v25：隐宽从网络形状推导（宽 MLP 存档照常挂价值头）
        h2 = net.sizes[2] if getattr(net, "sizes", None) else 128
        net.Wv = rng.standard_normal((h2, 1)) * 0.01
        net.bv = np.zeros(1)


def forward_ac(net, X):
    """一次前向，两个头：P(动作概率) + V(局面分)。cache[2][0] 就是共享躯干的出口 h2。"""
    P, cache = net.forward(X)
    h2 = cache[2][0]
    V = h2 @ net.Wv + net.bv          # (B,1)
    return P, V[:, 0], cache


def batch_loss_grad_ac(net, X, Y, A, R, value_coef):
    """演员 loss 照旧；评论员 loss = ½(V−R)²；两头的锅在 h2 汇合后一起逆流。
    返回 (loss, dWs, dbs, dWv, dbv, 平均|V误差|)。"""
    P, V, cache = forward_ac(net, X)
    B = len(Y)
    # —— 演员（与 batch_loss_grad 相同） ——
    logp = np.log(P[np.arange(B), Y] + 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(axis=1)
    loss_pg = float(-(logp * A).mean() - ENTROPY_BETA * H.mean())
    dZ = P.copy()
    dZ[np.arange(B), Y] -= 1.0
    dZ *= A[:, None]
    dZ += ENTROPY_BETA * P * (np.log(P + 1e-12) + H[:, None])
    dZ /= B
    # —— 评论员：½(V−R)² → dV=(V−R)/B ——
    dV = (V - R)[:, None] / B
    loss_v = float(0.5 * ((V - R) ** 2).mean())
    h2 = cache[2][0]
    dWv = h2.T @ dV * value_coef
    dbv = dV.sum(axis=0) * value_coef
    dh2_from_critic = (dV @ net.Wv.T) * value_coef   # 评论员给躯干的锅
    # —— 逆流：h2 处两锅合并 ——
    dWs, dbs = [0] * 3, [0] * 3
    for i in (2, 1, 0):
        h_in, _z = cache[i]
        dWs[i] = h_in.T @ dZ
        dbs[i] = dZ.sum(axis=0)
        if i > 0:
            dh = dZ @ net.Ws[i].T
            if i == 2:
                dh = dh + dh2_from_critic            # ← 汇合点
            dZ = dh * (cache[i - 1][1] > 0)
    # 梯度裁剪（含价值头，全局范数）
    all_g = dWs + dbs + [dWv, dbv]
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in all_g))
    if norm > CLIP_NORM:
        scale = CLIP_NORM / norm
        all_g = [g * scale for g in all_g]
        dWs, dbs, dWv, dbv = all_g[:3], all_g[3:6], all_g[6], all_g[7]
    return loss_pg + value_coef * loss_v, dWs, dbs, dWv, dbv, float(np.abs(V - R).mean())


class AdamAC(Adam):
    """Adam 扩展到 8 个张量（+Wv,bv）。"""
    def __init__(self, net, lr=1e-4):
        super().__init__(net, lr)
        self.m += [np.zeros_like(net.Wv), np.zeros_like(net.bv)]
        self.v += [np.zeros_like(net.Wv), np.zeros_like(net.bv)]

    def step(self, net, dWs, dbs, dWv, dbv):
        self.t += 1
        params = net.Ws + net.bs + [net.Wv, net.bv]
        grads = dWs + dbs + [dWv, dbv]
        for i in range(len(params)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grads[i] ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------- v12: PPO（同一批数据学 K 轮 + 信任域裁剪） ----------------
def batch_loss_grad_ppo(net, X, Y, A, logp_old, R, value_coef, eps_clip):
    """PPO 裁剪代理目标：min(ratio·A, clip(ratio,1±ε)·A)。
    梯度权重 w = ratio·A·mask（mask=走未裁剪支路；被裁=步子太大，该步梯度归零=刹车）。
    返回 (loss, dWs, dbs, dWv, dbv, V误差, 裁剪率)。"""
    P, V, cache = forward_ac(net, X)
    B = len(Y)
    logp = np.log(P[np.arange(B), Y] + 1e-12)
    ratio = np.exp(logp - logp_old)
    s1 = ratio * A
    s2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * A
    mask = s1 <= s2                      # min 走哪条支路（s2 被裁时常数化→梯度归零）
    w = ratio * A * mask
    H = -(P * np.log(P + 1e-12)).sum(axis=1)
    loss_pg = float(-np.minimum(s1, s2).mean() - ENTROPY_BETA * H.mean())
    dZ = P.copy()
    dZ[np.arange(B), Y] -= 1.0
    dZ *= w[:, None]
    dZ += ENTROPY_BETA * P * (np.log(P + 1e-12) + H[:, None])
    dZ /= B
    dV = (V - R)[:, None] / B
    loss_v = float(0.5 * ((V - R) ** 2).mean())
    h2 = cache[2][0]
    dWv = h2.T @ dV * value_coef
    dbv = dV.sum(axis=0) * value_coef
    dh2_from_critic = (dV @ net.Wv.T) * value_coef
    dWs, dbs = [0] * 3, [0] * 3
    for i in (2, 1, 0):
        h_in, _z = cache[i]
        dWs[i] = h_in.T @ dZ
        dbs[i] = dZ.sum(axis=0)
        if i > 0:
            dh = dZ @ net.Ws[i].T
            if i == 2:
                dh = dh + dh2_from_critic
            dZ = dh * (cache[i - 1][1] > 0)
    all_g = dWs + dbs + [dWv, dbv]
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in all_g))
    if norm > CLIP_NORM:
        scale = CLIP_NORM / norm
        all_g = [g * scale for g in all_g]
        dWs, dbs, dWv, dbv = all_g[:3], all_g[3:6], all_g[6], all_g[7]
    return (loss_pg + value_coef * loss_v, dWs, dbs, dWv, dbv,
            float(np.abs(V - R).mean()), float(1 - mask.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=64, help="每轮自对弈局数")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--init", default=BC_W)
    ap.add_argument("--out", default=OUT_W)
    ap.add_argument("--stuck-penalty", type=float, default=0.01)
    ap.add_argument("--idle-penalty", type=float, default=0.01)
    ap.add_argument("--idle-thresh", type=float, default=3.0, help="站桩判定位移阈值（像素/10步）")
    ap.add_argument("--close-bonus", type=float, default=0.0005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--draw-penalty", type=float, default=0.05, help="平局罚分（正数=罚）")
    ap.add_argument("--los-bonus", type=float, default=0.0, help="射界奖/步（我能打敌且敌打不到我）")
    ap.add_argument("--blind-penalty", type=float, default=0.0, help="盲射罚/炮（射界被挡还开火）")
    ap.add_argument("--fire-cost", type=float, default=0.0, help="弹药费/炮（无条件，spam经济学死刑）")
    ap.add_argument("--value-coef", type=float, default=0.0, help=">0 开启 critic（评论员loss系数，荐0.5）")
    ap.add_argument("--speed-bonus", type=float, default=0.0, help="速胜奖系数：胜时 +(1−步数/上限)×系数")
    ap.add_argument("--pool-prob", type=float, default=0.0, help="联赛：每局从对手池抽冻结历史脑的概率")
    ap.add_argument("--ppo-epochs", type=int, default=0, help=">0 开启 PPO：每批数据学 K 轮（荐4，需配合critic）")
    ap.add_argument("--ppo-eps", type=float, default=0.2, help="PPO 信任域裁剪半径")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE 平滑系数（PPO 模式生效）")
    ap.add_argument("--hit-reward", type=float, default=0.2, help="命中奖/挨打罚的绝对值（v14 地形改造）")
    ap.add_argument("--stall-coef", type=float, default=0.0, help="僵局流血系数（v17：每步 −coef×t/上限）")
    ap.add_argument("--destructible", action="store_true", help="掩体可摧毁（v15 几何手术）")
    ap.add_argument("--obs-v2", action="store_true",
                    help="v27 观测升级：37→40 维（敌方速度 vx/vy + 敌方冷却）")
    args = ap.parse_args()

    out_log = args.out.replace(".npz", "-log.txt")

    d = np.load(args.init)
    net = Net(seed=0)
    net.Ws = [d[f"w{i}"] for i in range(3)]
    net.bs = [d[f"b{i}"] for i in range(3)]
    if args.obs_v2 and net.Ws[0].shape[0] == 37:
        # 火种扩列：旧 37 维权重搬进新 40 维，新增 3 列置零
        # → 初始策略与旧冠军逐位相同（新输入先不说话），之后由梯度自己决定怎么用
        net.Ws[0] = np.vstack([net.Ws[0], np.zeros((3, net.Ws[0].shape[1]))])
    global R_HIT_DEALT, R_HIT_TAKEN
    R_HIT_DEALT, R_HIT_TAKEN = args.hit_reward, -args.hit_reward
    use_critic = args.value_coef > 0
    if args.ppo_epochs > 0 and not use_critic:
        ap.error("--ppo-epochs 需要 --value-coef > 0（PPO 必须有评论员）")
    if use_critic:
        attach_critic(net, d)
        opt = AdamAC(net, lr=args.lr)
    else:
        opt = Adam(net, lr=args.lr)

    # v10: 对手池（历代最强冻结脑）——蹲坑的天敌：它们不全都会"自己走过来"
    POOL_FILES = ["bc-v1.npz", "rl-v1.npz", "rl-v2-best140.npz", "rl-v3.npz",
                  "rl-v4-best140.npz", "rl-v7.npz"]
    pool = []
    if args.pool_prob > 0:
        for f in POOL_FILES:
            p = os.path.join(HERE, "..", "data", f)
            if not os.path.exists(p):
                continue
            pd_ = np.load(p)
            pn = Net(seed=0)
            pn.Ws = [pd_[f"w{i}"] for i in range(3)]
            pn.bs = [pd_[f"b{i}"] for i in range(3)]
            pool.append(pn)
    rng = np.random.default_rng(7)
    logf = open(out_log, "a", encoding="utf-8")
    best_score = -1e9
    t0 = time.time()

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"=== RL 自对弈开始 {time.strftime('%H:%M:%S')} init={os.path.basename(args.init)} "
        f"iters={args.iters} games/iter={args.games} lr={args.lr} ===")
    for it in range(1, args.iters + 1):
        # ① 打一批局
        Xs, Ys, As = [], [], []
        LPs, Rs = [], []             # PPO：旧策略 logp（信任域的锚）+ 价值目标
        n_draw = 0
        hits_tot = steps_tot = 0
        stuck_tot = []
        gae = args.gae_lambda if args.ppo_epochs > 0 else None
        for _ in range(args.games):
            foe_net = pool[int(rng.integers(0, len(pool)))] if pool and rng.random() < args.pool_prob else None
            obs_buf, act_buf, ret_buf, adv_buf, logp_buf, outcome, steps, hits, stuck = play_game(
                net, rng, args.stuck_penalty, args.close_bonus, args.gamma, args.idle_penalty,
                idle_thresh=args.idle_thresh, draw_penalty=args.draw_penalty, los_bonus=args.los_bonus,
                blind_penalty=args.blind_penalty, fire_cost=args.fire_cost, speed_bonus=args.speed_bonus,
                foe_net=foe_net, gae_lambda=gae, destructible=args.destructible,
                stall_coef=args.stall_coef, obs_v2=args.obs_v2)
            sides = (0,) if foe_net is not None else (0, 1)   # 联赛局只训练蓝方（冻结脑的经验不回收）
            for i in sides:
                if act_buf[i]:
                    Xs.extend(obs_buf[i])
                    Ys.extend(act_buf[i])
                    if gae is not None:
                        As.extend(adv_buf[i])        # GAE 优势（基于旧策略 V）
                        LPs.extend(logp_buf[i])      # 旧策略 logp
                        Rs.extend(ret_buf[i])        # 价值目标 = A + V_old
                    else:
                        As.extend(ret_buf[i])
            n_draw += outcome == "draw"
            hits_tot += sum(hits)
            steps_tot += steps
            stuck_tot.append(stuck)
        X = np.asarray(Xs, dtype=np.float64)
        Y = np.asarray(Ys)
        R_all = np.asarray(Rs, dtype=np.float64)
        clipfrac = None
        if args.ppo_epochs > 0:
            # PPO：A 用 GAE（已含 V 基线）→ 批归一化；同批数据学 K 轮，信任域裁剪防崩
            A = np.asarray(As, dtype=np.float64)
            A = (A - A.mean()) / (A.std() + 1e-8)
            LP = np.asarray(LPs, dtype=np.float64)
            for ep in range(args.ppo_epochs):
                loss, dWs, dbs, dWv, dbv, verr, clipfrac = batch_loss_grad_ppo(
                    net, X, Y, A, LP, R_all, args.value_coef, args.ppo_eps)
                opt.step(net, dWs, dbs, dWv, dbv)
        else:
            R_all = np.asarray(As, dtype=np.float64)
            if use_critic:
                # A = 回报 − 评论员的预期（随局面精准定价），再做批归一化
                _, V_pred, _ = forward_ac(net, X)
                A = R_all - V_pred
                A = (A - A.mean()) / (A.std() + 1e-8)
            else:
                A = (R_all - R_all.mean()) / (R_all.std() + 1e-8)   # 批次内归一化 = 粗糙 baseline
            # ② 拧一次旋钮（全批量）
            if use_critic:
                loss, dWs, dbs, dWv, dbv, verr = batch_loss_grad_ac(net, X, Y, A, R_all, args.value_coef)
                opt.step(net, dWs, dbs, dWv, dbv)
            else:
                loss, dWs, dbs = batch_loss_grad(net, X, Y, A)
                opt.step(net, dWs, dbs)

        vcol = f" | V误差 {verr:.3f}" if use_critic else ""
        ccol = f" | 裁剪率 {clipfrac*100:.0f}%" if clipfrac is not None else ""
        log(f"iter {it:3d} | loss {loss:+.4f} | 僵局 {n_draw}/{args.games} "
            f"| 局均命中 {hits_tot/args.games:.1f} | 局均步 {steps_tot/args.games:.0f} "
            f"| 顶墙率 {np.mean(stuck_tot)*100:.1f}%{vcol}{ccol} | {time.time()-t0:.0f}s")

        # ③ 月考
        if it % args.eval_every == 0:
            class _B:
                def act(self, obs, sample=False, rng=None):
                    P, _ = net.forward(np.asarray(obs, dtype=np.float64)[None])
                    return int(P[0].argmax()), None
            wa, dr, wb, hr, st = match(_B(), TeacherBrain(), 30, seed=50000 + it, destructible=args.destructible, obs_v2=args.obs_v2)
            wr, dr2, wl, hr2, _ = match(_B(), type("R", (), {"act": lambda self, o, sample=False, rng=None:
                                      (int(np.random.default_rng().integers(0, 5)), None)})(), 30, seed=60000 + it, obs_v2=args.obs_v2)
            score = (wa - wb) + 0.5 * (wr - wl)
            log(f"  📋 月考 @{it}: vs老师 {wa}胜{dr}平{wb}负(命中{hr*100:.0f}%) "
                f"vs随机 {wr}胜{dr2}平{wl}负 | score {score:+.0f}")
            if score > best_score:
                best_score = score
                save = {f"w{i}": w for i, w in enumerate(net.Ws)}
                save.update({f"b{i}": b for i, b in enumerate(net.bs)})
                if use_critic:
                    save.update({"wv": net.Wv, "bv": net.bv})
                np.savez(args.out, **save)
                log(f"  💾 新最佳，权重已存 {args.out}")
    log("=== 训练结束 ===")
    logf.close()


if __name__ == "__main__":
    main()
