#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v16 · 数据工厂：28 工人打局 + 1 学徒 PPO 更新（A3C 的最后一个字母 "A" 落地）。

架构：每轮学徒把当前权重分发给 28 个进程 → 并行打局（复用 train_rl.play_game 全部
shaping + GAE 记账）→ 经验压缩包回传 → 学徒批量 PPO 更新 → best-checkpoint。
断点续训：每轮存 resume 存档（网络 + Adam m/v + 轮数 + best_score），--resume 接力。

用法：
  python train_parallel.py --workers 28 --games 256 --iters 10000 \
      --init data/rl-v14.npz --out data/rl-v16.npz --destructible \
      --hit-reward 1.0 --fire-cost 0.0 --value-coef 0.5 --ppo-epochs 4 --lr 3e-5
  中断后：同上命令 + --resume
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bc import Net  # noqa: E402
from train_rl import (play_game, attach_critic, batch_loss_grad_ppo, AdamAC,  # noqa: E402
                      CLIP_NORM, R_WIN)
from eval_bc import TeacherBrain, match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- 工人进程（模块级，可 pickle） ----------------
class BotOneHot:
    """把剧本机器人包成 play_game 的 foe_net 接口：确定性动作 = one-hot 分布。"""
    def __init__(self, bot):
        self._b = bot

    def forward(self, x):
        P = np.zeros((1, 5)); P[0, self._b.decide(x[0])] = 1.0
        return P, None


def worker_games(payload):
    """一个工人打 n 局，返回压缩经验包。复用 train_rl.play_game 的完整奖励表与 GAE。
    v18: vs_teacher 模式——红方=永不漂移的规则老师，河流永不死，只训练蓝方。"""
    wsd, seed, n_games, cfg = payload
    # v25：隐宽从分发权重 shape 推导（旧 128 与宽 MLP 都能跑）
    net = Net(seed=0, hidden1=wsd["w0"].shape[1], hidden2=wsd["w1"].shape[1])
    net.Ws = [wsd[f"w{i}"] for i in range(3)]
    net.bs = [wsd[f"b{i}"] for i in range(3)]
    attach_critic(net, wsd)
    rng = np.random.default_rng(seed)
    use_bots = cfg.get("vs_teacher")
    foe = None
    packs, stats = [], dict(draw=0, hits=0, steps=0, stuck=[])
    for _ in range(n_games):
        if use_bots:                      # v18: 每局随机抽一个剧本对手（多样性=蹲坑克星）
            from bot_roster import BOTS
            foe = BotOneHot(BOTS[int(rng.integers(0, len(BOTS)))][1]())
        obs_buf, act_buf, ret_buf, adv_buf, logp_buf, outcome, steps, hits, stuck = play_game(
            net, rng, cfg["stuck_penalty"], cfg["close_bonus"], cfg["gamma"], cfg["idle_penalty"],
            idle_thresh=cfg["idle_thresh"], draw_penalty=cfg["draw_penalty"],
            los_bonus=cfg["los_bonus"], blind_penalty=cfg["blind_penalty"],
            fire_cost=cfg["fire_cost"], speed_bonus=0.0, stall_coef=cfg["stall_coef"],
            gae_lambda=cfg["gae_lambda"], destructible=cfg["destructible"],
            no_cover=cfg.get("no_cover", False), random_cover=cfg.get("random_cover", False),
            fire_idle_penalty=cfg.get("fire_idle_penalty", 0.0),
            fire_idle_thresh=cfg.get("fire_idle_thresh", 150.0),
            foe_net=foe, obs_v2=cfg.get("obs_v2", False))
        sides = (0,) if foe is not None else (0, 1)   # 老师陪练局只回收蓝方经验
        for i in sides:
            if act_buf[i]:
                packs.append((np.asarray(obs_buf[i], dtype=np.float32),
                              np.asarray(act_buf[i], dtype=np.int8),
                              np.asarray(adv_buf[i], dtype=np.float32),
                              np.asarray(logp_buf[i], dtype=np.float32),
                              np.asarray(ret_buf[i], dtype=np.float32)))
        stats["draw"] += outcome == "draw"
        stats["hits"] += sum(hits)
        stats["steps"] += steps
        stats["stuck"].append(stuck)
    return packs, stats


def weights_dict(net):
    d = {f"w{i}": w for i, w in enumerate(net.Ws)}
    d.update({f"b{i}": b for i, b in enumerate(net.bs)})
    d.update({"wv": net.Wv, "bv": net.bv})
    return d


def _chk_width(d, width1, width2):
    """v25：校验存档隐宽与 --width1/--width2 一致（错配即早失败，别等 matmul 炸）。"""
    if d["w0"].shape[1] != width1 or d["w1"].shape != (width1, width2) \
            or d["w0"].shape[0] not in (37, 40):
        raise SystemExit(f"存档隐宽 w0{d['w0'].shape} w1{d['w1'].shape} 与 "
                         f"--width1 {width1} --width2 {width2} 不符")


def save_resume(path, net, opt, t_adam, it, best_score):
    d = weights_dict(net)
    d.update({f"adam_m{k}": m for k, m in enumerate(opt.m)})
    d.update({f"adam_v{k}": v for k, v in enumerate(opt.v)})
    d.update({"t_adam": np.array(t_adam), "iter": np.array(it), "best": np.array(best_score)})
    np.savez(path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--games", type=int, default=256, help="每轮全局局数（分摊给工人）")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--init", default=os.path.join(HERE, "..", "data", "rl-v14.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "rl-v16.npz"))
    ap.add_argument("--resume", action="store_true")
    # 奖励表（默认 = v14 验证配方 + v15 物理）
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
    ap.add_argument("--destructible", action="store_true")
    ap.add_argument("--obs-v2", action="store_true",
                    help="v27 观测升级：37→40 维（敌方速度 + 敌方冷却）")
    ap.add_argument("--init-new-std", type=float, default=0.0,
                    help="扩列时新增列的初始化标准差（0=置零；>0=小随机，逼迫网络一开始就读新特征）")
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-eps", type=float, default=0.2)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--stall-coef", type=float, default=0.0)
    ap.add_argument("--vs-teacher", action="store_true", help="v18：红方=规则老师陪练（非自对弈）")
    ap.add_argument("--no-cover", action="store_true", help="v19：裸场地，无掩体")
    ap.add_argument("--random-cover", action="store_true", help="v25：每局随机掩体布局")
    ap.add_argument("--width1", type=int, default=128, help="v25：隐层1宽度")
    ap.add_argument("--width2", type=int, default=128, help="v25：隐层2宽度")
    ap.add_argument("--bc-lambda", type=float, default=0.0, help="v21：BC拴绳强度（AlphaStar式锚定）")
    ap.add_argument("--bc-data", default=os.path.join(HERE, "..", "data", "bc-nocover.npz"))
    ap.add_argument("--bc-anneal", default=None, help="v22：退火 start,end,iters 例 0.5,0.05,2000")
    ap.add_argument("--fire-idle-penalty", type=float, default=0.0, help="v26：哑火罚/步")
    ap.add_argument("--fire-idle-thresh", type=float, default=150.0, help="哑火豁免步数")
    args = ap.parse_args()

    import train_rl
    train_rl.R_HIT_DEALT, train_rl.R_HIT_TAKEN = args.hit_reward, -args.hit_reward

    out_log = args.out.replace(".npz", "-log.txt")
    resume_path = args.out.replace(".npz", "-resume.npz")
    logf = open(out_log, "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    net = Net(seed=0, hidden1=args.width1, hidden2=args.width2)
    start_it, best_score, t_adam = 0, -1e9, 0
    if args.resume and os.path.exists(resume_path):
        d = dict(np.load(resume_path))
        _chk_width(d, args.width1, args.width2)
        net.Ws = [d[f"w{i}"] for i in range(3)]
        net.bs = [d[f"b{i}"] for i in range(3)]
        attach_critic(net, d)
        opt = AdamAC(net, lr=args.lr)
        opt.m = [d[f"adam_m{k}"] for k in range(len(opt.m))]
        opt.v = [d[f"adam_v{k}"] for k in range(len(opt.v))]
        t_adam = int(d["t_adam"]); start_it = int(d["iter"]); best_score = float(d["best"])
        opt.t = t_adam
        log(f"=== 断点续训 @{start_it}（best {best_score:+.0f}）===")
    else:
        d = dict(np.load(args.init))
        _chk_width(d, args.width1, args.width2)
        if args.obs_v2 and d["w0"].shape[0] == 37:
            # 火种扩列：37→40。默认置零（起点=旧冠军逐位一致）；
            # --init-new-std>0 时用小随机数，逼迫网络从第一步就开始读新特征
            if args.init_new_std > 0:
                rng0 = np.random.default_rng(123)
                extra = rng0.normal(0, args.init_new_std, (3, d["w0"].shape[1]))
            else:
                extra = np.zeros((3, d["w0"].shape[1]))
            d["w0"] = np.vstack([d["w0"], extra])
            log(f"观测升级 37→40：旧权重扩列，新增 3 列 sd={args.init_new_std}")
        net.Ws = [d[f"w{i}"] for i in range(3)]
        net.bs = [d[f"b{i}"] for i in range(3)]
        attach_critic(net, d if "wv" in d else None)
        opt = AdamAC(net, lr=args.lr)

    ANNEAL = None
    if args.bc_anneal:
        ANNEAL = tuple(float(x) for x in args.bc_anneal.split(","))

    if args.bc_lambda > 0:
        _d = np.load(args.bc_data)
        DX = _d["X"].astype(np.float64); DY = _d["y"].astype(np.int64)
        if args.obs_v2 and DX.shape[1] == 37:
            # 老师拳谱是 37 维时代录的：新增 3 维补零（= 老师对这三条信息"不表态"）
            DX = np.hstack([DX, np.zeros((len(DX), 3))])
            log(f"拴绳拳谱扩列 37→40（新增 3 维置零）")
        _cnt = np.bincount(DY, minlength=5).astype(np.float64)
        DCW = np.clip(len(DY) / (5 * _cnt + 1e-9), 0.5, 5.0)   # 类别再平衡
        log(f"拴绳拳谱 {len(DY)} 张 λ={args.bc_lambda} 类权重 {np.round(DCW,2)}")

    cfg = dict(stuck_penalty=args.stuck_penalty, close_bonus=args.close_bonus, gamma=args.gamma,
               idle_penalty=args.idle_penalty, idle_thresh=args.idle_thresh,
               draw_penalty=args.draw_penalty, los_bonus=args.los_bonus,
               blind_penalty=args.blind_penalty, fire_cost=args.fire_cost,
               gae_lambda=args.gae_lambda, destructible=args.destructible,
               stall_coef=args.stall_coef, vs_teacher=args.vs_teacher,
               no_cover=args.no_cover, random_cover=args.random_cover, obs_v2=args.obs_v2,
               fire_idle_penalty=args.fire_idle_penalty, fire_idle_thresh=args.fire_idle_thresh)

    per_worker = max(1, args.games // args.workers)
    t0 = time.time()
    log(f"=== 数据工厂开工 {time.strftime('%H:%M:%S')} 工人={args.workers} 局/轮={per_worker*args.workers} "
        f"目标={args.iters}轮 lr={args.lr} ===")

    with mp.Pool(args.workers) as pool:
        for it in range(start_it + 1, args.iters + 1):
            # ① 分发权重，并行打局
            wsd = weights_dict(net)
            payloads = [(wsd, int(1e9) + it * 100 + w, per_worker, cfg) for w in range(args.workers)]
            results = pool.map(worker_games, payloads)
            # ② 汇总
            Xs, Ys, As, LPs, Rs = [], [], [], [], []
            n_draw = hits_tot = steps_tot = games = 0
            stuck_tot = []
            for packs, st in results:
                for X, Y, A, LP, R in packs:
                    Xs.append(X); Ys.append(Y); As.append(A); LPs.append(LP); Rs.append(R)
                n_draw += st["draw"]; hits_tot += st["hits"]; steps_tot += st["steps"]
                games += per_worker
                stuck_tot.extend(st["stuck"])
            X = np.concatenate(Xs).astype(np.float64)
            Y = np.concatenate(Ys).astype(np.int64)
            A = np.concatenate(As).astype(np.float64)
            LP = np.concatenate(LPs).astype(np.float64)
            R = np.concatenate(Rs).astype(np.float64)
            A = (A - A.mean()) / (A.std() + 1e-8)
            # ③ PPO 更新 K 轮（v21：每轮掺一份 BC 拴绳梯度——边打边抄拳谱）
            for ep in range(args.ppo_epochs):
                loss, dWs, dbs, dWv, dbv, verr, clipfrac = batch_loss_grad_ppo(
                    net, X, Y, A, LP, R, args.value_coef, args.ppo_eps)
                if args.bc_lambda > 0:
                    lam = args.bc_lambda
                    if ANNEAL:   # v22 退火：λ 从 start 线性放到 end（前 N 轮），此后保持 end
                        s, e, n = ANNEAL
                        lam = max(e, s + (e - s) * it / n)
                    bi = np.random.default_rng(it * 100 + ep).integers(0, len(DX), 4096)
                    P, cache = net.forward(DX[bi])
                    dWs_bc, dbs_bc = net.backward(P, DY[bi], DCW, cache)
                    for k in range(3):
                        dWs[k] = dWs[k] + lam * dWs_bc[k]
                        dbs[k] = dbs[k] + lam * dbs_bc[k]
                opt.step(net, dWs, dbs, dWv, dbv)
            t_adam = opt.t
            # ④ 断点存档（每轮，原子感：先写临时再替换；np.savez 会自动加 .npz 后缀）
            tmp = resume_path + ".tmp.npz"
            save_resume(tmp, net, opt, t_adam, it, best_score)
            os.replace(tmp, resume_path)
            if it % 10 == 0:
                log(f"iter {it:5d} | 僵局 {n_draw}/{games} | 局均命中 {hits_tot/games:.1f} "
                    f"| 局均步 {steps_tot/games:.0f} | 顶墙率 {np.mean(stuck_tot)*100:.1f}% "
                    f"| V误差 {verr:.3f} | 裁剪率 {clipfrac*100:.0f}% | {time.time()-t0:.0f}s")
            # ⑤ 月考（best-checkpoint 军规）
            if it % args.eval_every == 0:
                class _B:
                    def act(self, obs, sample=False, rng=None):
                        P, _ = net.forward(np.asarray(obs, dtype=np.float64)[None])
                        return int(P[0].argmax()), None
                wa, dr, wb, hr, st = match(_B(), TeacherBrain(), 30, seed=50000 + it,
                                           destructible=args.destructible, obs_v2=args.obs_v2)
                rnd = type("R", (), {"act": lambda self, o, sample=False, rng=None:
                                     (int(np.random.default_rng().integers(0, 5)), None)})()
                wr, dr2, wl, hr2, _ = match(_B(), rnd, 30, seed=60000 + it,
                                            destructible=args.destructible, obs_v2=args.obs_v2)
                score = (wa - wb) + 0.5 * (wr - wl)
                log(f"  📋 月考 @{it}: vs老师 {wa}胜{dr}平{wb}负(命中{hr*100:.0f}%) "
                    f"vs随机 {wr}胜{dr2}平{wl}负 | score {score:+.0f}")
                if score > best_score:
                    best_score = score
                    tmp_best = args.out + ".tmp.npz"          # 原子写：断电任意瞬间无半成品
                    np.savez(tmp_best, **weights_dict(net))
                    os.replace(tmp_best, args.out)
                    log(f"  💾 新最佳，权重已存 {args.out}")
    log("=== 数据工厂收工 ===")
    logf.close()


if __name__ == "__main__":
    main()
