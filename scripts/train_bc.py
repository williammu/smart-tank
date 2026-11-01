#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线C 第②步：行为克隆（SL）——手写反向传播训练 37→128→128→5。

全流程就是对话里推过的那套：
  前向记账（存 z、h、输入）→ 加权交叉熵 loss → P−y 起手 → 锅逆流三关 → Adam 拧旋钮。

军规：先跑 --grad-check（数值梯度校验），过了才准 --train。
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "dataset-v1.npz")
OUT_W = os.path.join(HERE, "..", "data", "bc-v1.npz")      # 训练出的权重（gitignored）
OUT_LOG = os.path.join(HERE, "..", "data", "bc-v1-log.txt")

ACTION_NAMES = ["左转", "右转", "前进", "后退", "开火"]
SIZES = (37, 128, 128, 5)


# ---------------- 网络：前向记账 + 反向分锅 ----------------
class Net:
    def __init__(self, seed=0, sizes=SIZES, hidden1=128, hidden2=128):
        # v25：hidden1/hidden2 可配置隐层宽度（不传 = 默认 37→128→128→5 零改动）
        if (hidden1, hidden2) != (128, 128):
            sizes = (sizes[0], hidden1, hidden2, sizes[3])
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.Ws, self.bs = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # He 初始化：把每层"音量"校准到 1（防梯度消失/爆炸）
            self.Ws.append(rng.standard_normal((fan_in, sizes[i + 1])) * np.sqrt(2.0 / fan_in))
            self.bs.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        """X: (B,37) → (P(B,5), 缓存)。小本本：每层的输入、z（ReLU前）、h。"""
        cache = []
        h = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = h @ W + b
            last = i == len(self.Ws) - 1
            out = z if last else np.maximum(z, 0)  # 输出层不过 ReLU
            cache.append((h, z))                   # 记下：这层的输入、ReLU前的z
            h = out
        z3 = h
        z3 = z3 - z3.max(axis=1, keepdims=True)    # 数值稳定
        e = np.exp(z3)
        P = e / e.sum(axis=1, keepdims=True)
        return P, cache

    def backward(self, P, y, cw, cache):
        """锅逆流：dZ3=(P−onehot)×类别权重/B，逐层分锅。返回 (dWs, dbs)。"""
        B = len(y)
        dZ = P.copy()
        dZ[np.arange(B), y] -= 1.0                 # P − y：起手式
        dZ *= cw[y][:, None] / B                   # 加权 + batch 平均
        dWs, dbs = [0] * 3, [0] * 3
        for i in (2, 1, 0):
            h_in, z = cache[i]
            dWs[i] = h_in.T @ dZ                   # 锅 = δ ⊗ 输入（×搭档）
            dbs[i] = dZ.sum(axis=0)                # 加法工位：锅全额照收
            if i > 0:
                dh = dZ @ self.Ws[i].T             # 锅传给上一层
                dZ = dh * (cache[i - 1][1] > 0)    # ReLU 闸门用【上一层】的 z：死了清零
        return dWs, dbs

    def loss(self, P, y, cw):
        return float(-(np.log(P[np.arange(len(y)), y] + 1e-12) * cw[y]).mean())


# ---------------- 军规：数值梯度校验 ----------------
def grad_check(net, X, y, cw, n_per_tensor=8, eps=1e-5):
    """笨办法（逐个拧动）vs 链式法则：对每个张量抽几个参数对答案。"""
    P, cache = net.forward(X)
    dWs, dbs = net.backward(P, y, cw, cache)
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, param, grad in [("W1", net.Ws[0], dWs[0]), ("W2", net.Ws[1], dWs[1]),
                              ("W3", net.Ws[2], dWs[2]), ("b2", net.bs[1], dbs[1]),
                              ("b3", net.bs[2], dbs[2])]:
        flat_p, flat_g = param.ravel(), grad.ravel()
        for idx in rng.choice(len(flat_p), size=min(n_per_tensor, len(flat_p)), replace=False):
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp = net.loss(net.forward(X)[0], y, cw)
            flat_p[idx] = orig - eps
            lm = net.loss(net.forward(X)[0], y, cw)
            flat_p[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = flat_g[idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


# ---------------- Adam 优化器 ----------------
class Adam:
    def __init__(self, net, lr=3e-4):
        self.lr, self.b1, self.b2, self.eps = lr, 0.9, 0.999, 1e-8
        self.t = 0
        self.m = [np.zeros_like(p) for p in net.Ws + net.bs]
        self.v = [np.zeros_like(p) for p in net.Ws + net.bs]

    def step(self, net, dWs, dbs):
        self.t += 1
        params = net.Ws + net.bs
        grads = dWs + dbs
        for i in range(len(params)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grads[i] ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------- 评估 ----------------
def evaluate(net, X, y, cw):
    P, _ = net.forward(X)
    pred = P.argmax(axis=1)
    acc = float((pred == y).mean())
    recalls = []
    for a in range(5):
        mask = y == a
        recalls.append(float((pred[mask] == a).mean()) if mask.sum() else float("nan"))
    return net.loss(P, y, cw), acc, recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    d = np.load(DATA)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    counts = np.bincount(ytr, minlength=5)
    cw = len(ytr) / (5.0 * counts)               # 类别权重：开火 ≈28×前进
    print(f"train {len(ytr):,} / val {len(yva):,} | 类别权重: {np.round(cw, 2)}")

    net = Net(seed=0)

    if args.grad_check:
        worst = grad_check(net, Xtr[:256], ytr[:256], cw)
        print(f"梯度校验：最大相对误差 = {worst:.2e}  {'✅ 通过' if worst < 1e-4 else '❌ 未过'}")
        return

    if args.train:
        opt = Adam(net, lr=args.lr)
        n_batches = (len(ytr) + args.batch - 1) // args.batch
        log_lines = []
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            idx = np.random.default_rng(1000 + ep).permutation(len(ytr))
            tot = 0.0
            for b in range(n_batches):
                sl = idx[b * args.batch:(b + 1) * args.batch]
                X, y = Xtr[sl], ytr[sl]
                P, cache = net.forward(X)                       # 押注（顺手记账）
                tot += net.loss(P, y, cw) * len(y)              # 扣分
                dWs, dbs = net.backward(P, y, cw, cache)        # 分锅
                opt.step(net, dWs, dbs)                         # 拧旋钮
            vl, va, vr = evaluate(net, Xva, yva, cw)
            line = (f"epoch {ep:2d} | train_loss {tot/len(ytr):.4f} | val_loss {vl:.4f} "
                    f"| val_acc {va*100:.2f}% | 每类召回: "
                    + " ".join(f"{ACTION_NAMES[a]} {vr[a]*100:.1f}%" for a in range(5))
                    + f" | {time.time()-t0:.0f}s")
            print(line)
            log_lines.append(line)
        np.savez(OUT_W, **{f"w{i}": w for i, w in enumerate(net.Ws)},
                 **{f"b{i}": b for i, b in enumerate(net.bs)})
        with open(OUT_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"权重已存: {OUT_W}")


if __name__ == "__main__":
    main()
