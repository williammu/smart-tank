#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 网络（手写，纯 numpy）：编码器(37→128, 可继承 bc-v1) + LSTM(128→64) + 双头(政策5/价值1)。

门：i=输入门 f=遗忘门 g=候选 o=输出门（sigmoid/sigmoid/tanh/sigmoid）
  z = x@Wxh + h@Whh + b   （一次性算 4 个门，再切开）
  c' = f⊙c + i⊙g          （遗忘门=记忆的橡皮擦，输入门=写字的笔）
  h' = o⊙tanh(c')         （输出门=此刻愿意说出多少）

BPTT：锅从最后一步逆流回第一步；dh 走两条路（直接输出 + 经 c 绕路），dc 只走细胞线。
"""
import numpy as np

ENC_IN, ENC_OUT, LSTM_H = 37, 128, 64


class LSTMNet:
    def __init__(self, seed=0, enc_from=None):
        rng = np.random.default_rng(seed)
        # 编码器：可继承 bc-v1 的 W0/b0（视觉皮层），否则 He 初始化
        if enc_from is not None:
            d = np.load(enc_from)
            self.W0, self.b0 = d["w0"].copy(), d["b0"].copy()
        else:
            self.W0 = rng.standard_normal((ENC_IN, ENC_OUT)) * np.sqrt(2.0 / ENC_IN)
            self.b0 = np.zeros(ENC_OUT)
        n_in = ENC_OUT + LSTM_H
        # LSTM：单个打包权重 (128+64, 256)；遗忘门偏置初始化为 1（经典技巧：默认"记得"）
        self.Wl = rng.standard_normal((n_in, 4 * LSTM_H)) * np.sqrt(1.0 / n_in)
        self.bl = np.zeros(4 * LSTM_H)
        self.bl[LSTM_H:2 * LSTM_H] = 1.0
        self.Wp = rng.standard_normal((LSTM_H, 5)) * np.sqrt(2.0 / LSTM_H)   # 政策头
        self.bp = np.zeros(5)
        self.Wv = rng.standard_normal((LSTM_H, 1)) * 0.01                     # 价值头
        self.bv = np.zeros(1)

    def params(self):
        return [self.W0, self.b0, self.Wl, self.bl, self.Wp, self.bp, self.Wv, self.bv]

    def step(self, x, h, c):
        """单步前向。x:(37,) h,c:(64,) → (P(5,), V标量, h', c', cache)"""
        h1 = np.maximum(x @ self.W0 + self.b0, 0)          # 编码器+ReLU
        z = h1 @ self.Wl[:ENC_OUT] + h @ self.Wl[ENC_OUT:] + self.bl
        i = 1 / (1 + np.exp(-z[:LSTM_H]))
        f = 1 / (1 + np.exp(-z[LSTM_H:2 * LSTM_H]))
        g = np.tanh(z[2 * LSTM_H:3 * LSTM_H])
        o = 1 / (1 + np.exp(-z[3 * LSTM_H:]))
        c2 = f * c + i * g
        tc = np.tanh(c2)
        h2 = o * tc
        zp = h2 @ self.Wp + self.bp
        zp = zp - zp.max()
        e = np.exp(zp)
        P = e / e.sum()
        V = float(h2 @ self.Wv[:, 0] + self.bv[0])
        cache = (x, h1, h, c, i, f, g, o, c2, tc, h2)
        return P, V, h2, c2, cache

    def bptt(self, caches, dP_list, dV_list, T):
        """随时间反向传播。dP_list[t]: (5,) 政策头起手δ；dV_list[t]: 标量 V 的 δ（已含系数）。
        返回 8 个梯度张量（与 params() 对齐，累加语义：调用方先清零）。"""
        gW0 = np.zeros_like(self.W0); gb0 = np.zeros_like(self.b0)
        gWl = np.zeros_like(self.Wl); gbl = np.zeros_like(self.bl)
        gWp = np.zeros_like(self.Wp); gbp = np.zeros_like(self.bp)
        gWv = np.zeros_like(self.Wv); gbv = np.zeros_like(self.bv)
        dh_next = np.zeros(LSTM_H); dc_next = np.zeros(LSTM_H)
        for t in range(T - 1, -1, -1):
            (x, h1, h, c, i, f, g, o, c2, tc, h2) = caches[t]
            # 双头的锅
            gWp += np.outer(h2, dP_list[t]); gbp += dP_list[t]
            gWv += np.outer(h2, [dV_list[t]]); gbv += dV_list[t]
            dh = self.Wp @ dP_list[t] + self.Wv[:, 0] * dV_list[t] + dh_next
            # 门内逆流
            dc = dc_next + dh * o * (1 - tc ** 2)
            do = dh * tc * o * (1 - o)
            di = dc * g * i * (1 - i)
            df = dc * c * f * (1 - f)
            dg = dc * i * (1 - g ** 2)
            dz = np.concatenate([di, df, dg, do])
            gWl[:ENC_OUT] += np.outer(h1, dz)
            gWl[ENC_OUT:] += np.outer(h, dz)
            gbl += dz
            dh1 = dz @ self.Wl[:ENC_OUT].T
            # 编码器逆流（ReLU 闸门：h1>0 才放行）
            dz0 = dh1 * (h1 > 0)
            gW0 += np.outer(x, dz0); gb0 += dz0
            dh_next = dz @ self.Wl[ENC_OUT:].T
            dc_next = dc * f
        return [gW0, gb0, gWl, gbl, gWp, gbp, gWv, gbv]

    def save(self, path):
        np.savez(path, w0=self.W0, b0=self.b0, wl=self.Wl, bl=self.bl,
                 wp=self.Wp, bp=self.bp, wv=self.Wv, bv=self.bv)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(seed=0)
        net.W0, net.b0 = d["w0"], d["b0"]
        net.Wl, net.bl = d["wl"], d["bl"]
        net.Wp, net.bp = d["wp"], d["bp"]
        net.Wv, net.bv = d["wv"], d["bv"]
        return net


class LSTMBrain:
    """有记忆的战斗脑：局内携带 (h,c)，局间 reset。eval_bc.match 会在每局开头调 reset。"""
    def __init__(self, path):
        self.net = LSTMNet.load(path)
        self.reset()

    def reset(self):
        self.h = np.zeros(LSTM_H)
        self.c = np.zeros(LSTM_H)

    def act(self, obs, sample=False, rng=None):
        P, _, self.h, self.c, _ = self.net.step(np.asarray(obs, dtype=np.float64), self.h, self.c)
        if sample and rng is not None:
            return int(rng.choice(5, p=P / P.sum())), None
        return int(P.argmax()), None
