#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能坦克 · 竞技场 v0.1 —— 随机初始化权重演示

对应 notes/001 的网络设定：
  观测(20维) → h¹(128) → h²(128) → 输出(9个动作) → softmax → 按概率采样
两个坦克各持一份**随机初始化**的权重互打——这就是"未训练的大脑"的行为基线。
训练之后再来对比，同一份代码，只换权重。

用法：
  python arena.py --live                 # 桌面窗口实时看（需 DISPLAY）
  python arena.py --record out/x.gif     # 无头录制 GIF
  python arena.py --headless             # 纯文本跑一局，验证机制
"""
import argparse
import math
import os
import sys

import numpy as np

# ---------------- 竞技场参数 ----------------
W, H = 640, 480               # 场地尺寸（像素）
TANK_R = 12                   # 坦克半径
TANK_SPEED = 2.2              # 每步移动距离
BULLET_SPEED = 5.0
BULLET_R = 3
DAMAGE = 25                   # 命中扣血（4 发带走）
HP_MAX = 100
FIRE_COOLDOWN = 18            # 开火冷却（步）
MAX_STEPS = 1800              # 一局上限（超时算平局）

OBSTACLES = [                 # 掩体：(x, y, w, h)——均避开中线 y=240，保证出生点直线弹道贯通
    (280, 100, 80, 60),
    (280, 320, 80, 60),
    (120, 150, 60, 60),
    (460, 270, 60, 60),
]

# 5 个动作：左转 / 右转 / 前进 / 后退 / 开火
# （v0.1 教训：初版用「8方向移动+开火」，瞄准被量子化到 45°，随机命中率≈0，
#   奖励稀疏到无法学习 → 改为转向制，炮塔朝向连续，指哪打哪。见 notes/002）
ACTIONS = ["turn_left", "turn_right", "forward", "backward", "fire"]
ACTION_NAMES = ["⟲ 左转", "⟳ 右转", "⤒ 前进", "⤓ 后退", "🔥 开火"]
TURN_RATE = 0.06            # 每步转向弧度（≈3.4°）
BACK_SPEED = 1.4            # 后退比前进慢


# ---------------- 神经网络（numpy 手写，对应讨论过的结构） ----------------
class MLP:
    """20 → 128 → 128 → 9，每层 h = ReLU(W·x + b)，输出层接 softmax。"""

    def __init__(self, seed, sizes=(37, 128, 128, 5)):
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He 初始化：方差 2/n_in，让 ReLU 网络各层信号尺度稳定（随机起步的常用起点）
            self.weights.append(rng.normal(0, math.sqrt(2 / n_in), (n_out, n_in)).astype(np.float32))
            self.biases.append(np.zeros(n_out, dtype=np.float32))
        self.n_params = sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, x):
        """前向传播：就是讨论里的那句话，重复三遍。"""
        h = np.asarray(x, dtype=np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ h + b
            h = np.maximum(0, z) if i < len(self.weights) - 1 else z  # 输出层不过 ReLU
        return h  # 9 个 logits

    def act(self, x, sample=True, rng=None):
        """softmax → 按概率采样（训练/演示模式）或取 argmax（对战模式）。"""
        z = self.forward(x)
        p = np.exp(z - z.max())
        p /= p.sum()
        if sample:
            return int(rng.choice(len(p), p=p)), p
        return int(np.argmax(p)), p

    @classmethod
    def load(cls, path):
        """载入 train_bc.py 存的权重（那边按 (in,out) 存，转置回本类的 (out,in)）。
        v25：隐宽从 w1/w2 shape 自动推断，旧 128 存档与宽 MLP 存档都能加载。"""
        d = np.load(path)
        net = cls(seed=0, sizes=(d["w0"].shape[0], d["w0"].shape[1],
                                 d["w1"].shape[1], d["w2"].shape[1]))
        net.weights = [d[f"w{i}"].T.astype(np.float32) for i in range(3)]
        net.biases = [d[f"b{i}"].astype(np.float32) for i in range(3)]
        return net


# ---------------- 游戏实体 ----------------
class Tank:
    def __init__(self, x, y, color, name):
        self.x, self.y = x, y
        self.angle = 0.0          # 朝向（弧度），初始朝右
        self.hp = HP_MAX
        self.cooldown = 0
        self.color = color
        self.name = name
        self.alive = True
        self.vx, self.vy = 0.0, 0.0   # 上一步的实际位移（obs v2：敌方运动状态）

    @property
    def facing(self):
        return math.cos(self.angle), math.sin(self.angle)


class Bullet:
    def __init__(self, x, y, vx, vy, owner):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.owner = owner        # 谁打的（不自伤）
        self.alive = True


def _circle_hit_rect(cx, cy, r, rect):
    """圆（坦克）与矩形（掩体）碰撞检测。"""
    rx, ry, rw, rh = rect
    nx = max(rx, min(cx, rx + rw))
    ny = max(ry, min(cy, ry + rh))
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r


# 出生禁区：四角 110×110（随机掩体不得压入，保证出生点附近有回旋空间）
SPAWN_ZONE = 110
SPAWN_ZONES = [
    (0, 0, SPAWN_ZONE, SPAWN_ZONE),
    (W - SPAWN_ZONE, 0, SPAWN_ZONE, SPAWN_ZONE),
    (0, H - SPAWN_ZONE, SPAWN_ZONE, SPAWN_ZONE),
    (W - SPAWN_ZONE, H - SPAWN_ZONE, SPAWN_ZONE, SPAWN_ZONE),
]


def _rects_clear(rect_a, rect_b, gap=30.0):
    """两矩形（含 gap 间距）是否互不侵犯：分离轴上拉开 gap 即算通过。"""
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    return (ax + aw + gap <= bx or bx + bw + gap <= ax or
            ay + ah + gap <= by or by + bh + gap <= ay)


def _rect_hits_spawn_zone(rect):
    """矩形是否压入四角出生禁区。"""
    ax, ay, aw, ah = rect
    return any(not (ax + aw <= zx or zx + zw <= ax or ay + ah <= zy or zy + zh <= ay)
               for zx, zy, zw, zh in SPAWN_ZONES)


class Arena:
    """游戏世界：规则全部在这里，神经网络只负责给动作。"""

    def __init__(self, seed=0, destructible=False, no_cover=False, random_cover=False,
                 obs_v2=False):
        self.obs_v2 = obs_v2               # v27：观测加敌方运动状态 + 敌方冷却（37→40）
        self.destructible = destructible   # v15 几何手术：掩体 3HP，被炮弹打掉后视线贯通
        self.no_cover = no_cover           # v19：裸场地——掩体全部下线，避难所消失
        self.random_cover = random_cover   # v25：每局随机掩体布局（2~4 块，轴对齐矩形）
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _gen_random_layout(self):
        """v25：随机布局——2~4 块 40~140px 轴对齐矩形，互不重叠且间距 ≥30px，
        不压四角出生禁区。重试兜底：500 次凑不齐 ≥2 块就退回固定掩体（换算成 float）。"""
        for _attempt in range(200):
            n_target = int(self.rng.integers(2, 5))
            rects = []
            for _try in range(500):
                if len(rects) >= n_target:
                    break
                rw = float(self.rng.uniform(40, 140))
                rh = float(self.rng.uniform(40, 140))
                rx = float(self.rng.uniform(30, W - 30 - rw))
                ry = float(self.rng.uniform(30, H - 30 - rh))
                rect = (rx, ry, rw, rh)
                if any(not _rects_clear(rect, other) for other in rects):
                    continue
                if _rect_hits_spawn_zone(rect):
                    continue
                rects.append(rect)
            if len(rects) >= 2:
                return rects
        return [tuple(float(v) for v in ob) for ob in OBSTACLES]

    def live_obstacles(self):
        # 存活的掩体（含索引）
        if self.no_cover:
            return []
        return [(i, ob) for i, ob in enumerate(self.obstacles)
                if not self.destructible or self.obs_hp[i] > 0]

    def reset(self):
        # v25：随机掩体模式每局重生成布局；默认路径逐字节保持旧行为
        if self.random_cover:
            self.obstacles = self._gen_random_layout()
            n = len(self.obstacles)
            self.obs_hp = [3] * n if self.destructible else [1] * n
        else:
            self.obstacles = OBSTACLES
            self.obs_hp = [3] * len(OBSTACLES)
        # 随机出生（notes/004：固定出生点会让对局退化成"中线对射"，数据永远学不到转向）
        # v25.1 修复：搜索失败逐级放宽，最后用四角保底（掩体生成器从不压四角禁区，
        #            角落必然无掩体）——修复 pos[1] IndexError（v25 首次夜跑事故）
        pos = []
        for _ in range(2):
            for _try in range(100):
                x = self.rng.uniform(60, W - 60)
                y = self.rng.uniform(60, H - 60)
                if any(_circle_hit_rect(x, y, TANK_R + 5, ob) for _, ob in self.live_obstacles()):
                    continue
                if pos and math.hypot(x - pos[0][0], y - pos[0][1]) < 250:
                    continue
                pos.append((x, y))
                break
        # 逐级放宽兜底：无距离约束 → 四角保底
        corners = [(60, 60), (W - 60, H - 60), (60, H - 60), (W - 60, 60)]
        for cx, cy in corners:
            if len(pos) >= 2:
                break
            if not any(_circle_hit_rect(cx, cy, TANK_R + 5, ob) for _, ob in self.live_obstacles()):
                if not pos or math.hypot(cx - pos[0][0], cy - pos[0][1]) >= 100:
                    pos.append((cx, cy))
        while len(pos) < 2:   # 绝对兜底：对角硬坐标（掩体生成器保证四角禁区无掩体）
            pos.append(corners[len(pos) % 2])
        self.tanks = [
            Tank(*pos[0], (91, 160, 250), "蓝"),
            Tank(*pos[1], (240, 98, 86), "红"),
        ]
        for tank in self.tanks:
            tank.angle = self.rng.uniform(-math.pi, math.pi)
        self.bullets = []
        self.step_n = 0
        self.done = False
        self.winner = None
        self.stats = {"hits": [0, 0], "shots": [0, 0]}

    # ---- 观测：把局面压成 29 个数（notes/001 的输入向量 + notes/003 的掩体感）----
    def observe(self, me, foe):
        bl = [b for b in self.bullets if b.owner is not me and b.alive]
        if bl:  # 最近的敌方子弹
            b = min(bl, key=lambda b: (b.x - me.x) ** 2 + (b.y - me.y) ** 2)
            bdx, bdy = (b.x - me.x) / W, (b.y - me.y) / H
            bvx, bvy = b.vx / BULLET_SPEED, b.vy / BULLET_SPEED
        else:
            bdx = bdy = bvx = bvy = 0.0
        fx, fy = me.facing
        efx, efy = foe.facing
        dist = math.hypot(foe.x - me.x, foe.y - me.y)
        obs = [
            me.x / W * 2 - 1, me.y / H * 2 - 1, fx, fy, me.hp / HP_MAX,
            foe.x / W * 2 - 1, foe.y / H * 2 - 1, efx, efy, foe.hp / HP_MAX,
            (foe.x - me.x) / W, (foe.y - me.y) / H, dist / math.hypot(W, H),
            bdx, bdy, bvx, bvy,
            1.0 if me.cooldown == 0 else 0.0,
            1.0 - self.step_n / MAX_STEPS,
            1.0,  # 常数项（偏置之外多给一个自由度，演示用）
        ]
        # 掩体感（notes/003）：输出恒 16 维（4 槽 × dx,dy,hw,hh；v25 随机布局数量 ≤4，
        # 不足槽位补零；数量可超时只取前 4 块）
        obstacles = self.obstacles[:4]
        for i in range(4):
            if i >= len(obstacles) or self.no_cover or \
                    (self.destructible and self.obs_hp[i] <= 0):
                obs += [0.0, 0.0, 0.0, 0.0]   # 槽位空/已被摧毁/裸场地：零尺寸信号
            else:
                ox, oy, ow, oh = obstacles[i]
                obs += [((ox + ow / 2) - me.x) / W, ((oy + oh / 2) - me.y) / H,
                        ow / 2 / W, oh / 2 / H]
        # 视线：敌我连线是否被掩体挡住（0=畅通可开火，1=被挡）
        blocked = 0.0
        steps = max(1, int(dist / 10))
        for k in range(1, steps):
            px = me.x + (foe.x - me.x) * k / steps
            py = me.y + (foe.y - me.y) * k / steps
            if any(ox <= px <= ox + ow and oy <= py <= oy + oh
                   for _, (ox, oy, ow, oh) in self.live_obstacles()):
                blocked = 1.0
                break
        obs.append(blocked)
        if self.obs_v2:
            # v27：绕柱所需的两条信息——敌方运动状态（速度矢量）+ 敌方冷却余量
            obs += [foe.vx / TANK_SPEED, foe.vy / TANK_SPEED,
                    foe.cooldown / FIRE_COOLDOWN]
        return np.array(obs, dtype=np.float32)

    # ---- 一步游戏循环 ----
    def step(self, actions):
        if self.done:
            return
        self.step_n += 1

        # 1) 坦克按动作转向/移动/开火（动作来自神经网络或规则，规则引擎不关心来源）
        for tank, act in zip(self.tanks, actions):
            if not tank.alive:
                continue
            name = ACTIONS[act]
            tank.vx = tank.vy = 0.0          # 转向/开火/被挡 = 速度 0
            if name == "turn_left":
                tank.angle -= TURN_RATE
            elif name == "turn_right":
                tank.angle += TURN_RATE
            elif name == "fire":
                if tank.cooldown == 0:
                    fx, fy = tank.facing
                    self.bullets.append(Bullet(
                        tank.x + fx * (TANK_R + BULLET_R + 1),
                        tank.y + fy * (TANK_R + BULLET_R + 1),
                        fx * BULLET_SPEED, fy * BULLET_SPEED, tank))
                    tank.cooldown = FIRE_COOLDOWN
                    self.stats["shots"][self.tanks.index(tank)] += 1
            else:  # forward / backward：沿当前朝向移动
                speed = TANK_SPEED if name == "forward" else -BACK_SPEED
                fx, fy = tank.facing
                nx = tank.x + fx * speed
                ny = tank.y + fy * speed
                # 边界 + 掩体碰撞：撞了就停在原地
                nx_c = max(TANK_R, min(W - TANK_R, nx))
                ny_c = max(TANK_R, min(H - TANK_R, ny))
                if not any(_circle_hit_rect(nx_c, ny_c, TANK_R, ob) for _, ob in self.live_obstacles()):
                    tank.vx = nx_c - tank.x
                    tank.vy = ny_c - tank.y
                    tank.x, tank.y = nx_c, ny_c
            if tank.cooldown > 0:
                tank.cooldown -= 1

        # 2) 子弹飞行 + 命中判定
        for b in self.bullets:
            if not b.alive:
                continue
            b.x += b.vx
            b.y += b.vy
            if not (0 <= b.x <= W and 0 <= b.y <= H):
                b.alive = False
                continue
            hit_ob = next((i for i, ob in self.live_obstacles()
                           if _circle_hit_rect(b.x, b.y, BULLET_R, ob)), None)
            if hit_ob is not None:
                b.alive = False
                if self.destructible:
                    self.obs_hp[hit_ob] -= 1   # 掩体掉血，归零即消失
                continue
            for tank in self.tanks:
                if tank is b.owner or not tank.alive:
                    continue
                if (b.x - tank.x) ** 2 + (b.y - tank.y) ** 2 < (TANK_R + BULLET_R) ** 2:
                    tank.hp -= DAMAGE
                    b.alive = False
                    self.stats["hits"][self.tanks.index(b.owner)] += 1
                    if tank.hp <= 0:
                        tank.alive = False
                        self.done = True
                        self.winner = b.owner
                    break
        self.bullets = [b for b in self.bullets if b.alive]

        # 3) 超时平局
        if self.step_n >= MAX_STEPS:
            self.done = True


# ---------------- 渲染 ----------------
def draw(surface, arena, extra=""):
    import pygame
    surface.fill((13, 17, 23))
    for x in range(0, W, 40):  # 淡网格
        pygame.draw.line(surface, (22, 27, 34), (x, 0), (x, H))
    for y in range(0, H, 40):
        pygame.draw.line(surface, (22, 27, 34), (0, y), (W, y))
    obstacles = getattr(arena, "obstacles", OBSTACLES)  # v25：随机布局时画实例掩体
    for i, ob in enumerate(obstacles):  # 掩体（可摧毁模式下按 HP 深浅，归零跳过）
        hp = getattr(arena, "obs_hp", None)
        if hp is not None and hp[i] <= 0:
            continue
        shade = 1.0 if hp is None else (0.4 + 0.6 * hp[i] / 3)
        pygame.draw.rect(surface, tuple(int(c * shade) for c in (62, 68, 76)), ob)
        pygame.draw.rect(surface, tuple(int(c * shade) for c in (110, 118, 129)), ob, 1)
    for b in arena.bullets:  # 子弹
        pygame.draw.circle(surface, (227, 179, 65), (int(b.x), int(b.y)), BULLET_R)
    for tank in arena.tanks:  # 坦克：方形底盘 + 履带 + 炮管 + 车头亮条（转向一眼可见）+ 血条
        draw_tank(surface, tank)
    if extra:  # 顶部状态行
        font = pygame.font.SysFont("sans", 16)
        surface.blit(font.render(extra, True, (139, 148, 158)), (8, 6))


def draw_tank(surface, tank):
    """方形坦克：先画一张朝右的小图，再按 angle 旋转贴上。
    车头亮条 + 偏置炮管制造不对称，朝向（含转向过程）肉眼可读。"""
    import pygame
    if not tank.alive:  # 残骸：灰色空方框
        r = pygame.Rect(0, 0, 22, 22)
        r.center = (tank.x, tank.y)
        pygame.draw.rect(surface, (80, 80, 80), r, 2)
        return
    C = 24  # 小图半径（留足旋转空间）
    img = pygame.Surface((C * 2, C * 2), pygame.SRCALPHA)
    dark = tuple(int(c * 0.55) for c in tank.color)
    pygame.draw.rect(img, dark, (C - 11, C - 11, 22, 5), border_radius=2)   # 上履带
    pygame.draw.rect(img, dark, (C - 11, C + 6, 22, 5), border_radius=2)    # 下履带
    pygame.draw.rect(img, tank.color, (C - 9, C - 8, 18, 16), border_radius=3)  # 车身
    pygame.draw.line(img, (230, 237, 243), (C, C), (C + 17, C), 4)          # 炮管（伸出车头）
    pygame.draw.circle(img, (230, 237, 243), (C, C), 4)                     # 炮塔座
    rot = pygame.transform.rotate(img, -math.degrees(tank.angle))  # 屏幕 y 向下，取负号
    surface.blit(rot, rot.get_rect(center=(tank.x, tank.y)))
    bw = 30  # 血条（不跟车转）
    pygame.draw.rect(surface, (48, 54, 61), (tank.x - bw / 2, tank.y - 22, bw, 4))
    pygame.draw.rect(surface, (63, 185, 80),
                     (tank.x - bw / 2, tank.y - 22, bw * max(0, tank.hp) / HP_MAX, 4))


# ---------------- 主流程 ----------------
def run_episode(arena, brains, render_hook=None, verbose=False):
    """一局：两个随机大脑互打。render_hook(arena, info) 每步调用（画图/录帧）。"""
    while not arena.done:
        actions = []
        for tank, foe, brain in zip(arena.tanks, arena.tanks[::-1], brains):
            if not tank.alive:
                actions.append(0)
                continue
            obs = arena.observe(tank, foe)
            act, _ = brain.act(obs, sample=True, rng=arena.rng)  # 按概率采样 = 探索模式
            actions.append(act)
        arena.step(actions)
        if render_hook:
            t0, t1 = arena.tanks
            render_hook(arena,
                        f"step {arena.step_n} | B(blue) hp {t0.hp} | R(red) hp {t1.hp} | bullets {len(arena.bullets)}")
    if verbose:
        w = arena.winner.name if arena.winner else "平局"
        print(f"  结果: {w} | 步数 {arena.step_n} | 蓝 {arena.stats['hits'][0]}中/{arena.stats['shots'][0]}射"
              f" | 红 {arena.stats['hits'][1]}中/{arena.stats['shots'][1]}射")
    return arena


def record_gif(arena, brains, path, fps=30, every=3):
    """无头录制一局到 GIF（供 eval_bc 等复用）。brains 需有 act(obs, sample, rng)。"""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    from PIL import Image
    pygame.init()
    surface = pygame.Surface((W, H))
    frames = []

    def hook(a, info):
        draw(surface, a, info)
        if a.step_n % every == 0:
            frames.append(Image.fromarray(
                np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))))

    run_episode(arena, brains, render_hook=hook, verbose=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 * every / (fps * 2)), loop=0, optimize=True)
    print(f"已录制 {len(frames)} 帧 → {path}（{os.path.getsize(path)/1e6:.1f} MB）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="桌面窗口实时演示")
    ap.add_argument("--record", metavar="GIF", help="无头录制到 GIF")
    ap.add_argument("--headless", action="store_true", help="纯文本验证（跑 3 局）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    arena = Arena(seed=args.seed)
    # 两个坦克各持一份随机初始化的权重（不同种子 = 两个不同的"未训练大脑"）
    brains = [MLP(seed=args.seed + 1), MLP(seed=args.seed + 1000)]
    print(f"网络结构: 37 → 128 → 128 → 5 | 每个大脑 {brains[0].n_params:,} 参数（随机初始化）")

    if args.headless:
        for ep in range(3):
            arena.reset()
            print(f"第 {ep + 1} 局:")
            run_episode(arena, brains, verbose=True)
        return

    if args.record:
        record_gif(arena, brains, args.record)
    else:  # --live 或默认：桌面窗口
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("智能坦克 · 随机初始化权重演示")
        clock = pygame.time.Clock()
        running = True

        def hook(a, info):
            nonlocal running
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    a.done = True
            draw(screen, a, info)
            pygame.display.flip()
            clock.tick(args.fps)

        while running:
            arena.reset()
            run_episode(arena, brains, render_hook=hook)
            if not running:
                break
            pygame.time.wait(1500)  # 一局结束停 1.5s 再开新局
        pygame.quit()


if __name__ == "__main__":
    main()
