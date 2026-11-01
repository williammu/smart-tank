# 017 · v25：随机掩体 + 宽 MLP（guns-roses 执行记录与夜跑手册）

> 2026-11-22 · 执行机 guns-roses（Guns-and-Roses-3.local / macOS 26.6.2 / 10 核）
> 任务书：`TASK-v25-gunsroses.md`

## 交付内容

- `scripts/arena.py`：`Arena(random_cover=False)`。True 时 `reset()` 每局生成 2~4 块
  40~140px 轴对齐矩形掩体（互不重叠且间距 ≥30px、不压四角 110×110 出生禁区），
  `obs_hp = [3]*n`（destructible）或 `[1]*n`；`observe()` 掩体特征恒 16 维（不足补零）；
  所有实例引用改 `self.obstacles`。默认路径与 HEAD 逐字节 parity（200 步 × 5 seeds 全等）。
- `scripts/train_bc.py`：`Net(hidden1=128, hidden2=128)` 参数化，默认零改动。
- `scripts/arena.py MLP.load` / `eval_bc.NetBrain`：从 npz `w0/w1` shape 自动推断隐宽，
  旧 128 存档照常加载。
- `scripts/train_parallel.py`：`--width1/--width2`（默认 128/128）、`--random-cover`；
  工人侧隐宽从分发权重 shape 推导；`_chk_width` 存档/参数错配早失败。
- `scripts/train_rl.py`：`play_game(random_cover=False)` 透传；`attach_critic` 随机初始化
  隐宽从 `net.sizes[2]` 推导。
- `scripts/test_v25_parity.py`：parity（git HEAD 旧 arena 基准）+ 随机布局校验 +
  destructible 3 炮拆除。`ALL PASS`，exit=0。
- `scripts/warmup_v25.py`：录 400 局 RuleTeacher（random_cover+destructible，红方六流派
  随机抽签）→ 316,792 帧 → 256 宽 Net 1 epoch val_acc **85.06%**（线 70%）→ `data/warm-v25.npz`。
- 冒烟：`--width1 256 --width2 256 --random-cover --destructible --vs-teacher
  --init data/warm-v25.npz --workers 2 --games 4 --iters 4 --eval-every 2`
  → 月考 + best-checkpoint 正常，exit=0；`--resume` 从 @4 接力到 @10，exit=0。

## v25 正式夜跑命令（guns-roses）

```bash
cd ~/.openclaw/workspace/work/dnn-learning/subprojects/smart-tank
[ -x .venv/bin/python ] || { python3 -m venv .venv && ./.venv/bin/pip install numpy pygame; }
# 热身存档丢了才需要重录：
# ./.venv/bin/python scripts/warmup_v25.py
nohup ./.venv/bin/python scripts/train_parallel.py \
  --width1 256 --width2 256 --random-cover --destructible --vs-teacher \
  --init data/warm-v25.npz --out data/rl-v25.npz \
  --workers 8 --games 256 --iters 4000 --eval-every 25 \
  --hit-reward 1.0 --lr 3e-5 \
  >> data/v25-night.log 2>&1 &
# 中断后接力：同命令 + --resume
```
