#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 dataset-v1.npz 的可视化样本注入 viz/dataset.html（分层抽样：每个动作类 80 条）。
可重复运行：替换 HTML 里的 const DATA = /*__INJECT__*/null; 占位行。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_CLASS = 80
ACTION_NAMES = ["左转", "右转", "前进", "后退", "开火"]

d = np.load(os.path.join(HERE, "..", "data", "dataset-v1.npz"))
X, y = d["X_train"], d["y_train"]
rng = np.random.default_rng(7)

samples = []
for a in range(5):
    idx = np.where(y == a)[0]
    pick = rng.choice(idx, size=min(N_PER_CLASS, len(idx)), replace=False)
    for i in sorted(pick):
        samples.append({"i": int(i), "y": int(a), "obs": [round(float(v), 4) for v in X[i]]})

counts = {ACTION_NAMES[a]: int(np.sum(y == a)) for a in range(5)}
rng.shuffle(samples)  # 打散类别顺序，"全部"视图五类混排
data = {"n_train": int(len(y)), "n_val": int(len(d["y_val"])),
        "counts": counts, "names": ACTION_NAMES, "samples": samples}

html_path = os.path.join(HERE, "..", "viz", "dataset.html")
html = open(html_path, encoding="utf-8").read()
marker = "const DATA = /*__INJECT__*/null;"
assert marker in html, "HTML 占位行不见了"
html = html.replace(marker, "const DATA = " + json.dumps(data, ensure_ascii=False) + ";")
open(html_path, "w", encoding="utf-8").write(html)
print(f"注入 {len(samples)} 条样本 → viz/dataset.html（{os.path.getsize(html_path)//1024} KB）")
