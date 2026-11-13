"""MTG資料（2026-11-12）用の実験図を、研究リポジトリのログから作る（読み取りのみ）。

python3 make_2026-11-11_mtg_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = Path("/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log")
HERE = Path(__file__).resolve().parent
EVAL = LOG / "temporal_order_eval/2026-11-11_10-36-56/summary.json"
SEQ = LOG / "moco_vit_seq_activitynet_clip/lora/activitynet_sequential_clips/lr_0.0003"
SEQ_RUNS = [(SEQ / "2026-11-05_02-08-19-443327/metrics.jsonl", 1, 2000),
           (SEQ / "2026-11-08_17-26-03-055153/metrics.jsonl", 2001, 6255)]

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "savefig.dpi": 200,
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
})
BLUE, ORANGE, AQUA, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"


def fig_order_change():
    s = json.load(open(EVAL))
    # summary.json のモデル名（評価時のkey）。表示名は labels で付ける
    models = ["A", "B-LoRA", "B-FT", "C-LoRA", "C-FT", "G2-LoRA"]
    labels = {"A": "学習前（画像ViTのまま）",
              "B-LoRA": "単一フレームで学習：LoRA", "B-FT": "単一フレームで学習：Full FT",
              "C-LoRA": "clipをランダムに選んで学習：LoRA", "C-FT": "clipをランダムに選んで学習：Full FT",
              "G2-LoRA": "動画を先頭から順に読んで学習：LoRA"}
    conds = [("shuffle", "並べ替え", BLUE, "o", BLUE),
             ("reverse", "逆再生", ORANGE, "D", ORANGE),
             ("static", "静止", AQUA, "s", AQUA),
             ("same_video", "同じ動画の10秒後のclip", GRAY, "o", "none")]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(models))[::-1]
    for key, lab, c, m, fc in conds:
        x = [1 - s["results"][mo][key]["cls"]["all"]["mean"] for mo in models]
        ax.scatter(x, y, s=64, marker=m, facecolors=fc, edgecolors=c, linewidths=2,
                   label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_yticks(y, [labels[m] for m in models])
    ax.set_xlabel("元の順序のclipとの特徴の変化  1 − cos（対数軸）")
    ax.set_title("並び順を変えたときの特徴（CLS）の変化", loc="left", fontsize=13, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=4, fontsize=9.5, frameon=False)
    fig.tight_layout()
    out = HERE / "2026-11-11-temporal-order-sensitivity-sequential-lora/01-order-change-cls.png"
    fig.savefig(out)
    print(out)


def load_sequential():
    rows = []
    for path, lo, hi in SEQ_RUNS:
        for line in open(path):
            r = json.loads(line)
            if lo <= r["global_step"] <= hi:
                rows.append(r)
    rows.sort(key=lambda r: r["global_step"])
    steps = np.array([r["global_step"] for r in rows])
    assert (steps == np.arange(1, 6256)).all(), "step 1..6255 が1つずつ揃っていない"
    get = lambda k: np.array([r[k] for r in rows], dtype=float)
    return steps, get("loss"), get("positive_cosine_mean"), get("hardest_negative_cosine_mean")


def rolling(x, steps, w=250, start=257):
    """step >= start（Queueが本物のkeyで埋まった後）だけで計算する中心移動平均。"""
    out = np.full_like(x, np.nan)
    idx = np.where(steps >= start)[0]
    v = x[idx]
    c = np.cumsum(np.insert(v, 0, 0.0))
    h = w // 2
    for j in range(h, len(v) - (w - h) + 1):
        out[idx[j]] = (c[j + w - h] - c[j - h]) / w
    return out


def fig_sequential_loss():
    steps, loss, pos, hneg = load_sequential()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(steps, loss, color="#c9c8c2", lw=0.6, label="各step")
    a1.plot(steps, rolling(loss, steps), color=BLUE, lw=2, label="250step移動平均")
    a1.set_ylabel("MoCo loss")
    a1.set_ylim(6.6, 9.0)
    a1.legend(loc="upper right", fontsize=9.5, frameon=False)
    a1.set_title("動画を先頭から順に読んで学習したLoRAのlossとcos", loc="left", fontsize=13, pad=10)
    a2.plot(steps, rolling(pos, steps), color=AQUA, lw=2, label="正例とのcos")
    a2.plot(steps, rolling(hneg, steps), color=ORANGE, lw=2, label="最も似た負例とのcos")
    a2.set_ylabel("cos（250step移動平均）")
    a2.set_ylim(0.4, 1.0)
    a2.set_xlabel("step")
    a2.set_xlim(0, 6300)
    a2.legend(loc="lower right", fontsize=9.5, frameon=False)
    fig.tight_layout()
    out = HERE / "2026-11-09-sequential-loader-lora-loss/01-loss-and-cosine.png"
    fig.savefig(out)
    print(out)


RUNS_LR = {  # (学習率, 方式) -> run directory（固定の確認用画像64枚の診断がある）
    ("0.03", "LoRA"): LOG / "moco_vit_query_collapse/lora/imagenet/2026-10-22_04-02-22-441192",
    ("0.03", "Full FT"): LOG / "moco_vit_query_collapse/full_ft/imagenet/2026-10-22_04-20-32-327570",
    ("3e-4", "LoRA"): LOG / "moco_vit_lr_long/lora/imagenet/lr_0.0003/2026-11-01_00-11-06-779221",
    ("3e-4", "Full FT"): LOG / "moco_vit_lr_long/full_ft/imagenet/lr_0.0003/2026-11-01_09-37-00-940051",
}
RUNS_STAGE = [
    ("ImageNetの画像（1,000step）", {
        "LoRA": RUNS_LR[("3e-4", "LoRA")], "Full FT": RUNS_LR[("3e-4", "Full FT")]}),
    ("ActivityNetの単一フレーム（459step）", {
        "LoRA": LOG / "moco_vit_lr_activitynet/lora/activitynet_frames/lr_0.0003/2026-11-03_10-48-36-408110",
        "Full FT": LOG / "moco_vit_lr_activitynet/full_ft/activitynet_frames/lr_0.0003/2026-11-03_11-49-57-818631"}),
    ("ActivityNetの4フレームclip（459step）", {
        "LoRA": LOG / "moco_vit_lr_activitynet_clip/lora/activitynet_clips/lr_0.0003/2026-11-03_15-22-36-985580",
        "Full FT": LOG / "moco_vit_lr_activitynet_clip/full_ft/activitynet_clips/lr_0.0003/2026-11-03_16-46-46-287044"}),
]


def probe_p50(run, stage):
    rows = [json.loads(l) for l in open(run / "probe_metrics.jsonl")]
    rows = sorted((r for r in rows if r["rank"] == 0 and r["stage"] == stage),
                  key=lambda r: r["probe_step"])
    return [r["probe_step"] for r in rows], [r["offdiag_cosine_p50"] for r in rows]


def fig_learning_rate():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    panels = [("z_q", "MoCoの出力（Projector）"), ("cls", "ViTの出力（CLS）")]
    style = {"0.03": ORANGE, "3e-4": BLUE}
    ls = {"LoRA": ("-", "o"), "Full FT": ("--", "s")}
    for ax, (stage, title) in zip(axes, panels):
        for (lr, mode), run in RUNS_LR.items():
            x, y = probe_p50(run, stage)
            ax.plot(x, y, color=style[lr], ls=ls[mode][0], marker=ls[mode][1], ms=6, lw=2,
                    label=f"学習率 {lr}：{mode}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.1, 1300)
        ax.set_xticks([0, 1, 2, 10, 100, 1000], ["0", "1", "2", "10", "100", "1000"])
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("step（対数軸）")
    axes[0].set_ylabel("確認用画像64枚どうしのcos（中央値）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle("学習率ごとの、確認用画像どうしの特徴のcosの推移（ImageNet）",
                 x=0.01, y=0.985, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    out = HERE / "2026-11-01-imagenet-learning-rate-collapse/01-probe-cosine-by-lr.png"
    fig.savefig(out)
    print(out)


def fig_stage_loss():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.2), sharey=True)
    color = {"LoRA": BLUE, "Full FT": ORANGE}
    for ax, (title, runs) in zip(axes, RUNS_STAGE):
        for mode, run in runs.items():
            rows = [json.loads(l) for l in open(run / "metrics.jsonl")]
            steps = np.array([r["global_step"] for r in rows])
            loss = np.array([r["loss"] for r in rows], dtype=float)
            ax.plot(steps, loss, color=color[mode], lw=0.6, alpha=0.3, label=f"{mode}（各step）")
            ax.plot(steps, rolling(loss, steps, w=50), color=color[mode], lw=2,
                    label=f"{mode}（50step移動平均）")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("step")
        ax.set_xlim(0, steps.max() + 10)
    axes[0].set_ylabel("MoCo loss")
    axes[0].set_ylim(6.7, 9.3)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               ncol=4, fontsize=9.5, frameon=False)
    fig.suptitle("段階ごとのlossの推移（学習率3e-4）", x=0.01, y=0.985, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    out = HERE / "2026-11-03-pretraining-loss-by-stage/01-loss-by-stage.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    fig_order_change()
    fig_sequential_loss()
    fig_learning_rate()
    fig_stage_loss()
