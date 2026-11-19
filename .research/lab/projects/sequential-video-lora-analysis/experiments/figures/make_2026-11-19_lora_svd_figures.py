"""MTG資料（2026-11-19）用の LoRA 特異値分解の図を、書き出し済みの CSV から作る（読み取りのみ）。

/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/.venv/bin/python make_2026-11-19_lora_svd_figures.py

入力：研究リポジトリ log/analysis/lora_svd/2026-11-19_06-56-19/（schema lora_svd/2、commit 0aa523f）
"""
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPORT = Path("/mnt/HDD12TB-1/ishikawa/2026_09_ishikawa_sequential-video-lora/log/analysis/lora_svd/2026-11-19_06-56-19")
OUT = Path(__file__).resolve().parent / "2026-11-19-lora-svd-current-state"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "savefig.dpi": 200,
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
})
BLUE, ORANGE, AQUA, PURPLE, GRAY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#8a8984", "#52514e"

LABEL = {
    "imagenet-lora": "ImageNet画像（1,000step）",
    "frame-lora": "単一フレーム（459step）",
    "clip-lora": "4フレームclip・シャッフル（459step）",
    "seq16-lora": "順に読む・4フレーム・stride 16",
    "seq8-lora": "順に読む・4フレーム・stride 8",
    "seq16-t8-lora": "順に読む・8フレーム・stride 16",
    "imagenet-ft": "ImageNet画像（Full FT）",
    "frame-ft": "単一フレーム（Full FT）",
    "clip-ft": "4フレームclip（Full FT）",
}
SHORT = {
    "imagenet-lora": "ImageNet", "frame-lora": "単一フレーム", "clip-lora": "clip・シャッフル",
    "seq16-lora": "順・stride16", "seq8-lora": "順・stride8", "seq16-t8-lora": "順・8フレーム",
    "imagenet-ft": "ImageNet FT", "frame-ft": "単一フレーム FT", "clip-ft": "clip FT",
}
SEQ = {"seq16-lora": BLUE, "seq8-lora": ORANGE, "seq16-t8-lora": PURPLE}
SHORT_LORA = {"imagenet-lora": "o", "frame-lora": "s", "clip-lora": "D"}
FT = {"imagenet-ft": "o", "frame-ft": "s", "clip-ft": "D"}

s = pd.read_csv(EXPORT / "svd_summary.csv")
w0 = s[s.run_id == "pretrained"].set_index(["layer", "module"])
t = s[s.run_id != "pretrained"].copy()
key = list(zip(t.layer, t.module))
t["s1_ratio"] = t.sigma_1.values / w0.loc[key, "sigma_1"].values
t["fro_ratio"] = t.fro.values / w0.loc[key, "fro"].values
assert t.groupby(["run_id", "step"]).size().eq(24).all()
med = t.groupby(["run_id", "step"])[["s1_ratio", "fro_ratio"]].median().reset_index()


def fig_size_vs_step():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title in [(axes[0], "s1_ratio", r"最大特異値の比  $\sigma_1(\Delta W)\,/\,\sigma_1(W_0)$"),
                           (axes[1], "fro_ratio", r"フロベニウスノルムの比  $\|\Delta W\|_F\,/\,\|W_0\|_F$")]:
        for run, c in SEQ.items():
            m = med[med.run_id == run].sort_values("step")
            x = np.r_[0, m.step.values]; y = np.r_[0, m[col].values]
            ax.plot(x, y, color=c, marker="o", ms=4, lw=1.8, label=LABEL[run])
        for run, mk in SHORT_LORA.items():
            m = med[med.run_id == run]
            ax.scatter(m.step, m[col], color=AQUA, marker=mk, s=46, zorder=3, label=LABEL[run] + "：LoRA")
        for run, mk in FT.items():
            m = med[med.run_id == run]
            ax.scatter(m.step, m[col], facecolor="none", edgecolor=INK2, marker=mk, s=46, zorder=3,
                       label=LABEL[run])
        ax.set_xlabel("step（パラメータの更新回数）")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("24行列（12層 × q・v）の中央値")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle(r"LoRA の更新量 $\Delta W$ の大きさ（事前学習の重み $W_0$ との比）", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01-size-vs-step.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_layer():
    runs = [("seq16-lora", 6255, BLUE, "-", "o"), ("seq16-t8-lora", 6255, PURPLE, "-", "o"),
            ("seq8-lora", 6255, ORANGE, "-", "o"),
            ("clip-lora", 459, AQUA, "-", "D"), ("clip-ft", 459, INK2, "--", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mod in zip(axes, ["q", "v"]):
        for run, st, c, ls, mk in runs:
            m = t[(t.run_id == run) & (t.step == st) & (t.module == mod)].sort_values("layer")
            ax.plot(m.layer, m.s1_ratio, color=c, ls=ls, marker=mk, ms=4, lw=1.6,
                    label=LABEL[run] if run.startswith(("clip", "imagenet", "frame")) else f"{LABEL[run]}（step {st:,}）")
        ax.axhline(1.0, color=GRAY, lw=1, ls=":")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("層（1 = 入力側、12 = 出力側）")
        ax.set_title(f"{mod}（{'query' if mod == 'q' else 'value'} の射影）", fontsize=11, loc="left")
    axes[0].set_ylabel(r"$\sigma_1(\Delta W)\,/\,\sigma_1(W_0)$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(r"層ごとの最大特異値の比（最後の step。点線 = $W_0$ の最大特異値と同じ大きさ）",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "02-per-layer-ratio.png", bbox_inches="tight")
    plt.close(fig)


def fig_direction_overlap():
    final = {"imagenet-lora": 1000, "frame-lora": 459, "clip-lora": 459,
             "seq16-lora": 6255, "seq8-lora": 6255, "seq16-t8-lora": 6255,
             "imagenet-ft": 1000, "frame-ft": 459, "clip-ft": 459}
    vec = {}
    for ch in pd.read_csv(EXPORT / "svd_vectors.csv", chunksize=2000):
        ch = ch[(ch.kind == "U") & (ch.step == ch.run_id.map(final))]
        for r in ch.itertuples(index=False):
            vec.setdefault((r.run_id, r.layer, r.module), {})[r.component] = np.asarray(r[6:], float)

    def mat(run, l, m):
        d = vec[(run, l, m)]
        return np.stack([d[c] for c in sorted(d)], 1)

    runs = list(final)
    phi = np.zeros((len(runs), len(runs)))
    for (i, a), (j, b) in itertools.product(enumerate(runs), enumerate(runs)):
        vals = []
        for l in range(1, 13):
            for m in "qv":
                A, B = mat(a, l, m), mat(b, l, m)
                r = min(A.shape[1], B.shape[1])
                vals.append(np.linalg.norm(A[:, :r].T @ B[:, :r]) ** 2 / r)
        phi[i, j] = np.median(vals)
    pd.DataFrame(phi, index=runs, columns=runs).to_csv(OUT / "03-direction-overlap.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(phi, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(runs)):
        for j in range(len(runs)):
            ax.text(j, i, f"{phi[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if phi[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(runs))); ax.set_yticks(range(len(runs)))
    ax.set_xticklabels([SHORT[r] for r in runs], rotation=45, ha="right")
    ax.set_yticklabels([SHORT[r] for r in runs])
    ax.axhline(5.5, color="#ffffff", lw=3); ax.axvline(5.5, color="#ffffff", lw=3)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="φ（1 = 同じ部分空間、無関係なら約 0.01）")
    ax.set_title(r"$\Delta W$ の上位 8 本の左特異ベクトル（U）が張る部分空間の重なり φ" "\n"
                 "最後の step どうし、24 行列の中央値（上 6 つ = LoRA、下 3 つ = Full FT）",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "03-direction-overlap.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(phi, index=runs, columns=runs)


if __name__ == "__main__":
    fig_size_vs_step()
    fig_per_layer()
    print(fig_direction_overlap().round(3).to_string())
    print(med.round(4).to_string())
