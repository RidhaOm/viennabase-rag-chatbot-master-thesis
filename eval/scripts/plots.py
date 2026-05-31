"""
plots.py
--------
Generates the figures used in the thesis results section.

Output files:
  fig_latency_box_swarm.png   — boxplot + swarm plot with Wilcoxon p-values
  fig_metric_bars.png         — grouped bar chart with 95% Wilson CIs
  fig_mcnemar_heatmap.png     — McNemar effect direction heatmap
  fig_fallback_behavior.png   — stacked bar chart: cf=1 / cf=0 / NA per config

Usage:
    python plots.py \
        --labels-dir eval/labels \
        --results-dir eval/results \
        --analysis-dir eval/analysis \
        --out-dir eval/figures
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


CONFIGS = ["baseline", "no_threshold", "no_history"]
COLORS = {
    "baseline":     "#2E7D32",
    "no_threshold": "#E65100",
    "no_history":   "#1565C0",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_results(results_dir: Path, config: str) -> pd.DataFrame:
    path = results_dir / f"results_{config}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8")


def load_labels(labels_dir: Path, config: str) -> pd.DataFrame:
    path = labels_dir / f"labels_{config}.csv"
    return pd.read_csv(path, encoding="utf-8")


def load_ci(analysis_dir: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_dir / "confidence_intervals.csv", encoding="utf-8")


def load_mcnemar(analysis_dir: Path) -> pd.DataFrame:
    return pd.read_csv(analysis_dir / "mcnemar_results.csv", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 1: Latency boxplot with swarm overlay
# ---------------------------------------------------------------------------

def plot_latency(results: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    data = []
    labels = []
    positions = []
    for i, cfg in enumerate(CONFIGS, start=1):
        df = results.get(cfg)
        if df is None:
            continue
        lats = pd.to_numeric(df["latency_s"], errors="coerce").dropna().values
        if len(lats) == 0:
            continue
        data.append(lats)
        labels.append(cfg)
        positions.append(i)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, cfg in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(cfg, "gray"))
        patch.set_alpha(0.35)

    rng = np.random.default_rng(seed=42)
    for pos, vals, cfg in zip(positions, data, labels):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter, vals, s=20,
            color=COLORS.get(cfg, "gray"),
            edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3,
        )

    for pos, vals in zip(positions, data):
        m = np.mean(vals)
        ax.hlines(
            m, pos - 0.25, pos + 0.25,
            colors="red", linestyles="dashed", linewidth=1.5,
            label="mean" if pos == positions[0] else "",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title(
        "Response latency per configuration\n"
        "(box: IQR, dashed red: mean, dots: individual queries)",
        fontsize=12, pad=20,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if data:
        max_val = max(max(v) for v in data)
        ax.set_ylim(top=max_val * 1.25)

    if len(data) >= 2:
        y_top = max(max(v) for v in data)
        for j, cfg in enumerate(labels[1:], start=1):
            try:
                a = data[0]
                b = data[j]
                if len(a) == len(b):
                    res = stats.wilcoxon(a, b)
                    p = res.pvalue
                    y_annot = y_top * (1.05 + (j - 1) * 0.05)
                    ax.annotate(
                        f"baseline vs {cfg}: p={p:.3f}",
                        xy=((positions[0] + positions[j]) / 2, y_annot),
                        ha="center", fontsize=9, color="dimgray",
                    )
            except Exception:
                pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Metric bars with Wilson confidence intervals
# ---------------------------------------------------------------------------

def plot_metric_bars(ci_df: pd.DataFrame, out_path: Path) -> None:
    metrics_to_plot = ["acceptable", "good", "grounded", "correct_fallback"]
    metric_labels = {
        "acceptable":       "Acceptable\n(good or ok)",
        "good":             "Strict good",
        "grounded":         "Grounded",
        "correct_fallback": "Correct\nfallback",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_metrics = len(metrics_to_plot)
    bar_width = 0.26
    x = np.arange(n_metrics)

    for i, cfg in enumerate(CONFIGS):
        rates = []
        lows  = []
        highs = []
        for m in metrics_to_plot:
            sub = ci_df[(ci_df["config"] == cfg) & (ci_df["metric"] == m)]
            if len(sub) == 0:
                rates.append(0); lows.append(0); highs.append(0)
                continue
            r  = float(sub.iloc[0]["rate"])
            lo = float(sub.iloc[0]["ci_low_95"])
            hi = float(sub.iloc[0]["ci_high_95"])
            rates.append(r)
            lows.append(r - lo)
            highs.append(hi - r)

        pos  = x + (i - 1) * bar_width
        bars = ax.bar(
            pos, rates, bar_width,
            color=COLORS.get(cfg, "gray"),
            edgecolor="black", linewidth=0.5,
            label=cfg, alpha=0.85,
        )
        ax.errorbar(pos, rates, yerr=[lows, highs],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        for bar, rate in zip(bars, rates):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{rate:.2f}", ha="center", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot], fontsize=10)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(
        "Evaluation metrics per configuration\n"
        "(error bars: 95% Wilson confidence interval)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Fallback behaviour stacked bar chart
# ---------------------------------------------------------------------------

def plot_fallback_behavior(labels: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))

    categories   = ["Correct fallback\n(cf=1)", "Wrong decision\n(cf=0)", "N/A\n(answered correctly)"]
    colors_cat   = ["#2E7D32", "#C62828", "#BDBDBD"]

    bottom = np.zeros(len(CONFIGS))
    for cat_idx, cat_label in enumerate(categories):
        heights = []
        for cfg in CONFIGS:
            df = labels[cfg]

            def _cf_str(v):
                if pd.isna(v):
                    return "NA"
                try:
                    return str(int(float(v)))
                except (ValueError, TypeError):
                    return str(v).strip().upper()

            cf = df["correct_fallback"].apply(_cf_str)
            if cat_idx == 0:
                count = int((cf == "1").sum())
            elif cat_idx == 1:
                count = int((cf == "0").sum())
            else:
                count = int((cf == "NA").sum())
            heights.append(count)

        ax.bar(
            range(len(CONFIGS)), heights, bottom=bottom,
            color=colors_cat[cat_idx], edgecolor="black",
            linewidth=0.5, label=cat_label,
        )
        for i, h in enumerate(heights):
            if h > 0:
                ax.text(
                    i, bottom[i] + h / 2, str(h),
                    ha="center", va="center", fontsize=10,
                    color="white" if cat_idx < 2 else "black",
                    fontweight="bold",
                )
        bottom += heights

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels(CONFIGS)
    ax.set_ylabel("Number of questions (of 79)", fontsize=11)
    ax.set_title("Fallback behaviour per configuration", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: McNemar p-value heatmap
# ---------------------------------------------------------------------------

def plot_mcnemar_heatmap(mc_df: pd.DataFrame, out_path: Path) -> None:
    metrics    = mc_df["metric"].unique().tolist()
    pairs      = [(a, b) for a, b in zip(mc_df["config_a"], mc_df["config_b"])]
    pair_labels = sorted(set(f"{a}\nvs {b}" for a, b in pairs))

    matrix      = np.zeros((len(pair_labels), len(metrics)))
    annotations = np.empty_like(matrix, dtype=object)

    for i, pl in enumerate(pair_labels):
        a, b = pl.split("\nvs ")
        for j, m in enumerate(metrics):
            sub = mc_df[
                (mc_df["config_a"] == a) &
                (mc_df["config_b"] == b) &
                (mc_df["metric"]   == m)
            ]
            if len(sub) == 0:
                matrix[i, j] = 0
                annotations[i, j] = ""
                continue
            p = float(sub.iloc[0]["p_value_two_sided"])
            neg_log_p = -math.log10(p) if p > 0 else 3
            matrix[i, j] = min(neg_log_p, 3)
            direction = sub.iloc[0]["effect_direction"]
            sig = "*" if p < 0.05 else ""
            annotations[i, j] = f"p={p:.3f}{sig}\n{direction}"

    fig, ax = plt.subplots(figsize=(10, 3 + 0.8 * len(pair_labels)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=0, fontsize=10)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)

    for i in range(len(pair_labels)):
        for j in range(len(metrics)):
            ax.text(
                j, i, annotations[i, j],
                ha="center", va="center", fontsize=8,
                color="black" if matrix[i, j] < 1.5 else "white",
            )

    plt.colorbar(im, ax=ax, label="-log10(p)  (capped at 3)")
    ax.set_title(
        "Pairwise McNemar tests (two-sided)\n* = significant at α=0.05",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir",   default="eval/labels")
    ap.add_argument("--results-dir",  default="eval/results")
    ap.add_argument("--analysis-dir", default="eval/analysis")
    ap.add_argument("--out-dir",      default="eval/figures")
    args = ap.parse_args()

    labels_dir   = Path(args.labels_dir)
    results_dir  = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {cfg: load_results(results_dir, cfg) for cfg in CONFIGS}
    labels  = {cfg: load_labels(labels_dir,   cfg) for cfg in CONFIGS}
    ci_df   = load_ci(analysis_dir)
    mc_df   = load_mcnemar(analysis_dir)

    print("Generating figures...")
    plot_latency(results, out_dir / "fig_latency_box_swarm.png")
    plot_metric_bars(ci_df, out_dir / "fig_metric_bars.png")
    plot_fallback_behavior(labels, out_dir / "fig_fallback_behavior.png")
    plot_mcnemar_heatmap(mc_df, out_dir / "fig_mcnemar_heatmap.png")

    print(f"\nAll figures written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
