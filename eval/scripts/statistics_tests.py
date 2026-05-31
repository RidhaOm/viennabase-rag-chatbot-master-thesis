"""
statistics_tests.py
-------------------
Runs pairwise statistical comparisons between the three evaluation configurations
for each binary metric (acceptable, grounded, correct_fallback, false_refusal).

Methods:
  - McNemar test (exact, paired, dichotomous): for paired binary results per question id.
    Suitable for comparing the same 79 questions across configurations.
  - Wilson score confidence intervals: for individual proportions per configuration.

Output files:
  eval/analysis/mcnemar_results.csv       — pairwise test results
  eval/analysis/confidence_intervals.csv  — point estimates with 95% CIs

Usage:
    python statistics_tests.py \
        --labels-dir eval/labels \
        --out-dir eval/analysis
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy import stats


CONFIGS = ["baseline", "no_threshold", "no_history"]

# Pairs relevant for the RQ3 ablation analysis
PAIRS = [
    ("baseline", "no_threshold"),  # effect of the distance threshold
    ("baseline", "no_history"),    # effect of history-aware reformulation
]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom  = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    low  = (center - spread) / denom
    high = (center + spread) / denom
    return (max(0.0, low), min(1.0, high))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact McNemar test using a binomial distribution.
    b = discordant pairs where A=1, B=0
    c = discordant pairs where A=0, B=1
    Returns a two-sided p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(labels_dir: Path, config: str) -> pd.DataFrame:
    path = labels_dir / f"labels_{config}.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df = df.set_index("id")
    return df


# ---------------------------------------------------------------------------
# Binary converters
# ---------------------------------------------------------------------------

def to_binary_acceptable(aq) -> int:
    """good or ok -> 1, bad -> 0, missing -> -1"""
    s = str(aq).strip().lower()
    if s in ("good", "ok"):
        return 1
    if s == "bad":
        return 0
    return -1


def to_binary_good(aq) -> int:
    """good -> 1, ok/bad -> 0, missing -> -1"""
    s = str(aq).strip().lower()
    if s == "good":
        return 1
    if s in ("ok", "bad"):
        return 0
    return -1


def to_binary_grounded(v) -> int:
    try:
        iv = int(v)
        if iv in (0, 1):
            return iv
    except (ValueError, TypeError):
        pass
    return -1


def to_binary_correct_fallback(v) -> int:
    """1 -> 1, 0 -> 0, NA -> -1 (not applicable, skipped in analysis)"""
    if pd.isna(v):
        return -1
    try:
        iv = int(float(v))
        if iv in (0, 1):
            return iv
    except (ValueError, TypeError):
        pass
    s = str(v).strip().upper()
    if s == "1":
        return 1
    if s == "0":
        return 0
    return -1


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRICS = [
    ("acceptable",       to_binary_acceptable,       "answer_quality",    "good or ok = success, bad = failure"),
    ("good",             to_binary_good,             "answer_quality",    "Strict: only 'good' is a success"),
    ("grounded",         to_binary_grounded,         "is_grounded",       "1 = grounded, 0 = ungrounded"),
    ("correct_fallback", to_binary_correct_fallback, "correct_fallback",  "1 = correct fallback, 0 = wrong; NA skipped"),
]


def extract_series(df: pd.DataFrame, col: str, fn) -> pd.Series:
    return df[col].apply(fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="eval/labels")
    ap.add_argument("--out-dir",    default="eval/analysis")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for cfg in CONFIGS:
        data[cfg] = load_labels(labels_dir, cfg)
    print(f"Loaded {len(CONFIGS)} configs, {len(data['baseline'])} questions each")

    # ------------------------------------------------------------------
    # 1) Wilson confidence intervals (one row per config + metric)
    # ------------------------------------------------------------------
    ci_rows = []
    for cfg in CONFIGS:
        df = data[cfg]
        for metric_name, fn, col, desc in METRICS:
            series = extract_series(df, col, fn)
            valid  = series[series != -1]
            n      = len(valid)
            successes = int((valid == 1).sum())
            rate      = successes / n if n else 0.0
            low, high = wilson_ci(successes, n)
            ci_rows.append({
                "config":      cfg,
                "metric":      metric_name,
                "n_valid":     n,
                "successes":   successes,
                "rate":        round(rate, 4),
                "ci_low_95":   round(low,  4),
                "ci_high_95":  round(high, 4),
                "description": desc,
            })

    ci_df  = pd.DataFrame(ci_rows)
    ci_out = out_dir / "confidence_intervals.csv"
    ci_df.to_csv(ci_out, index=False, encoding="utf-8")
    print(f"\nWrote confidence intervals: {ci_out}")
    print(ci_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 2) Paired McNemar tests
    # ------------------------------------------------------------------
    mc_rows = []
    for cfg_a, cfg_b in PAIRS:
        df_a = data[cfg_a]
        df_b = data[cfg_b]
        common_ids = df_a.index.intersection(df_b.index)

        for metric_name, fn, col, desc in METRICS:
            a = extract_series(df_a.loc[common_ids], col, fn)
            b = extract_series(df_b.loc[common_ids], col, fn)

            mask = (a != -1) & (b != -1)
            a_v  = a[mask]
            b_v  = b[mask]
            n_paired = len(a_v)

            if n_paired == 0:
                mc_rows.append({
                    "config_a": cfg_a, "config_b": cfg_b,
                    "metric": metric_name,
                    "n_paired": 0,
                    "a_successes": 0, "b_successes": 0,
                    "both_1": 0, "both_0": 0,
                    "only_a_1": 0, "only_b_1": 0,
                    "p_value_two_sided": None,
                    "effect_direction": "n/a",
                })
                continue

            both_1   = int(((a_v == 1) & (b_v == 1)).sum())
            both_0   = int(((a_v == 0) & (b_v == 0)).sum())
            only_a_1 = int(((a_v == 1) & (b_v == 0)).sum())
            only_b_1 = int(((a_v == 0) & (b_v == 1)).sum())

            p_value = mcnemar_exact(b=only_a_1, c=only_b_1)

            if only_a_1 > only_b_1:
                direction = f"{cfg_a} > {cfg_b}"
            elif only_b_1 > only_a_1:
                direction = f"{cfg_b} > {cfg_a}"
            else:
                direction = "tie"

            mc_rows.append({
                "config_a":             cfg_a,
                "config_b":             cfg_b,
                "metric":               metric_name,
                "n_paired":             n_paired,
                "a_successes":          int((a_v == 1).sum()),
                "b_successes":          int((b_v == 1).sum()),
                "both_1":               both_1,
                "both_0":               both_0,
                "only_a_1":             only_a_1,
                "only_b_1":             only_b_1,
                "p_value_two_sided":    round(p_value, 4),
                "significant_at_0.05":  p_value < 0.05,
                "effect_direction":     direction,
            })

    mc_df  = pd.DataFrame(mc_rows)
    mc_out = out_dir / "mcnemar_results.csv"
    mc_df.to_csv(mc_out, index=False, encoding="utf-8")
    print(f"\nWrote McNemar results: {mc_out}")
    print(mc_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
