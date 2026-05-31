"""
metrics.py
----------
Computes evaluation metrics per configuration from the annotated label CSVs
and writes a summary table.

Metrics:
  1) answer_quality counts and rates (good / ok / bad)
  2) acceptable_rate = (good + ok) / total
  3) good_rate = good / total
  4) grounded_rate = is_grounded=1 / total
  5) fallback_precision = correct_fallbacks / total refusals
  6) fallback_recall = correct_fallbacks / total unanswerable questions
  7) false_refusal_rate = wrong refusals / total answerable questions
  8) retrieval_hit_rate = fraction of answerable questions citing an expected source
  9) latency stats: median, p90, mean

Usage:
    python metrics.py \
        --labels-dir eval/labels \
        --results-dir eval/results \
        --out eval/analysis/metrics_summary.csv

Expects label CSVs and result CSVs named:
    labels_<config>.csv
    results_<config>.csv  (from eval_runner.py)
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


CONFIGS = ["baseline", "no_threshold", "no_history"]


def load_labels(labels_dir: Path, config: str) -> pd.DataFrame:
    path = labels_dir / f"labels_{config}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")


def load_results(results_dir: Path, config: str) -> pd.DataFrame:
    """Load results CSV for latency and source data. Returns None if not found."""
    path = results_dir / f"results_{config}.csv"
    if not path.exists():
        print(f"  Note: {path} not found — latency stats will be skipped")
        return None
    return pd.read_csv(path, encoding="utf-8")


def normalize_expected_sources(s: str) -> List[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    tokens = [t.strip() for t in str(s).split("|")]
    return [t for t in tokens if t]


def parse_returned_labels(s: str) -> List[str]:
    """Parse 'faq | AGBs' into ['faq', 'AGBs']."""
    if pd.isna(s) or not str(s).strip():
        return []
    return [t.strip() for t in str(s).split("|") if t.strip()]


def compute_retrieval_hit_rate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    For each answerable question, check whether the expected source appears
    among the returned source labels.
    """
    relevant = df[df["answerable"] == 1].copy()
    if len(relevant) == 0:
        return {"n": 0, "hits": 0, "hit_rate": None}

    hits = 0
    for _, row in relevant.iterrows():
        expected = normalize_expected_sources(row.get("expected_sources", ""))
        returned = parse_returned_labels(row.get("sources_returned_labels", ""))
        if not expected:
            continue
        if any(e in returned for e in expected):
            hits += 1

    return {
        "n": len(relevant),
        "hits": hits,
        "hit_rate": hits / len(relevant) if len(relevant) else None,
    }


def quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    w = idx - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def compute_metrics(
    config: str,
    labels_df: pd.DataFrame,
    results_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Compute all metrics for one configuration."""
    total = len(labels_df)

    aq = labels_df["answer_quality"].astype(str).str.strip()
    n_good = (aq == "good").sum()
    n_ok   = (aq == "ok").sum()
    n_bad  = (aq == "bad").sum()

    acceptable_rate = (n_good + n_ok) / total if total else 0.0
    good_rate = n_good / total if total else 0.0
    bad_rate  = n_bad  / total if total else 0.0

    ig = pd.to_numeric(labels_df["is_grounded"], errors="coerce")
    n_grounded = (ig == 1).sum()
    grounded_rate = n_grounded / total if total else 0.0

    def cf_str(v):
        if pd.isna(v):
            return "NA"
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()

    cf = labels_df["correct_fallback"].apply(cf_str)
    answerable = labels_df["answerable"].astype(int)

    unanswerable_total = (answerable == 0).sum()
    answerable_total   = (answerable == 1).sum()

    cf_eq_1 = cf == "1"
    cf_eq_0 = cf == "0"

    correct_fallbacks_on_unanswerable = ((answerable == 0) & cf_eq_1).sum()
    wrong_answers_on_unanswerable     = ((answerable == 0) & cf_eq_0).sum()
    wrong_refusals_on_answerable      = ((answerable == 1) & cf_eq_0).sum()

    total_refusals = correct_fallbacks_on_unanswerable + wrong_refusals_on_answerable
    fallback_precision = (
        correct_fallbacks_on_unanswerable / total_refusals
        if total_refusals > 0 else None
    )
    fallback_recall = (
        correct_fallbacks_on_unanswerable / unanswerable_total
        if unanswerable_total > 0 else None
    )
    false_refusal_rate = (
        wrong_refusals_on_answerable / answerable_total
        if answerable_total > 0 else None
    )

    metrics = {
        "config": config,
        "n_total": int(total),
        "n_answerable": int(answerable_total),
        "n_unanswerable": int(unanswerable_total),
        "n_good": int(n_good),
        "n_ok": int(n_ok),
        "n_bad": int(n_bad),
        "acceptable_rate": round(acceptable_rate, 4),
        "good_rate": round(good_rate, 4),
        "bad_rate": round(bad_rate, 4),
        "n_grounded": int(n_grounded),
        "grounded_rate": round(grounded_rate, 4),
        "correct_fallbacks": int(correct_fallbacks_on_unanswerable),
        "wrong_answers_on_unanswerable": int(wrong_answers_on_unanswerable),
        "wrong_refusals_on_answerable": int(wrong_refusals_on_answerable),
        "fallback_precision": round(fallback_precision, 4) if fallback_precision is not None else None,
        "fallback_recall": round(fallback_recall, 4) if fallback_recall is not None else None,
        "false_refusal_rate": round(false_refusal_rate, 4) if false_refusal_rate is not None else None,
    }

    retrieval_stats = compute_retrieval_hit_rate(labels_df)
    metrics["retrieval_n"] = retrieval_stats["n"]
    metrics["retrieval_hits"] = retrieval_stats["hits"]
    metrics["retrieval_hit_rate"] = (
        round(retrieval_stats["hit_rate"], 4)
        if retrieval_stats["hit_rate"] is not None else None
    )

    if results_df is not None and "latency_s" in results_df.columns:
        lats = pd.to_numeric(results_df["latency_s"], errors="coerce").dropna().tolist()
        if lats:
            metrics["latency_median_s"] = round(statistics.median(lats), 3)
            metrics["latency_mean_s"]   = round(sum(lats) / len(lats), 3)
            metrics["latency_p90_s"]    = round(quantile(lats, 0.9), 3)
            metrics["latency_min_s"]    = round(min(lats), 3)
            metrics["latency_max_s"]    = round(max(lats), 3)
        else:
            for k in ("latency_median_s", "latency_mean_s", "latency_p90_s",
                      "latency_min_s", "latency_max_s"):
                metrics[k] = None

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir",  default="eval/labels",                    help="Directory with labels_*.csv")
    ap.add_argument("--results-dir", default="eval/results",                   help="Directory with results_*.csv")
    ap.add_argument("--out",         default="eval/analysis/metrics_summary.csv", help="Output summary CSV")
    args = ap.parse_args()

    labels_dir  = Path(args.labels_dir)
    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg} ===")
        labels  = load_labels(labels_dir, cfg)
        results = load_results(results_dir, cfg)

        if results is not None:
            merged = labels.merge(
                results[["id", "latency_s", "sources_returned"]].drop_duplicates("id"),
                on="id", how="left", suffixes=("", "_res"),
            )
            m = compute_metrics(cfg, merged, results)
        else:
            m = compute_metrics(cfg, labels, None)

        rows.append(m)
        for k, v in m.items():
            print(f"  {k:30s} {v}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote summary: {out_path}")

    json_path = out_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
