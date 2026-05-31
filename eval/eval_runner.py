"""
eval_runner.py
--------------
Runs the 79 questions from questions_clean.csv against the local FastAPI (/ask)
and writes a results CSV and a run metadata file.

Usage:
    # The FastAPI server must be running (in a separate terminal):
    #   uvicorn api:app --reload
    #
    # Run from the project root:
    python eval/eval_runner.py --config baseline
    python eval/eval_runner.py --config no_threshold
    python eval/eval_runner.py --config no_history

The --config argument is used only as a label for output filenames.
The actual configuration (toggle flags in chatbot.py and api.py) must be
set manually before each run — see EVAL_GUIDE.md for details.

For quick tests: --limit 3 runs only the first 3 questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_URL = os.getenv("APP_API_URL", "http://127.0.0.1:8000/ask")
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_SLEEP_S = 0.2


# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_CSV = EVAL_DIR / "data" / "questions_clean.csv"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(csv_path: Path) -> list[dict]:
    """Load and parse questions from the CSV file, including chat history."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hist_raw = (r.get("chat_history") or "[]").strip()
            try:
                history = json.loads(hist_raw)
            except json.JSONDecodeError as e:
                print(f"  WARN id={r.get('id')}: chat_history not parseable: {e}")
                history = []

            rows.append({
                "id": int(r["id"]),
                "question": r["question"],
                "chat_history": history,
                "is_followup": int(r.get("is_followup", "0") or "0"),
                "answerable": int(r.get("answerable", "0") or "0"),
                "topic": r.get("topic", ""),
                "expected_answer": r.get("expected_answer", "") or "",
                "expected_sources": r.get("expected_sources", "") or "",
            })

    if not rows:
        raise ValueError("No questions found in CSV.")
    return rows


def call_api(
    api_url: str,
    api_key: str,
    question: str,
    chat_history: list,
    timeout_s: float,
) -> dict:
    """Send a question to the API and return a normalized response dict."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    payload = {
        "question": question,
        "chat_history": chat_history,
        "debug": True,  # retrieval_query is logged for analysis
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout_s)
        latency_s = time.perf_counter() - t0
    except requests.exceptions.RequestException as e:
        return {
            "ok": False,
            "http_status": "",
            "latency_s": time.perf_counter() - t0,
            "answer": "",
            "sources_returned": [],
            "retrieval_query": "",
            "no_context": None,
            "error": f"request_error: {type(e).__name__}: {e}",
        }

    status_code = resp.status_code

    try:
        data = resp.json()
    except Exception:
        data = None

    if status_code != 200:
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("detail", ""))[:500]
        else:
            detail = (resp.text or "")[:500]
        return {
            "ok": False,
            "http_status": status_code,
            "latency_s": latency_s,
            "answer": "",
            "sources_returned": [],
            "retrieval_query": "",
            "no_context": None,
            "error": f"http_{status_code}: {detail}",
        }

    if not isinstance(data, dict):
        return {
            "ok": False,
            "http_status": status_code,
            "latency_s": latency_s,
            "answer": "",
            "sources_returned": [],
            "retrieval_query": "",
            "no_context": None,
            "error": "invalid_json_response",
        }

    answer = str(data.get("answer", "") or "")
    sources_raw = data.get("sources", []) or []
    if not isinstance(sources_raw, list):
        sources_raw = []

    sources_clean = []
    for s in sources_raw:
        if isinstance(s, dict):
            sources_clean.append({
                "source": str(s.get("source", "")),
                "label": str(s.get("label", "")),
            })

    return {
        "ok": True,
        "http_status": status_code,
        "latency_s": latency_s,
        "answer": answer,
        "sources_returned": sources_clean,
        "retrieval_query": str(data.get("retrieval_query", "") or ""),
        "no_context": data.get("debug_no_context"),
        "error": "",
    }


def sources_to_str(sources: list[dict]) -> str:
    return json.dumps(sources, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run evaluation against the RAG chatbot API.")
    ap.add_argument(
        "--config",
        required=True,
        choices=["baseline", "no_threshold", "no_history"],
        help="Config label — used only for output filename and metadata.",
    )
    ap.add_argument("--questions", default=str(QUESTIONS_CSV), help="Path to questions_clean.csv")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="Output directory")
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S)
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N questions (0 = all)")
    args = ap.parse_args()

    questions_path = Path(args.questions)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"results_{args.config}.csv"
    out_meta = results_dir / f"run_meta_{args.config}.json"

    print(f"Loading: {questions_path}")
    questions = load_questions(questions_path)
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"  Loaded {len(questions)} questions")
    print(f"  Config label: {args.config}")
    print(f"  API URL:      {args.api_url}")
    print(f"  Output:       {out_csv}")
    print()

    # Health check before starting
    try:
        hc = requests.get(args.api_url.replace("/ask", "/health"), timeout=10)
        if hc.status_code == 200:
            print("  Health check: OK\n")
        else:
            print(f"  WARN: health check returned {hc.status_code}\n")
    except Exception as e:
        print(f"  WARN: health check failed: {e}")
        print("  If all calls fail, start FastAPI: uvicorn api:app --reload\n")

    fieldnames = [
        "id", "question", "is_followup", "answerable", "topic",
        "expected_answer", "expected_sources",
        "ok", "http_status", "latency_s",
        "answer", "sources_returned", "retrieval_query", "no_context",
        "error", "timestamp_utc", "config",
    ]

    start_time = time.perf_counter()
    ok_count = 0
    err_count = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, q in enumerate(questions, 1):
            preview = q["question"] if len(q["question"]) <= 70 else q["question"][:67] + "..."
            print(
                f"[{i:3d}/{len(questions)}] id={q['id']:2d} "
                f"{'[FU] ' if q['is_followup'] else '     '}"
                f"{'[ANS]' if q['answerable'] else '[N/A]'} {preview}"
            )

            res = call_api(
                api_url=args.api_url,
                api_key=args.api_key,
                question=q["question"],
                chat_history=q["chat_history"],
                timeout_s=args.timeout,
            )

            if res["ok"]:
                ok_count += 1
            else:
                err_count += 1
                print(f"       ERROR: {res['error'][:100]}")

            writer.writerow({
                "id": q["id"],
                "question": q["question"],
                "is_followup": q["is_followup"],
                "answerable": q["answerable"],
                "topic": q["topic"],
                "expected_answer": q["expected_answer"],
                "expected_sources": q["expected_sources"],
                "ok": res["ok"],
                "http_status": res["http_status"],
                "latency_s": f"{res['latency_s']:.4f}",
                "answer": res["answer"],
                "sources_returned": sources_to_str(res["sources_returned"]),
                "retrieval_query": res["retrieval_query"],
                "no_context": res["no_context"] if res["no_context"] is not None else "",
                "error": res["error"],
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "config": args.config,
            })
            f.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    total_time = time.perf_counter() - start_time

    meta = {
        "config": args.config,
        "n_questions": len(questions),
        "n_ok": ok_count,
        "n_error": err_count,
        "total_seconds": round(total_time, 2),
        "avg_seconds_per_question": round(total_time / max(1, len(questions)), 3),
        "api_url": args.api_url,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Config label is metadata only. The actual toggles "
            "(USE_EVIDENCE_THRESHOLD in chatbot.py, USE_HISTORY_REFORMULATION "
            "in api.py) must be set manually before running this script."
        ),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Done. Config: {args.config}")
    print(f"  Total: {len(questions)} questions")
    print(f"  OK:    {ok_count}")
    print(f"  ERR:   {err_count}")
    print(f"  Time:  {total_time:.1f}s  ({total_time / max(1, len(questions)):.2f}s/question)")
    print(f"  Wrote: {out_csv}")
    print(f"  Meta:  {out_meta}")
    print("=" * 60)

    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
