#!/usr/bin/env python3
"""
Merges all *.jsonl result files (solo / debate) and compares accuracy
for every combination of (dataset x model setup).

Outputs:
    - tables printed to the console
    - results/summary.csv one row per dataset x model setup
    - results/combined.csv pooled result over the datasets
    - results/all_records.csv flattened, merged records from all input files

Answer categories (mutually exclusive, they sum to 100%):
    correct  - the model answered and the answer matches gold
    wrong    - the model answered, but incorrectly
    abstain  - the debate ended with ABSTAIN (no consensus; `correct` == None)
    null     - no answer at all (solo: answer could not be parsed; `answer` == None)

Metrics:
    correct_% = correct / all tasks (strict accuracy)
    correct_%_answered = correct / (correct + wrong) (accuracy when the model actually answered)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def read_jsonl(path: Path):
    """Reads a JSONL file, robust to records glued together on one line
    (i.e. a missing '\\n' between two JSON objects)."""
    dec = json.JSONDecoder()
    records, skipped = [], 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            i = 0
            while i < len(line):
                try:
                    obj, i = dec.raw_decode(line, i)
                except json.JSONDecodeError as e:
                    print(f"  [!] {path.name}:{lineno} skipped corrupted fragment ({e})",
                          file=sys.stderr)
                    skipped += 1
                    break
                records.append(obj)
                while i < len(line) and line[i].isspace():
                    i += 1
    return records, skipped


def parse_filename(path: Path):
    """medical_solo_qwen3_5-9b.jsonl -> ('medical', 'solo');
    thinking_math_debate_... -> ('thinking_math', 'debate')."""
    m = re.match(r"^(?P<dataset>(?:thinking_)?[a-z]+)_(?P<cond>solo|debate)_", path.stem)
    if not m:
        return path.stem, "unknown"
    return m["dataset"], m["cond"]


def classify(rec: dict) -> str:
    """Returns one of: correct / wrong / abstain / null."""
    if rec.get("abstained") is True or str(rec.get("answer")).upper() == "ABSTAIN":
        return "abstain"
    if rec.get("answer") is None:
        return "null"
    if rec.get("correct") is True:
        return "correct"
    return "wrong"


def model_combo(rec: dict) -> str:
    if rec.get("models"):                      # debate
        return " + ".join(rec["models"])
    return rec.get("model") or "?"             # solo


def load_all(directory: Path) -> pd.DataFrame:
    rows = []
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {directory}")
    for f in files:
        recs, skipped = read_jsonl(f)
        dataset, cond_from_name = parse_filename(f)
        for r in recs:
            cfg = r.get("config") or {}
            rows.append({
                "source_file": f.name,
                "dataset": dataset,
                "condition": r.get("condition", cond_from_name),
                "models": model_combo(r),
                "task_id": r.get("task_id"),
                "gold": r.get("gold"),
                "answer": r.get("answer"),
                "correct_raw": r.get("correct"),
                "abstained": r.get("abstained", False),
                "debate_rounds": r.get("debate_rounds"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "think": cfg.get("think"),
                "status": classify(r),
            })
        print(f"  {f.name:<58} {len(recs):>4} records" + (f"  ({skipped} skipped)" if skipped else ""))
    return pd.DataFrame(rows)


def sort_setups(df: pd.DataFrame, extra_keys=()) -> pd.DataFrame:
    """Solo first, then debate; within each group sorted by correct_% descending."""
    df = df.copy()
    df["_c"] = df["condition"].map({"solo": 0, "debate": 1}).fillna(9)
    keys = [*extra_keys, "_c", "correct_%"]
    asc = [*([True] * len(extra_keys)), True, False]
    return df.sort_values(keys, ascending=asc).drop(columns="_c").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, condition, models), g in df.groupby(["dataset", "condition", "models"], sort=False):
        n = len(g)
        cnt = g["status"].value_counts()
        c, w, a, nl = (int(cnt.get(k, 0)) for k in ("correct", "wrong", "abstain", "null"))
        answered = c + w
        out.append({
            "dataset": dataset,
            "condition": condition,
            "models": models,
            "n": n,
            "correct": c,
            "wrong": w,
            "abstain": a,
            "null": nl,
            "correct_%": round(100 * c / n, 1),
            "wrong_%": round(100 * w / n, 1),
            "abstain_%": round(100 * a / n, 1),
            "null_%": round(100 * nl / n, 1),
            "correct_%_answered": round(100 * c / answered, 1) if answered else float("nan"),
            "avg_time_s": round(g["elapsed_seconds"].mean(), 1),
        })
    return sort_setups(pd.DataFrame(out), extra_keys=("dataset",))


def combine(summary: pd.DataFrame, datasets) -> pd.DataFrame:
    """Pooled result: sums the tasks of the given datasets for every setup (condition + models)."""
    sub = summary[summary["dataset"].isin(datasets)]
    g = sub.groupby(["condition", "models"], sort=False).agg(
        datasets=("dataset", lambda x: "+".join(x)),
        n=("n", "sum"), correct=("correct", "sum"), wrong=("wrong", "sum"),
        abstain=("abstain", "sum"), null=("null", "sum"),
    ).reset_index()
    for k in ("correct", "wrong", "abstain", "null"):
        g[f"{k}_%"] = (100 * g[k] / g["n"]).round(1)
    g["correct_%_answered"] = (100 * g["correct"] / (g["correct"] + g["wrong"])).round(1)
    return sort_setups(g)


def main():
    ap = argparse.ArgumentParser(description="Compare accuracy across datasets and model setups.")
    ap.add_argument("directory", nargs="?", default=str(RESULTS_DIR),
                    help="directory containing the .jsonl files (default: <script dir>/results)")
    ap.add_argument("--out", default=str(RESULTS_DIR),
                    help="directory for the output files (default: <script dir>/results)")
    ap.add_argument("--combine", nargs="+", default=["math", "medical"],
                    help="datasets to pool into the combined result (default: math medical)")
    args = ap.parse_args()

    print(f"Loading files from: {Path(args.directory).resolve()}")
    df = load_all(Path(args.directory))

    dups = df.duplicated(["dataset", "condition", "models", "task_id"]).sum()
    if dups:
        print(f"\n[!] Warning: {dups} duplicated (dataset, condition, models, task_id) rows",
              file=sys.stderr)

    summary = summarize(df)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)

    print("\n" + "=" * 100)
    print("SUMMARY: accuracy per dataset x model setup")
    print("=" * 100)
    for dataset, g in summary.groupby("dataset", sort=False):
        print(f"\n### {dataset}")
        print(g.drop(columns="dataset").to_string(index=False))

    # Matrix: correct_%, rows = setup, columns = dataset (+ pooled column)
    summary["setup"] = summary["condition"] + ": " + summary["models"]
    pivot = summary.pivot(index="setup", columns="dataset", values="correct_%")

    total = combine(summary, args.combine)
    label = "+".join(args.combine)
    total["setup"] = total["condition"] + ": " + total["models"]
    pivot[f"{label} (combined)"] = total.set_index("setup")["correct_%"]

    print("\n" + "=" * 100)
    print("MATRIX: correct_% (rows = setup, columns = dataset)")
    print("=" * 100)
    print(pivot.to_string())

    print("\n" + "=" * 100)
    print(f"COMBINED RESULT: {label} (tasks from the datasets pooled together)")
    print("=" * 100)
    print(total.drop(columns=["setup", "datasets"]).to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.drop(columns="setup").to_csv(out / "summary.csv", index=False)
    total.drop(columns="setup").to_csv(out / "combined.csv", index=False)
    df.to_csv(out / "all_records.csv", index=False)
    print(f"\nSaved: summary.csv, combined.csv, all_records.csv in {out}")


if __name__ == "__main__":
    main()