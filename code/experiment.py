import json
import math
import os
import time
from datetime import datetime, timezone

from Agent import check_ollama_model, quit_ollama
from datasets_loader import load_records
from debate import run_debate, DEFAULT_ROUNDS, SOLVE_TEMPERATURE, DEBATE_TEMPERATURE, THINK_ENABLED
from baselines import run_solo, SOLO_TEMPERATURE, SOLO_ITERATIONS
from grading import is_correct, ABSTAIN
from utils import log

MODEL_QWEN = "qwen3.5:9b"
MODEL_GEMMA = "gemma4:12b"
SOLO_MODELS = [MODEL_QWEN, MODEL_GEMMA]
DEBATE_MODELS = [MODEL_QWEN, MODEL_GEMMA]

ROUNDS = DEFAULT_ROUNDS
SEED = 12345
DOMAINS = ["math", "medical"]
LIMIT = 100
RESULTS_DIR = "results"
CONSOLE_LOGS = True
THINK = THINK_ENABLED


def _sanitize_model_name(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _debate_tag(models: list[str]) -> str:
    return "-".join(_sanitize_model_name(m) for m in models)


def _solo_path(domain: str, model: str) -> str:
    return os.path.join(RESULTS_DIR, f"{domain}_solo_{_sanitize_model_name(model)}.jsonl")


def _debate_path(domain: str, models: list[str]) -> str:
    return os.path.join(RESULTS_DIR, f"{domain}_debate_{_debate_tag(models)}.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_row(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_solo_experiment(domain: str, model: str, records: list[dict]) -> list[dict]:
    path = _solo_path(domain, model)
    rows = []
    for i, rec in enumerate(records, 1):
        log(CONSOLE_LOGS, f"\n[{domain} solo {_sanitize_model_name(model)} {i}/{len(records)}] {rec['id']}")
        start = time.perf_counter()
        result = run_solo(rec["question"], domain, model, seed=SEED)
        elapsed = time.perf_counter() - start
        answer = result["answer"]
        correct = None if answer == ABSTAIN else is_correct(answer, rec["gold"], domain)

        row = {
            "task_id": rec["id"],
            "domain": domain,
            "condition": "solo",
            "model": model,
            "gold": rec["gold"],
            "answer": answer,
            "correct": correct,
            "iterations": SOLO_ITERATIONS,
            "elapsed_seconds": elapsed,
            "raw": result["raw"],
            "timestamp": _now_iso(),
            "config": {
                "model": model, "seed": SEED, "solo_temp": SOLO_TEMPERATURE,
                "think": THINK, "iterations": SOLO_ITERATIONS,
            },
        }
        _append_row(path, row)
        rows.append(row)
    return rows


def run_debate_experiment(domain: str, models: list[str], records: list[dict], rounds: int = ROUNDS) -> list[dict]:
    path = _debate_path(domain, models)
    rows = []
    for i, rec in enumerate(records, 1):
        log(CONSOLE_LOGS, f"\n[{domain} debate {_debate_tag(models)} {i}/{len(records)}] {rec['id']}")
        deb = run_debate(rec["question"], domain, models, rounds=rounds, seed=SEED,
                         console=CONSOLE_LOGS)
        answer = deb["answer"]
        correct = None if answer == ABSTAIN else is_correct(answer, rec["gold"], domain)

        row = {
            "task_id": rec["id"],
            "domain": domain,
            "condition": "debate",
            "models": models,
            "gold": rec["gold"],
            "answer": answer,
            "abstained": deb["abstained"],
            "debate_rounds": deb["rounds"],
            "correct": correct,
            "elapsed_seconds": deb["elapsed_seconds"],
            "trace": deb["trace"],
            "timestamp": _now_iso(),
            "config": {
                "models": models, "rounds": rounds, "seed": SEED,
                "solve_temp": SOLVE_TEMPERATURE, "debate_temp": DEBATE_TEMPERATURE,
                "think": THINK,
            },
        }
        _append_row(path, row)
        rows.append(row)
    return rows


def accuracy(rows) -> tuple[int, int, float]:
    """Accuracy counting abstentions/no-answer as wrong (overall accuracy)."""
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"] is True)
    return correct, n, (correct / n if n else 0.0)


def coverage_and_answered_acc(rows) -> tuple[float, float]:
    """Coverage = fraction answered; accuracy-on-answered = accuracy among the
    non-abstentions. (For non-abstaining conditions coverage is always 1.0.)"""
    answered = [r for r in rows if r["answer"] != ABSTAIN and r["answer"] is not None]
    n = len(rows)
    coverage = len(answered) / n if n else 0.0
    acc_answered = (sum(1 for r in answered if r["correct"] is True) / len(answered)
                    if answered else 0.0)
    return coverage, acc_answered


def mcnemar(rows_a, rows_b) -> dict:
    """Exact McNemar test on the paired correct/incorrect outcomes, matched by task_id."""
    by_id_b = {r["task_id"]: r for r in rows_b}
    first_ok_sec_wrong = first_wrong_sec_ok = 0
    for ra in rows_a:
        rb = by_id_b.get(ra["task_id"])
        if rb is None:
            continue
        a_ok = ra["correct"] is True
        b_ok = rb["correct"] is True
        if a_ok and not b_ok:
            first_ok_sec_wrong += 1
        elif b_ok and not a_ok:
            first_wrong_sec_ok += 1
    n = first_ok_sec_wrong + first_wrong_sec_ok
    if n == 0:
        p = 1.0
    else:
        k = min(first_ok_sec_wrong, first_wrong_sec_ok)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p = min(1.0, 2 * tail)
    return {"first_ok_sec_wrong": first_ok_sec_wrong, "first_wrong_sec_ok": first_wrong_sec_ok, "discordant": n, "p_value": p}


def report(domain: str, condition_rows: dict):
    """Print the results table and paired tests"""
    print(f"\n{'=' * 60}\nDOMAIN: {domain}\n{'=' * 60}")
    print(f"{'condition':<40}{'n':>6}{'acc':>8}{'coverage':>11}{'acc|answered':>15}")
    for label, rows in condition_rows.items():
        _, n, acc = accuracy(rows)
        cov, acc_ans = coverage_and_answered_acc(rows)
        print(f"{label:<40}{n:>6}{acc:>8.3f}{cov:>11.3f}{acc_ans:>15.3f}")

    debate_labels = [label for label in condition_rows if label.startswith("debate_")]
    if debate_labels:
        print("\nDebate vs baseline (b=debate-only-right, c=baseline-only-right)")
        for debate_label in debate_labels:
            for label, rows in condition_rows.items():
                if label == debate_label:
                    continue
                m = mcnemar(condition_rows[debate_label], rows)
                print(f"  {debate_label} vs {label:<25} first_ok_sec_wrong={m['first_ok_sec_wrong']:<4} "
                      f"first_wrong_sec_ok={m['first_wrong_sec_ok']:<4} "
                      f"discordant={m['discordant']:<4} p={m['p_value']:.4f}")


def run_all(domains: list[str] = DOMAINS, limit: int = LIMIT, solo_models: list[str] = SOLO_MODELS,
           debate_models: list[str] = DEBATE_MODELS, rounds: int = ROUNDS) -> dict:
    """Every condition on the same dataset records and same task order per domain,
    written to condition-specific files, so runs are directly comparable."""
    all_models = list(dict.fromkeys(solo_models + debate_models))
    try:
        for m in all_models:
            check_ollama_model(m)

        results = {}
        for domain in domains:
            records = load_records(domain, limit=limit)
            condition_rows = {}

            for model in solo_models:
                label = f"{_sanitize_model_name(model)}_solo"
                condition_rows[label] = run_solo_experiment(domain, model, records)

            for i in range(0, len(debate_models)):
                for j in range(i, len(debate_models)):
                    chosen_debate_models = [debate_models[i], debate_models[j]]
                    debate_label = f"debate_{_debate_tag(chosen_debate_models)}"
                    condition_rows[debate_label] = run_debate_experiment(
                        domain, chosen_debate_models, records, rounds=rounds)

            results[domain] = condition_rows
        return results
    finally:
        for m in all_models:
            quit_ollama(m)
        log(True, "Finished shutting down all models")


def main():
    results = run_all()
    for domain, condition_rows in results.items():
        report(domain, condition_rows)


if __name__ == "__main__":
    main()
