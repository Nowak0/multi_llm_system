import json
import math
import os

from Agent import check_ollama_model, quit_ollama
from datasets_loader import load_records
from debate import run_debate, DEFAULT_ROUNDS, SOLVE_TEMPERATURE, DEBATE_TEMPERATURE
from baselines import run_solo, run_self_consistency, SOLO_TEMPERATURE, SC_TEMPERATURE
from grading import is_correct, ABSTAIN
from utils import log

MODEL_QWEN = "qwen3.5:9b"
MODEL_GEMMA = "gemma4:12b"
MODELS = [MODEL_QWEN, MODEL_GEMMA]

ROUNDS = DEFAULT_ROUNDS
K = 2 * (ROUNDS + 1)
SEED = 12345
DOMAINS = ["math", "medical"]
LIMIT = 100
RESULTS_DIR = "results"
CONSOLE_LOGS = True

CONDITIONS = ["qwen_solo", "gemma_solo", "qwen_sc", "gemma_sc", "debate"]


def _run_conditions(record: dict) -> dict:
    """Run every condition on one question and return {condition: answer}."""
    q, domain = record["question"], record["domain"]
    answers = {}

    answers["qwen_solo"] = run_solo(q, domain, MODEL_QWEN, seed=SEED)["answer"]
    answers["gemma_solo"] = run_solo(q, domain, MODEL_GEMMA, seed=SEED)["answer"]
    # answers["qwen_sc"] = run_self_consistency(q, domain, MODEL_QWEN, K, seed=SEED)["answer"]
    # answers["gemma_sc"] = run_self_consistency(q, domain, MODEL_GEMMA, K, seed=SEED)["answer"]

    deb = run_debate(q, domain, MODELS, rounds=ROUNDS, seed=SEED, console=CONSOLE_LOGS)
    answers["debate"] = deb["answer"]
    answers["debate_rounds"] = deb["rounds"]
    return answers


def run_domain(domain: str, limit: int) -> list[dict]:
    """Run all conditions over one domain; append each question's results to a
    JSONL file and return the in-memory rows."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{domain}.jsonl")
    records = load_records(domain, limit=limit)
    rows = []

    with open(path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            log(CONSOLE_LOGS, f"\n[{domain} {i}/{len(records)}] {rec['id']}")
            answers = _run_conditions(rec)

            correct = {
                cond: (None if answers[cond] == ABSTAIN
                       else is_correct(answers[cond], rec["gold"], domain))
                for cond in CONDITIONS
            }

            row = {
                "id": rec["id"],
                "domain": domain,
                "gold": rec["gold"],
                "answers": answers,
                "correct": correct,
                "config": {
                    "models": MODELS, "rounds": ROUNDS, "k": K, "seed": SEED,
                    "solve_temp": SOLVE_TEMPERATURE, "debate_temp": DEBATE_TEMPERATURE,
                    "solo_temp": SOLO_TEMPERATURE, "sc_temp": SC_TEMPERATURE,
                },
            }

            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)

    return rows


def accuracy(rows, cond) -> tuple[int, int, float]:
    """Accuracy counting abstentions as wrong (overall accuracy)."""
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"][cond] is True)
    return correct, n, (correct / n if n else 0.0)


def coverage_and_answered_acc(rows, cond) -> tuple[float, float]:
    """Coverage = fraction answered; accuracy-on-answered = accuracy among the
    non-abstentions. (For non-abstaining conditions coverage is always 1.0.)"""
    answered = [r for r in rows if r["answers"][cond] != ABSTAIN]
    n = len(rows)
    coverage = len(answered) / n if n else 0.0
    acc_answered = (sum(1 for r in answered if r["correct"][cond] is True) / len(answered)
                    if answered else 0.0)
    return coverage, acc_answered


def mcnemar(rows, cond_a, cond_b) -> dict:
    """Exact McNemar test on the paired correct/incorrect outcomes."""
    first_ok_sec_wrong = first_wrong_sec_ok = 0
    for r in rows:
        a_ok = r["correct"][cond_a] is True
        b_ok = r["correct"][cond_b] is True
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


def report(all_rows: dict):
    """Print the results table and paired tests for each domain."""
    for domain, rows in all_rows.items():
        print(f"\n{'=' * 60}\nDOMAIN: {domain}  (N={len(rows)})\n{'=' * 60}")
        print(f"{'condition':<14}{'acc':>8}{'coverage':>11}{'acc|answered':>15}")
        for cond in CONDITIONS:
            _, _, acc = accuracy(rows, cond)
            cov, acc_ans = coverage_and_answered_acc(rows, cond)
            print(f"{cond:<14}{acc:>8.3f}{cov:>11.3f}{acc_ans:>15.3f}")

        print("\nMcNemar: debate vs baseline (b=debate-only-right, c=baseline-only-right)")
        for cond in CONDITIONS:
            if cond == "debate":
                continue
            m = mcnemar(rows, "debate", cond)
            print(f"  debate vs {cond:<11} first_ok_sec_wrong={m['first_ok_sec_wrong']:<4} "
                  f"first_wrong_sec_ok={m['first_wrong_sec_ok']:<4} "
                  f"discordant={m['discordant']:<4} p={m['p_value']:.4f}")


def main():
    try:
        for m in MODELS:
            check_ollama_model(m)
        all_rows = {domain: run_domain(domain, LIMIT) for domain in DOMAINS}
        report(all_rows)
    finally:
        for m in MODELS:
            quit_ollama(m)
        log(True, "Finished shutting down all models")


if __name__ == "__main__":
    main()
