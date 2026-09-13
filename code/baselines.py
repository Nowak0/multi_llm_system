import time

from debate import _run_turn, SOLVE_SCHEMA, ANSWER_FORMATS, _answer_of
from prompts import SOLVE

SOLO_TEMPERATURE = 0.1
MAX_TOKENS = 6000
SOLO_ITERATIONS = 1


def _solve_system(domain: str) -> str:
    return SOLVE.format(answer_format=ANSWER_FORMATS[domain])


def run_solo(question: str, domain: str, model: str, seed: int = None) -> dict:
    """Exactly SOLO_ITERATIONS greedy call, independent of debate rounds.
    Returns {answer, calls, raw, elapsed_seconds}."""
    start = time.perf_counter()
    sol = _run_turn(model, _solve_system(domain), question, SOLVE_SCHEMA,
                    SOLO_TEMPERATURE, seed)
    elapsed = time.perf_counter() - start
    return {
        "answer": _answer_of(sol),
        "calls": SOLO_ITERATIONS,
        "raw": sol,
        "elapsed_seconds": elapsed,
    }
