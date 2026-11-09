import time

from debate import _run_turn, _emit, SOLVE_SCHEMA, ANSWER_FORMATS, _answer_of, _thought_of
from prompts import SOLVE

SOLO_TEMPERATURE = 0.1
MAX_TOKENS = 6000
SOLO_ITERATIONS = 1


def _solve_system(domain: str) -> str:
    return SOLVE.format(answer_format=ANSWER_FORMATS[domain])


def run_solo(question: str, domain: str, model: str, seed: int = None, on_event=None) -> dict:
    """Exactly SOLO_ITERATIONS greedy call, independent of debate rounds.
    Returns {answer, calls, raw, elapsed_seconds}.
    on_event: optional progress observer (same event format as run_debate)."""
    start = time.perf_counter()
    _emit(on_event, "agent_start", round=0, agent=0, model=model)
    sol = _run_turn(model, _solve_system(domain), question, SOLVE_SCHEMA,
                    SOLO_TEMPERATURE, seed)
    elapsed = time.perf_counter() - start
    _emit(on_event, "agent_done", round=0, agent=0, model=model, failed=sol is None,
          answer=_answer_of(sol), thought=_thought_of(sol), agree=None, elapsed=elapsed)
    _emit(on_event, "final", answer=_answer_of(sol), abstained=False, rounds=0, elapsed=elapsed)
    return {
        "answer": _answer_of(sol),
        "calls": SOLO_ITERATIONS,
        "raw": sol,
        "elapsed_seconds": elapsed,
    }
