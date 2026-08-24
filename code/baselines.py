from debate import _run_turn, SOLVE_SCHEMA, ANSWER_FORMATS, _answer_of, _is_answered
from prompts import SOLVE
from grading import equiv_fn

SOLO_TEMPERATURE = 0.0
SC_TEMPERATURE = 0.2
MAX_TOKENS = 6000


def _solve_system(domain: str) -> str:
    return SOLVE.format(answer_format=ANSWER_FORMATS[domain])


def run_solo(question: str, domain: str, model: str, seed: int = None) -> dict:
    """One greedy call. Returns {answer, calls, raw}."""
    sol = _run_turn(model, _solve_system(domain), question, SOLVE_SCHEMA,
                    SOLO_TEMPERATURE, seed)
    return {"answer": _answer_of(sol), "calls": 1, "raw": sol}


def _majority(samples: list[str], equiv) -> str:
    groups = []  # [representative, count]
    for a in samples:
        for g in groups:
            if equiv(g[0], a):
                g[1] += 1
                break
        else:
            groups.append([a, 1])
    if not groups:
        return None
    groups.sort(key=lambda g: -g[1])
    return groups[0][0]


def run_self_consistency(question: str, domain: str, model: str, k: int,
                         seed: int = None) -> dict:
    """k sampled calls + majority vote. Self-consistency always commits to an
    answer (no abstain) - its wrong answers are pure hallucination, which is the
    point of the comparison. Returns {answer, calls, votes}."""
    system = _solve_system(domain)
    equiv = equiv_fn(domain)
    votes = []
    for j in range(k):
        s = None if seed is None else seed + j
        sol = _run_turn(model, system, question, SOLVE_SCHEMA, SC_TEMPERATURE, s)
        ans = _answer_of(sol)
        if _is_answered(ans):
            votes.append(ans)
    return {"answer": _majority(votes, equiv), "calls": k, "votes": votes}
