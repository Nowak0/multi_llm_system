import json
import re
import time
import requests
from utils import log

from Agent import Agent
from prompts import (
    SOLVE, DEBATE, ANSWER_FORMAT_MATH, ANSWER_FORMAT_MCQ,
)
from grading import equiv_fn, ABSTAIN, NO_SOLUTION

DEFAULT_ROUNDS = 3
SOLVE_TEMPERATURE = 0.05
DEBATE_TEMPERATURE = 0.1
MAX_TOKENS = 6000
THINK_ENABLED = False

ANSWER_FORMATS = {"math": ANSWER_FORMAT_MATH, "medical": ANSWER_FORMAT_MCQ}

SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "final_answer": {"type": "string"},
    },
    "required": ["thought", "final_answer"],
}
DEBATE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "agree": {"type": "string"},
        "final_answer": {"type": "string"},
    },
    "required": ["thought", "final_answer"],
}


def _extract_json(raw: str):
    text = (raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else text


def _parse(raw: str):
    try:
        return json.loads(_extract_json(raw))
    except json.JSONDecodeError:
        return None


def _turn_result(parsed, response, model, elapsed):
    return {
        "parsed": parsed,
        "response": response,
        "model": model,
        "elapsed_seconds": elapsed,
    }


def _seed_for(base_seed, round_idx: int, agent_idx: int, n_agents: int):
    """Deterministic per-(round, agent) seed derived from the condition's base
    seed, so a whole debate replays identically. None => let Ollama randomise."""
    if base_seed is None:
        return None
    return base_seed + round_idx * n_agents + agent_idx


def _answer_of(turn):
    """The final_answer string from a turn result (or a raw parsed dict), or None."""
    if not turn:
        return None

    parsed = turn.get("parsed") if "parsed" in turn else turn
    if not parsed:
        return None
    ans = parsed.get("final_answer")
    return None if ans is None else str(ans).strip()


def _thought_of(turn):
    if not turn:
        return ""

    parsed = turn.get("parsed") if "parsed" in turn else turn
    return (parsed or {}).get("thought", "")


def _is_answered(ans) -> bool:
    return ans is not None and ans != "" and ans.lower() != NO_SOLUTION.lower()


def _consensus(answers, equiv):
    real = [a for a in answers if _is_answered(a)]
    if len(real) != len(answers) or not real:
        return None
    first = real[0]
    if all(equiv(first, a) for a in real[1:]):
        return first
    return None


def _peers_block(solutions, self_idx):
    """Render every current solution for the debate prompt."""
    lines = []
    for i, sol in enumerate(solutions):
        who = "Your previous solution" if i == self_idx else f"Agent {i + 1}"
        ans = _answer_of(sol)
        thought = _thought_of(sol)
        if ans is None:
            lines.append(f"[{who}] (no valid answer produced)")
        else:
            lines.append(f"[{who}] answer: {ans}\n    reasoning: {thought}")
    return "\n".join(lines)


def _run_turn(model, system, user, schema, temperature, seed):
    """One model call -> turn result dict, or None."""
    agent = Agent(model=model, role=system)
    prompt = agent.build_chat_prompt(user)

    try:
        response = agent.ollama_chat(
            prompt=prompt, temperature=temperature, max_tokens=MAX_TOKENS,
            schema=schema, seed=seed, think=THINK_ENABLED,
        )
    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed for {model}: {e}")
        return None
    parsed = _parse(response["content"])
    return _turn_result(parsed, response, model, response.get("wall_time_seconds"))


def run_debate(question: str, domain: str, models: list[str], rounds: int = DEFAULT_ROUNDS,
               seed: int = None, console: bool = False) -> dict:
    """Run the full debate protocol for one question"""
    equiv = equiv_fn(domain)
    n = len(models)
    trace = []
    start = time.perf_counter()

    solutions = _run_independent_round(question, domain, models, seed, n, console)
    trace.append(_snapshot(0, models, solutions))

    agreed = _consensus([_answer_of(s) for s in solutions], equiv)
    if agreed is not None:
        log(console, f"Consensus on the independent round: {agreed!r}")
        return _make_result(agreed, abstained=False, rounds=0, trace=trace, elapsed=time.perf_counter() - start)

    for r in range(1, rounds + 1):
        solutions = _run_debate_round(question, domain, models, solutions, seed, r, n, console)
        trace.append(_snapshot(r, models, solutions))

        agreed = _consensus([_answer_of(s) for s in solutions], equiv)
        if agreed is not None:
            log(console, f"Consensus after debate round {r}: {agreed!r}")
            return _make_result(agreed, abstained=False, rounds=r, trace=trace, elapsed=time.perf_counter() - start)

    log(console, f"No consensus after {rounds} rounds -> ABSTAIN")
    return _make_result(ABSTAIN, abstained=True, rounds=rounds, trace=trace, elapsed=time.perf_counter() - start)


def _run_independent_round(question, domain, models, seed, n, console):
    """Round 0: every agent solves alone, with no visibility into peers."""
    answer_format = ANSWER_FORMATS[domain]
    solve_system = SOLVE.format(answer_format=answer_format)

    solutions = []
    for i, model in enumerate(models):
        sol = _run_turn(
            model, solve_system, question, SOLVE_SCHEMA,
            SOLVE_TEMPERATURE, _seed_for(seed, 0, i, n)
        )
        solutions.append(sol)
        log(console, f"[round 0] Agent {i + 1} ({model}) -> {_answer_of(sol)!r}")
    return solutions


def _run_debate_round(question, domain, models, solutions, seed, r, n, console):
    """One debate round: every agent sees peers' current answers and may revise."""
    answer_format = ANSWER_FORMATS[domain]
    debate_system = DEBATE.format(answer_format=answer_format)

    new_solutions = []
    for i, model in enumerate(models):
        user = (
            f"PROBLEM:\n{question}\n\n"
            f"CURRENT SOLUTIONS FROM ALL AGENTS:\n{_peers_block(solutions, i)}"
        )
        sol = _run_turn(
            model, debate_system, user, DEBATE_SCHEMA,
            DEBATE_TEMPERATURE, _seed_for(seed, r, i, n),
        )
        new_solutions.append(sol if sol is not None else solutions[i])
        log(console, f"[round {r}] Agent {i + 1} ({model}) -> {_answer_of(new_solutions[-1])!r}")
    return new_solutions


def _make_result(answer, abstained, rounds, trace, elapsed):
    """Build the standard return shape for run_debate."""
    return {
        "answer": answer,
        "abstained": abstained,
        "rounds": rounds,
        "trace": trace,
        "elapsed_seconds": elapsed,
    }


def _snapshot(round_idx, models, solutions):
    """Full, serialisable record of a round for the trace/JSONL log."""
    return {
        "round": round_idx,
        "solutions": [
            {
                "model": models[i],
                "thought": _thought_of(sol),
                "final_answer": _answer_of(sol),
                "elapsed_seconds": (sol or {}).get("elapsed_seconds"),
                "response": (sol or {}).get("response"),
            }
            for i, sol in enumerate(solutions)
        ],
    }
