import json
import re
from utils import log

from Agent import Agent
from prompts import (
    SOLVE, DEBATE, ANSWER_FORMAT_MATH, ANSWER_FORMAT_MCQ,
)
from grading import equiv_fn, ABSTAIN, NO_SOLUTION

DEFAULT_ROUNDS = 3
SOLVE_TEMPERATURE = 0.1
DEBATE_TEMPERATURE = 0.2
MAX_TOKENS = 6000

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


def _seed_for(base_seed, round_idx: int, agent_idx: int, n_agents: int):
    """Deterministic per-(round, agent) seed derived from the condition's base
    seed, so a whole debate replays identically. None => let Ollama randomise."""
    if base_seed is None:
        return None
    return base_seed + round_idx * n_agents + agent_idx


def _answer_of(solution):
    """The final_answer string from a parsed turn, or None."""
    if not solution:
        return None
    ans = solution.get("final_answer")
    return None if ans is None else str(ans).strip()


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
        thought = (sol or {}).get("thought", "") if sol else ""
        if ans is None:
            lines.append(f"[{who}] (no valid answer produced)")
        else:
            lines.append(f"[{who}] answer: {ans}\n    reasoning: {thought}")
    return "\n".join(lines)


def _run_turn(model, system, user, schema, temperature, seed):
    """One model call -> parsed dict or None."""
    import requests
    agent = Agent(model=model, role=system)
    prompt = agent.build_chat_prompt(user)
    try:
        raw = agent.ollama_chat(
            prompt=prompt, temperature=temperature, max_tokens=MAX_TOKENS,
            schema=schema, seed=seed, think=False,
        )
    except requests.exceptions.RequestException as e:
        print(f"Ollama request failed for {model}: {e}")
        return None
    return _parse(raw)


def run_debate(question: str, domain: str, models: list[str], rounds: int = DEFAULT_ROUNDS,
               seed: int = None, console: bool = False) -> dict:
    """Run the full debate protocol for one question"""
    answer_format = ANSWER_FORMATS[domain]
    equiv = equiv_fn(domain)
    n = len(models)
    trace = []

    solve_system = SOLVE.format(answer_format=answer_format)
    solutions = []
    for i, model in enumerate(models):
        sol = _run_turn(
            model, solve_system, question, SOLVE_SCHEMA,
            SOLVE_TEMPERATURE, _seed_for(seed, 0, i, n),
        )
        solutions.append(sol)
        log(console, f"[round 0] Agent {i + 1} ({model}) -> {_answer_of(sol)!r}")
    trace.append(_snapshot(0, models, solutions))

    agreed = _consensus([_answer_of(s) for s in solutions], equiv)
    if agreed is not None:
        log(console, f"Consensus on the independent round: {agreed!r}")
        return {"answer": agreed, "abstained": False, "rounds": 0, "trace": trace}

    debate_system = DEBATE.format(answer_format=answer_format)
    for r in range(1, rounds + 1):
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
        solutions = new_solutions
        trace.append(_snapshot(r, models, solutions))

        agreed = _consensus([_answer_of(s) for s in solutions], equiv)
        if agreed is not None:
            log(console, f"Consensus after debate round {r}: {agreed!r}")
            return {"answer": agreed, "abstained": False, "rounds": r, "trace": trace}

    log(console, f"No consensus after {rounds} rounds -> ABSTAIN")
    return {"answer": ABSTAIN, "abstained": True, "rounds": rounds, "trace": trace}


def _snapshot(round_idx, models, solutions):
    """Compact, serialisable record of a round for the trace/JSONL log."""
    return {
        "round": round_idx,
        "solutions": [
            {
                "model": models[i],
                "thought": (sol or {}).get("thought", "") if sol else "",
                "final_answer": _answer_of(sol),
            }
            for i, sol in enumerate(solutions)
        ],
    }
