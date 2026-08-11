import json
import random

from Pure.Agent import *

CALCULATION_RUNS = 3
CONSOLE_LOGS = True
N_NO_ROLE_WORKERS = 2
MODEL_OLD = "llama3.1:8b"
MODEL_HEAVY = "deepseek-r1:14b"
MODEL_REGULAR_1 = "qwen2.5:7b"
MODEL_REGULAR_2 = "gemma2:9b"
MODEL_LIGHT_ANALYTICAL = "phi4-mini"
MODEL_LIGHT_KNOWLEDGE = "gemma2:2b"

ROLE_RESEARCHER = """You are a researcher that gathers insight about given math problem.

TASK:
Provide (A) theory/method notes and (B) optional checkable constraints that help downstream agents.
You must NOT compute the specific solution. 
Your output MUST help downstream agents (calculators + evaluator) by providing:
1) factual background / methods (theory), AND
2) explicit, checkable constraints ONLY WHEN they are warranted (implied by the question and by standard definitions).

ABSOLUTE RULES:
- Do NOT perform any calculations.
- Do NOT solve the problem.
- Do NOT approximate any numeric answer for the specific instance.
- Do NOT select among candidate answers.
- Do NOT speculate or add heuristics.
- Output MUST be valid JSON only (no markdown, no extra text).

CONSTRAINT POLICY (VERY IMPORTANT):
- It is OK to output mostly null/empty constraints.
- You may fill a constraint ONLY if:
  (1) it is explicitly stated in the question, OR
  (2) it is a direct consequence of a standard definition that you explicitly wrote in theory_notes/methods.
- If you set ANY constraint field to a non-null value, add a corresponding entry to constraint_sources.

QUALITY REQUIREMENT:
- theory_notes MUST contain at least 3 items.
- methods MUST contain at least 1 item when the problem is non-trivial.
- constraints.validity_checks_for_evaluator should be present even if empty ([]).

REMINDER:
No numeric evaluation of candidates. No choosing “best fraction”. No computing convergents.
"""
ROLE_CALCULATOR_BASE = """You are a math specialist that calculates equations. 

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field without any reasoning.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

OUTPUT FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLE_CALCULATOR_ALGEBRA = """You are a math calculation engine specialized in symbolic simplification.

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

METHOD:
1) Rewrite the problem into a single simplified symbolic expression.
2) Simplify algebraically (factor/cancel/common denominators) before evaluating.
3) Only at the end produce the final exact result.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLE_CALCULATOR_STEPWISE = """You are a math calculation engine specialized in careful stepwise arithmetic.

TASK:
Compute the final numeric/symbolic result for the given expression/problem. You will be given
a full research and knowledge in a prompt with a problem. Be sure to use this research in order
to maximize the accuracy of given result.

METHOD:
1) Evaluate operations in a strict, explicit order (parentheses, powers, mult/div, add/sub).
2) Keep results as exact fractions whenever possible.
3) Perform cancellation and gcd reductions frequently to avoid overflow/mistakes.

OUTPUT (STRICT):
- Show your step-by-step work in the "thought" field.
- Provide the final exact result in the "final_answer" field.

FINAL_ANSWER RULES:
- Provide ONE final result only.
- Prefer EXACT form when possible (e.g., 2/3, sqrt(2), pi, 3*sqrt(5)/2).
- Do NOT round unless the user explicitly requests rounding/decimal approximation.
- If you must output a decimal (because the prompt requires it), output full precision available from exact conversion, without commentary.
- If there is a common fraction, do not decompose it into nominator and denominator -> provide x/y style. 
- Use standard ASCII: "sqrt(2)" not "√2".
- If there is no valid solution, output {"final_answer": "#no_solution"}.

FORMAT:
{
    "thought": "Your step-by-step reasoning here...",
    "final_answer": result
}
"""
ROLES_CALCULATOR = [ROLE_CALCULATOR_BASE, ROLE_CALCULATOR_ALGEBRA,ROLE_CALCULATOR_STEPWISE]
ROLE_EVALUATOR = """You are a STRICT result selector for a math task.

YOU WILL RECEIVE (as JSON in the user message):
- question: the original user question/expression
- research: factual, non-calculational insights relevant to the problem 
    (may include constraints, domains, definitions, typical pitfalls)
- possible_results: an array of candidate final answers (strings or numbers)

GOAL:
Select the best final answer ONLY from possible_results.
If no candidate is good enough OR quality is insufficient, output "#not_good".

ABSOLUTE CONSTRAINTS:
- Output EXACTLY one JSON object with exactly one key: "final_answer".
- "final_answer" MUST be either:
  (a) one element copied from possible_results EXACTLY as it appears, or
  (b) the string "#not_good".
- Do not explain and do not fix formatting
- NEVER accept a solution that is in a non-final form, 
    example: (-2 +- sqrt(20/3) * 18) or (10 choose 5) * 0.5^5 * 0.5^(10-5), etc.
    In such solutions it is better to return output "#not_good".

SELECTION POLICY (STRICT ORDER):

1) Filter: Remove invalid/null entries. Discard candidates violating 'research' (domain, precision, format).
2) Cluster: Group equivalent values (e.g., "0.5", "1/2", "0.50").
3) Evaluate: 
   - If a specific precision (e.g., "5 decimals") is required and no candidate meets it -> "#not_good".
   - If no precision specified prefer fractions
   - Pick the cluster with the most support.
   - If there is a tie or no clear majority (support < 2) and the problem is non-trivial -> "#not_good".

OUTPUT FORMAT (exactly):
{
  "final_answer": "<one of possible_results or #not_good>"
}

EXAMPLE (precision):
question: "Approximate pi to 5 decimal places"
possible_results: ["3.14", "3.14159", "pi"]
-> choose "3.14159"
If possible_results: ["3.14", "pi"]
-> "#not_good"
"""

def console_log(message):
    if CONSOLE_LOGS:
        print(message)

def run_agent(agent: Agent, input: str, temperature: float = None, max_tokens: int = None):
    """Build a proper prompt for given agent and runs a chat with it"""
    prompt = agent.build_chat_prompt(input)

    if temperature is not None and max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    elif temperature is not None:
        return agent.ollama_chat(prompt=prompt, temperature=temperature)
    elif max_tokens is not None:
        return agent.ollama_chat(prompt=prompt, max_tokens=max_tokens)
    else:
        return agent.ollama_chat(prompt=prompt)


def handle_research(agent: Agent, user_input, temperature: float, max_tokens: int):
    """Gathers insight from researcher and injects it into user's query"""
    raw_insight = run_agent(agent=agent, input=user_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.dumps(json.loads(raw_insight), indent=2)
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Research agent failed to produce valid JSON")
    finally:
        quit_ollama(agent.model)


def run_worker(role: str, input: str, model: str, max_tokens: int):
    agent = Agent(model=model, role=role)
    result = run_agent(agent=agent, input=input, temperature=0.05, max_tokens=max_tokens)

    try:
        data = json.loads(result)
        console_log(f"Worker thought:\n{data.get('thought')}\n")
        return data.get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")


def handle_worker(start_input: str, possible_results: str, model: str, max_tokens: int, number_of_runs: int = 1):
    for _ in range(number_of_runs):
        idx = random.randint(0, len(ROLES_CALCULATOR) - 1)
        role = ROLES_CALCULATOR[idx]
        result = run_worker(role=role, input=start_input, model=model, max_tokens=max_tokens)
        possible_results += f"\n- {result}"
        console_log(f"single calculation ({idx + 1}): {result}")

    quit_ollama(model)
    return possible_results


def handle_calculations(evaluator: Agent, user_input: str, research: str, max_tokens: int):
    """Runs calculations with varying temperature"""
    possible_results = ""
    output_evaluation = ""
    start_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}
    """
    console_log("START CALCULATIONS")

    possible_results = handle_worker(start_input=start_input, possible_results=possible_results, model=MODEL_HEAVY,
                                     max_tokens=max_tokens, number_of_runs=CALCULATION_RUNS)
    count_runs = 0
    while count_runs < CALCULATION_RUNS * 3:
        console_log("POSSIBLE ANSWERS: " + possible_results)

        output_evaluation = handle_evaluation(agent=evaluator, user_input=user_input, research=research,
                                              results=possible_results, temperature=0.05, max_tokens=1000)
        console_log("evaluation: " + output_evaluation)

        if output_evaluation != "#not_good":
            break

        console_log(f"Answers not good enough")

        full_input = f"""{start_input}

        POSSIBLE ANSWERS: {possible_results}"""
        possible_results = handle_worker(start_input=full_input, possible_results=possible_results, model=MODEL_HEAVY,
                                         max_tokens=max_tokens, number_of_runs=1)
        count_runs += 1

    if count_runs >= CALCULATION_RUNS * 3:
        raise Exception("Could not find reliable answer")

    quit_ollama(evaluator.model)
    return output_evaluation


def handle_evaluation(agent: Agent, user_input, research: str, results: str, temperature: float, max_tokens: int):
    """Adds possible results to user's query and evaluates them"""
    new_input = f"""
    QUESTION: {user_input}

    RESEARCH: {research}

    POSSIBLE ANSWERS: {results}
    """

    output = run_agent(agent=agent, input=new_input, temperature=temperature, max_tokens=max_tokens)

    try:
        return json.loads(output).get("final_answer")
    except json.decoder.JSONDecodeError:
        raise RuntimeError("Calculation agent failed to produce valid JSON")
    except KeyError:
        raise RuntimeError("Evaluation JSON missing 'final_answer' key")


def different_roles_defined(user_input: str):
    """Model where different roles for agents are defined (researcher, calculator, evaluator)"""
    agent_researcher = Agent(model=MODEL_LIGHT_ANALYTICAL, role=ROLE_RESEARCHER)
    agent_evaluator = Agent(model=MODEL_REGULAR_1, role=ROLE_EVALUATOR)
    user_input = input("> ")

    research = handle_research(agent=agent_researcher, user_input=user_input, temperature=0.15, max_tokens=2000)
    console_log(research)

    results = handle_calculations(evaluator=agent_evaluator, user_input=user_input,
                                  research=research, max_tokens=2000)

    print("AGENT EVALUATION: ", results)