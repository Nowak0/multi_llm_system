"""Interactive single-question entry point to the multi-agent debate system.

Type a problem at the prompt and get one final answer (or ABSTAIN if the agents
never reach consensus).
"""

from Agent import check_ollama_model, quit_ollama
from debate import run_debate, DEFAULT_ROUNDS
from grading import ABSTAIN
from utils import log

MODEL_QWEN = "qwen3.5:9b"
MODEL_GEMMA = "gemma4:12b"
MODELS = [MODEL_QWEN, MODEL_GEMMA]
CONSOLE_LOGS = True
ROUNDS = DEFAULT_ROUNDS


def main():
    try:
        for model in MODELS:
            check_ollama_model(model)

        domain = input("Domain [math/medical] (default math): ").strip().lower() or "math"
        if domain not in ("math", "medical"):
            domain = "math"

        question = input("> ")
        result = run_debate(question, domain, MODELS, rounds=ROUNDS, console=CONSOLE_LOGS)

        if result["answer"] == ABSTAIN:
            log(CONSOLE_LOGS,"\n\nFINAL ANSWER: ABSTAIN (agents did not reach consensus)\n\n")
        else:
            log(CONSOLE_LOGS,f"\n\nFINAL ANSWER: {result['answer']} "
                  f"(consensus after {result['rounds']} debate round(s))\n\n")
    finally:
        for model in MODELS:
            quit_ollama(model)
        log(CONSOLE_LOGS,"Finished shutting down all used models")


if __name__ == "__main__":
    main()
