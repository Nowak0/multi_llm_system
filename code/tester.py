import json

from Agent import quit_ollama, check_ollama_model, Agent
from hive_mind import prepare_hive_mind
from prompts import SINGLE_MODEL

MODEL_1 = "qwen3.5:9b"
MODEL_2 = "gemma4:12b"
CONSOLE_LOGS = True
N_WORKERS = 2
N_RUNS_PER_TEST = 1
USED_MODELS = [MODEL_2, MODEL_1]


def run_tests():
    try:
        for m in USED_MODELS:
            check_ollama_model(m)

        questions = prepare_questions()
        for q in questions:
            handle_question(q)
    finally:
        for m in USED_MODELS:
            quit_ollama(m)
        pass


def prepare_questions():
    file = open('questions.txt')
    questions = file.read().splitlines()
    questions = [q for q in questions if q != '']
    return questions


def handle_question(question: str):
    result_dict = {
        "question": question,
        "hive_mind": {},
        "single_model": {}
    }

    for i in range(1,N_RUNS_PER_TEST+1):
        result_dict["hive_mind"][i] = run_hive_mind(question)

        for j, model in enumerate(USED_MODELS, 1):
            result_dict["single_model"][j] = run_single_model(question, model)

    with open('test.txt', 'a') as f:
        f.write(json.dumps(result_dict, indent=4))
        f.write("\n\n")
        f.flush()

    print(json.dumps(result_dict, indent=4))
    print("\n\n")


def run_hive_mind(question: str):
    return prepare_hive_mind(question, CONSOLE_LOGS, N_WORKERS, USED_MODELS)


def run_single_model(question: str, model: str):
    agent = Agent(model=model, role=SINGLE_MODEL)
    prompt = agent.build_chat_prompt(user_input=question)
    result = agent.ollama_chat(prompt=prompt, temperature=0.5, max_tokens=2000)
    return json.loads(result)["final_answer"]


run_tests()
