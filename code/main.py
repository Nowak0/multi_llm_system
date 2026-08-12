from Agent import check_ollama_model, quit_ollama
from hive_mind import prepare_hive_mind

MODEL_1 = "qwen3.5:9b"
MODEL_2 = "gemma4:12b"
CONSOLE_LOGS = True
N_WORKERS = 2
USED_MODELS = [MODEL_1, MODEL_2]


def main():
    try:
        for model in USED_MODELS:
            check_ollama_model(model)

        user_input = input("> ")
        final_answer = prepare_hive_mind(user_input, CONSOLE_LOGS, N_WORKERS, USED_MODELS)
        print("\n\nFINAL ANSWER:", final_answer, "\n\n")
    finally:
        for model in USED_MODELS:
            quit_ollama(model)
        if CONSOLE_LOGS:
            print(f"Finished shutting down all used models")

if __name__ == "__main__":
    main()
