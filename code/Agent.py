import subprocess
import time
import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 600


def check_ollama_model(model: str):
    """Ensure an Ollama model is available locally; pull if missing"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        installed = {line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()}
        if model not in installed:
            subprocess.run(["ollama", "pull", model], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while checking or pulling model '{model}':\n{e}")
        raise


def quit_ollama(model: str):
    """Quit Ollama. Saves a lot of RAM"""
    try:
        subprocess.run(["ollama", "stop", model])
    except Exception as e:
        print(e)


class Agent():
    def __init__(self, model, role):
        self.model = model
        self.role = role

    def build_chat_prompt(self, user_input):
        """Build a chat prompt"""
        return [
            {"role": "system", "content": self.role},
            {"role": "user", "content": user_input}
        ]

    def ollama_chat(self, prompt: list[dict], temperature: float = 0.7, max_tokens: int = 2000, schema: dict = None,
                     seed: int = None, think: bool = False):
        """Get a response from Ollama /api/chat. Returns the full response
        envelope plus content and wall-clock timing."""
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        if seed is not None:
            options["seed"] = seed

        package = {
            "model": self.model,
            "messages": prompt,
            "options": options,
            "stream": False,
            "format": schema if schema is not None else "json",
            "think": think
        }

        start = time.perf_counter()
        response = requests.post(OLLAMA_CHAT_URL, json=package, timeout=OLLAMA_TIMEOUT)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        body = response.json()
        message = body.get("message", {}) or {}

        return {
            "content": message.get("content", "") or "",
            "thinking": message.get("thinking"),
            "tool_calls": message.get("tool_calls"),
            "model": body.get("model"),
            "created_at": body.get("created_at"),
            "done": body.get("done"),
            "done_reason": body.get("done_reason"),
            "total_duration": body.get("total_duration"),
            "load_duration": body.get("load_duration"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "prompt_eval_duration": body.get("prompt_eval_duration"),
            "eval_count": body.get("eval_count"),
            "eval_duration": body.get("eval_duration"),
            "elapsed_seconds": elapsed,
        }
