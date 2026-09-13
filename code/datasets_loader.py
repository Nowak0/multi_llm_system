"""Dataset loading into a single uniform record shape.

Every question - math or medical - becomes:

    {
        "id":       str,          # stable per-question id (for the paired design)
        "domain":   "math" | "medical",
        "question": str,          # the full prompt
        "choices":  dict | None,  # {"A": "...", ...} for MCQ, None for math
        "gold":     str,          # gold answer: exact expression (math) / letter (MCQ)
    }

Datasets used:
  - Math:    HuggingFaceH4/MATH-500  (short/numeric answers, standard grader)
  - Medical: openlifescienceai/medmcqa  (4-option MCQ, exact-match on letter)
Both pull from the HuggingFace Hub.
"""

from grading import extract_letter
from utils import log

_LETTERS = ["A", "B", "C", "D", "E"]
CONSOLE_LOGS = True

def _load_hf(path: str, split: str):
    """Thin wrapper so the (heavy) import only happens when a loader runs."""
    from datasets import load_dataset
    return load_dataset(path, split=split)


def load_math(limit: int = 100, split: str = "test") -> list[dict]:
    """MATH-500: free-form math with a short exact gold answer."""
    ds = _load_hf("HuggingFaceH4/MATH-500", split)
    records = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        records.append({
            "id": f"math-{row.get('unique_id', i)}",
            "domain": "math",
            "question": row["problem"],
            "choices": None,
            "gold": str(row["answer"]).strip(),
        })
    return records


def load_medical(limit: int = 100, split: str = "validation") -> list[dict]:
    ds = _load_hf("openlifescienceai/medmcqa", split)
    records = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        choices = {_LETTERS[j]: str(opt) for j, opt in enumerate(options)}
        options_block = "\n".join(f"{letter}. {text}" for letter, text in choices.items())
        question = f"{row['question']}\n\n{options_block}"
        gold_letter = _LETTERS[int(row["cop"])]
        records.append({
            "id": f"med-{row.get('id', i)}",
            "domain": "medical",
            "question": question,
            "choices": choices,
            "gold": gold_letter,
        })
    return records


def load_records(domain: str, limit: int = 100) -> list[dict]:
    """Unified loader. domain in {"math", "medical"}."""
    if domain == "math":
        return load_math(limit=limit)
    if domain == "medical":
        return load_medical(limit=limit)
    raise ValueError(f"Unknown domain: {domain!r}")


if __name__ == "__main__":
    for domain in ("math", "medical"):
        recs = load_records(domain, limit=2)
        log(CONSOLE_LOGS,f"\n=== {domain}: {len(recs)} records ===")
        for r in recs:
            log(CONSOLE_LOGS,r["id"], "| gold:", r["gold"])
            log(CONSOLE_LOGS,r["question"][:200], "...")
            if r["domain"] == "medical":
                assert extract_letter(r["gold"]) == r["gold"]
