import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import grading

RESULTS_DIR = BASE_DIR / "results"
TEXT_DIR = BASE_DIR.parent / "text"
FIGURES_DIR = TEXT_DIR / "figures"
TABLES_DIR = TEXT_DIR / "tabels"

EXPECTED_TASKS = 250
DOMAINS = ("math", "medical")
DOMAIN_TITLES = {"math": "Matematyka", "medical": "Medycyna"}
DOMAIN_LOCATIVE = {"math": "w matematyce", "medical": "w medycynie"}
OUTCOMES = ("correct", "wrong", "abstain", "null")
DASH = "--"

QWEN = "qwen3.5:9b"
GEMMA = "gemma4:12b"


class Config(NamedTuple):
    label: str
    condition: str
    models: tuple
    kind: str


CONFIGS = (
    Config("Q", "solo", (QWEN,), "solo"),
    Config("G", "solo", (GEMMA,), "solo"),
    Config("Q+Q", "debate", (QWEN, QWEN), "homogeneous"),
    Config("Q+G", "debate", (QWEN, GEMMA), "mixed"),
    Config("G+G", "debate", (GEMMA, GEMMA), "homogeneous"),
)
CONFIG_BY_LABEL = {config.label: config for config in CONFIGS}
SOLO_LABELS = ("Q", "G")
DEBATE_CONFIGS = tuple(config for config in CONFIGS if config.condition == "debate")
DEBATE_ROUNDS = (1, 2, 3)

THINKING_MODEL_LABELS = {QWEN: "Q", GEMMA: "G"}
REVISION_PATTERN = re.compile(
    r"\b(Wait|Actually|Hmm|Alternatively|Let me (?:re-?check|double-?check|re-?evaluate|re-?read|re-?think|think about this again|verify)|double-?check|re-?check|re-?evaluate|On second thought)\b",
    re.IGNORECASE,
)
REPEAT_FRAGMENT_CHARS = 250
REPEAT_SCAN_STEP = 50
MIN_REPEATED_LINE_CHARS = 40
CONTEXT_WINDOW_TOKENS = 4096
LONG_PROMPT_TOKENS = CONTEXT_WINDOW_TOKENS // 2
LONG_GENERATION_TOKENS = 1000
REPEATED_SENTENCE_MIN_CHARS = 30
REPEATED_SENTENCE_LIMIT = 3
MARKER_DENSITY_QUANTILE = 0.9
LOOP_MARKER_PATTERN = re.compile(
    r"\b(?:Wait|Actually|Hmm+|Alternatively|Oops|On second thought"
    r"|Let me (?:re-?check|re-?calculate|double-?check|re-?evaluate|re-?read|re-?think|verify|try again)"
    r"|double-?check|re-?check)\b"
    r"|(?:^|[.?!\"}\n]\s*)(?:But|No|However|Or)\b[,.]?",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
DIGIT_RUN = re.compile(r"\d+")
CAUSE_PRIORITY = ("loop", "long_prompt", "other")

PAIR_ROWS = (
    ("Q+Q", "Q"),
    ("G+G", "G"),
    ("Q+G", "best_solo"),
    ("Q+G", "Q+Q"),
    ("Q+G", "G+G"),
    ("G", "Q"),
)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
FIGURE_WIDTH_INCHES = 16 / 2.54

OUTCOME_LABELS = {"correct": "poprawne", "wrong": "błędne", "abstain": "abstain", "null": "null"}
OUTCOME_FILL = {"correct": BLUE, "wrong": "#eb6834", "abstain": "#1baf7a", "null": "#eda100"}
OUTCOME_HATCH = {"correct": None, "wrong": "\\\\\\", "abstain": "///", "null": None}

BAR_POSITIONS = (0.0, 1.0, 2.35, 3.35, 4.35)
BAR_WIDTH = 0.62
SEGMENT_GAP = 0.7
MIN_LABELED_SEGMENT = 7.0

MARKERS = {"solo": "o", "homogeneous": "s", "mixed": "D"}
MARKER_LEGEND = (
    ("solo", "tryb jednomodelowy"),
    ("homogeneous", "debata jednorodna"),
    ("mixed", "debata mieszana"),
)
TOKEN_AXES = {"math": (0, 5000, 1000), "medical": (0, 2500, 500)}
LABEL_PLACEMENTS = {
    "above": ((0, 7), "center", "bottom"),
    "below": ((0, -7), "center", "top"),
    "right": ((8, 0), "left", "center"),
    "left": ((-8, 0), "right", "center"),
}
LABEL_PLACEMENT_OVERRIDES = {("math", "Q"): "right", ("medical", "G"): "left"}
TIME_AXES = {"math": (0, 140, 20), "medical": (0, 40, 10)}
TIME_LABEL_PLACEMENT_OVERRIDES = {}


def darken(color, amount):
    channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    return "#" + "".join(f"{round(channel * (1 - amount)):02x}" for channel in channels)


OUTCOME_HATCH_INK = {outcome: darken(OUTCOME_FILL[outcome], 0.4) for outcome in OUTCOMES}


def sanitize(model):
    return model.replace(":", "-").replace("/", "-")


def result_path(prefix, domain, config):
    tag = "-".join(sanitize(model) for model in config.models)
    return RESULTS_DIR / f"{prefix}{domain}_{config.condition}_{tag}.jsonl"


def read_jsonl(path):
    decoder = json.JSONDecoder()
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            position = 0
            while position < len(line):
                record, position = decoder.raw_decode(line, position)
                records.append(record)
                while position < len(line) and line[position].isspace():
                    position += 1
    return records


def load_experiment(prefix):
    data = {}
    for domain in DOMAINS:
        for config in CONFIGS:
            path = result_path(prefix, domain, config)
            if path.exists():
                data[(domain, config.label)] = read_jsonl(path)
    return data


def outcome_of(record):
    if record.get("abstained") is True or str(record.get("answer")).upper() == grading.ABSTAIN:
        return "abstain"
    if record.get("answer") is None:
        return "null"
    return "correct" if record.get("correct") is True else "wrong"


def llm_calls(record):
    if record["condition"] == "solo":
        response = (record.get("raw") or {}).get("response")
        return [response] if response else []
    return [
        solution["response"]
        for round_record in record.get("trace", [])
        for solution in round_record["solutions"]
        if solution.get("response")
    ]


def token_count(call):
    return (call.get("prompt_eval_count") or 0) + (call.get("eval_count") or 0)


def summarize(records):
    n = len(records)
    counts = {outcome: 0 for outcome in OUTCOMES}
    for record in records:
        counts[outcome_of(record)] += 1
    tokens = sum(token_count(call) for record in records for call in llm_calls(record))
    return {
        "n": n,
        **counts,
        "mean_time": sum(record["elapsed_seconds"] for record in records) / n,
        "tokens_per_task": tokens / n,
    }


def share(stats, outcome):
    return 100 * stats[outcome] / stats["n"]


def verify_alignment(data):
    for domain in DOMAINS:
        id_sets = {
            frozenset(record["task_id"] for record in data[(domain, config.label)])
            for config in CONFIGS
        }
        if len(id_sets) != 1 or len(next(iter(id_sets))) != EXPECTED_TASKS:
            raise SystemExit(f"Niezgodne zbiory zadań dla zbioru {domain}")


def verify_against_summary(data, prefix):
    with open(RESULTS_DIR / "summary.csv", encoding="utf-8") as handle:
        rows = {
            (row["dataset"], row["condition"], row["models"]): row
            for row in csv.DictReader(handle)
        }
    checked = 0
    for (domain, label), records in data.items():
        config = CONFIG_BY_LABEL[label]
        row = rows[(prefix + domain, config.condition, " + ".join(config.models))]
        stats = summarize(records)
        for field in ("n", *OUTCOMES):
            if stats[field] != int(row[field]):
                raise SystemExit(f"Niezgodność z summary.csv: {prefix}{domain} {label} pole {field}")
        if abs(round(stats["mean_time"], 1) - float(row["avg_time_s"])) > 1e-9:
            raise SystemExit(f"Niezgodność z summary.csv: {prefix}{domain} {label} czas")
        checked += 1
    return checked


def decimal(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value):
    text = decimal(abs(value))
    if value > 0:
        return "$+$" + text
    if value < 0:
        return "$-$" + text
    return text


def p_cell(p_value):
    text = r"$<$0,001" if p_value < 0.001 else decimal(p_value, 3)
    return r"\textbf{%s}" % text if p_value < 0.05 else text


def thousands(value):
    rounded = round(value)
    if rounded >= 1000:
        return f"{rounded // 1000}\\,{rounded % 1000:03d}"
    return str(rounded)


def center_column(width):
    return r">{\centering\arraybackslash}p{%s}" % width


def column_spec(columns):
    return "|" + "|".join(columns) + "|"


def table_row(cells, rule):
    return " & ".join(cells) + r" \\ " + rule


def header_row(cells, rule=r"\hline"):
    return [r"\rowcolor{gray!20}", table_row(cells, rule)]


def render_table(label, title, spec, header, body, description=None, fit_width=False):
    caption = f"\\caption[{title}]{{{title}. {description}}}" if description else f"\\caption{{{title}}}"
    lines = [
        "% !TEX root = ../main.tex",
        r"\begin{table}[H]",
        r"    \centering",
        "    " + caption,
        f"    \\label{{{label}}}",
        r"    \small",
        r"    \setlength{\tabcolsep}{2.5pt}",
        r"    \renewcommand{\arraystretch}{1.35}",
        r"    \sbox0{%",
        f"    \\begin{{tabular}}{{{spec}}}",
        r"        \hline",
    ]
    lines += ["        " + line for line in header + body]
    lines += [r"    \end{tabular}}"]
    if fit_width:
        lines += [r"    \ifdim\wd0>\textwidth\sbox0{\resizebox{\textwidth}{!}{\usebox0}}\fi"]
    lines += [r"    \usebox{0}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_text(path, content):
    path.write_text(content, encoding="utf-8")


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_outcome_table(stats):
    best_correct = {
        domain: max(CONFIGS, key=lambda config: stats[(domain, config.label)]["correct"]).label
        for domain in DOMAINS
    }
    header = header_row(
        [
            r"\multicolumn{2}{|c|}{}",
            r"\multicolumn{4}{c|}{\textbf{Odsetek zadań [\%]}}",
        ],
        r"\cline{3-6}",
    ) + header_row([
        r"\textbf{Zbiór}",
        r"\textbf{Konfiguracja}",
        r"\textbf{Poprawne}",
        r"\textbf{Błędne}",
        r"\textbf{Abstain}",
        r"\textbf{Null}",
    ])
    body = []
    for domain in DOMAINS:
        for index, config in enumerate(CONFIGS):
            s = stats[(domain, config.label)]
            correct = decimal(share(s, "correct"))
            if config.label == best_correct[domain]:
                correct = r"\textbf{%s}" % correct
            cells = [
                DOMAIN_TITLES[domain] if index == 0 else "",
                config.label,
                correct,
                decimal(share(s, "wrong")),
                DASH if config.condition == "solo" else decimal(share(s, "abstain")),
                DASH if config.condition == "debate" else decimal(share(s, "null")),
            ]
            is_last = index == len(CONFIGS) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-6}"))
    spec = column_spec(
        [center_column("2.0cm"), center_column("2.2cm")]
        + [center_column(width) for width in ("1.7cm", "1.45cm", "1.55cm", "1.25cm")]
    )
    title = "Odsetek zadań według kategorii wyniku"
    description = (
        r"$n = 250$ zadań na zbiór; Q -- \texttt{qwen3.5:9b}, G -- \texttt{gemma4:12b}, "
        "Q+Q, Q+G i G+G -- debaty odpowiednich par modeli. Pogrubiono najwyższy odsetek poprawnych."
    )
    return render_table("tab:wyniki_zbiorcze", title, spec, header, body, description)


def build_cost_table(stats, truncation_tasks):
    truncated = {(row["dataset"], row["config"]): row for row in truncation_tasks}
    header = header_row([
        r"\textbf{Zbiór}",
        r"\textbf{Konfiguracja}",
        r"\textbf{Średni czas [s]}",
        r"\textbf{Tokeny na zadanie}",
        r"\textbf{Tokeny w zadaniach z ucięciem [\%]}",
    ])
    body = []
    for domain in DOMAINS:
        for index, config in enumerate(CONFIGS):
            s = stats[(domain, config.label)]
            cells = [
                DOMAIN_TITLES[domain] if index == 0 else "",
                config.label,
                decimal(s["mean_time"]),
                thousands(s["tokens_per_task"]),
                decimal(truncated[(domain, config.label)]["truncated_tasks_tokens_pct"]),
            ]
            is_last = index == len(CONFIGS) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-5}"))
    spec = column_spec(
        [center_column("2.0cm"), center_column("2.2cm"), center_column("2.4cm"), center_column("2.4cm"), center_column("3.2cm")]
    )
    title = "Średni czas rozwiązania i liczba tokenów na zadanie"
    description = (
        r"$n = 250$ zadań na zbiór; oznaczenia konfiguracji jak w tabeli~\ref{tab:wyniki_zbiorcze}. "
        "Tokeny (wejściowe i wygenerowane) zliczono we wszystkich wywołaniach. "
        r"Ostatnia kolumna to udział tokenów zużytych w zadaniach, w których wystąpiło ucięcie generacji "
        r"(sekcja~\ref{sec:niepowodzenia})."
    )
    return render_table("tab:koszt_zbiorczy", title, spec, header, body, description)


def main_table_rows(stats):
    rows = []
    for domain in DOMAINS:
        for config in CONFIGS:
            s = stats[(domain, config.label)]
            rows.append({
                "dataset": domain,
                "config": config.label,
                "n": s["n"],
                **{outcome: s[outcome] for outcome in OUTCOMES},
                **{f"{outcome}_pct": round(share(s, outcome), 1) for outcome in OUTCOMES},
                "mean_time_s": round(s["mean_time"], 1),
                "tokens_per_task": round(s["tokens_per_task"]),
            })
    return rows


def paired_counts(records_a, records_b):
    correct_b = {record["task_id"]: record["correct"] is True for record in records_b}
    only_a = only_b = 0
    for record in records_a:
        a_correct = record["correct"] is True
        b_correct = correct_b[record["task_id"]]
        if a_correct and not b_correct:
            only_a += 1
        elif b_correct and not a_correct:
            only_b += 1
    return only_a, only_b


def exact_mcnemar_p(only_a, only_b):
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_a, only_b) + 1)) * 0.5 ** discordant
    return min(1.0, 2 * tail)


def best_solo_label(stats, domain):
    return max(SOLO_LABELS, key=lambda label: stats[(domain, label)]["correct"])


def paired_results(data, stats):
    results = []
    for first, second in PAIR_ROWS:
        for domain in DOMAINS:
            second_label = best_solo_label(stats, domain) if second == "best_solo" else second
            records_a = data[(domain, first)]
            records_b = data[(domain, second_label)]
            only_a, only_b = paired_counts(records_a, records_b)
            n = stats[(domain, first)]["n"]
            delta = 100 * (stats[(domain, first)]["correct"] - stats[(domain, second_label)]["correct"]) / n
            results.append({
                "row": f"{first} vs {second}",
                "dataset": domain,
                "A": first,
                "B": second_label,
                "delta_pp": round(delta, 1),
                "only_A": only_a,
                "only_B": only_b,
                "p_value": exact_mcnemar_p(only_a, only_b),
            })
    return results


def build_paired_table(results, stats):
    by_key = {(row["row"], row["dataset"]): row for row in results}
    sub_labels = [r"$\Delta$ [pp]", "tylko A", "tylko B", "p"]
    header = header_row(
        [r"\multicolumn{1}{|c|}{}"]
        + [r"\multicolumn{4}{c|}{\textbf{%s}}" % DOMAIN_TITLES[domain] for domain in DOMAINS],
        r"\cline{2-9}",
    ) + header_row([r"\textbf{Porównanie}"] + [r"\textbf{%s}" % label for label in sub_labels] * 2)
    body = []
    for first, second in PAIR_ROWS:
        name = "Q+G vs lepszy solo" if second == "best_solo" else f"{first} vs {second}"
        cells = [name]
        for domain in DOMAINS:
            row = by_key[(f"{first} vs {second}", domain)]
            cells += [
                signed(row["delta_pp"]),
                str(row["only_A"]),
                str(row["only_B"]),
                p_cell(row["p_value"]),
            ]
        body.append(table_row(cells, r"\hline"))
    block = [center_column(width) for width in ("1.3cm", "1.35cm", "1.35cm", "1.4cm")]
    spec = column_spec([center_column("3.2cm")] + block * 2)
    title = "Porównania parami konfiguracji na tych samych zadaniach"
    best = ", ".join(
        f"{best_solo_label(stats, domain)} {DOMAIN_LOCATIVE[domain]}" for domain in DOMAINS
    )
    description = (
        r"A i B -- pierwsza i druga konfiguracja w kolumnie „Porównanie”; $\Delta$ -- różnica dokładności A $-$ B "
        "w punktach procentowych (pp); „tylko A” i „tylko B” -- zadania rozwiązane poprawnie wyłącznie przez daną "
        r"konfigurację; $p$ -- dokładny test McNemara, pogrubiono $p < 0{,}05$. "
        f"Lepszy solo: {best}."
    )
    return render_table("tab:testy_parowane", title, spec, header, body, description, fit_width=True)


def answer_is_correct(answer, gold, domain):
    if answer in (None, ""):
        return False
    return grading.is_correct(answer, gold, domain)


def debate_balance(records, domain):
    balance = {
        "n": len(records),
        "agreed_first_round": 0,
        "agreed_first_round_correct": 0,
        "rescued": 0,
        "rescued_after_no_answer": 0,
        "lost": 0,
        "lost_to_abstain": 0,
        "debated": 0,
        "debated_nobody_correct": 0,
        "debated_someone_correct": 0,
        "debated_someone_correct_kept": 0,
        "debated_someone_correct_other_wrong": 0,
    }
    for record in records:
        first_round = record["trace"][0]["solutions"]
        agent_correct = [
            answer_is_correct(solution["final_answer"], record["gold"], domain)
            for solution in first_round
        ]
        any_no_answer = any(solution["final_answer"] in (None, "") for solution in first_round)
        final_correct = record["correct"] is True
        if record["debate_rounds"] == 0 and not record["abstained"]:
            balance["agreed_first_round"] += 1
            balance["agreed_first_round_correct"] += int(final_correct)
        if not (record["debate_rounds"] == 0 and not record["abstained"]):
            balance["debated"] += 1
            if any(agent_correct):
                balance["debated_someone_correct"] += 1
                balance["debated_someone_correct_kept"] += int(final_correct)
                balance["debated_someone_correct_other_wrong"] += int(not any_no_answer)
            else:
                balance["debated_nobody_correct"] += 1
        if not any(agent_correct) and final_correct:
            balance["rescued"] += 1
            balance["rescued_after_no_answer"] += int(any_no_answer)
        if any(agent_correct) and not final_correct:
            balance["lost"] += 1
            balance["lost_to_abstain"] += int(bool(record["abstained"]))
    return balance


def debate_balance_rows(data):
    rows = []
    for domain in DOMAINS:
        for config in DEBATE_CONFIGS:
            rows.append({
                "dataset": domain,
                "config": config.label,
                **debate_balance(data[(domain, config.label)], domain),
            })
    return rows


def build_balance_table(rows):
    group = lambda first, second: r"\multicolumn{2}{c|}{\shortstack{\textbf{%s}\\\textbf{%s}}}" % (first, second)
    sub = lambda label: r"\textbf{%s}" % label
    header = header_row(
        [
            r"\multicolumn{2}{|c|}{}",
            group("Zgoda", "od razu"),
            group("Żaden model", "poprawny"),
            group("Jeden model", "poprawny"),
        ],
        r"\cline{3-8}",
    ) + header_row(
        [sub("Zbiór"), sub("Debata"), sub("razem"), sub("w tym poprawna"), sub("razem"),
         sub("w tym uratowane"), sub("razem"), sub("w tym utracone")]
    )
    body = []
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["dataset"] == domain]
        for index, row in enumerate(domain_rows):
            cells = [
                DOMAIN_TITLES[domain] if index == 0 else "",
                row["config"],
                str(row["agreed_first_round"]),
                str(row["agreed_first_round_correct"]),
                str(row["debated_nobody_correct"]),
                str(row["rescued"]),
                str(row["debated_someone_correct"]),
                str(row["lost"]),
            ]
            assert row["agreed_first_round"] + row["debated_nobody_correct"] + row["debated_someone_correct"] == row["n"]
            is_last = index == len(domain_rows) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-8}"))
    widths = ("1.3cm", "1.9cm", "1.3cm", "1.9cm", "1.3cm", "1.9cm")
    spec = column_spec(
        [center_column("2.0cm"), center_column("1.4cm")] + [center_column(width) for width in widths]
    )
    title = "Bilans debaty względem rundy niezależnej"
    description = (
        "Liczba zadań, dla każdej debaty $n = 250$. „Zgoda od razu” -- modele zgodziły się w rundzie niezależnej, "
        "więc debata się nie odbyła. „Uratowane” -- zadania z poprawnym wynikiem końcowym, gdy w rundzie niezależnej "
        "nikt nie miał poprawnej odpowiedzi; „utracone” -- zadania z niepoprawnym wynikiem końcowym (także abstain), "
        "gdy poprawną odpowiedź miał jeden z modeli."
    )
    return render_table("tab:bilans_debaty", title, spec, header, body, description)


def is_truncated(call):
    return call.get("done_reason") == "length"


def truncated_pct(calls):
    return round(100 * sum(is_truncated(call) for call in calls) / len(calls), 1)


def truncation_rows(data):
    rows = []
    for domain in DOMAINS:
        for config in CONFIGS:
            records = data[(domain, config.label)]
            calls = [call for record in records for call in llm_calls(record)]
            truncated = [call for call in calls if is_truncated(call)]
            row = dict.fromkeys((
                "dataset", "config", "calls", "truncated_calls", "truncated_calls_pct",
                "independent_calls", "independent_truncated_pct",
                "debate_calls", "debate_truncated_pct", "truncated_tokens_pct",
            ), "")
            row.update(
                dataset=domain, config=config.label, calls=len(calls), truncated_calls=len(truncated),
                truncated_calls_pct=truncated_pct(calls),
                truncated_tokens_pct=round(
                    100 * sum(token_count(call) for call in truncated) / sum(token_count(call) for call in calls), 1
                ),
            )
            if config.condition == "debate":
                independent = [call for record in records for call in round_calls(record, 0)]
                debate = [call for record in records for call in debate_phase_calls(record)]
                assert len(independent) + len(debate) == len(calls)
                row.update(
                    independent_calls=len(independent), independent_truncated_pct=truncated_pct(independent),
                    debate_calls=len(debate), debate_truncated_pct=truncated_pct(debate),
                )
            rows.append(row)
    return rows


def build_truncation_table(rows):
    sub = lambda label: r"\textbf{%s}" % label
    header = header_row(
        [
            r"\multicolumn{2}{|c|}{}",
            r"\multicolumn{2}{c|}{\textbf{Zadania z ucięciem}}",
            r"\multicolumn{3}{c|}{\textbf{w tym z powodu}}",
        ],
        r"\cline{3-7}",
    ) + header_row([
        sub("Zbiór"), sub("Konfiguracja"), sub("liczba"), sub(r"[\%]"),
        sub("zapętlenia"), sub("długiego promptu"), sub("długiego rozumowania"),
    ])
    body = []
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["dataset"] == domain]
        for index, row in enumerate(domain_rows):
            assert row["truncated_tasks"] == row["loop_tasks"] + row["long_prompt_tasks"] + row["other_tasks"]
            cells = [
                DOMAIN_TITLES[domain] if index == 0 else "",
                row["config"],
                str(row["truncated_tasks"]),
                decimal(row["truncated_tasks_pct"]),
                str(row["loop_tasks"]),
                str(row["long_prompt_tasks"]),
                str(row["other_tasks"]),
            ]
            is_last = index == len(domain_rows) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-7}"))
    spec = column_spec(
        [center_column(width) for width in ("1.9cm", "2.1cm", "1.4cm", "1.3cm", "2.0cm", "2.3cm", "2.9cm")]
    )
    title = "Zadania z uciętą generacją według przyczyny"
    description = (
        "$n = 250$ zadań na konfigurację. Zadanie liczono, gdy co najmniej jedno jego wywołanie zostało ucięte po "
        "wyczerpaniu okna kontekstu. Przyczyny są rozłączne i sprawdzane w kolejności kolumn: zapętlenie (powtórzenia "
        "lub nadmiar słów-poprawek), długi prompt (co najmniej połowa okna) i długie rozumowanie bez powtórzeń."
    )
    return render_table("tab:uciete_generacje", title, spec, header, body, description)


def verify_context_window(data):
    truncated = [call for records in data.values() for record in records for call in llm_calls(record) if is_truncated(call)]
    assert truncated and all(
        (call.get("prompt_eval_count") or 0) + (call.get("eval_count") or 0) == CONTEXT_WINDOW_TOKENS
        for call in truncated
    )
    return len(truncated)


def repeated_sentence_count(text):
    sentences = (part.strip() for part in SENTENCE_SPLIT.split(text))
    counts = Counter(sentence for sentence in sentences if len(sentence) >= REPEATED_SENTENCE_MIN_CHARS)
    return max(counts.values()) if counts else 0


def marker_density(call, text_key="content"):
    tokens = call.get("eval_count") or 0
    if tokens < LONG_GENERATION_TOKENS:
        return 0.0
    return 1000 * len(LOOP_MARKER_PATTERN.findall(call.get(text_key) or "")) / tokens


def marker_density_limit(data):
    densities = sorted(
        marker_density(call)
        for records in data.values()
        for record in records
        for call in llm_calls(record)
        if not is_truncated(call) and (call.get("eval_count") or 0) >= LONG_GENERATION_TOKENS
    )
    return densities[int(MARKER_DENSITY_QUANTILE * len(densities))]


def truncation_cause(call, density_limit, text_key="content"):
    text = call.get(text_key) or ""
    if (
        has_repeated_fragment(DIGIT_RUN.sub("#", text))
        or repeated_sentence_count(text) >= REPEATED_SENTENCE_LIMIT
        or marker_density(call, text_key) > density_limit
    ):
        return "loop"
    if (call.get("prompt_eval_count") or 0) >= LONG_PROMPT_TOKENS:
        return "long_prompt"
    return "other"


def task_truncation_cause(record, density_limit, text_key="content"):
    causes = {truncation_cause(call, density_limit, text_key) for call in llm_calls(record) if is_truncated(call)}
    return next((cause for cause in CAUSE_PRIORITY if cause in causes), None)


def truncated_task_rows(data):
    density_limit = marker_density_limit(data)
    rows = []
    for domain in DOMAINS:
        for config in CONFIGS:
            records = data[(domain, config.label)]
            causes = Counter()
            tokens = seconds = truncated_tokens = truncated_seconds = 0
            for record in records:
                cause = task_truncation_cause(record, density_limit)
                task_tokens = sum(token_count(call) for call in llm_calls(record))
                tokens += task_tokens
                seconds += record["elapsed_seconds"]
                if cause:
                    causes[cause] += 1
                    truncated_tokens += task_tokens
                    truncated_seconds += record["elapsed_seconds"]
            truncated = sum(causes.values())
            rows.append({
                "dataset": domain,
                "config": config.label,
                "tasks": len(records),
                "truncated_tasks": truncated,
                "truncated_tasks_pct": round(100 * truncated / len(records), 1),
                "loop_tasks": causes["loop"],
                "long_prompt_tasks": causes["long_prompt"],
                "other_tasks": causes["other"],
                "truncated_tasks_tokens_pct": round(100 * truncated_tokens / tokens, 1),
                "truncated_tasks_time_pct": round(100 * truncated_seconds / seconds, 1),
                "marker_density_limit_per_1k_tokens": round(density_limit, 1),
            })
    return rows


TRUNCATION_MODES = (("main", "content"), ("reasoning", "thinking"))
TRUNCATION_GROUPS = (
    ("all", lambda label: True),
    ("with_G", lambda label: "G" in label),
    ("only_Q", lambda label: "G" not in label),
)


def mode_truncation_rows(main_data, pilot_data):
    density_limit = marker_density_limit(main_data)
    rows = []
    for mode, text_key in TRUNCATION_MODES:
        data = main_data if mode == "main" else pilot_data
        for group, belongs in TRUNCATION_GROUPS:
            records = [
                record
                for (_, label), config_records in data.items()
                if belongs(label)
                for record in config_records
            ]
            causes = Counter()
            first_feature = no_answer = 0
            for record in records:
                truncated = [call for call in llm_calls(record) if is_truncated(call)]
                if not truncated:
                    continue
                causes[task_truncation_cause(record, density_limit, text_key)] += 1
                first_feature += any(
                    has_repeated_fragment(DIGIT_RUN.sub("#", call.get(text_key) or "")) for call in truncated
                )
                no_answer += all(not (call.get("content") or "").strip() for call in truncated)
            truncated_tasks = sum(causes.values())
            rows.append({
                "mode": mode,
                "configs": group,
                "tasks": len(records),
                "truncated_tasks": truncated_tasks,
                "loop_tasks": causes["loop"],
                "loop_tasks_pct": round(100 * causes["loop"] / truncated_tasks, 1),
                "long_prompt_tasks": causes["long_prompt"],
                "other_tasks": causes["other"],
                "repeated_fragment_tasks": first_feature,
                "repeated_fragment_tasks_pct": round(100 * first_feature / truncated_tasks, 1),
                "no_answer_tasks": no_answer,
                "marker_density_limit_per_1k_tokens": round(density_limit, 1),
            })
    return rows


def solo_reference_label(domain, label, stats):
    if label == "Q+Q":
        return "Q"
    if label == "G+G":
        return "G"
    return max(("Q", "G"), key=lambda solo: stats[(domain, solo)]["correct"])


def truncation_detail_rows(data, stats):
    density_limit = marker_density_limit(data)
    rows = []

    def add(item, dataset, metric, value):
        rows.append({"item": item, "dataset": dataset, "metric": metric, "value": value})

    prompts = [
        call["prompt_eval_count"]
        for records in data.values()
        for record in records
        if task_truncation_cause(record, density_limit) == "long_prompt"
        for call in llm_calls(record)
        if is_truncated(call) and (call.get("prompt_eval_count") or 0) >= LONG_PROMPT_TOKENS
    ]
    add("long_prompt", "all", "calls", len(prompts))
    add("long_prompt", "all", "mean_prompt_tokens", round(sum(prompts) / len(prompts)))
    add("long_prompt", "all", "mean_room_tokens", round(sum(CONTEXT_WINDOW_TOKENS - prompt for prompt in prompts) / len(prompts)))

    for domain in DOMAINS:
        labels_by_task = {}
        for config in CONFIGS:
            for record in data[(domain, config.label)]:
                if task_truncation_cause(record, density_limit) == "other":
                    labels_by_task.setdefault(record["task_id"], set()).add(config.label)
        runs = []
        for config in CONFIGS:
            for record in data[(domain, config.label)]:
                if record["task_id"] in labels_by_task and config.label not in labels_by_task[record["task_id"]]:
                    runs.append(record)
        finished = [record for record in runs if not any(is_truncated(call) for call in llm_calls(record))]
        add("long_reasoning", domain, "tasks", len(labels_by_task))
        add("long_reasoning", domain, "runs_in_other_configs", len(runs))
        add("long_reasoning", domain, "runs_without_truncation", len(finished))
        add("long_reasoning", domain, "runs_without_truncation_correct", sum(outcome_of(record) == "correct" for record in finished))

    for domain in DOMAINS:
        records = data[(domain, "Q+G")]
        by_round = {}
        for record in records:
            for round_record in record["trace"]:
                for solution in round_record["solutions"]:
                    call = solution.get("response")
                    if call:
                        by_round.setdefault(round_record["round"], []).append(call)
        for round_index in sorted(by_round):
            calls = by_round[round_index]
            add(f"mixed_round_{round_index}", domain, "calls", len(calls))
            add(f"mixed_round_{round_index}", domain, "mean_prompt_tokens", round(sum(call.get("prompt_eval_count") or 0 for call in calls) / len(calls)))
            add(f"mixed_round_{round_index}", domain, "mean_generated_tokens", round(sum(call.get("eval_count") or 0 for call in calls) / len(calls)))
        loading = sum((call.get("load_duration") or 0) for record in records for call in llm_calls(record)) / 1e9
        add("mixed_model_loading", domain, "mean_seconds_per_task", round(loading / len(records), 1))

    for domain in DOMAINS:
        abstained = truncated = solo_null = solo_correct = 0
        for config in CONFIGS:
            if config.condition == "solo":
                continue
            solo = {record["task_id"]: record for record in data[(domain, solo_reference_label(domain, config.label, stats))]}
            for record in data[(domain, config.label)]:
                if outcome_of(record) != "abstain":
                    continue
                abstained += 1
                truncated += any(is_truncated(call) for call in llm_calls(record))
                solo_outcome = outcome_of(solo[record["task_id"]])
                solo_null += solo_outcome == "null"
                solo_correct += solo_outcome == "correct"
        add("abstain", domain, "tasks", abstained)
        add("abstain", domain, "tasks_with_truncated_call", truncated)
        add("abstain", domain, "tasks_solo_reference_null", solo_null)
        add("abstain", domain, "tasks_solo_reference_correct", solo_correct)
    return rows


COST_COLUMNS = (
    "dataset", "config", "part", "item", "tasks", "tokens_per_task", "time_s_per_task",
    "tokens_share_pct", "time_share_pct", "base_config", "tokens_vs_base", "time_vs_base",
    "accuracy_pct", "truncated_calls", "truncated_tokens_pct",
    "ended_tasks", "ended_correct_pct", "ended_abstain_pct",
)


def cost_row(dataset, config, part, item, **values):
    row = dict.fromkeys(COST_COLUMNS, "")
    row.update(dataset=dataset, config=config, part=part, item=item, **values)
    return row


def call_seconds(call):
    return call.get("elapsed_seconds") or 0


def sum_cost(calls):
    return sum(token_count(call) for call in calls), sum(call_seconds(call) for call in calls)


def round_calls(record, round_index):
    return [
        solution["response"]
        for round_record in record.get("trace", [])
        if round_record["round"] == round_index
        for solution in round_record["solutions"]
        if solution.get("response")
    ]


def debate_phase_calls(record):
    return [call for round_index in DEBATE_ROUNDS for call in round_calls(record, round_index)]


def model_calls(record, model):
    return [
        solution["response"]
        for round_record in record["trace"]
        for solution in round_record["solutions"]
        if solution.get("response") and solution["model"] == model
    ]


def cost_part_row(dataset, config, part, item, per_task_calls, total_tokens, total_seconds, **values):
    executed = [calls for calls in per_task_calls if calls]
    tokens, seconds = sum_cost([call for calls in executed for call in calls])
    count = len(executed)
    return cost_row(
        dataset, config, part, item,
        tasks=count,
        tokens_per_task=round(tokens / count) if count else "",
        time_s_per_task=round(seconds / count, 1) if count else "",
        tokens_share_pct=round(100 * tokens / total_tokens, 1),
        time_share_pct=round(100 * seconds / total_seconds, 1),
        **values,
    )


def ended_in_round(records, round_index):
    ended = [record for record in records if record["debate_rounds"] == round_index]
    if not ended:
        return {"ended_tasks": 0, "ended_correct_pct": "", "ended_abstain_pct": ""}
    return {
        "ended_tasks": len(ended),
        "ended_correct_pct": round(100 * sum(record["correct"] is True for record in ended) / len(ended), 2),
        "ended_abstain_pct": round(100 * sum(bool(record["abstained"]) for record in ended) / len(ended), 2),
    }


def cost_breakdown_rows(data, stats):
    rows = []
    for domain in DOMAINS:
        best = best_solo_label(stats, domain)
        for config in CONFIGS:
            records = data[(domain, config.label)]
            s = stats[(domain, config.label)]
            calls = [llm_calls(record) for record in records]
            all_calls = [call for per_task in calls for call in per_task]
            tokens, seconds = sum_cost(all_calls)
            truncated = [call for call in all_calls if call.get("done_reason") == "length"]
            base = {"Q+Q": "Q", "G+G": "G", "Q+G": best}.get(config.label, "")
            row = cost_row(
                domain, config.label, "config", "total",
                tasks=s["n"],
                tokens_per_task=round(s["tokens_per_task"]),
                time_s_per_task=round(s["mean_time"], 1),
                base_config=base,
                accuracy_pct=round(share(s, "correct"), 1),
                truncated_calls=len(truncated),
                truncated_tokens_pct=round(100 * sum(token_count(call) for call in truncated) / tokens, 1),
            )
            if base:
                row["tokens_vs_base"] = round(s["tokens_per_task"] / stats[(domain, base)]["tokens_per_task"], 2)
                row["time_vs_base"] = round(s["mean_time"] / stats[(domain, base)]["mean_time"], 2)
            rows.append(row)
            if config.condition == "solo":
                for outcome in ("correct", "wrong"):
                    group = [per_task for record, per_task in zip(records, calls) if outcome_of(record) == outcome]
                    rows.append(cost_part_row(domain, config.label, "outcome", outcome, group, tokens, seconds))
                continue
            rows.append(cost_part_row(
                domain, config.label, "phase", "independent",
                [round_calls(record, 0) for record in records], tokens, seconds,
            ))
            rows.append(cost_part_row(
                domain, config.label, "phase", "debate",
                [debate_phase_calls(record) for record in records], tokens, seconds,
            ))
            for round_index in DEBATE_ROUNDS:
                rows.append(cost_part_row(
                    domain, config.label, "round", f"round_{round_index}",
                    [round_calls(record, round_index) for record in records], tokens, seconds,
                    **ended_in_round(records, round_index),
                ))
            if config.kind == "mixed":
                for label, model in (("Q", QWEN), ("G", GEMMA)):
                    rows.append(cost_part_row(
                        domain, config.label, "model", label,
                        [model_calls(record, model) for record in records], tokens, seconds,
                    ))
    return rows


def pilot_rows(main_data, pilot_data):
    rows = []
    for domain in DOMAINS:
        for config in CONFIGS:
            key = (domain, config.label)
            if key not in pilot_data:
                continue
            thinking_records = pilot_data[key]
            task_ids = {record["task_id"] for record in thinking_records}
            baseline_records = [record for record in main_data[key] if record["task_id"] in task_ids]
            if len(baseline_records) != len(thinking_records):
                raise SystemExit(f"Brak zadań pilotażu w głównym przebiegu: {domain} {config.label}")
            baseline = summarize(baseline_records)
            thinking = summarize(thinking_records)
            rows.append({
                "dataset": domain,
                "config": config.label,
                "n": thinking["n"],
                "off_correct": baseline["correct"],
                "on_correct": thinking["correct"],
                "off_wrong": baseline["wrong"],
                "on_wrong": thinking["wrong"],
                "off_no_answer": baseline["abstain"] + baseline["null"],
                "on_no_answer": thinking["abstain"] + thinking["null"],
                "off_time_s": round(baseline["mean_time"], 1),
                "on_time_s": round(thinking["mean_time"], 1),
                "off_tokens_per_task": round(baseline["tokens_per_task"]),
                "on_tokens_per_task": round(thinking["tokens_per_task"]),
            })
    return rows


def build_pilot_table(rows):
    arrow = r" $\rightarrow$ "
    sub = lambda label: r"\textbf{%s}" % label
    header = header_row(
        [r"\multicolumn{3}{|c|}{}", r"\multicolumn{5}{c|}{\textbf{bez rozumowania%sz rozumowaniem}}" % arrow],
        r"\cline{4-8}",
    ) + header_row([
        sub("Zbiór"),
        sub("Konfiguracja"),
        sub("Liczba zadań"),
        sub("Poprawne"),
        sub("Błędne"),
        sub("Brak odpowiedzi"),
        sub("Średni czas na zadanie [s]"),
        sub("Tokeny na zadanie"),
    ])
    body = []
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["dataset"] == domain]
        for index, row in enumerate(domain_rows):
            cells = [
                DOMAIN_TITLES[domain] if index == 0 else "",
                row["config"],
                str(row["n"]),
                f"{row['off_correct']}{arrow}{row['on_correct']}",
                f"{row['off_wrong']}{arrow}{row['on_wrong']}",
                f"{row['off_no_answer']}{arrow}{row['on_no_answer']}",
                f"{decimal(row['off_time_s'])}{arrow}{decimal(row['on_time_s'])}",
                f"{thousands(row['off_tokens_per_task'])}{arrow}{thousands(row['on_tokens_per_task'])}",
            ]
            is_last = index == len(domain_rows) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-8}"))
    spec = column_spec(
        [center_column("2.0cm"), center_column("2.25cm"), center_column("1.2cm")]
        + [center_column("1.7cm"), center_column("1.4cm"), center_column("1.9cm"), center_column("2.35cm"), center_column("2.7cm")]
    )
    title = "Tryb bez rozumowania i z rozumowaniem na tych samych zadaniach"
    description = (
        "„Brak odpowiedzi” to null w trybie solo (generacja ucięta po wyczerpaniu okna kontekstu) i abstain w debacie "
        "(brak konsensusu). Poprawne, błędne i brak odpowiedzi podano w liczbach zadań, a tokeny obejmują prompt "
        "i wygenerowaną odpowiedź."
    )
    return render_table("tab:pilotaz_rozumowanie", title, spec, header, body, description, fit_width=True)


def has_repeated_fragment(text):
    last_start = max(0, len(text) - 2 * REPEAT_FRAGMENT_CHARS)
    for start in range(0, last_start, REPEAT_SCAN_STEP):
        fragment = text[start:start + REPEAT_FRAGMENT_CHARS]
        if text.find(fragment, start + REPEAT_FRAGMENT_CHARS) != -1:
            return True
    return False


def max_line_repeat(text):
    lines = (line.strip() for line in text.split("\n"))
    counts = Counter(line for line in lines if len(line) >= MIN_REPEATED_LINE_CHARS)
    return max(counts.values()) if counts else 0


def new_thinking_group():
    return {
        "calls": 0,
        "empty_content": 0,
        "eval_tokens": 0,
        "repeated_fragment": 0,
        "revision_words": 0,
        "max_line_repeat": 0,
        "max_line_repeat_task": "",
    }


def add_thinking_call(group, call, task_id):
    text = call.get("thinking") or ""
    group["calls"] += 1
    group["empty_content"] += not (call.get("content") or "").strip()
    group["eval_tokens"] += call.get("eval_count") or 0
    group["repeated_fragment"] += has_repeated_fragment(text)
    group["revision_words"] += len(REVISION_PATTERN.findall(text))
    repeat = max_line_repeat(text)
    if repeat > group["max_line_repeat"]:
        group["max_line_repeat"] = repeat
        group["max_line_repeat_task"] = task_id


def thinking_truncation_rows(pilot_data):
    groups = {}
    for (domain, _), records in pilot_data.items():
        for record in records:
            for call in llm_calls(record):
                model = THINKING_MODEL_LABELS[call["model"]]
                status = "truncated" if call.get("done_reason") == "length" else "finished"
                for scope in (domain, "all"):
                    group = groups.setdefault((scope, model, status), new_thinking_group())
                    add_thinking_call(group, call, record["task_id"])
    rows = []
    for scope in (*DOMAINS, "all"):
        for model in THINKING_MODEL_LABELS.values():
            for status in ("truncated", "finished"):
                group = groups.get((scope, model, status))
                if group is None:
                    continue
                calls = group["calls"]
                rows.append({
                    "dataset": scope,
                    "model": model,
                    "status": status,
                    "calls": calls,
                    "empty_content": group["empty_content"],
                    "mean_generated_tokens": round(group["eval_tokens"] / calls),
                    "repeated_fragment_calls": group["repeated_fragment"],
                    "repeated_fragment_pct": round(100 * group["repeated_fragment"] / calls, 1),
                    "revision_words": group["revision_words"],
                    "revision_words_per_1k_tokens": round(1000 * group["revision_words"] / group["eval_tokens"], 2),
                    "max_line_repeat": group["max_line_repeat"],
                    "max_line_repeat_task": group["max_line_repeat_task"],
                })
    return rows


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.unicode_minus": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "hatch.linewidth": 0.7,
    })


def style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0, labelsize=8.5)
    ax.set_axisbelow(True)


def draw_segment(ax, x, bottom, value, outcome):
    top = bottom + value
    lower = bottom + SEGMENT_GAP / 2 if bottom > 0 else bottom
    upper = top - SEGMENT_GAP / 2 if top < 100 - 1e-9 else top
    if upper <= lower:
        return
    hatch = OUTCOME_HATCH[outcome]
    ax.bar(
        x, upper - lower, bottom=lower, width=BAR_WIDTH,
        color=OUTCOME_FILL[outcome],
        edgecolor=OUTCOME_HATCH_INK[outcome] if hatch else OUTCOME_FILL[outcome],
        linewidth=0, hatch=hatch, zorder=3,
    )
    if outcome in ("correct", "wrong") and value >= MIN_LABELED_SEGMENT:
        ax.text(x, (lower + upper) / 2, decimal(value), ha="center", va="center",
                fontsize=8, color="#ffffff", zorder=4)


def annotate_group(ax, text, start, end):
    transform = ax.get_xaxis_transform()
    ax.plot([start, end], [-0.15, -0.15], transform=transform, color=AXIS, linewidth=0.8, clip_on=False)
    ax.text((start + end) / 2, -0.18, text, transform=transform, ha="center", va="top",
            fontsize=8.5, color=INK_SECONDARY)


def plot_outcome_composition(stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 3.8), sharey=True)
    half_width = BAR_WIDTH / 2 + 0.05
    for ax, domain in zip(axes, DOMAINS):
        for config, x in zip(CONFIGS, BAR_POSITIONS):
            s = stats[(domain, config.label)]
            bottom = 0.0
            for outcome in OUTCOMES:
                value = share(s, outcome)
                if value > 0:
                    draw_segment(ax, x, bottom, value, outcome)
                bottom += value
        style_axes(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
        ax.set_xlim(BAR_POSITIONS[0] - 0.55, BAR_POSITIONS[-1] + 0.55)
        ax.set_xticks(BAR_POSITIONS)
        ax.set_xticklabels([config.label for config in CONFIGS], fontsize=9, color=INK)
        ax.set_title(DOMAIN_TITLES[domain], loc="left", fontsize=10, fontweight="bold", color=INK, pad=6)
        annotate_group(ax, "tryb jednomodelowy", BAR_POSITIONS[0] - half_width, BAR_POSITIONS[1] + half_width)
        annotate_group(ax, "debata", BAR_POSITIONS[2] - half_width, BAR_POSITIONS[4] + half_width)
    axes[0].set_ylabel("Udział zadań [%]", fontsize=9)
    axes[1].tick_params(labelleft=False)
    handles = [
        Patch(facecolor=OUTCOME_FILL[outcome],
              edgecolor=OUTCOME_HATCH_INK[outcome] if OUTCOME_HATCH[outcome] else OUTCOME_FILL[outcome],
              hatch=OUTCOME_HATCH[outcome], linewidth=0, label=OUTCOME_LABELS[outcome])
        for outcome in OUTCOMES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               handlelength=1.8, handleheight=1.1, columnspacing=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.27, wspace=0.05)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def has_close_point_above(point, points):
    tokens, accuracy, _ = point
    return any(
        0 < other_accuracy - accuracy < 4.5 and abs(math.log(other_tokens / tokens)) < 0.45
        for other_tokens, other_accuracy, _ in points
    )


def plot_accuracy_vs_cost(stats, path, metric="tokens_per_task", axes_limits=TOKEN_AXES,
                          xlabel="Tokeny na zadanie", overrides=LABEL_PLACEMENT_OVERRIDES):
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 3.5), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        points = [
            (stats[(domain, config.label)][metric], share(stats[(domain, config.label)], "correct"), config)
            for config in CONFIGS
        ]
        for point in points:
            tokens, accuracy, config = point
            ax.plot(tokens, accuracy, linestyle="none", marker=MARKERS[config.kind], markersize=6.5,
                    markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            placement = overrides.get(
                (domain, config.label), "below" if has_close_point_above(point, points) else "above"
            )
            offset, horizontal, vertical = LABEL_PLACEMENTS[placement]
            ax.annotate(config.label, (tokens, accuracy), xytext=offset, textcoords="offset points",
                        ha=horizontal, va=vertical, fontsize=8.5, color=INK)
        style_axes(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.6, linestyle="-")
        low, high, step = axes_limits[domain]
        ticks = range(low, high, step)
        ax.set_xlim(low, high)
        ax.set_ylim(55, 90)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:,}".replace(",", " ") for tick in ticks])
        ax.minorticks_off()
        ax.set_title(DOMAIN_TITLES[domain], loc="left", fontsize=10, fontweight="bold", color=INK, pad=6)
        ax.set_xlabel(xlabel, fontsize=9)
    axes[0].set_ylabel("Poprawne [%]", fontsize=9)
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D([], [], linestyle="none", marker=MARKERS[kind], markersize=6.5, markerfacecolor=BLUE,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        for kind, label in MARKER_LEGEND
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               handlelength=1.8, columnspacing=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.25, wspace=0.05)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    main_data = load_experiment("")
    pilot_data = load_experiment("thinking_")
    verify_alignment(main_data)
    checked = verify_against_summary(main_data, "") + verify_against_summary(pilot_data, "thinking_")
    print(f"Zgodność z summary.csv: {checked} konfiguracji OK")

    stats = {key: summarize(records) for key, records in main_data.items()}
    print(f"Ucięte wywołania z oknem {CONTEXT_WINDOW_TOKENS} tokenów: {verify_context_window(main_data)}")
    truncation_tasks = truncated_task_rows(main_data)

    write_csv(RESULTS_DIR / "main_table.csv", main_table_rows(stats))
    write_text(TABLES_DIR / "wyniki-zbiorcze.tex", build_outcome_table(stats))
    write_text(TABLES_DIR / "koszt-zbiorczy.tex", build_cost_table(stats, truncation_tasks))

    paired = paired_results(main_data, stats)
    write_csv(RESULTS_DIR / "paired_tests.csv", paired)
    write_text(TABLES_DIR / "testy-parowane.tex", build_paired_table(paired, stats))

    balance = debate_balance_rows(main_data)
    write_csv(RESULTS_DIR / "debate_balance.csv", balance)
    write_text(TABLES_DIR / "bilans-debaty.tex", build_balance_table(balance))

    truncation = truncation_rows(main_data)
    write_csv(RESULTS_DIR / "truncation.csv", truncation)
    write_csv(RESULTS_DIR / "truncation_tasks.csv", truncation_tasks)
    write_csv(RESULTS_DIR / "truncation_details.csv", truncation_detail_rows(main_data, stats))
    write_text(TABLES_DIR / "uciete-generacje.tex", build_truncation_table(truncation_tasks))

    cost = cost_breakdown_rows(main_data, stats)
    write_csv(RESULTS_DIR / "cost_breakdown.csv", cost)

    pilot = pilot_rows(main_data, pilot_data)
    write_csv(RESULTS_DIR / "pilot_thinking.csv", pilot)
    write_csv(RESULTS_DIR / "thinking_truncation.csv", thinking_truncation_rows(pilot_data))
    write_csv(RESULTS_DIR / "truncation_modes.csv", mode_truncation_rows(main_data, pilot_data))
    write_text(TABLES_DIR / "pilotaz-rozumowanie.tex", build_pilot_table(pilot))

    plot_outcome_composition(stats, FIGURES_DIR / "rys-sklad-wynikow.png")
    plot_accuracy_vs_cost(stats, FIGURES_DIR / "rys-dokladnosc-koszt.png")
    plot_accuracy_vs_cost(
        stats, FIGURES_DIR / "rys-dokladnosc-czas.png", metric="mean_time", axes_limits=TIME_AXES,
        xlabel="Średni czas na zadanie [s]", overrides=TIME_LABEL_PLACEMENT_OVERRIDES,
    )
    print("Zapisano 6 tabel (text/tabels), 3 rysunki (text/figures) i 10 plików csv (code/results)")


if __name__ == "__main__":
    main()
