import csv
import json
import math
import sys
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
MUTED = "#898781"
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
TOKEN_TICKS = (500, 1000, 2000, 5000)
LABEL_PLACEMENTS = {
    "above": ((0, 7), "center", "bottom"),
    "below": ((0, -7), "center", "top"),
    "right": ((8, 0), "left", "center"),
    "left": ((-8, 0), "right", "center"),
}
LABEL_PLACEMENT_OVERRIDES = {("math", "Q"): "right", ("medical", "G"): "left"}


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


def render_table(label, caption, spec, header, body, note=None):
    lines = [
        "% !TEX root = ../main.tex",
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \small",
        r"    \setlength{\tabcolsep}{2.5pt}",
        r"    \renewcommand{\arraystretch}{1.35}",
        r"    \sbox0{%",
        f"    \\begin{{tabular}}{{{spec}}}",
        r"        \hline",
    ]
    lines += ["        " + line for line in header + body]
    lines += [r"    \end{tabular}}", r"    \usebox{0}"]
    if note:
        lines += [
            r"    \par\smallskip",
            r"    \begin{minipage}{\wd0}",
            r"        \raggedright\footnotesize",
            "        " + note,
            r"    \end{minipage}",
        ]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_text(path, content):
    path.write_text(content, encoding="utf-8")


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_main_table(stats):
    best_correct = {
        domain: max(CONFIGS, key=lambda config: stats[(domain, config.label)]["correct"]).label
        for domain in DOMAINS
    }
    header = header_row(
        [
            r"\multicolumn{2}{|c|}{}",
            r"\multicolumn{4}{c|}{\textbf{Odsetek zadań [\%]}}",
            r"\textbf{Średni}",
            r"\textbf{Tokeny}",
        ],
        r"\cline{3-6}",
    ) + header_row([
        r"\textbf{Zbiór}",
        r"\textbf{Konfiguracja}",
        r"\textbf{Poprawne}",
        r"\textbf{Błędne}",
        r"\textbf{Abstain}",
        r"\textbf{Null}",
        r"\textbf{czas [s]}",
        r"\textbf{na zadanie}",
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
                decimal(s["mean_time"]),
                thousands(s["tokens_per_task"]),
            ]
            is_last = index == len(CONFIGS) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-8}"))
    spec = column_spec(
        [center_column("2.0cm"), center_column("2.2cm")]
        + [center_column(width) for width in ("1.7cm", "1.45cm", "1.55cm", "1.25cm")]
        + [center_column("1.7cm"), center_column("2.0cm")]
    )
    caption = r"Wyniki zbiorcze dla obu zbiorów danych ($n = 250$ zadań na zbiór)"
    note = (
        r"Q -- \texttt{qwen3.5:9b}, G -- \texttt{gemma4:12b}; Q+Q, Q+G i G+G -- debata odpowiednich par modeli. "
        "Czas i liczba tokenów dotyczą jednego zadania, a tokeny (wejściowe i wygenerowane) zliczono "
        "we wszystkich wywołaniach modeli. Pogrubiono najwyższy odsetek poprawnych odpowiedzi w zbiorze."
    )
    return render_table("tab:wyniki_zbiorcze", caption, spec, header, body, note)


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
    caption = "Porównania parami konfiguracji na tych samych zadaniach"
    best = ", ".join(
        f"{best_solo_label(stats, domain)} {DOMAIN_LOCATIVE[domain]}" for domain in DOMAINS
    )
    note = (
        r"A i B -- pierwsza i druga konfiguracja w wierszu; $\Delta$ -- różnica odsetka poprawnych odpowiedzi "
        r"A $-$ B w punktach procentowych (pp); „tylko A” i „tylko B” -- liczba zadań rozwiązanych poprawnie "
        r"wyłącznie przez daną konfigurację; p -- wartość p dokładnego testu McNemara, pogrubiona dla "
        rf"$p < 0{{,}}05$. Lepszy solo: {best}."
    )
    return render_table("tab:testy_parowane", caption, spec, header, body, note)


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
    caption = r"Bilans debaty względem rundy niezależnej (liczba zadań, dla każdej debaty $n = 250$)"
    note = (
        "Zgoda od razu -- zadania, w których oba modele podały równoważne odpowiedzi w rundzie niezależnej, "
        "przed debatą; „w tym poprawna” -- gdy ta odpowiedź jest poprawna. Pozostałe zadania trafiły do debaty. "
        "Żaden model poprawny i Jeden model poprawny -- zadania z debatą, w których w rundzie niezależnej "
        "poprawną odpowiedź podał odpowiednio żaden albo jeden z modeli. Uratowane -- zadania z pierwszej z tych "
        "grup, w których wynik końcowy jest poprawny. Utracone -- zadania z drugiej, w których wynik końcowy nie "
        "jest poprawny, także gdy debata zakończyła się abstain."
    )
    return render_table("tab:bilans_debaty", caption, spec, header, body, note)


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
            })
    return rows


def build_pilot_table(rows):
    arrow = r" $\rightarrow$ "
    sub = lambda label: r"\textbf{%s}" % label
    header = header_row(
        [r"\multicolumn{3}{|c|}{}", r"\multicolumn{4}{c|}{\textbf{bez rozumowania%sz rozumowaniem}}" % arrow],
        r"\cline{4-7}",
    ) + header_row([
        sub("Zbiór"),
        sub("Konfiguracja"),
        sub("Liczba zadań"),
        sub("Poprawne"),
        sub("Błędne"),
        sub("Brak odpowiedzi"),
        sub("Średni czas na zadanie [s]"),
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
            ]
            is_last = index == len(domain_rows) - 1
            body.append(table_row(cells, r"\hline" if is_last else r"\cline{2-7}"))
    spec = column_spec(
        [center_column("2.0cm"), center_column("2.3cm"), center_column("1.5cm")]
        + [center_column("1.9cm"), center_column("1.9cm"), center_column("2.0cm"), center_column("2.7cm")]
    )
    caption = "Tryb bez rozumowania i z rozumowaniem na tych samych zadaniach"
    note = (
        "W kolumnach Poprawne, Błędne, Brak odpowiedzi i Średni czas każda komórka ma postać „x $\\rightarrow$ y”, "
        "gdzie x to wynik bez rozumowania, a y to wynik z rozumowaniem. Brak odpowiedzi oznacza w trybie "
        "jednomodelowym odpowiedź obciętą limitem tokenów (null), a w debacie brak konsensusu (abstain). "
        "Liczby w kolumnach Poprawne, Błędne i Brak odpowiedzi to liczby zadań."
    )
    return render_table("tab:pilotaz_rozumowanie", caption, spec, header, body, note)


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


def efficiency_frontier(points):
    frontier = []
    best_accuracy = float("-inf")
    for point in sorted(points, key=lambda item: item[0]):
        if point[1] > best_accuracy:
            frontier.append(point)
            best_accuracy = point[1]
    return frontier


def has_close_point_above(point, points):
    tokens, accuracy, _ = point
    return any(
        0 < other_accuracy - accuracy < 4.5 and abs(math.log(other_tokens / tokens)) < 0.45
        for other_tokens, other_accuracy, _ in points
    )


def plot_accuracy_vs_cost(stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 3.5), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        points = [
            (stats[(domain, config.label)]["tokens_per_task"], share(stats[(domain, config.label)], "correct"), config)
            for config in CONFIGS
        ]
        frontier = efficiency_frontier(points)
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], color=MUTED, linewidth=1.2, zorder=2)
        for point in points:
            tokens, accuracy, config = point
            ax.plot(tokens, accuracy, linestyle="none", marker=MARKERS[config.kind], markersize=6.5,
                    markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            placement = LABEL_PLACEMENT_OVERRIDES.get(
                (domain, config.label), "below" if has_close_point_above(point, points) else "above"
            )
            offset, horizontal, vertical = LABEL_PLACEMENTS[placement]
            ax.annotate(config.label, (tokens, accuracy), xytext=offset, textcoords="offset points",
                        ha=horizontal, va=vertical, fontsize=8.5, color=INK)
        style_axes(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.6, linestyle="-")
        ax.set_xscale("log")
        ax.set_xlim(300, 9000)
        ax.set_ylim(55, 90)
        ax.set_xticks(TOKEN_TICKS)
        ax.set_xticklabels([f"{tick:,}".replace(",", " ") for tick in TOKEN_TICKS])
        ax.minorticks_off()
        ax.set_title(DOMAIN_TITLES[domain], loc="left", fontsize=10, fontweight="bold", color=INK, pad=6)
        ax.set_xlabel("Tokeny na zadanie (skala logarytmiczna)", fontsize=9)
    axes[0].set_ylabel("Poprawne [%]", fontsize=9)
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D([], [], linestyle="none", marker=MARKERS[kind], markersize=6.5, markerfacecolor=BLUE,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        for kind, label in MARKER_LEGEND
    ]
    handles.append(Line2D([], [], color=MUTED, linewidth=1.2, label="granica efektywności"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
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

    write_csv(RESULTS_DIR / "main_table.csv", main_table_rows(stats))
    write_text(TABLES_DIR / "wyniki-zbiorcze.tex", build_main_table(stats))

    paired = paired_results(main_data, stats)
    write_csv(RESULTS_DIR / "paired_tests.csv", paired)
    write_text(TABLES_DIR / "testy-parowane.tex", build_paired_table(paired, stats))

    balance = debate_balance_rows(main_data)
    write_csv(RESULTS_DIR / "debate_balance.csv", balance)
    write_text(TABLES_DIR / "bilans-debaty.tex", build_balance_table(balance))

    pilot = pilot_rows(main_data, pilot_data)
    write_csv(RESULTS_DIR / "pilot_thinking.csv", pilot)
    write_text(TABLES_DIR / "pilotaz-rozumowanie.tex", build_pilot_table(pilot))

    plot_outcome_composition(stats, FIGURES_DIR / "rys-sklad-wynikow.png")
    plot_accuracy_vs_cost(stats, FIGURES_DIR / "rys-dokladnosc-koszt.png")
    print("Zapisano 4 tabele (text/tabels), 2 rysunki (text/figures) i 4 pliki csv (code/results)")


if __name__ == "__main__":
    main()
