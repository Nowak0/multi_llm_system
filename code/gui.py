import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from Agent import check_ollama_model, quit_ollama
from baselines import run_solo
from debate import run_debate, DEFAULT_ROUNDS
from grading import ABSTAIN
from main import MODELS as AVAILABLE_MODELS

DOMAINS = {"Matematyka": "math", "Medycyna (ABCD)": "medical"}
AGENT_COLORS = ["#1f6feb", "#c2410c", "#15803d", "#7c3aed"]
POLL_MS = 100

BG = "#f6f7f9"
CARD = "#ffffff"
MUTED = "#6b7280"
ACCENT = "#1f6feb"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("System przetwarzania danych oparty na pojedynczym oraz współpracujących modelach LLM")
        self.geometry("980x780")
        self.minsize(760, 600)
        self.configure(bg=BG)

        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.used_models: set[str] = set()
        self.busy_since: float | None = None
        self.busy_label = ""

        self._init_style()
        self._build_settings()
        self._build_question()
        self._build_status()
        self._build_answer()
        self._build_log()
        self._on_mode_change()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._poll_events)

    # ------------------------------------------------------------------ layout
    def _init_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        base = ("Segoe UI", 10)
        style.configure(".", font=base, background=BG)
        style.configure("Card.TLabelframe", background=CARD, bordercolor="#e5e7eb", relief="solid")
        style.configure("Card.TLabelframe.Label", background=CARD, foreground="#111827",
                        font=("Segoe UI Semibold", 10))
        style.configure("Card.TFrame", background=CARD)
        style.configure("Card.TLabel", background=CARD)
        style.configure("Card.TRadiobutton", background=CARD)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), foreground="white",
                        background=ACCENT, padding=(16, 6), borderwidth=0)
        style.map("Accent.TButton",
                  background=[("disabled", "#9ca3af"), ("active", "#1a5fd0")])
        style.configure("Answer.TLabel", background=CARD, font=("Segoe UI Semibold", 16))
        style.configure("AnswerMeta.TLabel", background=CARD, foreground=MUTED)

    def _card(self, title, side="top"):
        frame = ttk.LabelFrame(self, text=f" {title} ", style="Card.TLabelframe", padding=10)
        frame.pack(side=side, fill="x", padx=14, pady=(10, 0))
        return frame

    def _build_settings(self):
        card = self._card("Ustawienia")

        self.mode = tk.StringVar(value="debate")
        self.domain = tk.StringVar(value=next(iter(DOMAINS)))
        self.solo_model = tk.StringVar(value=AVAILABLE_MODELS[0])
        self.agent_a = tk.StringVar(value=AVAILABLE_MODELS[0])
        self.agent_b = tk.StringVar(value=AVAILABLE_MODELS[-1])
        self.rounds = tk.IntVar(value=DEFAULT_ROUNDS)

        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Tryb:", style="Card.TLabel").pack(side="left")
        for text, value in (("Solo (jeden model)", "solo"), ("Debata (dwa modele)", "debate")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.mode, style="Card.TRadiobutton",
                            command=self._on_mode_change).pack(side="left", padx=(10, 0))
        ttk.Label(row, text="Dziedzina:", style="Card.TLabel").pack(side="left", padx=(30, 6))
        ttk.Combobox(row, textvariable=self.domain, values=list(DOMAINS), state="readonly",
                     width=16).pack(side="left")

        # Solo options
        self.solo_frame = ttk.Frame(card, style="Card.TFrame")
        ttk.Label(self.solo_frame, text="Model:", style="Card.TLabel").pack(side="left")
        ttk.Combobox(self.solo_frame, textvariable=self.solo_model, values=AVAILABLE_MODELS,
                     state="readonly", width=20).pack(side="left", padx=(6, 0))

        # Debate options
        self.debate_frame = ttk.Frame(card, style="Card.TFrame")
        for label, var in (("Agent 1:", self.agent_a), ("Agent 2:", self.agent_b)):
            ttk.Label(self.debate_frame, text=label, style="Card.TLabel").pack(side="left")
            ttk.Combobox(self.debate_frame, textvariable=var, values=AVAILABLE_MODELS,
                         state="readonly", width=20).pack(side="left", padx=(6, 20))
        ttk.Label(self.debate_frame, text="Maks. rund debaty:", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(self.debate_frame, from_=1, to=10, textvariable=self.rounds, width=4,
                    state="readonly").pack(side="left", padx=(6, 0))

    def _build_question(self):
        card = self._card("Pytanie")
        self.question = tk.Text(card, height=4, wrap="word", font=("Segoe UI", 11), relief="solid",
                                borderwidth=1, highlightthickness=0, padx=8, pady=6)
        self.question.pack(fill="x")
        self.question.bind("<Control-Return>", lambda e: (self._start(), "break")[1])

        bar = ttk.Frame(card, style="Card.TFrame")
        bar.pack(fill="x", pady=(8, 0))
        ttk.Label(bar, text="Ctrl+Enter, aby wysłać", style="Card.TLabel",
                  foreground=MUTED).pack(side="left")
        self.run_btn = ttk.Button(bar, text="Zadaj pytanie", style="Accent.TButton", command=self._start)
        self.run_btn.pack(side="right")

    def _build_log(self):
        card = ttk.LabelFrame(self, text=" Przebieg ", style="Card.TLabelframe", padding=10)
        card.pack(fill="both", expand=True, padx=14, pady=(10, 0))
        self.log = scrolledtext.ScrolledText(card, wrap="word", font=("Segoe UI", 10), relief="flat",
                                             borderwidth=0, padx=6, pady=4, state="disabled")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("round", font=("Segoe UI Semibold", 11), foreground="#111827",
                               spacing1=14, spacing3=4)
        self.log.tag_configure("muted", foreground=MUTED)
        self.log.tag_configure("answer", font=("Consolas", 10, "bold"))
        self.log.tag_configure("thought", foreground="#374151", lmargin1=18, lmargin2=18, spacing3=6)
        self.log.tag_configure("ok", foreground="#15803d", font=("Segoe UI Semibold", 10), spacing1=6)
        self.log.tag_configure("warn", foreground="#b45309", font=("Segoe UI Semibold", 10), spacing1=6)
        self.log.tag_configure("error", foreground="#b91c1c")
        for i, color in enumerate(AGENT_COLORS):
            self.log.tag_configure(f"agent{i}", foreground=color, font=("Segoe UI Semibold", 10))

    def _build_answer(self):
        card = self._card("Ostateczna odpowiedź", side="bottom")
        self.answer_var = tk.StringVar(value="-")
        self.answer_meta = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.answer_var, style="Answer.TLabel",
                  wraplength=900).pack(anchor="w")
        ttk.Label(card, textvariable=self.answer_meta, style="AnswerMeta.TLabel").pack(anchor="w")

    def _build_status(self):
        bar = ttk.Frame(self, padding=(14, 8))
        bar.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.pack(side="right")
        self.status = tk.StringVar(value="Gotowy")
        ttk.Label(bar, textvariable=self.status, style="Muted.TLabel").pack(side="left")

    # ----------------------------------------------------------------- helpers
    def _on_mode_change(self):
        self.solo_frame.pack_forget()
        self.debate_frame.pack_forget()
        target = self.solo_frame if self.mode.get() == "solo" else self.debate_frame
        target.pack(fill="x", pady=(10, 0))

    def _write(self, text, *tags):
        self.log.configure(state="normal")
        self.log.insert("end", text, tags)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_busy(self, label):
        self.busy_label = label
        self.busy_since = time.perf_counter()

    def _set_running(self, running):
        self.run_btn.configure(state="disabled" if running else "normal")
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.busy_since = None

    # --------------------------------------------------------------- run logic
    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        question = self.question.get("1.0", "end").strip()
        if not question:
            messagebox.showinfo("Brak pytania", "Wpisz pytanie, zanim je wyślesz.")
            return

        domain = DOMAINS[self.domain.get()]
        mode = self.run_mode = self.mode.get()
        models = [self.solo_model.get()] if mode == "solo" else [self.agent_a.get(), self.agent_b.get()]
        rounds = int(self.rounds.get())
        self.agent_names = [f"Agent {i + 1} ({m})" if mode == "debate" else m for i, m in enumerate(models)]

        self._clear_log()
        self.answer_var.set("…")
        self.answer_meta.set("")
        self._write(f"Tryb: {'debata' if mode == 'debate' else 'solo'} · dziedzina: {domain} · "
                    f"modele: {', '.join(models)}\n", "muted")
        self._set_running(True)
        self._set_busy("Sprawdzanie dostępności modeli w Ollama")

        self.worker = threading.Thread(target=self._run, args=(mode, question, domain, models, rounds),
                                       daemon=True)
        self.worker.start()

    def _run(self, mode, question, domain, models, rounds):
        """Worker thread: never touches tkinter, only pushes events to the queue."""
        emit = self.events.put
        try:
            for m in dict.fromkeys(models):
                emit({"type": "status", "text": f"Sprawdzanie modelu {m}"})
                check_ollama_model(m)
                self.used_models.add(m)
            if mode == "solo":
                run_solo(question, domain, models[0], on_event=emit)
            else:
                run_debate(question, domain, models, rounds=rounds, on_event=emit)
        except Exception as e:
            emit({"type": "error", "text": f"{type(e).__name__}: {e}"})

    def _poll_events(self):
        try:
            while True:
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        if self.busy_since is not None:
            self.status.set(f"{self.busy_label}…  {time.perf_counter() - self.busy_since:.0f} s")
        self.after(POLL_MS, self._poll_events)

    def _handle(self, ev):
        kind = ev["type"]
        if kind == "status":
            self._set_busy(ev["text"])

        elif kind == "round_start":
            title = "Runda 0 - niezależne rozwiązania" if ev["round"] == 0 else f"Runda debaty {ev['round']}"
            self._write(f"{title}\n", "round")

        elif kind == "agent_start":
            self._set_busy(f"{self.agent_names[ev['agent']]} przetwarza")

        elif kind == "agent_done":
            i = ev["agent"]
            self._write(f"{self.agent_names[i]}", f"agent{i % len(AGENT_COLORS)}")
            if ev["failed"]:
                self._write("  - błąd zapytania", "error")
                if ev["round"] > 0:
                    self._write(" (zachowano poprzednią odpowiedź)", "muted")
            self._write("  →  ")
            self._write(f"{ev['answer'] if ev['answer'] is not None else 'brak odpowiedzi'}", "answer")
            if ev.get("elapsed") is not None:
                self._write(f"   ({ev['elapsed']:.1f} s)", "muted")
            self._write("\n")
            if ev.get("thought"):
                self._write(f"{ev['thought']}\n", "thought")

        elif kind == "consensus":
            self._write(f"Konsensus w rundzie {ev['round']}: {ev['answer']}\n", "ok")

        elif kind == "final":
            self._set_running(False)
            if ev["abstained"] or ev["answer"] == ABSTAIN:
                self._write(f"Brak konsensusu po {ev['rounds']} rundach → ABSTAIN\n", "warn")
                self.answer_var.set("ABSTAIN - modele nie doszły do porozumienia")
            else:
                self.answer_var.set(ev["answer"] if ev["answer"] is not None else "brak odpowiedzi")
            meta = f"Czas: {ev['elapsed']:.1f} s"
            if self.run_mode == "debate" and not ev["abstained"]:
                meta += f" · rundy debaty: {ev['rounds']}"
            self.answer_meta.set(meta)
            self.status.set("Gotowe")

        elif kind == "error":
            self._set_running(False)
            self._write(f"\nBłąd: {ev['text']}\n", "error")
            self.answer_var.set("-")
            self.status.set("Błąd - czy Ollama jest uruchomiona?")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Trwa przetwarzanie", "Modele nadal pracują. Zamknąć mimo to?"):
                return
        for m in self.used_models:
            quit_ollama(m)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
