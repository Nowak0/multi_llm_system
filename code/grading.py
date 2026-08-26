"""Answer grading and equivalence checking.

Two jobs live here:

1. GRADING a prediction against the gold answer (`is_correct`) - used to score
   every condition.
2. EQUIVALENCE between two predictions (`equiv_fn`) - used by the debate loop to
   decide whether the agents have reached consensus.
"""

import re
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
from utils import log

NO_SOLUTION = "#no_solution"
ABSTAIN = "ABSTAIN"
CONSOLE_LOGS = True


def _is_blank(value) -> bool:
    """True for answers that carry no information (None / empty / no-solution)."""
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == NO_SOLUTION.lower()


#
# Math: Hendrycks MATH `is_equiv` string normalisation
#

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    a = substr[0]
                    b = substr[1]
                except IndexError:
                    return string
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
        return string
    except (ValueError, TypeError):
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ ...}" trailing units, keep only the part before it
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _hendrycks_equiv(a: str, b: str) -> bool:
    try:
        return _strip_string(a) == _strip_string(b)
    except Exception:
        return a == b


#
# Math: sympy symbolic/numeric fallback (catches things string-equality misses,
# e.g. 0.5 == 1/2 written differently, 6*sqrt(2) == 6\sqrt2)
#

def _latex_to_expr(s: str) -> str:
    """Turn a light-LaTeX / plain math answer into a sympy-parseable string."""
    s = s.strip().strip("$")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # \frac{a}{b} -> ((a)/(b)); repeat to catch simple nesting
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    return s


def _sympy_equal(a: str, b: str) -> bool:
    transforms = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )

    ea, eb = _p(a, transforms), _p(b, transforms)
    if ea is None or eb is None:
        return False
    try:
        if sympy.simplify(ea - eb) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(float(ea.evalf()) - float(eb.evalf())) < 1e-9
    except Exception:
        return False


def _p(x, transforms):
    try:
        return parse_expr(_latex_to_expr(x), transformations=transforms, evaluate=True)
    except Exception:
        return None


def math_equiv(a, b) -> bool:
    """True if two math answers are mathematically equivalent."""
    if _is_blank(a) or _is_blank(b):
        return False
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    return _hendrycks_equiv(a, b) or _sympy_equal(a, b)


def grade_math(pred, gold) -> bool:
    return math_equiv(pred, gold)


# ===========================================================================
# Multiple choice (medical)
# ===========================================================================
_LETTER_PATTERNS = [
    re.compile(r"\b(?:answer|option|choice)\b[^A-Za-z]{0,6}\(?([A-Ea-e])\)?"),
    re.compile(r"[\(\[]\s*([A-Ea-e])\s*[\)\]]"),
    re.compile(r"^\s*([A-Ea-e])\b"),
    re.compile(r"\b([A-Ea-e])\b"),
]


def extract_letter(pred):
    """Pull the chosen option letter (A-E, uppercased) out of a prediction, or
    None if nothing letter-like can be found."""
    if _is_blank(pred):
        return None
    text = str(pred).strip()
    for pattern in _LETTER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    return None


def mcq_equiv(a, b) -> bool:
    extracted_a, extracted_b = extract_letter(a), extract_letter(b)
    return extracted_a is not None and extracted_a == extracted_b


def grade_mcq(pred, gold) -> bool:
    extracted_pred = extract_letter(pred)
    extracted_gold = extract_letter(gold) or (str(gold).strip().upper() if gold is not None else None)
    return extracted_pred is not None and extracted_gold is not None and extracted_pred == extracted_gold


# ===========================================================================
# Unified entry points
# ===========================================================================
def is_correct(pred, gold, domain: str) -> bool:
    """Single grading entry point. domain in {"math", "medical"}."""
    if domain == "math":
        return grade_math(pred, gold)
    if domain == "medical":
        return grade_mcq(pred, gold)
    raise ValueError(f"Unknown domain: {domain!r}")


def equiv_fn(domain: str):
    """Return the pred-vs-pred equivalence function the debate loop uses to
    detect consensus for this domain."""
    if domain == "math":
        return math_equiv
    if domain == "medical":
        return mcq_equiv
    raise ValueError(f"Unknown domain: {domain!r}")


def main():
    math_checks = [
        ("21/31", "\\frac{21}{31}", True),
        ("0.5", "1/2", True),
        ("6*sqrt(2)", "6\\sqrt2", True),
        ("\\frac{1}{2}", "0.5", True),
        ("3*sqrt(5)/2", "\\frac{3\\sqrt5}{2}", True),
        ("2", "3", False),
        ("1/2", "1/3", False),
    ]
    ok = True
    for a, b, expected in math_checks:
        got = math_equiv(a, b)
        flag = "OK" if got == expected else "FAIL"
        ok = ok and got == expected
        log(CONSOLE_LOGS, f"  [{flag}] math_equiv({a!r}, {b!r}) = {got}  (expected {expected})")

    mcq_checks = [
        ("The answer is (B).", "B", True),
        ("I choose option C", "C", True),
        ("A", "B", False),
        ("D", "D", True),
    ]
    for a, b, expected in mcq_checks:
        got = grade_mcq(a, b)
        flag = "OK" if got == expected else "FAIL"
        ok = ok and got == expected
        log(CONSOLE_LOGS, f"  [{flag}] grade_mcq({a!r}, {b!r}) = {got}  (expected {expected})")

    log(True, "\nALL PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()