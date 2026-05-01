"""Per-question scoring.

Self-contained — does not call the course-provided `judger.py` because that
path requires `sympy.parsing.latex.parse_latex`, which in turn needs
`antlr4-python3-runtime>=4.11`. Hydra/omegaconf pin the same package at 4.9
and the two are mutually incompatible at runtime. Rather than monkey-patch
sympy or regenerate omegaconf grammars, this module uses `sympify` directly
on regex-normalized prediction strings — covers every gold-answer shape that
appears in `data/public.jsonl` (numeric, expression, fraction, interval,
equation, T/F, letter / multi-letter, multi-answer lists).
"""

import re
from dataclasses import dataclass
from typing import Optional

from sympy import simplify, sympify
from sympy.core.sympify import SympifyError


# ─── Public API ──────────────────────────────────────────────────────────────


@dataclass
class ScoredResult:
    correct: bool
    extracted: str
    finished_with_box: bool


def score_one(item: dict, response: str) -> ScoredResult:
    """Score one (question, response) pair.

    `item` is one row from `data/public.jsonl`: `{id, question, answer, options?}`.
    """
    if item.get("options"):
        return _score_mcq(item, response)
    return _score_free(item, response)


# ─── Common extraction ───────────────────────────────────────────────────────

_BOXED_LETTER_RE = re.compile(r"\\boxed\{([A-Za-z])\}")


def _strip_thinking(text: str) -> str:
    """Drop everything up to and including the last </think> tag."""
    end = text.rfind("</think>")
    return text[end + len("</think>"):] if end >= 0 else text


def _last_boxed_content(text: str) -> str:
    """Return the inner content of the last \\boxed{...} after </think>, else ''.

    Walks brace-balanced so nested braces inside the boxed expression survive.
    """
    visible = _strip_thinking(text)
    idx = visible.rfind("\\boxed")
    if idx < 0:
        return ""
    i = visible.find("{", idx)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(visible)):
        if visible[j] == "{":
            depth += 1
        elif visible[j] == "}":
            depth -= 1
            if depth == 0:
                return visible[i + 1 : j]
    return ""


def _extract_letter(text: str) -> str:
    visible = _strip_thinking(text)
    m = _BOXED_LETTER_RE.search(visible)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", visible.upper())
    return matches[-1] if matches else ""


# ─── MCQ scoring ─────────────────────────────────────────────────────────────


def _score_mcq(item: dict, response: str) -> ScoredResult:
    extracted = _extract_letter(response)
    finished = bool(_BOXED_LETTER_RE.search(_strip_thinking(response)))
    gold = item.get("answer")
    correct = extracted == str(gold).strip().upper()
    return ScoredResult(correct, extracted, finished)


# ─── LaTeX → plain-Python normalization ──────────────────────────────────────

_TF_MAP = {
    "true": True, "yes": True, "correct": True, "right": True, "1": True,
    "false": False, "no": False, "incorrect": False, "wrong": False, "0": False,
}

_LATEX_STRIP = [
    r"\\displaystyle", r"\\textstyle", r"\\scriptstyle",
    r"\\left", r"\\right",
    r"\\!", r"\\,", r"\\;", r"\\:", r"\\quad", r"\\qquad",
    r"\\text\{[^}]*\}",
]

_LATEX_REPLACE = {
    r"\\pi": "pi",
    r"\\infty": "oo", r"\\infin": "oo", r"infinity": "oo", r"\binfty\b": "oo",
    r"\\cdot": "*", r"\\times": "*", r"\\div": "/",
    r"\\dfrac": r"\\frac", r"\\tfrac": r"\\frac",
    r"\\le\b": "<=", r"\\ge\b": ">=", r"\\leq": "<=", r"\\geq": ">=", r"\\neq": "!=",
    r"\\sin": "sin", r"\\cos": "cos", r"\\tan": "tan",
    r"\\sec": "sec", r"\\csc": "csc", r"\\cot": "cot",
    r"\\log": "log", r"\\ln": "ln", r"\\exp": "exp",
    r"\\arcsin": "asin", r"\\arccos": "acos", r"\\arctan": "atan",
    r"°": "", r"\\circ": "",
}


def _normalize_latex(s: str) -> str:
    """Convert a string that may contain LaTeX into something `sympify` can read."""
    if not isinstance(s, str):
        return str(s)
    s = s.strip()
    # Strip math-mode dollars
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Wrappers + cosmetic commands
    for pat in _LATEX_STRIP:
        s = re.sub(pat, "", s)
    # Alias \dfrac / \tfrac to \frac BEFORE fraction processing.
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # \frac{a}{b} → ((a)/(b)) — repeat to handle nesting
    for _ in range(8):
        s2 = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    # \sqrt{x} → sqrt(x)  /  \sqrt[n]{x} → (x)**(1/n)
    s = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    # Token replacements
    for pat, repl in _LATEX_REPLACE.items():
        s = re.sub(pat, repl, s)
    # Imaginary unit: bare 'i' as a token (preceded by digit or operator) → I
    s = re.sub(r"(?<![A-Za-z_])i(?![A-Za-z_])", "I", s)
    # ^ → ** for powers
    s = s.replace("^", "**")
    # 2x → 2*x, 2(x+1) → 2*(x+1)
    s = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", s)
    # )( → )*( implicit multiplication
    s = re.sub(r"\)\s*\(", ")*(", s)
    # Strip stray backslashes that survived
    s = re.sub(r"\\(?=[A-Za-z])", "", s)
    # Commas in numbers: 105,950 → 105950 (but preserve commas separating
    # arguments by only collapsing groups-of-three after a digit)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    # Trim residual whitespace
    return s.strip()


def _is_decimal(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _percent_to_number(s: str) -> Optional[str]:
    """`0.18%` → `0.0018` (return string for sympify); else None."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*%\s*", s)
    if m:
        return str(float(m.group(1)) / 100.0)
    return None


# ─── Symbolic + numeric equivalence ──────────────────────────────────────────

_NUMERIC_TOL = 1e-6


def _safe_sympify(s: str):
    """Try sympify; on failure, try to coerce. Returns a sympy expr or None."""
    if not s:
        return None
    s = s.strip().rstrip(".")
    # Trim spurious trailing comma (e.g. "5/8,")
    s = s.rstrip(",")
    # Percentage shortcut
    pct = _percent_to_number(s)
    if pct is not None:
        s = pct
    try:
        return sympify(s, rational=False)
    except (SympifyError, SyntaxError, TypeError, AttributeError):
        return None


def _scalars_equal(pred_expr, gold_expr) -> bool:
    """Symbolic + numeric comparison. Both are sympy expressions or None."""
    if pred_expr is None or gold_expr is None:
        return False
    try:
        diff = simplify(pred_expr - gold_expr)
        if diff == 0:
            return True
    except Exception:
        pass
    try:
        a = complex(pred_expr.evalf())
        b = complex(gold_expr.evalf())
        if abs(a - b) <= _NUMERIC_TOL * max(1.0, abs(a), abs(b)):
            return True
    except Exception:
        pass
    return False


def _string_equal(pred: str, gold: str) -> bool:
    """Case-insensitive equality after stripping whitespace + trailing punctuation."""
    return pred.strip().rstrip(".,").lower() == gold.strip().rstrip(".,").lower()


# ─── Shape-aware comparators ─────────────────────────────────────────────────


def _is_pair(s: str) -> bool:
    s = s.strip()
    return (
        len(s) >= 3
        and s[0] in "([" and s[-1] in ")]"
        and s.count(",") >= 1
    )


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` outside of brackets. Treats (), [], {} as nesting."""
    parts: list[str] = []
    depth = 0
    cur = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    for ch in s:
        if ch in pairs:
            depth += 1
            cur.append(ch)
        elif ch in closers:
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _strip_outer_brackets(s: str) -> str:
    """Strip exactly one matching outer bracket pair, if any."""
    s = s.strip()
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        return s[1:-1].strip()
    return s


def _pair_equal(pred: str, gold: str) -> bool:
    """Compare ordered pairs `(a, b)` componentwise. Used for ordered-list /
    point / interval-pair golds."""
    pred = _strip_outer_brackets(pred)
    gold = _strip_outer_brackets(gold)
    pred_parts = _split_top_level(pred)
    gold_parts = _split_top_level(gold)
    if len(pred_parts) != len(gold_parts):
        return False
    for p, g in zip(pred_parts, gold_parts):
        if not _atom_equal(p, g):
            return False
    return True


def _is_equation(s: str) -> bool:
    return s.count("=") == 1 and "==" not in s and "<=" not in s and ">=" not in s and "!=" not in s


def _equation_equal(pred: str, gold: str) -> bool:
    """Compare two equations by checking that `lhs - rhs` is a scalar multiple."""
    if not _is_equation(pred) or not _is_equation(gold):
        return False
    p_lhs, p_rhs = pred.split("=", 1)
    g_lhs, g_rhs = gold.split("=", 1)
    pe = _safe_sympify(_normalize_latex(p_lhs))
    pe2 = _safe_sympify(_normalize_latex(p_rhs))
    ge = _safe_sympify(_normalize_latex(g_lhs))
    ge2 = _safe_sympify(_normalize_latex(g_rhs))
    if None in (pe, pe2, ge, ge2):
        return False
    p_diff = pe - pe2
    g_diff = ge - ge2
    if _scalars_equal(p_diff, g_diff):
        return True
    # Check scalar multiple
    try:
        ratio = simplify(p_diff / g_diff)
        if ratio.is_constant() and ratio != 0:
            return True
    except Exception:
        pass
    return False


def _atom_equal(pred: str, gold: str) -> bool:
    """Compare two atomic answers (no commas/equals at top level)."""
    pred_n = _normalize_latex(pred)
    gold_n = _normalize_latex(gold)
    # T/F shortcut
    if pred_n.lower() in _TF_MAP and gold_n.lower() in _TF_MAP:
        return _TF_MAP[pred_n.lower()] == _TF_MAP[gold_n.lower()]
    # Single letter or short word: case-insensitive equality
    if (
        gold_n.isalpha()
        and (
            (len(gold_n) <= 6 and gold_n.isupper())
            or (len(gold_n) <= 12 and gold_n.lower() in {"up", "down", "left", "right",
                                                          "yes", "no", "increasing", "decreasing"})
        )
    ):
        return _string_equal(pred_n, gold_n)
    # Numeric / expression via sympify
    pe = _safe_sympify(pred_n)
    ge = _safe_sympify(gold_n)
    if _scalars_equal(pe, ge):
        return True
    # Fallback to string equality after normalization
    return _string_equal(pred_n, gold_n)


def _multi_letter_equal(pred: str, gold: str) -> bool:
    """For golds like 'BCEG' (multiple-choice multi-select). Compare as letter sets."""
    p = re.sub(r"[^A-Za-z]", "", pred).upper()
    g = re.sub(r"[^A-Za-z]", "", gold).upper()
    return set(p) == set(g) and len(p) == len(g)


def _free_form_match(pred: str, gold: str) -> bool:
    """Top-level shape dispatch for one (pred, gold) pair."""
    if pred is None:
        return False
    p = pred.strip()
    g = gold.strip() if isinstance(gold, str) else str(gold).strip()
    if not p or not g:
        return False
    # Equation
    if _is_equation(g):
        return _equation_equal(p, g)
    # Pair / interval / tuple
    if _is_pair(g):
        # Try componentwise; also accept the pred being unwrapped
        if _is_pair(p) and _pair_equal(p, g):
            return True
        return False
    # Multi-letter MCQ-as-free-form
    if g.isalpha() and len(g) >= 2 and g.isupper():
        return _multi_letter_equal(p, g)
    # Atomic
    return _atom_equal(p, g)


# ─── Free-form scorer ────────────────────────────────────────────────────────


def _score_free(item: dict, response: str) -> ScoredResult:
    extracted = _last_boxed_content(response)
    finished = bool(extracted)
    gold = item.get("answer")
    gold_list: list = gold if isinstance(gold, list) else [gold]

    if not extracted:
        return ScoredResult(False, "", finished)

    try:
        if len(gold_list) == 1:
            correct = _free_form_match(extracted, str(gold_list[0]))
        else:
            # Multi-answer: try to split prediction on commas (top-level) or
            # newlines, then match component-wise in order.
            pred_parts = _split_top_level(extracted, sep=",")
            if len(pred_parts) != len(gold_list):
                # Maybe model emitted multiple \boxed{}'s — collect them all.
                visible = _strip_thinking(response)
                all_boxes = _all_boxed(visible)
                if len(all_boxes) == len(gold_list):
                    pred_parts = all_boxes
            if len(pred_parts) != len(gold_list):
                return ScoredResult(False, extracted, finished)
            correct = all(
                _free_form_match(p, str(g)) for p, g in zip(pred_parts, gold_list)
            )
    except Exception:
        return ScoredResult(False, extracted, finished)

    return ScoredResult(correct, extracted, finished)


def _all_boxed(visible: str) -> list[str]:
    """Return the inner contents of every \\boxed{...} in left-to-right order."""
    out: list[str] = []
    i = 0
    while True:
        idx = visible.find("\\boxed", i)
        if idx < 0:
            return out
        b = visible.find("{", idx)
        if b < 0:
            return out
        depth = 0
        for j in range(b, len(visible)):
            if visible[j] == "{":
                depth += 1
            elif visible[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(visible[b + 1 : j])
                    i = j + 1
                    break
        else:
            return out
