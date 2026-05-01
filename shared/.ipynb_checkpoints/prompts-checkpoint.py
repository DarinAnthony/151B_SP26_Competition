"""Prompt dataclass + the canonical baseline prompt.

Owned by `shared/` so every experiment compares against the same reference. The
literal strings in `BASELINE_STARTER` are copied verbatim from the starter
notebook (`starter_code_cse151b_comp.ipynb`, cell `4e5169ac`) — do not edit
without a deliberate reason, and update every experiment's writeup if you do.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FewShotExemplar:
    """One worked example. Rendered as alternating user/assistant turns."""
    question: str
    options: Optional[list[str]] = None
    assistant_response: str = ""


@dataclass
class Prompt:
    id: str
    category: str
    kind: str  # "single" | "self_consistency" | "php" | "self_refine"
    system_free: str
    system_mcq: str
    few_shot_free: list[FewShotExemplar] = field(default_factory=list)
    few_shot_mcq: list[FewShotExemplar] = field(default_factory=list)
    php_hint_template: str = (
        "Hint: a previous attempt produced \\boxed{{{prev}}}. "
        "Verify carefully, then output your final answer in \\boxed{{}}."
    )
    self_refine_critique: str = (
        "Critique your answer above for any errors, then produce a corrected "
        "final answer inside \\boxed{}."
    )
    notes: str = ""


# ─── Canonical reference prompt (copied verbatim from starter notebook) ──────

_STARTER_SYSTEM_FREE = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

_STARTER_SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

BASELINE_STARTER = Prompt(
    id="baseline_starter",
    category="reference",
    kind="single",
    system_free=_STARTER_SYSTEM_FREE,
    system_mcq=_STARTER_SYSTEM_MCQ,
    notes=(
        "Literal copy of the starter notebook's SYSTEM_PROMPT_MATH and "
        "SYSTEM_PROMPT_MCQ. Reference baseline for cross-experiment comparison."
    ),
)
