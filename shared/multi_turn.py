"""Multi-turn inference helpers (PHP, Self-Refine).

Both helpers iterate over a *batch* of items but keep per-item state so
convergence behavior is preserved. Each round, only items that haven't yet
converged are re-batched — recovering most of the throughput of single-shot
batched inference while still letting items "settle" at different rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from shared.prompt_format import append_php_hint, append_self_refine_critique, build_chat_messages
from shared.prompts import Prompt
from shared.runner import ModelHandle
from shared.schemas import SamplingCfg
from shared.telemetry import Timer

# Reuse the same boxed-extractor used by scoring so PHP's convergence check is
# consistent with what eval.py records as the final answer.
from shared.scoring import _extract_letter, _last_boxed_content

logger = logging.getLogger(__name__)

# Floor on per-turn cap when splitting max_tokens across multi-turn iterations.
# Keeps tiny max_tokens sweeps from pinching every turn below the length of a
# \boxed{} commit; if the floor activates, we log once and accept that the run
# exceeds the configured total budget.
MIN_PER_TURN_TOKENS = 256


def _extract_for_convergence(item: dict, response: str) -> str:
    return _extract_letter(response) if item.get("options") else _last_boxed_content(response)


@dataclass
class MultiTurnResult:
    response: str  # the final-iteration response used for scoring
    all_responses: list[str]  # one per iteration in order
    n_iters: int  # 1 if first turn already converged
    n_response_tokens: int  # total generation tokens summed across all iterations


def run_php(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
    max_iters: int = 3,
) -> list[MultiTurnResult]:
    """Progressive-Hint Prompting over a batch.

    Round 1: solve plainly. Round k+1: re-prompt with previous answer as a hint.
    Converge per-item when two consecutive iterations produce equal extracted
    answers, or stop at `max_iters`.

    `max_tokens` is the **total** generation budget per item; it is split equally
    across `max_iters` turns so PHP is apples-to-apples with single-prompt kinds.
    Items that converge early simply leave their remaining share unused.
    """
    n = len(items)
    per_turn_max = max(MIN_PER_TURN_TOKENS, max_tokens // max_iters)
    if max_tokens // max_iters < MIN_PER_TURN_TOKENS:
        logger.warning(
            "run_php: max_tokens=%d split across max_iters=%d would give %d tok/turn, "
            "below floor %d. Using %d tok/turn; total budget may exceed max_tokens.",
            max_tokens, max_iters, max_tokens // max_iters, MIN_PER_TURN_TOKENS, per_turn_max,
        )
    with Timer("build_chat_messages"):
        chat_state: list[list[dict]] = [build_chat_messages(item, prompt) for item in items]
    all_responses: list[list[str]] = [[] for _ in range(n)]
    last_extracted: list[Optional[str]] = [None] * n
    final_response: list[str] = [""] * n
    final_tokens: list[int] = [0] * n
    n_iters: list[int] = [0] * n
    active = list(range(n))

    for it in range(max_iters):
        if not active:
            break
        with Timer("php.iter"):
            active_msgs = [chat_state[i] for i in active]
            outputs = handle.generate_batch(active_msgs, sampling, per_turn_max)

            next_active: list[int] = []
            for slot, idx in enumerate(active):
                response = outputs[slot].responses[0]
                tokens = outputs[slot].n_response_tokens[0] if outputs[slot].n_response_tokens else 0
                extracted = _extract_for_convergence(items[idx], response)

                all_responses[idx].append(response)
                final_response[idx] = response
                final_tokens[idx] += tokens
                n_iters[idx] = it + 1

                if last_extracted[idx] is not None and extracted and extracted == last_extracted[idx]:
                    continue
                last_extracted[idx] = extracted

                if it < max_iters - 1:
                    # Prepare next-turn messages: append the response + hint
                    hint_template = prompt.php_hint_template or (
                        "Hint: a previous attempt produced \\boxed{{{prev}}}. "
                        "Verify carefully, then output your final answer in \\boxed{{}}."
                    )
                    chat_state[idx] = append_php_hint(
                        chat_state[idx], response, hint_template, extracted or ""
                    )
                    next_active.append(idx)
            active = next_active

    return [
        MultiTurnResult(
            response=final_response[i],
            all_responses=all_responses[i],
            n_iters=n_iters[i],
            n_response_tokens=final_tokens[i],
        )
        for i in range(n)
    ]


def run_self_refine(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[MultiTurnResult]:
    """Self-Refine: solve, then critique-and-correct in a second turn.

    No convergence loop — exactly 2 iterations.

    `max_tokens` is the **total** generation budget per item; it is split equally
    across the 2 turns so Self-Refine is apples-to-apples with single-prompt kinds.
    """
    n = len(items)
    per_turn_max = max(MIN_PER_TURN_TOKENS, max_tokens // 2)
    if max_tokens // 2 < MIN_PER_TURN_TOKENS:
        logger.warning(
            "run_self_refine: max_tokens=%d split across 2 turns would give %d tok/turn, "
            "below floor %d. Using %d tok/turn; total budget may exceed max_tokens.",
            max_tokens, max_tokens // 2, MIN_PER_TURN_TOKENS, per_turn_max,
        )
    with Timer("build_chat_messages"):
        chat_state: list[list[dict]] = [build_chat_messages(item, prompt) for item in items]
    all_responses: list[list[str]] = [[] for _ in range(n)]
    turn1_tokens: list[int] = [0] * n

    # Turn 1
    with Timer("self_refine.turn1"):
        outputs = handle.generate_batch(chat_state, sampling, per_turn_max)
        for i in range(n):
            response = outputs[i].responses[0]
            turn1_tokens[i] = outputs[i].n_response_tokens[0] if outputs[i].n_response_tokens else 0
            all_responses[i].append(response)
            chat_state[i] = append_self_refine_critique(
                chat_state[i], response, prompt.self_refine_critique
            )

    # Turn 2
    with Timer("self_refine.turn2"):
        outputs2 = handle.generate_batch(chat_state, sampling, per_turn_max)
        results: list[MultiTurnResult] = []
        for i in range(n):
            response = outputs2[i].responses[0]
            tokens = outputs2[i].n_response_tokens[0] if outputs2[i].n_response_tokens else 0
            all_responses[i].append(response)
            results.append(
                MultiTurnResult(
                    response=response,
                    all_responses=all_responses[i],
                    n_iters=2,
                    n_response_tokens=turn1_tokens[i] + tokens,
                )
            )
    return results
