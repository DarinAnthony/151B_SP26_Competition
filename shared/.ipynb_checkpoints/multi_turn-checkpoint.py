"""Multi-turn inference helpers (PHP, Self-Refine).

Both helpers iterate over a *batch* of items but keep per-item state so
convergence behavior is preserved. Each round, only items that haven't yet
converged are re-batched — recovering most of the throughput of single-shot
batched inference while still letting items "settle" at different rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.prompt_format import append_php_hint, append_self_refine_critique, build_chat_messages
from shared.prompts import Prompt
from shared.runner import ModelHandle
from shared.schemas import SamplingCfg

# Reuse the same boxed-extractor used by scoring so PHP's convergence check is
# consistent with what eval.py records as the final answer.
from shared.scoring import _extract_boxed_content, _extract_letter


def _extract_for_convergence(item: dict, response: str) -> str:
    return _extract_letter(response) if item.get("options") else _extract_boxed_content(response)


@dataclass
class MultiTurnResult:
    response: str  # the final-iteration response used for scoring
    all_responses: list[str]  # one per iteration in order
    n_iters: int  # 1 if first turn already converged
    n_response_tokens: int  # tokens in the final response (for the result row)
    converged: bool  # PHP-only: did consecutive iters produce equal extracts


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
    """
    n = len(items)
    chat_state: list[list[dict]] = [build_chat_messages(item, prompt) for item in items]
    all_responses: list[list[str]] = [[] for _ in range(n)]
    last_extracted: list[Optional[str]] = [None] * n
    final_response: list[str] = [""] * n
    final_tokens: list[int] = [0] * n
    n_iters: list[int] = [0] * n
    converged: list[bool] = [False] * n
    active = list(range(n))

    for it in range(max_iters):
        if not active:
            break
        active_msgs = [chat_state[i] for i in active]
        outputs = handle.generate_batch(active_msgs, sampling, max_tokens)

        next_active: list[int] = []
        for slot, idx in enumerate(active):
            response = outputs[slot].responses[0]
            tokens = outputs[slot].n_response_tokens[0] if outputs[slot].n_response_tokens else 0
            extracted = _extract_for_convergence(items[idx], response)

            all_responses[idx].append(response)
            final_response[idx] = response
            final_tokens[idx] = tokens
            n_iters[idx] = it + 1

            if last_extracted[idx] is not None and extracted and extracted == last_extracted[idx]:
                converged[idx] = True
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
            converged=converged[i],
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
    """
    n = len(items)
    chat_state: list[list[dict]] = [build_chat_messages(item, prompt) for item in items]
    all_responses: list[list[str]] = [[] for _ in range(n)]

    # Turn 1
    outputs = handle.generate_batch(chat_state, sampling, max_tokens)
    for i in range(n):
        response = outputs[i].responses[0]
        all_responses[i].append(response)
        chat_state[i] = append_self_refine_critique(
            chat_state[i], response, prompt.self_refine_critique
        )

    # Turn 2
    outputs2 = handle.generate_batch(chat_state, sampling, max_tokens)
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
                n_response_tokens=tokens,
                converged=False,
            )
        )
    return results
