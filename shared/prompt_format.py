"""Build chat-message lists from (item, prompt) pairs.

Few-shot exemplars are rendered as alternating user/assistant turns rather than
crammed into a single user turn — chat-tuned models (Qwen3) respond better to
that shape.
"""

from typing import Optional

from shared.prompts import FewShotExemplar, Prompt


def _format_user_turn(question: str, options: Optional[list[str]]) -> str:
    if not options:
        return question
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"


def _exemplar_turns(exemplars: list[FewShotExemplar]) -> list[dict]:
    turns: list[dict] = []
    for ex in exemplars:
        turns.append({"role": "user", "content": _format_user_turn(ex.question, ex.options)})
        turns.append({"role": "assistant", "content": ex.assistant_response})
    return turns


def build_chat_messages(item: dict, prompt: Prompt) -> list[dict]:
    """Return a list of `{role, content}` dicts ready for `apply_chat_template`."""
    options = item.get("options")
    is_mcq = bool(options)
    system = prompt.system_mcq if is_mcq else prompt.system_free
    exemplars = prompt.few_shot_mcq if is_mcq else prompt.few_shot_free

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(_exemplar_turns(exemplars))
    messages.append({"role": "user", "content": _format_user_turn(item["question"], options)})
    return messages


def append_php_hint(messages: list[dict], assistant_response: str, hint_template: str, prev_extracted: str) -> list[dict]:
    """Extend a message list with the previous assistant response + a hint user turn.

    Used by `shared.multi_turn.run_php` to build turn N+1 from turn N.
    """
    new_messages = list(messages)
    new_messages.append({"role": "assistant", "content": assistant_response})
    new_messages.append(
        {"role": "user", "content": hint_template.format(prev=prev_extracted)}
    )
    return new_messages


def append_self_refine_critique(messages: list[dict], assistant_response: str, critique_prompt: str) -> list[dict]:
    new_messages = list(messages)
    new_messages.append({"role": "assistant", "content": assistant_response})
    new_messages.append({"role": "user", "content": critique_prompt})
    return new_messages
