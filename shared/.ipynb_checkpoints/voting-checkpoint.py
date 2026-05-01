"""Self-consistency majority voting.

Used by self-consistency prompts (`sc_*`): given N responses sampled at high
temperature, extract the `\\boxed{}` answer from each and return the most
common. Future work: replace exact-string voting with judger-equivalence-class
voting so `42` and `42.0` get bucketed together.
"""

from collections import Counter


def majority_vote(extracted_list: list[str]) -> str:
    """Return the most common non-empty extraction, breaking ties by first occurrence.

    Empty extractions ("") are excluded from the vote unless every entry is empty,
    in which case "" is returned.
    """
    if not extracted_list:
        return ""
    non_empty = [s for s in extracted_list if s]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    top_count = counts.most_common(1)[0][1]
    for s in non_empty:
        if counts[s] == top_count:
            return s
    return non_empty[0]


def vote_index(extracted_list: list[str]) -> int:
    """Return the index of the first response whose extraction matches the majority vote.

    Useful for selecting which raw response (from the N samples) to record as the
    canonical `response` field.
    """
    if not extracted_list:
        return 0
    winner = majority_vote(extracted_list)
    for i, s in enumerate(extracted_list):
        if s == winner:
            return i
    return 0
