"""Canonical result schema + JSONL helpers.

Every experiment writes rows in this exact shape so leaderboards across experiments
stay comparable. The schema is deliberately wider than what any single run uses
(`all_responses`, `n_iters`) so multi-turn / multi-sample runs round-trip
without losing information.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class ResultRow:
    id: int
    run_id: str
    prompt_id: str
    is_mcq: bool
    gold: Union[str, list[str]]
    response: str
    extracted: str
    correct: bool
    finished_with_box: bool
    n_response_tokens: int
    n_iters: int
    sampling: dict[str, Any]
    max_tokens: int
    all_responses: Optional[list[str]] = None


def load_jsonl(path: Union[str, Path]) -> list[dict]:
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
