import json
import re
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from src.config import TRAIN_SPLIT, TEST_SPLIT
from prm800k.grading.grader import grade_answer


@dataclass
class MATHQuestion:
    problem: str
    answer: str
    solution: str
    subject: str
    level: int
    unique_id: str

    def get_prompt(self, instruction: str | None = None) -> str:
        if instruction is None:
            return f"{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."
        else:
            return f"{instruction}\n\n{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."

    @staticmethod
    def parse_response_for_answer(response: str) -> str:
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        return lines[-1] if lines else ""


def _load_questions(path: Path, limit: int | None = 200):
    with path.open() as f:
        raw = [json.loads(line) for line in f]
    if limit:
        raw = raw[:limit]
    return [MATHQuestion(**d) for d in raw]


def load_questions(split: Literal["train", "test"], limit: int | None = 200):
    if split == "train":
        return _load_questions(TRAIN_SPLIT, limit)
    else:
        return _load_questions(TEST_SPLIT, limit)


def eval_model_answers(dataset, model_answers):
    return [
        grade_answer(ans, q.answer)
        for q, ans in zip(dataset, model_answers, strict=True)
    ]
