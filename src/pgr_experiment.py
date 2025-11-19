import numpy as np
from typing import List

from src.math_dataset import MATHQuestion, eval_model_answers
from src.llm_client import call_many_with_prefix, get_few_shot_prompt, format_answer


async def ask_with_gold(
    model: str,
    train_dataset: List[MATHQuestion],
    test_dataset: List[MATHQuestion],
    indices: List[int],
) -> float:
    shots = []
    for idx in indices:
        q = train_dataset[idx]
        shots.append((q.get_prompt(), q.answer))
    few_shot_prompt = get_few_shot_prompt(shots)

    test_prompts = [q.get_prompt() for q in test_dataset]
    responses = await call_many_with_prefix(few_shot_prompt, test_prompts, model=model)
    answers = [format_answer(r.choices[0].message.content) for r in responses]
    acc = float(np.mean(eval_model_answers(test_dataset, answers)))
    return acc


async def ask_with_weak_labels(
    weak_model: str,
    strong_model: str,
    train_dataset: List[MATHQuestion],
    test_dataset: List[MATHQuestion],
    indices: List[int],
) -> float:
    few_shot_questions = [train_dataset[i] for i in indices]
    weak_prompts = [q.get_prompt() for q in few_shot_questions]
    weak_responses = await call_many_with_prefix([], weak_prompts, model=weak_model)
    weak_answers = [format_answer(r.choices[0].message.content) for r in weak_responses]

    few_shot_prompt = get_few_shot_prompt(
        [(q.get_prompt(), a) for q, a in zip(few_shot_questions, weak_answers)]
    )

    test_prompts = [q.get_prompt() for q in test_dataset]
    strong_responses = await call_many_with_prefix(
        few_shot_prompt, test_prompts, model=strong_model
    )
    strong_answers = [format_answer(r.choices[0].message.content) for r in strong_responses]
    acc = float(np.mean(eval_model_answers(test_dataset, strong_answers)))
    return acc


async def run_pgr_experiment(
    weak_model: str,
    strong_model: str,
    train_dataset: List[MATHQuestion],
    test_dataset: List[MATHQuestion],
    indices: List[int],
    verbose: bool = False,
) -> float:
    acc_strong_weak = await ask_with_weak_labels(
        weak_model, strong_model, train_dataset, test_dataset, indices
    )
    if verbose:
        print(f"Acc (strong model with weak labels): {acc_strong_weak:.4f}")

    acc_strong_gold = await ask_with_gold(
        strong_model, train_dataset, test_dataset, indices
    )
    if verbose:
        print(f"Acc (strong model with gold labels): {acc_strong_gold:.4f}")
    acc_weak_gold = await ask_with_gold(
        weak_model, train_dataset, test_dataset, indices
    )
    if verbose:
        print(f"Acc (weak model with gold labels): {acc_weak_gold:.4f}")

    denom = (acc_strong_gold - acc_weak_gold)
    if denom <= 0:
        pgr = np.nan
    else:
        pgr = (acc_strong_weak - acc_weak_gold) / denom
        pgr = max(0.0, min(1.0, pgr))
    return pgr, acc_strong_weak, acc_strong_gold, acc_weak_gold
