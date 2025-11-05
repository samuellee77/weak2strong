import random
from typing import List

def choose_indices(n: int, k: int, seed: int = None) -> List[int]:
    if seed is not None:
        random.seed(seed)
    return random.sample(range(n), k)
