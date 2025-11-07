import argparse
import asyncio

from src.math_dataset import load_questions
from src.pgr_experiment import run_pgr_experiment
from src.utils import choose_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Loading datasets...")
    train_ds = load_questions("train", limit=args.train_limit)
    test_ds = load_questions("test", limit=args.test_limit)
    indices = choose_indices(len(train_ds), args.k)
    print("Running PGR experiment...")
    async def runner():
        pgr = await run_pgr_experiment(
            args.weak_model,
            args.strong_model,
            train_ds,
            test_ds,
            indices,
            verbose=args.verbose
        )
        print("====================================")
        print(f"weak model    : {args.weak_model}")
        print(f"strong model  : {args.strong_model}")
        print(f"indices       : {indices}")
        print(f"PGR           : {pgr:.3f}")
        print("====================================")

    asyncio.run(runner())

if __name__ == "__main__":
    main()
