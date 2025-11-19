import argparse
import asyncio
import wandb

from src.math_dataset import load_questions
from src.pgr_experiment import run_pgr_experiment
from src.utils import choose_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="pgr_experiment", config=vars(args))
    config = wandb.config

    print("Loading datasets...")
    train_ds = load_questions("train", limit=config.train_limit)
    test_ds = load_questions("test", limit=config.test_limit)
    indices = choose_indices(len(train_ds), config.k, seed=42)

    print("Running PGR experiment...")
    async def runner():
        pgr = await run_pgr_experiment(
            config.weak_model,
            config.strong_model,
            train_ds,
            test_ds,
            indices,
            verbose=config.verbose
        )
        result = (
            "====================================\n"
            f"weak model    : {config.weak_model}\n"
            f"strong model  : {config.strong_model}\n"
            f"k             : {config.k}\n"
            f"indices       : {indices}\n"
            f"PGR           : {pgr:.3f}\n"
            "====================================\n"
        )
        print(result)

        # Log results to wandb
        wandb.log({
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
            "indices": indices,
            "PGR": pgr
        })

    asyncio.run(runner())
    wandb.finish()

if __name__ == "__main__":
    main()
