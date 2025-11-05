# src/config.py
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

PRM800K_LOCAL = ROOT / "external" / "prm800k"
if PRM800K_LOCAL.exists():
    sys.path.insert(0, str(PRM800K_LOCAL))
    sys.path.insert(0, str(PRM800K_LOCAL / "prm800k"))
else:
    print("[warn] external/prm800k not found; MATH dataset may not load")

try:
    from src.prm800k_fix import patch_grader
    patch_grader()
except Exception as e:
    print(f"[warn] prm800k patch failed: {e}")

EXTERNAL_DIR = ROOT / "external"
PRM800K_DIR = EXTERNAL_DIR / "prm800k"

TRAIN_SPLIT = PRM800K_DIR / "prm800k" / "math_splits" / "train.jsonl"
TEST_SPLIT = PRM800K_DIR / "prm800k" / "math_splits" / "test.jsonl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

MODELS_LIST = ['gpt-5-mini',
          'o3-mini',
          'gpt-4.1-mini',
          'gpt-4o-mini',
          'gpt-4.1-nano']