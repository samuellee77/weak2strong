# src/prm800k_fix.py
from pathlib import Path

def patch_grader() -> None:
    try:
        import prm800k
    except ImportError:
        return

    prm_path = getattr(prm800k, "__file__", None)
    if prm_path is None:
        return

    pkg_path = Path(prm_path).resolve().parent
    grader_path = pkg_path / "grading" / "grader.py"
    if not grader_path.exists():
        return

    text = grader_path.read_text()
    old = "from grading import math_normalize"
    new = "from . import math_normalize"
    if old in text and new not in text:
        grader_path.write_text(text.replace(old, new))
