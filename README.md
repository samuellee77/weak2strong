# Weak-to-Strong: PGR Experiments on the MATH Dataset

This repository reproduces **Weak-to-Strong (W2S)** style experiments using few-shot prompting on the **MATH** dataset.  
It measures **Performance Gap Recovered (PGR)** when a strong model is prompted with examples labeled by a weaker one.

---

## Repository Structure

```txt
weak2strong/
├── src/                     # Core experiment code
│   ├── config.py            # env + dataset paths, prm800k loader patch
│   ├── math_dataset.py      # MATH dataset + evaluation helpers
│   ├── llm_client.py        # Async OpenAI client with retries
│   ├── prompting.py         # Few-shot prompt construction utilities
│   ├── pgr_experiment.py    # PGR metric implementation
│   ├── utils.py             # Misc. helpers
│   └── prm800k_fix.py       # Patches prm800k grader import issue
│
├── scripts/
│   └── run_pgr.py           # Main entrypoint (CLI)
│
├── external/
│   └── prm800k/             # Cloned OpenAI dataset repo (via Git LFS)
│
├── results/                 # Experiment results are stored here
│
├── .env.example             # Template for API keys
├── pyproject.toml           # Editable install config
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/samuellee77/weak2strong.git
cd weak2strong
````

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate.bat  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Pull the MATH dataset (`prm800k`) with Git LFS

The dataset uses Git Large File Storage.

```bash
brew install git-lfs        # macOS (or apt install git-lfs)
git lfs install
cd external/prm800k
git lfs pull
cd ../..
```

---

## Environment Variables

Copy `.env.example` → `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-your-openai-key
```

The script automatically loads these from `.env`.

---

## Run an Experiment

Go to `main.ipynb` and run the explore different models and parameters.

## What It Does

1. Loads the **MATH** dataset from `prm800k/prm800k/math_splits/`
2. Builds few-shot prompts using examples from the **weak** model
3. Evaluates both weak and strong models using OpenAI’s async API
4. Computes **Performance Gap Recovered (PGR)**:
   [
   \text{PGR} = \frac{A_{S|W} - A_{W|G}}{A_{S|G} - A_{W|G}}
   ]
   where:

   * (A_{S|W}): strong model accuracy using weak-labeled examples
   * (A_{S|G}): strong model accuracy using gold examples
   * (A_{W|G}): weak model accuracy using gold examples

---

## 📚 References

* [OpenAI: Weak-to-Strong Generalization](https://openai.com/index/weak-to-strong-generalization/)
* [PRM800K Dataset (OpenAI)](https://github.com/openai/prm800k)
* [arXiv: Weak-to-Strong Generalization Paper (2023)](https://arxiv.org/abs/2312.09390)

---
