# CLAUDE.md

Guidance for [Claude Code](https://claude.com/claude-code) when working in this repository.

## Use of Claude in this project

Parts of this codebase were written, refactored, or documented with the assistance of
Claude (Anthropic), used through Claude Code. All AI-assisted output was reviewed by the
authors, who remain responsible for the correctness of the code and of the scientific
claims in the accompanying paper. Contributors are welcome to use AI assistance under the
same expectation: review what you submit.

## What this repository is

ExAct is a domain-specific language that represents human motion as underspecified
programs. Programs compile to reward models for zero-shot policy inference, and are
aggregated by action label into *executable behaviour representations* (EBR). The repo
evaluates EBR on human action segmentation and human action anomaly detection.

## Layout

- `exact/` — the library (installed package, `hatchling` build).
  - `programs/` — the ExAct DSL: `grammar.lark` (Lark grammar), `rewards.py`
    (program → reward model), `generator.py` (synthetic program sampling),
    `edit_distance.py` (tree edit distance over programs, with joint-aware costs),
    `selection.py` (diverse program selection / deduplication). See
    `exact/programs/README.md` for the grammar in prose.
  - `parser/` — motion → program parser (LLM with LoRA + a motion encoder,
    grammar-constrained decoding via `syncode`).
  - `models/` — `ExecutableActivityModel` / `ActivityModelCollection`: EBR built by
    aggregating parsed programs per action label.
  - `encoder/` — ST-GCN motion encoder.
  - `data/` — datasets, HumEnv wrapper, ESK loading, trajectory generation.
  - `anomaly/` — anomaly-detection models (normalising flow and sigmoid-score baselines)
    and their trainer.
  - `bm.py` — `BehaviourModel` (metamotivo-backed policy inference).
  - `config.py` — pydantic `TrainConfig`.
- `scripts/` — entrypoints, numbered `1_`–`5_` shell wrappers for the pipeline stages,
  plus `scripts/data`, `scripts/parsing`, `scripts/tasks` Python entrypoints.
- `configs/` — Hydra/OmegaConf configs for `parser`, `segmentation`, `anomaly_detection`.
- `tests/` — pytest suite (grammar, edit distance, executable models).

## Conventions

- Python `>=3.10`, dependencies managed with **uv**. Use `uv sync`, not bare `pip`.
- Run everything from the repository root, inside the venv (`source .venv/bin/activate`).
- Data and generated artifacts live **outside** the repo, under `../exact_data/`
  (`benchmarks/`, `programs/`, `models/`). Never commit datasets, checkpoints, or
  `results/`/`outputs/` contents.
- Experiment config is Hydra YAML under `configs/`; scripts take overrides via
  environment variables (see README stages). Prefer adding a config over hardcoding.
- Logging uses `loguru`; experiment tracking uses `wandb`.
- Public API is re-exported from `exact/__init__.py` — keep `__all__` in the package and
  subpackage `__init__.py` files in sync when adding or renaming exports.

## Working on the DSL

`exact/programs/grammar.lark` is the reference grammar, but constrained decoding in the
parser uses a hand-written state machine rather than the Lark grammar. If you change the
program syntax, update **both**, plus `exact/programs/README.md` and
`tests/test_grammar.py`.

## Checks

```bash
uv sync --extra dev
pytest                      # full suite
pytest tests/test_grammar.py -q
```

The pipeline stages (`scripts/1_*.sh` … `scripts/5_*.sh`) are long-running and need GPUs
and the `../exact_data/` tree; do not launch them as a substitute for tests. Note that
`scripts/1_generate_data.sh`, `2_train_and_parse.sh`, and `3_generate_augmented.sh`
contain a hardcoded `cd /pvc/exact` that must match the local checkout path.

## Citation

If you use this code, please cite the paper: <https://arxiv.org/abs/2604.18064>.
