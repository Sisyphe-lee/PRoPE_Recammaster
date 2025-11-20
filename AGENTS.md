以后用中文回答
# Repository Guidelines

## Project Structure & Module Organization
`src/` hosts all runnable modules: `train_recammaster.py` for Lightning training, `inference_recammaster.py` for evaluation, `dataset.py` for Wan2.1/2.2 loaders, and `prope.py` for camera-aware attention. Automation sits in `scripts/` (`train.sh`, `run_parallel.sh`, `inference*.sh`) so experiments can be re-run verbatim. Use `metadata/`, `evaluation/`, and `assets/` for configs, captions, and metrics, and place generated artifacts under `models/`, `results/`, or `test_output/`. `third_party/DiffSynth-Studio/` mirrors the upstream Wan stack—treat it as read-only and document any patch you must carry.

## Build, Test & Development Commands
Create a virtual env (`python -m venv .venv && source .venv/bin/activate`) and install with `pip install -r requirements.txt && pip install -e .`. Export `PYTHONPATH=$PWD/third_party/DiffSynth-Studio:$PYTHONPATH` so DiffSynth modules resolve. Key commands:
- `python src/train_recammaster.py --dataset_path /data --pipeline_type v2v --output_path models/train`
- `bash scripts/train.sh --pipeline-type i2v --dataset-path /nas/a,/nas/b`
- `python src/inference_recammaster.py --ckpt_path models/.../step1100.ckpt --output_dir test_output`
- `bash scripts/inference.sh --pipeline-type i2v` for distributed smoke tests
Run `uv run ruff check . --fix` and `uv run pytest -q` (plus `uv run ruff format` if needed) before pushing.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and concise docstrings for non-obvious math. Keep files/functions snake_case (`vis_cam.py`, `collect_camera_stats`), classes PascalCase, and CLI flags kebab-case. Let Ruff handle imports and formatting; only override when conveying intent. Centralize helpers in `src/utils.py` or `wandb_module.py` rather than scattering script-local utilities.

## Testing Guidelines
Place fast unit tests under `tests/` mirroring `src/` names (e.g., `tests/test_prope.py`). Prefer fixtures from `example_test_data/`, and gate GPU-heavy assertions with `@pytest.mark.cuda`. For camera or sampler edits, capture a short `scripts/inference.sh --debug` trajectory, drop evidence in `test_output/` with metrics (`pose_metrics.csv`), and rerun at least one v2v plus one i2v inference after touching attention, datasets, or schedulers.

## Commit & Pull Request Guidelines
Branches follow `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`, etc. Commits use Conventional Commits with bullet bodies for change, impact, and verification. Every PR should explain motivation, list updated commands/configs, attach validation outputs (W&B link, `results/` plots, or hashes), and note any third-party touchpoints. Update `CHANGELOG.md` for user-visible changes and mention required assets so reviewers can replay the run.

## Security & Configuration Tips
Never commit checkpoints or raw datasets—reference their paths and keep them in `models/` or external storage. Store WANDB tokens, dataset keys, and NCCL overrides in your shell profile, not scripts. When converting camera metadata, use `scripts/convert_camera_json_to_npz.py` or `tools/` so EXIF or pose traces are scrubbed before sharing logs.
