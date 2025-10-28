必须用中文回答我
# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds training (`train_recammaster.py`), inference, dataset, VAE, and logging utilities for PRoPE on ReCamMaster.
- `scripts/` provides launchers such as `train.sh` for multi-GPU execution and reproducible experiments.
- `delta_prope_tests/` contains the Triton Δ-RoPE reference implementation and regression tests—treat it as the ground truth when modifying attention code.
- `third_party/DiffSynth-Studio/` is vendored upstream DiffSynth; touch only when coordinating an upstream sync.
- `docs/`, `metadata/`, `assets/`, and `test_output/` store experiment notes, CSVs, and qualitative artefacts referenced by the pipelines.

## Environment Setup & Build
- Use Python ≥3.8 with CUDA 12.x GPUs; the Triton kernels rely on CUDA-aligned toolchains.
- Install dependencies inside a fresh environment: `pip install -r requirements.txt && pip install -e .`.
- Extend `PYTHONPATH` for vendored modules: `export PYTHONPATH=$PWD/third_party/DiffSynth-Studio:$PYTHONPATH`.
- Keep large checkpoints external and feed their paths through CLI flags (see defaults in `scripts/train.sh`).

## Build, Test, and Development Commands
- `bash scripts/train.sh --help` lists reproducible training toggles; prefer this wrapper for distributed runs.
- `python src/train_recammaster.py --dataset_path /data --output_path ./models/train` starts the Lightning trainer with PRoPE rotary injection.
- `python src/inference_recammaster.py --dataset_path /data --ckpt_path ./models/ReCamMaster/checkpoints/step20000.ckpt` writes evaluation videos to `--output_dir`.
- `python delta_prope_tests/test_delta_rope.py` or `python delta_prope_tests/custom_fla.py` verifies Δ-RoPE math against the PyTorch baseline.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, `snake_case` functions, `CamelCase` classes, and concise module docstrings.
- Mirror the explicit seeding and CLI patterns in `train_recammaster.py`; expose new flags via both the Python script and `scripts/train.sh`.
- Avoid editing `third_party/` unless a vendor bump is coordinated; document local patches in `docs/`.

## Testing Guidelines
- Place new tests alongside existing ones in `delta_prope_tests/` and name them `test_<feature>.py`.
- Use `pytest delta_prope_tests/test_delta_rope.py -k <pattern>` for fast iteration; run the full scripts on GPU hardware before submitting PRs.
- When training changes affect logging or decoding, capture short artefacts in `test_output/` and clean them up inside helper scripts.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) as shown in `COMMIT_INFO.md`; keep scopes meaningful.
- Update documentation, metadata, or configs alongside code that depends on them.
- PRs should include a summary, the exact command(s) executed, links to WANDB runs or metrics, and representative frames/videos; call out dataset or checkpoint requirements explicitly.
