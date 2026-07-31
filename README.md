# Apoptotic Wildfire

Simulator, controllers, and trained model for the wildfire participation-control
experiments.

The simulator models a swarm of firefighting drones suppressing a wildfire.

## Requirements

- Python 3.11 or newer
- The dependencies in `pyproject.toml` (PyTorch, NumPy, pandas, Matplotlib,
  OpenCV, etc.). A GPU is optional; CPU, CUDA, and Apple MPS are supported.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run a small simulation

Visualize one short run with the frustration strategy and the `fast` preset:

```bash
python main.py --visualize --preset fast --strategies frustration
```

Use the trained MAPPO policy instead (loads the released checkpoint, see below):

```bash
python main.py --visualize --preset fast --strategies mappo
```

## Run a batch and analysis

`--batch` runs the strategy/scenario grid from `config/`, writes per-run metrics,
and produces analysis plots. Start small with the `fast` preset:

```bash
python main.py --batch --preset fast
```

Results and plots are written under a run directory (ignored by git). Pass
`--analyze` to re-run analysis on existing results.

## Training

`train.py` trains the MAPPO policy. The released checkpoint used in the paper is
already provided under `artifacts/`, so training is not required to reproduce the
MAPPO runs.

For a quick test:

```bash
python train.py --n-drones 100 --train-batch-size 64 --max-steps 64 --updates 3
```

A full run uses the `training` preset defaults:

```bash
python train.py
```

Each update is logged to stdout and appended to `training_metrics.csv` in the
run's `outputs/<timestamp>_training/data/` directory. Checkpoints are written to
`models/` as `mappo_latest.pt` and `mappo_final.pt` (git-ignored); resume with
`python train.py --resume models/mappo_latest.pt`.

The post-training video render and batch sweep are off by default because they
are slow; enable them with `--final-video` and `--final-batch-analysis`. For
evaluation and paper plots, use `python main.py --batch` / `python main.py --analyze`.

## Artifacts and checkpoint

- `artifacts/checkpoints/mappo_final.pt` — the released MAPPO policy. It is the
  default checkpoint loaded by the `mappo` strategy. Override it per run with
  `--model <path>`. If the checkpoint is missing, the run fails with a clear
  error rather than falling back to random weights.
- `artifacts/paper.pdf` — the paper.


## Repository layout

- `main.py` — run simulations, visualization, batch experiments, and analysis
- `train.py` — train the MAPPO policy
- `src/` — simulator core, physics, swarm control, RL, and strategies
- `config/` — base configuration, detail presets, and scenarios
- `artifacts/` — released MAPPO checkpoint and the paper PDF
- `external/open-yanapay/` — copy of the extended Yanapay evacuation toolkit,
  used as a reference baseline (see its own README and licence)

## Citation

See [`CITATION.cff`](CITATION.cff).
