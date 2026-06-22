# Robot-Assisted Evacuation (apoptotic-wildfire fork)

This is our **extended copy** of the Yanapay robot-assisted-evacuation toolkit.

Yanapay is a Docker-based wrapper around the `IMPACT+` agent-based evacuation
model (NetLogo). `IMPACT+` simulates evacuating a transport hub with a
search-and-rescue (SAR) robot that, on finding a fallen victim, decides whether
to ask a nearby passenger (zero-responder) or a staff member (first-responder)
for help.

## Relation to the original

The base toolkit and model are by Carlos Gavidia-Calderon et al. (Yanapay) and
van der Wal et al. (`IMPACT` / `IMPACT+`). Project repository:
<https://github.com/kangkelidis/robot-assisted-evacuation>.

What we added/changed for the wildfire paper:

- `workspace/src/wildfire_transfer.py` — bridges the wildfire participation-control
  logic into the evacuation SAR-robot's help-request decision.
- Participation controllers in `workspace/participation_controllers/` —
  `frustration` and `labella`, used to evaluate the wildfire-derived adaptation
  strategies in the evacuation domain (alongside the baseline `always`).
- `workspace/src/evacuation_paper_plotter.py` — produces the evacuation figures
  used in the paper.
- Batch/resume infrastructure (`experiment_checkpoint.py`, `retry_policy.py`,
  `simulation_manager.py`) for long sweeps.

For the full base-toolkit documentation (all configuration options, scenarios,
room types, and how to write new strategies), see the project repository above.

## Requirements

Docker — the simulation runs inside a container that bundles NetLogo and the
Python dependencies. Install Docker: <https://docs.docker.com/get-docker/>.

## Run

From this directory:

```bash
# Build the image locally (first time, or after changing the code)
./build-docker-image.sh

# Run the sweeps defined in workspace/config.json
./run-container.sh

# ...or build and run in one step
./build-and-run.sh
```

Add `hub` after the script name to use the prebuilt image from Docker Hub
instead of a local build, e.g. `./run-container.sh hub`.

Re-analyse or resume a saved run (FOLDER is a directory under
`workspace/results/`):

```bash
./run-container.sh [hub] --analyse FOLDER
./run-container.sh [hub] --resume FOLDER
```

## Configuration

`workspace/config.json` defines the scenarios. Each entry sets the participation
strategy (`always`, `frustration`, `labella`), the robot adaptation strategy, and
the robot counts. Edit it to change what is run.

## Plots

The paper figures are generated during analysis when `--paper-mode` is passed:

```bash
./run-container.sh [hub] --paper-mode standard --metric-variants mean
```

This release reproduces only the two figures used in the paper, written under the
run's `img/paper/` directory:

- `02_tradeoff_pareto_overall`
- `13_population_performance_cost_combined_overall`

## Results

Each run writes to `workspace/results/<timestamp>/` with `data/` (metrics),
`img/` (plots), and `video/` (optional recordings).

## License and citation

MIT — see [`LICENSE`](LICENSE). The original toolkit is © 2022 Carlos
Gavidia-Calderon; the apoptotic-wildfire extensions are © 2026 Alexandros
Kangkelidis. If you use this code, please cite the Yanapay paper (see
[`CITATION.cff`](CITATION.cff)) and the `IMPACT+` model, in addition to the
apoptotic-wildfire paper.
