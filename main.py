"""
Apoptotic Wildfire Swarm Simulator - Main Entry Point

MODES:
  --visualize (-v)  : Run single simulation with video recording
  --batch (-b)      : Run batch of simulations, full sweep (Strategy x Parameters).
  --analyze (-a)    : Analyze existing result files and generate plots
"""

import argparse
import gc
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from src.core.batch_runner import BatchRunner
from src.core.engine import create_engine
from src.utils.analysis import MultiExperimentAnalyzer
from src.utils.config_loader import load_config
from src.utils.hardware import detect_device, generate_seeds
from src.utils.outputs import (create_output_directory, find_latest_output,
                               find_output_by_timestamp,
                               output_paths_from_root)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apoptotic Wildfire Swarm Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Configuration Hierarchy:
                1. CLI arguments (highest priority)
                2. Preset from presets.yaml
                3. base.yaml (base configuration)

                Output Structure:
                Each run creates a timestamped directory:
                    outputs/YYYYMMDD_HHMMSS/
                    data/       - Metrics and analysis tables (Parquet, CSV fallback)
                    videos/     - Video recordings
                    plots/      - Analysis plots
                    summary.txt - Analysis report
                """,
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--batch", "-b", action="store_true",
                       help="Run batch combinations")
    group.add_argument("--analyze", "-a", action="store_true",
                       help="Run only post-simulation analysis")
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Enable visualization (works standalone or with batch/analyze)"
    )
    parser.add_argument("--_visualize_once", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_output_dir", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_video_tag", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_seed", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_n_drones", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_max_steps", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_grid_size", type=int, default=None,
                        help=argparse.SUPPRESS)

    parser.add_argument("--strategies", "-s", nargs="+",
                        default=None, help="List of strategies to test")
    parser.add_argument("--scenarios", "-c", nargs="+",
                        default=None, help="List of scenarios to test")
    parser.add_argument("--preset", "-p", type=str,
                        default=None, help="Detail preset")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "MAPPO checkpoint: an explicit .pt path, or a selector resolved under "
            "models/ (auto, latest, final). Omit to use the released checkpoint at "
            "artifacts/checkpoints/mappo_final.pt."
        ),
    )
    parser.add_argument("--no-auto-analysis", action="store_true",
                        help="Skip analysis after batch mode")
    parser.add_argument(
        "--exclude-presets",
        nargs="+",
        metavar="PRESET",
        default=None,
        help="Exclude runs with these presets from analysis (e.g. --exclude-presets fast)",
    )
    parser.add_argument(
        "--exclude-strategies",
        nargs="+",
        metavar="STRATEGY",
        default=None,
        help="Exclude runs with these strategies from analysis (e.g. --exclude-strategies never always)",
    )
    parser.add_argument(
        "--exclude-scenarios",
        nargs="+",
        metavar="SCENARIO",
        default=None,
        help="Exclude runs with these scenarios from analysis (e.g. --exclude-scenarios baseline)",
    )
    parser.add_argument(
        "--override-existing",
        action="store_true",
        help="In batch resume mode, rerun matching run_ids and replace existing rows instead of skipping them",
    )
    parser.add_argument(
        "--batch-multiprocess",
        action="store_true",
        help=(
            "Enable process-level parallel batch execution across independent "
            "engine packs (CPU-only recommended)."
        ),
    )
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=None,
        help=(
            "Worker count for --batch-multiprocess. "
            "If omitted, uses batch_run.multiprocess_workers from config."
        ),
    )
    parser.add_argument(
        "--batch-multiprocess-allow-accelerator",
        action="store_true",
        help=(
            "Allow --batch-multiprocess on non-CPU devices (mps/cuda). "
            "Use only if your machine is stable with accelerator contention."
        ),
    )
    parser.add_argument(
        "--batch-strategy-checkpoint-flush",
        action="store_true",
        help=(
            "Flush batch outputs after each strategy-pack completion so crashes "
            "resume from the last completed strategy."
        ),
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help=(
            "Existing output directory or timestamp. In analyze mode it selects "
            "the run to analyze; in batch mode it resumes/appends into that run."
        ),
    )

    return parser


def resolve_mappo_model_spec(spec: str) -> str:
    """
    Resolve a model selector into a checkpoint path.

    Supported selectors:
      - auto / recent       (most recently modified of latest/final)
      - latest / final
      - <folder>            (most recently modified in that folder)
      - <folder>/latest|final
      - explicit .pt path
    """
    def newest_checkpoint(latest_path: Path, final_path: Path) -> Path | None:
        candidates = [p for p in (latest_path, final_path) if p.exists()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))

    token = spec.strip()
    if not token:
        raise ValueError("Empty MAPPO model selector.")

    if token.endswith(".pt"):
        return token

    if token in ("auto", "recent"):
        latest = Path("models") / "mappo_latest.pt"
        final = Path("models") / "mappo_final.pt"
        chosen = newest_checkpoint(latest, final)
        if chosen is None:
            raise ValueError(
                "No MAPPO checkpoint found in models/. "
                f"Tried '{latest}' and '{final}'."
            )
        return str(chosen)

    if token in ("latest", "final"):
        return f"models/mappo_{token}.pt"

    if "/" in token:
        folder, variant = token.split("/", 1)
        if variant not in ("latest", "final"):
            raise ValueError(
                f"Invalid MAPPO selector '{spec}'. "
                "Use <folder>/latest or <folder>/final."
            )
        return f"models/{folder}/mappo_{variant}.pt"

    latest = Path("models") / token / "mappo_latest.pt"
    final = Path("models") / token / "mappo_final.pt"
    chosen = newest_checkpoint(latest, final)
    if chosen is not None:
        return str(chosen)

    raise ValueError(
        f"No MAPPO checkpoint found for folder '{token}'. "
        f"Tried '{latest}' and '{final}'."
    )


def _run_single_visualization(
    config: dict,
    strategy: str,
    scenario: str,
    seed: int,
    paths=None,
    video_tag: str | None = None,
) -> None:
    if paths is None:
        paths = create_output_directory()
    print(f"\n📁 Output directory: {paths.root}")
    print(f"\n🎬 VISUALIZE MODE ({strategy} | {scenario} | seed={seed})")

    vis_cfg = deepcopy(config)
    vis_cfg['simulation']['batch_size'] = 1
    if video_tag:
        vis_cfg.setdefault("runtime", {})
        vis_cfg["runtime"]["video_tag"] = video_tag
    engine = create_engine(vis_cfg, strategy, scenario, paths)
    engine.run_experiment(seed=seed, visualize=True)


def _spawn_visualization_subprocess(
    preset: str | None,
    strategy: str,
    scenario: str,
    model: str | None,
    output_dir: str | None = None,
    video_tag: str | None = None,
    seed: int | None = None,
    n_drones: int | None = None,
    max_steps: int | None = None,
    grid_size: int | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_visualize_once",
        "--strategies", strategy,
        "--scenarios", scenario,
    ]
    if preset:
        cmd.extend(["--preset", preset])
    if model:
        cmd.extend(["--model", model])
    if output_dir:
        cmd.extend(["--_output_dir", output_dir])
    if video_tag:
        cmd.extend(["--_video_tag", video_tag])
    if seed is not None:
        cmd.extend(["--_seed", str(int(seed))])
    if n_drones is not None:
        cmd.extend(["--_n_drones", str(int(n_drones))])
    if max_steps is not None:
        cmd.extend(["--_max_steps", str(int(max_steps))])
    if grid_size is not None:
        cmd.extend(["--_grid_size", str(int(grid_size))])
    subprocess.run(cmd, check=True)


def _resolve_output_paths(run_dir: str):
    raw = Path(run_dir).expanduser()
    if raw.exists() and raw.is_dir():
        return output_paths_from_root(raw.resolve())
    return find_output_by_timestamp(run_dir)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not (args.batch or args.analyze or args.visualize or args._visualize_once):
        parser.error(
            "Select at least one mode: --visualize, --batch, or --analyze.")

    visualize_only = args.visualize and not args.batch and not args.analyze and not args._visualize_once

    # Standalone visualize should use pure base config unless preset is explicitly provided.
    if visualize_only and args.preset is None:
        preset_to_use = None
        print("\nLoading configuration (base only)...")
    else:
        preset_to_use = args.preset or "fast"
        print("\nLoading configuration...")

    config = load_config(preset=preset_to_use)
    # Internal visualize overrides for exact batch-variant replay.
    if args._n_drones is not None:
        config["swarm"]["n_drones"] = int(args._n_drones)
    if args._max_steps is not None:
        config["simulation"]["max_steps"] = int(args._max_steps)
    if args._grid_size is not None:
        config["simulation"]["grid_size"] = int(args._grid_size)
    device = detect_device(config['simulation']['device'])
    config['simulation']['device'] = device
    print(f"Using device: {device.type.upper()}")
    if args.batch:
        if preset_to_use is None:
            print("Bootstrap preset: <base only>")
        else:
            print(f"Bootstrap preset: {preset_to_use}")
        batch_presets = config.get("batch_run", {}).get("presets")
        if batch_presets:
            print(f"Batch presets (effective): {list(batch_presets)}")
        else:
            fallback = preset_to_use or "fast"
            print(f"Batch presets (effective): [{fallback}]")
    else:
        if preset_to_use is None:
            print("Preset: <base only>")
        else:
            print(f"Preset: {preset_to_use}")

    if args.model is not None:
        model_path = resolve_mappo_model_spec(args.model)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MAPPO model not found: {model_path}"
            )
        config.setdefault("runtime", {})
        config["runtime"]["mappo_model_path"] = model_path
        print(f"MAPPO model: {model_path}")

    batch_cfg = config.get("batch_run", {})
    if visualize_only:
        # Do not consume batch_run defaults in standalone visualize mode.
        strategies = args.strategies or ["always"]
        scenarios = args.scenarios or ["baseline"]
    else:
        strategies = args.strategies or batch_cfg.get("strategies") or [
            "always"]
        scenarios = args.scenarios or batch_cfg.get("scenarios") or [
            "baseline"]
    # -------------------------------------------------------------------------
    # Internal Mode: Single visualize run (used by subprocess fan-out)
    # -------------------------------------------------------------------------
    if args._visualize_once:
        paths = output_paths_from_root(
            args._output_dir) if args._output_dir else None
        vis_seed = int(args._seed) if args._seed is not None else int(
            config['simulation']['seed'])
        _run_single_visualization(
            config=config,
            strategy=strategies[0],
            scenario=scenarios[0],
            seed=vis_seed,
            paths=paths,
            video_tag=args._video_tag,
        )
        return 0

    # -------------------------------------------------------------------------
    # Mode: Batch
    # -------------------------------------------------------------------------
    if args.batch:
        config.setdefault("batch_run", {})
        if args.batch_multiprocess:
            config["batch_run"]["multiprocess_engines"] = True
        if args.batch_workers is not None:
            config["batch_run"]["multiprocess_workers"] = int(
                args.batch_workers)
        if args.batch_multiprocess_allow_accelerator:
            config["batch_run"]["multiprocess_allow_accelerator"] = True
        if args.batch_strategy_checkpoint_flush:
            config["batch_run"]["strategy_checkpoint_flush"] = True

        strategy_checkpoint_flush = bool(
            config.get("batch_run", {}).get("strategy_checkpoint_flush", False)
        )

        if args.run_dir:
            paths = _resolve_output_paths(args.run_dir)
            if not paths:
                print(f"No output directory found matching '{args.run_dir}'")
                return 1
            print(f"\n📁 Resuming output directory: {paths.root}")
        else:
            paths = create_output_directory()
        print(f"\n📁 Output directory: {paths.root}")
        print("\n🔄 BATCH MODE")

        if strategy_checkpoint_flush and len(strategies) > 1:
            print(
                "Strategy-checkpoint mode: running one strategy at a time "
                "as isolated batch suites."
            )
            for idx, strategy in enumerate(strategies, start=1):
                print(
                    f"\n🧩 Strategy pass {idx}/{len(strategies)}: {strategy} "
                    f"(scenarios={scenarios})"
                )
                runner = BatchRunner(
                    config, paths, default_preset=preset_to_use)
                did_work = runner.run_suite(
                    [strategy],
                    scenarios,
                    override_existing=args.override_existing,
                )

                if not args.no_auto_analysis and did_work:
                    print(
                        "   Updating summary.txt (plots disabled for strategy pass)...")
                    analyzer = MultiExperimentAnalyzer(
                        paths,
                        exclude_presets=args.exclude_presets,
                        exclude_strategies=args.exclude_strategies,
                        exclude_scenarios=args.exclude_scenarios,
                    )
                    analyzer.analyze(generate_plots=False)

                # Release Python/accelerator caches between strategy passes.
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
        else:
            runner = BatchRunner(config, paths, default_preset=preset_to_use)
            did_work = runner.run_suite(
                strategies,
                scenarios,
                override_existing=args.override_existing,
            )

        if not args.no_auto_analysis:
            print("\n📊 Running post-batch analysis...")
            analyzer = MultiExperimentAnalyzer(
                paths,
                exclude_presets=args.exclude_presets,
                exclude_strategies=args.exclude_strategies,
                exclude_scenarios=args.exclude_scenarios,
            )
            analyzer.analyze()

        if args.visualize:
            print(
                "\n🎬 Rendering one video per strategy/scenario "
                "in subprocesses..."
            )
            variant_runner = BatchRunner(
                config,
                paths,
                default_preset=preset_to_use,
            )
            config_variants = variant_runner._build_config_variants()
            for strategy in strategies:
                for scenario in scenarios:
                    for variant in config_variants:
                        preset_name = str(variant["preset"])
                        n_drones = int(variant["n_drones"])
                        max_steps = int(variant["max_steps"])
                        grid_size = int(variant["grid_size"])
                        tag = (
                            f"{strategy}_{scenario}_preset-{preset_name}"
                            f"_n{n_drones}_t{max_steps}_g{grid_size}"
                        )
                        print(
                            "  - visualize "
                            f"strategy={strategy} scenario={scenario} "
                            f"preset={preset_name} n_drones={n_drones} "
                            f"max_steps={max_steps} grid={grid_size}"
                        )
                        try:
                            _spawn_visualization_subprocess(
                                preset=preset_name,
                                strategy=strategy,
                                scenario=scenario,
                                model=args.model,
                                output_dir=str(paths.root),
                                video_tag=tag,
                                n_drones=n_drones,
                                max_steps=max_steps,
                                grid_size=grid_size,
                            )
                        except subprocess.CalledProcessError as exc:
                            print(
                                "⚠️ Visualization subprocess failed for "
                                f"{strategy}/{scenario}/{preset_name}: {exc}"
                            )
                            continue

    # -------------------------------------------------------------------------
    # Mode: Analysis
    # -------------------------------------------------------------------------
    elif args.analyze:
        # Find the run directory to analyze
        if args.run_dir:
            paths = _resolve_output_paths(args.run_dir)
            if not paths:
                print(f"No run found matching '{args.run_dir}'")
                return 1
        else:
            # Use most recent run
            paths = find_latest_output()
            if not paths:
                print("No runs found in outputs/")
                return 1

        print(f"\n📁 Analyzing run: {paths.root}")
        analyzer = MultiExperimentAnalyzer(
            paths,
            exclude_presets=args.exclude_presets,
            exclude_strategies=args.exclude_strategies,
            exclude_scenarios=args.exclude_scenarios,
        )
        if not analyzer.analyze():
            return 1

        if args.visualize:
            strategy = strategies[0]
            scenario = scenarios[0]
            print(
                "\n🎬 Visualization requested with analysis. "
                f"Rendering one video for {strategy}/{scenario} in subprocess..."
            )
            try:
                _spawn_visualization_subprocess(
                    preset=preset_to_use,
                    strategy=strategy,
                    scenario=scenario,
                    model=args.model,
                    output_dir=str(paths.root),
                )
            except subprocess.CalledProcessError as exc:
                print(
                    "⚠️ Visualization subprocess failed for "
                    f"{strategy}/{scenario}: {exc}"
                )
                return 1

    # -------------------------------------------------------------------------
    # Mode: Visualize-only workflow
    # -------------------------------------------------------------------------
    elif args.visualize:
        # Standalone visualize mode: one run only (no batch_run, no analysis).
        master_seed = int(config['simulation']['seed'])
        first_seed = int(generate_seeds(master_seed, 1)[0])
        print(
            "\n🎬 VISUALIZE MODE | "
            f"strategy={strategies[0]} scenario={scenarios[0]} seed={first_seed}"
        )

        paths = create_output_directory()
        _run_single_visualization(
            config=config,
            strategy=strategies[0],
            scenario=scenarios[0],
            seed=first_seed,
            paths=paths,
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)
