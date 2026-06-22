"""Core: batch runner for strategy/scenario seed sweeps."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.engine import create_engine
from src.core.metrics import build_run_summary_from_steps
from src.utils.config_loader import load_config
from src.utils.hardware import generate_seeds
from src.utils.outputs import OutputPaths, output_paths_from_root


def _sort_jobs_for_chunk_worker(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(jobs)


def _run_pack_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one homogeneous pack of jobs in a worker process."""
    paths = output_paths_from_root(payload["paths_root"])
    base_cfg = deepcopy(payload["base_cfg"])
    pack_jobs = list(payload["pack_jobs"])
    rows_per_engine = int(payload["rows_per_engine"])
    rows_per_seed = int(payload["rows_per_seed"])
    engine_strategy = str(payload["engine_strategy"])
    scenario = str(payload["scenario"])
    n_drones_by_scenario = dict(payload.get("n_drones_by_scenario", {}))
    write_step_metrics = bool(payload.get("write_step_metrics", True))

    run_summaries: list[dict[str, Any]] = []
    step_tables: list[pd.DataFrame] = []
    timing_totals = BatchRunner._empty_timing()

    for i in range(0, len(pack_jobs), rows_per_engine):
        chunk = _sort_jobs_for_chunk_worker(pack_jobs[i:i + rows_per_engine])
        chunk_cfg = deepcopy(base_cfg)
        chunk_cfg["simulation"]["batch_size"] = len(chunk)

        engine = create_engine(
            chunk_cfg,
            engine_strategy,
            str(scenario),
            paths=paths,
        )

        run_layout = []
        chunk_seeds = []
        for local_idx, job in enumerate(chunk):
            n_active = int(n_drones_by_scenario.get(
                str(job["scenario"]), int(job["n_drones"])))
            run_layout.append({
                "run_id": str(job["run_id"]),
                "strategy": str(job["strategy"]),
                "scenario": str(job["scenario"]),
                "seed": int(job["seed"]),
                "n_active_agents": int(n_active),
                "preset": str(job["preset"]),
                "n_drones": int(job["n_drones"]),
                "max_steps": int(job["max_steps"]),
                "run_max_steps": int(job["max_steps"]),
                "grid_size": int(job["grid_size"]),
                "config_id": str(job["config_id"]),
                "start": int(local_idx),
                "end": int(local_idx + rows_per_seed),
            })
            chunk_seeds.append(int(job["seed"]))

        engine.configure_parallel_runs(run_layout)
        step_df = engine.run_experiment(seed=chunk_seeds, visualize=False)

        profile = getattr(engine, "last_profile", {})
        if profile:
            BatchRunner._accumulate_timing(
                timing_totals,
                {
                    "engine_total_s": float(profile.get("run_total_s", 0.0)),
                    "engine_reset_s": float(profile.get("reset_s", 0.0)),
                    "engine_loop_s": float(profile.get("loop_s", 0.0)),
                    "engine_step_compute_s": float(profile.get("step_compute_s", 0.0)),
                    "engine_metrics_s": float(profile.get("metrics_s", 0.0)),
                    "engine_record_s": float(profile.get("record_s", 0.0)),
                    "engine_other_loop_s": float(profile.get("other_loop_s", 0.0)),
                },
            )

        summaries = engine.metrics.get_run_summaries()
        if summaries:
            run_summaries.extend(summaries)

        if write_step_metrics and not step_df.empty:
            step_tables.append(step_df.copy())

    return {
        "run_summaries": run_summaries,
        "step_tables": step_tables,
        "timing": timing_totals,
        "pack_meta": payload["pack_meta"],
    }


class BatchRunner:
    """Executes full experiment suites and saves machine-readable results."""

    # Ceiling for packed parallel simulation rows in batch mode.
    # BatchRunner uses one row per seed and sets
    # effective_batch = min(iterations, max_parallel_simulations) per chunk.
    DEFAULT_MAX_PARALLEL_SIMULATIONS = 4096
    DEFAULT_MULTIPROCESS_WORKERS = 0
    WRITE_STEP_METRICS = True
    WRITE_PER_RUN_STEP_FILES = False
    WRITE_TIMING_REPORT = True

    def __init__(
        self,
        config: dict,
        paths: OutputPaths,
        default_preset: str | None = None
    ):
        self.cfg = config
        self.paths = paths
        self.default_preset = default_preset

    @staticmethod
    def _clean_token(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())

    def _run_id(
        self,
        strategy: str,
        scenario: str,
        seed: int,
        config_id: str = "default",
    ) -> str:
        k = self._clean_token(config_id)
        s = self._clean_token(strategy)
        c = self._clean_token(scenario)
        return f"run_{k}_{s}_{c}_seed{seed}"

    @staticmethod
    def _iter_chunks(values: list[Any], size: int):
        for i in range(0, len(values), size):
            yield values[i:i + size]

    @staticmethod
    def _write_table(df: pd.DataFrame, path_stem: Path) -> tuple[Path, float]:
        t0 = time.perf_counter()
        parquet_path = path_stem.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path, (time.perf_counter() - t0)
        except Exception as exc:
            csv_path = path_stem.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(
                f"⚠️ Parquet write failed ({exc}). "
                f"Fell back to CSV: {csv_path}"
            )
            return csv_path, (time.perf_counter() - t0)

    @staticmethod
    def _read_table(path_stem: Path) -> pd.DataFrame:
        parquet_path = path_stem.with_suffix(".parquet")
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception as exc:
                print(f"⚠️ Failed reading {parquet_path}: {exc}")
        csv_path = path_stem.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    @staticmethod
    def _ordered_unique_ints(values: pd.Series) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for value in values.tolist():
            seed = int(value)
            if seed in seen:
                continue
            seen.add(seed)
            ordered.append(seed)
        return ordered

    @staticmethod
    def _dedupe_table(
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        subset: list[str] | None,
    ) -> pd.DataFrame:
        parts = []
        if existing_df is not None and not existing_df.empty:
            parts.append(existing_df.copy())
        if new_df is not None and not new_df.empty:
            parts.append(new_df.copy())
        if not parts:
            return pd.DataFrame()
        merged = pd.concat(parts, ignore_index=True, sort=False)
        if not subset:
            return merged
        if any(col not in merged.columns for col in subset):
            return merged
        return (
            merged.drop_duplicates(subset=subset, keep="last")
            .reset_index(drop=True)
        )

    @classmethod
    def _manifest_dedupe_columns(cls, df: pd.DataFrame) -> list[str]:
        if "run_id" in df.columns:
            return ["run_id"]
        cols = [
            "strategy",
            "scenario",
            "seed",
            "preset",
            "n_drones",
            "max_steps",
            "grid_size",
            "config_id",
        ]
        return [col for col in cols if col in df.columns]

    def _ensure_manifest_run_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "run_id" in df.columns:
            return df
        required = {"strategy", "scenario", "seed", "config_id"}
        if not required.issubset(df.columns):
            return df
        out = df.copy()
        out["run_id"] = [
            self._run_id(
                str(strategy),
                str(scenario),
                int(seed),
                str(config_id),
            )
            for strategy, scenario, seed, config_id in zip(
                out["strategy"],
                out["scenario"],
                out["seed"],
                out["config_id"],
            )
        ]
        return out

    @staticmethod
    def _empty_timing() -> dict[str, float]:
        return {
            "engine_total_s": 0.0,
            "engine_reset_s": 0.0,
            "engine_loop_s": 0.0,
            "engine_step_compute_s": 0.0,
            "engine_metrics_s": 0.0,
            "engine_record_s": 0.0,
            "engine_other_loop_s": 0.0,
            "io_write_s": 0.0,
        }

    @staticmethod
    def _timing_delta(end: dict[str, float], start: dict[str, float]) -> dict[str, float]:
        keys = set(start.keys()) | set(end.keys())
        return {k: float(end.get(k, 0.0) - start.get(k, 0.0)) for k in keys}

    @staticmethod
    def _accumulate_timing(dst: dict[str, float], src: dict[str, float]) -> None:
        for key, val in src.items():
            dst[key] = float(dst.get(key, 0.0) + float(val))

    @staticmethod
    def _print_timing_summary(prefix: str, timing: dict[str, float]) -> None:
        total = float(timing.get("engine_total_s", 0.0) +
                      timing.get("io_write_s", 0.0))
        if total <= 0:
            print(f"{prefix} timing unavailable")
            return

        engine_total = float(timing.get("engine_total_s", 0.0))
        reset_s = float(timing.get("engine_reset_s", 0.0))
        loop_s = float(timing.get("engine_loop_s", 0.0))
        step_s = float(timing.get("engine_step_compute_s", 0.0))
        metrics_s = float(timing.get("engine_metrics_s", 0.0))
        record_s = float(timing.get("engine_record_s", 0.0))
        other_s = float(timing.get("engine_other_loop_s", 0.0))
        io_s = float(timing.get("io_write_s", 0.0))

        def pct(x: float) -> float:
            return (100.0 * x / total) if total > 0 else 0.0

        print(
            f"{prefix} total={total:.2f}s "
            f"engine={engine_total:.2f}s ({pct(engine_total):.1f}%) "
            f"io={io_s:.2f}s ({pct(io_s):.1f}%)"
        )
        print(
            f"  engine breakdown: reset={reset_s:.2f}s loop={loop_s:.2f}s "
            f"step={step_s:.2f}s metrics={metrics_s:.2f}s "
            f"record={record_s:.2f}s other={other_s:.2f}s"
        )

    def _resolve_parallel_groups(
        self,
        n_seeds: int,
        max_parallel_simulations: int,
    ) -> int:
        if n_seeds <= 0:
            raise ValueError("n_seeds must be > 0")
        if max_parallel_simulations <= 0:
            raise ValueError("simulation.max_parallel_simulations must be > 0")
        return max(1, min(int(n_seeds), int(max_parallel_simulations)))

    def _resolve_n_active_agents(self, scenario: str, default: int) -> int:
        overrides = self.cfg.get('batch_run', {}).get(
            'n_drones_by_scenario', {})
        if isinstance(overrides, dict) and scenario in overrides:
            return int(overrides[scenario])
        return default

    @staticmethod
    def _as_list(value: Any, fallback: list[Any]) -> list[Any]:
        if value is None:
            return list(fallback)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _build_config_variants(self) -> list[dict[str, Any]]:
        batch_cfg = self.cfg.get("batch_run", {})
        base_preset = self.default_preset or "fast"

        preset_names = [str(x) for x in self._as_list(
            batch_cfg.get("presets"), [base_preset])]
        n_drones_override = batch_cfg.get("n_drones")
        max_steps_override = batch_cfg.get("max_steps")
        grid_size_override = batch_cfg.get("grid_size")

        runtime_cfg = deepcopy(self.cfg.get("runtime", {}))
        detected_device = self.cfg["simulation"]["device"]

        variants: list[dict[str, Any]] = []
        for preset_name in preset_names:
            # Load each preset once and use its values as defaults.
            preset_cfg = load_config(preset=str(preset_name))
            preset_default_n = int(preset_cfg["swarm"]["n_drones"])
            preset_default_steps = int(preset_cfg["simulation"]["max_steps"])
            preset_default_grid = int(preset_cfg["simulation"]["grid_size"])

            n_drones_vals = [int(x) for x in self._as_list(
                n_drones_override, [preset_default_n])]
            max_steps_vals = [int(x) for x in self._as_list(
                max_steps_override, [preset_default_steps])]
            grid_size_vals = [int(x) for x in self._as_list(
                grid_size_override, [preset_default_grid])]

            if not n_drones_vals:
                n_drones_vals = [preset_default_n]
            if not max_steps_vals:
                max_steps_vals = [preset_default_steps]
            if not grid_size_vals:
                grid_size_vals = [preset_default_grid]

            for n_drones, max_steps, grid_size in product(
                n_drones_vals, max_steps_vals, grid_size_vals
            ):
                variant_cfg = deepcopy(preset_cfg)
                variant_cfg.setdefault("runtime", {})
                if runtime_cfg:
                    variant_cfg["runtime"] = deepcopy(runtime_cfg)
                variant_cfg["simulation"]["device"] = detected_device
                variant_cfg["swarm"]["n_drones"] = int(n_drones)
                variant_cfg["simulation"]["max_steps"] = int(max_steps)
                variant_cfg["simulation"]["grid_size"] = int(grid_size)

                config_id = (
                    f"preset={preset_name}"
                    f"|n_drones={int(n_drones)}"
                    f"|max_steps={int(max_steps)}"
                    f"|grid_size={int(grid_size)}"
                )
                variants.append({
                    "config": variant_cfg,
                    "config_id": config_id,
                    "preset": str(preset_name),
                    "n_drones": int(n_drones),
                    "max_steps": int(max_steps),
                    "grid_size": int(grid_size),
                })

        return variants

    def _pack_key(self, job: dict[str, Any]) -> tuple[Any, ...]:
        """
        Build a pack key for grouping runs into one engine launch.

        Packs are homogeneous in the dimensions that control tensor sizes and
        runtime length, so each engine launch only allocates what it needs.
        """
        return (
            str(job["strategy"]),
            str(job["scenario"]),
            str(job["preset"]),
            int(job["grid_size"]),
            int(job["n_drones"]),
            int(job["max_steps"]),
        )

    @staticmethod
    def _pack_sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        strategy, scenario, preset, grid_size, n_drones, max_steps = key
        return (int(n_drones), int(max_steps), str(preset), int(grid_size), str(strategy), str(scenario))

    @staticmethod
    def _sort_jobs_for_chunk(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(jobs)

    def _resolve_seed_schedule(
        self,
        master_seed: int,
        n_iters: int,
    ) -> tuple[list[int], pd.DataFrame, bool]:
        requested_seeds = [int(x)
                           for x in generate_seeds(master_seed, n_iters)]
        existing_seed_df = self._read_table(self.paths.data / "seed_schedule")
        if not existing_seed_df.empty:
            if "seed" not in existing_seed_df.columns:
                raise ValueError(
                    f"{self.paths.data / 'seed_schedule'} is missing required "
                    "'seed' column."
                )
            existing_seeds = self._ordered_unique_ints(
                existing_seed_df["seed"])
            if requested_seeds != existing_seeds:
                existing_master = None
                if "master_seed" in existing_seed_df.columns and not existing_seed_df.empty:
                    existing_master = int(
                        existing_seed_df["master_seed"].iloc[0])
                raise ValueError(
                    "Seed schedule mismatch for resumed batch run. "
                    f"Existing seeds ({len(existing_seeds)}) do not match the "
                    f"requested schedule ({len(requested_seeds)}). "
                    f"Requested master_seed={master_seed}; "
                    f"existing master_seed={existing_master}."
                )
            return existing_seeds, existing_seed_df.copy(), True

        existing_run_df = self._read_table(self.paths.data / "run_summary")
        if existing_run_df.empty:
            existing_step_df = self._read_table(
                self.paths.data / "step_metrics")
            if not existing_step_df.empty:
                existing_run_df = self._rebuild_run_summary_from_steps(
                    existing_step_df)

        if not existing_run_df.empty and "seed" in existing_run_df.columns:
            existing_seeds = sorted({int(x)
                                    for x in existing_run_df["seed"].tolist()})
            if sorted(requested_seeds) != existing_seeds:
                raise ValueError(
                    "Seed schedule mismatch for resumed batch run. "
                    f"Existing outputs imply seeds={existing_seeds}, "
                    f"but the requested schedule is {requested_seeds}."
                )
            inferred_df = pd.DataFrame({
                "master_seed": [master_seed] * len(existing_seeds),
                "seed": existing_seeds,
            })
            return existing_seeds, inferred_df, False

        return requested_seeds, pd.DataFrame({
            "master_seed": [master_seed] * len(requested_seeds),
            "seed": requested_seeds,
        }), False

    def _load_completed_run_ids(self) -> set[str]:
        existing_run_df = self._read_table(self.paths.data / "run_summary")
        if not existing_run_df.empty and "run_id" in existing_run_df.columns:
            return set(existing_run_df["run_id"].astype(str))
        existing_step_df = self._read_table(self.paths.data / "step_metrics")
        if not existing_step_df.empty and "run_id" in existing_step_df.columns:
            return set(existing_step_df["run_id"].astype(str))
        return set()

    def _rebuild_run_summary_from_steps(
        self,
        step_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if step_df.empty or "run_id" not in step_df.columns:
            return pd.DataFrame()

        summaries: list[dict[str, Any]] = []
        for run_id, run_steps in step_df.groupby("run_id", sort=False):
            run_steps = run_steps.copy()
            meta = run_steps.iloc[0]
            default_n_drones = meta["n_drones"] if "n_drones" in run_steps.columns else None
            default_run_max_steps = (
                meta["run_max_steps"]
                if "run_max_steps" in run_steps.columns
                else meta["max_steps"] if "max_steps" in run_steps.columns else None
            )
            summary = build_run_summary_from_steps(
                run_steps,
                sps=float(
                    meta["steps_per_second"]) if "steps_per_second" in run_steps.columns else 0.0,
                strategy=str(
                    meta["strategy"]) if "strategy" in run_steps.columns else "unknown",
                scenario=str(
                    meta["scenario"]) if "scenario" in run_steps.columns else "unknown",
                seed=int(meta["seed"]) if "seed" in run_steps.columns else 0,
                default_n_drones=default_n_drones,
                default_run_max_steps=default_run_max_steps,
            )
            summary["run_id"] = str(run_id)
            for col in (
                "preset",
                "n_drones",
                "max_steps",
                "run_max_steps",
                "grid_size",
                "config_id",
            ):
                if col in run_steps.columns:
                    summary[col] = meta[col]
            summaries.append(summary)
        return pd.DataFrame(summaries)

    def _flush_incremental_outputs(
        self,
        *,
        new_run_summaries: list[dict[str, Any]] | None = None,
        new_step_tables: list[pd.DataFrame] | None = None,
        new_timing_rows: list[dict[str, Any]] | None = None,
        checkpoint_row: dict[str, Any] | None = None,
    ) -> float:
        """Flush partial batch artifacts so resumes can continue after crashes."""
        io_total_s = 0.0

        if new_run_summaries:
            existing_summary_df = self._read_table(
                self.paths.data / "run_summary")
            new_summary_df = pd.DataFrame(new_run_summaries)
            merged_summary_df = self._dedupe_table(
                existing_summary_df,
                new_summary_df,
                ["run_id"],
            )
            if not merged_summary_df.empty:
                summary_path, io_s = self._write_table(
                    merged_summary_df,
                    self.paths.data / "run_summary",
                )
                io_total_s += io_s
                print(
                    f"Checkpoint flush: run summary ({len(new_summary_df)} new rows) -> "
                    f"{summary_path}"
                )

        if new_step_tables:
            new_steps_df = pd.concat(
                new_step_tables, ignore_index=True, sort=False)
            existing_steps_df = self._read_table(
                self.paths.data / "step_metrics")
            merged_steps_df = self._dedupe_table(
                existing_steps_df,
                new_steps_df,
                ["run_id", "step"],
            )
            if not merged_steps_df.empty:
                step_path, io_s = self._write_table(
                    merged_steps_df,
                    self.paths.data / "step_metrics",
                )
                io_total_s += io_s
                print(
                    f"Checkpoint flush: step metrics ({len(new_steps_df)} new rows) -> "
                    f"{step_path}"
                )

        if self.WRITE_TIMING_REPORT and new_timing_rows:
            existing_timing_df = self._read_table(
                self.paths.data / "batch_timing")
            timing_df = pd.concat(
                [existing_timing_df, pd.DataFrame(new_timing_rows)],
                ignore_index=True,
                sort=False,
            ) if not existing_timing_df.empty else pd.DataFrame(new_timing_rows)
            timing_path, io_s = self._write_table(
                timing_df,
                self.paths.data / "batch_timing",
            )
            io_total_s += io_s
            print(f"Checkpoint flush: timing rows -> {timing_path}")

        if checkpoint_row:
            existing_checkpoint_df = self._read_table(
                self.paths.data / "batch_flush_checkpoint"
            )
            checkpoint_df = pd.DataFrame([checkpoint_row])
            merged_checkpoint_df = pd.concat(
                [existing_checkpoint_df, checkpoint_df],
                ignore_index=True,
                sort=False,
            ) if not existing_checkpoint_df.empty else checkpoint_df
            checkpoint_path, io_s = self._write_table(
                merged_checkpoint_df,
                self.paths.data / "batch_flush_checkpoint",
            )
            io_total_s += io_s
            print(f"Checkpoint flush: manifest row -> {checkpoint_path}")

        return float(io_total_s)

    def run_suite(
        self,
        strategies: list[str] | None = None,
        scenarios: list[str] | None = None,
        override_existing: bool = False,
    ) -> bool:
        """Runs all strategy/scenario pairs against the same deterministic seeds."""
        batch_cfg = self.cfg.get("batch_run", {})
        strategies = [str(x) for x in self._as_list(
            strategies, batch_cfg.get("strategies") or ["always"])]
        scenarios = [str(x) for x in self._as_list(
            scenarios, batch_cfg.get("scenarios") or ["baseline"])]
        strategy_checkpoint_flush = bool(
            batch_cfg.get("strategy_checkpoint_flush", False)
        )
        if not strategies:
            raise ValueError("No strategies provided for batch run.")
        if not scenarios:
            raise ValueError("No scenarios provided for batch run.")
        mappo_required = {
            "mappo",
        }
        if any(strategy in mappo_required for strategy in strategies):
            runtime_cfg = self.cfg.get("runtime", {})
            model_path = runtime_cfg.get("mappo_model_path") if isinstance(runtime_cfg, dict) else None
            if not isinstance(model_path, str) or not model_path.strip():
                raise ValueError(
                    "runtime.mappo_model_path is required for batch runs that include mappo-backed strategies."
                )
            if not Path(model_path).exists():
                raise FileNotFoundError(f"runtime.mappo_model_path not found: {model_path}")

        n_iters = int(batch_cfg.get('iterations', 1))
        if n_iters <= 0:
            raise ValueError("batch_run.iterations must be > 0")
        master_seed = int(self.cfg['simulation']['seed'])
        seeds, seed_schedule_df, using_existing_seed_schedule = self._resolve_seed_schedule(
            master_seed=master_seed,
            n_iters=n_iters,
        )
        config_variants = self._build_config_variants()
        if not config_variants:
            raise ValueError(
                "No configuration variants resolved for batch run.")
        effective_presets = sorted(
            {str(v["preset"]) for v in config_variants}
        )
        print(f"Effective batch presets: {effective_presets}")

        jobs: list[dict[str, Any]] = []
        for strategy in strategies:
            for scenario in scenarios:
                for variant in config_variants:
                    for seed in seeds:
                        run_id = self._run_id(
                            str(strategy),
                            str(scenario),
                            int(seed),
                            str(variant["config_id"]),
                        )
                        jobs.append({
                            "run_id": run_id,
                            "strategy": str(strategy),
                            "scenario": str(scenario),
                            "preset": str(variant["preset"]),
                            "n_drones": int(variant["n_drones"]),
                            "max_steps": int(variant["max_steps"]),
                            "grid_size": int(variant["grid_size"]),
                            "config_id": str(variant["config_id"]),
                            "seed": int(seed),
                        })
        if not jobs:
            raise ValueError("Batch job list is empty.")

        completed_run_ids = self._load_completed_run_ids()
        existing_jobs = [
            job for job in jobs if str(job["run_id"]) in completed_run_ids
        ]
        if override_existing:
            pending_jobs = list(jobs)
        else:
            pending_jobs = [
                job for job in jobs if str(job["run_id"]) not in completed_run_ids
            ]

        max_parallel_simulations = int(
            batch_cfg.get(
                'max_parallel_simulations',
                self.cfg['simulation'].get(
                    'max_parallel_simulations',
                    self.DEFAULT_MAX_PARALLEL_SIMULATIONS
                )
            )
        )
        # Batch mode uses one simulation row per run.
        rows_per_seed = 1
        if pending_jobs:
            rows_per_engine = self._resolve_parallel_groups(
                n_seeds=len(pending_jobs),
                max_parallel_simulations=max_parallel_simulations,
            )
        else:
            rows_per_engine = 0
        effective_batch = rows_per_seed * rows_per_engine
        if pending_jobs and rows_per_engine < len(pending_jobs):
            print(
                "Parallel groups clamped by simulation.max_parallel_simulations: "
                f"requested_rows={len(pending_jobs)}, used_rows={rows_per_engine}"
            )

        total_runs = len(jobs)
        total_pending = len(pending_jobs)
        print(
            f"Batch plan: {len(strategies)} strategies x {len(scenarios)} scenarios "
            f"x {len(seeds)} seeds x {len(config_variants)} config variants "
            f"= {total_runs} runs"
        )
        print(
            f"Resume status: requested={total_runs} "
            f"{'override_existing' if override_existing else 'skipped_existing'}={len(existing_jobs)} "
            f"pending={len(pending_jobs)}"
        )
        print(
            "Seed schedule source: "
            f"{'existing output directory' if using_existing_seed_schedule else 'requested batch config'}"
        )
        print(
            f"Master seed: {master_seed} | Seed preview: {seeds[:min(5, len(seeds))]}"
        )
        print(
            "Strategy checkpoint flushing: "
            f"{'enabled' if strategy_checkpoint_flush else 'disabled'}"
        )
        if pending_jobs:
            print(
                f"Parallel packing: {rows_per_engine} rows/engine "
                f"(rows_per_seed={rows_per_seed}, effective_batch={effective_batch}, "
                f"max_parallel={max_parallel_simulations})"
            )
        else:
            print("No pending batch jobs. Existing output already covers this request.")
            return False
        if override_existing and existing_jobs:
            print("Existing run_ids will be rerun and replaced in merged outputs.")

        grouped_jobs: dict[tuple[Any, ...],
                           list[dict[str, Any]]] = defaultdict(list)
        for job in pending_jobs:
            grouped_jobs[self._pack_key(job)].append(job)

        run_summaries: list[dict] = []
        step_tables: list[pd.DataFrame] = []
        timing_totals = self._empty_timing()
        timing_rows: list[dict] = []

        runtime_cfg = deepcopy(self.cfg.get("runtime", {}))
        detected_device = self.cfg["simulation"]["device"]
        multiprocess_engines = bool(
            batch_cfg.get("multiprocess_engines", False)
        )
        multiprocess_workers = int(
            batch_cfg.get(
                "multiprocess_workers",
                self.DEFAULT_MULTIPROCESS_WORKERS,
            )
        )
        allow_accelerator_multiprocess = bool(
            batch_cfg.get("multiprocess_allow_accelerator", False)
        )
        if multiprocess_engines and str(detected_device).lower() != "cpu":
            if allow_accelerator_multiprocess:
                print(
                    "Multiprocess engine mode requested on accelerator device "
                    f"'{detected_device}' with explicit override enabled. "
                    "Proceeding may increase memory pressure or contention."
                )
            else:
                print(
                    "Multiprocess engine mode requested, but simulation.device is "
                    f"'{detected_device}'. Falling back to single-process mode. "
                    "Set batch_run.multiprocess_allow_accelerator=true to override."
                )
                multiprocess_engines = False

        run_idx = 0

        if multiprocess_engines and grouped_jobs:
            pack_payloads: list[dict[str, Any]] = []
            for key in sorted(
                grouped_jobs.keys(),
                key=self._pack_sort_key,
            ):
                strategy, scenario, preset, grid_size, n_drones_key, max_steps_key = key
                engine_strategy = str(strategy)
                strategy_names = [str(strategy)]
                timing_strategy = str(strategy)

                pack_jobs = self._sort_jobs_for_chunk(
                    grouped_jobs[key],
                )
                base_cfg = load_config(preset=str(preset))
                base_cfg.setdefault("runtime", {})
                if runtime_cfg:
                    base_cfg["runtime"] = deepcopy(runtime_cfg)
                base_cfg["simulation"]["device"] = detected_device
                base_cfg["simulation"]["grid_size"] = int(grid_size)
                base_cfg["simulation"]["max_steps"] = int(max_steps_key)
                base_cfg["swarm"]["n_drones"] = int(n_drones_key)

                pack_payloads.append({
                    "paths_root": str(self.paths.root),
                    "base_cfg": base_cfg,
                    "pack_jobs": pack_jobs,
                    "rows_per_engine": int(rows_per_engine),
                    "rows_per_seed": int(rows_per_seed),
                    "engine_strategy": engine_strategy,
                    "scenario": str(scenario),
                    "n_drones_by_scenario": dict(
                        self.cfg.get("batch_run", {}).get(
                            "n_drones_by_scenario", {})
                    ),
                    "write_step_metrics": bool(self.WRITE_STEP_METRICS),
                    "pack_meta": {
                        "timing_strategy": timing_strategy,
                        "scenario": str(scenario),
                        "preset": str(preset),
                        "n_drones": int(n_drones_key),
                        "max_steps": int(max_steps_key),
                        "grid_size": int(grid_size),
                        "strategy_names": strategy_names,
                    },
                })

            if len(pack_payloads) <= 1:
                multiprocess_engines = False
            else:
                cpu_cap = os.cpu_count() or 1
                if multiprocess_workers <= 0:
                    worker_count = min(cpu_cap, len(pack_payloads))
                else:
                    worker_count = max(1, min(multiprocess_workers, cpu_cap))

                print(
                    "Multiprocess engine mode: enabled "
                    f"(workers={worker_count}, packs={len(pack_payloads)})"
                )
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(_run_pack_worker, payload): payload
                        for payload in pack_payloads
                    }
                    for future in as_completed(future_map):
                        result = future.result()
                        pack_meta = result["pack_meta"]
                        pack_summaries = list(result["run_summaries"])
                        pack_df = pd.DataFrame(pack_summaries)
                        run_summaries.extend(pack_summaries)
                        run_idx += len(pack_summaries)

                        if self.WRITE_STEP_METRICS:
                            step_tables.extend(result["step_tables"])

                        pack_timing = dict(result["timing"])
                        self._accumulate_timing(timing_totals, pack_timing)

                        strategy_tag = ",".join(pack_meta["strategy_names"])
                        pack_tag = (
                            f"n_drones={pack_meta['n_drones']} max_steps={pack_meta['max_steps']} "
                            f"preset={pack_meta['preset']} grid_size={pack_meta['grid_size']} "
                            f"strategies={strategy_tag} scenario={pack_meta['scenario']}"
                        )
                        if not pack_df.empty:
                            avg = pack_df[[
                                "fuel_preserved_frac", "fuel_loss_frac",
                                "active_exposure_frac", "efficiency"
                            ]].mean()
                            print(
                                f"[{run_idx:03d}/{total_pending:03d}] {pack_tag} "
                                f"pack_runs={len(pack_df)} "
                                f"avg_fuel_preserved={avg['fuel_preserved_frac']:.3f} "
                                f"avg_fuel_loss={avg['fuel_loss_frac']:.3f} "
                                f"avg_active_exposure={avg['active_exposure_frac']:.3f} "
                                f"avg_efficiency={avg['efficiency']:.3f}"
                            )
                        self._print_timing_summary(
                            prefix=f"Timing {pack_tag}",
                            timing=pack_timing,
                        )
                        timing_row = {
                            "strategy": pack_meta["timing_strategy"],
                            "scenario": str(pack_meta["scenario"]),
                            "preset": str(pack_meta["preset"]),
                            "n_drones": int(pack_meta["n_drones"]),
                            "max_steps": int(pack_meta["max_steps"]),
                            "grid_size": int(pack_meta["grid_size"]),
                            "config_id": "__pack__",
                            "runs": int(len(pack_df)),
                            "strategy_count_in_pack": int(len(pack_meta["strategy_names"])),
                            "strategy_list": strategy_tag,
                            "parallel_groups": int(rows_per_engine),
                            "rows_per_seed": int(rows_per_seed),
                            "effective_batch": int(effective_batch),
                            "max_parallel_simulations": int(max_parallel_simulations),
                            **pack_timing,
                        }
                        if strategy_checkpoint_flush:
                            checkpoint_io_s = self._flush_incremental_outputs(
                                new_run_summaries=pack_summaries,
                                new_step_tables=(
                                    result["step_tables"]
                                    if self.WRITE_STEP_METRICS else []
                                ),
                                new_timing_rows=[timing_row],
                                checkpoint_row={
                                    "strategy": timing_row["strategy"],
                                    "scenario": timing_row["scenario"],
                                    "preset": timing_row["preset"],
                                    "n_drones": timing_row["n_drones"],
                                    "max_steps": timing_row["max_steps"],
                                    "grid_size": timing_row["grid_size"],
                                    "runs": timing_row["runs"],
                                    "mode": "multiprocess",
                                    "flushed_at_unix_s": float(time.time()),
                                },
                            )
                            timing_totals["io_write_s"] += checkpoint_io_s
                        else:
                            timing_rows.append(timing_row)

                # Prevent sequential path from rerunning packs.
                grouped_jobs = defaultdict(list)

        for key in sorted(
            grouped_jobs.keys(),
            key=self._pack_sort_key,
        ):
            strategy, scenario, preset, grid_size, n_drones_key, max_steps_key = key
            engine_strategy = str(strategy)
            pack_jobs = list(grouped_jobs[key])
            strategy_names = [str(strategy)]
            strategy_tag = str(strategy)
            timing_strategy = str(strategy)
            group_max_steps = int(max_steps_key)
            group_max_drones = int(n_drones_key)

            pack_tag = (
                f"n_drones={group_max_drones} max_steps={group_max_steps} "
                f"preset={preset} grid_size={grid_size} "
                f"strategies={strategy_tag} scenario={scenario}"
            )

            pack_run_start = len(run_summaries)
            pack_step_start = len(step_tables)
            pack_timing_start = dict(timing_totals)

            base_cfg = load_config(preset=str(preset))
            base_cfg.setdefault("runtime", {})
            if runtime_cfg:
                base_cfg["runtime"] = deepcopy(runtime_cfg)
            base_cfg["simulation"]["device"] = detected_device
            base_cfg["simulation"]["grid_size"] = int(grid_size)
            base_cfg["simulation"]["max_steps"] = int(group_max_steps)
            base_cfg["swarm"]["n_drones"] = int(group_max_drones)

            for chunk in self._iter_chunks(pack_jobs, rows_per_engine):
                chunk = self._sort_jobs_for_chunk(list(chunk))
                chunk_cfg = deepcopy(base_cfg)
                chunk_cfg["simulation"]["batch_size"] = len(chunk)
                engine = create_engine(
                    chunk_cfg,
                    engine_strategy,
                    str(scenario),
                    paths=self.paths
                )

                run_layout = []
                chunk_seeds = []
                for local_idx, job in enumerate(chunk):
                    n_active = self._resolve_n_active_agents(
                        str(job["scenario"]),
                        int(job["n_drones"])
                    )
                    run_layout.append({
                        "run_id": str(job["run_id"]),
                        "strategy": str(job["strategy"]),
                        "scenario": str(job["scenario"]),
                        "seed": int(job["seed"]),
                        "n_active_agents": int(n_active),
                        "preset": str(job["preset"]),
                        "n_drones": int(job["n_drones"]),
                        "max_steps": int(job["max_steps"]),
                        "run_max_steps": int(job["max_steps"]),
                        "grid_size": int(job["grid_size"]),
                        "config_id": str(job["config_id"]),
                        "start": int(local_idx),
                        "end": int(local_idx + rows_per_seed),
                    })
                    chunk_seeds.append(int(job["seed"]))

                engine.configure_parallel_runs(run_layout)
                step_df = engine.run_experiment(
                    seed=chunk_seeds, visualize=False)
                profile = getattr(engine, "last_profile", {})
                if profile:
                    self._accumulate_timing(
                        timing_totals,
                        {
                            "engine_total_s": float(profile.get("run_total_s", 0.0)),
                            "engine_reset_s": float(profile.get("reset_s", 0.0)),
                            "engine_loop_s": float(profile.get("loop_s", 0.0)),
                            "engine_step_compute_s": float(profile.get("step_compute_s", 0.0)),
                            "engine_metrics_s": float(profile.get("metrics_s", 0.0)),
                            "engine_record_s": float(profile.get("record_s", 0.0)),
                            "engine_other_loop_s": float(profile.get("other_loop_s", 0.0)),
                        }
                    )

                summaries = engine.metrics.get_run_summaries()
                if summaries:
                    run_summaries.extend(summaries)
                    run_idx += len(summaries)
                    chunk_summary_df = pd.DataFrame(summaries)
                    chunk_avg = chunk_summary_df[[
                        "fuel_preserved_frac", "fuel_loss_frac",
                        "active_exposure_frac", "efficiency"
                    ]].mean()
                    print(
                        f"[{run_idx:03d}/{total_pending:03d}] {pack_tag} "
                        f"chunk_runs={len(summaries)} "
                        f"avg_fuel_preserved={chunk_avg['fuel_preserved_frac']:.3f} "
                        f"avg_fuel_loss={chunk_avg['fuel_loss_frac']:.3f} "
                        f"avg_active_exposure={chunk_avg['active_exposure_frac']:.3f} "
                        f"avg_efficiency={chunk_avg['efficiency']:.3f}"
                    )

                if self.WRITE_STEP_METRICS and not step_df.empty:
                    step_df = step_df.copy()
                    step_tables.append(step_df)
                    if self.WRITE_PER_RUN_STEP_FILES:
                        for run_id, run_steps in step_df.groupby("run_id"):
                            run_path, io_s = self._write_table(
                                run_steps,
                                self.paths.data / str(run_id)
                            )
                            _ = run_path
                            timing_totals["io_write_s"] += io_s

            pack_df = pd.DataFrame(run_summaries[pack_run_start:])
            pack_timing = self._timing_delta(timing_totals, pack_timing_start)
            if not pack_df.empty:
                avg = pack_df[[
                    "fuel_preserved_frac", "fuel_loss_frac",
                    "active_exposure_frac", "efficiency"
                ]].mean()
                print(
                    f"Summary {pack_tag}: n={len(pack_df)} "
                    f"fuel_preserved={avg['fuel_preserved_frac']:.3f} "
                    f"fuel_loss={avg['fuel_loss_frac']:.3f} "
                    f"active_exposure={avg['active_exposure_frac']:.3f} "
                    f"efficiency={avg['efficiency']:.3f}"
                )
            self._print_timing_summary(
                prefix=f"Timing {pack_tag}",
                timing=pack_timing
            )
            timing_row = {
                "strategy": timing_strategy,
                "scenario": str(scenario),
                "preset": str(preset),
                "n_drones": int(group_max_drones),
                "max_steps": int(group_max_steps),
                "grid_size": int(grid_size),
                "config_id": "__pack__",
                "runs": int(len(pack_df)),
                "strategy_count_in_pack": int(len(strategy_names)),
                "strategy_list": ",".join(strategy_names),
                "parallel_groups": int(rows_per_engine),
                "rows_per_seed": int(rows_per_seed),
                "effective_batch": int(effective_batch),
                "max_parallel_simulations": int(max_parallel_simulations),
                **pack_timing,
            }
            if strategy_checkpoint_flush:
                checkpoint_io_s = self._flush_incremental_outputs(
                    new_run_summaries=list(run_summaries[pack_run_start:]),
                    new_step_tables=(
                        list(step_tables[pack_step_start:])
                        if self.WRITE_STEP_METRICS else []
                    ),
                    new_timing_rows=[timing_row],
                    checkpoint_row={
                        "strategy": timing_row["strategy"],
                        "scenario": timing_row["scenario"],
                        "preset": timing_row["preset"],
                        "n_drones": timing_row["n_drones"],
                        "max_steps": timing_row["max_steps"],
                        "grid_size": timing_row["grid_size"],
                        "runs": timing_row["runs"],
                        "mode": "sequential",
                        "flushed_at_unix_s": float(time.time()),
                    },
                )
                timing_totals["io_write_s"] += checkpoint_io_s
            else:
                timing_rows.append(timing_row)

        existing_steps_df = self._read_table(self.paths.data / "step_metrics")
        existing_summary_df = self._read_table(self.paths.data / "run_summary")
        existing_manifest_df = self._ensure_manifest_run_id(
            self._read_table(self.paths.data / "batch_manifest")
        )
        if existing_summary_df.empty and not existing_steps_df.empty:
            existing_summary_df = self._rebuild_run_summary_from_steps(
                existing_steps_df)

        new_summary_df = pd.DataFrame(run_summaries)
        merged_summary_df = self._dedupe_table(
            existing_summary_df,
            new_summary_df,
            ["run_id"],
        )
        if not merged_summary_df.empty:
            summary_path, io_s = self._write_table(
                merged_summary_df,
                self.paths.data / "run_summary"
            )
            timing_totals["io_write_s"] += io_s
            print(f"Saved run summary: {summary_path}")

        new_steps_df = (
            pd.concat(step_tables, ignore_index=True, sort=False)
            if step_tables else pd.DataFrame()
        )
        merged_steps_df = self._dedupe_table(
            existing_steps_df,
            new_steps_df,
            ["run_id", "step"],
        )
        if not merged_steps_df.empty:
            step_path, io_s = self._write_table(
                merged_steps_df,
                self.paths.data / "step_metrics"
            )
            timing_totals["io_write_s"] += io_s
            print(f"Saved step metrics: {step_path}")

        requested_manifest_df = self._ensure_manifest_run_id(
            pd.DataFrame(jobs))
        manifest_subset = self._manifest_dedupe_columns(
            requested_manifest_df if not requested_manifest_df.empty else existing_manifest_df
        )
        merged_manifest_df = self._dedupe_table(
            existing_manifest_df,
            requested_manifest_df,
            manifest_subset,
        )
        manifest_path, io_s = self._write_table(
            merged_manifest_df,
            self.paths.data / "batch_manifest"
        )
        timing_totals["io_write_s"] += io_s
        print(f"Saved batch manifest: {manifest_path}")

        if using_existing_seed_schedule:
            print(
                f"Preserved seed schedule: {self.paths.data / 'seed_schedule'}")
        else:
            seed_path, io_s = self._write_table(
                seed_schedule_df,
                self.paths.data / "seed_schedule"
            )
            timing_totals["io_write_s"] += io_s
            print(f"Saved seed schedule: {seed_path}")

        if pending_jobs:
            self._print_timing_summary(
                prefix="Timing total", timing=timing_totals)
        if self.WRITE_TIMING_REPORT:
            existing_timing_df = self._read_table(
                self.paths.data / "batch_timing")
            rows_to_write: list[dict[str, Any]] = []
            if strategy_checkpoint_flush:
                if pending_jobs:
                    rows_to_write.append({
                        "strategy": "__all__",
                        "scenario": "__all__",
                        "preset": "__all__",
                        "n_drones": -1,
                        "max_steps": -1,
                        "grid_size": -1,
                        "config_id": "__all__",
                        "runs": int(len(run_summaries)),
                        "parallel_groups": int(rows_per_engine),
                        "rows_per_seed": int(rows_per_seed),
                        "effective_batch": int(effective_batch),
                        "max_parallel_simulations": int(max_parallel_simulations),
                        **timing_totals,
                    })
            else:
                rows_to_write.extend(timing_rows)
                rows_to_write.append({
                    "strategy": "__all__",
                    "scenario": "__all__",
                    "preset": "__all__",
                    "n_drones": -1,
                    "max_steps": -1,
                    "grid_size": -1,
                    "config_id": "__all__",
                    "runs": int(len(run_summaries)),
                    "parallel_groups": int(rows_per_engine),
                    "rows_per_seed": int(rows_per_seed),
                    "effective_batch": int(effective_batch),
                    "max_parallel_simulations": int(max_parallel_simulations),
                    **timing_totals,
                })

            if rows_to_write:
                timing_df = pd.concat(
                    [existing_timing_df, pd.DataFrame(rows_to_write)],
                    ignore_index=True,
                    sort=False,
                ) if not existing_timing_df.empty else pd.DataFrame(rows_to_write)
                timing_path, _ = self._write_table(
                    timing_df,
                    self.paths.data / "batch_timing"
                )
                print(f"Saved timing report: {timing_path}")

        return True
