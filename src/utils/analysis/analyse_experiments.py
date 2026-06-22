"""Multi-experiment analysis for strategy/scenario comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.metrics import build_run_summary_from_steps
from src.utils.outputs import OutputPaths

from .paper_plotter import PaperPlotter


class MultiExperimentAnalyzer:
    """Loads experiment outputs, computes statistics, and generates reports."""

    FIRE_OUT_THRESHOLD = 1e-3
    GROUP_DIM_PRIORITY = [
        "strategy", "scenario", "preset",
        "n_drones", "max_steps", "grid_size", "config_id"
    ]

    def __init__(
        self,
        output_paths: OutputPaths,
        exclude_presets: list[str] | None = None,
        exclude_strategies: list[str] | None = None,
        exclude_scenarios: list[str] | None = None,
    ):
        self.paths = output_paths
        self.data_dir = output_paths.data
        self.exclude_presets: set[str] = set(
            exclude_presets) if exclude_presets else set()
        self.exclude_strategies: set[str] = set(
            exclude_strategies) if exclude_strategies else set()
        self.exclude_scenarios: set[str] = set(
            exclude_scenarios) if exclude_scenarios else set()
        self.paper_plotter = PaperPlotter(output_paths.plots)

    @staticmethod
    def _ci95(series: pd.Series) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        n = len(series)
        if n <= 1:
            return 0.0
        std = float(series.std(ddof=1))
        return 1.96 * (std / (n ** 0.5))

    @staticmethod
    def _read_table(path_stem: Path) -> pd.DataFrame:
        parquet = path_stem.with_suffix(".parquet")
        if parquet.exists():
            try:
                return pd.read_parquet(parquet)
            except Exception as exc:
                print(f"⚠️ Failed reading {parquet}: {exc}")

        csv = path_stem.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv)

        return pd.DataFrame()

    @staticmethod
    def _write_table(df: pd.DataFrame, path_stem: Path) -> Path:
        parquet = path_stem.with_suffix(".parquet")
        try:
            df.to_parquet(parquet, index=False)
            return parquet
        except Exception as exc:
            csv = path_stem.with_suffix(".csv")
            df.to_csv(csv, index=False)
            print(
                f"⚠️ Parquet write failed ({exc}). "
                f"Fell back to CSV: {csv}"
            )
            return csv

    def _load_run_summary(self) -> pd.DataFrame:
        return self._read_table(self.data_dir / "run_summary")

    def _load_step_metrics(self) -> pd.DataFrame:
        table = self._read_table(self.data_dir / "step_metrics")
        if not table.empty:
            return table

        candidates = sorted(self.data_dir.glob("run_*.parquet"))
        if not candidates:
            candidates = sorted(self.data_dir.glob("run_*.csv"))
        if not candidates:
            return pd.DataFrame()

        tables = []
        for file in candidates:
            try:
                if file.suffix == ".parquet":
                    tables.append(pd.read_parquet(file))
                else:
                    tables.append(pd.read_csv(file))
            except Exception:
                continue
        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

    @staticmethod
    def _merge_rebuilt_summary(
        run_df: pd.DataFrame,
        rebuilt_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if run_df.empty:
            return rebuilt_df.copy()
        if rebuilt_df.empty:
            return run_df.copy()

        match_cols = [c for c in (
            "run_id",) if c in run_df.columns and c in rebuilt_df.columns]
        if not match_cols:
            match_cols = [
                c for c in (
                    "strategy", "scenario", "seed", "preset",
                    "n_drones", "max_steps", "grid_size", "config_id"
                )
                if c in run_df.columns and c in rebuilt_df.columns
            ]
        if not match_cols:
            return rebuilt_df.copy()

        extra_cols = [
            c for c in run_df.columns
            if c in match_cols or c not in rebuilt_df.columns
        ]
        merged = rebuilt_df.merge(
            run_df[extra_cols],
            on=match_cols,
            how="left",
        )
        return merged

    def _rebuild_summary_from_steps(self, step_df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "run_id", "strategy", "scenario", "seed",
            "step", "fuel_pct_mean", "fire_coverage_mean", "active_ratio_mean"
        }
        if step_df.empty or not required.issubset(step_df.columns):
            return pd.DataFrame()

        rows = []
        grouped = step_df.groupby("run_id", sort=True)
        for run_id, run_steps in grouped:
            run_steps = run_steps.sort_values("step")
            summary = build_run_summary_from_steps(
                run_steps,
                sps=float(
                    run_steps["steps_per_second"].iloc[-1]
                ) if "steps_per_second" in run_steps.columns else 0.0,
                strategy=str(run_steps["strategy"].iloc[0]),
                scenario=str(run_steps["scenario"].iloc[0]),
                seed=int(run_steps["seed"].iloc[0]),
                default_n_drones=(
                    int(pd.to_numeric(
                        run_steps["n_drones"], errors="coerce"
                    ).iloc[0]) if "n_drones" in run_steps.columns else None
                ),
                default_run_max_steps=(
                    int(pd.to_numeric(
                        run_steps["run_max_steps"], errors="coerce"
                    ).iloc[0]) if "run_max_steps" in run_steps.columns else
                    int(pd.to_numeric(
                        run_steps["max_steps"], errors="coerce"
                    ).iloc[0]) if "max_steps" in run_steps.columns else None
                ),
                fire_out_threshold=self.FIRE_OUT_THRESHOLD,
            )
            summary["run_id"] = run_id
            for col in (
                "preset", "n_drones", "max_steps", "grid_size",
                "config_id", "n_active_agents", "run_max_steps"
            ):
                if col in run_steps.columns:
                    summary[col] = run_steps[col].iloc[0]
            rows.append(summary)
        return pd.DataFrame(rows)

    def _ensure_core_columns(self, run_df: pd.DataFrame) -> pd.DataFrame:
        out = run_df.copy()
        numeric_seed_cols = [
            "fuel_start_total", "fuel_end_total", "fuel_loss_total",
            "fuel_start_pct", "fuel_end_pct", "fuel_saved_pct",
            "fuel_loss_frac", "fuel_preserved_frac",
            "active_drone_steps_raw", "active_exposure_frac",
            "post_fire_active_steps_raw", "post_fire_active_exposure_frac",
            "fire_out_reached", "post_fire_window_steps",
            "stand_down_step", "stand_down_latency_steps", "stand_down_reached",
            "active_ratio_mean", "active_ratio_post_fire",
            "fire_coverage_mean", "fire_coverage_peak",
            "waiting_drone_steps_raw", "exploring_drone_steps_raw",
            "firefighting_drone_steps_raw", "returning_drone_steps_raw",
            "cost", "utility", "utility_internal", "efficiency",
            "n_drones", "n_active_agents", "max_steps", "run_max_steps",
            "steps", "grid_size", "fire_out_fraction",
        ]
        for col in numeric_seed_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if "fuel_end_pct" not in out.columns:
            if "fuel_saved_pct" in out.columns:
                out["fuel_end_pct"] = out["fuel_saved_pct"]
            else:
                out["fuel_end_pct"] = 0.0
        if "fuel_start_pct" not in out.columns:
            if "fuel_start_total" in out.columns and "grid_size" in out.columns:
                denom = out["grid_size"].fillna(0.0).pow(2)
                out["fuel_start_pct"] = (
                    out["fuel_start_total"].fillna(
                        0.0) / denom.where(denom > 0.0, 1.0)
                )
            else:
                out["fuel_start_pct"] = 0.0

        grid_scale = (
            out["grid_size"].fillna(0.0).pow(2)
            if "grid_size" in out.columns else
            pd.Series(1.0, index=out.index, dtype=float)
        )
        if "fuel_start_total" not in out.columns:
            out["fuel_start_total"] = out["fuel_start_pct"].fillna(
                0.0) * grid_scale
        if "fuel_end_total" not in out.columns:
            out["fuel_end_total"] = out["fuel_end_pct"].fillna(
                0.0) * grid_scale
        if "fuel_loss_total" not in out.columns:
            out["fuel_loss_total"] = (
                out["fuel_start_total"].fillna(0.0) -
                out["fuel_end_total"].fillna(0.0)
            ).clip(lower=0.0)

        start_total = out["fuel_start_total"].fillna(0.0)
        end_total = out["fuel_end_total"].fillna(0.0)
        out["fuel_preserved_frac"] = (
            end_total / start_total.where(start_total > 0.0, 1.0)
        ).where(start_total > 0.0, 0.0)
        out["fuel_loss_frac"] = (
            out["fuel_loss_total"].fillna(0.0) /
            start_total.where(start_total > 0.0, 1.0)
        ).where(start_total > 0.0, 0.0)

        agent_capacity = (
            out["n_drones"].fillna(0.0) if "n_drones" in out.columns else
            out["n_active_agents"].fillna(0.0) if "n_active_agents" in out.columns else
            pd.Series(0.0, index=out.index, dtype=float)
        )
        run_steps = (
            out["run_max_steps"].fillna(0.0) if "run_max_steps" in out.columns else
            out["max_steps"].fillna(0.0) if "max_steps" in out.columns else
            out["steps"].fillna(0.0) if "steps" in out.columns else
            pd.Series(0.0, index=out.index, dtype=float)
        )
        exposure_denom = agent_capacity * run_steps
        if "active_drone_steps_raw" not in out.columns:
            if "active_exposure_frac" in out.columns:
                out["active_drone_steps_raw"] = (
                    out["active_exposure_frac"].fillna(0.0) * exposure_denom
                )
            elif "cost" in out.columns:
                out["active_drone_steps_raw"] = out["cost"].fillna(0.0)
            elif "active_ratio_mean" in out.columns:
                out["active_drone_steps_raw"] = (
                    out["active_ratio_mean"].fillna(0.0) * exposure_denom
                )
            else:
                out["active_drone_steps_raw"] = 0.0
        if "active_exposure_frac" not in out.columns:
            out["active_exposure_frac"] = (
                out["active_drone_steps_raw"].fillna(0.0) /
                exposure_denom.where(exposure_denom > 0.0, 1.0)
            ).where(exposure_denom > 0.0, 0.0)

        if "post_fire_active_steps_raw" not in out.columns:
            if "post_fire_active_exposure_frac" in out.columns:
                out["post_fire_active_steps_raw"] = (
                    out["post_fire_active_exposure_frac"].fillna(
                        0.0) * exposure_denom
                )
            elif {"active_ratio_post_fire", "fire_out_fraction"}.issubset(out.columns):
                out["post_fire_active_steps_raw"] = (
                    out["active_ratio_post_fire"].fillna(0.0) *
                    out["fire_out_fraction"].fillna(0.0) *
                    exposure_denom
                )
            else:
                out["post_fire_active_steps_raw"] = 0.0
        if "post_fire_active_exposure_frac" not in out.columns:
            out["post_fire_active_exposure_frac"] = (
                out["post_fire_active_steps_raw"].fillna(0.0) /
                exposure_denom.where(exposure_denom > 0.0, 1.0)
            ).where(exposure_denom > 0.0, 0.0)
        if "fire_out_reached" not in out.columns:
            if "fire_out_step" in out.columns:
                out["fire_out_reached"] = (
                    pd.to_numeric(out["fire_out_step"], errors="coerce") >= 0
                ).astype(float)
            elif "fire_out_fraction" in out.columns:
                out["fire_out_reached"] = (
                    pd.to_numeric(out["fire_out_fraction"], errors="coerce")
                    .fillna(0.0)
                    .gt(0.0)
                    .astype(float)
                )
            else:
                out["fire_out_reached"] = 0.0
        if "post_fire_window_steps" not in out.columns:
            if {"fire_out_fraction", "max_steps"}.issubset(out.columns):
                out["post_fire_window_steps"] = (
                    out["fire_out_fraction"].fillna(0.0) *
                    out["max_steps"].fillna(0.0)
                ).round()
            elif {"fire_out_fraction", "run_max_steps"}.issubset(out.columns):
                out["post_fire_window_steps"] = (
                    out["fire_out_fraction"].fillna(0.0) *
                    out["run_max_steps"].fillna(0.0)
                ).round()
            else:
                out["post_fire_window_steps"] = 0.0
        if "stand_down_step" not in out.columns:
            out["stand_down_step"] = -1.0
        if "stand_down_latency_steps" not in out.columns:
            out["stand_down_latency_steps"] = pd.Series(
                float("nan"), index=out.index, dtype=float
            )
        if "stand_down_reached" not in out.columns:
            out["stand_down_reached"] = pd.Series(
                float("nan"), index=out.index, dtype=float
            )

        out["cost"] = out["active_drone_steps_raw"].fillna(0.0)
        out["utility_internal"] = (
            out["fuel_preserved_frac"].fillna(0.0) -
            out["active_exposure_frac"].fillna(0.0)
        )
        out["utility"] = out["utility_internal"]
        out["efficiency"] = (
            out["fuel_preserved_frac"].fillna(0.0) /
            (out["active_exposure_frac"].fillna(0.0) + 1e-6)
        )

        numeric_cols = [
            "fuel_start_total", "fuel_end_total", "fuel_loss_total",
            "fuel_start_pct", "fuel_end_pct", "fuel_preserved_frac",
            "fuel_loss_frac", "active_drone_steps_raw", "active_exposure_frac",
            "post_fire_active_steps_raw", "post_fire_active_exposure_frac",
            "fire_out_reached", "post_fire_window_steps",
            "stand_down_step", "stand_down_latency_steps", "stand_down_reached",
            "cost", "utility", "utility_internal", "efficiency",
            "active_ratio_mean", "active_ratio_post_fire",
            "fire_coverage_mean", "fire_coverage_peak",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        zero_fill_cols = [
            "fuel_start_total", "fuel_end_total", "fuel_loss_total",
            "fuel_start_pct", "fuel_end_pct", "fuel_preserved_frac",
            "fuel_loss_frac", "active_drone_steps_raw", "active_exposure_frac",
            "post_fire_active_steps_raw", "post_fire_active_exposure_frac",
            "fire_out_reached", "post_fire_window_steps",
            "cost", "utility", "utility_internal", "efficiency",
            "active_ratio_mean", "active_ratio_post_fire",
            "fire_coverage_mean", "fire_coverage_peak",
        ]
        for col in zero_fill_cols:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

        return out

    def _add_never_control_metrics(self, run_df: pd.DataFrame) -> pd.DataFrame:
        if run_df.empty or "strategy" not in run_df.columns:
            return run_df

        context_cols = [
            c for c in (
                "seed", "scenario", "preset", "n_drones",
                "max_steps", "grid_size", "config_id"
            )
            if c in run_df.columns
        ]
        if not context_cols:
            return run_df

        control = run_df[run_df["strategy"] == "never"]
        if control.empty or "fuel_loss_frac" not in control.columns:
            return run_df

        control = control[context_cols + ["fuel_loss_frac"]].rename(
            columns={"fuel_loss_frac": "never_fuel_loss_frac"}
        )
        out = run_df.merge(control, on=context_cols, how="left")
        denom = out["never_fuel_loss_frac"].fillna(0.0)
        out["loss_avoided_vs_never"] = denom - \
            out["fuel_loss_frac"].fillna(0.0)
        out["relative_loss_reduction_vs_never"] = (
            out["loss_avoided_vs_never"] /
            denom.where(denom > 0.0, 1.0)
        ).where(denom > 0.0)
        return out

    @classmethod
    def _group_dims(
        cls,
        df: pd.DataFrame,
        *,
        include_strategy: bool = True,
    ) -> list[str]:
        dims = [c for c in cls.GROUP_DIM_PRIORITY if c in df.columns]
        if include_strategy:
            if "strategy" not in dims and "strategy" in df.columns:
                dims = ["strategy", *dims]
        else:
            dims = [c for c in dims if c != "strategy"]
        return dims

    @staticmethod
    def _format_context(cols: list[str], values) -> str:
        if not cols:
            return "all-runs"
        if not isinstance(values, tuple):
            values = (values,)
        parts = [f"{col}={values[idx]}" for idx, col in enumerate(cols)]
        return ", ".join(parts)

    def _group_stats(
        self,
        run_df: pd.DataFrame,
        group_dims: list[str] | None = None,
    ) -> pd.DataFrame:
        if run_df.empty:
            return pd.DataFrame()

        group_dims = list(group_dims) if group_dims is not None else self._group_dims(
            run_df, include_strategy=True
        )
        group_dims = [c for c in group_dims if c in run_df.columns]
        if not group_dims:
            return pd.DataFrame()

        rows = []
        grouped = run_df.groupby(group_dims, dropna=False, sort=True)
        for group_vals, sub in grouped:
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            n = len(sub)
            if "fire_out_reached" in sub.columns:
                fire_out_mask = pd.to_numeric(
                    sub["fire_out_reached"], errors="coerce"
                ).fillna(0.0).gt(0.0)
            else:
                fire_out_mask = pd.Series(False, index=sub.index, dtype=bool)
            post_fire_sub = sub[fire_out_mask]
            stand_down_valid = (
                pd.to_numeric(
                    post_fire_sub["stand_down_latency_steps"], errors="coerce").dropna()
                if "stand_down_latency_steps" in post_fire_sub.columns else
                pd.Series(dtype=float)
            )
            stand_down_reached = (
                pd.to_numeric(
                    post_fire_sub["stand_down_reached"], errors="coerce").dropna()
                if "stand_down_reached" in post_fire_sub.columns else
                pd.Series(dtype=float)
            )
            post_fire_active_exposure = (
                pd.to_numeric(
                    post_fire_sub["post_fire_active_exposure_frac"], errors="coerce").dropna()
                if "post_fire_active_exposure_frac" in post_fire_sub.columns else
                pd.Series(dtype=float)
            )
            post_fire_active_steps = (
                pd.to_numeric(
                    post_fire_sub["post_fire_active_steps_raw"], errors="coerce").dropna()
                if "post_fire_active_steps_raw" in post_fire_sub.columns else
                pd.Series(dtype=float)
            )
            fire_out_steps = (
                pd.to_numeric(post_fire_sub["fire_out_step"], errors="coerce")
                if "fire_out_step" in post_fire_sub.columns else
                pd.Series(dtype=float)
            )
            fire_out_steps = fire_out_steps[fire_out_steps >= 0].dropna()
            fire_out_rate = float(fire_out_mask.mean()) if n > 0 else 0.0
            row = {
                "n_runs": int(n),
                "fuel_preserved_mean": float(sub["fuel_preserved_frac"].mean()),
                "fuel_preserved_std": float(sub["fuel_preserved_frac"].std(ddof=1)) if n > 1 else 0.0,
                "fuel_preserved_ci95": self._ci95(sub["fuel_preserved_frac"]),
                "fuel_loss_mean": float(sub["fuel_loss_frac"].mean()),
                "fuel_loss_std": float(sub["fuel_loss_frac"].std(ddof=1)) if n > 1 else 0.0,
                "fuel_loss_ci95": self._ci95(sub["fuel_loss_frac"]),
                "active_exposure_mean": float(sub["active_exposure_frac"].mean()),
                "active_exposure_std": float(sub["active_exposure_frac"].std(ddof=1)) if n > 1 else 0.0,
                "active_exposure_ci95": self._ci95(sub["active_exposure_frac"]),
                "active_drone_steps_raw_mean": float(sub["active_drone_steps_raw"].mean()) if "active_drone_steps_raw" in sub else 0.0,
                "active_headcount_mean": (
                    float(sub["active_headcount_run_mean"].mean())
                    if "active_headcount_run_mean" in sub.columns else float("nan")
                ),
                "active_headcount_ci95": (
                    self._ci95(sub["active_headcount_run_mean"])
                    if "active_headcount_run_mean" in sub.columns else 0.0
                ),
                "post_transient_active_headcount_mean": (
                    float(sub["post_transient_active_headcount_mean"].mean())
                    if "post_transient_active_headcount_mean" in sub.columns else float("nan")
                ),
                "post_transient_active_headcount_ci95": (
                    self._ci95(sub["post_transient_active_headcount_mean"])
                    if "post_transient_active_headcount_mean" in sub.columns else 0.0
                ),
                "post_fire_active_exposure_mean": (
                    float(post_fire_active_exposure.mean())
                    if not post_fire_active_exposure.empty else float("nan")
                ),
                "post_fire_active_exposure_ci95": (
                    self._ci95(post_fire_active_exposure)
                    if not post_fire_active_exposure.empty else 0.0
                ),
                "post_fire_active_steps_raw_mean": (
                    float(post_fire_active_steps.mean())
                    if not post_fire_active_steps.empty else float("nan")
                ),
                "fire_out_rate": fire_out_rate,
                "fire_out_rate_ci95": self._ci95(fire_out_mask.astype(float)),
                "fire_out_step_mean": (
                    float(fire_out_steps.mean())
                    if not fire_out_steps.empty else float("nan")
                ),
                "fire_out_step_ci95": (
                    self._ci95(fire_out_steps)
                    if not fire_out_steps.empty else 0.0
                ),
                "stand_down_latency_steps_mean": (
                    float(stand_down_valid.mean())
                    if not stand_down_valid.empty else float("nan")
                ),
                "stand_down_latency_steps_ci95": (
                    self._ci95(stand_down_valid)
                    if not stand_down_valid.empty else 0.0
                ),
                "stand_down_success_rate": (
                    float(stand_down_reached.mean())
                    if not stand_down_reached.empty else float("nan")
                ),
                "cost_mean": float(sub["cost"].mean()),
                "cost_std": float(sub["cost"].std(ddof=1)) if n > 1 else 0.0,
                "cost_ci95": self._ci95(sub["cost"]),
                "utility_mean": (
                    float(sub["utility_internal"].mean())
                    if "utility_internal" in sub.columns else float("nan")
                ),
                "utility_std": (
                    float(sub["utility_internal"].std(ddof=1))
                    if "utility_internal" in sub.columns and n > 1 else 0.0
                ),
                "utility_ci95": (
                    self._ci95(sub["utility_internal"])
                    if "utility_internal" in sub.columns else 0.0
                ),
                "efficiency_mean": float(sub["efficiency"].mean()),
                "efficiency_std": float(sub["efficiency"].std(ddof=1)) if n > 1 else 0.0,
                "efficiency_ci95": self._ci95(sub["efficiency"]),
                "active_ratio_mean": float(sub["active_ratio_mean"].mean()) if "active_ratio_mean" in sub else 0.0,
                "active_ratio_post_fire_mean": (
                    float(pd.to_numeric(
                        post_fire_sub["active_ratio_post_fire"], errors="coerce"
                    ).dropna().mean())
                    if "active_ratio_post_fire" in post_fire_sub and not post_fire_sub.empty else float("nan")
                ),
            }
            if "neighbor_count_abs_run_mean" in sub.columns:
                row["neighbor_count_abs_run_mean"] = float(
                    pd.to_numeric(
                        sub["neighbor_count_abs_run_mean"], errors="coerce"
                    ).dropna().mean()
                )
            if "loss_avoided_vs_never" in sub.columns:
                row["loss_avoided_vs_never_mean"] = float(
                    sub["loss_avoided_vs_never"].mean())
                row["loss_avoided_vs_never_ci95"] = self._ci95(
                    sub["loss_avoided_vs_never"])
            if "relative_loss_reduction_vs_never" in sub.columns:
                valid = pd.to_numeric(
                    sub["relative_loss_reduction_vs_never"], errors="coerce"
                ).dropna()
                row["relative_loss_reduction_vs_never_mean"] = float(
                    valid.mean()) if not valid.empty else 0.0
                row["relative_loss_reduction_vs_never_ci95"] = self._ci95(
                    valid) if not valid.empty else 0.0
            for idx, col in enumerate(group_dims):
                row[col] = group_vals[idx]
            rows.append(row)

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        sort_cols = [c for c in group_dims if c != "strategy"] + ["strategy"]
        sort_cols = [c for c in sort_cols if c in result.columns]
        ascending = [True] * len(sort_cols)
        return result.sort_values(sort_cols, ascending=ascending)

    @staticmethod
    def _pareto_front(stats_df: pd.DataFrame) -> pd.DataFrame:
        """Pareto front where fuel_preserved_mean is maximized and cost_mean minimized."""
        if stats_df.empty:
            return stats_df

        points = stats_df.reset_index(drop=True)
        keep = [True] * len(points)
        for i in range(len(points)):
            row_i = points.iloc[i]
            for j in range(len(points)):
                if i == j:
                    continue
                row_j = points.iloc[j]
                dominates = (
                    (row_j["fuel_preserved_mean"] >= row_i["fuel_preserved_mean"]) and
                    (row_j["cost_mean"] <= row_i["cost_mean"]) and
                    (
                        (row_j["fuel_preserved_mean"] > row_i["fuel_preserved_mean"]) or
                        (row_j["cost_mean"] < row_i["cost_mean"])
                    )
                )
                if dominates:
                    keep[i] = False
                    break
        return points[pd.Series(keep)].copy()

    def _pairwise_seed_deltas(self, run_df: pd.DataFrame) -> pd.DataFrame:
        """Paired deltas on preserved/loss/cost/efficiency between strategies."""
        rows = []
        if "seed" not in run_df.columns:
            return pd.DataFrame()

        context_cols = self._group_dims(run_df, include_strategy=False)
        grouped = (
            run_df.groupby(context_cols, dropna=False, sort=True)
            if context_cols else [((), run_df)]
        )

        for context_vals, sub in grouped:
            if not isinstance(context_vals, tuple):
                context_vals = (context_vals,)
            strategies = sorted(sub["strategy"].unique())
            for i, a in enumerate(strategies):
                for b in strategies[i + 1:]:
                    a_df = sub[sub["strategy"] == a][[
                        "seed", "fuel_preserved_frac", "fuel_loss_frac",
                        "cost", "active_exposure_frac", "efficiency", "utility_internal"]]
                    b_df = sub[sub["strategy"] == b][[
                        "seed", "fuel_preserved_frac", "fuel_loss_frac",
                        "cost", "active_exposure_frac", "efficiency", "utility_internal"]]
                    merged = a_df.merge(
                        b_df, on="seed", suffixes=("_a", "_b"), how="inner"
                    )
                    if merged.empty:
                        continue

                    delta_preserved = (
                        merged["fuel_preserved_frac_b"] -
                        merged["fuel_preserved_frac_a"]
                    )
                    delta_loss = merged["fuel_loss_frac_b"] - \
                        merged["fuel_loss_frac_a"]
                    delta_cost = (
                        merged["cost_b"] - merged["cost_a"]
                    )
                    delta_efficiency = (
                        merged["efficiency_b"] - merged["efficiency_a"]
                    )
                    delta_utility = (
                        merged["utility_internal_b"] -
                        merged["utility_internal_a"]
                    )
                    row = {
                        "strategy_a": a,
                        "strategy_b": b,
                        "n_pairs": int(len(merged)),
                        "delta_fuel_preserved_mean_b_minus_a": float(delta_preserved.mean()),
                        "delta_fuel_preserved_ci95": self._ci95(delta_preserved),
                        "delta_fuel_loss_mean_b_minus_a": float(delta_loss.mean()),
                        "delta_fuel_loss_ci95": self._ci95(delta_loss),
                        "delta_cost_mean_b_minus_a": float(delta_cost.mean()),
                        "delta_efficiency_mean_b_minus_a": float(delta_efficiency.mean()),
                        "delta_efficiency_ci95": self._ci95(delta_efficiency),
                        "delta_utility_internal_mean_b_minus_a": float(delta_utility.mean()),
                        "delta_utility_internal_ci95": self._ci95(delta_utility),
                        "win_rate_b_on_fuel_preserved": float((delta_preserved > 0.0).mean()),
                        "win_rate_b_on_fuel_loss": float((delta_loss < 0.0).mean()),
                        "win_rate_b_on_efficiency": float((delta_efficiency > 0.0).mean()),
                    }
                    for idx, col in enumerate(context_cols):
                        row[col] = context_vals[idx]
                    rows.append(row)
        return pd.DataFrame(rows)

    def _build_report(
        self,
        run_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        pareto_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        plots_created: list[str],
        output_table_paths: list[Path],
        paper_plots: list[str] | None = None,
    ) -> str:
        def _fmt(value, spec: str = ".4f", percent: bool = False) -> str:
            if pd.isna(value):
                return "n/a"
            if percent:
                return format(float(value), ".1%")
            return format(float(value), spec)

        def _render_table(
            frame: pd.DataFrame,
            columns: list[tuple[str, str]],
        ) -> list[str]:
            if frame.empty:
                return ["(no data)"]
            headers = [label for _, label in columns]
            body = [
                [str(frame.iloc[row_idx][key]) for key, _ in columns]
                for row_idx in range(len(frame))
            ]
            widths = []
            for col_idx, header in enumerate(headers):
                values = [row[col_idx] for row in body]
                widths.append(max(len(header), *(len(value)
                              for value in values)))

            def _row(values: list[str]) -> str:
                return " | ".join(
                    value.ljust(widths[idx]) for idx, value in enumerate(values)
                )

            separator = "-+-".join("-" * width for width in widths)
            lines_out = [_row(headers), separator]
            lines_out.extend(_row(row) for row in body)
            return lines_out

        def _strategy_table(
            frame: pd.DataFrame,
            pareto_members: set[str],
        ) -> pd.DataFrame:
            if frame.empty:
                return pd.DataFrame()
            ranked = frame.sort_values(
                ["fuel_preserved_mean", "cost_mean",
                    "efficiency_mean", "strategy"],
                ascending=[False, True, False, True],
                na_position="last",
            ).reset_index(drop=True)
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            ranked["fuel"] = ranked["fuel_preserved_mean"].map(
                lambda value: _fmt(value, percent=True)
            )
            ranked["cost"] = ranked["cost_mean"].map(
                lambda value: _fmt(value, ".1f")
            )
            ranked["active_ratio"] = ranked["active_exposure_mean"].map(
                lambda value: _fmt(value, percent=True)
            )
            ranked["eff"] = ranked["efficiency_mean"].map(
                lambda value: _fmt(value, ".2f")
            )
            ranked["pareto"] = ranked["strategy"].astype(str).map(
                lambda strategy: "yes" if strategy in pareto_members else ""
            )
            ranked["runs"] = ranked["n_runs"].map(
                lambda value: "n/a" if pd.isna(value) else str(int(value))
            )
            return ranked[["rank", "strategy", "fuel", "cost", "active_ratio", "eff", "pareto", "runs"]]

        def _pareto_decision(frame: pd.DataFrame) -> tuple[str, str]:
            if frame.empty:
                return ("n/a", "n/a")
            front = self._pareto_front(frame)
            if front.empty:
                return ("n/a", "n/a")
            front_ranked = front.sort_values(
                ["fuel_preserved_mean", "cost_mean", "strategy"],
                ascending=[False, True, True],
                na_position="last",
            ).reset_index(drop=True)
            pareto_pick = str(front_ranked.iloc[0]["strategy"])
            pareto_set = ", ".join(
                front_ranked["strategy"].astype(str).tolist())
            return pareto_pick, pareto_set

        def _scenario_label(scenario: str) -> str:
            return str(scenario).replace("_", " ").title()

        lines = []
        lines.append("Apoptotic Wildfire Analysis Summary")
        lines.append("=" * 36)
        lines.append(f"Run directory: {self.paths.root.name}")
        lines.append("")
        lines.append("Dataset")
        lines.append("-" * 36)
        lines.append(f"Runs analyzed: {len(run_df)}")
        lines.append(
            "Strategies: " +
            ", ".join(sorted(run_df["strategy"].astype(str).unique()))
        )
        lines.append(
            "Scenarios: " +
            ", ".join(sorted(run_df["scenario"].astype(str).unique()))
        )
        if "preset" in run_df.columns:
            lines.append(
                "Presets: " +
                ", ".join(sorted(run_df["preset"].astype(str).unique()))
            )
        for col in ("n_drones", "max_steps", "grid_size"):
            if col in run_df.columns:
                values = ", ".join(
                    str(v) for v in sorted(run_df[col].dropna().unique())
                )
                lines.append(f"{col}: {values}")
        lines.append("")
        lines.append("Metric guide")
        lines.append("-" * 36)
        lines.append(
            "Fuel preserved = final fuel / initial fuel. Higher is better."
        )
        lines.append(
            "Cost = cumulative active drone-steps. Each non-WAITING drone contributes 1 cost unit per step."
        )
        lines.append(
            "Active-time ratio = active drone-steps / (N x episode steps). Lower is better, but this is secondary to the raw cost count."
        )
        lines.append(
            "Efficiency = fuel preserved / active-time ratio. Use it as a secondary ratio, not the primary ranking."
        )
        lines.append(
            "Pareto winner = the preserve-first pick taken from the non-dominated Pareto set."
        )
        lines.append("")

        overall_stats = self._group_stats(run_df, group_dims=["strategy"])
        overall_pareto = self._pareto_front(overall_stats)
        overall_pareto_members = set(overall_pareto["strategy"].astype(str))
        overall_pick, overall_set = _pareto_decision(overall_stats)

        lines.append("Overall strategy table")
        lines.append("-" * 36)
        lines.extend(
            _render_table(
                _strategy_table(overall_stats, overall_pareto_members),
                [
                    ("rank", "Rank"),
                    ("strategy", "Strategy"),
                    ("fuel", "Fuel preserved"),
                    ("cost", "Cost"),
                    ("active_ratio", "Active-time ratio"),
                    ("eff", "Efficiency"),
                    ("pareto", "Pareto"),
                    ("runs", "Runs"),
                ],
            )
        )
        lines.append("")
        lines.append(f"Pareto set: {overall_set}")
        lines.append(f"Pareto winner (preserve-first pick): {overall_pick}")
        lines.append("")

        if "preset" in run_df.columns and run_df["preset"].dropna().nunique() > 1:
            lines.append("Overall by preset")
            lines.append("-" * 36)
            for preset_val in sorted(run_df["preset"].dropna().unique()):
                preset_run_df = run_df[run_df["preset"] == preset_val]
                preset_stats = self._group_stats(
                    preset_run_df, group_dims=["strategy"])
                preset_pareto = self._pareto_front(preset_stats)
                preset_pareto_members = set(
                    preset_pareto["strategy"].astype(str))
                p_pick, p_set = _pareto_decision(preset_stats)
                lines.append(f"Preset: {preset_val}")
                lines.extend(
                    _render_table(
                        _strategy_table(preset_stats, preset_pareto_members),
                        [
                            ("rank", "Rank"),
                            ("strategy", "Strategy"),
                            ("fuel", "Fuel preserved"),
                            ("cost", "Cost"),
                            ("active_ratio", "Active-time ratio"),
                            ("eff", "Efficiency"),
                            ("pareto", "Pareto"),
                            ("runs", "Runs"),
                        ],
                    )
                )
                lines.append(f"Pareto set: {p_set}")
                lines.append(f"Pareto winner (preserve-first pick): {p_pick}")
                lines.append("")

        scenario_stats = (
            self._group_stats(run_df, group_dims=["scenario", "strategy"])
            if "scenario" in run_df.columns
            else pd.DataFrame()
        )
        if not scenario_stats.empty:
            varying_pool_cols = [
                col for col in ("preset", "n_drones", "max_steps", "grid_size")
                if col in run_df.columns and run_df[col].dropna().nunique() > 1
            ]
            lines.append("Scenario tables")
            lines.append("-" * 36)
            if varying_pool_cols:
                lines.append(
                    "Each scenario table pools over: " +
                    ", ".join(varying_pool_cols)
                )
                lines.append("")
            for scenario, sub in scenario_stats.groupby("scenario", dropna=False, sort=True):
                pareto_sub = self._pareto_front(sub)
                pareto_members = set(pareto_sub["strategy"].astype(str))
                pick, pareto_set = _pareto_decision(sub)
                lines.append(f"Scenario: {_scenario_label(scenario)}")
                lines.extend(
                    _render_table(
                        _strategy_table(sub, pareto_members),
                        [
                            ("rank", "Rank"),
                            ("strategy", "Strategy"),
                            ("fuel", "Fuel preserved"),
                            ("cost", "Cost"),
                            ("active_ratio", "Active-time ratio"),
                            ("eff", "Efficiency"),
                            ("pareto", "Pareto"),
                            ("runs", "Runs"),
                        ],
                    )
                )
                lines.append(f"Pareto set: {pareto_set}")
                lines.append(f"Pareto winner (preserve-first pick): {pick}")
                lines.append("")

                if "preset" in run_df.columns and run_df["preset"].dropna().nunique() > 1:
                    for preset_val in sorted(run_df["preset"].dropna().unique()):
                        scen_preset_df = run_df[
                            (run_df["scenario"] == scenario) &
                            (run_df["preset"] == preset_val)
                        ]
                        if scen_preset_df.empty:
                            continue
                        sp_stats = self._group_stats(
                            scen_preset_df, group_dims=["strategy"])
                        sp_pareto = self._pareto_front(sp_stats)
                        sp_pareto_members = set(
                            sp_pareto["strategy"].astype(str))
                        sp_pick, sp_set = _pareto_decision(sp_stats)
                        lines.append(f"  Preset: {preset_val}")
                        lines.extend(
                            _render_table(
                                _strategy_table(sp_stats, sp_pareto_members),
                                [
                                    ("rank", "Rank"),
                                    ("strategy", "Strategy"),
                                    ("fuel", "Fuel preserved"),
                                    ("cost", "Cost"),
                                    ("active_ratio", "Active-time ratio"),
                                    ("eff", "Efficiency"),
                                    ("pareto", "Pareto"),
                                    ("runs", "Runs"),
                                ],
                            )
                        )
                        lines.append(f"  Pareto set: {sp_set}")
                        lines.append(
                            f"  Pareto winner (preserve-first pick): {sp_pick}")
                        lines.append("")

        if {"fire_out_rate", "stand_down_latency_steps_mean", "post_fire_active_exposure_mean"}.issubset(stats_df.columns):
            lines.append("Adaptability summary")
            lines.append("-" * 36)
            adapt_cols = ["strategy"]
            if "scenario" in stats_df.columns:
                adapt_cols.append("scenario")
            adapt_stats = self._group_stats(run_df, group_dims=adapt_cols)
            if "scenario" in adapt_stats.columns:
                for scenario, sub in adapt_stats.groupby("scenario", dropna=False, sort=True):
                    table = sub.sort_values(
                        ["fire_out_rate", "post_fire_active_exposure_mean",
                            "stand_down_latency_steps_mean", "strategy"],
                        ascending=[False, True, True, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    table["strategy"] = table["strategy"].astype(str)
                    table["fire_out"] = table["fire_out_rate"].map(
                        lambda value: _fmt(value, percent=True))
                    table["post_fire_cost"] = table["post_fire_active_exposure_mean"].map(
                        lambda value: _fmt(value, percent=True))
                    table["stand_down"] = table["stand_down_latency_steps_mean"].map(
                        lambda value: _fmt(value, ".1f"))
                    lines.append(f"Scenario: {_scenario_label(scenario)}")
                    lines.extend(
                        _render_table(
                            table[["strategy", "fire_out",
                                   "post_fire_cost", "stand_down"]],
                            [
                                ("strategy", "Strategy"),
                                ("fire_out", "Fire-out rate"),
                                ("post_fire_cost", "Post-fire cost"),
                                ("stand_down", "Stand-down"),
                            ],
                        )
                    )
                    lines.append("")
            else:
                table = adapt_stats.sort_values(
                    ["fire_out_rate", "post_fire_active_exposure_mean",
                        "stand_down_latency_steps_mean", "strategy"],
                    ascending=[False, True, True, True],
                    na_position="last",
                ).reset_index(drop=True)
                table["strategy"] = table["strategy"].astype(str)
                table["fire_out"] = table["fire_out_rate"].map(
                    lambda value: _fmt(value, percent=True))
                table["post_fire_cost"] = table["post_fire_active_exposure_mean"].map(
                    lambda value: _fmt(value, percent=True))
                table["stand_down"] = table["stand_down_latency_steps_mean"].map(
                    lambda value: _fmt(value, ".1f"))
                lines.extend(
                    _render_table(
                        table[["strategy", "fire_out",
                               "post_fire_cost", "stand_down"]],
                        [
                            ("strategy", "Strategy"),
                            ("fire_out", "Fire-out rate"),
                            ("post_fire_cost", "Post-fire cost"),
                            ("stand_down", "Stand-down"),
                        ],
                    )
                )
                lines.append("")

        if not pairwise_df.empty:
            lines.append("Best paired comparison on fuel preserved")
            lines.append("-" * 36)
            pairwise_context_cols = [
                c for c in ("scenario", "preset", "n_drones", "max_steps", "grid_size")
                if c in pairwise_df.columns
            ]
            sort_cols = pairwise_context_cols + \
                ["delta_fuel_preserved_mean_b_minus_a"]
            ascending = [True] * len(pairwise_context_cols) + [False]
            best_rows = (
                pairwise_df.sort_values(sort_cols, ascending=ascending)
                .groupby(pairwise_context_cols, dropna=False, sort=True)
                .head(1)
                if pairwise_context_cols
                else pairwise_df.sort_values(sort_cols, ascending=ascending).head(1)
            )
            if not isinstance(best_rows, pd.DataFrame):
                best_rows = pd.DataFrame(best_rows)
            for _, row in best_rows.iterrows():
                context_label = self._format_context(
                    pairwise_context_cols,
                    tuple(
                        row[c] for c in pairwise_context_cols) if pairwise_context_cols else (),
                )
                lines.append(
                    f"{context_label}: {row['strategy_b']} over {row['strategy_a']} "
                    f"(Δfuel={_fmt(row['delta_fuel_preserved_mean_b_minus_a'], '.4f')}, "
                    f"Δcost={_fmt(row['delta_cost_mean_b_minus_a'], '.4f')}, "
                    f"Δeff={_fmt(row['delta_efficiency_mean_b_minus_a'], '.4f')}, "
                    f"n={int(row['n_pairs'])})"
                )
            lines.append("")

        lines.append("Output tables")
        lines.append("-" * 36)
        for table_path in output_table_paths:
            lines.append(table_path.name)
        return "\n".join(lines) + "\n"

    def analyze(self, generate_plots: bool = True) -> bool:
        """Run multi-experiment analysis for one output directory."""
        run_df = self._load_run_summary()
        step_df = self._load_step_metrics()

        rebuilt_df = self._rebuild_summary_from_steps(
            step_df) if not step_df.empty else pd.DataFrame()
        if run_df.empty and not rebuilt_df.empty:
            run_df = rebuilt_df
        elif not run_df.empty and not rebuilt_df.empty:
            run_df = self._merge_rebuilt_summary(run_df, rebuilt_df)

        if run_df.empty:
            print(
                f"No experiment summary found in {self.data_dir}. "
                "Expected run_summary.parquet/csv or run_*.parquet/csv with step_metrics."
            )
            return False

        if self.exclude_presets and "preset" in run_df.columns:
            before = len(run_df)
            run_df = run_df[~run_df["preset"].isin(self.exclude_presets)]
            dropped = before - len(run_df)
            if dropped:
                print(
                    f"ℹ️ Excluded {dropped} runs with preset(s): {sorted(self.exclude_presets)}")
            if run_df.empty:
                print("No data remaining after excluding preset(s).")
                return False

        if self.exclude_strategies and "strategy" in run_df.columns:
            before = len(run_df)
            run_df = run_df[~run_df["strategy"].isin(self.exclude_strategies)]
            dropped = before - len(run_df)
            if dropped:
                print(
                    f"ℹ️ Excluded {dropped} runs with strategy(s): {sorted(self.exclude_strategies)}")
            if run_df.empty:
                print("No data remaining after excluding strategy(s).")
                return False

        if self.exclude_scenarios and "scenario" in run_df.columns:
            before = len(run_df)
            run_df = run_df[~run_df["scenario"].isin(self.exclude_scenarios)]
            dropped = before - len(run_df)
            if dropped:
                print(
                    f"ℹ️ Excluded {dropped} runs with scenario(s): {sorted(self.exclude_scenarios)}")
            if run_df.empty:
                print("No data remaining after excluding scenario(s).")
                return False

        required = {"strategy", "scenario", "seed"}
        missing = required.difference(run_df.columns)
        if missing:
            print(
                f"Run summary is missing required columns: {sorted(missing)}. "
                "Expected at least strategy/scenario/seed metadata."
            )
            return False

        run_df = self._ensure_core_columns(run_df)
        run_df = self._add_never_control_metrics(run_df)
        stats_df = self._group_stats(run_df)
        if stats_df.empty:
            print("Unable to compute grouped statistics from run summary data.")
            return False

        pareto_df = self._pareto_front(stats_df)
        pairwise_df = self._pairwise_seed_deltas(run_df)

        out_runs = self._write_table(
            run_df, self.data_dir / "analysis_runs_enriched")
        out_stats = self._write_table(
            stats_df, self.data_dir / "analysis_group_stats")
        out_pareto = self._write_table(
            pareto_df, self.data_dir / "analysis_pareto")
        out_pairwise = self._write_table(
            pairwise_df, self.data_dir / "analysis_pairwise")

        if not generate_plots:
            report = self._build_report(
                run_df=run_df,
                stats_df=stats_df,
                pareto_df=pareto_df,
                pairwise_df=pairwise_df,
                plots_created=[],
                output_table_paths=[
                    out_runs,
                    out_stats,
                    out_pareto,
                    out_pairwise,
                ],
                paper_plots=[],
            )
            self.paths.summary.write_text(report)
            print(f"Analysis report: {self.paths.summary}")
            return True

        paper_plots: list[str] = []
        try:
            paper_plots = self.paper_plotter.plot_suite(
                run_df, stats_df, step_df)
        except Exception as exc:
            print(f"⚠️ Failed to generate paper figures: {exc}")

        report = self._build_report(
            run_df=run_df,
            stats_df=stats_df,
            pareto_df=pareto_df,
            pairwise_df=pairwise_df,
            plots_created=paper_plots,
            output_table_paths=[
                out_runs,
                out_stats,
                out_pareto,
                out_pairwise
            ],
            paper_plots=paper_plots,
        )
        self.paths.summary.write_text(report)

        print(f"Analysis report: {self.paths.summary}")
        if paper_plots:
            print(f"Analysis plots: {self.paths.plots}")
        return True
