"""Journal-ready narrative plots for paper-facing experiment analysis."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd


class PaperPlotter:
    """Builds a fixed paper figure suite from analyzer summary tables."""

    CONTEXT_COLUMNS = [
        "scenario",
        "preset",
        "n_drones",
        "max_steps",
        "grid_size",
        "config_id",
    ]
    COLUMN_LABELS = {
        "scenario": "scenario",
        "preset": "preset",
        "n_drones": "N",
        "max_steps": "steps",
        "grid_size": "grid",
        "config_id": "config",
    }
    FIGURE_STEMS = {
        "headline": "01_headline_ranking",
        "pareto": "02_tradeoff_pareto",
        "temporal": "03_temporal_response",
        "mechanism": "04_behavior_mechanism",
        "adaptability": "05_adaptability_after_fire_out",
        "population": "06_population_scaling",
        "distributions": "07_run_distributions",
        "attrition": "08_attrition",
        "fire_out": "09_fire_out_rate",
        "population_perf_cost": "10_population_performance_cost",
        "population_cost": "11_population_cost_scaling",
        "density": "12_average_density",
        "population_perf_cost_combined": "13_population_performance_cost_combined",
        "population_active_ratio": "14_population_active_ratio_scaling",
    }
    VARIANT_SUFFIX = {
        "overall": "overall",
        "scenario": "by_scenario",
    }
    PALETTE = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#000000",  # black
        "#F0E442",  # yellow
    ]
    MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*", "h", "<", ">"]
    SCENARIO_PANEL_ORDER = {
        "baseline": 0,
        "fast_spreading_fire": 0,
        "stress_test": 1,
    }
    META_STRATEGIES: set[str] = set()
    FUEL_LABEL = "Fuel preserved (%)"
    ACTIVE_LABEL = "Exposure ratio (%)"
    COST_LABEL = "Operational cost (active drone-steps)"
    DENSITY_LABEL = "Average active density (drones per cell)"
    HEADCOUNT_LABEL = "Post-transient active headcount"
    UTILITY_LABEL = "Objective (fuel minus exposure ratio)"
    FUEL_DEF = "Fuel preserved (%) = final fuel / initial fuel."
    ACTIVE_DEF = (
        "Exposure ratio (%) = active drone-steps / (N x episode steps)."
    )
    COST_DEF = (
        "Operational cost = cumulative active drone-steps; each non-waiting drone contributes 1 cost unit per step."
    )
    DENSITY_DEF = (
        "Average active density = mean over episode of active drones divided by grid cells."
    )
    HEADCOUNT_DEF = (
        "Post-transient active headcount = mean active drones after the first 20% "
        "of the episode."
    )
    UTILITY_DEF = (
        "Objective = fuel preserved fraction minus exposure ratio."
    )
    PRIMARY_NOTE = (
        "Preserve-first ranking: strategies are ordered by fuel preserved at episode end; ties break toward lower exposure ratio."
    )
    TRADEOFF_NOTE = ""
    TEMPORAL_NOTE = (
        "Temporal panels show containment speed, operational load, and how much of the swarm commits to firefighting over time."
    )
    MECHANISM_NOTE = (
        "Mechanism panels explain whether a strategy wins by committing earlier, conserving battery, or returning to base sooner."
    )
    ADAPTABILITY_NOTE = (
        "Adaptability compares exposure ratio while fire is still present against exposure ratio after fire-out. Good stand-down behavior keeps the post-fire value low without collapsing activity too early."
    )
    FIRE_OUT_NOTE = (
        "Fire-out rate is the share of runs that reached sustained fire-out within the episode horizon."
    )
    SCALING_NOTE = (
        "Population scaling shows whether added drones improve fuel preserved faster than they increase exposure ratio."
    )
    COST_SCALING_NOTE = (
        "Operational cost scaling shows how cumulative active drone-steps grow with swarm size."
    )
    DISTRIBUTION_NOTE = (
        "Violin bodies show run distributions. Box-and-whisker overlays show median and spread for fuel preserved, exposure ratio, operational cost, and efficiency."
    )
    ATTRITION_NOTE = (
        "Attrition panels show when agents die and how losses accumulate across the episode."
    )
    DENSITY_NOTE = (
        "Average density measures how densely the environment is occupied by active drones over time."
    )
    FIRE_PRESENT_THRESHOLD = 1e-3

    def __init__(
        self,
        plots_dir: Path,
        *,
        paper_subdir: str = "paper",
    ):
        self.root_plots_dir = Path(plots_dir)
        self.paper_dir = self.root_plots_dir / Path(paper_subdir)
        self.paper_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_pyplot():
        try:
            import matplotlib.pyplot as plt

            return plt
        except Exception as exc:
            print(
                f"⚠️ Paper plotting disabled: matplotlib unavailable ({exc})")
            return None

    @staticmethod
    def _warn(message: str) -> None:
        print(f"⚠️ PaperPlotter: {message}")

    @classmethod
    def _context_columns(
        cls,
        df: pd.DataFrame,
        *,
        exclude: tuple[str, ...] = (),
    ) -> list[str]:
        return [
            col for col in cls.CONTEXT_COLUMNS
            if col in df.columns and col not in exclude
        ]

    @staticmethod
    def _value_text(value) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, float):
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.4g}"
        return str(value)

    @staticmethod
    def _humanize(text: str) -> str:
        return str(text).replace("_", " ").strip()

    @classmethod
    def _display_context_value(cls, col: str, value) -> str:
        if pd.isna(value):
            return "n/a"
        if col == "scenario":
            return cls._humanize(value).title()
        if col == "preset":
            return f"Preset={value}"
        if col == "n_drones":
            return f"N={int(float(value))}"
        if col == "max_steps":
            return f"T={int(float(value))}"
        if col == "grid_size":
            return f"Grid={int(float(value))}"
        if col == "config_id":
            return cls._humanize(value)
        label = cls.COLUMN_LABELS.get(col, col)
        return f"{label}={cls._value_text(value)}"

    @classmethod
    def _varying_columns(cls, df: pd.DataFrame, cols: list[str]) -> list[str]:
        varying: list[str] = []
        for col in cols:
            uniq = df[col].drop_duplicates()
            if len(uniq) > 1:
                varying.append(col)
        return varying

    @classmethod
    def _group_context_columns(
        cls,
        df: pd.DataFrame,
        *,
        exclude: tuple[str, ...] = (),
    ) -> list[str]:
        preferred = [
            col for col in cls.CONTEXT_COLUMNS
            if col in df.columns and col not in exclude and col != "config_id"
        ]
        if preferred:
            return preferred
        if "config_id" in df.columns and "config_id" not in exclude:
            return ["config_id"]
        return []

    @classmethod
    def _panel_context_columns(
        cls,
        df: pd.DataFrame,
        *,
        exclude: tuple[str, ...] = (),
    ) -> list[str]:
        group_cols = cls._group_context_columns(df, exclude=exclude)
        panel_cols = cls._varying_columns(df, group_cols)
        return panel_cols

    @classmethod
    def _context_label(cls, cols: list[str], values) -> str:
        if not cols:
            return "All runs"
        if not isinstance(values, tuple):
            values = (values,)
        if len(cols) == 1:
            return cls._display_context_value(cols[0], values[0])
        parts = [
            cls._display_context_value(col, values[idx])
            for idx, col in enumerate(cols)
        ]
        return fill(" | ".join(parts), width=34)

    @classmethod
    def _fixed_context_note(
        cls,
        df: pd.DataFrame,
        *,
        exclude: tuple[str, ...] = (),
    ) -> str:
        group_cols = cls._group_context_columns(df, exclude=exclude)
        panel_cols = cls._panel_context_columns(df, exclude=exclude)
        fixed_cols = [col for col in group_cols if col not in panel_cols]
        parts: list[str] = []
        for col in fixed_cols:
            unique = df[col].dropna().unique()
            if len(unique) != 1:
                continue
            parts.append(cls._display_context_value(col, unique[0]))
        return " | ".join(parts)

    @classmethod
    def _iter_context_groups(
        cls,
        df: pd.DataFrame,
        *,
        exclude: tuple[str, ...] = (),
        include_all_n: bool = False,
    ) -> list[tuple[tuple, pd.DataFrame, str, list[str]]]:
        context_cols = cls._panel_context_columns(df, exclude=exclude)
        groups: list[tuple[tuple, pd.DataFrame, str, list[str]]] = []
        if not context_cols:
            groups = [((), df.copy(), "All runs", context_cols)]
        else:
            grouped = df.groupby(context_cols, dropna=False, sort=True)
            for values, sub in grouped:
                if not isinstance(values, tuple):
                    values = (values,)
                groups.append((values, sub.copy(), cls._context_label(
                    context_cols, values), context_cols))

        if (
            include_all_n
            and "n_drones" in context_cols
            and "n_drones" in df.columns
            and pd.to_numeric(df["n_drones"], errors="coerce").nunique(dropna=True) > 1
        ):
            collapsed_cols = [col for col in context_cols if col != "n_drones"]
            aggregate_groups: list[tuple[tuple,
                                         pd.DataFrame, str, list[str]]] = []
            if not collapsed_cols:
                aggregate_groups.append(
                    ((), df.copy(), "All N", collapsed_cols))
            else:
                grouped = df.groupby(collapsed_cols, dropna=False, sort=True)
                for values, sub in grouped:
                    if not isinstance(values, tuple):
                        values = (values,)
                    base = cls._context_label(collapsed_cols, values)
                    label = f"{base} | All N" if base != "All runs" else "All N"
                    aggregate_groups.append(
                        (values, sub.copy(), label, collapsed_cols))
            return aggregate_groups + groups
        return groups

    @staticmethod
    def _display_strategy(strategy: str, *, width: int = 16) -> str:
        label_map = {
            "always": "NAB",
            "frustration": "AFT",
            "frustration_threshold_adaptive": "AFT",
            "labella": "AFH",
            "labella_sortie_tuned": "AFH",
            "mappo": "MAPPO",
        }
        return fill(
            label_map.get(str(strategy), str(strategy).replace("_", " ")),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )

    @classmethod
    def _strategy_color_map(cls, strategies: list[str]) -> dict[str, str]:
        ordered = sorted({str(strategy) for strategy in strategies})
        return {
            strategy: cls.PALETTE[idx % len(cls.PALETTE)]
            for idx, strategy in enumerate(ordered)
        }

    @classmethod
    def _strategy_marker_map(cls, strategies: list[str]) -> dict[str, str]:
        ordered = sorted({str(strategy) for strategy in strategies})
        return {
            strategy: cls.MARKERS[idx % len(cls.MARKERS)]
            for idx, strategy in enumerate(ordered)
        }

    @classmethod
    def _ordered_strategies(cls, df: pd.DataFrame) -> list[str]:
        if "strategy" not in df.columns:
            return []
        return sorted(df["strategy"].dropna().astype(str).unique())

    @classmethod
    def _is_meta_strategy(cls, strategy: str) -> bool:
        return str(strategy) in cls.META_STRATEGIES

    @staticmethod
    def _legend_handles(color_map: dict[str, str], strategies: list[str]):
        from matplotlib.lines import Line2D

        return [
            Line2D(
                [0],
                [0],
                color=color_map[strategy],
                linewidth=2.3,
                marker="o",
                markersize=6,
                label=strategy,
            )
            for strategy in strategies
        ]

    @classmethod
    def _pareto_mode_groups(
        cls,
        df: pd.DataFrame,
        mode: str,
    ) -> list[tuple[tuple, pd.DataFrame, str, list[str]]]:
        if mode != "overall" or "scenario" not in df.columns:
            return cls._iter_mode_groups(df, mode)

        scenarios = [
            str(scenario)
            for scenario in df["scenario"].dropna().astype(str).unique()
        ]
        if len(scenarios) != 2:
            return cls._iter_mode_groups(df, mode)

        scenarios = sorted(
            scenarios,
            key=lambda scenario: (
                cls.SCENARIO_PANEL_ORDER.get(scenario, 99),
                scenario,
            ),
        )
        titles = ["Static", "Dynamic"]
        return [
            (
                (scenario,),
                df[df["scenario"].astype(str) == scenario].copy(),
                titles[idx],
                ["scenario"],
            )
            for idx, scenario in enumerate(scenarios)
        ]

    @staticmethod
    def _rank_rows(sub: pd.DataFrame) -> pd.DataFrame:
        return sub.sort_values(
            ["fuel_preserved_mean", "active_exposure_mean", "strategy"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
        valid = values.notna() & weights.notna()
        if not valid.any():
            return float("nan")
        w = weights.loc[valid].astype(float)
        if float(w.sum()) <= 0.0:
            return float(values.loc[valid].astype(float).mean())
        return float(np.average(values.loc[valid].astype(float), weights=w))

    @classmethod
    def _collapse_strategy_rows(cls, sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty or "strategy" not in sub.columns:
            return sub.copy()
        if sub["strategy"].astype(str).nunique() == len(sub):
            return sub.copy()

        weights = (
            pd.to_numeric(sub["n_runs"], errors="coerce").fillna(1.0)
            if "n_runs" in sub.columns
            else pd.Series(1.0, index=sub.index, dtype=float)
        )
        mean_like = {
            col
            for col in sub.columns
            if col.endswith("_mean")
            or col.endswith("_std")
            or col.endswith("_ci95")
            or col in {"fire_out_rate", "stand_down_success_rate", "n_runs"}
        }
        rows: list[dict] = []
        for strategy, strat_df in sub.groupby("strategy", sort=True, dropna=False):
            strat_weights = weights.loc[strat_df.index]
            row: dict[str, object] = {"strategy": strategy}
            for col in strat_df.columns:
                if col == "strategy":
                    continue
                if col == "n_runs":
                    row[col] = float(strat_weights.sum())
                    continue
                if col in mean_like:
                    row[col] = cls._weighted_average(
                        strat_df[col], strat_weights)
                    continue
                unique = strat_df[col].dropna().unique()
                row[col] = unique[0] if len(unique) == 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def preserve_first_winners(cls, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Return one preserve-first winner per available context."""
        required = {"strategy", "fuel_preserved_mean", "active_exposure_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            return pd.DataFrame()

        rows: list[dict] = []
        for values, sub, _, context_cols in cls._iter_context_groups(stats_df):
            ranked = cls._rank_rows(sub)
            if ranked.empty:
                continue
            winner = ranked.iloc[0].to_dict()
            winner["winner_rank"] = 1
            for idx, col in enumerate(context_cols):
                winner[col] = values[idx]
            rows.append(winner)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @staticmethod
    def _pareto_mask(sub: pd.DataFrame) -> pd.Series:
        keep = np.ones(len(sub), dtype=bool)
        values = sub.reset_index(drop=True)
        for i in range(len(values)):
            row_i = values.iloc[i]
            for j in range(len(values)):
                if i == j:
                    continue
                row_j = values.iloc[j]
                dominates = (
                    row_j["fuel_preserved_mean"] >= row_i["fuel_preserved_mean"]
                    and row_j["active_exposure_mean"] <= row_i["active_exposure_mean"]
                    and (
                        row_j["fuel_preserved_mean"] > row_i["fuel_preserved_mean"]
                        or row_j["active_exposure_mean"] < row_i["active_exposure_mean"]
                    )
                )
                if dominates:
                    keep[i] = False
                    break
        return pd.Series(keep, index=sub.index)

    @staticmethod
    def _paper_rc() -> dict[str, object]:
        return {
            "axes.facecolor": "white",
            "axes.edgecolor": "#3A3A3A",
            "axes.labelcolor": "#222222",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "font.size": 10,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }

    @staticmethod
    def _add_footer(fig, note: str) -> None:
        if not note:
            return
        fig.text(
            0.5,
            0.005,
            note,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="dimgray",
            wrap=True,
        )

    @staticmethod
    def _compose_note(*parts: str) -> str:
        return " | ".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def _percent_text(value: float, digits: int = 0) -> str:
        return f"{100.0 * float(value):.{digits}f}%"

    @staticmethod
    def _ci95(series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        n = len(clean)
        if n <= 1:
            return 0.0
        std = float(clean.std(ddof=1))
        return 1.96 * (std / (n ** 0.5))

    @staticmethod
    def _metric_ascending(metric: str) -> bool:
        return metric in {
            "active_exposure_mean",
            "active_exposure_frac",
            "cost_mean",
            "cost",
            "post_fire_active_exposure_mean",
            "stand_down_latency_steps_mean",
        }

    @classmethod
    def _sort_by_metric(cls, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        return df.sort_values(
            [metric, "strategy"],
            ascending=[cls._metric_ascending(metric), True],
            na_position="last",
        ).reset_index(drop=True)

    def _variant_filename(self, key: str, mode: str) -> Path:
        name = f"{self.FIGURE_STEMS[key]}_{self.VARIANT_SUFFIX[mode]}.png"
        return self.paper_dir / name

    def _save_variant_figure(self, fig, key: str, mode: str) -> str:
        output_path = self._variant_filename(key, mode)
        fig.savefig(output_path, dpi=600)
        return str(output_path)

    @classmethod
    def _mode_dims(cls, df: pd.DataFrame, mode: str) -> list[str]:
        available = cls._group_context_columns(df)
        if mode == "overall":
            return []
        if mode == "scenario":
            return [col for col in ["scenario"] if col in available]
        raise ValueError(f"unknown plot mode: {mode}")

    @classmethod
    def _iter_mode_groups(
        cls,
        df: pd.DataFrame,
        mode: str,
    ) -> list[tuple[tuple, pd.DataFrame, str, list[str]]]:
        dims = cls._mode_dims(df, mode)
        if not dims:
            return [((), df.copy(), "All runs", dims)]
        groups: list[tuple[tuple, pd.DataFrame, str, list[str]]] = []
        grouped = df.groupby(dims, dropna=False, sort=True)
        for values, sub in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            groups.append(
                (values, sub.copy(), cls._context_label(dims, values), dims))
        return groups

    @classmethod
    def _fixed_context_note_for_mode(cls, df: pd.DataFrame, mode: str) -> str:
        group_cols = cls._group_context_columns(df)
        dims = cls._mode_dims(df, mode)
        fixed_cols = [col for col in group_cols if col not in dims]
        parts: list[str] = []
        for col in fixed_cols:
            unique = df[col].dropna().unique()
            if len(unique) != 1:
                continue
            parts.append(cls._display_context_value(col, unique[0]))
        return " | ".join(parts)

    @staticmethod
    def _flatten_axes(axes) -> list:
        if isinstance(axes, np.ndarray):
            return list(axes.flat)
        return [axes]

    @staticmethod
    def _ensure_2d_axes(axes, nrows: int, ncols: int) -> np.ndarray:
        arr = np.asarray(axes, dtype=object)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(nrows, ncols)
        return arr

    @staticmethod
    def _step_means(step_df: pd.DataFrame, value_cols: list[str], context_cols: list[str]) -> pd.DataFrame:
        required = {"strategy", "step", *value_cols, *context_cols}
        if step_df.empty or not required.issubset(step_df.columns):
            return pd.DataFrame()
        return (
            step_df
            .groupby([*context_cols, "strategy", "step"], dropna=False, sort=True)[value_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

    @staticmethod
    def _collapse_step_series(sub: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        if sub.empty:
            return sub.copy()
        required = {"strategy", "step", *value_cols}
        if not required.issubset(sub.columns):
            return sub.copy()
        grouped = (
            sub.groupby(["strategy", "step"], dropna=False,
                        sort=True)[value_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        return grouped

    @staticmethod
    def _collapse_scaling_series(sub: pd.DataFrame, value_col: str) -> pd.DataFrame:
        required = {"strategy", "n_drones", value_col}
        if sub.empty or not required.issubset(sub.columns):
            return sub.copy()
        return (
            sub.groupby(["strategy", "n_drones"],
                        dropna=False, sort=True)[value_col]
            .mean(numeric_only=True)
            .reset_index()
        )

    @classmethod
    def _phase_activity_stats(cls, step_df: pd.DataFrame) -> pd.DataFrame:
        required = {"run_id", "strategy",
                    "active_ratio_mean", "fire_coverage_mean"}
        if step_df.empty or not required.issubset(step_df.columns):
            return pd.DataFrame()

        context_cols = cls._group_context_columns(step_df)
        run_group_cols = [*context_cols, "strategy", "run_id"]
        rows: list[dict[str, object]] = []
        for values, sub in step_df.groupby(run_group_cols, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = {col: values[idx] for idx, col in enumerate(run_group_cols)}
            active = pd.to_numeric(sub["active_ratio_mean"], errors="coerce")
            fire = pd.to_numeric(sub["fire_coverage_mean"], errors="coerce")
            fire_mask = fire > cls.FIRE_PRESENT_THRESHOLD
            row["during_fire_active_ratio_mean"] = float(
                active.loc[fire_mask].mean()) if fire_mask.any() else np.nan
            row["post_fire_active_ratio_mean"] = float(
                active.loc[~fire_mask].mean()) if (~fire_mask).any() else np.nan
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        run_level = pd.DataFrame(rows)
        agg_cols = [*context_cols, "strategy"]
        summary = (
            run_level
            .groupby(agg_cols, dropna=False, sort=True)
            .agg(
                n_runs=("run_id", "nunique"),
                during_fire_active_ratio_mean=(
                    "during_fire_active_ratio_mean", "mean"),
                during_fire_active_ratio_ci95=(
                    "during_fire_active_ratio_mean", cls._ci95),
                post_fire_active_ratio_mean=(
                    "post_fire_active_ratio_mean", "mean"),
                post_fire_active_ratio_ci95=(
                    "post_fire_active_ratio_mean", cls._ci95),
            )
            .reset_index()
        )
        return summary

    @staticmethod
    def _prepare_attrition_step_df(step_df: pd.DataFrame) -> pd.DataFrame:
        required = {"attrition_events_mean", "attrition_total_cumulative_mean"}
        if step_df.empty or not required.issubset(step_df.columns):
            return pd.DataFrame()

        prepared = step_df.copy()
        denom = (
            pd.to_numeric(prepared["initial_alive_mean"], errors="coerce")
            if "initial_alive_mean" in prepared.columns
            else pd.Series(np.nan, index=prepared.index, dtype=float)
        )
        if "n_drones" in prepared.columns:
            denom = denom.where(
                denom > 0.0,
                pd.to_numeric(prepared["n_drones"], errors="coerce"),
            )
        denom = denom.where(denom > 0.0)

        prepared["attrition_events_share"] = (
            pd.to_numeric(prepared["attrition_events_mean"],
                          errors="coerce") / denom
        )
        prepared["attrition_total_share"] = (
            pd.to_numeric(
                prepared["attrition_total_cumulative_mean"], errors="coerce"
            )
            / denom
        )
        return prepared

    def plot_suite(
        self,
        run_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        step_df: pd.DataFrame,
    ) -> list[str]:
        """Render the release paper figure suite.

        The release keeps only the figures used by the paper:
        02_tradeoff_pareto_overall,
        13_population_performance_cost_combined_overall, and
        14_population_active_ratio_scaling_overall.
        Figures skip cleanly when data is missing.
        """

        generated: list[str] = []
        jobs = [
            lambda: self.plot_tradeoff_pareto(stats_df, mode="overall"),
            lambda: self.plot_population_performance_cost_combined(
                stats_df,
                mode="overall",
            ),
            lambda: self.plot_population_active_ratio_scaling(
                stats_df,
                mode="overall",
            ),
        ]
        for job in jobs:
            try:
                result = job()
            except Exception as exc:
                self._warn(f"failed to render paper figure: {exc}")
                continue
            if not result:
                continue
            if isinstance(result, list):
                generated.extend(str(item) for item in result)
            else:
                generated.append(str(result))
        return generated

    @classmethod
    def _active_density_summary(cls, step_df: pd.DataFrame) -> pd.DataFrame:
        required = {"run_id", "strategy", "active_agents_mean", "grid_size"}
        if step_df.empty or not required.issubset(step_df.columns):
            return pd.DataFrame()

        prepared = step_df.copy()
        active = pd.to_numeric(prepared["active_agents_mean"], errors="coerce")
        grid = pd.to_numeric(prepared["grid_size"], errors="coerce")
        denom = grid.pow(2).where(grid > 0.0, np.nan)
        prepared["active_density"] = active / denom

        context_cols = cls._group_context_columns(prepared)
        run_group_cols = [*context_cols, "strategy", "run_id"]
        run_level = (
            prepared
            .groupby(run_group_cols, dropna=False, sort=True)["active_density"]
            .mean()
            .reset_index(name="active_density_run_mean")
        )

        summary = (
            run_level
            .groupby([*context_cols, "strategy"], dropna=False, sort=True)
            .agg(
                n_runs=("run_id", "nunique"),
                active_density_mean=("active_density_run_mean", "mean"),
                active_density_ci95=("active_density_run_mean", cls._ci95),
            )
            .reset_index()
        )
        return summary

    def plot_average_density(self, step_df: pd.DataFrame, *, mode: str) -> str | None:
        density_df = self._active_density_summary(step_df)
        required = {"strategy", "active_density_mean"}
        if density_df.empty or not required.issubset(density_df.columns):
            self._warn("skipping density figure: missing active density inputs")
            return None
        if mode == "scenario" and "scenario" not in density_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None

        groups = self._iter_mode_groups(density_df, mode)
        fixed_note = self._fixed_context_note_for_mode(density_df, mode)
        strategies = self._ordered_strategies(density_df)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)
        ncols = 1 if n_panels == 1 else 2
        nrows = ceil(n_panels / ncols)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(6.8 * ncols, 2.7 * nrows + 1.0),
                squeeze=False,
            )
            flat_axes = self._flatten_axes(axes)

            for ax, (_, sub, title, _) in zip(flat_axes, groups):
                ranked = (
                    self._collapse_strategy_rows(sub)
                    .sort_values(
                        ["active_density_mean", "strategy"],
                        ascending=[False, True],
                        na_position="last",
                    )
                    .reset_index(drop=True)
                )
                y = np.arange(len(ranked))
                ci = (
                    ranked["active_density_ci95"].astype(float).to_numpy()
                    if "active_density_ci95" in ranked.columns else None
                )
                colors = [color_map[str(strategy)]
                          for strategy in ranked["strategy"]]
                ax.barh(
                    y,
                    ranked["active_density_mean"].astype(float).to_numpy(),
                    xerr=ci,
                    color=colors,
                    edgecolor="#2F2F2F",
                    linewidth=0.7,
                )
                ax.set_yticks(y)
                ax.set_yticklabels([
                    self._display_strategy(strategy)
                    for strategy in ranked["strategy"]
                ])
                ax.invert_yaxis()
                if len(groups) > 1 or title != "All runs":
                    ax.set_title(title, loc="left", fontweight="bold", pad=8)
                ax.set_xlabel(self.DENSITY_LABEL)
                xmax = float(
                    np.nanmax(
                        ranked["active_density_mean"].astype(float).to_numpy()
                        + (
                            ranked["active_density_ci95"].astype(
                                float).to_numpy()
                            if "active_density_ci95" in ranked.columns
                            else 0.0
                        )
                    )
                )
                ax.set_xlim(0.0, xmax * 1.18 if xmax > 0.0 else 1.0)
                ax.tick_params(axis="both", labelsize=9)
                ax.grid(axis="x", visible=True)
                ax.grid(axis="y", visible=False)

            for ax in flat_axes[len(groups):]:
                ax.set_visible(False)

            fig.suptitle("Average active density", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.DENSITY_NOTE,
                    self.DENSITY_DEF,
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
            output = self._save_variant_figure(fig, "density", mode)
            plt.close(fig)
            return output

    def plot_headline_ranking(self, stats_df: pd.DataFrame, *, mode: str) -> str | None:
        required = {"strategy", "fuel_preserved_mean", "active_exposure_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping headline ranking: missing required aggregated metrics")
            return None
        if mode == "scenario" and "scenario" not in stats_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        groups = self._iter_mode_groups(stats_df, mode)
        fixed_note = self._fixed_context_note_for_mode(stats_df, mode)
        strategies = stats_df["strategy"].astype(str).tolist()
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)
        ncols = 1 if n_panels == 1 else 2
        nrows = ceil(n_panels / ncols)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(6.8 * ncols, 2.7 * nrows + 1.0),
                squeeze=False,
            )
            flat_axes = self._flatten_axes(axes)

            for ax, (_, sub, title, _) in zip(flat_axes, groups):
                ranked = self._rank_rows(self._collapse_strategy_rows(sub))
                y = np.arange(len(ranked))
                ci = (
                    ranked["fuel_preserved_ci95"].astype(float).to_numpy()
                    if "fuel_preserved_ci95" in ranked.columns else None
                )
                colors = [color_map[str(strategy)]
                          for strategy in ranked["strategy"]]
                colors = [
                    color if idx == 0 else f"{color}CC"
                    for idx, color in enumerate(colors)
                ]
                ax.barh(
                    y,
                    ranked["fuel_preserved_mean"].astype(float).to_numpy(),
                    xerr=ci,
                    color=colors,
                    edgecolor="#2F2F2F",
                    linewidth=0.7,
                )
                ax.set_yticks(y)
                ax.set_yticklabels([
                    self._display_strategy(strategy)
                    for strategy in ranked["strategy"]
                ])
                ax.invert_yaxis()
                if len(groups) > 1 or title != "All runs":
                    ax.set_title(title, loc="left", fontweight="bold", pad=8)
                ax.set_xlabel(self.FUEL_LABEL)
                xmax = float(
                    np.nanmax(
                        ranked["fuel_preserved_mean"].astype(float).to_numpy()
                        + (ci if ci is not None else 0.0)
                    )
                )
                ax.set_xlim(0.0, min(1.10, xmax + 0.08))
                ax.xaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0))
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="both", labelsize=9)

                for idx, row in ranked.iterrows():
                    ax.text(
                        0.98,
                        idx,
                        f"exposure ratio {self._percent_text(float(row['active_exposure_mean']), 1)}",
                        transform=ax.get_yaxis_transform(),
                        va="center",
                        ha="right",
                        fontsize=8.5,
                        color="#3A3A3A",
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": 0.7, "pad": 0.3},
                    )

            for ax in flat_axes[len(groups):]:
                ax.set_visible(False)

            fig.suptitle("Preserve-first ranking", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.PRIMARY_NOTE,
                    self.FUEL_DEF,
                    self.ACTIVE_DEF,
                    "Right-edge labels show exposure ratio only.",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
            output = self._save_variant_figure(fig, "headline", mode)
            plt.close(fig)
            return output

    def plot_tradeoff_pareto(self, stats_df: pd.DataFrame, *, mode: str) -> str | None:
        required = {"strategy", "fuel_preserved_mean", "active_exposure_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping Pareto figure: missing preserve/exposure aggregates")
            return None
        if mode == "scenario" and "scenario" not in stats_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FormatStrFormatter, PercentFormatter

        groups = self._pareto_mode_groups(stats_df, mode)
        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)
        marker_map = self._strategy_marker_map(strategies)
        n_panels = len(groups)
        ncols = 1 if n_panels == 1 else 2
        nrows = ceil(n_panels / ncols)
        x_ci = (
            pd.to_numeric(stats_df["active_exposure_ci95"],
                          errors="coerce").fillna(0.0)
            if "active_exposure_ci95" in stats_df.columns else
            pd.Series(0.0, index=stats_df.index, dtype=float)
        )
        y_ci = (
            pd.to_numeric(stats_df["fuel_preserved_ci95"],
                          errors="coerce").fillna(0.0)
            if "fuel_preserved_ci95" in stats_df.columns else
            pd.Series(0.0, index=stats_df.index, dtype=float)
        )
        x_max = float(
            np.nanmax(
                pd.to_numeric(
                    stats_df["active_exposure_mean"], errors="coerce").to_numpy()
                + x_ci.to_numpy()
            )
        )
        y_max = float(
            np.nanmax(
                pd.to_numeric(stats_df["fuel_preserved_mean"],
                              errors="coerce").to_numpy()
                + y_ci.to_numpy()
            )
        )

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(6.4 * ncols, 4.4 * nrows + 1.2),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            flat_axes = self._flatten_axes(axes)

            for panel_idx, (ax, (_, sub, title, _)) in enumerate(zip(flat_axes, groups)):
                ranked = self._rank_rows(self._collapse_strategy_rows(sub))
                pareto_mask = self._pareto_mask(ranked)

                for _, row in ranked.iterrows():
                    strategy = str(row["strategy"])
                    x = float(row["active_exposure_mean"])
                    y = float(row["fuel_preserved_mean"])
                    xerr = float(row.get("active_exposure_ci95", 0.0) or 0.0)
                    yerr = float(row.get("fuel_preserved_ci95", 0.0) or 0.0)
                    is_pareto = bool(pareto_mask.loc[row.name])
                    marker = marker_map[strategy]
                    size = 150
                    edge = "#111111" if is_pareto else "white"
                    ax.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        fmt="none",
                        ecolor="#999999",
                        elinewidth=1.0,
                        capsize=2.0,
                        zorder=2,
                    )
                    ax.scatter(
                        [x],
                        [y],
                        s=size,
                        marker=marker,
                        color=color_map[strategy],
                        edgecolors=edge,
                        linewidths=1.2,
                        zorder=3,
                    )

                if len(groups) > 1 or title != "All runs":
                    ax.set_title(
                        title,
                        loc="left",
                        fontweight="bold",
                        fontsize=14,
                        pad=8,
                    )
                ax.set_xlabel(r"Active ratio ($R$)", fontsize=13)
                ax.set_ylabel(
                    r"Vegetation preserved ($P$)" if panel_idx % ncols == 0 else "",
                    fontsize=13,
                )
                ax.set_xlim(0.0, x_max + max(0.03, 0.08 * x_max))
                ax.set_ylim(0.0, min(1.02, y_max + max(0.02, 0.05 * y_max)))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0))
                ax.tick_params(axis="both", labelsize=11)

            for ax in flat_axes[len(groups):]:
                ax.set_visible(False)

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[strategy],
                    color="none",
                    markerfacecolor=color_map[strategy],
                    markeredgecolor="#111111",
                    markeredgewidth=0.9,
                    markersize=10,
                    linestyle="None",
                    label=self._display_strategy(
                        strategy, width=18).replace("\n", " "),
                )
                for strategy in strategies
            ]
            fig.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, len(strategies)),
                fontsize=12,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.985,
                top=0.82,
                bottom=0.16,
                wspace=0.08,
            )
            output = self._save_variant_figure(fig, "pareto", mode)
            plt.close(fig)
            return output

    def plot_temporal_response(self, step_df: pd.DataFrame, *, mode: str) -> str | None:
        metrics = [
            ("fire_coverage_mean", "Fire coverage"),
            ("active_ratio_mean", "Active fraction"),
            ("firefighting_ratio_mean", "Firefighting fraction"),
        ]
        required = {"strategy", "step", *(metric for metric, _ in metrics)}
        if step_df.empty or not required.issubset(step_df.columns):
            self._warn(
                "skipping temporal response: missing step-level fire/activation metrics")
            return None
        if mode == "scenario" and "scenario" not in step_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None

        context_cols = self._group_context_columns(step_df)
        grouped = self._step_means(
            step_df, [metric for metric, _ in metrics], context_cols)
        if grouped.empty:
            self._warn(
                "skipping temporal response: unable to aggregate step metrics")
            return None

        groups = self._iter_mode_groups(grouped, mode)
        fixed_note = self._fixed_context_note_for_mode(grouped, mode)
        strategies = self._ordered_strategies(grouped)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(metrics),
                ncols=n_panels,
                figsize=(4.2 * n_panels, 2.35 * len(metrics) + 1.4),
                squeeze=False,
                sharex="col",
                sharey="row",
            )
            axes = self._ensure_2d_axes(axes, len(metrics), n_panels)

            for col_idx, (_, sub, title, _) in enumerate(groups):
                sub = self._collapse_step_series(
                    sub, [metric for metric, _ in metrics])
                last_step = int(pd.to_numeric(
                    sub["step"], errors="coerce").max())
                for row_idx, (metric, label) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    for strategy, strat_df in sub.groupby("strategy", sort=True):
                        strat_df = strat_df.sort_values("step")
                        x = pd.to_numeric(
                            strat_df["step"], errors="coerce").to_numpy()
                        y = pd.to_numeric(
                            strat_df[metric], errors="coerce").to_numpy()
                        ax.plot(x, y, linewidth=2.0,
                                color=color_map[str(strategy)])
                        ax.scatter([x[-1]], [y[-1]], s=18,
                                   color=color_map[str(strategy)], zorder=3)
                    ax.set_xlim(0.0, last_step)
                    if row_idx == 0 and (len(groups) > 1 or title != "All runs"):
                        ax.set_title(title, loc="left",
                                     fontweight="bold", pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(label)
                    if row_idx == len(metrics) - 1:
                        ax.set_xlabel("Step")
                    ax.grid(axis="y", visible=True)
                    ax.tick_params(axis="both", labelsize=8.5)

            fig.legend(
                self._legend_handles(color_map, strategies),
                strategies,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
            )
            fig.suptitle("Temporal response", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.TEMPORAL_NOTE,
                    "Active fraction and firefighting fraction are shares of alive drones at each step.",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
            output = self._save_variant_figure(fig, "temporal", mode)
            plt.close(fig)
            return output

    def plot_behavior_mechanism(self, step_df: pd.DataFrame, *, mode: str) -> str | None:
        metrics = [
            ("exploring_ratio_mean", "Exploring fraction"),
            ("returning_ratio_mean", "Returning fraction"),
            ("battery_mean_alive", "Battery (alive)"),
            ("payload_mean_alive", "Payload (alive)"),
        ]
        required = {"strategy", "step", *(metric for metric, _ in metrics)}
        if step_df.empty or not required.issubset(step_df.columns):
            self._warn(
                "skipping behavior mechanism: missing required observable step metrics")
            return None
        if mode == "scenario" and "scenario" not in step_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None

        context_cols = self._group_context_columns(step_df)
        grouped = self._step_means(
            step_df, [metric for metric, _ in metrics], context_cols)
        if grouped.empty:
            self._warn(
                "skipping behavior mechanism: unable to aggregate step metrics")
            return None

        groups = self._iter_mode_groups(grouped, mode)
        fixed_note = self._fixed_context_note_for_mode(grouped, mode)
        strategies = self._ordered_strategies(grouped)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(metrics),
                ncols=n_panels,
                figsize=(4.2 * n_panels, 2.35 * len(metrics) + 1.4),
                squeeze=False,
                sharex="col",
                sharey="row",
            )
            axes = self._ensure_2d_axes(axes, len(metrics), n_panels)

            for col_idx, (_, sub, title, _) in enumerate(groups):
                sub = self._collapse_step_series(
                    sub, [metric for metric, _ in metrics])
                last_step = int(pd.to_numeric(
                    sub["step"], errors="coerce").max())
                for row_idx, (metric, label) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    for strategy, strat_df in sub.groupby("strategy", sort=True):
                        strat_df = strat_df.sort_values("step")
                        x = pd.to_numeric(
                            strat_df["step"], errors="coerce").to_numpy()
                        y = pd.to_numeric(
                            strat_df[metric], errors="coerce").to_numpy()
                        ax.plot(x, y, linewidth=2.0,
                                color=color_map[str(strategy)])
                        ax.scatter([x[-1]], [y[-1]], s=18,
                                   color=color_map[str(strategy)], zorder=3)
                    ax.set_xlim(0.0, last_step)
                    if row_idx == 0 and (len(groups) > 1 or title != "All runs"):
                        ax.set_title(title, loc="left",
                                     fontweight="bold", pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(label)
                    if row_idx == len(metrics) - 1:
                        ax.set_xlabel("Step")
                    ax.grid(axis="y", visible=True)
                    ax.tick_params(axis="both", labelsize=8.5)

            fig.legend(
                self._legend_handles(color_map, strategies),
                strategies,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
            )
            fig.suptitle("Behavioral mechanism", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.MECHANISM_NOTE,
                    "Exploring and returning fractions are shares of alive drones at each step.",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
            output = self._save_variant_figure(fig, "mechanism", mode)
            plt.close(fig)
            return output

    def plot_adaptability_after_fire_out(self, step_df: pd.DataFrame, *, mode: str) -> str | None:
        phase_df = self._phase_activity_stats(step_df)
        required = {
            "strategy",
            "during_fire_active_ratio_mean",
            "post_fire_active_ratio_mean",
        }
        if phase_df.empty or not required.issubset(phase_df.columns):
            self._warn(
                "skipping adaptability figure: missing fire-phase activity metrics")
            return None
        if mode == "scenario" and "scenario" not in phase_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        groups = self._iter_mode_groups(phase_df, mode)
        fixed_note = self._fixed_context_note_for_mode(phase_df, mode)
        phase_colors = {
            "While fire present": "#4C78A8",
            "After fire-out": "#F58518",
        }

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(groups),
                ncols=1,
                figsize=(8.6, 2.9 * len(groups) + 1.2),
                squeeze=False,
            )
            axes = self._ensure_2d_axes(axes, len(groups), 1)

            for row_idx, (_, sub, title, _) in enumerate(groups):
                ax = axes[row_idx, 0]
                ranked = self._collapse_strategy_rows(sub).sort_values(
                    ["during_fire_active_ratio_mean",
                        "post_fire_active_ratio_mean", "strategy"],
                    ascending=[False, True, True],
                    na_position="last",
                ).reset_index(drop=True)
                y = np.arange(len(ranked))
                bar_h = 0.36
                during = pd.to_numeric(
                    ranked["during_fire_active_ratio_mean"], errors="coerce")
                post = pd.to_numeric(
                    ranked["post_fire_active_ratio_mean"], errors="coerce")
                during_ci = pd.to_numeric(
                    ranked.get("during_fire_active_ratio_ci95",
                               pd.Series(0.0, index=ranked.index)),
                    errors="coerce",
                ).fillna(0.0)
                post_ci = pd.to_numeric(
                    ranked.get("post_fire_active_ratio_ci95",
                               pd.Series(0.0, index=ranked.index)),
                    errors="coerce",
                ).fillna(0.0)

                during_mask = during.notna()
                post_mask = post.notna()
                ax.barh(
                    y[during_mask] - bar_h / 2.0,
                    during[during_mask],
                    height=bar_h,
                    xerr=during_ci[during_mask].to_numpy(),
                    color=phase_colors["While fire present"],
                    edgecolor="#2F2F2F",
                    linewidth=0.7,
                    label="While fire present" if row_idx == 0 else None,
                )
                ax.barh(
                    y[post_mask] + bar_h / 2.0,
                    post[post_mask],
                    height=bar_h,
                    xerr=post_ci[post_mask].to_numpy(),
                    color=phase_colors["After fire-out"],
                    edgecolor="#2F2F2F",
                    linewidth=0.7,
                    label="After fire-out" if row_idx == 0 else None,
                )

                ax.set_yticks(y)
                ax.set_yticklabels([self._display_strategy(strategy)
                                   for strategy in ranked["strategy"]])
                ax.invert_yaxis()
                if len(groups) > 1 or title != "All runs":
                    ax.set_title(title, loc="left", fontweight="bold", pad=8)
                xmax = float(np.nanmax(np.r_[during.fillna(0.0).to_numpy(
                ), post.fillna(0.0).to_numpy()])) if len(ranked) else 0.0
                ax.set_xlim(
                    0.0, min(1.02, xmax + max(0.04, 0.12 * xmax if xmax > 0 else 0.2)))
                ax.xaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0))
                ax.set_xlabel(self.ACTIVE_LABEL)
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="both", labelsize=9)

            fig.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=2,
                fontsize=9,
            )
            fig.suptitle("Activity before and after fire-out",
                         y=0.995, fontsize=14, fontweight="bold")
            self._add_footer(fig, self._compose_note(
                self.ADAPTABILITY_NOTE, fixed_note))
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93), h_pad=1.1)
            output = self._save_variant_figure(fig, "adaptability", mode)
            plt.close(fig)
            return output

    def plot_fire_out_rate(self, stats_df: pd.DataFrame, *, mode: str) -> str | None:
        required = {"strategy", "fire_out_rate"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping fire-out rate figure: missing fire-out metrics")
            return None
        if mode == "scenario" and "scenario" not in stats_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        groups = self._iter_mode_groups(stats_df, mode)
        fixed_note = self._fixed_context_note_for_mode(stats_df, mode)
        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(groups),
                ncols=1,
                figsize=(8.2, 2.7 * len(groups) + 1.2),
                squeeze=False,
            )
            axes = self._ensure_2d_axes(axes, len(groups), 1)

            for row_idx, (_, sub, title, _) in enumerate(groups):
                ax = axes[row_idx, 0]
                ranked = self._sort_by_metric(
                    self._collapse_strategy_rows(sub), "fire_out_rate")
                y = np.arange(len(ranked))
                values = pd.to_numeric(
                    ranked["fire_out_rate"], errors="coerce")
                ci = pd.to_numeric(
                    ranked.get("fire_out_rate_ci95", pd.Series(
                        0.0, index=ranked.index)),
                    errors="coerce",
                ).fillna(0.0)
                mask = values.notna()
                ax.barh(
                    y[mask],
                    values[mask],
                    xerr=ci[mask].to_numpy(),
                    color=[color_map[str(strategy)]
                           for strategy in ranked.loc[mask, "strategy"]],
                    edgecolor="#2F2F2F",
                    linewidth=0.7,
                )
                ax.set_yticks(y)
                ax.set_yticklabels([self._display_strategy(strategy)
                                   for strategy in ranked["strategy"]])
                ax.invert_yaxis()
                if len(groups) > 1 or title != "All runs":
                    ax.set_title(title, loc="left", fontweight="bold", pad=8)
                ax.set_xlim(0.0, 1.02)
                ax.xaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0, decimals=0))
                ax.set_xlabel("Fire-out rate (%)")
                ax.grid(axis="y", visible=False)
                ax.tick_params(axis="both", labelsize=9)

            fig.suptitle("Fire-out rate", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(fig, self._compose_note(
                self.FIRE_OUT_NOTE, fixed_note))
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93), h_pad=1.1)
            output = self._save_variant_figure(fig, "fire_out", mode)
            plt.close(fig)
            return output

    def plot_population_scaling(self, stats_df: pd.DataFrame, *, mode: str) -> str | None:
        required = {"strategy", "n_drones",
                    "fuel_preserved_mean", "active_exposure_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping population scaling: missing N-dependent aggregates")
            return None
        if pd.to_numeric(stats_df["n_drones"], errors="coerce").nunique(dropna=True) <= 1:
            return None
        if mode == "scenario" and "scenario" not in stats_df.columns:
            return None
        if mode not in {"overall", "scenario"}:
            return None

        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        metrics = [
            ("fuel_preserved_mean", self.FUEL_LABEL),
            ("active_exposure_mean", self.ACTIVE_LABEL),
        ]
        if "post_transient_active_headcount_mean" in stats_df.columns:
            metrics.append(
                ("post_transient_active_headcount_mean", self.HEADCOUNT_LABEL)
            )
        if "utility_mean" in stats_df.columns:
            metrics.append(("utility_mean", self.UTILITY_LABEL))
        groups = self._iter_mode_groups(stats_df, mode)
        fixed_note = self._fixed_context_note_for_mode(stats_df, mode)
        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(metrics),
                ncols=n_panels,
                figsize=(4.8 * n_panels, 2.8 * len(metrics) + 1.4),
                squeeze=False,
                sharex="col",
            )
            axes = self._ensure_2d_axes(axes, len(metrics), n_panels)

            for col_idx, (_, sub, title, _) in enumerate(groups):
                for row_idx, (metric, label) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    scaling_df = self._collapse_scaling_series(sub, metric)
                    for strategy, strat_df in scaling_df.groupby("strategy", sort=True):
                        strat_df = strat_df.sort_values("n_drones")
                        x = pd.to_numeric(
                            strat_df["n_drones"], errors="coerce").to_numpy()
                        y = pd.to_numeric(
                            strat_df[metric], errors="coerce").to_numpy()
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            linewidth=2.0,
                            color=color_map[str(strategy)],
                        )
                    if row_idx == 0:
                        if len(groups) > 1 or title != "All runs":
                            ax.set_title(title, loc="left",
                                         fontweight="bold", pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(label)
                    if row_idx == len(metrics) - 1:
                        ax.set_xlabel("Number of drones")
                    ax.grid(axis="y", visible=True)
                    if metric in {"fuel_preserved_mean", "active_exposure_mean"}:
                        ax.yaxis.set_major_formatter(
                            PercentFormatter(xmax=1.0, decimals=0)
                        )
                    ax.tick_params(axis="both", labelsize=8.5)

            fig.legend(
                self._legend_handles(color_map, strategies),
                strategies,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
            )
            fig.suptitle("Population scaling", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.SCALING_NOTE,
                    self.FUEL_DEF,
                    self.ACTIVE_DEF,
                    self.HEADCOUNT_DEF if "post_transient_active_headcount_mean" in stats_df.columns else "",
                    self.UTILITY_DEF if "utility_mean" in stats_df.columns else "",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
            output = self._save_variant_figure(fig, "population", mode)
            plt.close(fig)
            return output

    @staticmethod
    def _cost_tick(value: float, _: int) -> str:
        if abs(value) >= 1000.0:
            return f"{value / 1000.0:.0f}k"
        return f"{value:.0f}"

    @staticmethod
    def _set_split_ylabel(ax, main: str, detail: str) -> None:
        ax.set_ylabel("")
        ax.text(
            -0.105,
            0.5,
            main,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            color="#222222",
        )
        ax.text(
            -0.067,
            0.5,
            detail,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#777777",
        )

    def _population_performance_series(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        weights = (
            pd.to_numeric(stats_df["n_runs"], errors="coerce").fillna(1.0)
            if "n_runs" in stats_df.columns
            else pd.Series(1.0, index=stats_df.index, dtype=float)
        )
        rows: list[dict[str, object]] = []
        for (strategy, n_drones), sub in stats_df.groupby(
            ["strategy", "n_drones"],
            dropna=False,
            sort=True,
        ):
            sub_weights = weights.loc[sub.index]
            row: dict[str, object] = {
                "strategy": str(strategy),
                "n_drones": n_drones,
                "fuel_preserved_mean": self._weighted_average(
                    pd.to_numeric(sub["fuel_preserved_mean"], errors="coerce"),
                    sub_weights,
                ),
            }
            if "fuel_preserved_ci95" in sub.columns:
                row["fuel_preserved_ci95"] = self._weighted_average(
                    pd.to_numeric(sub["fuel_preserved_ci95"], errors="coerce"),
                    sub_weights,
                )
            else:
                row["fuel_preserved_ci95"] = 0.0
            rows.append(row)

        performance = pd.DataFrame(rows)
        if performance.empty:
            return performance
        performance["n_drones"] = pd.to_numeric(
            performance["n_drones"],
            errors="coerce",
        )
        performance["fuel_preserved_mean"] = pd.to_numeric(
            performance["fuel_preserved_mean"],
            errors="coerce",
        )
        performance["fuel_preserved_ci95"] = pd.to_numeric(
            performance["fuel_preserved_ci95"],
            errors="coerce",
        ).fillna(0.0)
        return performance.dropna(
            subset=["strategy", "n_drones", "fuel_preserved_mean"]
        )

    def _population_cost_series(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        weights = (
            pd.to_numeric(stats_df["n_runs"], errors="coerce").fillna(1.0)
            if "n_runs" in stats_df.columns
            else pd.Series(1.0, index=stats_df.index, dtype=float)
        )
        rows: list[dict[str, object]] = []
        for (strategy, n_drones), sub in stats_df.groupby(
            ["strategy", "n_drones"],
            dropna=False,
            sort=True,
        ):
            sub_weights = weights.loc[sub.index]
            rows.append(
                {
                    "strategy": str(strategy),
                    "n_drones": n_drones,
                    "cost_mean": self._weighted_average(
                        pd.to_numeric(sub["cost_mean"], errors="coerce"),
                        sub_weights,
                    ),
                }
            )

        cost = pd.DataFrame(rows)
        if cost.empty:
            return cost
        cost["n_drones"] = pd.to_numeric(cost["n_drones"], errors="coerce")
        cost["cost_mean"] = pd.to_numeric(cost["cost_mean"], errors="coerce")
        return cost.dropna(subset=["strategy", "n_drones", "cost_mean"])

    def _population_active_ratio_series(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        weights = (
            pd.to_numeric(stats_df["n_runs"], errors="coerce").fillna(1.0)
            if "n_runs" in stats_df.columns
            else pd.Series(1.0, index=stats_df.index, dtype=float)
        )
        rows: list[dict[str, object]] = []
        for (strategy, n_drones), sub in stats_df.groupby(
            ["strategy", "n_drones"],
            dropna=False,
            sort=True,
        ):
            sub_weights = weights.loc[sub.index]
            rows.append(
                {
                    "strategy": str(strategy),
                    "n_drones": n_drones,
                    "active_ratio_mean": self._weighted_average(
                        pd.to_numeric(
                            sub["active_exposure_mean"],
                            errors="coerce",
                        ),
                        sub_weights,
                    ),
                }
            )

        active_ratio = pd.DataFrame(rows)
        if active_ratio.empty:
            return active_ratio
        active_ratio["n_drones"] = pd.to_numeric(
            active_ratio["n_drones"],
            errors="coerce",
        )
        active_ratio["active_ratio_mean"] = pd.to_numeric(
            active_ratio["active_ratio_mean"],
            errors="coerce",
        )
        return active_ratio.dropna(
            subset=["strategy", "n_drones", "active_ratio_mean"]
        )

    def _population_legend_handles(
        self,
        strategies: list[str],
        color_map: dict[str, str],
        marker_map: dict[str, str],
    ):
        from matplotlib.lines import Line2D

        return [
            Line2D(
                [0],
                [0],
                marker=marker_map[strategy],
                color=color_map[strategy],
                markerfacecolor=color_map[strategy],
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                markersize=7.0,
                linewidth=2.1,
                label=self._display_strategy(
                    strategy, width=18).replace("\n", " "),
            )
            for strategy in strategies
        ]

    def plot_population_performance_cost_combined(
        self,
        stats_df: pd.DataFrame,
        *,
        mode: str,
    ) -> str | None:
        required = {"strategy", "n_drones", "fuel_preserved_mean", "cost_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping combined performance/cost figure: missing required aggregates"
            )
            return None
        if pd.to_numeric(stats_df["n_drones"], errors="coerce").nunique(dropna=True) <= 1:
            return None
        if mode != "overall":
            return None

        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import FuncFormatter, PercentFormatter

        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)
        marker_map = self._strategy_marker_map(strategies)
        performance = self._population_performance_series(stats_df)
        cost = self._population_cost_series(stats_df)
        if performance.empty or cost.empty:
            return None

        n_values = sorted(
            set(performance["n_drones"].dropna().unique())
            | set(cost["n_drones"].dropna().unique())
        )
        if not n_values:
            return None
        group_spacing = 1.38
        centers = np.arange(len(n_values), dtype=float) * group_spacing
        n_lookup = {value: center for value, center in zip(n_values, centers)}
        cluster_width = 0.78
        bar_width = min(0.18, cluster_width / max(1, len(strategies)))
        y_upper = float(performance["fuel_preserved_mean"].max(skipna=True))
        y_limit = min(1.02, max(0.05, y_upper) * 1.08)

        with plt.rc_context(self._paper_rc()):
            fig, (ax_perf, ax_cost) = plt.subplots(
                2,
                1,
                figsize=(9.6, 7.2),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 0.12},
            )

            for strategy in strategies:
                series = performance[performance["strategy"] == strategy].sort_values(
                    "n_drones"
                )
                if series.empty:
                    continue
                x = np.array(
                    [n_lookup[n] for n in series["n_drones"]],
                    dtype=float,
                )
                y = pd.to_numeric(
                    series["fuel_preserved_mean"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    continue
                color = color_map[strategy]
                ax_perf.plot(
                    x[valid],
                    y[valid],
                    marker=marker_map[strategy],
                    markersize=6.5,
                    linewidth=2.1,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="#111111",
                    markeredgewidth=0.8,
                    zorder=3,
                )

            for n_drones in n_values:
                n_df = (
                    cost[cost["n_drones"] == float(n_drones)]
                    .sort_values(["cost_mean", "strategy"], ascending=[True, True])
                    .reset_index(drop=True)
                )
                if n_df.empty:
                    continue
                local_offsets = (
                    np.array([0.0], dtype=float)
                    if len(n_df) == 1
                    else np.linspace(-cluster_width / 2.0, cluster_width / 2.0, len(n_df))
                )
                for row_idx, row in n_df.iterrows():
                    strategy = str(row["strategy"])
                    ax_cost.bar(
                        n_lookup[n_drones] + local_offsets[row_idx],
                        float(row["cost_mean"]),
                        width=bar_width,
                        color=color_map[strategy],
                        edgecolor="#222222",
                        linewidth=0.6,
                        alpha=0.92,
                        zorder=3,
                    )

            self._set_split_ylabel(
                ax_perf, r"Performance ($P$)", "vegetation preserved")
            ax_perf.set_ylim(0.0, y_limit)
            ax_perf.yaxis.set_major_formatter(
                PercentFormatter(xmax=1.0, decimals=0))
            ax_perf.grid(axis="y", visible=True)
            ax_perf.grid(axis="x", visible=False)
            ax_perf.tick_params(axis="both", labelsize=9, labelbottom=False)

            self._set_split_ylabel(
                ax_cost, r"Cost ($J$)", "active-agents step")
            ax_cost.set_xlabel(r"Population Size ($\bar{n}$)")
            ax_cost.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
            ax_cost.set_xticks(centers)
            ax_cost.set_xticklabels([str(int(n)) for n in n_values])
            ax_cost.yaxis.set_major_formatter(FuncFormatter(self._cost_tick))
            ax_cost.grid(axis="y", visible=True)
            ax_cost.grid(axis="x", visible=False)
            ax_cost.tick_params(axis="both", labelsize=9)

            handles = self._population_legend_handles(
                strategies, color_map, marker_map)
            fig.legend(
                handles,
                [handle.get_label() for handle in handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, max(1, len(handles))),
                fontsize=9,
                frameon=False,
            )
            fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.1)
            output = self._save_variant_figure(
                fig,
                "population_perf_cost_combined",
                mode,
            )
            plt.close(fig)
            return output

    def plot_population_active_ratio_scaling(
        self,
        stats_df: pd.DataFrame,
        *,
        mode: str,
    ) -> str | None:
        required = {"strategy", "n_drones", "active_exposure_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping N-wise active-ratio figure: missing required aggregates")
            return None
        if pd.to_numeric(stats_df["n_drones"], errors="coerce").nunique(dropna=True) <= 1:
            return None
        if mode != "overall":
            return None

        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import FormatStrFormatter

        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)
        marker_map = self._strategy_marker_map(strategies)
        active_ratio = self._population_active_ratio_series(stats_df)
        if active_ratio.empty:
            return None

        n_values = sorted(active_ratio["n_drones"].dropna().unique())
        y_upper = float(active_ratio["active_ratio_mean"].max(skipna=True))
        y_limit = min(1.02, max(0.05, y_upper) * 1.12)

        with plt.rc_context(self._paper_rc()):
            fig, ax = plt.subplots(figsize=(9.4, 4.4))

            for strategy in strategies:
                series = active_ratio[active_ratio["strategy"] == strategy].sort_values(
                    "n_drones"
                )
                if series.empty:
                    continue
                x = pd.to_numeric(series["n_drones"],
                                  errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(
                    series["active_ratio_mean"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    continue
                color = color_map[strategy]
                ax.plot(
                    x[valid],
                    y[valid],
                    marker=marker_map[strategy],
                    markersize=6.5,
                    linewidth=2.1,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="#111111",
                    markeredgewidth=0.8,
                    zorder=3,
                )

            ax.set_xlabel(r"Population Size ($\bar{n}$)")
            ax.set_ylabel(r"Active ratio ($R$)", labelpad=12)
            x_pad = max(1.0, 0.03 * (float(n_values[-1]) - float(n_values[0])))
            ax.set_xlim(float(n_values[0]) - x_pad,
                        float(n_values[-1]) + x_pad)
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(int(n)) for n in n_values])
            ax.set_ylim(0.0, y_limit)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.grid(axis="y", visible=True)
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="both", labelsize=9)

            legend_handles = self._population_legend_handles(
                strategies,
                color_map,
                marker_map,
            )
            fig.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
                frameon=False,
            )
            fig.subplots_adjust(left=0.14, right=0.985, top=0.86, bottom=0.14)
            output = self._save_variant_figure(
                fig, "population_active_ratio", mode)
            plt.close(fig)
            return output

    def plot_population_cost_scaling(self, stats_df: pd.DataFrame, *, mode: str) -> str | None:
        required = {"strategy", "n_drones", "cost_mean"}
        if stats_df.empty or not required.issubset(stats_df.columns):
            self._warn(
                "skipping cost scaling: missing N-dependent cost aggregates")
            return None
        if pd.to_numeric(stats_df["n_drones"], errors="coerce").nunique(dropna=True) <= 1:
            return None
        if mode == "scenario" and "scenario" not in stats_df.columns:
            return None
        if mode not in {"overall", "scenario"}:
            return None

        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter

        groups = self._iter_mode_groups(stats_df, mode)
        strategies = self._ordered_strategies(stats_df)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)
        n_values = sorted(
            pd.to_numeric(stats_df["n_drones"], errors="coerce")
            .dropna()
            .unique()
        )
        group_spacing = 1.38
        centers = np.arange(len(n_values), dtype=float) * group_spacing
        n_lookup = {value: center for value, center in zip(n_values, centers)}
        cluster_width = 0.78
        bar_width = min(0.18, cluster_width / max(1, len(strategies)))

        def cost_tick(value: float, _: int) -> str:
            if abs(value) >= 1000.0:
                return f"{value / 1000.0:.0f}k"
            return f"{value:.0f}"

        with plt.rc_context(self._paper_rc()):
            panel_width = max(5.8, min(7.4, 2.5 + 0.7 * len(n_values)))
            fig, axes = plt.subplots(
                nrows=1,
                ncols=n_panels,
                figsize=(panel_width * n_panels, 4.4 +
                         (0.4 if n_panels > 1 else 0.0)),
                squeeze=False,
                sharey=True,
            )
            axes = self._ensure_2d_axes(axes, 1, n_panels)

            for col_idx, (_, sub, title, _) in enumerate(groups):
                ax = axes[0, col_idx]
                prepared = sub.copy()
                prepared["_n_drones_numeric"] = pd.to_numeric(
                    prepared["n_drones"], errors="coerce"
                )
                prepared["cost_mean"] = pd.to_numeric(
                    prepared["cost_mean"], errors="coerce"
                )
                if "cost_ci95" in prepared.columns:
                    prepared["cost_ci95"] = pd.to_numeric(
                        prepared["cost_ci95"], errors="coerce"
                    )
                weights = (
                    pd.to_numeric(prepared["n_runs"],
                                  errors="coerce").fillna(1.0)
                    if "n_runs" in prepared.columns
                    else pd.Series(1.0, index=prepared.index, dtype=float)
                )
                rows: list[dict[str, object]] = []
                for (strategy, n_drones), strat_df in prepared.dropna(
                    subset=["strategy", "_n_drones_numeric", "cost_mean"]
                ).groupby(["strategy", "_n_drones_numeric"], dropna=False, sort=True):
                    group_weights = weights.loc[strat_df.index]
                    row: dict[str, object] = {
                        "strategy": str(strategy),
                        "n_drones": float(n_drones),
                        "cost_mean": self._weighted_average(
                            strat_df["cost_mean"],
                            group_weights,
                        ),
                    }
                    row["cost_ci95"] = (
                        self._weighted_average(
                            strat_df["cost_ci95"], group_weights)
                        if "cost_ci95" in strat_df.columns
                        else 0.0
                    )
                    rows.append(row)
                scaling_df = pd.DataFrame(rows)
                for n_drones in n_values:
                    n_df = (
                        scaling_df[scaling_df["n_drones"] == float(n_drones)]
                        .sort_values(["cost_mean", "strategy"], ascending=[True, True])
                        .reset_index(drop=True)
                    )
                    if n_df.empty:
                        continue
                    local_offsets = (
                        np.array([0.0], dtype=float)
                        if len(n_df) == 1
                        else np.linspace(
                            -cluster_width / 2.0,
                            cluster_width / 2.0,
                            len(n_df),
                        )
                    )
                    for row_idx, row in n_df.iterrows():
                        strategy = str(row["strategy"])
                        cost = float(row["cost_mean"])
                        cost_ci = float(row.get("cost_ci95", 0.0) or 0.0)
                        ax.bar(
                            n_lookup[n_drones] + local_offsets[row_idx],
                            cost,
                            width=bar_width,
                            yerr=cost_ci if cost_ci > 0.0 else None,
                            capsize=2.0 if cost_ci > 0.0 else 0.0,
                            color=color_map[strategy],
                            edgecolor="#222222",
                            linewidth=0.6,
                            alpha=0.92,
                            zorder=3,
                        )
                if len(groups) > 1 or title != "All runs":
                    ax.set_title(title, loc="left", fontweight="bold", pad=8)
                if col_idx == 0:
                    ax.set_ylabel("Cost (active agent-steps)")
                ax.set_xlabel(r"Population Size ($\bar{n}$)")
                ax.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
                ax.set_xticks(centers)
                ax.set_xticklabels([str(int(n)) for n in n_values])
                ax.yaxis.set_major_formatter(FuncFormatter(cost_tick))
                ax.grid(axis="y", visible=True)
                ax.grid(axis="x", visible=False)
                ax.tick_params(axis="both", labelsize=8.5)

            legend_handles = [
                Patch(
                    facecolor=color_map[strategy],
                    edgecolor="#222222",
                    linewidth=0.6,
                    label=self._display_strategy(
                        strategy, width=18).replace("\n", " "),
                )
                for strategy in strategies
            ]
            fig.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
            )
            fig.suptitle(
                "Cost by number of agents",
                y=0.995,
                fontsize=14,
                fontweight="bold",
            )
            fig.subplots_adjust(
                left=0.1,
                right=0.985,
                top=0.82,
                bottom=0.14,
                wspace=0.08,
            )
            output = self._save_variant_figure(fig, "population_cost", mode)
            plt.close(fig)
            return output

    def plot_run_distributions(self, run_df: pd.DataFrame, *, mode: str) -> str | None:
        metrics = [
            ("fuel_preserved_frac", self.FUEL_LABEL),
            ("active_exposure_frac", self.ACTIVE_LABEL),
        ]
        if "cost" in run_df.columns:
            metrics.append(("cost", self.COST_LABEL))
        metrics.append(("efficiency", "Efficiency"))
        required = {"strategy", *(metric for metric, _ in metrics)}
        if run_df.empty or not required.issubset(run_df.columns):
            self._warn(
                "skipping run distributions: missing run-level performance/cost/efficiency metrics")
            return None
        if mode == "scenario" and "scenario" not in run_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        groups = self._iter_mode_groups(run_df, mode)
        fixed_note = self._fixed_context_note_for_mode(run_df, mode)
        strategies = self._ordered_strategies(run_df)
        color_map = self._strategy_color_map(strategies)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(groups),
                ncols=len(metrics),
                figsize=(4.6 * len(metrics), 3.0 * len(groups) + 1.3),
                squeeze=False,
            )
            axes = self._ensure_2d_axes(axes, len(groups), len(metrics))
            row_titles: list[str] = []

            for row_idx, (_, sub, title, _) in enumerate(groups):
                row_titles.append(title)
                for col_idx, (metric, label) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    series_pairs: list[tuple[str, list[float]]] = []
                    for strategy in strategies:
                        values = pd.to_numeric(
                            sub.loc[sub["strategy"] == strategy, metric],
                            errors="coerce",
                        ).dropna()
                        if values.empty:
                            continue
                        series_pairs.append((strategy, values.tolist()))

                    if not series_pairs:
                        ax.text(
                            0.5,
                            0.5,
                            "no data",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="dimgray",
                        )
                    else:
                        labels = [name for name, _ in series_pairs]
                        values = [vals for _, vals in series_pairs]
                        order = np.argsort([float(np.mean(vals))
                                           for vals in values])
                        if not self._metric_ascending(metric):
                            order = order[::-1]
                        labels = [labels[idx] for idx in order]
                        values = [values[idx] for idx in order]
                        positions = np.arange(1, len(labels) + 1)
                        violins = ax.violinplot(
                            values,
                            positions=positions,
                            widths=0.8,
                            showmeans=False,
                            showmedians=False,
                            showextrema=False,
                        )
                        for body_idx, body in enumerate(violins["bodies"]):
                            strategy = labels[body_idx]
                            body.set_facecolor(color_map[strategy])
                            body.set_edgecolor("#2F2F2F")
                            body.set_alpha(0.50)
                            body.set_linewidth(0.7)

                        box = ax.boxplot(
                            values,
                            positions=positions,
                            widths=0.18,
                            patch_artist=True,
                            showfliers=False,
                            medianprops={"color": "#111111", "linewidth": 1.4},
                            whiskerprops={
                                "color": "#111111", "linewidth": 1.0},
                            capprops={"color": "#111111", "linewidth": 1.0},
                        )
                        for patch in box["boxes"]:
                            patch.set_facecolor("white")
                            patch.set_edgecolor("#111111")
                            patch.set_linewidth(1.0)
                            patch.set_alpha(0.95)

                        means = [float(np.mean(vals)) for vals in values]
                        ax.scatter(
                            positions,
                            means,
                            marker="D",
                            s=26,
                            color="#111111",
                            zorder=3,
                        )
                        ax.set_xticks(positions)
                        ax.set_xticklabels(
                            [self._display_strategy(name, width=14)
                             for name in labels],
                            rotation=0,
                        )

                    if row_idx == 0:
                        ax.set_title(label, loc="left",
                                     fontweight="bold", pad=20)
                    if metric in {"fuel_preserved_frac", "active_exposure_frac"}:
                        ax.yaxis.set_major_formatter(
                            PercentFormatter(xmax=1.0, decimals=0))
                    ax.grid(axis="x", visible=False)
                    ax.tick_params(axis="both", labelsize=8.5)

            fig.suptitle("Run-level distributions", y=0.995,
                         fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.DISTRIBUTION_NOTE,
                    self.FUEL_DEF,
                    self.ACTIVE_DEF,
                    self.COST_DEF if "cost" in run_df.columns else "",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96), h_pad=1.1)
            if len(groups) > 1 or (row_titles and row_titles[0] != "All runs"):
                for row_idx, title in enumerate(row_titles):
                    left_pos = axes[row_idx, 0].get_position()
                    fig.text(
                        left_pos.x0,
                        left_pos.y1 + 0.008,
                        title,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
            output = self._save_variant_figure(fig, "distributions", mode)
            plt.close(fig)
            return output

    def plot_attrition(self, step_df: pd.DataFrame, *, mode: str) -> str | None:
        metrics = [
            ("attrition_events_share", "Deaths this step (% initial swarm)"),
            ("attrition_total_share", "Cumulative attrition (% initial swarm)"),
        ]
        required = {"strategy", "step", "attrition_events_mean",
                    "attrition_total_cumulative_mean"}
        if step_df.empty or not required.issubset(step_df.columns):
            self._warn("skipping attrition: missing attrition step metrics")
            return None
        if mode == "scenario" and "scenario" not in step_df.columns:
            return None
        plt = self._load_pyplot()
        if plt is None:
            return None
        from matplotlib.ticker import PercentFormatter

        prepared = self._prepare_attrition_step_df(step_df)
        if prepared.empty:
            self._warn(
                "skipping attrition: unable to normalize attrition metrics")
            return None

        context_cols = self._group_context_columns(prepared)
        grouped = self._step_means(
            prepared, [metric for metric, _ in metrics], context_cols
        )
        if grouped.empty:
            self._warn(
                "skipping attrition: unable to aggregate attrition step metrics")
            return None

        groups = self._iter_mode_groups(grouped, mode)
        fixed_note = self._fixed_context_note_for_mode(grouped, mode)
        strategies = self._ordered_strategies(grouped)
        color_map = self._strategy_color_map(strategies)
        n_panels = len(groups)

        with plt.rc_context(self._paper_rc()):
            fig, axes = plt.subplots(
                nrows=len(metrics),
                ncols=n_panels,
                figsize=(4.2 * n_panels, 2.35 * len(metrics) + 1.4),
                squeeze=False,
                sharex="col",
                sharey="row",
            )
            axes = self._ensure_2d_axes(axes, len(metrics), n_panels)

            for col_idx, (_, sub, title, _) in enumerate(groups):
                sub = self._collapse_step_series(
                    sub, [metric for metric, _ in metrics]
                )
                last_step = int(pd.to_numeric(
                    sub["step"], errors="coerce").max())
                for row_idx, (metric, label) in enumerate(metrics):
                    ax = axes[row_idx, col_idx]
                    for strategy, strat_df in sub.groupby("strategy", sort=True):
                        strat_df = strat_df.sort_values("step")
                        x = pd.to_numeric(
                            strat_df["step"], errors="coerce").to_numpy()
                        y = pd.to_numeric(
                            strat_df[metric], errors="coerce").to_numpy()
                        ax.plot(x, y, linewidth=2.0,
                                color=color_map[str(strategy)])
                        ax.scatter(
                            [x[-1]],
                            [y[-1]],
                            s=18,
                            color=color_map[str(strategy)],
                            zorder=3,
                        )
                    ax.set_xlim(0.0, last_step)
                    ax.yaxis.set_major_formatter(
                        PercentFormatter(xmax=1.0, decimals=0))
                    if row_idx == 0 and (len(groups) > 1 or title != "All runs"):
                        ax.set_title(title, loc="left",
                                     fontweight="bold", pad=8)
                    if col_idx == 0:
                        ax.set_ylabel(label)
                    if row_idx == len(metrics) - 1:
                        ax.set_xlabel("Step")
                    ax.grid(axis="y", visible=True)
                    ax.tick_params(axis="both", labelsize=8.5)

            fig.legend(
                self._legend_handles(color_map, strategies),
                strategies,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=min(4, len(strategies)),
                fontsize=9,
            )
            fig.suptitle("Attrition", y=0.995, fontsize=14, fontweight="bold")
            self._add_footer(
                fig,
                self._compose_note(
                    self.ATTRITION_NOTE,
                    "Percentages are normalized by each run's initial alive swarm size before aggregation.",
                    fixed_note,
                ),
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
            output = self._save_variant_figure(fig, "attrition", mode)
            plt.close(fig)
            return output
