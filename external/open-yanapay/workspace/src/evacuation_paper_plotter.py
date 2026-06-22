"""Paper-facing plot suite for evacuation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill
from typing import Iterable, Optional

import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

PLOT_STYLE = "seaborn-v0_8-darkgrid"
METRIC_VARIANTS = ("mean", "median", "p90")
METRIC_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "p90": "P90",
}
STRATEGY_ORDER = {
    "always": 0,
    "frustration": 1,
    "labella": 2,
}
PALETTE = {
    "always": "#0072B2",
    "frustration": "#D55E00",
    "labella": "#009E73",
}
MARKERS = {
    "always": "o",
    "frustration": "s",
    "labella": "^",
}


def resolve_metric_variants(metric_variants: str | None) -> list[str]:
    if metric_variants is None or metric_variants == "all":
        return list(METRIC_VARIANTS)
    if metric_variants not in METRIC_VARIANTS:
        raise ValueError(
            f"Unsupported metric_variants value '{metric_variants}'. "
            f"Expected one of {list(METRIC_VARIANTS) + ['all']}"
        )
    return [metric_variants]


def _display_strategy(strategy: str) -> str:
    return {
        "always": "NAB",
        "frustration": "AFT",
        "labella": "AFH",
    }.get(str(strategy), str(strategy).replace("_", " ").title())


def _ordered_strategies(values: Iterable[str]) -> list[str]:
    unique = sorted({str(value) for value in values if pd.notna(value)})
    return sorted(unique, key=lambda strategy: (STRATEGY_ORDER.get(strategy, 999), strategy))


def _effective_ticks(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    prepared["effective_evacuation_ticks"] = prepared.get("evacuation_ticks")
    if "failure_reason" in prepared.columns and "max_netlogo_ticks" in prepared.columns:
        max_ticks_mask = (
            prepared["effective_evacuation_ticks"].isna()
            & prepared["failure_reason"].fillna("").eq("max_ticks")
        )
        prepared.loc[max_ticks_mask, "effective_evacuation_ticks"] = prepared.loc[
            max_ticks_mask, "max_netlogo_ticks"
        ]
    return prepared


def _quantile_90(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(0.9))


def _summarise_runs(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    prepared = _effective_ticks(data)
    prepared["success"] = prepared.get("success", prepared["effective_evacuation_ticks"].notna())
    aggregations: dict[str, tuple[str, str | callable]] = {
        "runs": ("simulation_id", "size") if "simulation_id" in prepared.columns else ("effective_evacuation_ticks", "size"),
        "success_rate": ("success", "mean"),
        "mean_evacuation_ticks": ("effective_evacuation_ticks", "mean"),
        "median_evacuation_ticks": ("effective_evacuation_ticks", "median"),
        "p90_evacuation_ticks": ("effective_evacuation_ticks", _quantile_90),
    }
    optional_metrics = [
        "active_robot_ticks",
        "reserve_robot_ticks",
        "task_committed_robot_ticks",
        "duplicate_robot_contacts",
        "interference_delay_total",
        "busy_staff_distractions",
        "mean_active_robots",
        "mean_reserve_robots",
        "mean_task_committed_robots",
    ]
    for metric in optional_metrics:
        if metric not in prepared.columns:
            continue
        aggregations[f"mean_{metric}"] = (metric, "mean")
        aggregations[f"median_{metric}"] = (metric, "median")
        aggregations[f"p90_{metric}"] = (metric, _quantile_90)

    return (
        prepared.groupby(group_columns, observed=True)
        .agg(**aggregations)
        .reset_index()
    )


def _metric_column(variant: str, base_name: str) -> str:
    return f"{variant}_{base_name}"


def _set_zoomed_limits(
    ax,
    values: pd.Series,
    *,
    axis: str,
    pad_ratio: float = 0.12,
    min_pad: float = 2.0,
) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return
    lower = float(numeric.min())
    upper = float(numeric.max())
    spread = max(upper - lower, 1e-6)
    pad = max(spread * pad_ratio, min_pad)
    if axis == "x":
        ax.set_xlim(lower - pad, upper + pad)
    else:
        ax.set_ylim(lower - pad, upper + pad)


def _is_dominated(
    row: pd.Series,
    candidates: pd.DataFrame,
    *,
    performance_column: str,
    cost_column: str,
) -> bool:
    better_or_equal = (
        (candidates[cost_column] <= row[cost_column])
        & (candidates[performance_column] <= row[performance_column])
        & (candidates["success_rate"] >= row["success_rate"])
    )
    strictly_better = (
        (candidates[cost_column] < row[cost_column])
        | (candidates[performance_column] < row[performance_column])
        | (candidates["success_rate"] > row["success_rate"])
    )
    return bool((better_or_equal & strictly_better).any())


def _is_performance_dominated(
    row: pd.Series,
    candidates: pd.DataFrame,
    *,
    performance_column: str,
    cost_column: str,
) -> bool:
    better_or_equal = (
        (candidates[cost_column] <= row[cost_column])
        & (candidates[performance_column] >= row[performance_column])
        & (candidates["success_rate"] >= row["success_rate"])
    )
    strictly_better = (
        (candidates[cost_column] < row[cost_column])
        | (candidates[performance_column] > row[performance_column])
        | (candidates["success_rate"] > row["success_rate"])
    )
    return bool((better_or_equal & strictly_better).any())


def _cost_tick(value: float, _: int) -> str:
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.1f}k".replace(".0k", "k")
    return f"{value:.0f}"


def _set_split_ylabel(
    ax,
    main: str,
    detail: str,
    *,
    main_x: float = -0.105,
    detail_x: float = -0.067,
    main_fontsize: float = 11,
    detail_fontsize: float = 9.5,
) -> None:
    ax.set_ylabel("")
    ax.text(
        main_x,
        0.5,
        main,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=main_fontsize,
        color="#222222",
    )
    ax.text(
        detail_x,
        0.5,
        detail,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=detail_fontsize,
        color="#777777",
    )


def _strategy_legend_handles(strategies: list[str]):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS.get(strategy, "o"),
            color=PALETTE.get(strategy, "#444444"),
            markerfacecolor=PALETTE.get(strategy, "#444444"),
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            markersize=7.0,
            linewidth=2.1,
            label=_display_strategy(strategy),
        )
        for strategy in strategies
    ]


class EvacuationPaperPlotter:
    def __init__(
        self,
        img_folder: str | Path,
        *,
        data_folder: str | Path,
        plot_mode: str = "standard",
    ) -> None:
        if plot_mode not in {"standard", "extended"}:
            raise ValueError("plot_mode must be 'standard' or 'extended'")
        self.img_folder = Path(img_folder)
        self.paper_root = self.img_folder / "paper"
        self.paper_root.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_folder)
        self.paper_data_root = self.data_root / "paper"
        self.paper_data_root.mkdir(parents=True, exist_ok=True)
        self.plot_mode = plot_mode
        self.performance_reference_ticks = self._load_performance_reference_ticks()

    def _load_performance_reference_ticks(self) -> float | None:
        config_path = self.data_root.parent / "config.json"
        if not config_path.exists():
            return None
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = config.get("scenarioParams", {}).get("maxNetlogoTicks")
        try:
            reference = float(value)
        except (TypeError, ValueError):
            return None
        return reference if reference > 0.0 else None

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

    def _performance_from_ticks(self, values: pd.Series) -> pd.Series:
        ticks = pd.to_numeric(values, errors="coerce")
        reference = self.performance_reference_ticks
        if reference is None:
            max_ticks = ticks.max(skipna=True)
            reference = float(max(max_ticks, 1.0)) if pd.notna(max_ticks) else 1.0
        return (1.0 - (ticks / float(reference))).clip(lower=0.0, upper=1.0)

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

    def _activate_style(self) -> None:
        plt.style.use(PLOT_STYLE)
        plt.rcParams.update(self._paper_rc())

    @staticmethod
    def _save_figure(fig, output: Path) -> str:
        fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0)
        return str(output)

    def _metric_dir(self, variant: str) -> Path:
        metric_dir = self.paper_root / variant
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "strategy_specific").mkdir(parents=True, exist_ok=True)
        return metric_dir

    def _strategy_summary(self, experiment_data: pd.DataFrame) -> pd.DataFrame:
        return _summarise_runs(experiment_data, ["robot_participation_strategy"])

    def _strategy_n_summary(self, experiment_data: pd.DataFrame) -> pd.DataFrame:
        return _summarise_runs(
            experiment_data,
            ["robot_participation_strategy", "num_of_robots"],
        )

    def _strategy_equal_weight_summary(self, strategy_n_summary: pd.DataFrame) -> pd.DataFrame:
        if strategy_n_summary.empty:
            return strategy_n_summary.copy()
        numeric_columns = [
            column
            for column in strategy_n_summary.columns
            if column not in {"robot_participation_strategy", "num_of_robots"}
            and pd.api.types.is_numeric_dtype(strategy_n_summary[column])
        ]
        aggregated = (
            strategy_n_summary.groupby("robot_participation_strategy", observed=True)[numeric_columns]
            .mean()
            .reset_index()
        )
        return aggregated

    def _build_metric_comparison(
        self,
        strategy_summary: pd.DataFrame,
        strategy_n_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def build_row(scope: str, sub: pd.DataFrame, num_of_robots: Optional[int] = None) -> dict[str, object]:
            row: dict[str, object] = {
                "scope": scope,
                "num_of_robots": num_of_robots,
            }
            for variant in METRIC_VARIANTS:
                performance_column = _metric_column(variant, "evacuation_ticks")
                ranked = sub.sort_values(
                    [
                        performance_column,
                        "success_rate",
                        "median_active_robot_ticks",
                        "robot_participation_strategy",
                    ],
                    ascending=[True, False, True, True],
                    na_position="last",
                ).reset_index(drop=True)
                row[f"{variant}_top_strategy"] = (
                    ranked.iloc[0]["robot_participation_strategy"] if not ranked.empty else None
                )
                row[f"{variant}_ranking"] = ",".join(
                    ranked["robot_participation_strategy"].astype(str).tolist()
                )
                pareto_frame = sub[
                    ["robot_participation_strategy", performance_column, "median_active_robot_ticks", "success_rate"]
                ].dropna(subset=[performance_column, "median_active_robot_ticks"])
                if pareto_frame.empty:
                    row[f"{variant}_pareto_set"] = ""
                else:
                    mask = [
                        not _is_dominated(
                            pareto_row,
                            pareto_frame.drop(index=index),
                            performance_column=performance_column,
                            cost_column="median_active_robot_ticks",
                        )
                        for index, pareto_row in pareto_frame.iterrows()
                    ]
                    row[f"{variant}_pareto_set"] = ",".join(
                        sorted(
                            pareto_frame.loc[mask, "robot_participation_strategy"]
                            .astype(str)
                            .tolist()
                        )
                    )
            row["top_strategy_changed_vs_median"] = (
                row["mean_top_strategy"] != row["median_top_strategy"]
                or row["p90_top_strategy"] != row["median_top_strategy"]
            )
            row["pareto_set_changed_vs_median"] = (
                row["mean_pareto_set"] != row["median_pareto_set"]
                or row["p90_pareto_set"] != row["median_pareto_set"]
            )
            return row

        overall_summary = self._strategy_equal_weight_summary(strategy_n_summary)
        rows.append(build_row("overall", overall_summary if not overall_summary.empty else strategy_summary))
        for _, sub in strategy_n_summary.groupby("num_of_robots", observed=True):
            num_of_robots = int(sub["num_of_robots"].iloc[0])
            rows.append(build_row("num_of_robots", sub, num_of_robots=num_of_robots))
        return pd.DataFrame(rows)

    def _write_metric_comparison_markdown(self, comparison: pd.DataFrame) -> None:
        lines = ["# Evacuation Metric Comparison", ""]
        overall = comparison[comparison["scope"] == "overall"]
        if not overall.empty:
            row = overall.iloc[0]
            lines.append(
                f"- Overall top strategies: mean=`{row['mean_top_strategy']}`, "
                f"median=`{row['median_top_strategy']}`, p90=`{row['p90_top_strategy']}`."
            )
            lines.append(
                f"- Overall Pareto sets: mean=`{row['mean_pareto_set']}`, "
                f"median=`{row['median_pareto_set']}`, p90=`{row['p90_pareto_set']}`."
            )
            lines.append("")

        changed = comparison[
            comparison["top_strategy_changed_vs_median"]
            | comparison["pareto_set_changed_vs_median"]
        ]
        if changed.empty:
            lines.append("- Mean and p90 do not change the recommended strategy or Pareto set versus median.")
        else:
            lines.append("## Cells with changes vs median")
            lines.append("")
            for row in changed.itertuples():
                if row.scope == "overall":
                    lines.append(
                        f"- overall: top mean/median/p90 = "
                        f"`{row.mean_top_strategy}` / `{row.median_top_strategy}` / `{row.p90_top_strategy}`; "
                        f"Pareto mean/median/p90 = `{row.mean_pareto_set}` / `{row.median_pareto_set}` / `{row.p90_pareto_set}`."
                    )
                else:
                    lines.append(
                        f"- N={int(row.num_of_robots)}: top mean/median/p90 = "
                        f"`{row.mean_top_strategy}` / `{row.median_top_strategy}` / `{row.p90_top_strategy}`; "
                        f"Pareto mean/median/p90 = `{row.mean_pareto_set}` / `{row.median_pareto_set}` / `{row.p90_pareto_set}`."
                    )
        (self.paper_data_root / "paper_metric_comparison.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def write_summary_tables(self, experiment_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        strategy_summary = self._strategy_summary(experiment_data)
        strategy_n_summary = self._strategy_n_summary(experiment_data)
        strategy_summary.to_csv(self.paper_data_root / "strategy_summary.csv", index=False)
        strategy_n_summary.to_csv(self.paper_data_root / "strategy_by_num_of_robots_summary.csv", index=False)
        comparison = self._build_metric_comparison(strategy_summary, strategy_n_summary)
        comparison.to_csv(self.paper_data_root / "paper_metric_comparison.csv", index=False)
        self._write_metric_comparison_markdown(comparison)
        return strategy_summary, strategy_n_summary

    def _plot_headline_ranking(
        self,
        strategy_summary: pd.DataFrame,
        variant: str,
        metric_dir: Path,
    ) -> str | None:
        performance_column = _metric_column(variant, "evacuation_ticks")
        if performance_column not in strategy_summary.columns:
            return None
        ranked = strategy_summary.sort_values(
            [performance_column, "success_rate", "median_active_robot_ticks", "robot_participation_strategy"],
            ascending=[True, False, True, True],
            na_position="last",
        ).reset_index(drop=True)

        self._activate_style()
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        colors = [PALETTE.get(strategy, "#444444") for strategy in ranked["robot_participation_strategy"]]
        ax.barh(
            np.arange(len(ranked)),
            ranked[performance_column].astype(float).to_numpy(),
            color=colors,
            edgecolor="#2F2F2F",
        )
        ax.set_yticks(np.arange(len(ranked)))
        ax.set_yticklabels([_display_strategy(strategy) for strategy in ranked["robot_participation_strategy"]])
        ax.invert_yaxis()
        _set_zoomed_limits(ax, ranked[performance_column], axis="x")
        ax.set_xlabel(f"{METRIC_LABELS[variant]} Evacuation Ticks")
        ax.set_title(f"Evacuation Ranking ({METRIC_LABELS[variant]})", fontweight="bold")
        for idx, row in ranked.iterrows():
            cost = row.get("median_active_robot_ticks", float("nan"))
            value = float(row[performance_column])
            ax.text(
                value,
                idx - 0.26,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
            )
            label = (
                f"success {100 * float(row['success_rate']):.0f}% | "
                f"cost {int(cost) if pd.notna(cost) else 'n/a'}"
            )
            ax.text(
                value,
                idx,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
            )
        self._add_footer(
            fig,
            "Ranking uses evacuation performance as the primary axis; annotations show success rate and median active robot cost.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "01_headline_ranking_overall.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def _plot_tradeoff_pareto(
        self,
        strategy_n_summary: pd.DataFrame,
        variant: str,
        metric_dir: Path,
    ) -> str | None:
        performance_column = _metric_column(variant, "evacuation_ticks")
        active_column = _metric_column(variant, "mean_active_robots")
        required = {
            performance_column,
            active_column,
            "success_rate",
            "robot_participation_strategy",
            "num_of_robots",
        }
        if not required.issubset(strategy_n_summary.columns):
            return None

        prepared = strategy_n_summary.copy()
        prepared["evacuation_ticks"] = pd.to_numeric(prepared[performance_column], errors="coerce")
        prepared["active_time_ratio"] = (
            pd.to_numeric(prepared[active_column], errors="coerce")
            / pd.to_numeric(prepared["num_of_robots"], errors="coerce")
        )
        summary = (
            prepared.groupby("robot_participation_strategy", observed=True)
            .agg(
                evacuation_ticks=("evacuation_ticks", "median"),
                active_time_ratio=("active_time_ratio", "median"),
                success_rate=("success_rate", "mean"),
            )
            .reset_index()
            .dropna(subset=["evacuation_ticks", "active_time_ratio"])
        )
        summary["is_pareto_efficient"] = [
            not _is_dominated(
                row,
                summary.drop(index=index),
                performance_column="evacuation_ticks",
                cost_column="active_time_ratio",
            )
            for index, row in summary.iterrows()
        ]
        if summary.empty:
            return None

        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        ordered_strategies = _ordered_strategies(summary["robot_participation_strategy"])
        x_max = float(summary["active_time_ratio"].max(skipna=True))
        with plt.rc_context(self._paper_rc()):
            fig, ax = plt.subplots(figsize=(6.4, 5.6))
            text_scale = 0.7
            legend_text_scale = text_scale * 0.9
            for _, row in summary.iterrows():
                strategy = str(row["robot_participation_strategy"])
                edge = "#111111" if bool(row["is_pareto_efficient"]) else "white"
                ax.scatter(
                    [float(row["active_time_ratio"])],
                    [float(row["evacuation_ticks"])],
                    s=150,
                    marker=MARKERS.get(strategy, "o"),
                    color=PALETTE.get(strategy, "#444444"),
                    edgecolors=edge,
                    linewidths=1.2,
                    zorder=3,
                )

            ax.set_xlabel(r"Active ratio ($R$)", fontsize=13 * text_scale)
            _set_split_ylabel(
                ax,
                r"Evacuation time ($\hat{P}$)",
                "ticks",
                main_x=-0.14,
                detail_x=-0.095,
                main_fontsize=13 * text_scale,
                detail_fontsize=11 * text_scale,
            )
            ax.set_xlim(0.0, x_max + max(0.03, 0.08 * x_max))
            _set_zoomed_limits(ax, summary["evacuation_ticks"], axis="y", pad_ratio=0.2, min_pad=5.0)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(axis="y", visible=True)
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="both", labelsize=11 * text_scale)

            handles = _strategy_legend_handles(ordered_strategies)
            fig.legend(
                handles,
                [handle.get_label() for handle in handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, len(handles)),
                fontsize=12 * legend_text_scale,
                frameon=False,
            )
            fig.subplots_adjust(left=0.18, right=0.985, top=0.86, bottom=0.16)
            output = metric_dir / "02_tradeoff_pareto_overall.png"
            saved = self._save_figure(fig, output)
            plt.close(fig)
            return saved

    def _plot_population_scaling(
        self,
        strategy_n_summary: pd.DataFrame,
        variant: str,
        metric_dir: Path,
    ) -> str | None:
        performance_column = _metric_column(variant, "evacuation_ticks")
        if performance_column not in strategy_n_summary.columns:
            return None
        self._activate_style()
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
        ordered_strategies = _ordered_strategies(strategy_n_summary["robot_participation_strategy"])
        for strategy in ordered_strategies:
            sub = strategy_n_summary[
                strategy_n_summary["robot_participation_strategy"] == strategy
            ].sort_values("num_of_robots")
            color = PALETTE.get(strategy, "#444444")
            axes[0].plot(sub["num_of_robots"], sub[performance_column], marker="o", color=color, label=_display_strategy(strategy))
            if "median_active_robot_ticks" in sub.columns:
                axes[1].plot(sub["num_of_robots"], sub["median_active_robot_ticks"], marker="o", color=color, label=_display_strategy(strategy))
            axes[2].plot(sub["num_of_robots"], sub["success_rate"], marker="o", color=color, label=_display_strategy(strategy))
        axes[0].set_ylabel(f"{METRIC_LABELS[variant]} Evac. Ticks")
        axes[1].set_ylabel("Median Active\nRobot-Ticks")
        axes[2].set_ylabel("Success Rate")
        axes[2].set_xlabel("Number of Robots")
        axes[2].set_ylim(0, 1.05)
        axes[0].set_title(f"Population Scaling ({METRIC_LABELS[variant]})", fontweight="bold")
        axes[0].legend(title="Strategy")
        self._add_footer(
            fig,
            "Top: evacuation performance. Middle: active robot effort. Bottom: completion reliability across robot counts.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "03_population_scaling_overall.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def _plot_population_performance_cost(
        self,
        strategy_n_summary: pd.DataFrame,
        variant: str,
        metric_dir: Path,
    ) -> str | None:
        performance_column = _metric_column(variant, "evacuation_ticks")
        cost_column = _metric_column(variant, "active_robot_ticks")
        if not {performance_column, cost_column}.issubset(strategy_n_summary.columns):
            return None
        ordered_strategies = _ordered_strategies(strategy_n_summary["robot_participation_strategy"])
        n_values = sorted(strategy_n_summary["num_of_robots"].dropna().astype(int).unique())
        if not n_values:
            return None

        prepared = strategy_n_summary.copy()
        prepared["evacuation_ticks"] = pd.to_numeric(prepared[performance_column], errors="coerce")
        prepared[cost_column] = pd.to_numeric(prepared[cost_column], errors="coerce")
        group_spacing = 1.38
        centers = np.arange(len(n_values), dtype=float) * group_spacing
        n_lookup = {value: center for value, center in zip(n_values, centers)}
        cluster_width = 0.78
        bar_width = min(0.18, cluster_width / max(1, len(ordered_strategies)))

        from matplotlib.ticker import FuncFormatter, MaxNLocator

        with plt.rc_context(self._paper_rc()):
            text_scale = 1.2
            fig, (ax_perf, ax_cost) = plt.subplots(
                2,
                1,
                figsize=(9.6, 7.2),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 0.12},
            )

            for strategy in ordered_strategies:
                sub = prepared[
                    prepared["robot_participation_strategy"] == strategy
                ].sort_values("num_of_robots")
                if sub.empty:
                    continue
                x = np.array([n_lookup[int(n)] for n in sub["num_of_robots"]], dtype=float)
                y = pd.to_numeric(sub["evacuation_ticks"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    continue
                color = PALETTE.get(strategy, "#444444")
                ax_perf.plot(
                    x[valid],
                    y[valid],
                    marker=MARKERS.get(strategy, "o"),
                    markersize=6.5,
                    linewidth=2.1,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="#111111",
                    markeredgewidth=0.8,
                    zorder=3,
                )

            for n_value in n_values:
                n_df = (
                    prepared[prepared["num_of_robots"].astype(int) == n_value]
                    .sort_values([cost_column, "robot_participation_strategy"], ascending=[True, True])
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
                    strategy = str(row["robot_participation_strategy"])
                    ax_cost.bar(
                        n_lookup[n_value] + local_offsets[row_idx],
                        float(row[cost_column]),
                        width=bar_width,
                        color=PALETTE.get(strategy, "#444444"),
                        edgecolor="#222222",
                        linewidth=0.6,
                        alpha=0.92,
                        zorder=3,
                    )

            _set_split_ylabel(
                ax_perf,
                r"Evacuation time ($\hat{P}$)",
                "ticks",
                main_fontsize=11 * text_scale,
                detail_fontsize=9.5 * text_scale,
            )
            _set_zoomed_limits(ax_perf, prepared["evacuation_ticks"], axis="y", pad_ratio=0.18, min_pad=6.0)
            ax_perf.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax_perf.grid(axis="y", visible=True)
            ax_perf.grid(axis="x", visible=False)
            ax_perf.tick_params(axis="both", labelsize=9 * text_scale, labelbottom=False)

            _set_split_ylabel(
                ax_cost,
                r"Cost ($J$)",
                "active-robot ticks",
                main_fontsize=11 * text_scale,
                detail_fontsize=9.5 * text_scale,
            )
            ax_cost.set_xlabel(r"Population Size ($\bar{n}$)", fontsize=11 * text_scale)
            ax_cost.set_xlim(centers[0] - 0.65, centers[-1] + 0.65)
            ax_cost.set_xticks(centers)
            ax_cost.set_xticklabels([str(int(n)) for n in n_values])
            ax_cost.yaxis.set_major_formatter(FuncFormatter(_cost_tick))
            ax_cost.grid(axis="y", visible=True)
            ax_cost.grid(axis="x", visible=False)
            ax_cost.tick_params(axis="both", labelsize=9 * text_scale)

            handles = _strategy_legend_handles(ordered_strategies)
            fig.legend(
                handles,
                [handle.get_label() for handle in handles],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=min(5, max(1, len(handles))),
                fontsize=9 * text_scale,
                frameon=False,
            )
            fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.1)
            output = metric_dir / "13_population_performance_cost_combined_overall.png"
            saved = self._save_figure(fig, output)
            plt.close(fig)
            return saved

    def _plot_run_distributions(
        self,
        experiment_data: pd.DataFrame,
        metric_dir: Path,
    ) -> str | None:
        prepared = _effective_ticks(experiment_data)
        if "robot_participation_strategy" not in prepared.columns:
            return None
        metric_specs = [
            ("effective_evacuation_ticks", "Evacuation Ticks"),
            ("active_robot_ticks", "Active Robot-Ticks"),
            ("interference_delay_total", "Interference Delay"),
        ]
        available = [(column, label) for column, label in metric_specs if column in prepared.columns]
        if not available:
            return None
        ordered_strategies = _ordered_strategies(prepared["robot_participation_strategy"])
        self._activate_style()
        fig, axes = plt.subplots(1, len(available), figsize=(4.8 * len(available), 4.8), squeeze=False)
        for axis, (column, label) in zip(axes[0], available):
            sns.violinplot(
                data=prepared,
                x="robot_participation_strategy",
                y=column,
                order=ordered_strategies,
                palette=PALETTE,
                inner=None,
                ax=axis,
            )
            sns.boxplot(
                data=prepared,
                x="robot_participation_strategy",
                y=column,
                order=ordered_strategies,
                width=0.18,
                showcaps=True,
                showfliers=False,
                boxprops={"facecolor": "white", "edgecolor": "#111111"},
                medianprops={"color": "#111111"},
                whiskerprops={"color": "#111111"},
                ax=axis,
            )
            axis.set_title(label, fontweight="bold")
            axis.set_xlabel("")
            axis.set_xticklabels([fill(_display_strategy(strategy), 12) for strategy in ordered_strategies])
        fig.suptitle("Run Distributions", y=1.02, fontsize=14, fontweight="bold")
        self._add_footer(
            fig,
            "Violin bodies show run distributions; box overlays show median and spread for performance, effort, and interference.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "05_run_distributions_overall.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def _plot_behavior_mechanism(
        self,
        strategy_n_summary: pd.DataFrame,
        metric_dir: Path,
    ) -> str | None:
        required = {
            "num_of_robots",
            "robot_participation_strategy",
            "median_duplicate_robot_contacts",
            "median_interference_delay_total",
            "median_active_robot_ticks",
            "median_reserve_robot_ticks",
        }
        if not required.issubset(strategy_n_summary.columns):
            return None
        self._activate_style()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        plot_specs = [
            ("median_duplicate_robot_contacts", "Duplicate Contacts"),
            ("median_interference_delay_total", "Interference Delay"),
            ("median_active_robot_ticks", "Active Robot-Ticks"),
            ("median_reserve_robot_ticks", "Reserve Robot-Ticks"),
        ]
        ordered_strategies = _ordered_strategies(strategy_n_summary["robot_participation_strategy"])
        for axis, (column, title) in zip(axes.flat, plot_specs):
            for strategy in ordered_strategies:
                sub = strategy_n_summary[
                    strategy_n_summary["robot_participation_strategy"] == strategy
                ].sort_values("num_of_robots")
                axis.plot(
                    sub["num_of_robots"],
                    sub[column],
                    marker="o",
                    color=PALETTE.get(strategy, "#444444"),
                    label=_display_strategy(strategy),
                )
            axis.set_title(title, fontweight="bold")
        axes[0, 0].legend(title="Strategy")
        axes[1, 0].set_xlabel("Number of Robots")
        axes[1, 1].set_xlabel("Number of Robots")
        fig.suptitle("Behavior Mechanism", y=1.02, fontsize=14, fontweight="bold")
        self._add_footer(
            fig,
            "These panels separate coordination cost from work allocation so changes in evacuation time can be traced back to mechanism, not just outcome.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "06_behavior_mechanism_overall.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def _plot_headline_by_num_of_robots(
        self,
        strategy_n_summary: pd.DataFrame,
        variant: str,
        metric_dir: Path,
    ) -> str | None:
        performance_column = _metric_column(variant, "evacuation_ticks")
        if performance_column not in strategy_n_summary.columns:
            return None
        n_values = sorted(strategy_n_summary["num_of_robots"].dropna().astype(int).unique())
        if len(n_values) <= 1:
            return None
        self._activate_style()
        fig, axes = plt.subplots(len(n_values), 1, figsize=(8, 3.2 * len(n_values)), squeeze=False)
        for axis, n_value in zip(axes.flat, n_values):
            sub = strategy_n_summary[
                strategy_n_summary["num_of_robots"] == n_value
            ].sort_values(
                [performance_column, "success_rate", "median_active_robot_ticks", "robot_participation_strategy"],
                ascending=[True, False, True, True],
                na_position="last",
            )
            axis.barh(
                np.arange(len(sub)),
                sub[performance_column],
                color=[PALETTE.get(strategy, "#444444") for strategy in sub["robot_participation_strategy"]],
                edgecolor="#2F2F2F",
            )
            axis.set_yticks(np.arange(len(sub)))
            axis.set_yticklabels([_display_strategy(strategy) for strategy in sub["robot_participation_strategy"]])
            axis.invert_yaxis()
            _set_zoomed_limits(axis, sub[performance_column], axis="x")
            for idx, row in enumerate(sub.itertuples()):
                axis.text(
                    float(getattr(row, performance_column)),
                    idx - 0.24,
                    f"{float(getattr(row, performance_column)):.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#111111",
                )
            axis.set_title(f"N={n_value}", loc="left", fontweight="bold")
        axes[-1, 0].set_xlabel(f"{METRIC_LABELS[variant]} Evacuation Ticks")
        fig.suptitle(f"Ranking by Robot Count ({METRIC_LABELS[variant]})", y=1.01, fontsize=14, fontweight="bold")
        self._add_footer(
            fig,
            "Within each robot-count panel, strategies are ranked by evacuation performance with cost used as a secondary tie-breaker.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "strategy_specific" / "01_headline_ranking_by_num_of_robots.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def _plot_behavior_mechanism_by_strategy(
        self,
        tick_trace_data: Optional[pd.DataFrame],
        metric_dir: Path,
        *,
        expected_strategies: list[str],
    ) -> str | None:
        if tick_trace_data is None or tick_trace_data.empty:
            return None
        required = {
            "tick",
            "robot_participation_strategy",
            "active_robots",
            "reserve_robots",
            "unresolved_fallen",
            "attended_fallen",
        }
        if not required.issubset(tick_trace_data.columns):
            return None
        strategies = _ordered_strategies(tick_trace_data["robot_participation_strategy"])
        if len(strategies) <= 1 or sorted(strategies) != sorted(expected_strategies):
            return None
        grouped = (
            tick_trace_data.groupby(["robot_participation_strategy", "tick"], observed=True)[
                ["active_robots", "reserve_robots", "unresolved_fallen", "attended_fallen"]
            ]
            .median()
            .reset_index()
        )
        self._activate_style()
        fig, axes = plt.subplots(len(strategies), 4, figsize=(16, 3.2 * len(strategies)), squeeze=False, sharex=False)
        metrics = [
            ("active_robots", "Active"),
            ("reserve_robots", "Reserve"),
            ("unresolved_fallen", "Unresolved"),
            ("attended_fallen", "Attended"),
        ]
        for row_idx, strategy in enumerate(strategies):
            sub = grouped[grouped["robot_participation_strategy"] == strategy].sort_values("tick")
            for col_idx, (column, title) in enumerate(metrics):
                axis = axes[row_idx, col_idx]
                axis.plot(sub["tick"], sub[column], color=PALETTE.get(strategy, "#444444"))
                if row_idx == 0:
                    axis.set_title(title, fontweight="bold")
                if col_idx == 0:
                    axis.set_ylabel(_display_strategy(strategy))
                if row_idx == len(strategies) - 1:
                    axis.set_xlabel("Tick")
        fig.suptitle("Behavior Over Time", y=1.01, fontsize=14, fontweight="bold")
        self._add_footer(
            fig,
            "Temporal panels expose when strategies commit robots, hold them in reserve, and relieve the fallen-passenger backlog.",
        )
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        output = metric_dir / "strategy_specific" / "06_behavior_mechanism_by_strategy.png"
        saved = self._save_figure(fig, output)
        plt.close(fig)
        return saved

    def plot_suite(
        self,
        experiment_data: pd.DataFrame,
        tick_trace_data: Optional[pd.DataFrame],
        *,
        metric_variants: Iterable[str],
    ) -> list[str]:
        # The release reproduces only the two evacuation figures used in the paper:
        # 02_tradeoff_pareto_overall and 13_population_performance_cost_combined_overall.
        generated: list[str] = []
        _, strategy_n_summary = self.write_summary_tables(experiment_data)
        for variant in metric_variants:
            metric_dir = self._metric_dir(variant)
            for plot_fn in (
                lambda: self._plot_tradeoff_pareto(strategy_n_summary, variant, metric_dir),
                lambda: self._plot_population_performance_cost(strategy_n_summary, variant, metric_dir),
            ):
                result = plot_fn()
                if result:
                    generated.append(result)
        return generated
