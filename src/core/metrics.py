"""Core: research metrics for evaluation runs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
import torch

from src.swarm.constants import DroneState

if TYPE_CHECKING:
    from src.physics.manager import PhysicsManager
    from src.swarm.manager import SwarmManager


SUMMARY_EPS = 1e-6
STAND_DOWN_THRESHOLD_FRAC = 0.10


def _safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return float(numer / denom)


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _resolve_agent_capacity(
    df: pd.DataFrame,
    default_n_drones: int | float | None = None,
) -> float:
    for col in ("n_drones", "n_active_agents"):
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            return float(series.max())
    if default_n_drones is not None and float(default_n_drones) > 0.0:
        return float(default_n_drones)
    if "alive_agents_mean" in df.columns:
        series = pd.to_numeric(df["alive_agents_mean"], errors="coerce").dropna()
        if not series.empty:
            return float(series.max())
    return 0.0


def _resolve_run_max_steps(
    df: pd.DataFrame,
    default_run_max_steps: int | float | None = None,
) -> float:
    for col in ("run_max_steps", "max_steps"):
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    if default_run_max_steps is not None and float(default_run_max_steps) > 0.0:
        return float(default_run_max_steps)
    if "step" in df.columns:
        series = pd.to_numeric(df["step"], errors="coerce").dropna()
        if not series.empty:
            return float(series.max() + 1.0)
    return float(len(df))


def _metric_count_series(
    df: pd.DataFrame,
    *,
    count_col: str,
    ratio_col: str | None = None,
    alive_col: str = "alive_agents_mean",
) -> pd.Series:
    if count_col in df.columns:
        return _numeric_series(df, count_col)
    if ratio_col is not None and ratio_col in df.columns:
        alive = _numeric_series(df, alive_col)
        ratio = _numeric_series(df, ratio_col)
        return alive * ratio
    return pd.Series(0.0, index=df.index, dtype=float)


def _fuel_total_series(df: pd.DataFrame) -> pd.Series:
    if "fuel_total_mean" in df.columns:
        return _numeric_series(df, "fuel_total_mean")
    fuel_frac = _numeric_series(df, "fuel_pct_mean")
    if "grid_size" in df.columns:
        grid = _numeric_series(df, "grid_size", default=1.0)
        return fuel_frac * grid.pow(2)
    return fuel_frac


def build_run_summary_from_steps(
    df: pd.DataFrame,
    *,
    sps: float,
    strategy: str,
    scenario: str,
    seed: int,
    default_n_drones: int | float | None = None,
    default_run_max_steps: int | float | None = None,
    fire_out_threshold: float = 1e-3,
    stand_down_threshold_frac: float = STAND_DOWN_THRESHOLD_FRAC,
) -> dict:
    """Build the canonical run-summary schema from step-level metrics."""
    if df.empty:
        return {
            "strategy": strategy,
            "scenario": scenario,
            "seed": int(seed),
            "steps": 0,
            "steps_per_second": float(sps),
            "fuel_start_total": 0.0,
            "fuel_end_total": 0.0,
            "fuel_saved_total": 0.0,
            "fuel_loss_total": 0.0,
            "fuel_start_pct": 0.0,
            "fuel_end_pct": 0.0,
            "fuel_saved_pct": 0.0,
            "fuel_saved_frac": 0.0,
            "fuel_preserved_frac": 0.0,
            "fuel_loss_frac": 0.0,
            "fire_coverage_mean": 0.0,
            "fire_coverage_peak": 0.0,
            "active_drone_steps_raw": 0.0,
            "active_exposure_frac": 0.0,
            "waiting_drone_steps_raw": 0.0,
            "exploring_drone_steps_raw": 0.0,
            "firefighting_drone_steps_raw": 0.0,
            "returning_drone_steps_raw": 0.0,
            "post_fire_active_steps_raw": 0.0,
            "post_fire_active_exposure_frac": 0.0,
            "fire_out_reached": 0.0,
            "post_fire_window_steps": 0,
            "active_ratio_mean": 0.0,
            "active_ratio_post_fire": 0.0,
            "active_headcount_run_mean": 0.0,
            "post_transient_active_headcount_mean": 0.0,
            "fire_out_step": -1,
            "fire_out_fraction": 0.0,
            "stand_down_step": -1,
            "stand_down_latency_steps": float("nan"),
            "stand_down_reached": float("nan"),
            "attrition_total_mean": 0.0,
            "attrition_fraction": 0.0,
            "cost": 0.0,
            "efficiency": 0.0,
            "utility": 0.0,
            "utility_internal": 0.0,
        }

    run_df = df.sort_values("step").reset_index(drop=True)
    fuel_total = _fuel_total_series(run_df)
    fuel_frac = _numeric_series(run_df, "fuel_pct_mean")
    fire_cov = _numeric_series(run_df, "fire_coverage_mean")
    alive_agents = _metric_count_series(
        run_df,
        count_col="alive_agents_mean",
    )
    active_agents = _metric_count_series(
        run_df,
        count_col="active_agents_mean",
        ratio_col="active_ratio_mean",
    )
    waiting_agents = _metric_count_series(
        run_df,
        count_col="waiting_agents_mean",
        ratio_col="waiting_ratio_mean",
    )
    exploring_agents = _metric_count_series(
        run_df,
        count_col="exploring_agents_mean",
        ratio_col="exploring_ratio_mean",
    )
    firefighting_agents = _metric_count_series(
        run_df,
        count_col="firefighting_agents_mean",
        ratio_col="firefighting_ratio_mean",
    )
    returning_agents = _metric_count_series(
        run_df,
        count_col="returning_agents_mean",
        ratio_col="returning_ratio_mean",
    )
    attrition_events = _numeric_series(run_df, "attrition_events_mean")
    attrition_total_cumulative = _numeric_series(
        run_df, "attrition_total_cumulative_mean"
    )
    initial_alive_series = _numeric_series(run_df, "initial_alive_mean")

    fuel_start_total = float(fuel_total.iloc[0])
    fuel_end_total = float(fuel_total.iloc[-1])
    fuel_loss_total = max(0.0, fuel_start_total - fuel_end_total)
    fuel_start_frac = float(fuel_frac.iloc[0]) if not fuel_frac.empty else 0.0
    fuel_end_frac = float(fuel_frac.iloc[-1]) if not fuel_frac.empty else 0.0
    fuel_preserved_frac = _safe_ratio(fuel_end_total, fuel_start_total)
    fuel_loss_frac = _safe_ratio(fuel_loss_total, fuel_start_total)

    steps = int(len(run_df))
    run_max_steps = _resolve_run_max_steps(
        run_df, default_run_max_steps=default_run_max_steps
    )
    agent_capacity = _resolve_agent_capacity(
        run_df, default_n_drones=default_n_drones
    )
    exposure_denom = float(agent_capacity * run_max_steps)

    active_drone_steps_raw = float(active_agents.sum())
    waiting_drone_steps_raw = float(waiting_agents.sum())
    exploring_drone_steps_raw = float(exploring_agents.sum())
    firefighting_drone_steps_raw = float(firefighting_agents.sum())
    returning_drone_steps_raw = float(returning_agents.sum())
    active_exposure_frac = _safe_ratio(active_drone_steps_raw, exposure_denom)
    attrition_total_mean = float(
        attrition_total_cumulative.iloc[-1]
        if "attrition_total_cumulative_mean" in run_df.columns
        else attrition_events.sum()
    )
    initial_alive_agents = float(initial_alive_series.iloc[0])
    if initial_alive_agents <= 0.0:
        initial_alive_agents = float(alive_agents.iloc[0] + attrition_events.iloc[0])
    if initial_alive_agents <= 0.0:
        initial_alive_agents = max(
            agent_capacity,
            float(alive_agents.max()) if not alive_agents.empty else 0.0,
        )
    attrition_fraction = _safe_ratio(attrition_total_mean, initial_alive_agents)

    is_out = fire_cov <= float(fire_out_threshold)
    sustained_out = is_out.iloc[::-1].cummin().iloc[::-1]
    if sustained_out.any():
        fire_out_reached = 1.0
        fire_out_idx = sustained_out.idxmax()
        if "step" in run_df.columns:
            fire_out_step = int(pd.to_numeric(
                run_df.loc[fire_out_idx, "step"], errors="coerce"
            ))
        else:
            fire_out_step = int(fire_out_idx)
        post_fire_window_steps = int(sustained_out.sum())
        post_fire_active_steps_raw = float(active_agents[sustained_out].sum())
        alive_post_fire = float(alive_agents[sustained_out].sum())
        active_ratio_post_fire = _safe_ratio(post_fire_active_steps_raw, alive_post_fire)
        stand_down_threshold_agents = max(
            1.0,
            float(stand_down_threshold_frac) * max(agent_capacity, 0.0),
        ) if agent_capacity > 0.0 else 0.0
        stand_down_mask = sustained_out & (
            active_agents <= (stand_down_threshold_agents + SUMMARY_EPS)
        )
        if stand_down_mask.any():
            stand_down_idx = stand_down_mask.idxmax()
            if "step" in run_df.columns:
                stand_down_step = int(pd.to_numeric(
                    run_df.loc[stand_down_idx, "step"], errors="coerce"
                ))
            else:
                stand_down_step = int(stand_down_idx)
            stand_down_latency_steps = float(
                max(0, stand_down_step - fire_out_step)
            )
            stand_down_reached = 1.0
        else:
            stand_down_step = -1
            stand_down_latency_steps = float(max(0, post_fire_window_steps))
            stand_down_reached = 0.0
    else:
        fire_out_reached = 0.0
        fire_out_step = -1
        post_fire_window_steps = 0
        post_fire_active_steps_raw = 0.0
        active_ratio_post_fire = 0.0
        stand_down_step = -1
        stand_down_latency_steps = float("nan")
        stand_down_reached = float("nan")
    post_fire_active_exposure_frac = _safe_ratio(
        post_fire_active_steps_raw, exposure_denom
    )

    alive_agent_steps_raw = float(alive_agents.sum())
    active_ratio_mean = _safe_ratio(active_drone_steps_raw, alive_agent_steps_raw)
    active_headcount_run_mean = float(active_agents.mean()) if not active_agents.empty else 0.0
    post_transient_start_step = int(0.20 * max(run_max_steps, 1.0))
    if "step" in run_df.columns:
        step_index = pd.to_numeric(run_df["step"], errors="coerce").fillna(0.0)
    else:
        step_index = pd.Series(range(len(run_df)), index=run_df.index, dtype=float)
    post_transient_mask = step_index >= float(post_transient_start_step)
    post_transient_active_headcount_mean = (
        float(active_agents[post_transient_mask].mean())
        if post_transient_mask.any()
        else active_headcount_run_mean
    )
    efficiency = _safe_ratio(fuel_preserved_frac, active_exposure_frac + SUMMARY_EPS)
    utility_internal = fuel_preserved_frac - active_exposure_frac

    return {
        "strategy": strategy,
        "scenario": scenario,
        "seed": int(seed),
        "steps": steps,
        "steps_per_second": float(sps),
        "fuel_start_total": fuel_start_total,
        "fuel_end_total": fuel_end_total,
        "fuel_saved_total": fuel_end_total,
        "fuel_loss_total": fuel_loss_total,
        "fuel_start_pct": fuel_start_frac,
        "fuel_end_pct": fuel_end_frac,
        "fuel_saved_pct": fuel_end_frac,
        "fuel_saved_frac": fuel_preserved_frac,
        "fuel_preserved_frac": fuel_preserved_frac,
        "fuel_loss_frac": fuel_loss_frac,
        "fire_coverage_mean": float(fire_cov.mean()),
        "fire_coverage_peak": float(fire_cov.max()),
        "active_drone_steps_raw": active_drone_steps_raw,
        "active_exposure_frac": active_exposure_frac,
        "waiting_drone_steps_raw": waiting_drone_steps_raw,
        "exploring_drone_steps_raw": exploring_drone_steps_raw,
        "firefighting_drone_steps_raw": firefighting_drone_steps_raw,
        "returning_drone_steps_raw": returning_drone_steps_raw,
        "post_fire_active_steps_raw": post_fire_active_steps_raw,
        "post_fire_active_exposure_frac": post_fire_active_exposure_frac,
        "fire_out_reached": fire_out_reached,
        "post_fire_window_steps": post_fire_window_steps,
        "active_ratio_mean": active_ratio_mean,
        "active_ratio_post_fire": active_ratio_post_fire,
        "active_headcount_run_mean": active_headcount_run_mean,
        "post_transient_active_headcount_mean": post_transient_active_headcount_mean,
        "fire_out_step": fire_out_step,
        "fire_out_fraction": float(is_out.mean()),
        "stand_down_step": stand_down_step,
        "stand_down_latency_steps": stand_down_latency_steps,
        "stand_down_reached": stand_down_reached,
        "attrition_total_mean": attrition_total_mean,
        "attrition_fraction": attrition_fraction,
        # Compatibility / internal-only aliases.
        "cost": active_drone_steps_raw,
        "efficiency": efficiency,
        "utility": utility_internal,
        "utility_internal": utility_internal,
    }


class ResearchMetrics:
    """Collects per-step metrics and compiles run-level summary statistics."""

    FIRE_THRESHOLD = 0.05
    FIRE_OUT_THRESHOLD = 1e-3
    OBSERVE_EVERY_STEPS = 1
    # Set true for more detailed step metrics (impact performance)
    EMIT_EXTENDED_STEP_METRICS = True

    def __init__(self, config, strategy_name: str, scenario_name: str):
        self.config = config
        self.strategy_name = strategy_name
        self.scenario_name = scenario_name
        self.max_steps = int(config['simulation']['max_steps'])
        self.seed: int | None = None

        self.data_buffer: list[dict] = []
        self.last_summary: dict = {}
        self.last_summaries: list[dict] = []

        self.run_layout: list[dict] = []
        self.parallel_data_buffers: dict[str, list[dict]] = {}
        self._layout_fastpath = False
        self._layout_groups = 0
        self._layout_group_size = 0
        self.last_observed_batch_size = int(config["simulation"]["batch_size"])

    def set_run_layout(self, run_layout: list[dict] | None) -> None:
        """
        Configure packed parallel runs over the batch dimension.

        Each run dict must include:
            run_id, start, end
        Optional metadata:
            strategy, scenario, seed, n_active_agents
        """
        self.run_layout = []
        self.parallel_data_buffers = {}
        self._layout_fastpath = False
        self._layout_groups = 0
        self._layout_group_size = 0

        if not run_layout:
            return

        ordered = sorted(run_layout, key=lambda x: int(x["start"]))
        for idx, run in enumerate(ordered):
            spec = deepcopy(run)
            spec.setdefault("run_id", f"run_{idx:04d}")
            spec.setdefault("strategy", self.strategy_name)
            spec.setdefault("scenario", self.scenario_name)
            spec.setdefault("seed", -1)
            spec.setdefault(
                "n_active_agents",
                int(self.config['swarm']['n_drones'])
            )
            spec["start"] = int(spec["start"])
            spec["end"] = int(spec["end"])
            if spec["end"] <= spec["start"]:
                raise ValueError(
                    f"Invalid run range for {spec['run_id']}: "
                    f"{spec['start']}..{spec['end']}"
                )
            self.run_layout.append(spec)
            self.parallel_data_buffers[spec["run_id"]] = []

        spans = [spec["end"] - spec["start"] for spec in self.run_layout]
        if spans and all(span == spans[0] for span in spans):
            group_size = int(spans[0])
            contiguous = True
            for i, spec in enumerate(self.run_layout):
                expected_start = i * group_size
                if spec["start"] != expected_start or spec["end"] != expected_start + group_size:
                    contiguous = False
                    break
            if contiguous:
                self._layout_fastpath = True
                self._layout_groups = len(self.run_layout)
                self._layout_group_size = group_size

    def get_run_layout(self) -> list[dict]:
        return deepcopy(self.run_layout)

    @staticmethod
    def _run_metadata(run: dict) -> dict:
        """Extract stable run metadata from a run-layout spec."""
        out = {}
        for key, value in run.items():
            if key in {"start", "end"}:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    out[key] = value.item()
                else:
                    out[key] = value.detach().cpu().tolist()
            else:
                out[key] = value
        return out

    @staticmethod
    def _normalize_strategy_diagnostics(
        diagnostics: dict[str, torch.Tensor | float] | None,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(diagnostics, dict) or batch_size <= 0:
            return {}

        out: dict[str, torch.Tensor] = {}
        for key, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().float()
                if tensor.numel() == 1:
                    tensor = tensor.reshape(1).expand(batch_size)
                elif tensor.ndim >= 1 and tensor.shape[0] == batch_size:
                    if tensor.ndim == 1:
                        tensor = tensor.reshape(batch_size)
                    else:
                        tensor = tensor.reshape(batch_size, -1).mean(dim=1)
                elif tensor.numel() == batch_size:
                    tensor = tensor.reshape(batch_size)
                else:
                    continue
            else:
                try:
                    tensor = torch.full((batch_size,), float(value))
                except (TypeError, ValueError):
                    continue
            out[str(key)] = tensor
        return out

    @staticmethod
    def _merge_diagnostics_into_row(
        row: dict,
        diagnostics: dict[str, torch.Tensor],
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        if not diagnostics:
            return
        for key, values in diagnostics.items():
            if start is None or end is None:
                row[key] = float(values.mean().item())
            else:
                row[key] = float(values[start:end].mean().item())

    def _extract_step_metrics(
        self,
        heat,
        fuel,
        alive_mask,
        states,
        batteries,
        payloads,
        attrition_events,
        attrition_total,
        initial_alive,
    ) -> dict:
        alive = alive_mask.squeeze(-1).float()
        states = states.squeeze(-1)
        alive_count = alive.sum(dim=1).clamp(min=1.0)

        active = ((states != DroneState.WAITING) & (alive > 0.0)).float()
        waiting = ((states == DroneState.WAITING) & (alive > 0.0)).float()
        exploring = ((states == DroneState.EXPLORING) & (alive > 0.0)).float()
        firefighting = ((states == DroneState.FIREFIGHTING) & (alive > 0.0)).float()
        returning = ((states == DroneState.RETURNING) & (alive > 0.0)).float()
        active_ratio = active.sum(dim=1) / alive_count

        fuel_total = fuel.sum(dim=(1, 2))
        fuel_pct = fuel.mean(dim=(1, 2))
        fire_coverage = (heat > self.FIRE_THRESHOLD).float().mean(dim=(1, 2))

        out = {
            "fuel_total_mean": fuel_total.mean().item(),
            "fuel_pct_mean": fuel_pct.mean().item(),
            "fire_coverage_mean": fire_coverage.mean().item(),
            "active_ratio_mean": active_ratio.mean().item(),
            "active_agents_mean": active.sum(dim=1).mean().item(),
            "alive_agents_mean": alive.sum(dim=1).mean().item(),
            "waiting_agents_mean": waiting.sum(dim=1).mean().item(),
            "exploring_agents_mean": exploring.sum(dim=1).mean().item(),
            "firefighting_agents_mean": firefighting.sum(dim=1).mean().item(),
            "returning_agents_mean": returning.sum(dim=1).mean().item(),
            "attrition_events_mean": attrition_events.mean().item(),
            "attrition_total_cumulative_mean": attrition_total.mean().item(),
            "initial_alive_mean": initial_alive.mean().item(),
        }

        if self.EMIT_EXTENDED_STEP_METRICS:
            wait_ratio = waiting.sum(dim=1) / alive_count
            exploring_ratio = exploring.sum(dim=1) / alive_count
            firefighting_ratio = firefighting.sum(dim=1) / alive_count
            returning_ratio = returning.sum(dim=1) / alive_count

            mean_heat = heat.mean(dim=(1, 2))
            battery = batteries.squeeze(-1)
            payload = payloads.squeeze(-1)
            battery_mean_alive = (battery * alive).sum(dim=1) / alive_count
            payload_mean_alive = (payload * alive).sum(dim=1) / alive_count

            fire_out_mask = fire_coverage <= self.FIRE_OUT_THRESHOLD
            fire_out_batches = fire_out_mask.float().mean().item()
            post_fire_active = 0.0
            if fire_out_mask.any():
                post_fire_active = active_ratio[fire_out_mask].mean().item()

            out.update({
                "fuel_pct_std": fuel_pct.std(unbiased=False).item(),
                "fire_coverage_std": fire_coverage.std(unbiased=False).item(),
                "heat_mean": mean_heat.mean().item(),
                "active_ratio_std": active_ratio.std(unbiased=False).item(),
                "waiting_ratio_mean": wait_ratio.mean().item(),
                "exploring_ratio_mean": exploring_ratio.mean().item(),
                "firefighting_ratio_mean": firefighting_ratio.mean().item(),
                "returning_ratio_mean": returning_ratio.mean().item(),
                "battery_mean_alive": battery_mean_alive.mean().item(),
                "payload_mean_alive": payload_mean_alive.mean().item(),
                "fire_out_batch_fraction": fire_out_batches,
                "post_fire_active_ratio": post_fire_active,
            })

        return out

    def _extract_group_metrics_vectorized(
        self,
        heat,
        fuel,
        alive_mask,
        states,
        batteries,
        payloads,
        attrition_events,
        attrition_total,
        initial_alive,
    ) -> dict[str, torch.Tensor]:
        """
        Vectorized metric extraction over (groups, group_batch, ...).

        Returns tensors of shape (groups,).
        """
        alive = alive_mask.squeeze(-1).float()   # (G, S, N)
        states = states.squeeze(-1)              # (G, S, N)
        alive_count = alive.sum(dim=2).clamp(min=1.0)  # (G, S)

        active = ((states != DroneState.WAITING) & (alive > 0.0)).float()
        waiting = ((states == DroneState.WAITING) & (alive > 0.0)).float()
        exploring = ((states == DroneState.EXPLORING) & (alive > 0.0)).float()
        firefighting = ((states == DroneState.FIREFIGHTING) & (alive > 0.0)).float()
        returning = ((states == DroneState.RETURNING) & (alive > 0.0)).float()
        active_ratio = active.sum(dim=2) / alive_count

        fuel_total = fuel.sum(dim=(2, 3))
        fuel_pct = fuel.mean(dim=(2, 3))                       # (G, S)
        fire_coverage = (heat > self.FIRE_THRESHOLD).float().mean(dim=(2, 3))

        out = {
            "fuel_total_mean": fuel_total.mean(dim=1),
            "fuel_pct_mean": fuel_pct.mean(dim=1),
            "fire_coverage_mean": fire_coverage.mean(dim=1),
            "active_ratio_mean": active_ratio.mean(dim=1),
            "active_agents_mean": active.sum(dim=2).mean(dim=1),
            "alive_agents_mean": alive.sum(dim=2).mean(dim=1),
            "waiting_agents_mean": waiting.sum(dim=2).mean(dim=1),
            "exploring_agents_mean": exploring.sum(dim=2).mean(dim=1),
            "firefighting_agents_mean": firefighting.sum(dim=2).mean(dim=1),
            "returning_agents_mean": returning.sum(dim=2).mean(dim=1),
            "attrition_events_mean": attrition_events.mean(dim=1),
            "attrition_total_cumulative_mean": attrition_total.mean(dim=1),
            "initial_alive_mean": initial_alive.mean(dim=1),
        }

        if self.EMIT_EXTENDED_STEP_METRICS:
            wait_ratio = waiting.sum(dim=2) / alive_count
            exploring_ratio = exploring.sum(dim=2) / alive_count
            firefighting_ratio = firefighting.sum(dim=2) / alive_count
            returning_ratio = returning.sum(dim=2) / alive_count

            mean_heat = heat.mean(dim=(2, 3))
            battery = batteries.squeeze(-1)
            payload = payloads.squeeze(-1)
            battery_mean_alive = (battery * alive).sum(dim=2) / alive_count
            payload_mean_alive = (payload * alive).sum(dim=2) / alive_count

            fire_out_mask = fire_coverage <= self.FIRE_OUT_THRESHOLD  # (G, S)
            fire_out_batches = fire_out_mask.float().mean(dim=1)
            fire_out_denom = fire_out_mask.float().sum(dim=1)
            fire_out_num = (active_ratio * fire_out_mask.float()).sum(dim=1)
            post_fire_active = torch.zeros_like(fire_out_num)
            valid = fire_out_denom > 0
            post_fire_active[valid] = fire_out_num[valid] / \
                fire_out_denom[valid]

            out.update({
                "fuel_pct_std": fuel_pct.std(dim=1, unbiased=False),
                "fire_coverage_std": fire_coverage.std(dim=1, unbiased=False),
                "heat_mean": mean_heat.mean(dim=1),
                "active_ratio_std": active_ratio.std(dim=1, unbiased=False),
                "waiting_ratio_mean": wait_ratio.mean(dim=1),
                "exploring_ratio_mean": exploring_ratio.mean(dim=1),
                "firefighting_ratio_mean": firefighting_ratio.mean(dim=1),
                "returning_ratio_mean": returning_ratio.mean(dim=1),
                "battery_mean_alive": battery_mean_alive.mean(dim=1),
                "payload_mean_alive": payload_mean_alive.mean(dim=1),
                "fire_out_batch_fraction": fire_out_batches,
                "post_fire_active_ratio": post_fire_active,
            })

        return out

    def observe(
        self,
        step_idx: int,
        physics: "PhysicsManager",
        swarm: "SwarmManager"
    ) -> None:
        """Extract lightweight, batch-aggregated metrics for this step."""
        stride = max(1, int(self.OBSERVE_EVERY_STEPS))
        if stride > 1 and (step_idx % stride != 0) and (step_idx != self.max_steps - 1):
            return

        heat = physics.state[:, 0]
        fuel = physics.state[:, 1]
        attrition_events = swarm.attrition_deaths_this_step
        attrition_total = swarm.attrition_deaths_total
        initial_alive = swarm.initial_alive_counts
        self.last_observed_batch_size = int(heat.shape[0])
        strategy_step_diag = self._normalize_strategy_diagnostics(
            getattr(swarm.strategy, "get_step_diagnostics", lambda: {})(),
            self.last_observed_batch_size,
        )

        if not self.run_layout:
            row = self._extract_step_metrics(
                heat=heat,
                fuel=fuel,
                alive_mask=swarm.alive_mask,
                states=swarm.states,
                batteries=swarm.batteries,
                payloads=swarm.payloads,
                attrition_events=attrition_events,
                attrition_total=attrition_total,
                initial_alive=initial_alive,
            )
            self._merge_diagnostics_into_row(row, strategy_step_diag)
            row["step"] = int(step_idx)
            self.data_buffer.append(row)
            return

        if self._layout_fastpath:
            g = self._layout_groups
            s = self._layout_group_size
            if heat.shape[0] == g * s:
                heat_g = heat.reshape(g, s, *heat.shape[1:])
                fuel_g = fuel.reshape(g, s, *fuel.shape[1:])
                alive_g = swarm.alive_mask.reshape(
                    g, s, *swarm.alive_mask.shape[1:])
                states_g = swarm.states.reshape(g, s, *swarm.states.shape[1:])
                bat_g = swarm.batteries.reshape(
                    g, s, *swarm.batteries.shape[1:])
                pay_g = swarm.payloads.reshape(g, s, *swarm.payloads.shape[1:])
                attrition_events_g = attrition_events.reshape(
                    g, s, *attrition_events.shape[1:]
                )
                attrition_total_g = attrition_total.reshape(
                    g, s, *attrition_total.shape[1:]
                )
                initial_alive_g = initial_alive.reshape(
                    g, s, *initial_alive.shape[1:]
                )
            else:
                # Safety fallback if runtime batch shape diverges from layout.
                self._layout_fastpath = False
                heat_g = fuel_g = alive_g = states_g = bat_g = pay_g = None
                attrition_events_g = attrition_total_g = initial_alive_g = None

            if self._layout_fastpath:
                group_metrics = self._extract_group_metrics_vectorized(
                    heat=heat_g,
                    fuel=fuel_g,
                    alive_mask=alive_g,
                    states=states_g,
                    batteries=bat_g,
                    payloads=pay_g,
                    attrition_events=attrition_events_g.squeeze(-1),
                    attrition_total=attrition_total_g.squeeze(-1),
                    initial_alive=initial_alive_g.squeeze(-1),
                )
                fields = list(group_metrics.keys())
                if fields:
                    mat = torch.stack([group_metrics[k]
                                      for k in fields], dim=1)
                    mat_np = mat.detach().cpu().numpy()
                else:
                    mat_np = None

                for run_idx, run in enumerate(self.run_layout):
                    run_max_steps = int(run.get("run_max_steps", self.max_steps))
                    if step_idx >= run_max_steps:
                        continue
                    rid = run["run_id"]
                    row = {"step": int(step_idx), **self._run_metadata(run)}
                    if mat_np is not None:
                        vals = mat_np[run_idx]
                        for field_idx, key in enumerate(fields):
                            row[key] = float(vals[field_idx])
                    self._merge_diagnostics_into_row(
                        row,
                        strategy_step_diag,
                        start=int(run["start"]),
                        end=int(run["end"]),
                    )
                    self.parallel_data_buffers[rid].append(row)
                return

        for run in self.run_layout:
            run_max_steps = int(run.get("run_max_steps", self.max_steps))
            if step_idx >= run_max_steps:
                continue
            start = int(run["start"])
            end = int(run["end"])
            rid = run["run_id"]
            row = self._extract_step_metrics(
                heat=heat[start:end],
                fuel=fuel[start:end],
                alive_mask=swarm.alive_mask[start:end],
                states=swarm.states[start:end],
                batteries=swarm.batteries[start:end],
                payloads=swarm.payloads[start:end],
                attrition_events=attrition_events[start:end],
                attrition_total=attrition_total[start:end],
                initial_alive=initial_alive[start:end],
            )
            row.update({
                "step": int(step_idx),
                **self._run_metadata(run),
            })
            self._merge_diagnostics_into_row(
                row,
                strategy_step_diag,
                start=start,
                end=end,
            )
            self.parallel_data_buffers[rid].append(row)

    def _build_summary(
        self,
        df: pd.DataFrame,
        sps: float,
        strategy: str,
        scenario: str,
        seed: int,
    ) -> dict:
        return build_run_summary_from_steps(
            df,
            sps=sps,
            strategy=strategy,
            scenario=scenario,
            seed=seed,
            default_n_drones=int(self.config["swarm"]["n_drones"]),
            default_run_max_steps=int(self.config["simulation"]["max_steps"]),
            fire_out_threshold=self.FIRE_OUT_THRESHOLD,
        )

    def get_final_result(self, sps: float, strategy=None):
        """Return step-level dataframe and cache summary metrics."""
        summary_diag = self._normalize_strategy_diagnostics(
            getattr(strategy, "get_debug_summary", lambda: {})(),
            self.last_observed_batch_size,
        )
        if not self.run_layout:
            df = pd.DataFrame(self.data_buffer)
            summary = self._build_summary(
                df=df,
                sps=sps,
                strategy=self.strategy_name,
                scenario=self.scenario_name,
                seed=self.seed if self.seed is not None else -1,
            )
            self._merge_diagnostics_into_row(summary, summary_diag)
            self.last_summary = summary
            self.last_summaries = [summary]

            if not df.empty:
                df["strategy"] = self.strategy_name
                df["scenario"] = self.scenario_name
                df["seed"] = summary["seed"]
                df["n_drones"] = int(self.config["swarm"]["n_drones"])
                df["max_steps"] = int(self.config["simulation"]["max_steps"])
                df["run_max_steps"] = int(self.config["simulation"]["max_steps"])
                df["steps_per_second"] = summary["steps_per_second"]
            return df

        tables = []
        self.last_summaries = []
        self.last_summary = {}
        for run in self.run_layout:
            rid = run["run_id"]
            run_df = pd.DataFrame(self.parallel_data_buffers.get(rid, []))
            summary = self._build_summary(
                df=run_df,
                sps=sps,
                strategy=str(run["strategy"]),
                scenario=str(run["scenario"]),
                seed=int(run["seed"]),
            )
            self._merge_diagnostics_into_row(
                summary,
                summary_diag,
                start=int(run["start"]),
                end=int(run["end"]),
            )
            summary.update(self._run_metadata(run))
            self.last_summaries.append(summary)

            if not run_df.empty:
                run_df["steps_per_second"] = summary["steps_per_second"]
                tables.append(run_df)

        if len(self.last_summaries) == 1:
            self.last_summary = dict(self.last_summaries[0])

        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

    def get_run_summary(self) -> dict:
        if self.last_summary:
            return dict(self.last_summary)
        if self.last_summaries:
            return dict(self.last_summaries[0])
        return {}

    def get_run_summaries(self) -> list[dict]:
        if self.last_summaries:
            return [dict(row) for row in self.last_summaries]
        if self.last_summary:
            return [dict(self.last_summary)]
        return []

    def reset(self, seed: int | list[int] | None = None) -> None:
        if isinstance(seed, list):
            self.seed = int(seed[0]) if seed else None
        elif seed is None:
            self.seed = None
        else:
            self.seed = int(seed)

        self.data_buffer = []
        self.last_summary = {}
        self.last_summaries = []
        self.last_observed_batch_size = int(self.config["simulation"]["batch_size"])
        if self.run_layout:
            self.parallel_data_buffers = {
                run["run_id"]: [] for run in self.run_layout
            }
