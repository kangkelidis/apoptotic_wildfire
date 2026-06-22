"""Minimal private-threshold frustration controller."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from src.swarm.constants import DroneState
from src.swarm.sensors import PerceptionEngine

from .base import BaseStrategy, SensorData


@dataclass(frozen=True)
class FrustrationThresholdAdaptiveSpec:
    """Code-only tuning surface for a minimal private-threshold controller."""

    visual_heat_gain: float = float(PerceptionEngine.ACTOR_VISUAL_HEAT_GAIN)
    memory_heat_gain: float = float(PerceptionEngine.ACTOR_MEMORY_HEAT_GAIN)

    tau_init: float = 0.4
    tau_spread: float = 0.2
    tau_min: float = 0.01
    tau_max: float = 0.7
    tau_delta: float = 0.1
    decision_gain: float = 10.0
    max_failed_decisions_before_return: int = 3
    waiting_retry_steps: int = 3
    payload_drop_epsilon: float = 0.01

    @classmethod
    def high_performance(cls) -> "FrustrationThresholdAdaptiveSpec":
        """Aggressive performance-first preset preserved separately from defaults."""
        return cls(
            tau_init=0.7,
            tau_spread=0.2,
            tau_min=0.10,
            tau_max=0.90,
            tau_delta=0.1,
            decision_gain=10.0,
            max_failed_decisions_before_return=3,
            waiting_retry_steps=2,
            payload_drop_epsilon=0.01,
        )


class FrustrationThresholdScorer:
    REQUIRED_FEATURES = (
        "visual_heat",
        "memory_heat",
        "social_pressure",
        "trend_success",
        "trend_giveup",
    )

    def __init__(self, spec: FrustrationThresholdAdaptiveSpec):
        self.spec = spec

    def compute(
        self,
        *,
        actor_obs: torch.Tensor,
        feature_index: dict[str, int] | None,
    ) -> dict[str, torch.Tensor]:
        if feature_index is None:
            raise ValueError(
                "frustration_threshold_adaptive requires SensorData.feature_index."
            )

        missing = [
            name for name in self.REQUIRED_FEATURES if name not in feature_index]
        if missing:
            raise ValueError(
                "frustration_threshold_adaptive missing actor feature indices for "
                f"{missing}."
            )

        def feature(name: str) -> torch.Tensor:
            idx = int(feature_index[name])
            return actor_obs[..., idx:idx + 1]

        visual_heat = torch.clamp(
            feature("visual_heat") / self.spec.visual_heat_gain,
            0.0,
            1.0,
        )
        memory_heat = torch.clamp(
            feature("memory_heat") / self.spec.memory_heat_gain,
            0.0,
            1.0,
        )
        trend_success = torch.clamp(feature("trend_success"), 0.0, 1.0)
        trend_giveup = torch.clamp(feature("trend_giveup"), 0.0, 1.0)
        crowd_excess = torch.clamp(feature("social_pressure") - 1.0, 0.0, 1.0)

        utility = (visual_heat + memory_heat + trend_success) / 3.0
        density = (crowd_excess + trend_giveup) / 2.0
        frustration = torch.clamp((density + (1.0 - utility)) / 2.0, 0.0, 1.0)

        return {
            "utility": utility,
            "density": density,
            "frustration": frustration,
        }


class FrustrationThresholdAdaptiveStrategy(BaseStrategy):
    """
    Minimal threshold-based frustration controller.

    Each drone carries a private tolerance tau_i from the start. Decisions are
    probabilistic from tau_i - frustration_i. Sortie success raises tau_i and
    sortie failure lowers tau_i, so the threshold itself is the adaptive state.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "frustration_threshold_adaptive"
        self.B = int(config["simulation"]["batch_size"])
        self.N = int(config["swarm"]["n_drones"])
        self.spec = self._build_spec(config)
        self.scorer = FrustrationThresholdScorer(self.spec)
        self._init_internal_state()

    def _base_spec(self) -> FrustrationThresholdAdaptiveSpec:
        return FrustrationThresholdAdaptiveSpec()

    def _build_spec(self, config: dict) -> FrustrationThresholdAdaptiveSpec:
        spec = self._base_spec()
        runtime_cfg = config.get("runtime", {}).get(
            "frustration_threshold_adaptive")
        if not isinstance(runtime_cfg, dict):
            return spec
        updates = {}
        for key in (
            "visual_heat_gain", "memory_heat_gain",
            "tau_init", "tau_spread", "tau_min", "tau_max", "tau_delta",
            "decision_gain", "max_failed_decisions_before_return",
            "waiting_retry_steps", "payload_drop_epsilon",
        ):
            if key in runtime_cfg:
                updates[key] = runtime_cfg[key]
        return replace(spec, **updates)

    def _base_tau(self) -> torch.Tensor:
        if self.N <= 1:
            offsets = torch.zeros((1, 1, 1), device=self.device)
        else:
            rank = torch.linspace(-0.5, 0.5, self.N, device=self.device)
            offsets = (rank * self.spec.tau_spread).view(1, self.N, 1)
        base = self.spec.tau_init + offsets
        return torch.clamp(base, min=self.spec.tau_min, max=self.spec.tau_max)

    def _init_internal_state(self) -> None:
        shape = (self.B, self.N, 1)
        self.tau = self._base_tau().expand(shape).clone()
        self._tau_initial = self.tau.clone()
        self.streak = torch.zeros(shape, device=self.device)
        self.prev_payload = torch.ones(shape, device=self.device)
        self.prev_sortie_active = torch.zeros(
            shape, dtype=torch.bool, device=self.device)
        self.sortie_dropped_any = torch.zeros(
            shape, dtype=torch.bool, device=self.device)
        self.sortie_no_drop_decisions = torch.zeros(
            shape, dtype=torch.long, device=self.device)
        self.wait_retry = torch.zeros(
            shape, dtype=torch.long, device=self.device)

        self._last_alive_mask = torch.ones(
            shape, dtype=torch.bool, device=self.device)
        self._last_wait_prob_mean = torch.zeros(self.B, device=self.device)
        self._last_active_prob_mean = torch.zeros(self.B, device=self.device)
        self._last_wait_go_mean = torch.zeros(self.B, device=self.device)
        self._last_active_go_mean = torch.zeros(self.B, device=self.device)
        self._last_wait_block_mean = torch.zeros(self.B, device=self.device)
        self._last_launch_fraction = torch.zeros(self.B, device=self.device)
        self._last_return_fraction = torch.zeros(self.B, device=self.device)
        self._last_utility_mean = torch.zeros(self.B, device=self.device)
        self._last_density_mean = torch.zeros(self.B, device=self.device)
        self._last_frustration_mean = torch.zeros(self.B, device=self.device)
        self._last_margin_mean = torch.zeros(self.B, device=self.device)
        self._last_sortie_success_end_mean = torch.zeros(
            self.B, device=self.device)
        self._last_sortie_failure_end_mean = torch.zeros(
            self.B, device=self.device)

    def reset(self) -> None:
        self._init_internal_state()

    def _ensure_batch_shape(self, batch_size: int) -> None:
        if int(batch_size) != self.B:
            self.B = int(batch_size)
            self._init_internal_state()

    @staticmethod
    def _masked_batch_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = values.squeeze(-1).float()
        m = mask.squeeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    @classmethod
    def _masked_batch_std(cls, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = values.squeeze(-1).float()
        m = mask.squeeze(-1).float()
        mean = cls._masked_batch_mean(values, mask).unsqueeze(1)
        denom = m.sum(dim=1).clamp(min=1.0)
        centered = (x - mean) * m
        return torch.sqrt((centered * centered).sum(dim=1) / denom)

    def _require_fields(
        self, sensor_data: SensorData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        payload = sensor_data.payload
        alive_mask = sensor_data.alive_mask
        states = sensor_data.states
        feature_index = sensor_data.feature_index
        if payload is None or alive_mask is None or states is None:
            raise ValueError(
                "frustration_threshold_adaptive requires payload, alive_mask, and states."
            )
        if feature_index is None:
            raise ValueError(
                "frustration_threshold_adaptive requires SensorData.feature_index."
            )
        return payload, alive_mask, states, feature_index

    def observe_step(self, sensor_data: SensorData) -> None:
        payload, alive_mask, states, _feature_index = self._require_fields(
            sensor_data)
        self._ensure_batch_shape(states.shape[0])

        alive = alive_mask > 0.5
        self._last_alive_mask = alive.clone()
        in_sortie = alive & (states != int(DroneState.WAITING))
        waiting = alive & (states == int(DroneState.WAITING))

        payload_dropped = (
            in_sortie
            & self.prev_sortie_active
            & (payload < (self.prev_payload - self.spec.payload_drop_epsilon))
        )
        carried_success = self.sortie_dropped_any | payload_dropped
        sortie_ended = self.prev_sortie_active & (~in_sortie)
        success_end = sortie_ended & carried_success
        failure_end = sortie_ended & (~carried_success)

        next_streak = self.streak
        next_streak = torch.where(success_end, torch.clamp(
            next_streak + 1.0, min=1.0), next_streak)
        next_streak = torch.where(failure_end, torch.clamp(
            next_streak - 1.0, max=-1.0), next_streak)

        tau_update = torch.zeros_like(self.tau)
        tau_update = torch.where(
            success_end | failure_end,
            next_streak * self.spec.tau_delta,
            tau_update,
        )
        self.streak = torch.where(
            success_end | failure_end, next_streak, self.streak)
        self.tau = torch.clamp(
            self.tau + tau_update,
            min=self.spec.tau_min,
            max=self.spec.tau_max,
        )

        newly_active = in_sortie & (~self.prev_sortie_active)
        self.sortie_dropped_any = torch.where(
            in_sortie,
            torch.where(newly_active, payload_dropped, carried_success),
            torch.zeros_like(self.sortie_dropped_any),
        )
        self.sortie_no_drop_decisions = torch.where(
            in_sortie,
            torch.where(
                newly_active | payload_dropped,
                torch.zeros_like(self.sortie_no_drop_decisions),
                self.sortie_no_drop_decisions,
            ),
            torch.zeros_like(self.sortie_no_drop_decisions),
        )
        self.wait_retry = torch.where(
            waiting,
            torch.clamp(self.wait_retry - 1, min=0),
            torch.zeros_like(self.wait_retry),
        )
        self.prev_payload = torch.where(alive, payload, self.prev_payload)
        self.prev_sortie_active = in_sortie

        self._last_sortie_success_end_mean = self._masked_batch_mean(
            success_end.float(), alive)
        self._last_sortie_failure_end_mean = self._masked_batch_mean(
            failure_end.float(), alive)

    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        _payload, alive_mask, states, feature_index = self._require_fields(
            sensor_data)
        self._ensure_batch_shape(states.shape[0])

        alive = alive_mask > 0.5
        self._last_alive_mask = alive.clone()
        eligible = sensor_data.active_mask.bool() & alive

        signals = self.scorer.compute(
            actor_obs=sensor_data.actor_obs,
            feature_index=feature_index,
        )
        waiting_mask = eligible & (states == int(DroneState.WAITING))
        field_mask = eligible & (
            (states == int(DroneState.EXPLORING))
            | (states == int(DroneState.FIREFIGHTING))
        )

        decision_margin = self.tau - signals["frustration"]
        decision_prob = torch.sigmoid(
            self.spec.decision_gain * decision_margin)
        rolls = torch.rand_like(decision_prob)
        go_mask = eligible & (rolls < decision_prob)

        wait_ready = waiting_mask & (self.wait_retry <= 0)
        wait_go = wait_ready & go_mask

        next_fail_age = self.sortie_no_drop_decisions
        next_fail_age = torch.where(
            field_mask, self.sortie_no_drop_decisions + 1, next_fail_age)
        active_go = field_mask & (
            go_mask
            | (next_fail_age < self.spec.max_failed_decisions_before_return)
        )

        self.sortie_no_drop_decisions = torch.where(
            field_mask,
            torch.where(self.sortie_dropped_any, torch.zeros_like(
                next_fail_age), next_fail_age),
            self.sortie_no_drop_decisions,
        )
        retry_fill = torch.full_like(
            self.wait_retry, self.spec.waiting_retry_steps)
        self.wait_retry = torch.where(wait_ready & (
            ~wait_go), retry_fill, self.wait_retry)

        intent = torch.zeros_like(eligible, dtype=torch.long)
        intent = torch.where(waiting_mask, wait_go.long(), intent)
        intent = torch.where(field_mask, active_go.long(), intent)

        self._last_wait_prob_mean = self._masked_batch_mean(
            decision_prob, alive & waiting_mask)
        self._last_active_prob_mean = self._masked_batch_mean(
            decision_prob, alive & field_mask)
        self._last_wait_go_mean = self._masked_batch_mean(
            wait_go.float(), alive & waiting_mask)
        self._last_active_go_mean = self._masked_batch_mean(
            active_go.float(), alive & field_mask)
        self._last_wait_block_mean = self._masked_batch_mean(
            (waiting_mask & (~wait_ready)).float(),
            alive & waiting_mask,
        )
        self._last_launch_fraction = self._masked_batch_mean(
            wait_go.float(), alive)
        self._last_return_fraction = self._masked_batch_mean(
            (field_mask & (~active_go)).float(),
            alive,
        )
        self._last_utility_mean = self._masked_batch_mean(
            signals["utility"], alive)
        self._last_density_mean = self._masked_batch_mean(
            signals["density"], alive)
        self._last_frustration_mean = self._masked_batch_mean(
            signals["frustration"], alive)
        self._last_margin_mean = self._masked_batch_mean(
            decision_margin, alive)

        return intent

    def get_step_diagnostics(self) -> dict[str, torch.Tensor]:
        alive = self._last_alive_mask
        return {
            "fta_tau_mean": self._masked_batch_mean(self.tau, alive),
            "fta_tau_std": self._masked_batch_std(self.tau, alive),
            "fta_wait_prob_mean": self._last_wait_prob_mean.clone(),
            "fta_active_prob_mean": self._last_active_prob_mean.clone(),
            "fta_wait_go_mean": self._last_wait_go_mean.clone(),
            "fta_active_go_mean": self._last_active_go_mean.clone(),
            "fta_wait_block_mean": self._last_wait_block_mean.clone(),
            "fta_launch_fraction": self._last_launch_fraction.clone(),
            "fta_return_fraction": self._last_return_fraction.clone(),
            "fta_utility_mean": self._last_utility_mean.clone(),
            "fta_density_mean": self._last_density_mean.clone(),
            "fta_frustration_mean": self._last_frustration_mean.clone(),
            "fta_margin_mean": self._last_margin_mean.clone(),
            "fta_sortie_success_end_mean": self._last_sortie_success_end_mean.clone(),
            "fta_sortie_failure_end_mean": self._last_sortie_failure_end_mean.clone(),
        }

    def get_debug_summary(self) -> dict[str, torch.Tensor]:
        alive = self._last_alive_mask
        tau_delta = self.tau - self._tau_initial
        return {
            "fta_tau_final_mean": self._masked_batch_mean(self.tau, alive),
            "fta_tau_final_std": self._masked_batch_std(self.tau, alive),
            "fta_tau_delta_mean": self._masked_batch_mean(tau_delta, alive),
        }


class FrustrationThresholdAdaptiveHighPerformanceStrategy(
    FrustrationThresholdAdaptiveStrategy
):
    """Performance-first preset for the private-threshold frustration controller."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "frustration_threshold_adaptive_high_performance"

    def _base_spec(self) -> FrustrationThresholdAdaptiveSpec:
        return FrustrationThresholdAdaptiveSpec.high_performance()
