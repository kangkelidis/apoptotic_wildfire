"""LaBella-style sortie controller with mission-end adaptation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from src.swarm.constants import DroneState

from .base import BaseStrategy, SensorData


@dataclass(frozen=True)
class LabellaSortieTunedSpec:
    """Code-only tuning surface for a sortie-evaluated LaBella controller."""

    p_init: float = 0.06
    p_min: float = 0.02
    p_max: float = 0.85
    delta: float = 0.06
    max_failed_decisions_before_return: int = 3
    waiting_retry_steps: int = 3
    payload_drop_epsilon: float = 0.01

    @classmethod
    def high_performance(cls) -> "LabellaSortieTunedSpec":
        """Aggressive preserved preset optimized for preserve-first runs."""
        return cls(
            p_init=0.5,
            p_min=0.1,
            p_max=0.9,
            delta=0.06,
            max_failed_decisions_before_return=3,
            waiting_retry_steps=2,
            payload_drop_epsilon=0.01,
        )


class LabellaSortieTunedStrategy(BaseStrategy):
    """
    LaBella-style controller that updates only on sortie completion.

    The adaptation mechanism remains strictly success/failure streak based:
    - success: the sortie dropped payload at least once before returning
    - failure: the sortie returned or terminated without any payload drop

    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "labella"
        self.B = int(config["simulation"]["batch_size"])
        self.N = int(config["swarm"]["n_drones"])
        self.spec = self._build_spec(config)
        self._init_internal_state()

    def _base_spec(self) -> LabellaSortieTunedSpec:
        return LabellaSortieTunedSpec()

    def _build_spec(self, config: dict) -> LabellaSortieTunedSpec:
        spec = self._base_spec()
        runtime_cfg = config.get("runtime", {}).get("labella_sortie_tuned")
        if not isinstance(runtime_cfg, dict):
            return spec
        updates = {}
        for key in (
            "p_init",
            "p_min",
            "p_max",
            "delta",
            "max_failed_decisions_before_return",
            "waiting_retry_steps",
            "payload_drop_epsilon",
        ):
            if key in runtime_cfg:
                updates[key] = runtime_cfg[key]
        return replace(spec, **updates)

    def _init_internal_state(self) -> None:
        shape = (self.B, self.N, 1)
        self.p_l = torch.full(shape, self.spec.p_init, device=self.device)
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
        self._last_wait_go_mean = torch.zeros(self.B, device=self.device)
        self._last_active_go_mean = torch.zeros(self.B, device=self.device)
        self._last_fail_age_mean = torch.zeros(self.B, device=self.device)
        self._last_sortie_success_end_mean = torch.zeros(
            self.B, device=self.device)
        self._last_sortie_failure_end_mean = torch.zeros(
            self.B, device=self.device)
        self._last_wait_block_mean = torch.zeros(self.B, device=self.device)

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

    def observe_step(self, sensor_data: SensorData) -> None:
        states = sensor_data.states
        payload = sensor_data.payload
        alive_mask = sensor_data.alive_mask
        if states is None or payload is None:
            raise ValueError(
                "labella_sortie_tuned requires states and payload.")

        self._ensure_batch_shape(states.shape[0])

        alive = (
            alive_mask > 0.5
            if alive_mask is not None
            else torch.ones_like(states, dtype=torch.bool)
        )
        self._last_alive_mask = alive.clone()

        in_sortie = alive & (states != int(DroneState.WAITING))
        waiting = alive & (states == int(DroneState.WAITING))
        payload_dropped = (
            in_sortie &
            self.prev_sortie_active &
            (payload < (self.prev_payload - self.spec.payload_drop_epsilon))
        )
        carried_success = self.sortie_dropped_any | payload_dropped

        sortie_ended = self.prev_sortie_active & (~in_sortie)
        success_end = sortie_ended & carried_success
        failure_end = sortie_ended & (~carried_success)
        adapt_mask = success_end | failure_end

        next_streak = self.streak
        next_streak = torch.where(
            success_end,
            torch.clamp(next_streak + 1, min=1),
            next_streak,
        )
        next_streak = torch.where(
            failure_end,
            torch.clamp(next_streak - 1, max=-1),
            next_streak,
        )
        adapted_p = torch.clamp(
            self.p_l + (next_streak.float() * self.spec.delta),
            min=self.spec.p_min,
            max=self.spec.p_max,
        )
        self.streak = torch.where(adapt_mask, next_streak, self.streak)
        self.p_l = torch.where(adapt_mask, adapted_p, self.p_l)

        newly_active = in_sortie & (~self.prev_sortie_active)
        self.sortie_dropped_any = torch.where(
            in_sortie,
            torch.where(newly_active, payload_dropped, carried_success),
            torch.zeros_like(self.sortie_dropped_any),
        )
        self.sortie_no_drop_decisions = torch.where(
            in_sortie,
            torch.where(
                newly_active | self.sortie_dropped_any,
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
        states = sensor_data.states
        active_mask = sensor_data.active_mask.bool()
        alive_mask = sensor_data.alive_mask
        if states is None:
            raise ValueError("labella_sortie_tuned requires states.")

        self._ensure_batch_shape(states.shape[0])

        alive = (
            alive_mask > 0.5
            if alive_mask is not None
            else torch.ones_like(active_mask, dtype=torch.bool)
        )
        self._last_alive_mask = alive.clone()

        waiting_mask = active_mask & (states == int(DroneState.WAITING))
        field_mask = active_mask & (
            (states == int(DroneState.EXPLORING)) |
            (states == int(DroneState.FIREFIGHTING))
        )
        wait_ready = waiting_mask & (self.wait_retry <= 0)

        launch_rolls = torch.rand_like(self.p_l)
        wait_go = wait_ready & (launch_rolls < self.p_l)

        next_fail_age = self.sortie_no_drop_decisions
        active_no_success = field_mask & (~self.sortie_dropped_any)
        next_fail_age = torch.where(
            active_no_success, self.sortie_no_drop_decisions + 1, next_fail_age)
        next_fail_age = torch.where(
            field_mask & self.sortie_dropped_any,
            torch.zeros_like(next_fail_age),
            next_fail_age,
        )

        active_go = field_mask & (
            self.sortie_dropped_any |
            (next_fail_age < self.spec.max_failed_decisions_before_return)
        )

        self.sortie_no_drop_decisions = torch.where(
            field_mask,
            torch.where(self.sortie_dropped_any, torch.zeros_like(
                next_fail_age), next_fail_age),
            self.sortie_no_drop_decisions,
        )
        retry_fill = torch.full_like(
            self.wait_retry, self.spec.waiting_retry_steps)
        self.wait_retry = torch.where(
            wait_ready & (~wait_go),
            retry_fill,
            self.wait_retry,
        )

        intent = torch.zeros_like(active_mask, dtype=torch.long)
        intent = torch.where(waiting_mask, wait_go.long(), intent)
        intent = torch.where(field_mask, active_go.long(), intent)

        self._last_wait_go_mean = self._masked_batch_mean(
            wait_go.float(), alive & waiting_mask)
        self._last_wait_block_mean = self._masked_batch_mean(
            (waiting_mask & (~wait_ready)).float(), alive & waiting_mask)
        self._last_active_go_mean = self._masked_batch_mean(
            active_go.float(), alive & field_mask)
        self._last_fail_age_mean = self._masked_batch_mean(
            self.sortie_no_drop_decisions.float(),
            alive & (states != int(DroneState.WAITING)),
        )
        return intent

    def get_step_diagnostics(self) -> dict[str, torch.Tensor]:
        alive = self._last_alive_mask
        return {
            "lb_p_mean": self._masked_batch_mean(self.p_l, alive),
            "lb_p_std": self._masked_batch_std(self.p_l, alive),
            "lb_wait_go_mean": self._last_wait_go_mean.clone(),
            "lb_active_go_mean": self._last_active_go_mean.clone(),
            "lb_fail_age_mean": self._last_fail_age_mean.clone(),
            "lb_wait_block_mean": self._last_wait_block_mean.clone(),
            "lb_sortie_success_end_mean": self._last_sortie_success_end_mean.clone(),
            "lb_sortie_failure_end_mean": self._last_sortie_failure_end_mean.clone(),
        }


class LabellaSortieTunedHighPerformanceStrategy(LabellaSortieTunedStrategy):
    """Performance-first preserved preset for LaBella sortie control."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "labella_sortie_tuned_high_performance"

    def _base_spec(self) -> LabellaSortieTunedSpec:
        return LabellaSortieTunedSpec.high_performance()
