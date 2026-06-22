from __future__ import annotations

import math
from dataclasses import dataclass

from src.participation_controller import (
    ParticipationController,
    ParticipationObservation,
)


@dataclass
class WildfireTransferState:
    task_memory: float = 0.0
    encounter_memory: float = 1.0
    success_memory: float = 0.0
    giveup_memory: float = 0.0
    tau: float = 0.0
    streak: float = 0.0
    p_l: float = 0.0
    sortie_active: bool = False
    sortie_had_success: bool = False
    sortie_no_success_decisions: int = 0
    wait_retry: int = 0


@dataclass(frozen=True)
class WildfireTransferFeatures:
    visual_heat: float
    memory_heat: float
    social_pressure: float
    trend_success: float
    trend_giveup: float
    utility: float
    density: float
    frustration: float


@dataclass(frozen=True)
class SortieTransition:
    current_active: bool
    waiting: bool
    sortie_started: bool
    sortie_ended: bool
    success_end: bool
    failure_end: bool


class WildfireTransferParticipationController(ParticipationController):
    """Shared wildfire-style state and feature adapter for Yanapay participation."""

    MEMORY_ALPHA = 0.1

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        self._state: dict[tuple[str, int], WildfireTransferState] = {}
        self._num_robots = self._get_total_robots()

    def _get_total_robots(self) -> int:
        num_robots = getattr(self.scenario.netlogo_params, "num_of_robots", 1)
        if isinstance(num_robots, (list, tuple)):
            num_robots = num_robots[0] if num_robots else 1
        try:
            return max(1, int(num_robots))
        except Exception:
            return 1

    def _initial_state(self, observation: ParticipationObservation) -> WildfireTransferState:
        del observation
        return WildfireTransferState()

    def _get_state(self, observation: ParticipationObservation) -> WildfireTransferState:
        key = (observation.simulation_id, int(observation.robot_id))
        if key not in self._state:
            self._state[key] = self._initial_state(observation)
        return self._state[key]

    def cleanup_simulation(self, simulation_id: str) -> None:
        stale_keys = [key for key in self._state if key[0] == simulation_id]
        for key in stale_keys:
            del self._state[key]

    def _get_rank_offset(self, observation: ParticipationObservation, spread: float) -> float:
        if self._num_robots <= 1:
            return 0.0
        centred_rank = (
            (float(observation.robot_rank) / float(max(self._num_robots - 1, 1))) - 0.5
        )
        return centred_rank * float(spread)

    def _build_transfer_features(
        self,
        observation: ParticipationObservation,
        state: WildfireTransferState,
    ) -> tuple[WildfireTransferFeatures, SortieTransition]:
        current_active = bool(observation.active_flag or observation.task_committed)
        waiting = bool(observation.reserve_flag and not current_active)
        sortie_started = current_active and not state.sortie_active
        sortie_ended = state.sortie_active and not current_active

        visual_heat = self._scale_count(observation.unresolved_nearby, 3.0)
        state.task_memory = (
            (1.0 - self.MEMORY_ALPHA) * state.task_memory
            + self.MEMORY_ALPHA * visual_heat
        )

        current_density = max(float(observation.nearby_sars), 0.0)
        state.encounter_memory = max(
            (1.0 - self.MEMORY_ALPHA) * state.encounter_memory
            + self.MEMORY_ALPHA * current_density,
            1e-5,
        )

        success_count = max(float(observation.window_success_count), 0.0)
        giveup_count = max(
            float(observation.window_failed_request_count)
            + float(observation.window_staff_unavailable_count),
            0.0,
        )
        state.success_memory = (
            (1.0 - self.MEMORY_ALPHA) * state.success_memory
            + self.MEMORY_ALPHA * success_count
        )
        state.giveup_memory = (
            (1.0 - self.MEMORY_ALPHA) * state.giveup_memory
            + self.MEMORY_ALPHA * giveup_count
        )

        if sortie_started:
            state.sortie_had_success = False
            state.sortie_no_success_decisions = 0

        if current_active and success_count > 0:
            state.sortie_had_success = True

        success_end = sortie_ended and state.sortie_had_success
        failure_end = sortie_ended and not state.sortie_had_success

        state.sortie_active = current_active

        social_pressure = current_density / (state.encounter_memory + 1e-5)
        trend_success = state.success_memory / (state.encounter_memory + 1e-5)
        trend_giveup = state.giveup_memory / (state.encounter_memory + 1e-5)

        utility = (
            visual_heat + self._clamp(state.task_memory) + self._clamp(trend_success)
        ) / 3.0
        density = (
            self._clamp(social_pressure - 1.0) + self._clamp(trend_giveup)
        ) / 2.0
        frustration = self._clamp((density + (1.0 - utility)) / 2.0)

        return (
            WildfireTransferFeatures(
                visual_heat=visual_heat,
                memory_heat=self._clamp(state.task_memory),
                social_pressure=social_pressure,
                trend_success=trend_success,
                trend_giveup=trend_giveup,
                utility=utility,
                density=density,
                frustration=frustration,
            ),
            SortieTransition(
                current_active=current_active,
                waiting=waiting,
                sortie_started=sortie_started,
                sortie_ended=sortie_ended,
                success_end=success_end,
                failure_end=failure_end,
            ),
        )
