from __future__ import annotations

from dataclasses import dataclass

from src.participation_controller import (
    ParticipationDecision,
    ParticipationObservation,
)
from src.wildfire_transfer import WildfireTransferParticipationController


@dataclass(frozen=True)
class LabellaSpec:
    p_init: float = 0.10
    p_min: float = 0.02
    p_max: float = 0.85
    delta: float = 0.06
    max_failed_decisions_before_return: int = 3
    waiting_retry_steps: int = 3


class LabellaParticipationController(WildfireTransferParticipationController):
    """Yanapay port of wildfire's sortie-evaluated LaBella controller."""

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        self.name = "labella"
        self.spec = LabellaSpec(
            p_init=float(self.params.get("pInit", 0.10)),
            p_min=float(self.params.get("pMin", 0.02)),
            p_max=float(self.params.get("pMax", 0.85)),
            delta=float(self.params.get("delta", 0.06)),
            max_failed_decisions_before_return=self._get_non_negative_int(
                "maxFailedDecisionsBeforeReturn", 3
            ),
            waiting_retry_steps=self._get_non_negative_int("waitingRetrySteps", 3),
        )

    def _initial_state(self, observation: ParticipationObservation):
        state = super()._initial_state(observation)
        state.p_l = self.spec.p_init
        return state

    def decide(
        self,
        observation: ParticipationObservation,
    ) -> ParticipationDecision:
        state = self._get_state(observation)
        signals, transition = self._build_transfer_features(observation, state)
        del signals

        if transition.success_end:
            state.streak = max(state.streak + 1.0, 1.0)
            state.p_l = max(
                self.spec.p_min,
                min(self.spec.p_max, state.p_l + state.streak * self.spec.delta),
            )
            state.sortie_had_success = False
            state.sortie_no_success_decisions = 0
        elif transition.failure_end:
            state.streak = min(state.streak - 1.0, -1.0)
            state.p_l = max(
                self.spec.p_min,
                min(self.spec.p_max, state.p_l + state.streak * self.spec.delta),
            )
            state.sortie_had_success = False
            state.sortie_no_success_decisions = 0

        roll = self._deterministic_roll(
            observation.simulation_id,
            observation.robot_id,
            observation.tick,
            salt=self.name,
        )
        wait_ready = state.wait_retry <= 0
        wait_go = roll < state.p_l

        if observation.task_committed:
            action = self.ACTION_STAY_ACTIVE
        elif transition.waiting:
            sensed_local_need = float(observation.unresolved_nearby) > 0
            if wait_ready and wait_go and sensed_local_need:
                action = self.ACTION_STAY_ACTIVE
            else:
                if wait_ready:
                    state.wait_retry = self.spec.waiting_retry_steps
                else:
                    state.wait_retry = max(state.wait_retry - 1, 0)
                action = self.ACTION_RETURN_TO_RESERVE
        elif transition.current_active:
            next_fail_age = state.sortie_no_success_decisions + 1
            active_go = state.sortie_had_success or (
                next_fail_age < self.spec.max_failed_decisions_before_return
            )
            if state.sortie_had_success:
                state.sortie_no_success_decisions = 0
            else:
                state.sortie_no_success_decisions = next_fail_age
            action = (
                self.ACTION_STAY_ACTIVE
                if active_go
                else self.ACTION_RETURN_TO_RESERVE
            )
        else:
            action = self.ACTION_STAY_ACTIVE if wait_go else self.ACTION_RETURN_TO_RESERVE

        return ParticipationDecision(
            action=action,
            score=state.p_l,
            tau=state.p_l,
            regret_ema=float(state.sortie_no_success_decisions),
        )
