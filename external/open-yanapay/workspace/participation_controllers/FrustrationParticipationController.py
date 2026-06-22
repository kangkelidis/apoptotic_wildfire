from __future__ import annotations

from dataclasses import dataclass

from src.participation_controller import (
    ParticipationDecision,
    ParticipationObservation,
)
from src.wildfire_transfer import WildfireTransferParticipationController


@dataclass(frozen=True)
class FrustrationSpec:
    tau_init: float = 0.4
    tau_spread: float = 0.2
    tau_min: float = 0.01
    tau_max: float = 0.7
    tau_delta: float = 0.1
    decision_gain: float = 10.0
    max_failed_decisions_before_return: int = 3
    waiting_retry_steps: int = 3


class FrustrationParticipationController(WildfireTransferParticipationController):
    """Yanapay port of the wildfire private-threshold frustration controller."""

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        self.name = "frustration"
        self.spec = FrustrationSpec(
            tau_init=float(self.params.get("tauInit", 0.4)),
            tau_spread=float(self.params.get("tauSpread", 0.2)),
            tau_min=float(self.params.get("tauMin", 0.01)),
            tau_max=float(self.params.get("tauMax", 0.7)),
            tau_delta=float(self.params.get("tauDelta", 0.1)),
            decision_gain=float(self.params.get("decisionGain", 10.0)),
            max_failed_decisions_before_return=self._get_non_negative_int(
                "maxFailedDecisionsBeforeReturn", 3
            ),
            waiting_retry_steps=self._get_non_negative_int("waitingRetrySteps", 3),
        )

    def _initial_state(self, observation: ParticipationObservation):
        state = super()._initial_state(observation)
        base_tau = self.spec.tau_init + self._get_rank_offset(
            observation, self.spec.tau_spread
        )
        state.tau = max(self.spec.tau_min, min(self.spec.tau_max, base_tau))
        return state

    def decide(
        self,
        observation: ParticipationObservation,
    ) -> ParticipationDecision:
        state = self._get_state(observation)
        signals, transition = self._build_transfer_features(observation, state)

        if transition.success_end:
            state.streak = max(state.streak + 1.0, 1.0)
            state.tau = max(
                self.spec.tau_min,
                min(self.spec.tau_max, state.tau + state.streak * self.spec.tau_delta),
            )
            state.sortie_had_success = False
            state.sortie_no_success_decisions = 0
        elif transition.failure_end:
            state.streak = min(state.streak - 1.0, -1.0)
            state.tau = max(
                self.spec.tau_min,
                min(self.spec.tau_max, state.tau + state.streak * self.spec.tau_delta),
            )
            state.sortie_had_success = False
            state.sortie_no_success_decisions = 0

        decision_margin = state.tau - signals.frustration
        decision_prob = self._sigmoid(self.spec.decision_gain * decision_margin)
        roll = self._deterministic_roll(
            observation.simulation_id,
            observation.robot_id,
            observation.tick,
            salt=self.name,
        )
        go_signal = roll < decision_prob

        if observation.task_committed:
            action = self.ACTION_STAY_ACTIVE
        elif transition.waiting:
            wait_ready = state.wait_retry <= 0
            sensed_local_need = float(observation.unresolved_nearby) > 0
            wait_go = wait_ready and sensed_local_need and go_signal
            if wait_ready and not wait_go:
                state.wait_retry = self.spec.waiting_retry_steps
            elif transition.waiting and state.wait_retry > 0:
                state.wait_retry = max(state.wait_retry - 1, 0)
            action = (
                self.ACTION_STAY_ACTIVE
                if wait_go
                else self.ACTION_RETURN_TO_RESERVE
            )
        elif transition.current_active:
            next_fail_age = state.sortie_no_success_decisions + 1
            active_go = go_signal or (
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
            action = self.ACTION_STAY_ACTIVE if go_signal else self.ACTION_RETURN_TO_RESERVE

        return ParticipationDecision(
            action=action,
            score=signals.frustration,
            tau=state.tau,
            regret_ema=decision_prob,
        )
