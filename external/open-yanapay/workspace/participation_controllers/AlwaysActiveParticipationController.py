from __future__ import annotations

from src.participation_controller import (
    ParticipationController,
    ParticipationDecision,
    ParticipationObservation,
)


class AlwaysActiveParticipationController(ParticipationController):
    """Always keeps the robot active."""

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        self.name = "always"

    def decide(
        self,
        observation: ParticipationObservation,
    ) -> ParticipationDecision:
        del observation
        return ParticipationDecision(
            action=self.ACTION_STAY_ACTIVE,
            score=1.0,
            tau=1.0,
            regret_ema=0.0,
        )
