"""Curated strategy factory for the official strategy surface."""

from __future__ import annotations

from src.strategies.base import AlwaysStrategy, NeverStrategy, RLStrategy
from src.strategies.frustration_threshold_adaptive import (
    FrustrationThresholdAdaptiveHighPerformanceStrategy,
    FrustrationThresholdAdaptiveStrategy,
)
from src.strategies.labella_sortie_tuned import (
    LabellaSortieTunedHighPerformanceStrategy,
    LabellaSortieTunedStrategy,
)
from src.strategies.mappo import MAPPOStrategy


_ALIASES = {
    "frustration_threshold_adaptive": "frustration",
    "frustration_threshold_adaptive_hp": "frustration_threshold_adaptive_high_performance",
    "labella_sortie_tuned": "labella",
    "labella_sortie_tuned_hp": "labella_sortie_tuned_high_performance",
}


class StrategyFactory:
    @staticmethod
    def create(name: str, config: dict):
        key = _ALIASES.get(str(name).strip(), str(name).strip())

        builders = {
            "frustration": lambda cfg: FrustrationThresholdAdaptiveStrategy(cfg),
            "frustration_threshold_adaptive_high_performance": (
                lambda cfg: FrustrationThresholdAdaptiveHighPerformanceStrategy(cfg)
            ),
            "labella": lambda cfg: LabellaSortieTunedStrategy(cfg),
            "labella_sortie_tuned_high_performance": (
                lambda cfg: LabellaSortieTunedHighPerformanceStrategy(cfg)
            ),
            "mappo": lambda cfg: MAPPOStrategy(cfg),
            "always": lambda cfg: AlwaysStrategy(cfg),
            "never": lambda cfg: NeverStrategy(cfg),
            "rl_training": lambda cfg: RLStrategy(cfg),
        }
        builder = builders.get(key)
        if builder is None:
            raise ValueError(f"Unknown strategy: {name}")
        return builder(config)
