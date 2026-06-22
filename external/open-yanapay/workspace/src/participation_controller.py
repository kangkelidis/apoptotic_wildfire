"""
Base classes for SAR participation control.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from utils.helper import setup_logger
from utils.paths import PARTICIPATION_CONTROLLERS_FOLDER


@dataclass(frozen=True)
class ParticipationObservation:
    """Local state exposed by a SAR robot to the participation layer."""

    simulation_id: str
    robot_id: int
    robot_rank: int
    tick: int
    unresolved_nearby: float
    nearest_victim_distance: float
    available_bystanders: float
    available_staff: float
    recent_success_rate: float
    nearby_sars: float
    recent_failed_requests: float
    recent_staff_unavailable: float
    window_success_count: float
    window_failed_request_count: float
    window_staff_unavailable_count: float
    task_committed: bool
    reserve_flag: bool
    active_flag: bool
    distance_to_nearest_victim: float


@dataclass(frozen=True)
class ParticipationDecision:
    """Decision returned to NetLogo."""

    action: str
    score: float = 0.0
    tau: float = 0.0
    regret_ema: float = 0.0

    def to_netlogo_literal(self) -> str:
        return (
            f"[\"{self.action}\" "
            f"{self.score:.6f} "
            f"{self.tau:.6f} "
            f"{self.regret_ema:.6f}]"
        )


class ParticipationController(ABC):
    """Abstract base class for all participation controllers."""

    logger = setup_logger()
    ACTION_STAY_ACTIVE = "stay-active"
    ACTION_RETURN_TO_RESERVE = "return-to-reserve"
    CONTROLLER_ALIASES = {
        "always": "AlwaysActiveParticipationController",
        "always_active": "AlwaysActiveParticipationController",
        "frustration": "FrustrationParticipationController",
        "labella": "LabellaParticipationController",
    }

    @staticmethod
    def _resolve_controllers_folder(controllers_folder: str) -> str:
        if os.path.exists(controllers_folder):
            return controllers_folder
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "participation_controllers")
        )

    @staticmethod
    def get_controller(
        controller_name: str,
        scenario,
        controllers_folder: str = PARTICIPATION_CONTROLLERS_FOLDER,
    ) -> Optional["ParticipationController"]:
        controllers_folder = ParticipationController._resolve_controllers_folder(
            controllers_folder
        )
        controller_name = ParticipationController.CONTROLLER_ALIASES.get(
            controller_name,
            controller_name,
        )
        try:
            for file_name in os.listdir(controllers_folder):
                if file_name.endswith(".py") and file_name[:-3] == controller_name:
                    module = importlib.import_module(
                        "participation_controllers." + controller_name
                    )
                    controller_class = getattr(module, controller_name)
                    if issubclass(controller_class, ParticipationController):
                        return controller_class(scenario)
        except Exception as exc:
            ParticipationController.logger.error(
                "Error in get_participation_controller: %s", exc
            )
            traceback.print_exc()
        raise FileNotFoundError(
            f"Failed to get participation controller {controller_name}"
        )

    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.params = dict(getattr(scenario, "participation_params", {}) or {})
        self.name = "base"

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _inverse_distance(distance: float) -> float:
        if distance < 0:
            return 0.0
        return 1.0 / (1.0 + float(distance))

    @staticmethod
    def _scale_count(value: float, normaliser: float) -> float:
        if normaliser <= 0:
            return 0.0
        return ParticipationController._clamp(float(value) / float(normaliser))

    def _get_weight(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    def _get_tau_init(self, default: float = 0.45) -> float:
        return float(self.params.get("tauInit", default))

    def _get_non_negative_int(self, key: str, default: int = 0) -> int:
        return max(0, int(self.params.get(key, default)))

    @abstractmethod
    def decide(
        self,
        observation: ParticipationObservation,
    ) -> ParticipationDecision:
        """Return stay-active or return-to-reserve for the given observation."""

    def decide_batch(
        self,
        observations: list[ParticipationObservation],
    ) -> list[tuple[int, ParticipationDecision]]:
        return [
            (int(observation.robot_id), self.decide(observation))
            for observation in observations
        ]

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(value)))

    @staticmethod
    def _deterministic_roll(
        simulation_id: str,
        robot_id: int,
        tick: int,
        salt: str = "",
    ) -> float:
        token = f"{simulation_id}:{robot_id}:{tick}:{salt}".encode("ascii", "ignore")
        digest = hashlib.sha256(token).hexdigest()
        return int(digest[:12], 16) / float(16 ** 12)

    def cleanup_simulation(self, simulation_id: str) -> None:
        del simulation_id

    def __str__(self) -> str:
        return self.__class__.__name__
