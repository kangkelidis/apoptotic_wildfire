"""
Strategy: Abstract Base and Action Injection.

Defines the contract for all swarm decision logic.
Includes the 'ActionStrategy' puppet used for Reinforcement Learning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SensorData:
    """
    Standardized packet of tensors for the Brain.
    Matches the observations derived from SwarmSensors.

    For CTDE (Centralized Training, Decentralized Execution):
    - actor_obs: Local observations for decentralized decision-making
    - critic_obs: Full observations (optional, only used in training)
    """
    actor_obs: torch.Tensor     # (B, N, actor_obs_dim) - local features
    active_mask: torch.Tensor   # (B, N, 1)
    # (B, N, critic_obs_dim) - for training only
    critic_obs: torch.Tensor | None = None
    # (B, N, 1) - current payload level (optional, for classical strategies)
    payload: torch.Tensor | None = None
    # (B, N, 1) - current battery level [0, 1] (optional)
    battery: torch.Tensor | None = None
    # (B, N, 1) - alive/dead mask (optional)
    alive_mask: torch.Tensor | None = None
    # (B, N, 1) - current drone FSM state (optional)
    states: torch.Tensor | None = None
    # (B, N, 1) - raw count of nearby alive neighbors (optional)
    neighbor_count_abs: torch.Tensor | None = None
    # (B, N, 1) - raw count of nearby alive active neighbors (optional)
    neighbor_active_count_abs: torch.Tensor | None = None
    # (B, N, 1) - raw count of nearby firefighting neighbors (optional)
    neighbor_firefighting_count_abs: torch.Tensor | None = None
    # (B, N, 1) - raw count of nearby successful returns (optional)
    return_success_count_abs: torch.Tensor | None = None
    # (B, N, 1) - raw count of nearby unproductive returns (optional)
    return_giveup_count_abs: torch.Tensor | None = None
    # (B, N, N) - alive, in-range neighborhood mask with self excluded (optional)
    neighbor_mask: torch.Tensor | None = None
    # (B, N, 1) - drones currently under simulator-imposed startup override (optional)
    startup_override_mask: torch.Tensor | None = None
    # Actor feature-name to column-index mapping (optional)
    feature_index: dict[str, int] | None = None


class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.cfg = config
        self.device = config['simulation']['device']
        self.name = "base"

    @abstractmethod
    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        """
        Input: Observations + Mask.
        Output: Binary Intent (B, N, 1) -> 0: Base, 1: Field.
        """
        pass

    def reset(self):
        """Optional hook for stateful strategies to reset between episodes."""
        pass

    def observe_step(self, sensor_data: SensorData) -> None:
        """Optional per-step observation hook for stateful classical controllers."""
        del sensor_data

    def get_step_diagnostics(self) -> dict[str, torch.Tensor | float]:
        """Optional per-step diagnostics, ideally one scalar per batch row."""
        return {}

    def reset_debug_summary(self) -> None:
        """Optional hook to clear run-level debug accumulators."""
        pass

    def get_debug_summary(self) -> dict[str, torch.Tensor | float]:
        """Optional run-level diagnostics, ideally one scalar per batch row."""
        return {}


class AlwaysStrategy(BaseStrategy):
    """
    The 'Always Go' Strategy.
    Simple baseline that always commands drones to go to/extend in the field.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "always"

    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        """Returns a tensor of all ones (Go/Extend)."""
        # Return Long or Float, matches the 'intent_mask' usage
        return torch.ones_like(sensor_data.active_mask, dtype=torch.long)


class NeverStrategy(BaseStrategy):
    """
    The 'Never Go' Strategy.
    Simple baseline that always commands drones to return/stay at base.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "never"

    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        """Returns a tensor of all zeros (Return/Stay)."""
        return torch.zeros_like(sensor_data.active_mask, dtype=torch.long)


class RLStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.device = config['simulation']['device']
        self.batch_size = config['simulation']['batch_size']
        self.n_drones = config['swarm']['n_drones']

        # Buffer to hold the latest actions from the Env
        # Default to 0 (Return/Stay) to be safe
        self.current_actions = torch.zeros(
            (self.batch_size, self.n_drones, 1),
            device=self.device
        )

    def set_actions(self, actions: torch.Tensor):
        """Called by the RL Environment before the simulation step."""
        self.current_actions = actions

    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        """
        Called by SwarmManager.
        Simply returns the injected actions.
        """
        # The Manager expects 'intent' (0 or 1)
        return self.current_actions
