"""
Strategies: MAPPO (Inference).

Loads a trained PPO policy to control the swarm.
Uses CTDE (Centralized Training, Decentralized Execution) -
only actor_obs (local features) are needed for inference.
"""

from pathlib import Path

import torch

from ..rl.policy import SwarmPolicy
from .base import BaseStrategy, SensorData

# Default MAPPO inference checkpoint shipped with the release.
# Override per run with `--model <path>` (see main.py) or by setting
# runtime.mappo_model_path in the config.
DEFAULT_MAPPO_CHECKPOINT = "artifacts/checkpoints/mappo_final.pt"


class MAPPOStrategy(BaseStrategy):
    DEFAULT_ACTOR_OBS_DIM = 12
    DEFAULT_CRITIC_OBS_DIM = 18

    def __init__(self, config: dict):
        super().__init__(config)
        self.device = config['simulation']['device']

        # Allow CLI/runtime override from main.py (--model).
        runtime = config.get("runtime", {})
        configured_model = runtime.get("mappo_model_path")
        if configured_model:
            self.model_path = str(configured_model)
        else:
            self.model_path = DEFAULT_MAPPO_CHECKPOINT

        self.actor_obs_dim = self.DEFAULT_ACTOR_OBS_DIM
        self.critic_obs_dim = self.DEFAULT_CRITIC_OBS_DIM
        self.policy = self._build_policy(
            self.actor_obs_dim, self.critic_obs_dim)

        # Load Weights
        self.load_model()
        self.policy.eval()

    def _build_policy(self, actor_obs_dim: int, critic_obs_dim: int) -> SwarmPolicy:
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        return SwarmPolicy(
            config={'rl': {
                'actor_obs_dim': self.actor_obs_dim,
                'critic_obs_dim': self.critic_obs_dim
            }}).to(self.device)

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict) and 'policy' in ckpt:
            return ckpt['policy'], {
                "steps": ckpt.get('steps'),
                "update": ckpt.get('update'),
            }
        raise ValueError(
            "MAPPO checkpoint must be a current trainer checkpoint with a "
            "'policy' state dict. Regenerate it with train.py or use "
            f"{DEFAULT_MAPPO_CHECKPOINT}."
        )

    @staticmethod
    def _format_checkpoint_meta(meta: dict) -> str:
        steps = meta.get("steps")
        update = meta.get("update")
        bits = []
        if isinstance(steps, int):
            bits.append(f"steps={steps:,}")
        else:
            bits.append("steps=?")
        if isinstance(update, int):
            bits.append(f"update={update}")
        return ", ".join(bits)

    @staticmethod
    def _infer_obs_dims(state_dict: dict):
        actor_dim = None
        critic_dim = None
        actor_w = state_dict.get('actor.0.weight')
        critic_w = state_dict.get('critic.0.weight')
        if isinstance(actor_w, torch.Tensor):
            actor_dim = int(actor_w.shape[1])
        if isinstance(critic_w, torch.Tensor):
            critic_dim = int(critic_w.shape[1])
        return actor_dim, critic_dim

    def _validate_obs_dims(self, state_dict: dict) -> None:
        actor_dim, critic_dim = self._infer_obs_dims(state_dict)
        expected = (self.actor_obs_dim, self.critic_obs_dim)
        actual = (actor_dim, critic_dim)
        if actual != expected:
            raise ValueError(
                "MAPPO checkpoint observation dimensions do not match the "
                f"current model: actor/critic={actual}, expected={expected}."
            )

    def load_model(self):
        path = Path(self.model_path)
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            state_dict, meta = self._extract_state_dict(ckpt)
            meta_text = self._format_checkpoint_meta(meta)
            self._validate_obs_dims(state_dict)

            self.policy.load_state_dict(state_dict)
            print(
                "MAPPO Strategy: Loaded Agent "
                f"from {path} ({meta_text})"
            )
        else:
            raise FileNotFoundError(
                f"MAPPO checkpoint not found at {path}. "
                "Train a model with train.py, pass --model <path>, or set "
                "runtime.mappo_model_path. The released checkpoint lives at "
                f"{DEFAULT_MAPPO_CHECKPOINT}."
            )

    def decide(self, sensor_data: SensorData) -> torch.Tensor:
        """
        Executes the policy to determine intents (Decentralized Execution).

        Args:
            sensor_data: Contains actor_obs (B, N, actor_obs_dim) for local decisions.

        Returns:
            intent: (B, N, 1) Tensor where 1=Launch/Extend, 0=Stay/Return.
        """
        # CTDE: Use actor_obs for decentralized execution
        # We use 'deterministic=True' for evaluation/viz to get the best behavior.

        actor_obs = sensor_data.actor_obs
        if actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(
                "MAPPO actor observation dimension mismatch: "
                f"got {actor_obs.shape[-1]}, expected {self.actor_obs_dim}."
            )

        with torch.no_grad():
            actions, _, _ = self.policy.get_action(
                actor_obs, deterministic=True)

        return actions
