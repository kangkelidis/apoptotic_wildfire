"""
PPO Policy Network (Actor-Critic) with CTDE Architecture.

Implements Centralized Training, Decentralized Execution for Homogeneous Swarms.
- Actor: Decentralized. Uses only local observations to decide.
- Critic: Centralized. Uses full observations (local + global/social info)
  to estimate the value of the state.

Architecture:
    Actor: Input (actor_obs_dim from sensors actor feature set) -> [MLP] -> Logits (2)
           [Stay/Return vs Go/Extend]
    Critic: Input (critic_obs_dim from sensors critic feature set) -> [MLP] -> Value (1)
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class SwarmPolicy(nn.Module):
    """
    CTDE Actor-Critic network for the Wildfire Swarm.

    Attributes:
        actor (nn.Sequential): The policy network (decentralized).
        critic (nn.Sequential): The value network (centralized).
    """

    HIDDEN_DIM = 256

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing 'rl' parameters.
        """
        super().__init__()

        # CTDE observation dimensions
        # Actor: local features only
        # Critic: local + centralized swarm context features
        actor_obs_dim = config.get('rl', {}).get('actor_obs_dim', 12)
        critic_obs_dim = config.get('rl', {}).get('critic_obs_dim', 18)

        # ---------------------------------------------------------
        # 1. ACTOR (Decentralized)
        # ---------------------------------------------------------
        # Maps Local Observation -> Action Logits (Binary: 0 or 1)
        self.actor = nn.Sequential(
            nn.Linear(actor_obs_dim, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, 2)
        )

        # ---------------------------------------------------------
        # 2. CRITIC (Centralized)
        # ---------------------------------------------------------
        # Uses full observations including global/social information
        self.critic = nn.Sequential(
            nn.Linear(critic_obs_dim, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, 1)  # Outputs Value Scalar V(s)
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable PPO training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.414)  # Gain for Tanh
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Specific scaling for Actor output (Logits)
        actor_last = self.actor[-1]
        if isinstance(actor_last, nn.Linear):
            nn.init.orthogonal_(actor_last.weight, gain=0.01)

        # Specific scaling for Critic output (Value)
        critic_last = self.critic[-1]
        if isinstance(critic_last, nn.Linear):
            nn.init.orthogonal_(critic_last.weight, gain=1.0)

    @staticmethod
    def _categorical_from_logits(logits: torch.Tensor) -> Categorical:
        """
        Build a categorical distribution from logits.

        NOTE: On MPS, `validate_args=True` can raise internal out-of-bounds
        errors on large training tensors. PPO already controls legality through
        action masking and data flow, so we disable distribution validation here.
        """
        return Categorical(logits=logits, validate_args=False)

    @staticmethod
    def _sample_categorical(dist: Categorical) -> torch.Tensor:
        """Sample from a Categorical distribution, working around MPS bugs.

        ``torch.multinomial`` and softmax on MPS can produce out-of-bounds
        indices for large tensors. We move the raw logits to CPU, build a
        fresh distribution there, sample, and move the indices back.
        """
        logits = dist.logits
        device = logits.device
        if device.type == 'mps':
            shape = logits.shape[:-1]          # (B, N) or (Samples,)
            cpu_logits = logits.view(-1, logits.shape[-1]).cpu()
            cpu_dist = Categorical(logits=cpu_logits, validate_args=False)
            samples = cpu_dist.sample().to(device).view(shape)
            return samples
        return dist.sample()

    def get_value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """
        Estimates state value V(s) using full (centralized) observations.

        Args:
            critic_obs: Either (B, N, critic_obs_dim) structured or (Samples, critic_obs_dim) flattened.

        Returns:
            values: (B, N, 1) or (Samples, 1) depending on input shape.
        """
        if critic_obs.dim() == 2:
            # Flattened case: (Samples, D) -> (Samples, 1)
            return self.critic(critic_obs)
        else:
            # Structured case: (B, N, D) -> (B, N, 1)
            return self.critic(critic_obs)

    def get_action(
        self,
        actor_obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action using only local (actor) observations.

        Args:
            actor_obs: Local observations (B, N, actor_obs_dim).
            action_mask: (Optional) Boolean mask (B, N, 2) where False = Illegal.
            deterministic: If True, returns argmax (for evaluation).

        Returns:
            action: (B, N, 1) Selected action index.
            log_prob: (B, N, 1) Log probability of that action.
            entropy: (B, N, 1) Entropy of the distribution (exploration metric).
        """
        logits = self.actor(actor_obs)

        # Apply Action Masking
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        dist = self._categorical_from_logits(logits)

        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = self._sample_categorical(dist)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions.unsqueeze(-1), log_probs.unsqueeze(-1), entropy.unsqueeze(-1)

    def evaluate(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of actions for PPO Loss calculation (CTDE).

        Args:
            actor_obs: Flattened actor observations (Total_Samples, actor_obs_dim).
            critic_obs: Flattened critic observations (Total_Samples, critic_obs_dim).
            actions: Flattened actions taken (Total_Samples, 1).
            action_mask: Flattened mask (Total_Samples, 2).

        Returns:
            log_probs: (Total_Samples, 1) Log probabilities of taken actions.
            values: (Total_Samples, 1) State values from critic.
            entropy: (Total_Samples, 1) Policy entropy.
        """
        logits = self.actor(actor_obs)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        dist = self._categorical_from_logits(logits)

        log_probs = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()

        values = self.get_value(critic_obs)

        return log_probs.unsqueeze(-1), values, entropy.unsqueeze(-1)
