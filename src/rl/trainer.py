"""
RL: PPO Trainer.

Implements PPO with support for Asynchronous Multi-Agent Environments.
Key Features:
- Masked Updates: Only trains on steps where agents actually made decisions.
- GAE: Computes advantages over the full time horizon.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.rl.policy import SwarmPolicy


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    rollout_steps: int = 256
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5


class RolloutBuffer:
    """
    Stores transitions (Obs, Action, Reward, etc.) for a full episode segment.
    Supports CTDE with separate actor and critic observation buffers.
    """

    def __init__(
        self,
        config: PPOConfig,
        batch_size: int,
        n_agents: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        device: torch.device
    ):
        self.T = config.rollout_steps
        self.B = batch_size
        self.N = n_agents
        self.device = device
        self.ptr = 0

        # Pre-allocate Buffers [Time, Batch, Agents, Dim]
        # CTDE: Separate buffers for actor (local) and critic (full) observations
        self.actor_obs = torch.zeros(
            (self.T, self.B, self.N, actor_obs_dim), device=device)
        self.critic_obs = torch.zeros(
            (self.T, self.B, self.N, critic_obs_dim), device=device)

        self.actions = torch.zeros((self.T, self.B, self.N, 1), device=device)
        self.log_probs = torch.zeros(
            (self.T, self.B, self.N, 1), device=device)
        self.rewards = torch.zeros((self.T, self.B, self.N, 1), device=device)
        self.values = torch.zeros((self.T, self.B, self.N, 1), device=device)
        self.dones = torch.zeros(
            (self.T, self.B, self.N, 1), dtype=torch.bool, device=device)

        # Action Masks (Constraints: Can I deploy?)
        self.action_masks = torch.zeros(
            (self.T, self.B, self.N, 2), dtype=torch.bool, device=device)

        # Decision Masks (Timing: Did I act?)
        self.decision_masks = torch.zeros(
            (self.T, self.B, self.N, 1), dtype=torch.bool, device=device)

    def reset(self):
        self.ptr = 0

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        action_mask: torch.Tensor,
        decision_mask: torch.Tensor
    ):
        """Record a step with CTDE observations."""
        if self.ptr >= self.T:
            return  # Buffer full

        self.actor_obs[self.ptr] = actor_obs
        self.critic_obs[self.ptr] = critic_obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask
        self.decision_masks[self.ptr] = decision_mask

        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE).
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(last_value)

        # We iterate backwards
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            not_done = (~self.dones[t]).float()

            delta = self.rewards[t] + gamma * \
                next_value * not_done - self.values[t]
            advantages[t] = last_gae = delta + \
                gamma * gae_lambda * not_done * last_gae

        returns = advantages + self.values
        return returns, advantages


class PPOTrainer:
    """
    PPO Training Loop.
    """

    def __init__(self, policy: SwarmPolicy, config: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> Dict[str, float]:
        """
        Updates the policy network using collected experience (CTDE).
        """
        returns, advantages = buffer.compute_gae(
            last_value, self.cfg.gamma, self.cfg.gae_lambda
        )

        # We merge [Time, Batch, Agents] -> [Samples]
        # This creates a massive batch of (T * B * N) samples
        b_actor_obs = buffer.actor_obs[:buffer.ptr].reshape(
            -1, buffer.actor_obs.shape[-1])
        b_critic_obs = buffer.critic_obs[:buffer.ptr].reshape(
            -1, buffer.critic_obs.shape[-1])
        b_actions = buffer.actions[:buffer.ptr].reshape(-1, 1)
        b_log_probs = buffer.log_probs[:buffer.ptr].reshape(-1, 1)
        b_returns = returns[:buffer.ptr].reshape(-1, 1)
        b_advantages = advantages[:buffer.ptr].reshape(-1, 1)
        b_action_masks = buffer.action_masks[:buffer.ptr].reshape(-1, 2)
        b_values = buffer.values[:buffer.ptr].reshape(-1, 1)
        b_decision_masks = buffer.decision_masks[:buffer.ptr].reshape(-1, 1)
        total_rollout_samples = int(b_actor_obs.shape[0])

        # Most rollout steps are idle (drones waiting); train only on the steps
        # where a decision was actually made.
        # b_decision_masks is shaped (Samples, 1). We need row indices only.
        # MPS can produce unstable index tensors for very large `nonzero`/`randperm`
        # workloads; run index generation on CPU and only move batch indices back.
        if self.device.type == "mps":
            valid_indices = torch.nonzero(
                b_decision_masks.squeeze(-1).detach().cpu(), as_tuple=True
            )[0].to(self.device)
        else:
            valid_indices = torch.nonzero(
                b_decision_masks.squeeze(-1), as_tuple=True
            )[0]

        # Safety: If NO decisions happened in this entire rollout (rare but possible), skip
        if valid_indices.numel() == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'decision_sample_count': 0.0,
                'total_rollout_samples': float(total_rollout_samples),
                'decision_sample_fraction': 0.0,
                'minibatches_per_epoch': 0.0,
                'optimizer_steps': 0.0,
            }

        # Select only valid data
        f_actor_obs = b_actor_obs[valid_indices]
        f_critic_obs = b_critic_obs[valid_indices]
        f_actions = b_actions[valid_indices]
        f_log_probs = b_log_probs[valid_indices]
        f_returns = b_returns[valid_indices]
        f_advantages = b_advantages[valid_indices]
        f_masks = b_action_masks[valid_indices]
        f_values = b_values[valid_indices]

        # Normalize Advantages on the filtered set
        f_advantages = (f_advantages - f_advantages.mean()) / \
            (f_advantages.std() + 1e-8)

        # Stats accumulators
        stats = {'policy_loss': 0.0, 'value_loss': 0.0,
                 'entropy': 0.0, 'approx_kl': 0.0}
        n_updates = 0
        total_samples = f_actor_obs.shape[0]
        minibatches_per_epoch = int(
            math.ceil(total_samples / max(int(self.cfg.mini_batch_size), 1))
        )

        # PPO Epochs
        for _ in range(self.cfg.ppo_epochs):
            if self.device.type == "mps":
                indices = torch.randperm(
                    total_samples, device=torch.device("cpu"))
            else:
                indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = indices[start:end]
                if mb_idx.device != self.device:
                    mb_idx = mb_idx.to(self.device)

                # Mini-batch data (CTDE: separate actor and critic obs)
                mb_actor_obs = f_actor_obs[mb_idx]
                mb_critic_obs = f_critic_obs[mb_idx]
                mb_act = f_actions[mb_idx]
                mb_old_log = f_log_probs[mb_idx]
                mb_ret = f_returns[mb_idx]
                mb_adv = f_advantages[mb_idx]
                mb_mask_act = f_masks[mb_idx]
                mb_old_val = f_values[mb_idx]

                # Forward Pass (CTDE: actor_obs for policy, critic_obs for value)
                new_log_probs, new_values, entropy = self.policy.evaluate(
                    mb_actor_obs, mb_critic_obs, mb_act, mb_mask_act
                )

                # Policy Loss
                ratio = torch.exp(new_log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                v_clipped = mb_old_val + \
                    torch.clamp(new_values - mb_old_val, -
                                self.cfg.value_clip, self.cfg.value_clip)
                v_loss1 = (new_values - mb_ret).pow(2)
                v_loss2 = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Entropy Loss
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + self.cfg.value_coeff * \
                    value_loss + self.cfg.entropy_coeff * entropy_loss

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Log
                with torch.no_grad():
                    approx_kl = (mb_old_log - new_log_probs).mean().item()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                # Log raw entropy (positive)
                stats['entropy'] += entropy.mean().item()
                stats['approx_kl'] += approx_kl
                n_updates += 1

        # Average Stats
        for k in stats:
            stats[k] /= max(n_updates, 1)

        stats['decision_sample_count'] = float(total_samples)
        stats['total_rollout_samples'] = float(total_rollout_samples)
        stats['decision_sample_fraction'] = float(
            total_samples / max(total_rollout_samples, 1)
        )
        stats['minibatches_per_epoch'] = float(minibatches_per_epoch)
        stats['optimizer_steps'] = float(n_updates)

        return stats
