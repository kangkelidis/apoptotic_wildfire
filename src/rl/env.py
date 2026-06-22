"""
RL: Wildfire Environment Wrapper with CTDE Support.
Returns separate actor/critic observations for Centralized Training, Decentralized Execution.
"""

from typing import cast

import torch

from src.core.engine import create_engine
from src.rl.rewards import RewardEngine
from src.strategies.base import RLStrategy
from src.utils.hardware import generate_seeds


class WildfireEnv:
    def __init__(
        self,
        config: dict,
        scenario_name: str = 'baseline',
        seed_mode: str = 'static',
        seed_repeat: int = 1,
        seed_pool_size: int = 64
    ):
        self.cfg = config
        self.scenario_name = str(scenario_name)
        self.device = config['simulation']['device']
        self.batch_size = config['simulation']['batch_size']
        self.n_drones = config['swarm']['n_drones']
        self.base_seed = int(config['simulation']['seed'])

        self.seed_mode = str(seed_mode).lower()
        if self.seed_mode not in ('static', 'cycle'):
            raise ValueError(
                f"Invalid seed_mode='{self.seed_mode}'. Use 'static' or 'cycle'."
            )

        # Each seed is repeated for this many episodes before moving to the next.
        self.seed_repeat = max(1, int(seed_repeat))
        self.seed_pool_size = max(1, int(seed_pool_size))
        self.seed_pool = self._build_seed_pool()
        self.episode_idx = 0
        self.current_episode_seed = self.seed_pool[0]

        self.engine = create_engine(
            config,
            strategy_name='rl_training',
            scenario_name=self.scenario_name
        )

        self.strategy_proxy = cast(RLStrategy, self.engine.swarm.strategy)

        self.reward_engine = RewardEngine(config)
        self.step_count = 0

        # CTDE observation dimensions from swarm manager
        self.actor_obs_dim = self.engine.swarm.actor_obs_dim
        self.critic_obs_dim = self.engine.swarm.critic_obs_dim

    def _build_seed_pool(self) -> list[int]:
        """Build deterministic episode seed pool (first seed is always base_seed)."""
        if self.seed_mode == 'static':
            return [self.base_seed]

        pool = generate_seeds(self.base_seed, self.seed_pool_size)

        # Keep the very first episodes anchored on the configured base seed.
        if self.base_seed in pool:
            idx = pool.index(self.base_seed)
            pool[0], pool[idx] = pool[idx], pool[0]
        else:
            pool[0] = self.base_seed

        return pool

    def _next_episode_seed(self) -> int:
        if self.seed_mode == 'static':
            return self.base_seed

        pool_idx = (self.episode_idx // self.seed_repeat) % len(self.seed_pool)
        return self.seed_pool[pool_idx]

    def reset(self) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Resets the simulation and calculates the initial observation.

        Returns:
            Tuple of ((actor_obs, critic_obs), decision_mask):
                actor_obs: (B, N, actor_obs_dim) for decentralized execution
                critic_obs: (B, N, critic_obs_dim) for centralized training
                decision_mask: (B, N, 1) boolean mask of agents needing decisions
        """
        self.step_count = 0

        episode_seed = self._next_episode_seed()
        self.current_episode_seed = episode_seed
        self.episode_idx += 1

        self.engine.reset(episode_seed)
        self.reward_engine.reset(self.engine)

        actor_obs, critic_obs = self.engine.swarm.get_perception(
            self.engine.physics)

        decision_mask = self.engine.swarm.decision_requests

        return (actor_obs, critic_obs), decision_mask

    def step(self, actions: torch.Tensor):
        """
        1. Inject Action -> 2. Step Physics -> 3. Calculate Reward

        Returns:
            Tuple of ((actor_obs, critic_obs), reward, decision_mask, done, info)

            done: Boolean flag indicating episode termination (time limit reached)
                  This decouples the learning cycle from the episode cycle,
                  allowing the trainer to continue rollouts across episode boundaries
                  if desired, or to reset when episodes naturally end.
        """
        sim_step_idx = int(self.step_count)

        # This is the mask that corresponds to the action being applied now.
        decision_mask_step = self.engine.swarm.decision_requests

        # Use the exact observation that produced this action to avoid
        # actor-observation lag inside the transition.
        actor_obs_for_step = self.engine.swarm.latest_actor_obs
        if actor_obs_for_step is None:
            actor_obs_for_step, _ = self.engine.swarm.get_perception(
                self.engine.physics
            )

        # We tell the strategy what to do for this upcoming step
        self.strategy_proxy.set_actions(actions)

        self.engine.step(sim_step_idx, actor_obs_override=actor_obs_for_step)
        self.step_count += 1

        actor_obs, critic_obs = self.engine.swarm.get_perception(
            self.engine.physics
        )

        done = (self.step_count >= self.engine.max_steps)
        reward = self.reward_engine.compute(
            self.engine,
            done=done,
            actor_obs_step=actor_obs_for_step,
            actions=actions,
            decision_mask_step=decision_mask_step,
        )

        # This mask is for the subsequent action selection call.
        decision_mask = self.engine.swarm.decision_requests

        info = {
            'step_count': self.step_count,
            'max_steps': self.engine.max_steps,
            'episode_complete': done,
            'episode_seed': self.current_episode_seed,
            'decision_mask_step': decision_mask_step,
        }

        return (actor_obs, critic_obs), reward, decision_mask, done, info
