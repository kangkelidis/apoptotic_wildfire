"""RL reward engine.

Reward design target:
1. Utility-style objective (save fuel, minimize active-time cost).
2. Dense signal under sparse/random ignitions.
3. Clear carrot/stick around fire response and post-fire stand-down.
"""

from dataclasses import dataclass

import torch


@dataclass
class RewardConfig:
    # Performance proxy: discourage fuel loss.
    fuel_weight: float = 50.0

    # Cost proxy (utility/efficiency denominator): active drones.
    ops_cost: float = 0.0001

    # if fire exists, waiting agents are penalized.
    negligence: float = 0.001

    # reward response while fire exists.
    response_bonus: float = 0.01

    containment_bonus: float = 5.0
    heat_drop_bonus: float = 0.2

    # extra active cost once fire is out.
    post_fire_ops_cost: float = 0.001

    # Add terminal utility ((fuel_preserved_frac - active_exposure_frac) * weight)
    terminal_utility_weight: float = 1.0

    # Decision-aligned shaping (only applied on decision-request steps).
    decision_need_visual_threshold: float = 0.002
    decision_need_memory_threshold: float = 0.0002
    decision_need_trend_success_threshold: float = 0.20
    decision_need_trend_giveup_threshold: float = 0.5
    decision_go_when_need_bonus: float = 0.02
    decision_return_when_need_penalty: float = 0.02
    decision_go_when_no_need_penalty: float = 0.02
    decision_return_when_no_need_bonus: float = 0.01

    # Overall scaling for decision-aligned shaping terms.
    decision_weight: float = 0.05

    # Penalize large positive jumps in active ratio to discourage
    # synchronized "everyone launch now" behavior.
    launch_burst_penalty: float = 0.0
    launch_burst_tolerance: float = 0.01


REWARD_PROFILE_OVERRIDES: dict[str, dict[str, float]] = {
    "default": {},
    "fire_out_first": {
        "fuel_weight": 35.0,
        "ops_cost": 0.00005,
        "negligence": 0.0005,
        "response_bonus": 0.02,
        "containment_bonus": 12.0,
        "heat_drop_bonus": 0.5,
        "post_fire_ops_cost": 0.0003,
        "terminal_utility_weight": 0.0,
        "decision_go_when_need_bonus": 0.03,
        "decision_return_when_need_penalty": 0.03,
        "decision_go_when_no_need_penalty": 0.01,
        "decision_return_when_no_need_bonus": 0.0,
        "decision_weight": 0.08,
        "launch_burst_penalty": 0.0,
    },
    "deploy_balance": {
        "fuel_weight": 50.0,
        "ops_cost": 0.00008,
        "response_bonus": 0.01,
        "containment_bonus": 8.0,
        "heat_drop_bonus": 0.3,
        "post_fire_ops_cost": 0.0008,
        "terminal_utility_weight": 0.5,
        "decision_go_when_need_bonus": 0.02,
        "decision_return_when_need_penalty": 0.02,
        "decision_go_when_no_need_penalty": 0.015,
        "decision_return_when_no_need_bonus": 0.005,
        "decision_weight": 0.06,
        "launch_burst_penalty": 0.25,
    },
    "cost_tune": {
        "fuel_weight": 55.0,
        "ops_cost": 0.00018,
        "response_bonus": 0.003,
        "containment_bonus": 2.5,
        "heat_drop_bonus": 0.08,
        "post_fire_ops_cost": 0.0025,
        "terminal_utility_weight": 1.25,
        "decision_go_when_need_bonus": 0.01,
        "decision_return_when_need_penalty": 0.025,
        "decision_go_when_no_need_penalty": 0.05,
        "decision_return_when_no_need_bonus": 0.015,
        "decision_weight": 0.05,
        "launch_burst_penalty": 0.25,
    },
    "cost_refine": {
        "fuel_weight": 60.0,
        "ops_cost": 0.00024,
        "response_bonus": 0.002,
        "containment_bonus": 2.0,
        "heat_drop_bonus": 0.05,
        "post_fire_ops_cost": 0.0035,
        "terminal_utility_weight": 1.5,
        "decision_go_when_need_bonus": 0.008,
        "decision_return_when_need_penalty": 0.03,
        "decision_go_when_no_need_penalty": 0.06,
        "decision_return_when_no_need_bonus": 0.02,
        "decision_weight": 0.05,
        "launch_burst_penalty": 0.35,
    },
    "cost_hard_tune": {
        "fuel_weight": 15.0,
        "ops_cost": 0.00032,
        "response_bonus": 0.001,
        "containment_bonus": 1.5,
        "heat_drop_bonus": 0.03,
        "post_fire_ops_cost": 0.0050,
        "terminal_utility_weight": 2.0,
        "decision_go_when_need_bonus": 0.006,
        "decision_return_when_need_penalty": 0.035,
        "decision_go_when_no_need_penalty": 0.08,
        "decision_return_when_no_need_bonus": 0.025,
        "decision_weight": 0.05,
        "launch_burst_penalty": 0.50,
    },
}


def available_reward_profiles() -> tuple[str, ...]:
    return tuple(REWARD_PROFILE_OVERRIDES.keys())


def apply_reward_profile(cfg: RewardConfig, profile_name: str | None) -> str:
    name = str(profile_name or "default")
    if name not in REWARD_PROFILE_OVERRIDES:
        raise ValueError(
            f"Unknown reward profile '{name}'. Available: {sorted(REWARD_PROFILE_OVERRIDES)}"
        )
    overrides = dict(REWARD_PROFILE_OVERRIDES[name])
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return name


class RewardEngine:
    FIRE_THRESHOLD = 0.01
    # "Fire out" threshold for *coverage* (fraction of cells above FIRE_THRESHOLD).
    # Keep this very small so post-fire stand-down shaping triggers reliably.
    FIRE_OUT_THRESHOLD = 1e-4

    def __init__(self, config: dict):
        self.device = torch.device(config['simulation']['device'])

        # Use default values from dataclass
        self.cfg = RewardConfig()
        self.profile_name = apply_reward_profile(
            self.cfg,
            config.get("rl", {}).get("reward_profile"),
        )

        self.prev_fuel_pct = None
        self.initial_fuel_pct = None
        self.prev_fire_cov = None
        self.prev_heat_mean = None
        self.prev_active_ratio = None
        self.fire_seen = None
        self.stats = {
            'fuel': 0.0,
            'ops': 0.0,
            'neg': 0.0,
            'response': 0.0,
            'contain': 0.0,
            'heat': 0.0,
            'post': 0.0,
            'decision': 0.0,
            'burst': 0.0,
        }
        self.last_instant = {k: 0.0 for k in self.stats}
        self._episode_active_sum = None
        self._episode_post_fire_active_sum = None
        self._episode_post_fire_steps = None

    def reset(self, engine):
        metrics = engine.get_global_metrics()
        self.prev_fuel_pct = metrics['fuel_pct'].detach().view(-1, 1)
        self.initial_fuel_pct = self.prev_fuel_pct.clone()

        heat = engine.physics.state[:, 0]
        self.prev_fire_cov = (
            (heat > self.FIRE_THRESHOLD).float().mean(dim=(1, 2)).view(-1, 1)
        )
        self.prev_heat_mean = heat.mean(dim=(1, 2), keepdim=False).view(-1, 1)
        self.prev_active_ratio = metrics['active_mask'].float().mean(
            dim=1, keepdim=True
        )
        self.fire_seen = torch.zeros_like(self.prev_fire_cov, dtype=torch.bool)
        self._episode_active_sum = torch.zeros_like(self.prev_fuel_pct)
        self._episode_post_fire_active_sum = torch.zeros_like(
            self.prev_fuel_pct)
        self._episode_post_fire_steps = torch.zeros_like(self.prev_fuel_pct)

        # Reset stats
        self.stats = {k: 0.0 for k in self.stats}
        self.last_instant = {k: 0.0 for k in self.last_instant}
        self.last_instant['terminal'] = 0.0
        self.stats['terminal'] = 0.0
        self.step_count = 0

    def compute(
        self,
        engine,
        *,
        done: bool = False,
        actor_obs_step: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        decision_mask_step: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the step reward.
        Returns: (B, N, 1)
        """
        # 1. Get metrics/state
        metrics = engine.get_global_metrics()
        current_fuel = metrics['fuel_pct'].view(-1, 1)
        heat = engine.physics.state[:, 0]
        fire_cov = (heat > self.FIRE_THRESHOLD).float().mean(
            dim=(1, 2)).view(-1, 1)
        heat_mean = heat.mean(dim=(1, 2), keepdim=False).view(-1, 1)
        if self.fire_seen is None:
            self.fire_seen = torch.zeros_like(fire_cov, dtype=torch.bool)
        self.fire_seen = self.fire_seen | (fire_cov > self.FIRE_OUT_THRESHOLD)

        if self.prev_fuel_pct is None:
            self.prev_fuel_pct = current_fuel
        if self.prev_fire_cov is None:
            self.prev_fire_cov = fire_cov
        if self.prev_heat_mean is None:
            self.prev_heat_mean = heat_mean
        if self.prev_active_ratio is None:
            self.prev_active_ratio = metrics['active_mask'].float().mean(
                dim=1, keepdim=True
            )

        # A) Utility-style performance: penalize fuel loss (delta <= 0)
        delta_fuel = torch.clamp(current_fuel - self.prev_fuel_pct, max=0.0)
        r_fuel = delta_fuel * self.cfg.fuel_weight  # (B, 1)

        # B) Dense containment shaping: reward only improvements.
        fire_drop = torch.clamp(self.prev_fire_cov - fire_cov, min=0.0)
        heat_drop = torch.clamp(self.prev_heat_mean - heat_mean, min=0.0)
        r_contain = fire_drop * self.cfg.containment_bonus  # (B, 1)
        r_heat = heat_drop * self.cfg.heat_drop_bonus       # (B, 1)

        B = current_fuel.shape[0]
        N = engine.swarm.states.shape[1]

        alive = engine.swarm.alive_mask.squeeze(-1).bool()
        alive_float = alive.float().unsqueeze(-1)
        waiting = (
            (engine.swarm.states.squeeze(-1) ==
             engine.swarm.controller.STATE_WAITING) & alive
        )
        active = metrics['active_mask'].squeeze(-1).bool() & alive
        active_capacity_frac = (
            metrics['active_drones'].view(-1, 1) /
            float(max(int(engine.n_drones), 1))
        )
        active_ratio = active.float().mean(dim=1, keepdim=True).unsqueeze(-1)

        fire_gate = fire_cov.view(B, 1, 1)
        post_fire_gate = (
            self.fire_seen & (fire_cov <= self.FIRE_OUT_THRESHOLD)
        ).float().view(B, 1, 1)

        # C) Cost side: active-time penalty + extra post-fire active penalty.
        r_ops = -self.cfg.ops_cost * active.float().unsqueeze(-1)
        r_post = -self.cfg.post_fire_ops_cost * \
            active.float().unsqueeze(-1) * post_fire_gate
        burst_delta = torch.clamp(
            active_ratio - self.prev_active_ratio - self.cfg.launch_burst_tolerance,
            min=0.0,
        )
        burst_gate = (fire_cov > self.FIRE_OUT_THRESHOLD).float().view(B, 1, 1)
        r_burst = -self.cfg.launch_burst_penalty * burst_delta * burst_gate

        # D) Response shaping around active fires.
        r_negligence = -self.cfg.negligence * \
            waiting.float().unsqueeze(-1) * fire_gate
        r_response = self.cfg.response_bonus * \
            active.float().unsqueeze(-1) * fire_gate

        # E) Decision-aligned shaping (computed from actor-observable features).
        r_decision = torch.zeros((B, N, 1), device=self.device)
        obs = actor_obs_step if actor_obs_step is not None else engine.swarm.latest_actor_obs
        if (
            obs is not None and
            actions is not None and
            decision_mask_step is not None and
            (
                self.cfg.decision_go_when_need_bonus != 0.0 or
                self.cfg.decision_return_when_need_penalty != 0.0 or
                self.cfg.decision_go_when_no_need_penalty != 0.0 or
                self.cfg.decision_return_when_no_need_bonus != 0.0
            )
        ):
            idx = engine.swarm.perception.actor_feature_idx
            # Raw (un-gained) heat signals.
            raw_vis = obs[..., idx["visual_heat"]:idx["visual_heat"] + 1] / \
                engine.swarm.perception.ACTOR_VISUAL_HEAT_GAIN
            raw_mem = obs[..., idx["memory_heat"]:idx["memory_heat"] + 1] / \
                engine.swarm.perception.ACTOR_MEMORY_HEAT_GAIN
            trend_success = obs[..., idx["trend_success"]                                :idx["trend_success"] + 1]
            trend_giveup = obs[..., idx["trend_giveup"]                               :idx["trend_giveup"] + 1]

            need = (
                (raw_vis > self.cfg.decision_need_visual_threshold) |
                (raw_mem > self.cfg.decision_need_memory_threshold) |
                (trend_success > self.cfg.decision_need_trend_success_threshold)
            )
            # If neighbors are mostly giving up, dampen the global "need" signal.
            need = need & ~(trend_giveup >
                            self.cfg.decision_need_trend_giveup_threshold)

            eligible = decision_mask_step.bool() & alive_float.bool()
            is_go = (actions == 1) & eligible
            is_ret = (actions == 0) & eligible

            if self.cfg.decision_go_when_need_bonus != 0.0:
                r_decision = r_decision + \
                    self.cfg.decision_go_when_need_bonus * \
                    (need & is_go).float()
            if self.cfg.decision_return_when_need_penalty != 0.0:
                r_decision = r_decision - \
                    self.cfg.decision_return_when_need_penalty * \
                    (need & is_ret).float()
            if self.cfg.decision_go_when_no_need_penalty != 0.0:
                r_decision = r_decision - \
                    self.cfg.decision_go_when_no_need_penalty * \
                    ((~need) & is_go).float()
            if self.cfg.decision_return_when_no_need_bonus != 0.0:
                r_decision = r_decision + \
                    self.cfg.decision_return_when_no_need_bonus * \
                    ((~need) & is_ret).float()

            r_decision = r_decision * alive_float * self.cfg.decision_weight

        # F) Aggregate (global terms are broadcast to alive agents only).
        global_terms = (
            r_fuel + r_contain + r_heat
        ).view(B, 1, 1).expand(B, N, 1) * alive_float
        total = (
            global_terms +
            r_negligence +
            r_response +
            r_ops +
            r_post +
            r_burst.view(B, 1, 1).expand(B, N, 1) * alive_float +
            r_decision
        )

        # Track episode-level cost terms used by run utility metric.
        if self._episode_active_sum is None:
            self._episode_active_sum = torch.zeros_like(current_fuel)
        if self._episode_post_fire_active_sum is None:
            self._episode_post_fire_active_sum = torch.zeros_like(current_fuel)
        if self._episode_post_fire_steps is None:
            self._episode_post_fire_steps = torch.zeros_like(current_fuel)
        self._episode_active_sum += active_capacity_frac
        post_fire_active_step = active_capacity_frac * \
            post_fire_gate.view(B, 1)
        self._episode_post_fire_active_sum += post_fire_active_step
        self._episode_post_fire_steps += post_fire_gate.view(B, 1)

        r_terminal = torch.zeros_like(current_fuel)
        if done and self.cfg.terminal_utility_weight != 0.0:
            steps = float(max(self.step_count + 1, 1))
            active_exposure = self._episode_active_sum / steps
            fuel_preserved = current_fuel / \
                self.initial_fuel_pct.clamp(min=1e-6)
            utility = fuel_preserved - active_exposure
            r_terminal = utility * self.cfg.terminal_utility_weight
            total += r_terminal.view(B, 1, 1).expand(B, N, 1) * alive_float

        # Update history for next step.
        self.prev_fuel_pct = current_fuel.detach()
        self.prev_fire_cov = fire_cov.detach()
        self.prev_heat_mean = heat_mean.detach()
        self.prev_active_ratio = active_ratio.detach()

        # Logging (accumulate and track step count).
        self.step_count += 1

        # Store instant values as mean over (B, N, 1) so they sum to total.mean().
        # Global (B,1) terms must be broadcast to (B,N,1) with the alive mask
        # to match how they enter the total.
        def _agent_mean(x_bn1: torch.Tensor) -> float:
            return x_bn1.mean().item()

        def _global_agent_mean(x_b1: torch.Tensor) -> float:
            return (x_b1.view(B, 1, 1).expand(B, N, 1) * alive_float).mean().item()

        self.last_instant['fuel'] = _global_agent_mean(r_fuel)
        self.last_instant['contain'] = _global_agent_mean(r_contain)
        self.last_instant['heat'] = _global_agent_mean(r_heat)
        self.last_instant['neg'] = _agent_mean(r_negligence)
        self.last_instant['response'] = _agent_mean(r_response)
        self.last_instant['ops'] = _agent_mean(r_ops)
        self.last_instant['post'] = _agent_mean(r_post)
        self.last_instant['burst'] = _global_agent_mean(r_burst)
        self.last_instant['decision'] = _agent_mean(r_decision)
        self.last_instant['terminal'] = _global_agent_mean(
            r_terminal) if done else 0.0

        # Accumulate for episode stats
        for key in self.stats:
            self.stats[key] += self.last_instant[key]

        return total

    def get_stats_summary(self) -> str:
        """Returns reward breakdown (per-step averages)."""
        if self.step_count == 0:
            return "Rewards: No steps yet"
        avg_stats = {k: v / self.step_count for k, v in self.stats.items()}
        total = sum(avg_stats.values())
        parts = " ".join(f"{k}={avg_stats[k]:+.4f}" for k in sorted(avg_stats))
        return f"Rewards/step: Total={total:+.4f} [{parts}]"

    def get_instant_breakdown(self) -> str:
        """Get instant reward breakdown for the current step."""
        total = sum(self.last_instant.values())
        parts = " ".join(
            f"{k}={self.last_instant[k]:+.4f}" for k in sorted(self.last_instant)
        )
        return f"Instant: Total={total:+.4f} [{parts}]"
