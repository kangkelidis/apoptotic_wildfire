"""
Swarm Manager: Handles the Drone Swarm.

Manages the persistent states tensors for all agents across all batches.
Acts as a Mediator between Sensors, states Machine, and Navigation
"""

from typing import TYPE_CHECKING

import torch

from src.strategies.base import SensorData
from src.strategies.factory import StrategyFactory
from src.swarm.constants import DroneState
from src.swarm.controller import StateController
from src.swarm.navigation import NavigationEngine
from src.swarm.sensors import PerceptionEngine
from src.utils.hardware import SeedManager

if TYPE_CHECKING:
    from src.physics.manager import PhysicsManager


class SwarmManager:

    # Threshold for deciding when to drop water (0-1, relative to heat sensor reading)
    DROP_THRESHOLD = 0.1
    # Intensity of the drop (0-1)
    DROP_INTENSITY = 1.0
    # Fraction of launch-eligible drones that start directly in-field.
    DEFAULT_INITIAL_IN_AIR_FRACTION = 0.20
    # Must stay aligned with NavigationEngine's base proximity threshold.
    AT_BASE_THRESHOLD = 0.05

    def __init__(self, config: dict, strategy_name: str):
        """
        """
        self.config = config
        self.n_drones = config['swarm']['n_drones']
        self.batch_size = config['simulation']['batch_size']
        self.device = config['simulation']['device']

        self.navigation = NavigationEngine(config)
        self.controller = StateController(config)
        self.perception = PerceptionEngine(config)
        self.strategy = StrategyFactory.create(strategy_name, config)
        self.initial_in_air_fraction = float(
            config.get('swarm', {}).get(
                'initial_in_air_fraction',
                self.DEFAULT_INITIAL_IN_AIR_FRACTION,
            )
        )
        self.initial_force_go_decisions = int(
            config.get('swarm', {}).get('initial_force_go_decisions', 2)
        )
        queue_cfg = config.get('swarm', {}).get('base_service_queue', {})
        self.base_service_queue_enabled = bool(queue_cfg.get('enabled', False))
        self.service_slots_per_base = int(
            queue_cfg.get('service_slots_per_base', 12)
        )
        congestion_cfg = config.get('swarm', {}).get('congestion_effects', {})
        self.congestion_effects_enabled = bool(
            congestion_cfg.get('enabled', False)
        )
        self.congestion_radius = float(congestion_cfg.get('radius', 0.08))
        self.congestion_threshold = int(congestion_cfg.get('threshold', 10))
        self.full_congestion_count = int(
            congestion_cfg.get('full_congestion_count', 24)
        )
        attrition_cfg = config.get('swarm', {}).get('attrition', {})
        self.attrition_enabled = bool(attrition_cfg.get('enabled', False))
        self.attrition_battery_threshold = float(
            attrition_cfg.get('battery_threshold', 0.15)
        )
        self.attrition_fire_threshold = float(
            attrition_cfg.get('fire_threshold', 0.10)
        )
        self.attrition_safe_zone_radius = float(
            attrition_cfg.get('safe_zone_radius', 0.10)
        )
        self.attrition_crowding_radius = float(
            attrition_cfg.get('crowding_radius', 0.03)
        )
        self.attrition_crowding_threshold = int(
            attrition_cfg.get('crowding_threshold', 2)
        )
        self.attrition_fire_weight = float(attrition_cfg.get('fire_weight', 0.5))
        self.attrition_crowding_weight = float(
            attrition_cfg.get('crowding_weight', 0.5)
        )
        self.attrition_max_death_prob = float(
            attrition_cfg.get('max_death_prob', 0.20)
        )

        # --- 1. KINEMATICS ---
        # B: Batch Size, N: Number of Drones, D: Dimensions (2D: X,Y)
        # Used by: Navigation, Physics
        self.positions = torch.zeros(
            (self.batch_size, self.n_drones, 2), device=self.device)
        self.velocities = torch.zeros(
            (self.batch_size, self.n_drones, 2), device=self.device)

        # --- 2. HARDWARE STATUS ---
        # Used by: Controller
        # states: Discrete Enum (0=Returning, 1=Exploring, etc.)
        self.states = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device)

        self.batteries = torch.ones(
            (self.batch_size, self.n_drones, 1), device=self.device)
        self.payloads = torch.ones(
            (self.batch_size, self.n_drones, 1), device=self.device)

        # ALIVE: Bool is faster for masking operations
        self.alive_mask = torch.ones(
            (self.batch_size, self.n_drones, 1), dtype=torch.bool, device=self.device)
        self.launch_allowed_mask = torch.ones(
            (self.batch_size, self.n_drones, 1), dtype=torch.bool, device=self.device
        )
        self._random_spawn_mask = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.bool, device=self.device
        )

        # --- 3. LOGIC & COMMITMENT ---
        # Used by: Controller, Strategy
        # It represents "Commitment Timer" when Exploring, "Reload Timer" when at Base.
        self.timers = torch.zeros(
            (self.batch_size, self.n_drones, 1), device=self.device)
        self.commitment_decisions_remaining = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device
        )
        self.initial_force_go_remaining = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device
        )
        self.base_queue_entry_step = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device
        )
        self.wait_steps = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device
        )
        self.sortie_age_steps = torch.zeros(
            (self.batch_size, self.n_drones, 1), dtype=torch.long, device=self.device
        )

        # --- 4. CONFIGURATION ---
        # BASES: Hardcoded locations
        self.bases = torch.tensor(
            [[-0.9, -0.9], [0.9, -0.9], [-0.9, 0.9], [0.9, 0.9]], device=self.device)
        self._initialize_positions()

        # Stores who is at base (updated by Navigation)
        self._latest_at_base_mask = self._compute_at_base_mask(self.positions)

        # Pre-allocated scalar constant for hot-path eligibility checks (avoids MPS issues).
        self._one = torch.tensor(1.0, device=self.device)
        self._zero = torch.tensor(0.0, device=self.device)
        self._queue_step_sentinel = int(
            config['simulation'].get('max_steps', 0)
        ) + 10000
        self._not_self_mask = ~torch.eye(
            self.n_drones, device=self.device, dtype=torch.bool
        ).unsqueeze(0)
        self.attrition_rng = SeedManager.create_generator(self.device)
        self.attrition_deaths_this_step = torch.zeros(
            (self.batch_size, 1), device=self.device
        )
        self.attrition_deaths_total = torch.zeros(
            (self.batch_size, 1), device=self.device
        )
        self.initial_alive_counts = torch.zeros(
            (self.batch_size, 1), device=self.device
        )
        self._visual_heat_idx = int(self.perception.actor_feature_idx["visual_heat"])

        # CTDE observation dimensions
        self.actor_obs_dim = self.perception.actor_obs_dim
        self.critic_obs_dim = self.perception.critic_obs_dim

        # Cache for the latest perception data (B, N, D)
        self.latest_actor_obs = None
        self.latest_critic_obs = None
        self.latest_neighbor_count_abs = None
        self.latest_neighbor_active_count_abs = None
        self.latest_neighbor_firefighting_count_abs = None
        self.latest_return_success_count_abs = None
        self.latest_return_giveup_count_abs = None
        self.latest_neighbor_mask = None
        self.latest_startup_override_mask = None
        self.latest_congestion_factor = torch.zeros(
            (self.batch_size, self.n_drones, 1), device=self.device
        )

        self.step_idx = 0

    def _initialize_positions(self):
        """
        Assigns starting positions.
        Logic:
        1) Round-robin assignment to bases + random jitter to prevent stacking.
        2) Randomly place a fraction of alive drones directly in the field.
        """
        # 1. Create Assignments (0, 1, 2, 3, 0, ...)
        base_indices = torch.arange(
            self.n_drones, device=self.device) % self.bases.shape[0]

        # 2. Map Indices to Coordinates
        # self.bases is (4, 2) -> self.bases[base_indices] is (n_drones, 2)
        # We expand it to match the batch size (batch_size, n_drones, 2)
        base_coords = self.bases[base_indices].expand(self.batch_size, -1, -1)

        # 3. Generate Jitter
        # Random noise [-0.015, 0.015]
        jitter = (torch.rand_like(base_coords) - 0.5) * 0.03

        # 4. WRITE TO MASTER TENSOR
        base_positions = base_coords + jitter

        # 5. Random in-field spawn for a subset of launch-eligible drones.
        # Use a Bernoulli sampler on-device (MPS-safe) instead of index-heavy
        # argsort/nonzero pipelines, which can trigger accelerator index faults.
        alive = self.alive_mask.bool()
        launch_eligible = alive & self.launch_allowed_mask.bool()
        random_positions = (torch.rand_like(base_positions) * 2.0) - 1.0
        if self.initial_in_air_fraction <= 0.0:
            random_mask = torch.zeros_like(alive)
        elif self.initial_in_air_fraction >= 1.0:
            random_mask = launch_eligible
        else:
            random_mask = (torch.rand_like(base_positions[..., :1]) <
                           float(self.initial_in_air_fraction)) & launch_eligible
        self._random_spawn_mask = random_mask & launch_eligible
        self.positions.copy_(
            torch.where(self._random_spawn_mask,
                        random_positions, base_positions)
        )

    def _compute_at_base_mask(
        self,
        positions: torch.Tensor,
        radius: float | None = None,
    ) -> torch.Tensor:
        """
        Compute per-drone base proximity from current positions.
        """
        to_bases = self.bases.view(1, 1, -1, 2) - positions.unsqueeze(2)
        dist_bases = torch.norm(to_bases, dim=-1)
        min_dist = dist_bases.min(dim=2, keepdim=True).values
        threshold = self.AT_BASE_THRESHOLD if radius is None else float(radius)
        return min_dist < threshold

    def _nearest_base_indices(self, positions: torch.Tensor) -> torch.Tensor:
        to_bases = self.bases.view(1, 1, -1, 2) - positions.unsqueeze(2)
        dist_bases = torch.norm(to_bases, dim=-1)
        return torch.argmin(dist_bases, dim=2, keepdim=True)

    def _service_required_mask(
        self,
        *,
        state: torch.Tensor | None = None,
        timers: torch.Tensor | None = None,
        battery: torch.Tensor | None = None,
        payload: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.states if state is None else state
        timers = self.timers if timers is None else timers
        battery = self.batteries if battery is None else battery
        payload = self.payloads if payload is None else payload
        is_waiting = (state == self.controller.STATE_WAITING)
        return (
            is_waiting
            & self.alive_mask
            & (timers > 0.0)
            & ((battery < 0.999) | (payload < 0.999))
        )

    def _compute_service_progress_mask(self) -> torch.Tensor | None:
        if not self.base_service_queue_enabled:
            return None

        service_required = self._service_required_mask()
        service_mask = torch.zeros_like(service_required)
        if not service_required.any() or self.service_slots_per_base <= 0:
            return service_mask

        base_idx = self._nearest_base_indices(self.positions)
        drone_idx = torch.arange(
            self.n_drones, device=self.device, dtype=torch.long
        )
        for batch_idx in range(self.batch_size):
            for base_id in range(self.bases.shape[0]):
                candidate_mask = (
                    service_required[batch_idx, :, 0]
                    & (base_idx[batch_idx, :, 0] == base_id)
                )
                if not candidate_mask.any():
                    continue
                candidate_ids_cpu = torch.nonzero(
                    candidate_mask.detach().cpu(),
                    as_tuple=False,
                ).squeeze(-1)
                candidate_ids = candidate_ids_cpu.to(self.device)
                score = (
                    self.base_queue_entry_step[batch_idx, candidate_ids, 0]
                    .detach()
                    .cpu()
                    * (self.n_drones + 1)
                ) + drone_idx[candidate_ids].detach().cpu()
                ordered = candidate_ids_cpu[torch.argsort(score)].to(self.device)
                slots = min(self.service_slots_per_base, int(ordered.numel()))
                service_mask[batch_idx, ordered[:slots], 0] = True
        return service_mask

    def _compute_congestion_factor(
        self,
        dist_matrix: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        alive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.congestion_effects_enabled or self.congestion_radius <= 0.0:
            return torch.zeros_like(self.batteries)

        state = self.states if state is None else state
        alive_mask = self.alive_mask if alive_mask is None else alive_mask
        alive_2d = alive_mask.squeeze(-1).to(dtype=torch.bool)
        state_2d = state.squeeze(-1)
        airborne = alive_2d & (state_2d != int(DroneState.WAITING))
        # MPS has been brittle on boolean reduction kernels here; use a safe
        # integer count so baseline rollouts do not trip accelerator faults.
        airborne_count = int(airborne.to(dtype=torch.int32).sum().item())
        if airborne_count <= 0:
            return torch.zeros_like(self.batteries)

        nearby = (
            (dist_matrix <= self.congestion_radius)
            & airborne.unsqueeze(1)
            & airborne.unsqueeze(2)
            & self._not_self_mask
        )
        nearby_count = nearby.sum(dim=2, keepdim=True).float()
        denom = float(max(self.full_congestion_count - self.congestion_threshold, 1))
        congestion = torch.clamp(
            (nearby_count - float(self.congestion_threshold)) / denom,
            min=0.0,
            max=1.0,
        )
        return torch.where(
            airborne.unsqueeze(-1),
            congestion,
            torch.zeros_like(congestion),
        )

    def _compute_decision_requests(
        self,
        *,
        service_progress_mask: torch.Tensor | None,
        congestion_factor: torch.Tensor,
    ) -> torch.Tensor:
        new_timers, new_bat, new_pay, _service_finished = (
            self.controller.predict_physical_updates(
                state=self.states,
                timers=self.timers,
                battery=self.batteries,
                payload=self.payloads,
                service_progress_mask=service_progress_mask,
                congestion_factor=congestion_factor,
            )
        )

        timer_done = (new_timers <= 0)
        valid_waiting = (self.states == self.controller.STATE_WAITING) & (
            self.launch_allowed_mask > 0.5
        )
        valid_mode = valid_waiting | (self.states == self.controller.STATE_EXPLORING)
        is_healthy = (new_bat > self.controller.LOW_BATTERY) & (new_pay > 0)
        alive = (self.alive_mask > 0.5)
        return timer_done & valid_mode & is_healthy & alive

    def _update_base_queue_state(self, previous_states: torch.Tensor) -> None:
        if not self.base_service_queue_enabled:
            return

        sentinel = torch.full_like(
            self.base_queue_entry_step,
            self._queue_step_sentinel,
        )
        waiting_now = (self.states == self.controller.STATE_WAITING) & self.alive_mask
        service_required_now = self._service_required_mask()
        newly_waiting = (
            (previous_states != self.controller.STATE_WAITING)
            & waiting_now
            & service_required_now
        )
        step_tensor = torch.full_like(
            self.base_queue_entry_step,
            int(self.step_idx),
        )
        self.base_queue_entry_step = torch.where(
            newly_waiting,
            step_tensor,
            self.base_queue_entry_step,
        )
        self.base_queue_entry_step = torch.where(
            service_required_now,
            self.base_queue_entry_step,
            sentinel,
        )

    def step(
        self,
        physics: 'PhysicsManager',
        step_idx: int,
        actor_obs_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Performs a single simulation step for the swarm.
            1. Get Perception from Physics
            2. Execute Logic (Strategy + Controller)
            3. Update Movement (Navigation)
        Args:
            physics: Access to the Physics Manager for perception and state info.
            step_idx: Current simulation step index
        """
        self.step_idx = step_idx
        # ---------------------------------------------------------
        # GEOMETRY PRE-CALCULATION
        # ---------------------------------------------------------
        # We calculate the vector difference and distances ONCE.
        # Shape: (B, N, N, 2) and (B, N, N)
        # This is memory intensive but computationally efficient.

        diff_matrix = self.positions.unsqueeze(2) - self.positions.unsqueeze(1)
        dist_matrix = torch.norm(diff_matrix, dim=-1)

        # RL path can provide a pre-action observation so the transition uses
        # the exact same tensor that produced the sampled action.
        if actor_obs_override is None:
            actor_obs, _ = self.get_perception(physics, dist_mat=dist_matrix)
        else:
            actor_obs = actor_obs_override

        # Execute logic uses actor_obs (decentralized decision making)
        pre_logic_congestion = self._compute_congestion_factor(
            dist_matrix,
            state=self.states,
            alive_mask=self.alive_mask,
        )
        self.latest_congestion_factor = pre_logic_congestion
        service_progress_mask = self._compute_service_progress_mask()
        self.execute_logic(
            actor_obs,
            service_progress_mask=service_progress_mask,
            congestion_factor=pre_logic_congestion,
        )
        self._apply_attrition(actor_obs, dist_matrix)
        self.latest_congestion_factor = self._compute_congestion_factor(
            dist_matrix,
            state=self.states,
            alive_mask=self.alive_mask,
        )
        self.update_movement(physics, neighbor_data=(diff_matrix, dist_matrix))

        supression_data = self.get_suppression_data(physics)
        return supression_data

    def reset(
        self,
        n_active_agents: int | list[int] | torch.Tensor,
        seed: int | None = None,
        initial_launch_allowed_mask: torch.Tensor | None = None,
    ):
        """
        Resets the swarm hardware for a new mission.

        Args:
            n_active_agents: The number of drones that physically exist in this run.
        """
        # 1. Identity Management.
        self.alive_mask.fill_(False)

        if isinstance(n_active_agents, int):
            n = max(0, min(int(n_active_agents), self.n_drones))
            self.alive_mask[:, :n, :] = True
        else:
            counts = torch.as_tensor(
                n_active_agents,
                device=self.device,
                dtype=torch.long
            ).view(-1)
            if counts.numel() != self.batch_size:
                raise ValueError(
                    "n_active_agents must have one value per batch row. "
                    f"Got {counts.numel()} values for batch_size={self.batch_size}."
                )
            counts = counts.clamp(min=0, max=self.n_drones)
            idx = torch.arange(
                self.n_drones, device=self.device).view(1, -1, 1)
            limit = counts.view(self.batch_size, 1, 1)
            self.alive_mask.copy_(idx < limit)

        if initial_launch_allowed_mask is None:
            self.launch_allowed_mask.copy_(self.alive_mask)
        else:
            self.launch_allowed_mask.copy_(
                self.alive_mask & initial_launch_allowed_mask.to(
                    device=self.device, dtype=torch.bool
                )
            )

        # 2. Kinematics Reset: Place drones at bases in round-robin fashion
        self._initialize_positions()
        self.velocities.fill_(0.0)
        self._latest_at_base_mask = self._compute_at_base_mask(self.positions)
        self.attrition_deaths_this_step.zero_()
        self.attrition_deaths_total.zero_()
        self.initial_alive_counts.zero_()
        self.commitment_decisions_remaining.zero_()
        self.initial_force_go_remaining.zero_()
        self.base_queue_entry_step.fill_(self._queue_step_sentinel)
        self.wait_steps.zero_()
        self.sortie_age_steps.zero_()
        self.latest_congestion_factor.zero_()

        # 3. Hardware Reset
        self.states.fill_(DroneState.WAITING)
        self.states = torch.where(
            self._random_spawn_mask,
            torch.full_like(self.states, int(DroneState.EXPLORING)),
            self.states
        )
        if self.initial_force_go_decisions > 0:
            initial_force_go = torch.full_like(
                self.initial_force_go_remaining,
                self.initial_force_go_decisions,
            )
            self.initial_force_go_remaining = torch.where(
                self.alive_mask,
                initial_force_go,
                self.initial_force_go_remaining,
            )
        if self.controller.launch_commitment_decisions > 0:
            initial_commitment = torch.full_like(
                self.commitment_decisions_remaining,
                self.controller.launch_commitment_decisions,
            )
            self.commitment_decisions_remaining = torch.where(
                self._random_spawn_mask,
                initial_commitment,
                self.commitment_decisions_remaining,
            )
        self.batteries.fill_(1.0)
        self.payloads.fill_(1.0)  # 100% capacity
        # Start with random timer phases to stagger first decisions.
        if self.controller.decision_interval > 0:
            self.timers.copy_(
                torch.rand_like(self.timers) *
                float(self.controller.decision_interval)
            )
        else:
            self.timers.fill_(0.0)

        # Reset sensor memories each episode to avoid temporal leakage.
        if seed is None:
            seed = 0
        self.perception.reset(seed)
        SeedManager.seed_generator(self.attrition_rng, int(seed))

        # Reset stateful strategies (e.g. LaBella's P_l and streak counters).
        self.strategy.reset()

    def capture_initial_alive_counts(self) -> None:
        self.initial_alive_counts.copy_(
            self.alive_mask.squeeze(-1).float().sum(dim=1, keepdim=True)
        )

    def update_movement(self, physics: 'PhysicsManager', neighbor_data: tuple | None = None):
        """
        Orchestrates the movement step.
        1. Fetches environmental data from Physics(Fire Gradient).
        2. Delegates calculation to Navigation Engine.
        3. Updates internal Kinematics state.
        """

        # 1. FETCH DATA (The "Switchboard" Step)
        # We ask Physics for the gradient.
        # Because of your caching logic, this is cheap even if called often.
        fire_gradient = physics.get_gradient()

        # 2. DELEGATE TO NAV ENGINE
        # We pass the pure tensors. The Nav engine doesn't know 'physics_engine' exists.
        new_pos, new_vel, at_base_mask = self.navigation.update_movement(
            pos=self.positions,
            vel=self.velocities,
            states=self.states,
            alive_mask=self.alive_mask,
            fire_grad=fire_gradient,
            bases=self.bases,
            neighbor_data=neighbor_data,
            congestion_factor=self.latest_congestion_factor,
        )

        # 3. UPDATE STATE
        # We update our master tensors with the results
        self.positions = new_pos
        self.velocities = new_vel

        self._latest_at_base_mask = at_base_mask

    def execute_logic(
        self,
        actor_obs: torch.Tensor,
        *,
        service_progress_mask: torch.Tensor | None = None,
        congestion_factor: torch.Tensor | None = None,
    ):
        """
        The Gated Decision Cycle (CTDE: uses actor_obs for decentralized decisions).
        Args:
            actor_obs: (B, N, actor_obs_dim) Local observations for decision-making
        """
        # A. Identify Eligibility (Timer 0 + Resources OK)
        # Get the mask of agents who need decisions right now
        if congestion_factor is None:
            congestion_factor = self.latest_congestion_factor
        decision_mask = self._compute_decision_requests(
            service_progress_mask=service_progress_mask,
            congestion_factor=congestion_factor,
        )

        # B. Get Decision Intent (ONLY for eligible agents)
        # Initialize intent to 0 (Stay/Return) for all agents
        raw_intent = torch.zeros_like(self.states)
        sensor_data = SensorData(
            actor_obs=actor_obs,
            active_mask=decision_mask,
            payload=self.payloads,
            battery=self.batteries,
            alive_mask=self.alive_mask,
            states=self.states,
            neighbor_count_abs=self.latest_neighbor_count_abs,
            neighbor_active_count_abs=self.latest_neighbor_active_count_abs,
            neighbor_firefighting_count_abs=self.latest_neighbor_firefighting_count_abs,
            return_success_count_abs=self.latest_return_success_count_abs,
            return_giveup_count_abs=self.latest_return_giveup_count_abs,
            neighbor_mask=self.latest_neighbor_mask,
            startup_override_mask=self.latest_startup_override_mask,
            feature_index=self.perception.actor_feature_idx,
        )

        if hasattr(self.strategy, "observe_step"):
            self.strategy.observe_step(sensor_data)

        # Only call strategy for agents who need decisions
        if decision_mask.any():
            # Strategy returns (B, N, 1) of 0s and 1s
            strategic_intent = self.strategy.decide(sensor_data)
            # Apply the strategic intent only where decision_mask is True
            raw_intent = torch.where(
                decision_mask, strategic_intent, raw_intent)

        forced_go_mask = decision_mask & (self.initial_force_go_remaining > 0)
        if forced_go_mask.any():
            raw_intent = torch.where(
                forced_go_mask,
                torch.ones_like(raw_intent),
                raw_intent,
            )
            self.initial_force_go_remaining = torch.where(
                forced_go_mask,
                torch.clamp(self.initial_force_go_remaining - 1, min=0),
                self.initial_force_go_remaining,
            )

        # C. Parse Sensor Data for Controller
        # Index 0 is 'visual_heat' (See sensors.py actor_features)
        # We threshold it: If heat > 0.1, we "see" fire.
        fire_mask = (actor_obs[..., 0:1] > 0.1) & self.alive_mask

        # D. Update FSM
        previous_states = self.states.clone()
        (
            new_state,
            new_timers,
            new_bat,
            new_pay,
            new_commitment,
        ) = self.controller.update_states(
            state=self.states,
            timers=self.timers,
            battery=self.batteries,
            payload=self.payloads,
            intent_mask=raw_intent,
            launch_allowed_mask=self.launch_allowed_mask,
            commitment_decisions_remaining=self.commitment_decisions_remaining,
            at_base_mask=self._latest_at_base_mask,  # Calculated in update_movement
            fire_detected_mask=fire_mask,
            service_progress_mask=service_progress_mask,
            congestion_factor=congestion_factor,
        )

        self.states = new_state
        self.timers = new_timers
        self.batteries = new_bat
        self.payloads = new_pay
        self.commitment_decisions_remaining = new_commitment
        waiting_now = self.alive_mask & (
            self.states == self.controller.STATE_WAITING
        )
        in_field_now = self.alive_mask & (
            self.states != self.controller.STATE_WAITING
        )
        launched_now = (
            (previous_states == self.controller.STATE_WAITING)
            & in_field_now
        )
        newly_waiting = (
            (previous_states != self.controller.STATE_WAITING)
            & waiting_now
        )
        self.wait_steps = torch.where(
            waiting_now,
            torch.where(
                newly_waiting,
                torch.zeros_like(self.wait_steps),
                self.wait_steps + 1,
            ),
            torch.zeros_like(self.wait_steps),
        )
        self.sortie_age_steps = torch.where(
            in_field_now,
            torch.where(
                launched_now,
                torch.zeros_like(self.sortie_age_steps),
                self.sortie_age_steps + 1,
            ),
            torch.zeros_like(self.sortie_age_steps),
        )
        self._update_base_queue_state(previous_states)

    def _apply_attrition(
        self,
        actor_obs: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> None:
        self.attrition_deaths_this_step.zero_()
        if not self.attrition_enabled:
            return

        battery_threshold = max(self.attrition_battery_threshold, 0.0)
        if battery_threshold <= 0.0:
            return

        alive = self.alive_mask
        in_field = self.states != self.controller.STATE_WAITING
        safe_zone_radius = max(self.attrition_safe_zone_radius, self.AT_BASE_THRESHOLD)
        outside_safe_zone = ~self._compute_at_base_mask(
            self.positions,
            radius=safe_zone_radius,
        )
        low_battery = self.batteries <= battery_threshold
        eligible = alive & in_field & outside_safe_zone & low_battery
        if not eligible.any():
            return

        battery_factor = torch.clamp(
            (battery_threshold - self.batteries) / max(battery_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )

        visual_heat = actor_obs[..., self._visual_heat_idx:self._visual_heat_idx + 1]
        fire_threshold = min(max(self.attrition_fire_threshold, 0.0), 1.0)
        fire_factor = torch.clamp(
            (visual_heat - fire_threshold) / max(1.0 - fire_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )

        crowding_radius = max(self.attrition_crowding_radius, 0.0)
        if crowding_radius > 0.0:
            alive_2d = alive.squeeze(-1)
            nearby = (
                (dist_matrix <= crowding_radius)
                & alive_2d.unsqueeze(1)
                & alive_2d.unsqueeze(2)
                & self._not_self_mask
            )
            nearby_count = nearby.sum(dim=2, keepdim=True).float()
        else:
            nearby_count = torch.zeros_like(self.batteries)

        crowding_denom = float(max(self.attrition_crowding_threshold, 1))
        crowding_factor = torch.clamp(
            (nearby_count - float(self.attrition_crowding_threshold) + 1.0) / crowding_denom,
            min=0.0,
            max=1.0,
        )

        hazard = battery_factor * (
            (self.attrition_fire_weight * fire_factor) +
            (self.attrition_crowding_weight * crowding_factor)
        )
        death_prob = torch.clamp(
            hazard,
            min=0.0,
            max=min(max(self.attrition_max_death_prob, 0.0), 1.0),
        )
        death_prob = torch.where(eligible, death_prob, torch.zeros_like(death_prob))
        if not (death_prob > 0.0).any():
            return

        draws = torch.rand(
            death_prob.shape,
            device=self.device,
            generator=self.attrition_rng,
        )
        died = eligible & (draws < death_prob)
        if not died.any():
            return

        deaths_this_step = died.squeeze(-1).float().sum(dim=1, keepdim=True)
        self.attrition_deaths_this_step.copy_(deaths_this_step)
        self.attrition_deaths_total.add_(deaths_this_step)
        self.alive_mask &= ~died
        self.launch_allowed_mask &= ~died
        self.states = torch.where(died, self.controller.STATE_WAITING, self.states)
        self.timers = torch.where(died, self._zero, self.timers)
        self.commitment_decisions_remaining = torch.where(
            died,
            torch.zeros_like(self.commitment_decisions_remaining),
            self.commitment_decisions_remaining,
        )
        self.initial_force_go_remaining = torch.where(
            died,
            torch.zeros_like(self.initial_force_go_remaining),
            self.initial_force_go_remaining,
        )
        self.wait_steps = torch.where(
            died,
            torch.zeros_like(self.wait_steps),
            self.wait_steps,
        )
        self.sortie_age_steps = torch.where(
            died,
            torch.zeros_like(self.sortie_age_steps),
            self.sortie_age_steps,
        )
        self.base_queue_entry_step = torch.where(
            died,
            torch.full_like(self.base_queue_entry_step, self._queue_step_sentinel),
            self.base_queue_entry_step,
        )
        self.velocities = torch.where(
            died.expand_as(self.velocities),
            torch.zeros_like(self.velocities),
            self.velocities,
        )

    def get_perception(
        self, physics: 'PhysicsManager', dist_mat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pulls data from Physics to build CTDE observation streams.

        Returns:
            Tuple of (actor_obs, critic_obs):
                actor_obs: (B, N, actor_obs_dim) for decentralized execution
                critic_obs: (B, N, critic_obs_dim) for centralized training
        """
        fire_grid = physics.state
        actor_obs, critic_obs = self.perception.get_perception(
            self.positions,
            self.alive_mask,
            self.states,
            self.batteries,
            self.payloads,
            self.wait_steps,
            self.sortie_age_steps,
            fire_grid,
            dist_mat=dist_mat
        )
        self.latest_actor_obs = actor_obs
        self.latest_critic_obs = critic_obs
        self.latest_neighbor_count_abs = self.perception.neighbor_count_abs
        self.latest_neighbor_active_count_abs = self.perception.neighbor_active_count_abs
        self.latest_neighbor_firefighting_count_abs = self.perception.neighbor_firefighting_count_abs
        self.latest_return_success_count_abs = self.perception.return_success_count_abs
        self.latest_return_giveup_count_abs = self.perception.return_giveup_count_abs
        self.latest_neighbor_mask = self.perception.neighbor_mask
        self.latest_startup_override_mask = (
            self.alive_mask
            & (
                self._random_spawn_mask
                | (self.initial_force_go_remaining > 0)
                | (self.commitment_decisions_remaining > 0)
            )
        )
        return actor_obs, critic_obs

    def get_suppression_data(self, physics: 'PhysicsManager') -> torch.Tensor:
        """
        Determines drop commands for the 'Water Bomb' payload.

        Args:
            physics: Access to the fire grid for targeting.

        Returns:
            Tensor(B, N, 3) -> [x_norm, y_norm, drop_intensity]
        """
        B, N, _ = self.positions.shape

        heat_sample = torch.nn.functional.grid_sample(
            physics.state[:, 0:1],
            self.positions.view(B, N, 1, 2),
            align_corners=True,
            mode='bilinear'
        )
        current_heat = heat_sample.squeeze(-1).permute(0, 2, 1)  # (B, N, 1)

        in_mode = (self.states == self.controller.STATE_FIREFIGHTING)

        on_target = (current_heat > self.DROP_THRESHOLD)
        has_payload = (self.payloads > 0.0)
        should_drop = in_mode & on_target & has_payload

        drop_intensity = torch.where(
            should_drop,
            torch.tensor(self.DROP_INTENSITY, device=self.device),
            torch.tensor(0.0, device=self.device)
        )

        self.payloads = torch.where(
            should_drop,
            torch.tensor(0.0, device=self.device),
            self.payloads
        )

        # Stack [x, y, intensity]
        output = torch.cat([self.positions, drop_intensity], dim=2)

        return output

    def get_state(self) -> dict:
        """
        Facade Getter: Provides a read-only view of the swarm for
        the visualizer or metrics engine.
        """
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "states": self.states,
            "batteries": self.batteries,
            "alive_mask": self.alive_mask,
            "launch_allowed_mask": self.launch_allowed_mask,
        }

    def get_active_count(self) -> torch.Tensor:
        """Returns the number of drones currently in the field(B, 1)."""
        # Only count 'Alive' drones that are not in the WAITING states
        is_in_field = (self.states != DroneState.WAITING) & (
            self.alive_mask > 0.8)
        return is_in_field.float().sum(dim=1)

    @property
    def decision_requests(self) -> torch.Tensor:
        """
        Returns a boolean mask (B, N, 1) of agents who strictly need
        a new decision (Intents) right now.

        CRITICAL: This must match the FSM 'is_eligible' logic AFTER the
        controller does its timer countdown (timers - 1). Otherwise we get
        a timing bug where agents become eligible after we've already decided
        not to call the strategy for them.
        """
        diff_matrix = self.positions.unsqueeze(2) - self.positions.unsqueeze(1)
        dist_matrix = torch.norm(diff_matrix, dim=-1)
        congestion_factor = self._compute_congestion_factor(
            dist_matrix,
            state=self.states,
            alive_mask=self.alive_mask,
        )
        service_progress_mask = self._compute_service_progress_mask()
        return self._compute_decision_requests(
            service_progress_mask=service_progress_mask,
            congestion_factor=congestion_factor,
        )
