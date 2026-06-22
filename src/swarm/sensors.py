"""Swarm Perception Engine.

This module is responsible for translating the raw simulation state (physics tensors)
into the observation vector used by the Reinforcement Learning agent.

It uses Exponential Moving Averages (EMA) to track temporal trends.
"""

import torch
import torch.nn.functional as F

from src.utils.hardware import SeedManager

from .constants import COMM_RANGE, DroneState


class PerceptionEngine:
    """Handles sensor processing and observation assembly for the drone swarm.

    Attributes:
        active_features (list[str]): Ordered list of feature names to include in the output.
        heat_memory (Tensor): EMA buffer for visual heat sensing.
        encounter_memory (Tensor): EMA buffer for density (crowding) sensing.
        success_memory (Tensor): EMA buffer for 'Successful Mission' returns.
        giveup_memory (Tensor): EMA buffer for 'Aborted Mission' returns.
    """

    # Memory decay rate for EMAs. Higher alpha means more weight on recent observations.
    # 0.1 means the memory will reflect roughly the last 10 steps (1/alpha).
    ALPHA = 0.1
    # Actor-side heat emphasis multipliers.
    # Keep direct visual heat competitive with memory so fast-moving fire fronts
    # are visible to the actor instead of being dominated by stale EMA history.
    ACTOR_VISUAL_HEAT_GAIN = 200.0
    ACTOR_MEMORY_HEAT_GAIN = 500.0

    def __init__(self, config: dict):
        """Initializes the Perception Engine.

        Args:
            config: The full configuration dictionary
        """
        self.batch_size = config['simulation']['batch_size']
        self.n_drones = config['swarm']['n_drones']
        self.device = config['simulation']['device']
        self.grid_size = config['simulation']['grid_size']
        self.max_steps = float(config['simulation']['max_steps'])

        # CTDE (Centralized Training, Decentralized Execution) Feature Split
        #
        # Actor Features: Local/egocentric observations the agent can sense
        # independently during decentralized execution.
        self.actor_features = [
            'visual_heat',      # Local fire sensing (camera)
            'memory_heat',      # Heat trend (internal EMA)
            'battery',          # Local battery state for deploy/return timing
            'social_pressure',  # Crowding density
            'trend_success',    # Neighbors finding fire
            'trend_giveup',     # Neighbors failing
            'wait_time_norm',   # Time spent waiting at base
            'sortie_age_norm',  # Time spent in the current sortie
            'queue_pressure',   # Nearby waiting-drone fraction
            'airborne_pressure',  # Nearby airborne-drone fraction
            'agent_id_sin',     # Stable role signal to break policy symmetry
            'agent_id_cos',     # Stable role signal to break policy symmetry
        ]

        # Critic Features: Actor features + global/social info available
        # only during centralized training.
        self.critic_features = [
            # Local features (same as actor)
            'visual_heat',
            'memory_heat',
            'battery',
            'payload',
            'social_pressure',  # Crowding density
            'trend_success',    # Neighbors finding fire
            'trend_giveup',     # Neighbors failing
            'wait_time_norm',
            'sortie_age_norm',
            'queue_pressure',
            'airborne_pressure',
            'agent_id_sin',
            'agent_id_cos',
            # Global/social features (centralized only)
            'global_fire',      # Global fire coverage
            'global_mean_heat',     # Mean local heat seen by alive drones
            'global_mean_battery',  # Mean battery among alive drones
            'global_mean_payload',  # Mean payload among alive drones
            'global_active_ratio',  # Fraction of alive drones not waiting
        ]

        # Dimension info for policy network initialization
        self.actor_obs_dim = len(self.actor_features)
        self.critic_obs_dim = len(self.critic_features)

        # Name → index lookup for safe feature access from external code.
        self.actor_feature_idx = {
            name: i for i, name in enumerate(self.actor_features)
        }

        # Pre-allocate buffers on GPU
        shape = (self.batch_size, self.n_drones, 1)
        self.heat_memory = torch.zeros(shape, device=self.device)
        self.encounter_memory = torch.zeros(shape, device=self.device)
        self.success_memory = torch.zeros(shape, device=self.device)
        self.giveup_memory = torch.zeros(shape, device=self.device)
        self.neighbor_count_abs = torch.zeros(shape, device=self.device)
        self.neighbor_active_count_abs = torch.zeros(shape, device=self.device)
        self.neighbor_firefighting_count_abs = torch.zeros(
            shape, device=self.device)
        self.return_success_count_abs = torch.zeros(shape, device=self.device)
        self.return_giveup_count_abs = torch.zeros(shape, device=self.device)
        self.neighbor_mask = torch.zeros(
            (self.batch_size, self.n_drones, self.n_drones),
            dtype=torch.bool,
            device=self.device,
        )
        self._not_self_mask = ~torch.eye(
            self.n_drones, device=self.device, dtype=torch.bool
        ).unsqueeze(0)

        # Deterministic per-agent identity encoding.
        # This gives the shared actor a stable way to break symmetry.
        idx = torch.arange(
            self.n_drones, device=self.device).float().view(1, -1, 1)
        phase = (2.0 * torch.pi * idx) / max(self.n_drones, 1)
        self.agent_id_sin = torch.sin(phase).expand(
            self.batch_size, -1, -1).clone()
        self.agent_id_cos = torch.cos(phase).expand(
            self.batch_size, -1, -1).clone()

        # We pre-calculate the Box Blur kernel for visual sensing.
        # Kernel radius is half the communication range.
        k_radius = int((COMM_RANGE / 2.0) * self.grid_size)
        # Ensure kernel size is odd (e.g., 3, 5, 7) for proper centering.
        k_pixels = max(3, k_radius if k_radius % 2 != 0 else k_radius + 1)

        # A normalized averaging kernel (Sum = 1.0)
        self.kernel = torch.ones(
            (1, 1, k_pixels, k_pixels), device=self.device
        ) / (k_pixels**2)

        self.padding = k_pixels // 2

        # RNG for any future stochastic sensors
        self.rng = SeedManager.create_generator(self.device)

    def reset(self, seed: int):
        """Resets all temporal memory buffers.

        Args:
            seed: The random seed for the current episode.
        """
        self.heat_memory.fill_(0.0)
        self.encounter_memory.fill_(0.0)
        self.success_memory.fill_(0.0)
        self.giveup_memory.fill_(0.0)
        self.neighbor_count_abs.fill_(0.0)
        self.neighbor_active_count_abs.fill_(0.0)
        self.neighbor_firefighting_count_abs.fill_(0.0)
        self.return_success_count_abs.fill_(0.0)
        self.return_giveup_count_abs.fill_(0.0)
        self.neighbor_mask.zero_()
        SeedManager.seed_generator(self.rng, seed)

    def get_perception(self,
                       pos: torch.Tensor,
                       alive_mask: torch.Tensor,
                       drone_states: torch.Tensor,
                       battery: torch.Tensor,
                       payload: torch.Tensor,
                       wait_steps: torch.Tensor,
                       sortie_age_steps: torch.Tensor,
                       fire_grid: torch.Tensor,
                       dist_mat: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the observation vectors for all agents (CTDE split).

        Args:
            pos: (B, N, 2) Tensor of positions [-1, 1].
            alive_mask: (B, N, 1) Boolean mask of alive agents.
            drone_states: (B, N, 1) Enum state of agents.
            battery: (B, N, 1) Current battery levels [0-1].
            payload: (B, N, 1) Current water payload [0,1].
            wait_steps: (B, N, 1) Steps spent continuously in WAITING.
            sortie_age_steps: (B, N, 1) Steps spent continuously outside WAITING.
            fire_grid: (B, C, H, W) The physics grid (Channel 0 is Heat).
            dist_mat: (B, N, N) Optional pre-computed distance matrix between agents.

        Returns:
            Tuple of (actor_obs, critic_obs):
                actor_obs: (B, N, actor_obs_dim) Local observations for decentralized execution.
                critic_obs: (B, N, critic_obs_dim) Full observations for centralized training.
        """
        # Guard against invalid coordinates on accelerator backends.
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=-1.0)
        pos = torch.clamp(pos, -1.0, 1.0)
        features = {}

        # ---------------------------------------------------------
        # Visual Sensing (Heat Perception)
        # ---------------------------------------------------------
        # Apply the Box Blur kernel to the Heat Channel (Index 0).
        # This simulates the drone's camera footprint.
        perceived_map = F.conv2d(
            fire_grid[:, 0:1], self.kernel, padding=self.padding
        )

        # Sample the blurred map at the drone's exact coordinates.
        # grid_sample expects coordinates in (B, N, 1, 2).
        instant_heat = F.grid_sample(
            perceived_map,
            pos.view(self.batch_size, -1, 1, 2),
            align_corners=True
        ).view(self.batch_size, -1, 1)

        # Update Heat Memory (EMA)
        # Formula: Memory = (1 - alpha) * Memory + alpha * New
        self.heat_memory.mul_(1.0 - self.ALPHA).add_(
            instant_heat, alpha=self.ALPHA
        )

        features['visual_heat'] = instant_heat
        features['memory_heat'] = self.heat_memory.clone()

        # ---------------------------------------------------------
        # Global Awareness
        # ---------------------------------------------------------
        # Calculate what % of the grid is on fire (Intensity > 0.1)
        # Shape: (B, 1, H, W) -> (B, 1, 1)
        B, C, H, W = fire_grid.shape
        total_cells = float(H * W)

        # Sum over height/width dims
        fire_coverage = (fire_grid[:, 0] > 0.05).float().sum(
            dim=(1, 2), keepdim=True) / total_cells

        # Broadcast to all agents: (B, 1, 1) -> (B, N, 1)
        features['global_fire'] = fire_coverage.expand(-1, self.n_drones, -1)

        # ---------------------------------------------------------
        # Social Topology (Neighborhood Analysis)
        # ---------------------------------------------------------
        # Calculate pair-wise distances between all agents.
        if dist_mat is None:
            dist_mat = torch.cdist(pos, pos)

        # Determine Neighbors:
        # 1. Must be within Radio Range.
        # 2. Must be Alive (alive_mask).
        in_range = dist_mat < COMM_RANGE
        alive_bool = alive_mask.bool()
        valid_neighbors_bool = (
            in_range
            & alive_bool
            & alive_bool.transpose(1, 2)
            & self._not_self_mask
        )
        valid_neighbors = valid_neighbors_bool.float()
        self.neighbor_mask = valid_neighbors_bool.clone()

        # Calculate Density (Count of neighbors minus self)
        current_density = valid_neighbors.sum(dim=2, keepdim=True)
        self.neighbor_count_abs = current_density.clone()

        active_neighbors = (
            (drone_states != int(DroneState.WAITING)).float()
            * alive_mask.float()
        )
        neighbor_active_count = (
            torch.bmm(valid_neighbors, active_neighbors)
        )
        self.neighbor_active_count_abs = torch.clamp(
            neighbor_active_count,
            min=0.0,
        )
        waiting_neighbors = (
            (drone_states == int(DroneState.WAITING)).float()
            * alive_mask.float()
        )
        queue_neighbor_count = torch.bmm(valid_neighbors, waiting_neighbors)

        firefighting_neighbors = (
            (drone_states == int(DroneState.FIREFIGHTING)).float()
            * alive_mask.float()
        )
        neighbor_firefighting_count = (
            torch.bmm(valid_neighbors, firefighting_neighbors)
        )
        self.neighbor_firefighting_count_abs = torch.clamp(
            neighbor_firefighting_count,
            min=0.0,
        )

        # Update Density Memory (EMA)
        self.encounter_memory.mul_(1.0 - self.ALPHA).add_(
            current_density, alpha=self.ALPHA
        )

        # Calculate Social Pressure:
        # Ratio of "Current Crowding" / "Usual Crowding".
        # > 1.0 means the area is getting crowded. < 1.0 means thinning.
        features['social_pressure'] = current_density / \
            (self.encounter_memory + 1e-5)

        # ---------------------------------------------------------
        # Traffic Analysis (Social Trends)
        # ---------------------------------------------------------
        is_returning = (drone_states == DroneState.RETURNING)

        # Trend A: Success ("We found fire!")
        # Logic: Neighbors returning with EMPTY payload implies they dropped it.
        mask_success = (is_returning & (payload <= 0.0)).float()

        # Trend B: Give Up ("Dead end / Low Battery")
        # Logic: Neighbors returning with FULL payload implies they found nothing.
        mask_giveup = (is_returning & (payload > 0.0)).float()

        # Count occurrences in neighborhood
        count_success = torch.bmm(valid_neighbors, mask_success)
        count_giveup = torch.bmm(valid_neighbors, mask_giveup)
        self.return_success_count_abs = count_success.clone()
        self.return_giveup_count_abs = count_giveup.clone()

        # Update Trend Memories (EMA)
        self.success_memory.mul_(1.0 - self.ALPHA).add_(
            count_success, alpha=self.ALPHA
        )
        self.giveup_memory.mul_(1.0 - self.ALPHA).add_(
            count_giveup, alpha=self.ALPHA
        )

        # Normalize Trends
        # We divide by the "Usual Density" to get a stable ratio.
        # "What % of my usual social circle is returning successfully?"
        denom = self.encounter_memory + 1e-5
        features['trend_success'] = self.success_memory / denom
        features['trend_giveup'] = self.giveup_memory / denom

        # ---------------------------------------------------------
        # Internal State
        # ---------------------------------------------------------
        # Battery is already linear (1.0 -> 0.0). Clamping ensures safety.
        features['battery'] = torch.clamp(battery, 0.0, 1.0)
        features['payload'] = torch.clamp(payload, 0.0, 1.0)
        features['wait_time_norm'] = torch.clamp(
            wait_steps.float() / max(self.max_steps, 1.0), 0.0, 1.0
        )
        features['sortie_age_norm'] = torch.clamp(
            sortie_age_steps.float() / max(self.max_steps, 1.0), 0.0, 1.0
        )
        density_denom = current_density + 1e-5
        features['queue_pressure'] = torch.clamp(
            queue_neighbor_count / density_denom, 0.0, 1.0
        )
        features['airborne_pressure'] = torch.clamp(
            neighbor_active_count / density_denom, 0.0, 1.0
        )
        features['agent_id_sin'] = self.agent_id_sin
        features['agent_id_cos'] = self.agent_id_cos

        # ---------------------------------------------------------
        # Centralized Context (swarm-level summaries)
        # ---------------------------------------------------------
        alive_float = alive_mask.float()
        alive_count = alive_float.sum(dim=1, keepdim=True).clamp(min=1.0)

        def mean_over_alive(x: torch.Tensor) -> torch.Tensor:
            mean_val = (x * alive_float).sum(dim=1, keepdim=True) / alive_count
            return mean_val.expand(-1, self.n_drones, -1)

        features['global_mean_heat'] = mean_over_alive(features['visual_heat'])
        features['global_mean_battery'] = mean_over_alive(features['battery'])
        features['global_mean_payload'] = mean_over_alive(features['payload'])

        active_mask = (
            drone_states != DroneState.WAITING).float() * alive_float
        active_ratio = active_mask.sum(dim=1, keepdim=True) / alive_count
        features['global_active_ratio'] = active_ratio.expand(
            -1, self.n_drones, -1)

        # ---------------------------------------------------------
        # 5. Assembly (CTDE Split)
        # ---------------------------------------------------------
        # Build separate observation tensors for actor and critic.
        # Heat gains affect only actor stream (decision policy), not critic stream.
        actor_parts = []
        for key in self.actor_features:
            x = features[key]
            if key == 'visual_heat':
                x = x * self.ACTOR_VISUAL_HEAT_GAIN
            elif key == 'memory_heat':
                x = x * self.ACTOR_MEMORY_HEAT_GAIN
            actor_parts.append(x)
        actor_obs = torch.cat(actor_parts, dim=-1)
        critic_obs = torch.cat([features[key]
                               for key in self.critic_features], dim=-1)

        return actor_obs, critic_obs
