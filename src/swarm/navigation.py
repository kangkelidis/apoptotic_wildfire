"""
Swarm: Navigation Engine (The Pilot).

Updates:
1. Stability: Replaced Inverse-Square separation with Linear Falloff (No more explosions).
2. Smoothness: Implemented Cubic Boundary Pull (Soft edges).
3. Logic: Explicit masking for Boids vs Mission states.
"""

import torch
import torch.nn.functional as F

from src.swarm.constants import DroneState


class NavigationEngine:
    # --- 1. PHYSICAL CONSTANTS ---
    DRAG = 0.1             # Air resistance
    MAX_SPEED = 0.04       # Maximum velocity
    FORCE_SCALE = 0.05     # Acceleration scalar

    # --- 2. RADIUS DEFINITIONS ---
    # We use a single communication range for all boids logic for simplicity
    # but weight them differently.
    COMM_RANGE = 0.2

    # --- 3. WEIGHTS ---
    # Note: Weights are applied to normalized vectors (0.0-1.0), so they represent
    # the relative priority of each behavior.
    W_SEP = 0.2            # Keep personal space
    W_ALI = 0.2            # Flow with the group
    W_COH = 0.1            # Stay with the group
    W_FIRE = 2.0           # Mission priority

    # Boundary Weights
    W_BOUND = 0.3         # Strong pull to keep them in bounds
    WALL_MARGIN = 0.15     # Start feeling the pull at 0.75 (1.0 - 0.25)
    BOUND_POWER = 1.0      # Cubic curve (Smooth soft edge -> Hard wall)

    def __init__(self, config: dict):
        self.device = config['simulation']['device']
        self.B = config['simulation']['batch_size']
        self.N = config['swarm']['n_drones']
        congestion_cfg = config.get('swarm', {}).get('congestion_effects', {})
        self.congestion_effects_enabled = bool(
            congestion_cfg.get('enabled', False)
        )
        self.min_speed_multiplier = float(
            congestion_cfg.get('min_speed_multiplier', 0.45)
        )

    def _normalize(self, vectors: torch.Tensor):
        """
        Safely normalize vectors to unit length.
        Adds epsilon to norm to prevent division by zero.
        """
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        return vectors / (norms + 1e-6)

    def update_movement(self,
                        pos: torch.Tensor,
                        vel: torch.Tensor,
                        states: torch.Tensor,
                        alive_mask: torch.Tensor,
                        fire_grad: torch.Tensor,
                        bases: torch.Tensor,
                        neighbor_data: tuple | None = None,
                        congestion_factor: torch.Tensor | None = None) -> tuple:
        """
        Compute the next movement step using stable Linear Boids forces.

        Args:
            pos: (B, N, 2) Current positions of drones.
            vel: (B, N, 2) Current velocities of drones.
            states: (B, N, 1) Current state enum of drones.
            alive_mask: (B, N, 1) Boolean mask of alive drones.
            fire_grad: (B, 2, H, W) Pre-computed fire gradient for seeking behavior.
            bases: (M, 2) Coordinates of bases for return behavior.
            neighbor_data: Optional tuple (diff_matrix, dist_matrix) pre-computed neighbor info.
        """
        B, N, _ = pos.shape

        # ---------------------------------------------------------
        # 1. MASKS & STATE MANAGEMENT
        # ---------------------------------------------------------

        # Who is active? (Alive and not in the 'Waiting' pool)
        is_active = alive_mask & (states != DroneState.WAITING)

        # Who contributes to the flock?
        # (Only Exploring drones flock. Returning/Fighting drones do their own thing)
        is_flocker = (states == DroneState.EXPLORING)

        # Valid Neighbor Matrix:
        # A neighbor must be active AND a flocker to influence others.
        # (Returning drones are invisible to the flock to prevent dragging)
        valid_neighbor_mask = (is_active & is_flocker).float()

        # ---------------------------------------------------------
        # 2. NEIGHBORHOOD CALCULATIONS (Linear Falloff)
        # ---------------------------------------------------------

        if neighbor_data is not None:
            diff, dist_mat = neighbor_data
        else:
            diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # Vectors J->I
            dist_mat = torch.norm(diff, dim=-1)

        # Linear Weights: 1.0 at dist=0, 0.0 at dist=COMM_RANGE
        # This replaces the dangerous 1/dist^2 logic.
        dist_weights = torch.clamp(1.0 - dist_mat / self.COMM_RANGE, min=0.0)

        # Apply masks:
        # 1. Neighbor must be valid (Active & Flocker)
        # 2. Exclude self (diagonal)
        neighbor_mask = valid_neighbor_mask.transpose(1, 2)
        not_eye = ~torch.eye(N, device=self.device).bool().unsqueeze(0)

        final_weights = dist_weights * neighbor_mask * not_eye.float()

        # Sum of weights for averaging (Cohesion/Alignment)
        weight_sum = final_weights.sum(dim=2, keepdim=True) + 1e-6

        # ---------------------------------------------------------
        # 3. BOIDS FORCES (Sum-Then-Normalize)
        # ---------------------------------------------------------

        # A. SEPARATION
        # Weighted sum of vectors pointing FROM neighbor TO self.
        # Since 'diff' is (Self - Neighbor), it already points away.
        # We sum the weighted vectors, then normalize the result.
        raw_sep = (diff * final_weights.unsqueeze(-1)).sum(dim=2)
        f_sep = self._normalize(raw_sep)

        # B. ALIGNMENT
        # Weighted average of neighbor velocities
        avg_vel = (final_weights.unsqueeze(-1) *
                   vel.unsqueeze(1)).sum(dim=2) / weight_sum
        f_ali = self._normalize(avg_vel)

        # C. COHESION
        # Weighted average of neighbor positions -> Vector to Centroid
        centroid = (final_weights.unsqueeze(-1) *
                    pos.unsqueeze(1)).sum(dim=2) / weight_sum
        f_coh = self._normalize(centroid - pos)

        # ---------------------------------------------------------
        # 4. ENVIRONMENTAL FORCES
        # ---------------------------------------------------------

        # A. FIRE SEEKING (Gradient Sample)
        sampled_grads = F.grid_sample(
            fire_grad, pos.view(B, N, 1, 2), align_corners=True
        )
        fire_vec = sampled_grads.squeeze(-1).permute(0, 2, 1)
        has_fire = (torch.norm(fire_vec, dim=-1, keepdim=True) > 0.01).float()
        f_fire = self._normalize(fire_vec) * has_fire

        # B. BASE RETURN
        to_bases = bases.view(1, 1, -1, 2) - pos.unsqueeze(2)
        dist_bases = torch.norm(to_bases, dim=-1)
        closest_idx = torch.argmin(dist_bases, dim=2, keepdim=True)
        idx_expanded = closest_idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        base_vec = torch.gather(to_bases, 2, idx_expanded).squeeze(2)
        f_base = self._normalize(base_vec)

        # C. BOUNDARY FORCE (Cubic Center Pull)
        # Soft curve that gets very strong at the edge
        dist_from_center = torch.norm(pos, dim=-1, keepdim=True)
        threshold = 1.0 - self.WALL_MARGIN
        penetration = torch.clamp(dist_from_center - threshold, min=0.0)

        # Normalize penetration (0.0 to 1.0 within the margin)
        norm_pen = penetration / self.WALL_MARGIN

        # Cubic curve for smoothness: (x)^3
        boundary_strength = torch.pow(norm_pen, self.BOUND_POWER)

        # Vector pointing to center (0,0)
        to_center = self._normalize(-pos)
        f_bound = to_center * boundary_strength * self.W_BOUND

        # ---------------------------------------------------------
        # 5. INTEGRATION
        # ---------------------------------------------------------

        is_exploring = (states == DroneState.EXPLORING).float()
        is_returning = (states == DroneState.RETURNING).float()
        is_fighting = (states == DroneState.FIREFIGHTING).float()

        # Combine Forces
        steering = \
            is_exploring * (
                f_sep * self.W_SEP +
                f_ali * self.W_ALI +
                f_coh * self.W_COH +
                f_fire * self.W_FIRE +
                f_bound  # Active agents respect walls
            ) + \
            is_returning * (
                # Weaker separation when returning
                f_sep * (self.W_SEP * 0.5) +
                f_base * 2.0
            ) + \
            is_fighting * (
                f_sep * self.W_SEP +
                f_fire * self.W_FIRE +
                f_bound  # Fighters respect walls
            )

        # Apply Update
        new_vel = vel + (steering * self.FORCE_SCALE)

        # Drag
        new_vel = new_vel * (1.0 - self.DRAG)

        # Speed Limit
        speed = torch.norm(new_vel, dim=-1, keepdim=True)
        scale = torch.clamp(self.MAX_SPEED / (speed + 1e-6), max=1.0)
        new_vel = new_vel * scale

        if self.congestion_effects_enabled and congestion_factor is not None:
            congestion = torch.clamp(congestion_factor.float(), 0.0, 1.0)
            speed_multiplier = 1.0 - (
                congestion * (1.0 - self.min_speed_multiplier)
            )
            new_vel = new_vel * speed_multiplier

        # Explicitly zero out velocity for WAITING or DEAD agents.
        should_freeze = (
            states == DroneState.WAITING) | (~alive_mask)
        new_vel = torch.where(
            should_freeze, torch.zeros_like(new_vel), new_vel)

        # Final Safety for NaNs
        new_vel = torch.nan_to_num(new_vel, nan=0.0)

        # Position Update
        new_pos = pos + new_vel
        new_pos = torch.nan_to_num(
            new_pos, nan=0.0, posinf=1.0, neginf=-1.0
        )
        final_pos = torch.clamp(new_pos, -1.0, 1.0)

        dist_to_closest = torch.gather(dist_bases, 2, closest_idx)
        at_base_mask = (dist_to_closest < 0.05)

        return final_pos, new_vel, at_base_mask
