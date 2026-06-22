"""
Drone Renderer: Swarm Visualization Layer.

Responsible for rendering drone-related elements:
- Drone markers with state-based colors
- Base station markers
- Optional: trails, sensor ranges, etc.

This layer is rendered on top of the terrain.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DroneColors:
    """
    Color configuration for drone states (BGR format).

    Maps DroneState enum values to colors:
        WAITING (0): At base, reloading
        EXPLORING (1): Searching for fire
        FIREFIGHTING (2): Actively suppressing
        RETURNING (3): Heading back to base
    """
    waiting: Tuple[int, int, int] = (255, 200, 100)      # Light blue
    exploring: Tuple[int, int, int] = (255, 255, 255)    # White
    firefighting: Tuple[int, int, int] = (0, 165, 255)   # Orange
    returning: Tuple[int, int, int] = (255, 0, 255)      # Magenta

    # Fallback for unknown states
    inactive: Tuple[int, int, int] = (100, 100, 100)     # Gray

    def get_state_color(self, state: int) -> Tuple[int, int, int]:
        """Get color for a drone state enum value."""
        colors = [self.waiting, self.exploring,
                  self.firefighting, self.returning]
        if 0 <= state < len(colors):
            return colors[state]
        return self.inactive

    def as_list(self) -> List[Tuple[int, int, int]]:
        """Return colors as list indexed by state enum."""
        return [self.waiting, self.exploring, self.firefighting, self.returning]


@dataclass
class BaseColors:
    """Color configuration for base stations (BGR format)."""
    fill: Tuple[int, int, int] = (0, 255, 255)           # Yellow
    border: Tuple[int, int, int] = (0, 0, 0)             # Black


@dataclass
class DroneRenderConfig:
    """Configuration for drone rendering."""
    # Drone marker sizes
    active_radius: int = 2
    inactive_radius: int = 1

    # Base marker sizes
    base_radius: int = 6
    base_border_width: int = 2

    # Colors
    drone_colors: DroneColors = field(default_factory=DroneColors)
    base_colors: BaseColors = field(default_factory=BaseColors)


class DroneRenderer:
    """
    Renders drone swarm visualization.

    Draws drone markers with state-based coloring and base stations.
    Provides drone statistics for HUD display.
    """

    # State name mapping for stats
    STATE_NAMES = ["waiting", "exploring", "firefighting", "returning"]
    REFERENCE_RENDER_SIZE = 800

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[DroneRenderConfig] = None,
        render_bases: bool = True,
    ):
        """
        Initialize drone renderer.

        Args:
            width: Output frame width in pixels
            height: Output frame height in pixels
            config: Rendering configuration (uses defaults if None)
            render_bases: Whether to draw base station markers
        """
        self.width = width
        self.height = height
        self.config = self._scaled_config(config or DroneRenderConfig())
        self.render_bases = render_bases

        # Pre-compute color list for fast lookup
        self._state_colors = self.config.drone_colors.as_list()

    def _render_scale(self) -> float:
        return max(0.5, min(self.width, self.height) / self.REFERENCE_RENDER_SIZE)

    @staticmethod
    def _scale_int(value: int, scale: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(value) * scale)))

    def _scaled_config(self, config: DroneRenderConfig) -> DroneRenderConfig:
        scale = self._render_scale()
        return replace(
            config,
            active_radius=self._scale_int(
                config.active_radius, scale, minimum=1),
            inactive_radius=self._scale_int(
                config.inactive_radius, scale, minimum=1),
            base_radius=self._scale_int(config.base_radius, scale, minimum=1),
            base_border_width=self._scale_int(
                config.base_border_width, scale, minimum=1),
        )

    def render(
        self,
        frame: np.ndarray,
        positions: np.ndarray,
        states: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        base_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render drones and bases onto frame (modifies in place).

        Args:
            frame: BGR frame to draw on (modified in place)
            positions: (N, 2) drone coordinates in [-1, 1] normalized space
            states: (N,) or (N, 1) drone state enum values
            active_mask: (N,) or (N, 1) binary mask for active drones
            base_positions: (M, 2) base coordinates in [-1, 1] space

        Returns:
            Modified frame (same reference as input)
        """
        # Draw bases first (under drones)
        if self.render_bases and base_positions is not None:
            self._draw_bases(frame, base_positions)

        # Draw drones
        self._draw_drones(frame, positions, states, active_mask)

        return frame

    def _normalize_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized [-1, 1] coordinates to pixel coordinates."""
        px = int((x + 1) / 2 * (self.width - 1))
        py = int((y + 1) / 2 * (self.height - 1))
        return px, py

    def _draw_bases(self, frame: np.ndarray, positions: np.ndarray) -> None:
        """Draw base station markers."""
        cfg = self.config

        for i in range(len(positions)):
            px, py = self._normalize_to_pixel(positions[i, 0], positions[i, 1])

            # Draw border circle (larger)
            cv2.circle(
                frame, (px, py),
                cfg.base_radius + cfg.base_border_width,
                cfg.base_colors.border, -1
            )
            # Draw fill circle
            cv2.circle(
                frame, (px, py),
                cfg.base_radius,
                cfg.base_colors.fill, -1
            )

    def _draw_drones(
        self,
        frame: np.ndarray,
        positions: np.ndarray,
        states: Optional[np.ndarray],
        active_mask: Optional[np.ndarray],
    ) -> None:
        """Draw drone markers with state-based colors."""
        cfg = self.config
        n_drones = len(positions)

        # Flatten arrays if needed
        if states is not None and states.ndim > 1:
            states = states.flatten()
        if active_mask is not None and active_mask.ndim > 1:
            active_mask = active_mask.flatten()

        for i in range(n_drones):
            if active_mask is not None and active_mask[i] <= 0.5:
                # Reserved/not deployed drones are hidden in the main overlay.
                continue

            px, py = self._normalize_to_pixel(positions[i, 0], positions[i, 1])

            # Determine color and radius
            if states is not None:
                state_idx = int(states[i])
                if 0 <= state_idx < len(self._state_colors):
                    color = self._state_colors[state_idx]
                else:
                    color = cfg.drone_colors.inactive
                radius = cfg.active_radius
            elif active_mask is not None:
                is_active = active_mask[i] > 0.5
                color = self._state_colors[1] if is_active else cfg.drone_colors.inactive
                radius = cfg.active_radius if is_active else cfg.inactive_radius
            else:
                color = self._state_colors[1]  # Default: exploring color
                radius = cfg.active_radius

            # Draw filled circle
            cv2.circle(frame, (px, py), radius, color, -1)

    def get_drone_stats(
        self,
        states: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """
        Calculate drone statistics for HUD display.

        Args:
            states: (N,) or (N, 1) drone state enum values
            active_mask: (N,) or (N, 1) binary mask for deployed drones
                        (1 = deployed/in-action, 0 = reserved/not yet deployed)

        Returns:
            Dict with:
            - deployed: number of drones currently in action (not reserved)
            - waiting/exploring/firefighting/returning: counts per state
            - active: non-waiting deployed drones
            - active_pct: percentage of deployed drones that are not waiting
        """
        if states.ndim > 1:
            states = states.flatten()

        # Determine which drones are deployed (in action)
        if active_mask is not None:
            if active_mask.ndim > 1:
                active_mask = active_mask.flatten()
            deployed_mask = active_mask > 0.5
            deployed = int(deployed_mask.sum())

            # Count states only for deployed drones
            deployed_states = states[deployed_mask]
        else:
            # No mask means all drones are deployed
            deployed = len(states)
            deployed_states = states
            deployed_mask = np.ones(len(states), dtype=bool)

        # Count states among deployed drones
        waiting = int((deployed_states == 0).sum())
        exploring = int((deployed_states == 1).sum())
        firefighting = int((deployed_states == 2).sum())
        returning = int((deployed_states == 3).sum())

        # Active = deployed drones that are NOT waiting
        active = deployed - waiting
        active_pct = (active / deployed * 100) if deployed > 0 else 0

        stats = {
            "total": len(states),  # Total including reserved
            "deployed": deployed,   # Currently in action
            "waiting": waiting,
            "exploring": exploring,
            "firefighting": firefighting,
            "returning": returning,
            "active": active,       # Non-waiting deployed drones
            "active_pct": active_pct,
        }

        return stats
