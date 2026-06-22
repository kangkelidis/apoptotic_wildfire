"""
Frame Compositor: Combines Rendering Layers.

Orchestrates the rendering pipeline by combining:
1. TerrainRenderer (bottom layer)
2. DroneRenderer (middle layer)
3. HUDRenderer (top layer)

Provides a simple interface for the simulation loop.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .drones import DroneRenderConfig, DroneRenderer
from .hud import HUDConfig, HUDRenderer
from .terrain import TerrainColors, TerrainRenderer, TerrainThresholds


@dataclass
class CompositorConfig:
    """Configuration for the frame compositor."""
    # Layer visibility toggles
    render_terrain: bool = True
    render_drones: bool = True
    render_hud: bool = True

    # Sub-renderer configs (use defaults if None)
    terrain_colors: Optional[TerrainColors] = None
    terrain_thresholds: Optional[TerrainThresholds] = None
    drone_config: Optional[DroneRenderConfig] = None
    hud_config: Optional[HUDConfig] = None

    # Terrain options
    render_retardant: bool = True

    # Drone options
    render_bases: bool = True


class FrameCompositor:
    """
    Combines all rendering layers into final frames.

    This is the main entry point for frame rendering. It:
    1. Extracts state from tensors
    2. Renders terrain (fire, fuel, retardant)
    3. Overlays drones and bases
    4. Adds HUD with statistics

    Usage:
        compositor = FrameCompositor(width=600, height=600)

        frame = compositor.render(
            state_tensor=model.state[0],
            drone_positions=positions[0],
            drone_states=states[0],
            base_positions=bases,
            step=current_step,
            wind=config['physics']['wind'],
        )
    """

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[CompositorConfig] = None,
    ):
        """
        Initialize the frame compositor.

        Args:
            width: Output frame width in pixels
            height: Output frame height in pixels
            config: Compositor configuration (uses defaults if None)
        """
        self.width = width
        self.height = height
        self.config = config or CompositorConfig()

        # Initialize sub-renderers
        self.terrain = TerrainRenderer(
            width=width,
            height=height,
            colors=self.config.terrain_colors,
            thresholds=self.config.terrain_thresholds,
            render_retardant=self.config.render_retardant,
        )

        self.drones = DroneRenderer(
            width=width,
            height=height,
            config=self.config.drone_config,
            render_bases=self.config.render_bases,
        )

        self.hud = HUDRenderer(
            width=width,
            height=height,
            config=self.config.hud_config,
        )

    def render(
        self,
        state_tensor: torch.Tensor,
        drone_positions: Optional[torch.Tensor] = None,
        drone_states: Optional[torch.Tensor] = None,
        alive_mask: Optional[torch.Tensor] = None,
        base_positions: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        wind: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Render a complete frame with all layers.

        Args:
            state_tensor: Physics state (3, H, W) with heat, fuel, retardant
            drone_positions: (N, 2) drone coordinates in [-1, 1] space
            drone_states: (N,) or (N, 1) drone state enum values
            alive_mask: (N,) or (N, 1) binary alive mask
            base_positions: (M, 2) base coordinates in [-1, 1] space
            step: Current simulation step
            total_steps: Total simulation steps
            wind: Wind vector dict with 'x' and 'y' components

        Returns:
            BGR frame (height, width, 3) as uint8
        """
        cfg = self.config

        # Extract numpy arrays from tensors
        state_np = state_tensor.detach().cpu().numpy()
        heat = state_np[0]
        fuel = state_np[1]
        retardant = state_np[2]

        # Layer 1: Terrain
        if cfg.render_terrain:
            frame = self.terrain.render(heat, fuel, retardant)
        else:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (64, 89, 102)  # Background color

        # Layer 2: Drones
        if cfg.render_drones and drone_positions is not None:
            pos_np = drone_positions.detach().cpu().numpy()
            states_np = drone_states.detach().cpu().numpy() if drone_states is not None else None
            mask_np = alive_mask.detach().cpu().numpy() if alive_mask is not None else None
            base_np = base_positions.detach().cpu().numpy(
            ) if base_positions is not None else None

            self.drones.render(frame, pos_np, states_np, mask_np, base_np)

        # Layer 3: HUD
        if cfg.render_hud:
            # Calculate statistics
            drone_stats = None
            fire_stats = None

            if drone_states is not None:
                states_np = drone_states.detach().cpu().numpy()
                mask_np = alive_mask.detach().cpu().numpy() if alive_mask is not None else None
                drone_stats = self.drones.get_drone_stats(states_np, mask_np)

            fire_stats = self.terrain.get_fire_stats(heat, fuel)

            self.hud.render(
                frame,
                step=step,
                total_steps=total_steps,
                wind=wind,
                drone_stats=drone_stats,
                fire_stats=fire_stats,
            )

        return frame

    def set_hud_visibility(
        self,
        show_step: Optional[bool] = None,
        show_wind: Optional[bool] = None,
        show_drone_stats: Optional[bool] = None,
        show_fire_stats: Optional[bool] = None,
    ) -> None:
        """
        Toggle HUD element visibility.

        Args:
            show_step: Show step counter
            show_wind: Show wind indicator
            show_drone_stats: Show drone statistics
            show_fire_stats: Show fire statistics
        """
        hud_cfg = self.hud.config

        if show_step is not None:
            hud_cfg.show_step = show_step
        if show_wind is not None:
            hud_cfg.show_wind = show_wind
        if show_drone_stats is not None:
            hud_cfg.show_drone_stats = show_drone_stats
        if show_fire_stats is not None:
            hud_cfg.show_fire_stats = show_fire_stats

    def set_layer_visibility(
        self,
        terrain: Optional[bool] = None,
        drones: Optional[bool] = None,
        hud: Optional[bool] = None,
    ) -> None:
        """
        Toggle layer visibility.

        Args:
            terrain: Show terrain layer
            drones: Show drones layer
            hud: Show HUD layer
        """
        if terrain is not None:
            self.config.render_terrain = terrain
        if drones is not None:
            self.config.render_drones = drones
        if hud is not None:
            self.config.render_hud = hud
