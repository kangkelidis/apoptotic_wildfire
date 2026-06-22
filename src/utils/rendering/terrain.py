"""
Terrain Renderer: Fire, Fuel, and Retardant Visualization.

Responsible for rendering the physical simulation state:
- Vegetation (green gradient based on fuel density)
- Fire (thermal colormap from heat intensity)
- Burnt areas (dark brown/ash)
- Retardant coverage (blue overlay)

This is the bottom layer of the rendering stack.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TerrainColors:
    """Color configuration for terrain rendering (BGR format)."""
    # Background (bare ground)
    background: Tuple[int, int, int] = (64, 89, 102)

    # Burnt ground (fuel depleted)
    burnt: Tuple[int, int, int] = (50, 25, 30)

    # Unburnt vegetation base (used for gradient)
    vegetation_base: Tuple[int, int, int] = (38, 128, 38)


@dataclass
class TerrainThresholds:
    """Threshold configuration for terrain state detection."""
    # Fuel below this is considered "burnt"
    burnt_fuel: float = 0.05

    # Heat above this is considered "on fire"
    fire: float = 0.05

    # Retardant above this is rendered
    retardant: float = 0.01


class TerrainRenderer:
    """
    Renders the physical terrain state (fire, fuel, retardant).

    This renderer creates the base layer showing:
    1. Green vegetation with density gradient
    2. Thermal fire visualization
    3. Dark burnt areas
    4. Optional blue retardant overlay
    """

    def __init__(
        self,
        width: int,
        height: int,
        colors: Optional[TerrainColors] = None,
        thresholds: Optional[TerrainThresholds] = None,
        render_retardant: bool = True,
    ):
        """
        Initialize terrain renderer.

        Args:
            width: Output frame width in pixels
            height: Output frame height in pixels
            colors: Color configuration (uses defaults if None)
            thresholds: Threshold configuration (uses defaults if None)
            render_retardant: Whether to show retardant overlay
        """
        self.width = width
        self.height = height
        self.colors = colors or TerrainColors()
        self.thresholds = thresholds or TerrainThresholds()
        self.render_retardant = render_retardant

    def render(
        self,
        heat: np.ndarray,
        fuel: np.ndarray,
        retardant: np.ndarray,
    ) -> np.ndarray:
        """
        Render terrain state to a BGR frame.

        Args:
            heat: (H, W) fire intensity in [0, 1]
            fuel: (H, W) vegetation density in [0, 1]
            retardant: (H, W) suppression level in [0, 1]

        Returns:
            BGR frame (height, width, 3) as uint8
        """
        # Resize if needed
        if heat.shape[0] != self.height or heat.shape[1] != self.width:
            heat = cv2.resize(heat, (self.width, self.height),
                              interpolation=cv2.INTER_NEAREST)
            fuel = cv2.resize(fuel, (self.width, self.height),
                              interpolation=cv2.INTER_NEAREST)
            retardant = cv2.resize(
                retardant, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # Initialize with background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = self.colors.background

        # Layer 1: Vegetation (green gradient based on fuel)
        self._render_vegetation(frame, fuel)

        # Layer 2: Burnt areas (dark where fuel depleted and no fire)
        self._render_burnt(frame, heat, fuel)

        # Layer 3: Fire (thermal colormap)
        self._render_fire(frame, heat, fuel)

        # Layer 4: Retardant overlay (optional)
        if self.render_retardant:
            self._render_retardant(frame, heat, retardant)

        return frame

    def _render_vegetation(self, frame: np.ndarray, fuel: np.ndarray) -> None:
        """Render green vegetation with fuel density gradient."""
        unburnt_mask = fuel > self.thresholds.burnt_fuel

        if not np.any(unburnt_mask):
            return

        # Convert fuel to color intensity
        fuel_values = fuel[unburnt_mask]
        fuel_uint8 = np.clip(fuel_values * 255, 0, 255).astype(np.uint8)

        # Use viridis colormap and shift to green tones
        fuel_2d = fuel_uint8.reshape(-1, 1)
        colored = cv2.applyColorMap(fuel_2d, cv2.COLORMAP_VIRIDIS)
        colored = colored.squeeze(axis=1)

        # Adjust to more natural green
        colored[:, 0] = (colored[:, 0] * 0.5).astype(np.uint8)  # Reduce blue
        colored[:, 2] = (colored[:, 2] * 0.3).astype(np.uint8)  # Reduce red

        frame[unburnt_mask] = colored

    def _render_burnt(self, frame: np.ndarray, heat: np.ndarray, fuel: np.ndarray) -> None:
        """Render dark burnt areas where fuel is depleted."""
        burnt_mask = (fuel <= self.thresholds.burnt_fuel) & (
            heat <= self.thresholds.fire)
        frame[burnt_mask] = self.colors.burnt

    def _render_fire(self, frame: np.ndarray, heat: np.ndarray, fuel: np.ndarray) -> None:
        """Render fire with thermal colormap."""
        fire_mask = (heat > self.thresholds.fire) & (fuel > 0.0)

        if not np.any(fire_mask):
            return

        # Convert heat to thermal colors
        heat_uint8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
        colored_heat = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_HOT)

        frame[fire_mask] = colored_heat[fire_mask]

    def _render_retardant(
        self,
        frame: np.ndarray,
        heat: np.ndarray,
        retardant: np.ndarray
    ) -> None:
        """Render blue retardant overlay."""
        suppressed_mask = (retardant > self.thresholds.retardant) & (
            heat <= self.thresholds.fire)

        if not np.any(suppressed_mask):
            return

        retardant_uint8 = np.clip(retardant * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(retardant_uint8, cv2.COLORMAP_OCEAN)

        frame[suppressed_mask] = colored[suppressed_mask]

    def get_fire_stats(self, heat: np.ndarray, fuel: np.ndarray) -> dict:
        """
        Calculate fire statistics for HUD display.

        Args:
            heat: (H, W) fire intensity
            fuel: (H, W) fuel density

        Returns:
            Dict with fire stats (intensity, area, burnt_area)
        """
        fire_mask = heat > self.thresholds.fire
        burnt_mask = fuel <= self.thresholds.burnt_fuel

        total_cells = heat.shape[0] * heat.shape[1]

        return {
            "total_intensity": float(heat.sum()),
            "fire_cells": int(fire_mask.sum()),
            "fire_area_pct": float(fire_mask.sum() / total_cells * 100),
            "burnt_cells": int(burnt_mask.sum()),
            "burnt_area_pct": float(burnt_mask.sum() / total_cells * 100),
            "max_heat": float(heat.max()),
            "mean_heat": float(heat[fire_mask].mean()) if fire_mask.any() else 0.0,
        }
