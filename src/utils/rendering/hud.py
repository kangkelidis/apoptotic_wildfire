"""
HUD Renderer: Heads-Up Display Overlay Layer.

Responsible for rendering informational overlays:
- Step counter
- Wind direction indicator (arrow + magnitude)
- Drone state counts
- Fire statistics
- Optional: FPS, simulation time, events

This is the top layer of the rendering stack.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HUDColors:
    """Color configuration for HUD elements (BGR format)."""
    text: Tuple[int, int, int] = (255, 255, 255)          # White
    text_shadow: Tuple[int, int, int] = (0, 0, 0)         # Black
    panel_bg: Tuple[int, int, int] = (40, 40, 40)         # Dark gray
    panel_border: Tuple[int, int, int] = (100, 100, 100)  # Light gray

    # Wind indicator
    wind_arrow: Tuple[int, int, int] = (200, 200, 255)   # Light pink/white
    wind_bg: Tuple[int, int, int] = (60, 60, 60)         # Dark gray

    # State indicator colors (match drone colors)
    state_waiting: Tuple[int, int, int] = (255, 200, 100)
    state_exploring: Tuple[int, int, int] = (255, 255, 255)
    state_firefighting: Tuple[int, int, int] = (0, 165, 255)
    state_returning: Tuple[int, int, int] = (255, 0, 255)


@dataclass
class HUDConfig:
    """Configuration for HUD rendering."""
    # Font settings
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    font_thickness: int = 1
    line_height: int = 20

    # Panel settings
    panel_padding: int = 8
    panel_alpha: float = 0.7  # Transparency for background panels
    panel_corner_radius: int = 4

    # Element visibility toggles
    show_step: bool = True
    show_wind: bool = True
    show_drone_stats: bool = True
    show_fire_stats: bool = True

    # Layout positions (relative to corners)
    # "tl" = top-left, "tr" = top-right, "bl" = bottom-left, "br" = bottom-right
    step_position: str = "tl"
    wind_position: str = "tr"
    drone_stats_position: str = "bl"
    fire_stats_position: str = "br"

    # Wind indicator settings
    wind_indicator_size: int = 50  # Diameter of wind compass
    wind_arrow_scale: float = 0.8  # Arrow length relative to indicator size

    # Colors
    colors: HUDColors = field(default_factory=HUDColors)


class HUDRenderer:
    """
    Renders heads-up display overlays on simulation frames.

    Provides visual information about:
    - Current simulation step
    - Wind direction and magnitude
    - Drone counts by state
    - Fire intensity and area
    """
    REFERENCE_RENDER_SIZE = 800

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[HUDConfig] = None,
    ):
        """
        Initialize HUD renderer.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            config: HUD configuration (uses defaults if None)
        """
        self.width = width
        self.height = height
        self.config = self._scaled_config(config or HUDConfig())

        # Margin from edges
        self.margin = self._scale_int(10, self._render_scale(), minimum=4)

    def _render_scale(self) -> float:
        return max(0.5, min(self.width, self.height) / self.REFERENCE_RENDER_SIZE)

    @staticmethod
    def _scale_int(value: int, scale: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(value) * scale)))

    @staticmethod
    def _scale_float(value: float, scale: float, minimum: float = 0.1) -> float:
        return max(minimum, float(value) * scale)

    def _scaled_config(self, config: HUDConfig) -> HUDConfig:
        scale = self._render_scale()
        return replace(
            config,
            font_scale=self._scale_float(config.font_scale, scale, minimum=0.3),
            font_thickness=self._scale_int(config.font_thickness, scale, minimum=1),
            line_height=self._scale_int(config.line_height, scale, minimum=10),
            panel_padding=self._scale_int(config.panel_padding, scale, minimum=4),
            panel_corner_radius=self._scale_int(config.panel_corner_radius, scale, minimum=0),
            wind_indicator_size=self._scale_int(config.wind_indicator_size, scale, minimum=20),
        )

    def render(
        self,
        frame: np.ndarray,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        wind: Optional[Dict[str, float]] = None,
        drone_stats: Optional[Dict[str, int]] = None,
        fire_stats: Optional[Dict[str, float]] = None,
        extra_info: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Render HUD elements onto frame (modifies in place).

        Args:
            frame: BGR frame to draw on (modified in place)
            step: Current simulation step number
            total_steps: Total steps (for progress display)
            wind: Wind vector dict with 'x' and 'y' components
            drone_stats: Drone counts dict from DroneRenderer.get_drone_stats()
            fire_stats: Fire statistics dict from TerrainRenderer.get_fire_stats()
            extra_info: Additional key-value pairs to display

        Returns:
            Modified frame (same reference as input)
        """
        cfg = self.config

        # Render step counter
        if cfg.show_step and step is not None:
            self._draw_step_counter(frame, step, total_steps)

        # Render wind indicator
        if cfg.show_wind and wind is not None:
            self._draw_wind_indicator(frame, wind)

        # Render drone statistics
        if cfg.show_drone_stats and drone_stats is not None:
            self._draw_drone_stats(frame, drone_stats)

        # Render fire statistics
        if cfg.show_fire_stats and fire_stats is not None:
            self._draw_fire_stats(frame, fire_stats)

        return frame

    def _get_position(self, position: str, width: int, height: int) -> Tuple[int, int]:
        """
        Get top-left corner coordinates for a panel.

        Args:
            position: Position code ("tl", "tr", "bl", "br")
            width: Panel width
            height: Panel height

        Returns:
            (x, y) coordinates for panel top-left corner
        """
        m = self.margin

        if position == "tl":
            return m, m
        elif position == "tr":
            return self.width - width - m, m
        elif position == "bl":
            return m, self.height - height - m
        elif position == "br":
            return self.width - width - m, self.height - height - m
        else:
            return m, m  # Default to top-left

    def _draw_panel_background(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw semi-transparent panel background."""
        cfg = self.config

        # Create overlay for alpha blending
        overlay = frame.copy()

        # Draw rounded rectangle (or regular if no radius)
        if cfg.panel_corner_radius > 0:
            # OpenCV doesn't have built-in rounded rect, so we use regular
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                cfg.colors.panel_bg,
                -1
            )
        else:
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                cfg.colors.panel_bg,
                -1
            )

        # Blend with alpha
        cv2.addWeighted(overlay, cfg.panel_alpha, frame,
                        1 - cfg.panel_alpha, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            cfg.colors.panel_border,
            1
        )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Optional[Tuple[int, int, int]] = None,
        scale: Optional[float] = None,
        shadow: bool = True,
    ) -> None:
        """Draw text with optional shadow for readability."""
        cfg = self.config
        color = color or cfg.colors.text
        scale = scale or cfg.font_scale

        if shadow:
            # Draw shadow offset by 1 pixel
            cv2.putText(
                frame, text, (x + 1, y + 1),
                cfg.font, scale, cfg.colors.text_shadow, cfg.font_thickness + 1
            )

        cv2.putText(
            frame, text, (x, y),
            cfg.font, scale, color, cfg.font_thickness
        )

    def _draw_step_counter(
        self,
        frame: np.ndarray,
        step: int,
        total_steps: Optional[int] = None,
    ) -> None:
        """Draw step counter in configured position."""
        cfg = self.config

        if total_steps:
            text = f"Step: {step:,}/{total_steps:,}"
        else:
            text = f"Step: {step:,}"

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cfg.font, cfg.font_scale, cfg.font_thickness
        )

        # Panel dimensions
        pad = cfg.panel_padding
        panel_w = text_w + pad * 2
        panel_h = text_h + baseline + pad * 2

        # Get position
        px, py = self._get_position(cfg.step_position, panel_w, panel_h)

        # Draw panel and text
        self._draw_panel_background(frame, px, py, panel_w, panel_h)
        self._draw_text(frame, text, px + pad, py + pad + text_h)

    def _draw_wind_indicator(
        self,
        frame: np.ndarray,
        wind: Dict[str, float],
    ) -> None:
        """Draw wind direction arrow with magnitude."""
        cfg = self.config
        colors = cfg.colors

        wind_x = wind.get('x', 0)
        wind_y = wind.get('y', 0)
        magnitude = math.sqrt(wind_x**2 + wind_y**2)

        # Panel dimensions
        size = cfg.wind_indicator_size
        pad = cfg.panel_padding
        panel_w = size + pad * 2
        panel_h = size + pad * 2 + cfg.line_height  # Extra space for text

        # Get position
        px, py = self._get_position(cfg.wind_position, panel_w, panel_h)

        # Draw panel background
        self._draw_panel_background(frame, px, py, panel_w, panel_h)

        # Draw compass circle
        center_x = px + pad + size // 2
        center_y = py + pad + size // 2
        radius = size // 2 - self._scale_int(2, self._render_scale(), minimum=1)
        border_thickness = self._scale_int(1, self._render_scale(), minimum=1)
        arrow_thickness = self._scale_int(2, self._render_scale(), minimum=1)

        cv2.circle(frame, (center_x, center_y), radius, colors.wind_bg, -1)
        cv2.circle(frame, (center_x, center_y), radius, colors.panel_border, border_thickness)

        # Draw wind arrow
        if magnitude > 0.01:
            # Normalize direction
            dir_x = wind_x / magnitude
            dir_y = wind_y / magnitude

            # Arrow endpoint (scale by magnitude, max at edge)
            arrow_len = min(magnitude, 1.0) * radius * cfg.wind_arrow_scale
            end_x = int(center_x + dir_x * arrow_len)
            end_y = int(center_y + dir_y * arrow_len)

            # Draw arrow line
            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (end_x, end_y),
                colors.wind_arrow,
                arrow_thickness,
                tipLength=0.3
            )
        else:
            # Draw small dot for no wind
            cv2.circle(
                frame,
                (center_x, center_y),
                self._scale_int(3, self._render_scale(), minimum=2),
                colors.wind_arrow,
                -1,
            )

        # Draw magnitude text below compass
        mag_text = f"{magnitude:.2f}"
        (tw, th), _ = cv2.getTextSize(mag_text, cfg.font, cfg.font_scale * 0.8, 1)
        text_x = px + pad + (size - tw) // 2
        text_y = py + pad + size + cfg.line_height - self._scale_int(4, self._render_scale(), minimum=2)
        self._draw_text(frame, mag_text, text_x, text_y,
                        scale=cfg.font_scale * 0.8)

    def _draw_drone_stats(
        self,
        frame: np.ndarray,
        stats: Dict[str, int],
    ) -> None:
        """Draw drone state counts in a compact 2x2 grid layout."""
        cfg = self.config
        colors = cfg.colors

        # Get counts - deployed = drones in action (not reserved)
        deployed = stats.get("deployed", stats.get("total", 0))
        active_pct = stats.get("active_pct", 0)

        # State counts (only for deployed drones)
        waiting = stats.get("waiting", 0)
        exploring = stats.get("exploring", 0)
        firefighting = stats.get("firefighting", 0)
        returning = stats.get("returning", 0)

        # Layout configuration
        pad = cfg.panel_padding
        line_h = cfg.line_height
        col_width = self._scale_int(55, self._render_scale(), minimum=24)

        # Header: "D: XXX (XX%)" - combined on one line
        # Color the percentage based on activity level
        if active_pct >= 70:
            pct_color = (100, 200, 100)  # Green
        elif active_pct >= 40:
            pct_color = (100, 200, 255)  # Yellow/orange
        else:
            pct_color = (100, 100, 255)  # Red

        # Panel dimensions: wider but shorter (2x2 grid)
        panel_w = col_width * 2 + pad * 3  # Two columns
        panel_h = line_h * 3 + pad * 2     # Header + 2 rows of states

        # Get position
        px, py = self._get_position(cfg.drone_stats_position, panel_w, panel_h)

        # Draw panel
        self._draw_panel_background(frame, px, py, panel_w, panel_h)

        # Row 1: Header with drone count and active % on same line
        y = py + pad + line_h - self._scale_int(5, self._render_scale(), minimum=2)
        header_text = f"D:{deployed}"
        self._draw_text(frame, header_text, px + pad, y)

        # Calculate where the count text ends to place percentage
        (hw, _), _ = cv2.getTextSize(header_text,
                                     cfg.font, cfg.font_scale, cfg.font_thickness)
        pct_text = f"({active_pct:.0f}%)"
        self._draw_text(
            frame,
            pct_text,
            px + pad + hw + self._scale_int(4, self._render_scale(), minimum=2),
            y,
            color=pct_color,
        )

        # Row 2: W and E (2x2 grid top row)
        y += line_h
        # Waiting (left column)
        self._draw_state_cell(frame, px + pad, y, "W",
                              waiting, colors.state_waiting)
        # Exploring (right column)
        self._draw_state_cell(frame, px + pad + col_width +
                              pad, y, "E", exploring, colors.state_exploring)

        # Row 3: F and R (2x2 grid bottom row)
        y += line_h
        # Firefighting (left column)
        self._draw_state_cell(frame, px + pad, y, "F",
                              firefighting, colors.state_firefighting)
        # Returning (right column)
        self._draw_state_cell(frame, px + pad + col_width +
                              pad, y, "R", returning, colors.state_returning)

    def _draw_state_cell(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        label: str,
        count: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a single state cell with color dot and count."""
        scale = self._render_scale()
        dot_offset = self._scale_int(5, scale, minimum=3)
        dot_radius = self._scale_int(4, scale, minimum=2)
        text_offset = self._scale_int(12, scale, minimum=8)
        # Draw color dot
        cv2.circle(frame, (x + dot_offset, y - dot_offset), dot_radius, color, -1)
        # Draw label and count
        text = f"{label}:{count:3d}"
        self._draw_text(frame, text, x + text_offset, y)

    def _draw_fire_stats(
        self,
        frame: np.ndarray,
        stats: Dict[str, float],
    ) -> None:
        """Draw fire statistics panel."""
        cfg = self.config

        # Build lines
        intensity = stats.get("total_intensity", 0)
        fire_pct = stats.get("fire_area_pct", 0)
        burnt_pct = stats.get("burnt_area_pct", 0)

        lines = [
            f"Fire: {intensity:,.0f}",
            f"Area: {fire_pct:.1f}%",
            f"Burnt: {burnt_pct:.1f}%",
        ]

        # Calculate panel dimensions
        pad = cfg.panel_padding
        line_h = cfg.line_height

        max_w = 0
        for text in lines:
            (tw, _), _ = cv2.getTextSize(
                text, cfg.font, cfg.font_scale, cfg.font_thickness)
            max_w = max(max_w, tw)

        panel_w = max_w + pad * 2
        panel_h = len(lines) * line_h + pad * 2

        # Get position
        px, py = self._get_position(cfg.fire_stats_position, panel_w, panel_h)

        # Draw panel
        self._draw_panel_background(frame, px, py, panel_w, panel_h)

        # Draw each line
        for i, text in enumerate(lines):
            y = py + pad + (i + 1) * line_h - self._scale_int(5, self._render_scale(), minimum=2)

            # Color fire intensity in orange/red
            if i == 0:  # Fire intensity line
                color = (0, 165, 255) if intensity > 100 else cfg.colors.text
            else:
                color = cfg.colors.text

            self._draw_text(frame, text, px + pad, y, color=color)
