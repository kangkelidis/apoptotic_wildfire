"""
Modular Rendering System for Wildfire Simulation.

This package provides a layered rendering architecture where each component
has a single responsibility:

Layers (rendered bottom to top):
    - TerrainRenderer: Fire, fuel, burnt areas, retardant
    - DroneRenderer: Drone markers with state-based colors, bases
    - HUDRenderer: Overlays with stats, step counter, wind indicator

Compositor:
    - FrameCompositor: Combines layers and manages render pipeline

Encoder:
    - VideoEncoder: FFmpeg pipe for H.264 encoding

Usage:
    from src.utils.rendering import FrameCompositor, VideoEncoder

    compositor = FrameCompositor(width=600, height=600)
    encoder = VideoEncoder("output.mp4", width=600, height=600)

    for step in range(num_steps):
        frame = compositor.render(
            state_tensor=model.state[0],
            drone_positions=positions[0],
            drone_states=states[0],
            step=step,
            wind=config['physics']['wind'],
            ...
        )
        encoder.add_frame(frame)

    encoder.close()
"""

from .compositor import CompositorConfig, FrameCompositor
from .drones import DroneRenderConfig, DroneRenderer
from .encoder import EncoderConfig, VideoEncoder
from .hud import HUDColors, HUDConfig, HUDRenderer
from .terrain import TerrainColors, TerrainRenderer, TerrainThresholds

__all__ = [
    # Core renderers
    "TerrainRenderer",
    "DroneRenderer",
    "HUDRenderer",
    "FrameCompositor",
    "VideoEncoder",
    # Config classes
    "TerrainColors",
    "TerrainThresholds",
    "DroneRenderConfig",
    "HUDConfig",
    "HUDColors",
    "CompositorConfig",
    "EncoderConfig",
]
