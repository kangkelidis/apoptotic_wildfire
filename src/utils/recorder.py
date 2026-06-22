"""
Video Recorder: Captures simulation frames and encodes to video.

This class acts as a context manager that handles:
- Creating output directory structure
- Initializing compositor and encoder
- Capturing frames during simulation
- Finalizing video on exit
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.utils.rendering import FrameCompositor, VideoEncoder

if TYPE_CHECKING:
    from src.physics.manager import PhysicsManager
    from src.swarm.manager import SwarmManager


class Recorder:
    """
    Records simulation frames to video using the rendering module.

    Usage:
        with Recorder(config) as recorder:
            for step in range(max_steps):
                # ... run simulation step ...
                recorder.capture(physics, swarm, step, max_steps)
    """

    def __init__(
        self,
        config: dict,
        strategy_name: str,
        scenario_name: str,
        seed: int,
        video_output_dir: Optional[Path]
    ):
        """
        Initialize the recorder.

        Args:
            config: Configuration dictionary with simulation settings
            strategy_name: Name of strategy for video filename
            scenario_name: Name of scenario for video filename
            seed: Seed value for video filename
        """
        self.config = config
        self.strategy_name = strategy_name or "unknown_strategy"
        self.scenario_name = scenario_name or "unknown_scenario"
        self.seed = seed or 0

        self.render_w = config["simulation"]['render_width']
        self.render_h = config["simulation"]['render_height']

        runtime = config.get("runtime", {})
        video_tag = runtime.get("video_tag")
        suffix = f"_{video_tag}" if video_tag else ""
        video_filename = (
            f"{self.strategy_name}_{self.scenario_name}_seed{self.seed}{suffix}.mp4"
        )
        if video_output_dir is None:
            video_output_dir = Path("./outputs/videos")
        self.video_path = video_output_dir / video_filename

        # Initialize rendering components (created in __enter__)
        self.compositor = None
        self.encoder = None

    def __enter__(self):
        """
        Context manager entry - initialize compositor and encoder.
        """
        self.video_path.parent.mkdir(parents=True, exist_ok=True)

        self.compositor = FrameCompositor(
            width=self.render_w, height=self.render_h)
        self.encoder = VideoEncoder(
            str(self.video_path), width=self.render_w, height=self.render_h
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - finalize and close video encoder.
        """
        if self.encoder:
            self.encoder.close()
            print(f"Video saved to {self.video_path}")

    def capture(
        self,
        physics: "PhysicsManager",
        swarm: "SwarmManager",
        step_idx: int,
        max_steps: int,
    ) -> None:
        """
        Capture a single frame from the current simulation state.

        Args:
            physics: Physics manager with state tensor
            swarm: Swarm manager with drone positions and states
            step_idx: Current simulation step
            max_steps: Total number of steps
        """
        if not self.compositor or not self.encoder:
            return

        # Get physics state (use first batch element [0] for visualization)
        state = physics.state[0]  # (3, H, W)

        # Get drone data from swarm
        swarm_state = swarm.get_state()

        # Extract drone data from first batch element
        drone_positions = swarm_state["positions"][0]  # (N, 2)
        drone_states = swarm_state["states"][0]  # (N, 1)
        alive_mask = swarm_state["alive_mask"][0]  # (N, 1)

        # Get base positions from swarm
        base_positions = swarm.bases  # (4, 2) - same for all batches

        # Get wind from config
        wind = self.config['physics']['wind']

        # Render the frame
        frame = self.compositor.render(
            state_tensor=state,
            drone_positions=drone_positions,
            drone_states=drone_states,
            alive_mask=alive_mask,
            base_positions=base_positions,
            step=step_idx,
            total_steps=max_steps,
            wind=wind,
        )

        # Add frame to encoder
        self.encoder.add_frame(frame)
