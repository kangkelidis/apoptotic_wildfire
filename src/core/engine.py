"""
Core: Simulation Engine.

This class is the Facade for the simulation.
1. .step(): The atomic step. Vectorized math. No side effects. Used by RL.
2. .run_experiment(): The evaluation loop. Orchestrates tqdm, metrics, and video.
"""


from typing import TYPE_CHECKING, Optional

import torch

from src.core.event_manager import EventManager
from src.core.metrics import ResearchMetrics
from src.physics.manager import PhysicsManager
from src.swarm.manager import SwarmManager
from src.utils.hardware import setup_determinism

if TYPE_CHECKING:
    from src.utils.outputs import OutputPaths


class SimulationEngine:
    """
    Facade for the simulation. Orchestrates interactions between Physics,
    Swarm, and Event systems.
    """

    def __init__(self, config: dict, physics: PhysicsManager,
                 swarm: SwarmManager, events: EventManager, metrics: ResearchMetrics,
                 strategy_name: str = "unknown", scenario_name: str = "unknown",
                 paths: Optional["OutputPaths"] = None):
        self.config = config
        self.device = torch.device(config['simulation']['device'])
        self.n_drones = config['swarm']['n_drones']
        self.batch_size = config['simulation']['batch_size']

        self.physics = physics
        self.swarm = swarm
        self.events = events
        self.metrics = metrics

        self.strategy_name = strategy_name
        self.scenario_name = scenario_name
        self.paths = paths

        self.max_steps = config['simulation']['max_steps']
        self.substeps = config['simulation']['physics_substeps']

        self._parallel_run_layout: list[dict] = []
        self._active_agents_per_batch: list[int] | None = None
        self.last_profile: dict[str, float] = {}

    def configure_parallel_runs(self, run_layout: list[dict] | None) -> None:
        """
        Configure packed parallel runs over the batch dimension.

        Args:
            run_layout: List of run specs with {start, end, ...metadata}.
        """
        if not run_layout:
            self._parallel_run_layout = []
            self._active_agents_per_batch = None
            self.metrics.set_run_layout(None)
            return

        self._parallel_run_layout = [dict(spec) for spec in run_layout]
        self.metrics.set_run_layout(self._parallel_run_layout)

        counts = [int(self.n_drones)] * int(self.batch_size)
        for spec in self._parallel_run_layout:
            start = int(spec["start"])
            end = int(spec["end"])
            if start < 0 or end > self.batch_size or end <= start:
                raise ValueError(
                    f"Invalid run layout range: {start}..{end} for "
                    f"batch_size={self.batch_size}"
                )
            n_active = int(spec.get("n_active_agents", self.n_drones))
            for row in range(start, end):
                counts[row] = n_active
        self._active_agents_per_batch = counts

    def step(self, step_idx, actor_obs_override: Optional[torch.Tensor] = None):
        """
        Executes one atomic simulation step (Vectorised).
        Pipeline: Events -> Perception -> Decision -> Action -> Physics
        """
        updated_mask, _ = self.events.update(
            step_idx,
            launch_allowed_mask=self.swarm.launch_allowed_mask.float()
        )
        if updated_mask is not None:
            self.swarm.launch_allowed_mask.copy_(updated_mask > 0.5)

        suppression_data = self.swarm.step(
            self.physics,
            step_idx,
            actor_obs_override=actor_obs_override
        )

        self.physics.apply_suppression(suppression_data)

        substeps = self.substeps
        for _ in range(substeps):
            self.physics.step()

    def run_experiment(self, seed: int | list[int], visualize: bool = False):
        """
        Runs a full evaluation loop with progress and recording.
        """
        import time
        from contextlib import nullcontext

        from tqdm import tqdm

        from ..utils.recorder import Recorder

        if self.paths is None and visualize:
            raise ValueError(
                "Output paths must be provided for visualization.")
        video_output_dir = self.paths.videos if self.paths else None
        strategy = getattr(self.swarm, "strategy", None)
        if strategy is not None and hasattr(strategy, "reset_debug_summary"):
            strategy.reset_debug_summary()

        t_run_start = time.perf_counter()
        self.reset(seed)
        t_after_reset = time.perf_counter()

        recorder_ctx = Recorder(
            self.config,
            strategy_name=self.strategy_name,
            scenario_name=self.scenario_name,
            seed=seed,
            video_output_dir=video_output_dir
        ) if visualize else nullcontext()

        step_compute_s = 0.0
        metrics_s = 0.0
        record_s = 0.0

        with recorder_ctx as recorder:
            start_time = time.perf_counter()

            for step_idx in tqdm(range(self.max_steps), desc=""):
                t0 = time.perf_counter()
                self.step(step_idx)
                t1 = time.perf_counter()
                self.metrics.observe(step_idx, self.physics, self.swarm)
                t2 = time.perf_counter()

                if recorder:
                    recorder.capture(self.physics, self.swarm,
                                     step_idx, self.max_steps)
                t3 = time.perf_counter()

                step_compute_s += (t1 - t0)
                metrics_s += (t2 - t1)
                record_s += (t3 - t2)

        duration = time.perf_counter() - start_time
        sps = self.max_steps / duration if duration > 0 else 0.0
        reset_s = t_after_reset - t_run_start
        run_total_s = time.perf_counter() - t_run_start
        other_loop_s = max(0.0, duration - step_compute_s - metrics_s - record_s)
        self.last_profile = {
            "run_total_s": float(run_total_s),
            "reset_s": float(reset_s),
            "loop_s": float(duration),
            "step_compute_s": float(step_compute_s),
            "metrics_s": float(metrics_s),
            "record_s": float(record_s),
            "other_loop_s": float(other_loop_s),
            "sps": float(sps),
        }
        if strategy is not None and hasattr(strategy, "get_debug_summary"):
            summary = strategy.get_debug_summary()
            if isinstance(summary, dict) and summary.get("impact_n_samples", 0.0) > 0:
                samples = int(round(float(summary["impact_n_samples"])))
                top = []
                suffix = "_flip_rate"
                for key, value in summary.items():
                    if not (key.startswith("impact_") and key.endswith(suffix)):
                        continue
                    feature = key[len("impact_"):-len(suffix)]
                    flip = float(value)
                    delta = float(summary.get(f"impact_{feature}_delta_go_abs_mean", 0.0))
                    top.append((feature, flip, delta))
                top.sort(key=lambda x: (x[1], x[2]), reverse=True)
                top = top[:3]
                details = " | ".join(
                    f"{feat}(|d_go|)={delta:.4f} {feat}(flip)={flip:.4f}"
                    for feat, flip, delta in top
                )
                print(
                    "MAPPO impact summary: "
                    f"samples={samples} "
                    f"{details}"
                )

        return self.metrics.get_final_result(sps, strategy=strategy)

    def reset(self, seed: int | list[int]):
        """
        Initializes a unique universe based on the seed.
        """
        if isinstance(seed, list):
            if not seed:
                raise ValueError("seed list cannot be empty.")
            setup_seed = int(seed[0])
        else:
            setup_seed = int(seed)

        setup_determinism(setup_seed, self.device)

        self.physics.reset(seed, run_layout=self._parallel_run_layout or None)
        self.events.reset(seed=setup_seed)
        initial_launch_allowed_mask = self.events.get_initial_launch_allowed_mask(
            self.batch_size, self.n_drones
        )
        if self._active_agents_per_batch is None:
            self.swarm.reset(
                self.n_drones,
                seed=setup_seed,
                initial_launch_allowed_mask=initial_launch_allowed_mask,
            )
        else:
            self.swarm.reset(
                self._active_agents_per_batch,
                seed=setup_seed,
                initial_launch_allowed_mask=initial_launch_allowed_mask,
            )
        # Held-back drones should not count as deployed/alive until released.
        self.swarm.alive_mask &= (initial_launch_allowed_mask > 0.5)
        self.swarm.initial_force_go_remaining = torch.where(
            self.swarm.alive_mask,
            self.swarm.initial_force_go_remaining,
            torch.zeros_like(self.swarm.initial_force_go_remaining),
        )
        self.swarm.capture_initial_alive_counts()
        self.metrics.reset(seed=seed)

    def get_global_metrics(self) -> dict:
        """
        Extracts high-level simulation metrics for Rewards/Logging.
        Returns dictionary of Tensors (B, 1).
        """
        # 1. Fuel State (Layer 1 is Fuel, Layer 0 is Heat)
        # Sum over Height(2) and Width(3) dimensions
        # Shape: (B, C, H, W) -> (B, 1)
        grid_size = self.physics.grid_size
        max_fuel = float(grid_size * grid_size)

        current_fuel = self.physics.state[:, 1].sum(dim=(1, 2)).unsqueeze(1)
        # 0.0 to 1.0 (1.0 = All Trees Alive)
        fuel_pct = current_fuel / max_fuel

        # 2. Swarm Activity
        # Count how many drones are NOT in Waiting State
        # Shape: (B, N) -> (B, 1)
        active_mask = (self.swarm.states !=
                       self.swarm.controller.STATE_WAITING)
        active_count = active_mask.float().sum(dim=1, keepdim=True)

        return {
            'fuel_pct': fuel_pct,           # (B, 1) Range [0, 1]
            'active_drones': active_count,  # (B, 1) Count
            'active_mask': active_mask      # (B, N) Boolean
        }


def create_engine(config: dict, strategy_name: str, scenario_name: str,
                  paths: Optional["OutputPaths"] = None) -> SimulationEngine:
    """
    Builder Function: The only place where Managers are instantiated.

    Args:
        config: Configuration dictionary
        strategy_name: Name of the drone strategy
        scenario_name: Name of the scenario
        paths: Output paths
    Returns:
        SimulationEngine: Fully constructed simulation engine
    """
    physics = PhysicsManager(config)
    swarm = SwarmManager(config, strategy_name=strategy_name)
    events = EventManager(config, physics, swarm, scenario_name)
    metrics = ResearchMetrics(config, strategy_name, scenario_name)

    return SimulationEngine(
        config, physics, swarm, events, metrics,
        strategy_name=strategy_name,
        scenario_name=scenario_name,
        paths=paths
    )
