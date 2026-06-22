"""
Dynamic Environment Manager: Scripted Event System.

This module provides a timeline-based event system that modifies simulation
parameters mid-run. Events are defined in YAML and triggered at specific steps.

Supported Event Types:
    - wind_shift: Change wind direction/magnitude
    - ignite_fire: Start new fires at specified or random locations
    - deploy_drones: Release held drones to become launch-eligible
    - physics_param: Generic physics parameter modification

Example YAML:
    events:
      - step: 500
        type: wind_shift
        wind: {x: -0.5, y: 0.8}
      - step: 1000
        type: ignite_fire
        position: random
        radius: 8
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import yaml

from src.physics.wind import get_diffusion_kernel

if TYPE_CHECKING:
    from src.physics.manager import PhysicsManager
    from src.swarm.manager import SwarmManager


class EventType(Enum):
    """Supported dynamic event types."""
    WIND_SHIFT = auto()
    IGNITE_FIRE = auto()
    DEPLOY_DRONES = auto()
    PHYSICS_PARAM = auto()


@dataclass
class SimulationEvent:
    """
    A single scheduled event in the simulation timeline.

    Attributes:
        step: Simulation step when this event triggers
        event_type: Type of event (from EventType enum)
        params: Event-specific parameters dictionary
        triggered: Whether this event has already fired
    """
    step: int
    event_type: EventType
    params: Dict[str, Any] = field(default_factory=dict)
    triggered: bool = False

    def __post_init__(self):
        """Convert string event type to enum if needed."""
        if isinstance(self.event_type, str):
            type_map = {
                'wind_shift': EventType.WIND_SHIFT,
                'ignite_fire': EventType.IGNITE_FIRE,
                'deploy_drones': EventType.DEPLOY_DRONES,
                'physics_param': EventType.PHYSICS_PARAM,
            }
            self.event_type = type_map.get(
                self.event_type.lower(), EventType.PHYSICS_PARAM
            )


class EventManager:
    """
    Dynamic Environment Manager for mid-simulation parameter changes.

    This orchestrator monitors the simulation step count and triggers
    scripted events from a YAML timeline. It can modify physics parameters,
    spawn fires, shift wind, and release held drones in deployment batches.

    Usage:
        manager = EventManager(config, fire_model, swarm_controller)
        manager.load_scenario("scenarios/wind_shift.yaml")

        for step in range(num_steps):
            # Apply any events scheduled for this step
            manager.update(step, launch_allowed_mask)

            # Normal simulation loop
            fire_model.step()
            ...

    Attributes:
        events: List of scheduled SimulationEvent objects
        fire_model: Reference to FireModel for physics modifications
        swarm: Reference to SwarmController for drone management
        reserved_drones: Indices of drones held back in the deployment queue
    """

    def __init__(
        self,
        config: dict,
        physics: 'PhysicsManager | None' = None,
        swarm: 'SwarmManager | None' = None,
        scenario_name: Optional[str] = None,
    ):
        """
        Initialize the Event Manager.

        Args:
            config: Simulation configuration dictionary
            fire_model: FireModel instance for physics modifications
            swarm_controller: SwarmController for drone management
        """
        self.config = config
        self.physics = physics
        self.swarm = swarm
        self.device = config['simulation']['device']
        self.grid_size = config['simulation']['grid_size']

        # Event timeline (sorted by step)
        self.events: List[SimulationEvent] = []
        self._scenario_data: Optional[dict] = None
        self._scenario_max_steps: int = int(config['simulation']['max_steps'])

        # Held-back drone tracking (kept as Python lists to avoid MPS indexing issues)
        self.reserved_indices: Optional[list[int]] = None
        self._reserved_indices_template: Optional[list[int]] = None
        self.initial_active_count: int = 0

        # RNG for random fire positions
        self.rng = torch.Generator(device=self.device)
        seed = config['simulation']['seed']
        self.rng.manual_seed(seed + 9999)  # Offset to decouple from fire RNG

        # Load scenario events
        scenario_name = scenario_name or "baseline"
        self.load_scenario(scenario_name)

    def load_scenario(self, scenario_name: str) -> None:
        """
        Load events from a YAML scenario file.

        Args:
            scenario_path: Path to the YAML file containing events
        """
        scenario_path = "config/scenarios.yaml"
        max_steps = self.config['simulation']['max_steps']
        with open(scenario_path, 'r') as f:
            all_scenarios = yaml.safe_load(f) or {}

        scenario = all_scenarios.get(scenario_name, None)
        if not scenario:
            raise ValueError(f"No scenarios found in {scenario_path}")

        self._scenario_data = scenario
        self._scenario_max_steps = int(max_steps)
        self._parse_scenario(self._scenario_data, max_steps=self._scenario_max_steps)

    def _randint_inclusive(self, lo: int, hi: int) -> int:
        """Sample integer in [lo, hi] using manager RNG."""
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return int(lo)
        x = torch.randint(
            int(lo), int(hi) + 1, (1,),
            generator=self.rng, device=self.device
        )
        return int(x.item())

    @staticmethod
    def _parse_range(value: Any) -> tuple[float, float] | None:
        """
        Parse range-like config values.

        Supports:
          - [min, max]
          - {min: ..., max: ...}
        """
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
        if isinstance(value, dict) and ("min" in value) and ("max" in value):
            return float(value["min"]), float(value["max"])
        return None

    def _resolve_event_step(self, evt: dict, max_steps: int) -> int:
        """
        Resolve event step from config.

        Supported fields (priority order):
          1) step
          2) step_percentage
          3) step_range
          4) step_percentage_range
        """
        step = 0
        if "step" in evt:
            step = int(evt["step"])
        elif "step_percentage" in evt:
            step = int(float(evt["step_percentage"]) * max_steps)
        elif "step_range" in evt:
            parsed = self._parse_range(evt["step_range"])
            if parsed is None:
                raise ValueError(f"Invalid step_range value: {evt['step_range']}")
            lo, hi = int(parsed[0]), int(parsed[1])
            step = self._randint_inclusive(lo, hi)
        elif "step_percentage_range" in evt:
            parsed = self._parse_range(evt["step_percentage_range"])
            if parsed is None:
                raise ValueError(
                    f"Invalid step_percentage_range value: {evt['step_percentage_range']}"
                )
            lo_p, hi_p = parsed
            lo = int(lo_p * max_steps)
            hi = int(hi_p * max_steps)
            step = self._randint_inclusive(lo, hi)

        # The run loop is range(max_steps), so latest valid trigger is max_steps-1.
        return max(0, min(int(step), max_steps - 1))

    def _parse_scenario(self, data: dict, max_steps: int) -> None:
        """
        Parse scenario data into SimulationEvent objects.

        Args:
            data: Scenario configuration dict with 'events' list
            max_steps: Total simulation steps (for resolving step_percentage)
        """
        self.events.clear()

        events_data = data.get('events', [])
        for evt in events_data:
            step = self._resolve_event_step(evt, max_steps=max_steps)

            event = SimulationEvent(
                step=step,
                event_type=evt['type'],
                params={k: v for k, v in evt.items() if k not in [
                    'step', 'step_percentage', 'step_range',
                    'step_percentage_range', 'type']}
            )
            self.events.append(event)

        # Sort by step for efficient processing
        self.events.sort(key=lambda e: e.step)

        # Parse reserved drones configuration
        reserve_config = data.get('reserved_drones', {})
        if reserve_config:
            self._setup_reserved_drones(reserve_config)

    def _sanitize_reserved(self, reserved: Optional[list[int]], n_drones: int) -> list[int]:
        """Validate reserved index list to avoid out-of-bounds indices."""
        if not reserved:
            return []
        safe = []
        for raw in reserved:
            try:
                idx = int(raw)
            except Exception:
                continue
            if 0 <= idx < n_drones:
                safe.append(idx)
        if len(safe) != len(reserved):
            print("⚠️  Reserved drone indices contained invalid values; dropping out-of-range entries.")
        return safe

    def _setup_reserved_drones(self, reserve_config: dict) -> None:
        """
        Configure drones to be held in reserve for later deployment.

        Args:
            reserve_config: Dict with 'percentage' (0.0-1.0) or 'count'
        """
        if self.swarm is None:
            return

        n_drones = self.swarm.n_drones

        if 'percentage' in reserve_config:
            reserve_count = int(n_drones * float(reserve_config['percentage']))
            reserve_count = max(0, min(reserve_count, n_drones))
        elif 'count' in reserve_config:
            reserve_count = max(0, min(int(reserve_config['count']), n_drones))
        else:
            return

        if reserve_count <= 0:
            return

        # Select last N drones as reserved (deterministic selection)
        start_idx = n_drones - reserve_count
        self.reserved_indices = list(range(start_idx, n_drones))
        self._reserved_indices_template = list(self.reserved_indices)
        self.initial_active_count = n_drones - reserve_count

    def get_initial_launch_allowed_mask(self, batch_size: int, n_drones: int) -> torch.Tensor:
        """
        Get the initial launch-allowed mask with reserved drones held at base.

        Returns:
            (B, N, 1) tensor with 1.0 for launch-eligible drones, 0.0 for held drones
        """
        mask = torch.ones((batch_size, n_drones, 1), device=self.device)

        if self.reserved_indices:
            # Hold reserved drones until a deploy event releases them.
            reserved_idx = self._sanitize_reserved(self.reserved_indices, n_drones)
            if reserved_idx:
                for idx_int in reserved_idx:
                    mask[:, idx_int, :] = 0.0

        return mask

    def update(
        self,
        step: int,
        launch_allowed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], List[str]]:
        """
        Check and trigger any events scheduled for the current step.

        Args:
            step: Current simulation step
            launch_allowed_mask: (B, N, 1) current launch-eligibility mask

        Returns:
            updated_mask: New launch-eligibility mask if drones were deployed, else None
            triggered_events: List of event type names that fired
        """
        triggered = []
        updated_mask = None
        current_mask = launch_allowed_mask

        for event in self.events:
            if event.triggered:
                continue
            if event.step > step:
                break  # Events are sorted, no more to check
            if event.step == step:
                result = self._trigger_event(event, current_mask)
                event.triggered = True
                triggered.append(event.event_type.name)

                # Track mask updates from deploy_drones
                if result is not None and event.event_type == EventType.DEPLOY_DRONES:
                    updated_mask = result
                    current_mask = result

        return updated_mask, triggered

    def _trigger_event(
        self,
        event: SimulationEvent,
        launch_allowed_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Execute a single event.

        Returns:
            Updated launch-eligibility mask if event modified it, else None
        """
        handlers = {
            EventType.WIND_SHIFT: self._handle_wind_shift,
            EventType.IGNITE_FIRE: self._handle_ignite_fire,
            EventType.DEPLOY_DRONES: self._handle_deploy_drones,
            EventType.PHYSICS_PARAM: self._handle_physics_param,
        }

        handler = handlers.get(event.event_type)
        if handler:
            return handler(event.params, launch_allowed_mask)
        return None

    def _handle_wind_shift(
        self,
        params: dict,
        launch_allowed_mask: Optional[torch.Tensor],
    ) -> None:
        """
        Shift wind direction/magnitude.

        Params:
            wind: {x: float, y: float} - new wind vector
            transition: str - 'instant' or 'gradual' (future: interpolation)
        """
        if self.physics is None:
            raise ValueError(
                "PhysicsManager is required for wind_shift event.")

        new_wind = params.get('wind', {})
        if not new_wind:
            return

        # Update config
        self.config['physics']['wind']['x'] = new_wind.get(
            'x', self.config['physics']['wind']['x']
        )
        self.config['physics']['wind']['y'] = new_wind.get(
            'y', self.config['physics']['wind']['y']
        )

        # Regenerate diffusion kernel
        kernel_size = int(getattr(
            self.physics.fire_model, "kernel_size",
            self.physics.fire_model.KERNEL_SIZE
        ))
        self.physics.fire_model.kernel = get_diffusion_kernel(
            self.device,
            self.config['physics']['wind'],
            kernel_size=kernel_size
        )
        self.physics.fire_model.kernel_size = int(
            self.physics.fire_model.kernel.shape[-1]
        )
        self.physics.fire_model.padding = self.physics.fire_model.kernel_size // 2

    def _handle_ignite_fire(
        self,
        params: dict,
        launch_allowed_mask: Optional[torch.Tensor],
    ) -> None:
        """
        Start a new fire at specified or random location.

        Params:
            position: 'random', 'center', or {x: int, y: int}
            radius: int - fire radius in grid cells
        """
        if self.physics is None:
            raise ValueError(
                "PhysicsManager is required for ignite_fire event.")

        radius = params.get('radius', 5)

        # Determine position
        pos_spec = params.get('position', 'random')

        if pos_spec == 'random':
            # Choose a random cell that currently has fuel.
            fuel_grid = self.physics.state[:, 1]
            fuel_any = fuel_grid.max(dim=0).values
            fuel_threshold = max(
                float(getattr(self.physics.fire_model, "MIN_FUEL_THRESHOLD", 0.0)),
                1e-3
            )
            candidates = torch.nonzero(
                fuel_any > fuel_threshold, as_tuple=False)
            if candidates.numel() > 0:
                pick = torch.randint(
                    0,
                    int(candidates.shape[0]),
                    (1,),
                    generator=self.rng,
                    device=self.device
                )
                row = int(pick.item())
                y = int(candidates[row, 0].item())
                x = int(candidates[row, 1].item())
            else:
                # Fallback if terrain is unexpectedly empty.
                x = self.grid_size // 2
                y = self.grid_size // 2
        elif pos_spec == 'center':
            x = self.grid_size // 2
            y = self.grid_size // 2
        elif isinstance(pos_spec, dict):
            x = pos_spec.get('x', self.grid_size // 2)
            y = pos_spec.get('y', self.grid_size // 2)
        else:
            x = self.grid_size // 2
            y = self.grid_size // 2

        # Ignite fire
        self.physics.initialize_fire(x=x, y=y, radius=radius)

    def _handle_deploy_drones(
        self,
        params: dict,
        launch_allowed_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Release reserved drones to become launch-eligible at base.

        Params:
            count: int - number of drones to deploy (default: all reserved)
            percentage: float - percentage of reserved to deploy (0.0-1.0)
        """
        if self.swarm is None or self.reserved_indices is None:
            return None
        if launch_allowed_mask is None:
            return None

        # Determine how many to deploy
        available = len(self.reserved_indices)
        if available == 0:
            return None

        if 'count' in params:
            deploy_count = max(0, min(int(params['count']), available))
        elif 'percentage' in params:
            deploy_count = int(available * float(params['percentage']))
            deploy_count = max(0, min(deploy_count, available))
        else:
            deploy_count = available  # Deploy all

        if deploy_count <= 0:
            return None

        # Select drones to deploy (from front of reserved list)
        deploy_idx = self.reserved_indices[:deploy_count]
        self.reserved_indices = self.reserved_indices[deploy_count:]
        deploy_idx = [
            int(i) for i in deploy_idx
            if 0 <= int(i) < self.swarm.n_drones
        ]
        if not deploy_idx:
            return None

        # Update launch-eligibility mask
        new_mask = launch_allowed_mask.clone()
        for idx_int in deploy_idx:
            new_mask[:, idx_int, :] = 1.0

        # Reset newly released drones so they make a fresh launch decision.
        if hasattr(self.swarm, 'states'):
            from ..swarm.constants import DroneState

            for idx_int in deploy_idx:
                self.swarm.alive_mask[:, idx_int, :] = True
                self.swarm.states[:, idx_int, :] = DroneState.WAITING
                self.swarm.batteries[:, idx_int, :] = 1.0
                self.swarm.payloads[:, idx_int, :] = 1.0
                self.swarm.timers[:, idx_int, :] = 0.0
                self.swarm.commitment_decisions_remaining[:, idx_int, :] = 0
                self.swarm.initial_force_go_remaining[:, idx_int, :] = (
                    self.swarm.initial_force_go_decisions
                )

        return new_mask

    def _handle_physics_param(
        self,
        params: dict,
        launch_allowed_mask: Optional[torch.Tensor],
    ) -> None:
        """
        Generic physics parameter modification.

        Params:
            param: str - parameter name (e.g., 'IGNITION_THRESHOLD')
            value: float - new value
            factor: float - multiply current value by this factor
        """
        if self.physics is None:
            raise ValueError(
                "PhysicsManager is required for physics_param event.")

        param_name = params.get('param')
        if not param_name:
            return

        if not hasattr(self.physics.fire_model, param_name):
            print(f"⚠️  Unknown physics param: {param_name}")
            return

        old_value = getattr(self.physics.fire_model, param_name)
        if 'value' in params:
            setattr(self.physics.fire_model, param_name, params['value'])
        elif 'factor' in params:
            setattr(self.physics.fire_model, param_name,
                    old_value * params['factor'])

    def reset(self, seed: int | None = None) -> None:
        """Reset all events to untriggered state and optionally reseed event RNG."""
        if seed is not None:
            self.rng.manual_seed(int(seed) + 9999)

        # Rebuild timeline on each reset so random-step events resample every episode.
        if self._scenario_data is not None:
            self._parse_scenario(self._scenario_data, max_steps=self._scenario_max_steps)
        else:
            for event in self.events:
                event.triggered = False
            if self._reserved_indices_template is not None:
                self.reserved_indices = list(self._reserved_indices_template)
            else:
                self.reserved_indices = None
