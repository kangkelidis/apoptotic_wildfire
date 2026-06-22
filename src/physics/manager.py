"""
Physics: Manager that handles the Fire Grid and Terrain.
Maintains the master 4D Grid Tensor (B,C,H,W)
where B: Batch, C: Fire Channels (0: Heat, 1: Fuel, 2: Retardant), H: Height, W: Width.

"""

import torch
import torch.nn.functional as F

from src.physics.fire_model import FireModel
from src.physics.terrain import TerrainGenerator


class PhysicsManager:
    """
    Manages the lifecycle and state of the environmental tensors.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to allocate tensors on (cpu, cuda, mps).
    """

    def __init__(self, config: dict):
        self.config = config
        self.grid_size = config['simulation']['grid_size']
        self.batch_size = config['simulation']['batch_size']
        self.device = config['simulation']['device']
        self.splash_size = config['swarm']['splash_size']
        if int(self.splash_size) <= 0 or int(self.splash_size) % 2 == 0:
            raise ValueError(
                f"swarm.splash_size must be a positive odd integer, got {self.splash_size}"
            )
        saturation_cfg = config.get('physics', {}).get(
            'suppression_saturation', {}
        )
        self.suppression_saturation_enabled = bool(
            saturation_cfg.get('enabled', False)
        )
        self.suppression_saturation_cap = float(
            saturation_cfg.get('per_step_cap', 0.35)
        )

        self.state = torch.zeros(
            (self.batch_size, 3, self.grid_size, self.grid_size),
            device=self.device,
            dtype=torch.float32
        )

        self.fire_model = FireModel(config)
        self.terrain_gen = TerrainGenerator(config)

        # Create fixed coordinate grids [H, W]
        # We use indexing='ij' to match our [Channels, Y, X] tensor structure
        y_coords = torch.arange(self.grid_size, device=self.device)
        x_coords = torch.arange(self.grid_size, device=self.device)
        self.Y, self.X = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # PRE-ALLOCATE SOBEL KERNELS (No recalculation) for fire seeking
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        self._cached_gradient = None

        self.splash_kernel = self._generate_splash_kernel(
            size=self.splash_size,
        )

        # Calculate padding needed to keep grid size same
        # e.g., for size 7, padding is 3.
        self.padding = self.splash_size // 2

    def _generate_splash_kernel(self, size: int) -> torch.Tensor:
        """
        Creates a 2D Gaussian kernel dynamically.
        """
        sigma = self.splash_size / 4.0
        # Create a 1D coordinate vector: e.g. [-3, -2, -1, 0, 1, 2, 3]
        coords = torch.arange(
            size, device=self.device).float() - (size - 1) / 2

        # Gaussian Formula: exp(-x^2 / (2*sigma^2))
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))

        # we want the center to be 1.0 (Full Power).
        # Standard Gaussian sums to 1, which dilutes power.
        # So we divide by the MAX value to ensure the peak is 1.0.
        gauss_1d = gauss_1d / gauss_1d.max()

        # Outer Product to make it 2D (Matrix Multiplication)
        # (N, 1) * (1, N) = (N, N)
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]

        # Reshape for Conv2d: (Out=1, In=1, H, W)
        return kernel_2d.view(1, 1, size, size)

    def reset(
        self,
        universe_seed: int | list[int],
        run_layout: list[dict] | None = None
    ) -> None:
        """
        Builds a new parallel universes.
        Generates a unique fuel map for this seed.
        """
        # This removes all heat and retardant from the previous iteration.
        self.state[:, 0].fill_(0.0)
        self.state[:, 2].fill_(0.0)

        if isinstance(universe_seed, int):
            self.terrain_gen.reset(universe_seed)
            self.fire_model.reset(universe_seed + 1)
            self.state[:, 1:2] = self.terrain_gen.generate()
            return

        seeds = [int(s) for s in universe_seed]
        if not seeds:
            raise ValueError(
                "universe_seed list must contain at least one seed.")

        self.fire_model.reset(seeds[0] + 1)

        if run_layout is None:
            group_size = self.batch_size // len(seeds)
            if group_size <= 0:
                raise ValueError(
                    f"Invalid seed grouping: batch_size={self.batch_size}, "
                    f"n_seeds={len(seeds)}"
                )
            for i, seed in enumerate(seeds):
                start = i * group_size
                end = self.batch_size if i == len(
                    seeds) - 1 else (i + 1) * group_size
                span = max(0, end - start)
                if span == 0:
                    continue
                self.terrain_gen.reset(seed)
                self.state[start:end, 1:2] = self.terrain_gen.generate(
                    batch_size=span
                )
            return

        if len(run_layout) != len(seeds):
            raise ValueError(
                "run_layout length must match the number of seeds when using "
                "parallel reset."
            )

        for run, seed in zip(run_layout, seeds):
            start = int(run["start"])
            end = int(run["end"])
            if end <= start:
                continue
            span = end - start
            self.terrain_gen.reset(seed)
            self.state[start:end, 1:2] = self.terrain_gen.generate(
                batch_size=span)

    def step(self) -> None:
        """
        Advances all universes in the batch by one tick.
        - Diffuses heat (Wind).
        - Consumes fuel (Burning).
        - Applies suppression.
        """
        self.fire_model.propagate(self.state)
        # Invalidate cached gradient
        self._cached_gradient = None

    def apply_suppression(self, agent_data: torch.Tensor):
        """
        Converts agent drop commands into water on the grid (Channel 2).

        Args:
            agent_data: Tensor (Batch, N, 3) -> [x, y, intensity]
                        x, y are in range [-1.0, 1.0] normalized coordinates of the drop
        """
        batch_size = self.batch_size
        n_agents = self.config['swarm']['n_drones']

        x_raw = agent_data[:, :, 0]
        y_raw = agent_data[:, :, 1]
        intensity = agent_data[:, :, 2]

        # World [-1, 1] -> Normalized [0, 1]
        x_norm = (x_raw + 1.0) / 2.0
        y_norm = (y_raw + 1.0) / 2.0

        # Grid Index Calculation
        x_idx = (x_norm * (self.grid_size - 1)
                 ).long().clamp(0, self.grid_size - 1)
        y_idx = (y_norm * (self.grid_size - 1)
                 ).long().clamp(0, self.grid_size - 1)

        drop_grid = torch.zeros(
            (batch_size, 1, self.grid_size, self.grid_size),
            device=self.device
        )

        # Scatter drops onto the grid
        b_idx = torch.arange(batch_size, device=self.device).unsqueeze(
            1).expand(batch_size, n_agents)
        flat_indices = (b_idx * self.grid_size * self.grid_size) + \
            (y_idx * self.grid_size) + x_idx

        drop_grid_flat = drop_grid.view(-1)
        # We accumulate drops (if 2 drones hit same spot, power doubles)
        drop_grid_flat.scatter_add_(
            0, flat_indices.view(-1), intensity.view(-1))

        drop_grid = drop_grid_flat.view(
            batch_size, 1, self.grid_size, self.grid_size)

        # Uses the dynamic kernel and calculated padding
        splashed_water = F.conv2d(
            drop_grid,
            self.splash_kernel,
            padding=self.padding
        )
        if self.suppression_saturation_enabled:
            cap = max(self.suppression_saturation_cap, 1e-6)
            splashed_water = splashed_water / (1.0 + (splashed_water / cap))

        # 4. Update State
        self.state[:, 2:3] = torch.clamp(
            self.state[:, 2:3] + splashed_water, 0.0, 1.0)

    def initialize_fire(self, x=None, y=None, radius: int = 5) -> None:
        """
        Directly ignites a circular area of fire in all batches.

        Args:
            x: int - x coordinate in grid cells
            y: int - y coordinate in grid cells
            radius: int - fire radius in grid cells
        """
        # Default to center of grid if no coordinates provided
        if x is None:
            x = self.grid_size // 2
        if y is None:
            y = self.grid_size // 2

        # Resulting mask is [H, W]
        dist_sq = (self.X - x)**2 + (self.Y - y)**2
        mask = dist_sq < (radius**2)

        # self.grid is [B, 3, H, W].
        # We target Channel 0 (Heat) for ALL batches [:]
        # and apply the boolean mask to the spatial dimensions.
        self.state[:, 0, mask] = 1.0
        # reduce fuel slightly in the 'core' to represent the start of the burn
        self.state[:, 1, mask] *= 0.9

        # Fire shouldn't start on bare dirt.
        # We multiply by the fuel channel to ensure only burnable areas ignite.
        self.state[:, 0] *= (self.state[:, 1] > 0.1).float()

    def get_gradient(self) -> torch.Tensor:
        """
        Returns (B, 2, H, W) vector field of the fire.
        Calculates lazily only if the cache is empty.
        """
        if self._cached_gradient is not None:
            return self._cached_gradient

        # Only happens once per step
        heat = self.state[:, 0:1]

        grad_x = F.conv2d(heat, self.sobel_x, padding=1)
        grad_y = F.conv2d(heat, self.sobel_y, padding=1)

        self._cached_gradient = torch.cat([grad_x, grad_y], dim=1)

        return self._cached_gradient

    def get_state(self) -> dict:
        """
        The 'Getter' for the rest of the system.
        Returns a dictionary of references.

        Returns:
            dict: Keys are channel names, values are 4D tensors (B,C,H,W)
                    heat, fuel, retardant
        """
        return {
            "heat": self.state[:, 0:1],
            "fuel": self.state[:, 1:2],
            "retardant": self.state[:, 2:3]
        }
