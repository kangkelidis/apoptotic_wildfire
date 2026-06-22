"""
Physics: Vectorized Terrain Generation.

Generates fuel maps for the entire batch in parallel.
Uses convolutional smoothing to create clumps and firebreaks.
"""
import torch
import torch.nn.functional as F

from src.utils.hardware import SeedManager


class TerrainGenerator:
    """
    Vectorized procedural terrain generation.

    Pipeline:
    1. Seed: Random static (white noise).
    2. Clump: Box-blur creates organic blobs (low frequency).
    3. Sharpen: Contrast stretching forces values to 0.0 (Dirt) or 1.0 (Forest).
    """

    # --- TERRAIN GENERATION CONSTANTS ---

    # PATCH_SIZE: The size of the blur kernel (Must be Odd).
    # Larger = Larger, more connected islands of forest.
    # Smaller = Scattered, tiny bushes.
    PATCH_SIZE = 11

    # DENSITY_BIAS: Controls the ratio of Forest to Dirt.
    # 0.5 = Balanced (50% forest).
    # 0.49 = Sparse (More dirt).
    # 0.6 = Dense (More forest).
    DENSITY_BIAS = 0.499

    # EDGE_SHARPNESS: The contrast multiplier.
    # 10.0 = Sharp, distinct forest boundaries.
    # 1.0 = Foggy, gradual transitions (bad for distinct firebreaks).
    EDGE_SHARPNESS = 16.0

    # MIN_DENSITY_CUTOFF: The 'Tree Line'.
    # Any value below this is clamped to 0.0 (Dirt/Firebreak).
    # Higher values create wider gaps between forest patches.
    MIN_DENSITY_CUTOFF = 0.25

    def __init__(self, config: dict):
        self.device = config['simulation']['device']
        self.batch_size = config['simulation']['batch_size']
        self.grid_size = config['simulation']['grid_size']

        self.rng = SeedManager.create_generator(self.device)

        # Pre-calculate padding to keep map size consistent
        self.padding = self.PATCH_SIZE // 2

        # Create the smoothing kernel (Normalized Box Filter)
        # This acts as a "low-pass filter" to turn static into blobs
        self.kernel = torch.ones(
            (1, 1, self.PATCH_SIZE, self.PATCH_SIZE),
            device=self.device
        ) / (self.PATCH_SIZE ** 2)

    def reset(self, seed: int):
        """Rewind the RNG to a specific universe ID."""
        SeedManager.seed_generator(self.rng, seed)

    def generate(self, batch_size: int | None = None) -> torch.Tensor:
        """
        Generates B unique fuel maps in a single parallel operation.
        Returns: Fuel maps of shape (B, 1, H, H)
        """
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = int(batch_size)

        # 1. WHITE NOISE
        # Generate raw static
        noise = torch.rand(
            (batch_size, 1, self.grid_size, self.grid_size),
            generator=self.rng,
            device=self.device
        )

        # 2. CONVOLUTIONAL SMOOTHING (Clumping)
        # Turns white noise into "clouds" of values
        smoothed = F.conv2d(noise, self.kernel, padding=self.padding)

        # 3. CONTRAST STRETCHING (Sharpening)
        # Formula: (Pixel - Center) * Contrast + Center
        # This pushes grey values towards 0 or 1
        centered = smoothed - (1.0 - self.DENSITY_BIAS)
        sharpened = (centered * self.EDGE_SHARPNESS) + 0.5

        # 4. THRESHOLDING & CLAMPING
        # Apply the cutoff and ensure valid range [0, 1]
        fuel_map = torch.clamp(sharpened, 0.0, 1.0)

        # Hard cut to create clean firebreaks (Dirt)
        # Without this, you get "micro-fuel" (0.001) that lets fire leak across gaps
        fuel_map[fuel_map < self.MIN_DENSITY_CUTOFF] = 0.0

        # 5. BORDER MARGIN (10% on each side → 0 fuel)
        margin = max(1, int(self.grid_size * 0.05))
        fuel_map[:, :, :margin, :] = 0.0   # top
        fuel_map[:, :, -margin:, :] = 0.0  # bottom
        fuel_map[:, :, :, :margin] = 0.0   # left
        fuel_map[:, :, :, -margin:] = 0.0  # right

        return fuel_map
