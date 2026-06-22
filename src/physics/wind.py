"""
Physics: Advection-Diffusion Kernel Generator.

This module constructs the convolution kernels used to simulate the spread of heat
across a 2D grid. It uses a Gaussian distribution to define how heat from a
burning cell influences its neighbors.

To simulate wind, the center of the Gaussian distribution is shifted. For example,
if the wind blows North, the kernel weights are shifted South, effectively
"pulling" heat from southern neighbors into the current cell.
"""

import torch


def get_diffusion_kernel(
    device: torch.device,
    wind_vec: dict,
    kernel_size: int = 5,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    Generates a Gaussian diffusion kernel biased by a wind vector.

    Args:
        device: Torch device (CPU/GPU).
        wind_vec: Dict with 'x' and 'y' float components (-1.0 to 1.0).
        kernel_size: Odd integer (default 5).
        sigma: Spread standard deviation (default 1.0).

    Returns:
        Tensor shape (1, 1, kernel_size, kernel_size).
    """
    # Create coordinate grid centered at 0
    range_limit = kernel_size // 2
    xy = torch.arange(-range_limit, range_limit + 1, device=device).float()
    grid_y, grid_x = torch.meshgrid(xy, xy, indexing='ij')

    # Shift Gaussian center opposite to wind direction (Advection)
    shift_x = -wind_vec['x']
    shift_y = -wind_vec['y']

    # Calculate Gaussian weights
    dist_sq = (grid_x - shift_x)**2 + (grid_y - shift_y)**2
    weights = torch.exp(-dist_sq / (2 * sigma**2))

    # Zero out the center (Diffusion only considers neighbors)
    center_idx = range_limit
    weights[center_idx, center_idx] = 0.0

    # Normalize to conserve energy
    weights = weights / weights.sum()

    return weights.view(1, 1, kernel_size, kernel_size)
