import random

import numpy as np
import torch

"""
Intentions:
    - user provides a single master seed.
    - all RNG in the code is seeded from this master seed.
    - including terrain generation, fire model stochasticity, and any other randomness.
    - must work across CPU, CUDA, and MPS devices.
    - if we run batch experiments with multiple seeds, we can generate those seeds from the master seed.
"""


def setup_determinism(master_seed: int, device: torch.device) -> None:
    """
    Sets the global state for the entire process.
    Called once at the very start of main.py.
    """
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)

    if "cuda" in device.type:
        torch.cuda.manual_seed_all(master_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif "mps" in device.type:
        torch.mps.manual_seed(master_seed)


def detect_device(preferred: str = "auto") -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority: CUDA > MPS > CPU

    Args:
        preferred: Override auto-detection with specific device

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if preferred == 'cpu':
        return torch.device('cpu')

    available = None

    if torch.cuda.is_available():
        available = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available = torch.device("mps")
    else:
        available = torch.device("cpu")

    if preferred != "auto" and available.type != preferred:
        print(
            f"WARNING: {preferred.upper()} not available. Using device: {available.type.upper()}")

    return available


def generate_seeds(master_seed: int, n_seeds: int) -> list[int]:
    """
    Generate deterministic seed list from master seed.

    Args:
        master_seed: Base seed for the random generator
        n_seeds: Number of seeds to generate

    Returns:
        List of integer seeds
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be > 0")

    rng = np.random.default_rng(int(master_seed))
    return [int(rng.integers(0, 2**31)) for _ in range(n_seeds)]


class SeedManager:
    """
    Central Service for RNG management.
    Ensures consistent hardware-accelerated randomness.
    """
    @staticmethod
    def create_generator(device: torch.device) -> torch.Generator:
        """Allocates a stateful Generator object on the target device."""
        return torch.Generator(device=device)

    @staticmethod
    def seed_generator(gen: torch.Generator, seed: int):
        """Sets the state of an existing generator to a specific seed."""
        gen.manual_seed(seed)
