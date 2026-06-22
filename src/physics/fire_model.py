"""
Physics: Rothermel-style Fire Model.

Equation:
New_Heat = (Old_Heat * Decay) + (Diffusion_Influx * Spread_Scalar)
New_Fuel = Old_Fuel - (Combustion_Rate * New_Heat)
"""

import torch
import torch.nn.functional as F

from src.physics.wind import get_diffusion_kernel
from src.utils.hardware import SeedManager


class FireModel:
    """
    GPU-vectorized fire physics engine using reaction-diffusion dynamics.

    Dynamics:
    1. Diffusion: Heat moves from high to low (modified by wind).
    2. Ignition: Heat + Fuel -> Combustion.
    3. Burnout: Combustion consumes fuel until depleted.
    4. Cooling: Without fuel, heat dissipates rapidly.
    """
    # KERNEL_SIZE: 7x7 allows the fire to bleed into neighbours.
    KERNEL_SIZE = 7

    # # IGNITION_THRESHOLD: The 'Activation Energy'.
    # # A cell must reach 0.25 heat (25% max temp) from neighbours to start burning.
    # # Use the config value to override.
    # # s:200, t:200 = 0.25
    # # s50, t:100 = 0.3
    # IGNITION_THRESHOLD = 0.25

    # MIN_FUEL_THRESHOLD: The 'dregs' of the fuel.
    # Below 5% fuel, the fire flickers out.
    # s:200, t:200 = 0.05
    MIN_FUEL_THRESHOLD = 0.05

    # COMBUSTION_INTENSITY: The 'Heat Multiplier'.
    # When burning, how much heat is pumped out relative to current fuel?
    # Value > 1.0 creates an explosive front (heat > fuel).
    # s:200, t:200 = 1.5
    # s50, t:100 = 1.2
    COMBUSTION_INTENSITY = 1.2

    # FUEL_CONSUMPTION_SPEED: The 'Burn Duration'.
    # How much fuel is removed per step?
    # 0.05 = It takes roughly 20 steps to burn a cell completely (1.0 / 0.05).
    # s:200, t:200 = 0.05
    FUEL_CONSUMPTION_SPEED = 0.04

    # HEAT_RETENTION: The 'Cooling Rate'.
    # How much heat survives to the next tick?
    # 0.9 = Slow cooling (smouldering).
    # 0.8 = Fast cooling (creates a sharp, thin fire front).
    # s:200, t:200 = 0.7
    HEAT_RETENTION = 0.7

    WATER_EVAPORATION_RATE = 0.99  # Water lasts ~100 steps
    WATER_COOLING_POWER = 0.5      # How strongly water reduces heat
    WATER_DAMPING_EFFECT = 5.0     # High value = Wet fuel is very hard to burn

    def __init__(self, config: dict):
        self.cfg = config['physics']
        self.device = config['simulation']['device']

        self.IGNITION_THRESHOLD = self.cfg['ignition_threshold']

        self.rng = SeedManager.create_generator(self.device)

        self.kernel = get_diffusion_kernel(
            self.device, config['physics']['wind'], self.KERNEL_SIZE
        )
        self.kernel_size = int(self.kernel.shape[-1])
        if int(self.kernel.shape[-2]) != self.kernel_size:
            raise ValueError(
                f"Diffusion kernel must be square; got {tuple(self.kernel.shape)}"
            )
        self.padding = self.kernel_size // 2

    def propagate(self, state: torch.Tensor) -> None:
        """
        Advances the fire state by one time step.
        Applies diffusion, advection, and reaction processes.
        """
        heat = state[:, 0:1]
        fuel = state[:, 1:2]
        retardant = state[:, 2:3]

        # 1. DIFFUSION (The Push)
        # Heat spreads to neighbours. This is the "Sense" step.
        diffused_heat = F.conv2d(heat, self.kernel, padding=self.padding)
        if diffused_heat.shape[-2:] != heat.shape[-2:]:
            h, w = heat.shape[-2:]
            dh, dw = diffused_heat.shape[-2:]
            start_h = max(0, (dh - h) // 2)
            start_w = max(0, (dw - w) // 2)
            diffused_heat = diffused_heat[...,
                                          start_h:start_h + h, start_w:start_w + w]

        # 2. CHECK STATUS (The Logic)
        # To burn, you need:
        # A. Enough incoming heat to trigger ignition.
        # B. Enough fuel remaining.

        # EFFECT A: Shielding
        # Effective Fuel = Real Fuel / (1 + Wetness * Damping)
        # If Wetness is 1.0 and Damping is 5.0, Fuel is 6x harder to access.
        effective_fuel_access = fuel / \
            (1.0 + (retardant * self.WATER_DAMPING_EFFECT))

        has_fuel = effective_fuel_access > self.MIN_FUEL_THRESHOLD

        # EFFECT B: Ignition Threshold raised by wetness
        # It's harder to warm up wet ground.
        effective_ignition = self.IGNITION_THRESHOLD + (retardant * 0.5)

        is_ignited = diffused_heat > effective_ignition

        # Boolean mask: Who is actively burning this turn?
        active_fire_mask = (is_ignited & has_fuel).float()

        # 3. REACTION (The Output)
        # Generated Heat = Mask * Intensity * Current Fuel
        # If fuel is 1.0, we generate 1.8 heat. If fuel is 0.1, we generate 0.18 heat.
        generated_heat = active_fire_mask * \
            self.COMBUSTION_INTENSITY * effective_fuel_access

        # 4. UPDATE HEAT
        # We apply cooling (Retention) to the OLD heat, then add the NEW heat.
        # This creates the wave:
        # - Front: High Generated Heat + Incoming Diffusion
        # - Centre: No Generated Heat (no fuel) + Fast Cooling = Cold

        # EFFECT C: Direct Cooling
        # Subtract (Wetness * Power) from the heat
        cooling_factor = retardant * self.WATER_COOLING_POWER
        new_heat = (diffused_heat * self.HEAT_RETENTION) + \
            generated_heat - cooling_factor

        # 5. CONSUME FUEL & EVAPORATE WATER
        # We remove fuel wherever there is active fire.
        # We clamp to ensure we don't gain fuel or go negative.
        fuel_loss = active_fire_mask * self.FUEL_CONSUMPTION_SPEED
        new_fuel = fuel - fuel_loss

        new_retardant = retardant * self.WATER_EVAPORATION_RATE

        # 5. WRITE BACK & CLAMP
        state[:, 0:1] = torch.clamp(new_heat, 0.0, 1.0)
        state[:, 1:2] = torch.clamp(new_fuel, 0.0, 1.0)
        state[:, 2:3] = torch.clamp(new_retardant, 0.0, 1.0)

    def reset(self, seed: int):
        """Rewind the RNG to a specific universe ID."""
        SeedManager.seed_generator(self.rng, seed)
