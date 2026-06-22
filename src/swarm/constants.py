"""
Swarm: State Constants.

Defines the discrete modes of the drone state machine.
"""
from enum import IntEnum


class DroneState(IntEnum):
    WAITING = 0       # At base, idle or reloading/recharging
    EXPLORING = 1     # In field, searching for fire fronts
    FIREFIGHTING = 2  # In field, engaged with fire (dropping payload)
    # In field, heading back to base (low resources/withdrawn)
    RETURNING = 3


# Communication range (as fraction of grid size)
# and fire detection range
# 0.2, that represents 10% of the map width [-1,1].
COMM_RANGE = 0.2
