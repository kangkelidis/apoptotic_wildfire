from __future__ import annotations

MAX_SIMULATION_ATTEMPTS = 2
MAX_TICKS_FAILURE_REASON = "max_ticks"
WALL_TIMEOUT_FAILURE_REASON = "wall_timeout"
NETLOGO_EXCEPTION_FAILURE_REASON = "netlogo_exception"
EXCEPTION_FAILURE_REASON = "exception"


def should_retry_simulation(
    failure_reason: str | None,
    attempt_count: int,
    max_attempts: int = MAX_SIMULATION_ATTEMPTS,
) -> bool:
    if attempt_count >= max_attempts:
        return False
    return failure_reason in {
        WALL_TIMEOUT_FAILURE_REASON,
        NETLOGO_EXCEPTION_FAILURE_REASON,
        EXCEPTION_FAILURE_REASON,
    }
