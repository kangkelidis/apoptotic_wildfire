"""
Swarm: Controller (The Gated FSM).

Enforces physical rules and the commitment contract.
"""

import torch

from .constants import DroneState


class StateController:
    # 15% threshold to force return
    LOW_BATTERY = 0.15
    # 2% per step (50 steps life)
    BATTERY_DRAIN = 0.01
    # Timer desynchronization to avoid lockstep cycles.
    RELOAD_JITTER = 2.0
    DECISION_JITTER = 3.0

    def __init__(self, config: dict):
        self.device = config['simulation']['device']
        self.reload_time = config['swarm']['reload_time']
        self.decision_interval = config['swarm']['decision_interval']
        self.launch_commitment_decisions = int(
            config['swarm'].get('launch_commitment_decisions', 2)
        )
        congestion_cfg = config.get('swarm', {}).get('congestion_effects', {})
        self.congestion_effects_enabled = bool(
            congestion_cfg.get('enabled', False)
        )
        self.battery_drain_multiplier_at_full = float(
            congestion_cfg.get('battery_drain_multiplier_at_full', 2.0)
        )

        # Pre-allocate constants on GPU to avoid creation overhead in the loop
        self.STATE_WAITING = torch.tensor(
            DroneState.WAITING, device=self.device)
        self.STATE_EXPLORING = torch.tensor(
            DroneState.EXPLORING, device=self.device)
        self.STATE_RETURNING = torch.tensor(
            DroneState.RETURNING, device=self.device)
        self.STATE_FIREFIGHTING = torch.tensor(
            DroneState.FIREFIGHTING, device=self.device)

    def predict_physical_updates(
        self,
        *,
        state: torch.Tensor,
        timers: torch.Tensor,
        battery: torch.Tensor,
        payload: torch.Tensor,
        service_progress_mask: torch.Tensor | None = None,
        congestion_factor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mirror the controller's pre-transition physical updates so other callers
        can reason about eligibility with exactly the same logic.
        """
        new_timers = timers - 1.0
        is_waiting = (state == self.STATE_WAITING)

        service_required = (
            is_waiting
            & (timers > 0.0)
            & ((battery < 0.999) | (payload < 0.999))
        )
        if service_progress_mask is not None:
            service_progress_mask = service_progress_mask.bool() & service_required
            stalled_service = service_required & (~service_progress_mask)
            new_timers = torch.where(stalled_service, timers, new_timers)

        drain = torch.full_like(battery, self.BATTERY_DRAIN)
        if self.congestion_effects_enabled and congestion_factor is not None:
            congestion = torch.clamp(congestion_factor.float(), 0.0, 1.0)
            drain_multiplier = 1.0 + (
                congestion * (self.battery_drain_multiplier_at_full - 1.0)
            )
            in_field = (state != self.STATE_WAITING)
            drain = torch.where(in_field, drain * drain_multiplier, drain)

        new_bat = battery - drain
        if service_progress_mask is not None:
            new_bat = torch.where(service_required, battery, new_bat)

        service_finished = is_waiting & (new_timers <= 0)
        one = torch.ones_like(battery)
        new_bat = torch.where(service_finished, one, new_bat)
        new_pay = torch.where(service_finished, one, payload)

        return new_timers, new_bat, new_pay, service_finished

    def update_states(self,
                      state,
                      timers,
                      battery,
                      payload,
                      intent_mask,
                      launch_allowed_mask,
                      commitment_decisions_remaining,
                      at_base_mask,
                      fire_detected_mask,
                      service_progress_mask=None,
                      congestion_factor=None):
        """
        Full FSM Logic.
        args:
            intent_mask: 1=Go/Extend, 0=Return/Stay (from Strategy)
            at_base_mask: Boolean, True if distance_to_base < threshold (from Nav)
            fire_detected_mask: Boolean (from Sensors)
        """

        # ---------------------------------------------------------
        # 1. PHYSICAL UPDATES (Aging & Refueling)
        # ---------------------------------------------------------

        # Count down timers and apply battery/service updates.
        new_timers, new_bat, new_pay, _service_finished = self.predict_physical_updates(
            state=state,
            timers=timers,
            battery=battery,
            payload=payload,
            service_progress_mask=service_progress_mask,
            congestion_factor=congestion_factor,
        )

        # ---------------------------------------------------------
        # 2. ELIGIBILITY CHECKS
        # ---------------------------------------------------------

        # Who is allowed to listen to the Strategy?
        # 1. Must use the Strategy Timer (Timer <= 0)
        timer_done = (new_timers <= 0)

        # 2. Must be in a Valid Mode (Waiting or Exploring)
        # (We ignore Returning or Fighting drones - they are busy)
        waiting_mode = (state == self.STATE_WAITING) & launch_allowed_mask
        exploring_mode = (state == self.STATE_EXPLORING)
        valid_mode = waiting_mode | exploring_mode

        # 3. Must be Healthy (If Waiting, we must be fully reloaded)
        is_healthy = (new_bat > self.LOW_BATTERY) & (new_pay > 0)

        # Final Mask: "Agent X is ready for orders"
        is_eligible = timer_done & valid_mode & is_healthy

        # ---------------------------------------------------------
        # 3. STRATEGIC TRANSITIONS (Intent)
        # ---------------------------------------------------------

        new_state = state.clone()
        new_commitment = commitment_decisions_remaining.clone()
        committed_exploring = exploring_mode & is_eligible & (new_commitment > 0)
        effective_intent = torch.where(
            committed_exploring,
            torch.ones_like(intent_mask),
            intent_mask,
        )
        new_commitment = torch.where(
            committed_exploring,
            torch.clamp(new_commitment - 1, min=0),
            new_commitment,
        )

        # A. LAUNCH (Waiting -> Exploring)
        # If I am fully reloaded (Eligible + Waiting) and Strategy says "1" (Go)
        launch = waiting_mode & is_eligible & (effective_intent == 1)
        new_state = torch.where(launch, self.STATE_EXPLORING, new_state)

        # B. WITHDRAW (Exploring -> Returning)
        # If I am in the field (Eligible + Exploring) and Strategy says "0" (Return)
        withdraw = exploring_mode & is_eligible & (effective_intent == 0)
        new_state = torch.where(withdraw, self.STATE_RETURNING, new_state)

        # ---------------------------------------------------------
        # 4. REACTIVE TRANSITIONS (Overrides)
        # ---------------------------------------------------------

        # C. FAILSAFE (Low Battery / Empty Payload -> Returning)
        must_return = (new_bat < self.LOW_BATTERY) | (new_pay <= 0.0)

        # Don't trigger return if I am already at base (prevents flickering)
        force_return = must_return & (new_state != self.STATE_WAITING)
        new_state = torch.where(force_return, self.STATE_RETURNING, new_state)

        # D. DOCKING (Returning -> Waiting)
        docking = (new_state == self.STATE_RETURNING) & at_base_mask
        new_state = torch.where(docking, self.STATE_WAITING, new_state)

        # E. FIREFIGHTING (The "Instinct" Override)
        # If I see fire + I am healthy + I am not actively returning for survival
        can_fight = fire_detected_mask & (new_bat > self.LOW_BATTERY) & (
            new_pay > 0.0) & (~force_return)
        new_state = torch.where(can_fight, self.STATE_FIREFIGHTING, new_state)

        # F. FIRE OUT (Fighting -> Exploring)
        stop_fighting = (state == self.STATE_FIREFIGHTING) & (~can_fight)
        new_state = torch.where(stop_fighting, self.STATE_EXPLORING, new_state)

        reset_commitment = torch.zeros_like(new_commitment)
        launch_commitment = torch.full_like(
            new_commitment,
            self.launch_commitment_decisions,
        )
        new_commitment = torch.where(launch, launch_commitment, new_commitment)
        new_commitment = torch.where(
            (new_state == self.STATE_RETURNING) | (new_state == self.STATE_WAITING),
            reset_commitment,
            new_commitment,
        )

        # ---------------------------------------------------------
        # 5. TIMER RESET LOGIC
        # ---------------------------------------------------------

        # CASE A: DOCKING -> RELOAD
        reload_timer = torch.tensor(
            self.reload_time, device=self.device).float()
        if self.RELOAD_JITTER > 0.0:
            reload_timer = reload_timer + \
                (torch.rand_like(new_timers) * self.RELOAD_JITTER)
        new_timers = torch.where(docking, reload_timer, new_timers)

        # CASE B: COMMITTED TO FIELD -> INTERVAL
        # We ONLY reset the timer if the agent is actually EXPLORING.
        # - If it Launched: Reset.
        # - If it Extended: Reset.
        # - If it Stayed at Base: DO NOT RESET (Timer stays <= 0, asks again next frame).

        # Who ran the strategy?
        ran_strategy = is_eligible & (~docking)

        # Who is now exploring?
        is_committed = (new_state == self.STATE_EXPLORING)

        # Reset Logic
        should_reset_interval = ran_strategy & is_committed

        decision_timer = torch.tensor(
            self.decision_interval, device=self.device).float()
        if self.DECISION_JITTER > 0.0:
            decision_timer = decision_timer + \
                (torch.rand_like(new_timers) * self.DECISION_JITTER)
        new_timers = torch.where(
            should_reset_interval, decision_timer, new_timers)

        return new_state, new_timers, new_bat, new_pay, new_commitment
