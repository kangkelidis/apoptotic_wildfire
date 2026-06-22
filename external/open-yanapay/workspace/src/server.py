from __future__ import annotations

import logging
import os
import signal
import sys
import traceback
from multiprocessing import Lock
from typing import Any

from flask import Flask, request  # type: ignore

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace_path)

from utils.cleanup import signal_handler

PORT = 5000
BASE_URL = f'http://localhost:{PORT}'

# List of scenario objects
SCENARIOS = []
# Tracks the simulation IDs that have not finished yet
UNFINISHED_SIMULATION_IDS: set[str] = set()
COMPLETED_SIMULATION_IDS: set[str] = set()
CURRENT_EXPERIMENT_FOLDER: dict[str, str] | None = None
SIMULATION_ATTEMPTS: dict[str, int] = {}

app = Flask(__name__)

lock = Lock()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes"}


def _build_participation_observation(data: dict[str, Any]):
    from src.participation_controller import ParticipationObservation

    return ParticipationObservation(
        simulation_id=data["simulation_id"],
        robot_id=int(data["robot_id"]),
        robot_rank=int(data["robot_rank"]),
        tick=int(data["tick"]),
        unresolved_nearby=float(data["unresolved_nearby"]),
        nearest_victim_distance=float(data["nearest_victim_distance"]),
        available_bystanders=float(data["available_bystanders"]),
        available_staff=float(data["available_staff"]),
        recent_success_rate=float(data["recent_success_rate"]),
        nearby_sars=float(data["nearby_sars"]),
        recent_failed_requests=float(data["recent_failed_requests"]),
        recent_staff_unavailable=float(data["recent_staff_unavailable"]),
        window_success_count=float(data.get("window_success_count", 0.0)),
        window_failed_request_count=float(data.get("window_failed_request_count", 0.0)),
        window_staff_unavailable_count=float(
            data.get("window_staff_unavailable_count", 0.0)
        ),
        task_committed=_to_bool(data["task_committed"]),
        reserve_flag=_to_bool(data["reserve_flag"]),
        active_flag=_to_bool(data["active_flag"]),
        distance_to_nearest_victim=float(data["distance_to_nearest_victim"]),
    )


def _format_batch_decisions(decisions: list[tuple[int, Any]]) -> str:
    rows = []
    for robot_id, decision in sorted(decisions, key=lambda item: item[0]):
        rows.append(
            f"[{int(robot_id)} "
            f"\"{decision.action}\" "
            f"{decision.score:.6f} "
            f"{decision.tau:.6f} "
            f"{decision.regret_ema:.6f}]"
        )
    return "[" + " ".join(rows) + "]"


@app.route('/get_unfinished_simulations', methods=['GET'])
def get_unfinished_simulations():
    """
    Returns a list of simulation IDs that have not finished yet.
    """
    return {"ids": list(UNFINISHED_SIMULATION_IDS)}, 200


@app.route('/reset_simulation_state', methods=['POST'])
def reset_simulation_state():
    """
    Reset cached per-simulation state before a fresh NetLogo run starts.
    """
    from src.experiment_checkpoint import set_persisted_attempt_count
    from src.simulation import Result, Scenario, Simulation

    global SIMULATION_ATTEMPTS
    data = request.json
    simulation_id = data['simulation_id']
    scenario_name = Simulation.get_scenario_name(simulation_id)

    with lock:
        scenario: Scenario = Scenario.find_by_name(scenario_name, SCENARIOS)
        simulation: Simulation = Simulation.find_by_id(scenario, simulation_id)
        attempt_count = SIMULATION_ATTEMPTS.get(simulation_id, 0) + 1
        SIMULATION_ATTEMPTS[simulation_id] = attempt_count
        if CURRENT_EXPERIMENT_FOLDER is not None:
            set_persisted_attempt_count(CURRENT_EXPERIMENT_FOLDER, simulation_id, attempt_count)
        simulation.result = Result(attempt_count=attempt_count)
        if scenario.participation_controller:
            scenario.participation_controller.cleanup_simulation(simulation_id)

    return "Simulation state reset", 200


@app.route('/put_results', methods=['PUT'])
def put_results():
    """
    Updates the results of a simulation in the corresponding simulation object.
    Called when the simulation ends.
    """
    from src.experiment_checkpoint import append_simulation_checkpoint, set_persisted_attempt_count
    from src.retry_policy import MAX_SIMULATION_ATTEMPTS, should_retry_simulation
    from src.simulation import Scenario, Simulation

    data = request.json
    simulation_id = data['simulation_id']

    global COMPLETED_SIMULATION_IDS
    global CURRENT_EXPERIMENT_FOLDER
    global SIMULATION_ATTEMPTS
    global UNFINISHED_SIMULATION_IDS
    with lock:
        scenario_name = Simulation.get_scenario_name(simulation_id)
        scenario: Scenario = Scenario.find_by_name(scenario_name, SCENARIOS)
        simulation: Simulation = Simulation.find_by_id(scenario, simulation_id)
        attempt_count = SIMULATION_ATTEMPTS.get(simulation_id, simulation.result.attempt_count or 1)
        data['attempt_count'] = attempt_count
        simulation.result.update(data)
        if scenario.participation_controller:
            scenario.participation_controller.cleanup_simulation(simulation_id)
        if should_retry_simulation(data.get('failure_reason'), attempt_count):
            if CURRENT_EXPERIMENT_FOLDER is not None:
                set_persisted_attempt_count(CURRENT_EXPERIMENT_FOLDER, simulation_id, attempt_count)
            app.logger.warning(
                "Retrying simulation %s after %s on attempt %s/%s",
                simulation_id,
                data.get('failure_reason'),
                attempt_count,
                MAX_SIMULATION_ATTEMPTS,
            )
            return "Retry scheduled", 200

        UNFINISHED_SIMULATION_IDS.discard(simulation_id)
        SIMULATION_ATTEMPTS.pop(simulation_id, None)
        if CURRENT_EXPERIMENT_FOLDER is not None:
            set_persisted_attempt_count(CURRENT_EXPERIMENT_FOLDER, simulation_id, None)
        if CURRENT_EXPERIMENT_FOLDER is not None and simulation_id not in COMPLETED_SIMULATION_IDS:
            append_simulation_checkpoint(scenario, simulation, CURRENT_EXPERIMENT_FOLDER)
            COMPLETED_SIMULATION_IDS.add(simulation_id)

    return "Results saved", 200


@app.route('/passenger_response', methods=['POST'])
def passenger_response():
    """
    Save the response of a passenger when asked to help in the corresponding simulation object.
    """
    from src.simulation import Simulation

    data = request.json
    simulation_id: str = data["simulation_id"]
    response: str = data["response"]

    with lock:
        simulation: Simulation = Simulation.find_by_id(SCENARIOS, simulation_id)
        simulation.add_response(response)

    return "Response saved", 200


@app.route('/on_survivor_contact', methods=['POST'])
def on_survivor_contact_handler():
    """
    Called by the NetLogo model when the robot makes contact with a fallen victim.
    Calls the get_robot_action method of the adaptation strategy to return the robot's action.
    """
    from src.adaptation_strategy import Survivor
    from src.simulation import Scenario, Simulation
    from utils.helper import setup_logger

    logger = setup_logger()
    data = request.json

    candidate_helper = Survivor(data["helper_gender"], data["helper_culture"], data["helper_age"])
    victim = Survivor(data["fallen_gender"], data["fallen_culture"], data["fallen_age"])
    helper_victim_distance = float(data["helper_fallen_distance"])
    first_responder_victim_distance = float(data["staff_fallen_distance"])
    simulation_id = data["simulation_id"]

    logger.debug(f'PUT /on_survivor_contact called by {simulation_id}')
    scenario_name = Simulation.get_scenario_name(simulation_id)
    scenario: Scenario = Scenario.find_by_name(scenario_name, SCENARIOS)
    simulation: Simulation = Simulation.find_by_id(scenario, simulation_id)

    if scenario.adaptation_strategy is None:
        raise ValueError("No adaptation strategy provided.")

    action = scenario.adaptation_strategy.get_robot_action(
        simulation_id, candidate_helper, victim,
        helper_victim_distance, first_responder_victim_distance)

    with lock:
        simulation.add_action(action)

    return action, 200


@app.route('/robot_participation_batch', methods=['POST'])
def robot_participation_batch():
    """
    Called by NetLogo once per tick with every SAR robot whose decision timer expired.
    Returns a NetLogo-readable list of decision rows:
    [[robot_id "action" score tau regret_ema] ...]
    """
    from src.simulation import Scenario, Simulation
    from utils.helper import setup_logger

    logger = setup_logger()
    data = request.json
    simulation_id = data["simulation_id"]
    scenario_name = Simulation.get_scenario_name(simulation_id)
    scenario: Scenario = Scenario.find_by_name(scenario_name, SCENARIOS)

    if scenario.participation_controller is None:
        raise ValueError("No participation controller provided.")

    observations = [
        _build_participation_observation(observation)
        for observation in data.get("observations", [])
    ]

    logger.debug(
        "POST /robot_participation_batch called by %s robots=%s tick=%s",
        simulation_id,
        len(observations),
        observations[0].tick if observations else "n/a"
    )

    with lock:
        simulation = Simulation.find_by_id(scenario, simulation_id)
        decisions = scenario.participation_controller.decide_batch(observations)
        keep_trace = bool(scenario.netlogo_params.participation_debug_enabled)
        decisions_by_robot = {int(robot_id): decision for robot_id, decision in decisions}
        trace_rows = []
        for observation in observations:
            decision = decisions_by_robot[int(observation.robot_id)]
            trace_rows.append(
                {
                    "tick": int(observation.tick),
                    "robot_id": int(observation.robot_id),
                    "robot_rank": int(observation.robot_rank),
                    "previous_active": bool(observation.active_flag),
                    "previous_reserve": bool(observation.reserve_flag),
                    "task_committed": bool(observation.task_committed),
                    "unresolved_nearby": float(observation.unresolved_nearby),
                    "nearby_sars": float(observation.nearby_sars),
                    "available_bystanders": float(observation.available_bystanders),
                    "available_staff": float(observation.available_staff),
                    "recent_failed_requests": float(observation.recent_failed_requests),
                    "recent_staff_unavailable": float(observation.recent_staff_unavailable),
                    "window_success_count": float(observation.window_success_count),
                    "window_failed_request_count": float(
                        observation.window_failed_request_count
                    ),
                    "window_staff_unavailable_count": float(
                        observation.window_staff_unavailable_count
                    ),
                    "action": decision.action,
                    "score": float(decision.score),
                    "tau": float(decision.tau),
                    "regret_ema": float(decision.regret_ema),
                    "reactivated_from_reserve": bool(
                        observation.reserve_flag
                        and not observation.task_committed
                        and decision.action == "stay-active"
                    ),
                    "stayed_in_reserve": bool(
                        observation.reserve_flag
                        and decision.action == "return-to-reserve"
                    ),
                }
            )
        simulation.add_participation_decisions(trace_rows, keep_trace=keep_trace)

    return _format_batch_decisions(decisions), 200


@app.route('/start', methods=['POST'])
def start():
    try:
        import json

        from src.experiment_checkpoint import (
            get_persisted_attempt_counts,
            initialize_experiment_manifest,
            initialize_results_checkpoint_file,
            restore_checkpointed_results,
        )
        from src.load_config import load_config, load_scenarios
        from src.simulation import Scenario
        from src.simulation_manager import start_experiments
        from utils.paths import CONFIG_FILE

        data = request.json
        experiment_folder: dict[str, str] = data["experiment_folder"]
        config_file_path = data.get("config_file_path", CONFIG_FILE)
        requested_sample_window_size = data.get("sample_window_size")
        requested_paper_mode = data.get("paper_mode")

        config: dict[str, Any] = load_config(config_file_path)
        scenarios: list[Scenario] = load_scenarios(config)

        global CURRENT_EXPERIMENT_FOLDER
        CURRENT_EXPERIMENT_FOLDER = experiment_folder
        global SCENARIOS
        SCENARIOS = scenarios
        experiment_manifest = initialize_experiment_manifest(
            scenarios,
            experiment_folder,
            requested_sample_window_size=requested_sample_window_size,
            requested_paper_mode=requested_paper_mode,
        )
        effective_paper_mode = experiment_manifest.get("paper_mode")
        if effective_paper_mode is not None:
            for scenario in scenarios:
                if scenario._get_num_robots_scalar() is not None and scenario._get_num_robots_scalar() > 0:
                    scenario.netlogo_params.participation_debug_enabled = True
        initialize_results_checkpoint_file(scenarios, experiment_folder)
        # Persist the validated config snapshot used for this run. This keeps plain
        # future resumes aligned with any explicit resume-time config extension.
        file_path = os.path.join(experiment_folder['path'], 'config.json')
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=5)

        global COMPLETED_SIMULATION_IDS
        COMPLETED_SIMULATION_IDS = restore_checkpointed_results(scenarios, experiment_folder)

        global SIMULATION_ATTEMPTS
        SIMULATION_ATTEMPTS = get_persisted_attempt_counts(experiment_folder)
        global UNFINISHED_SIMULATION_IDS
        UNFINISHED_SIMULATION_IDS = set()
        for scenario in scenarios:
            for simulation in scenario.simulations:
                if simulation.id not in COMPLETED_SIMULATION_IDS:
                    UNFINISHED_SIMULATION_IDS.add(simulation.id)

        # Run the experiments, and saves the results
        start_experiments(
            config,
            scenarios,
            experiment_folder,
            sample_window_size=experiment_manifest.get("sample_window_size"),
        )
    except BaseException as e:
        error_message = f"Error on server: {str(e)}\n{traceback.format_exc()}"
        return error_message, 500

    return 'ok', 200


def main():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    # cleanup the workspace when the server stops
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle Docker stop
    # start the server
    app.run(debug=False, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()
