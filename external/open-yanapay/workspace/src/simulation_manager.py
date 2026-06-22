"""
This module, manages the parallel execution of simulations in NetLogo.

It provides functionality to run simulations and save the results.
It uses the pyNetLogo library, to configure simulation parameters and retrieve simulation results.
"""

import signal
import time
from io import StringIO
from multiprocessing import Pool, Process, Queue
from typing import Any, Optional

import pandas as pd  # type: ignore
import pyNetLogo
import requests
from pyNetLogo import NetLogoException
from src.retry_policy import (
    EXCEPTION_FAILURE_REASON,
    MAX_TICKS_FAILURE_REASON,
    NETLOGO_EXCEPTION_FAILURE_REASON,
    WALL_TIMEOUT_FAILURE_REASON,
)
from src.experiment_checkpoint import (
    finalize_checkpoint_files,
    load_experiment_manifest,
    update_experiment_manifest_window,
)
from src.server import BASE_URL
from src.simulation import NetLogoParams, Result, Scenario, Simulation
from tqdm import tqdm  # type: ignore
from utils.helper import (PBar, TimeoutException, get_available_cpus,
                          print_dots, setup_logger, timeout_handler)
from utils.netlogo_commands import *
from utils.paths import *
from utils.video_generation import generate_video

logger = setup_logger()


def reset_server_simulation_state(simulation_id: str) -> None:
    """
    Clears any cached server-side state for a simulation before a NetLogo rerun.
    """
    try:
        requests.post(BASE_URL + "/reset_simulation_state", json={"simulation_id": simulation_id})
    except Exception as exc:
        logger.warning("Failed to reset simulation state for %s: %s", simulation_id, exc)


def execute_commands(simulation_id: str,
                     netlogo_params: NetLogoParams,
                     netlogo_link: pyNetLogo.NetLogoLink) -> None:
    """
    Executes NetLogo commands to setup global model parameters in NetLogo.

    Each parameter is mapped to a NetLogo command and executed.
    The process is performed before the initail simulation setup.

    Args:
        simulation_id: The simulation id in the form of <scenario_indx>.
        netlogo_params: The parameters to be set in NetLogo.
        netlogo_link: The NetLogo link object.
    """
    commands = {
        SET_SIMULATION_ID_COMMAND: simulation_id,
        SET_NUM_OF_ROBOTS_COMMAND: netlogo_params.num_of_robots,
        SET_NUM_OF_PASSENGERS_COMMAND: netlogo_params.num_of_passengers,
        SET_NUM_OF_STAFF_COMMAND: netlogo_params.num_of_staff,
        SET_FALL_LENGTH_COMMAND: netlogo_params.fall_length,
        SET_FALL_CHANCE_COMMAND: netlogo_params.fall_chance,
        SET_ROBOT_PERSUASION_FACTOR: netlogo_params.robot_persuasion_factor,
        SET_PARTICIPATION_DECISION_INTERVAL: netlogo_params.participation_decision_interval,
        SET_PARTICIPATION_DEBUG_ENABLED: (
            "TRUE" if netlogo_params.participation_debug_enabled else "FALSE"
        ),
        SET_ROBOT_PARTICIPATION_STRATEGY: netlogo_params.robot_participation_strategy,
        SET_ROBOT_INTERFERENCE_DELAY: netlogo_params.robot_interference_delay,
        SET_FRAME_GENERATION_COMMAND: "TRUE" if netlogo_params.enable_video else "FALSE",
        SET_ROOM_ENVIRONMENT_TYPE: netlogo_params.room_type
    }

    try:
        for command, value in commands.items():
            netlogo_link.command(command.format(value))
            logger.debug(f"{simulation_id}: Executed {command.format(value)}")

    except Exception as e:
        logger.error(f"Commands failed for id: {simulation_id}. Exception: {e}")
    logger.debug(f"Commands executed for id: {simulation_id}")


def execute_runtime_commands(simulation_id: str,
                             netlogo_params: NetLogoParams,
                             netlogo_link: pyNetLogo.NetLogoLink) -> None:
    """
    Re-applies runtime settings that may be overwritten by NetLogo setup defaults.

    These settings do not affect environment construction, so applying them again after `setup`
    keeps interactive NetLogo defaults intact while ensuring batch runs use the configured values.
    """
    commands = {
        SET_PARTICIPATION_DECISION_INTERVAL: netlogo_params.participation_decision_interval,
        SET_PARTICIPATION_DEBUG_ENABLED: (
            "TRUE" if netlogo_params.participation_debug_enabled else "FALSE"
        ),
        SET_ROBOT_PARTICIPATION_STRATEGY: netlogo_params.robot_participation_strategy,
        SET_ROBOT_INTERFERENCE_DELAY: netlogo_params.robot_interference_delay,
    }

    try:
        for command, value in commands.items():
            netlogo_link.command(command.format(value))
            logger.debug(f"{simulation_id}: Re-applied {command.format(value)} after setup")
    except Exception as e:
        logger.error(f"Runtime commands failed for id: {simulation_id}. Exception: {e}")


def setup_simulation(simulation_id: str,
                     simulation_seed: int,
                     simulation_params: NetLogoParams,
                     netlogo_link: pyNetLogo.NetLogoLink) -> int:
    """
    Prepares the simulation.

    Clears the environment in NetLogo, executes the commands using the parameters provided
    and calls the set-up function of the NetLogo model.

    Args:
        simulation_id: The simulation id in the form of <scenario_indx>.
        simulation_seed: The seed for the simulation.
        simulation_params: The parameters to be set in NetLogo.
        netlogo_link: The NetLogo link object.

    Returns:
        current_seed: The seed used by netlogo for the simulation.
    """
    logger.debug(f'Setting up simulation for id: {simulation_id}.')
    reset_server_simulation_state(simulation_id)
    netlogo_link.command('clear')

    logger.debug(f'Cleared environment for id: {simulation_id}')
    execute_commands(simulation_id, simulation_params, netlogo_link)

    current_seed: int = int(netlogo_link.report(SEED_SIMULATION_REPORTER.format(simulation_seed)))
    logger.debug(f"Simulation {simulation_id},  Current seed: {current_seed}")

    netlogo_link.command('setup')
    execute_runtime_commands(simulation_id, simulation_params, netlogo_link)
    logger.debug(f"Setup completed for id: {simulation_id}")

    return current_seed


def initialise_netlogo_link(netlogo_model_path: str) -> pyNetLogo.NetLogoLink:
    """
    Initialises the NetLogo link and loads the model.

    Args:
        netlogo_model_path: The path to the NetLogo model.

    Returns:
        netlogo_link: The NetLogo link object.
    """
    logger.debug("Initialising NetLogo link from model path: %s", netlogo_model_path)
    netlogo_link: pyNetLogo.NetLogoLink = pyNetLogo.NetLogoLink(netlogo_home=NETLOGO_HOME,
                                                                netlogo_version=NETLOGO_VERSION,
                                                                gui=False)
    netlogo_link.load_model(netlogo_model_path)
    return netlogo_link


def _run_netlogo_model(netlogo_link: pyNetLogo.NetLogoLink, max_netlogo_ticks,
                       time_limit: int = 120) -> tuple[int | None, str | None]:
    """
    Runs the NetLogo model with a time limit and returns the number of ticks it took for the
    evacuation to finish. If the evacuation does not finish or exceeds the time limit,
    it returns None.

    Args:
        netlogo_link: The NetLogo link object.
        max_netlogo_ticks: The maximum number of ticks to run the model.
        time_limit: The time limit in seconds for running the model.

    Returns:
        A tuple of:
        - evacuation_ticks: The number of ticks it took for the evacuation to finish, or None.
        - failure_reason: A machine-readable reason when the run failed.
    """
    evacuation_ticks = None
    failure_reason = None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)
        ticks = 0
        while not netlogo_link.report(EVACUATION_FINISHED_REPORTER) and ticks < max_netlogo_ticks:
            netlogo_link.command('go')
            ticks += 1
        if ticks < max_netlogo_ticks:
            evacuation_ticks = ticks
        else:
            failure_reason = MAX_TICKS_FAILURE_REASON
        signal.alarm(0)
    except TimeoutException:
        logger.warning("Simulation timed out!")
        failure_reason = WALL_TIMEOUT_FAILURE_REASON
    except NetLogoException as e:
        logger.error(f"NetLogo exception: {e}")
        failure_reason = NETLOGO_EXCEPTION_FAILURE_REASON
    # ! cannot catch the exception in the java environment
    except BaseException as e:
        logger.error(f"Exception: {e}")
        failure_reason = EXCEPTION_FAILURE_REASON
    finally:
        signal.alarm(0)
    return evacuation_ticks, failure_reason


def run_simulation(simulation_id: str,
                   simulation_seed: int,
                   simulation_params: NetLogoParams,
                   netlogo_link: pyNetLogo.NetLogoLink,
                   simulation_timeout: int) -> Result:
    """
    Runs a single simulation with the provided parameters and returns the results.

    Sets up the simulation using Neltlogo commands,
    calculates the execution time, generates a video if applicable and
    creates and returns a Result object.

    Args:
        simulation_id: The simulation id in the form of <scenario_indx>.
        simulation_seed: The seed for the simulation. To be used in Netlogo to create a random seed.
        simulation_params: The parameters to be set in NetLogo.
        neltogo_link: The link to the NetLogo model.

    Returns:
        result: The result object containing the simulation results.
    """
    start_time = time.time()
    current_seed: int = setup_simulation(simulation_id, simulation_seed, simulation_params,
                                         netlogo_link)
    evacuation_ticks, failure_reason = _run_netlogo_model(netlogo_link,
                                                          simulation_params.max_netlogo_ticks,
                                                          simulation_timeout)

    def safe_report(reporter: str, default: int | float = 0) -> int | float:
        try:
            value = netlogo_link.report(reporter)
        except Exception as e:
            logger.warning("Reporter %s failed for %s: %s", reporter, simulation_id, e)
            return default
        if isinstance(default, int):
            return int(value)
        return float(value)

    def safe_report_str(reporter: str, default: str = "") -> str:
        try:
            value = netlogo_link.report(reporter)
        except Exception as e:
            logger.warning("Reporter %s failed for %s: %s", reporter, simulation_id, e)
            return default
        return str(value)

    def safe_report_csv(reporter: str, default: str = "") -> str:
        try:
            value = netlogo_link.report(reporter)
        except Exception as e:
            logger.warning("Reporter %s failed for %s: %s", reporter, simulation_id, e)
            return default
        return str(value)

    duplicate_robot_contacts = safe_report(DUPLICATE_ROBOT_CONTACTS_REPORTER, 0)
    busy_staff_distractions = safe_report(BUSY_STAFF_DISTRACTIONS_REPORTER, 0)
    interference_delay_total = safe_report(INTERFERENCE_DELAY_TOTAL_REPORTER, 0.0)
    interference_delay_applications = safe_report(INTERFERENCE_DELAY_APPLICATIONS_REPORTER, 0)
    duplicate_delay_applications = safe_report(DUPLICATE_DELAY_APPLICATIONS_REPORTER, 0)
    busy_staff_delay_applications = safe_report(BUSY_STAFF_DELAY_APPLICATIONS_REPORTER, 0)
    active_robot_ticks = safe_report(ACTIVE_ROBOT_TICKS_REPORTER, 0)
    reserve_robot_ticks = safe_report(RESERVE_ROBOT_TICKS_REPORTER, 0)
    task_committed_robot_ticks = safe_report(TASK_COMMITTED_ROBOT_TICKS_REPORTER, 0)
    sar_effort_samples = safe_report(SAR_EFFORT_SAMPLES_REPORTER, 0)
    runtime_robot_participation_strategy = safe_report_str(
        ROBOT_PARTICIPATION_STRATEGY_RUNTIME_REPORTER,
        "",
    )
    runtime_robot_interference_delay = safe_report(
        ROBOT_INTERFERENCE_DELAY_RUNTIME_REPORTER,
        0.0,
    )
    debug_tick_trace_csv = ""
    tick_trace_stats = {
        "mean_active_robots": 0.0,
        "peak_active_robots": 0,
        "mean_reserve_robots": 0.0,
        "peak_reserve_robots": 0,
        "mean_task_committed_robots": 0.0,
        "mean_unresolved_fallen": 0.0,
        "mean_attended_fallen": 0.0,
        "reserve_with_unresolved_ticks": 0,
    }
    if simulation_params.participation_debug_enabled:
        debug_tick_trace_csv = safe_report_csv(PARTICIPATION_DEBUG_TICK_TRACE_REPORTER, "")
        if debug_tick_trace_csv.strip():
            try:
                tick_df = pd.read_csv(StringIO(debug_tick_trace_csv))
                if not tick_df.empty:
                    tick_trace_stats = {
                        "mean_active_robots": float(tick_df["active_robots"].mean()),
                        "peak_active_robots": int(tick_df["active_robots"].max()),
                        "mean_reserve_robots": float(tick_df["reserve_robots"].mean()),
                        "peak_reserve_robots": int(tick_df["reserve_robots"].max()),
                        "mean_task_committed_robots": float(
                            tick_df["task_committed_robots"].mean()
                        ),
                        "mean_unresolved_fallen": float(
                            tick_df["unresolved_fallen"].mean()
                        ),
                        "mean_attended_fallen": float(
                            tick_df["attended_fallen"].mean()
                        ),
                        "reserve_with_unresolved_ticks": int(
                            ((tick_df["reserve_robots"] > 0)
                             & (tick_df["unresolved_fallen"] > 0)).sum()
                        ),
                    }
            except Exception as e:
                logger.warning(
                    "Failed to parse participation debug tick trace for %s: %s",
                    simulation_id,
                    e,
                )
    observed_ticks = evacuation_ticks if evacuation_ticks is not None else safe_report("ticks", 0)
    effort_denominator = sar_effort_samples if sar_effort_samples > 0 else observed_ticks
    if effort_denominator > 0:
        if tick_trace_stats["mean_active_robots"] == 0.0:
            tick_trace_stats["mean_active_robots"] = (
                float(active_robot_ticks) / float(effort_denominator)
            )
        if tick_trace_stats["mean_reserve_robots"] == 0.0:
            tick_trace_stats["mean_reserve_robots"] = (
                float(reserve_robot_ticks) / float(effort_denominator)
            )
        if tick_trace_stats["mean_task_committed_robots"] == 0.0:
            tick_trace_stats["mean_task_committed_robots"] = (
                float(task_committed_robot_ticks) / float(effort_denominator)
            )
    endtime = time.time()
    evacuation_time = round(endtime - start_time, 2)

    success: bool = evacuation_ticks is not None and \
        evacuation_ticks < simulation_params.max_netlogo_ticks
    return Result(netlogo_seed=current_seed,
                  evacuation_ticks=evacuation_ticks,
                  evacuation_time=evacuation_time,
                  success=success,
                  failure_reason=failure_reason,
                  duplicate_robot_contacts=int(duplicate_robot_contacts),
                  busy_staff_distractions=int(busy_staff_distractions),
                  interference_delay_total=float(interference_delay_total),
                  interference_delay_applications=int(interference_delay_applications),
                  duplicate_delay_applications=int(duplicate_delay_applications),
                  busy_staff_delay_applications=int(busy_staff_delay_applications),
                  active_robot_ticks=int(active_robot_ticks),
                  reserve_robot_ticks=int(reserve_robot_ticks),
                  task_committed_robot_ticks=int(task_committed_robot_ticks),
                  runtime_robot_participation_strategy=runtime_robot_participation_strategy,
                  runtime_robot_interference_delay=float(runtime_robot_interference_delay),
                  mean_active_robots=tick_trace_stats["mean_active_robots"],
                  peak_active_robots=tick_trace_stats["peak_active_robots"],
                  mean_reserve_robots=tick_trace_stats["mean_reserve_robots"],
                  peak_reserve_robots=tick_trace_stats["peak_reserve_robots"],
                  mean_task_committed_robots=tick_trace_stats["mean_task_committed_robots"],
                  mean_unresolved_fallen=tick_trace_stats["mean_unresolved_fallen"],
                  mean_attended_fallen=tick_trace_stats["mean_attended_fallen"],
                  reserve_with_unresolved_ticks=tick_trace_stats["reserve_with_unresolved_ticks"],
                  debug_tick_trace_csv=debug_tick_trace_csv)


def _build_server_result_payload(result: Result, simulation_id: str) -> dict[str, Any]:
    """
    Converts a result object into the subset of fields owned by the server.
    """
    server_owned_fields = {
        'debug_participation_trace',
        'participation_decision_count',
        'participation_stay_active_decisions',
        'participation_return_to_reserve_decisions',
        'participation_reactivation_decisions',
        'participation_reserve_wait_decisions',
    }
    data = {
        key: value for key, value in result.__dict__.items()
        if not key.startswith('robot_') and key not in server_owned_fields
    }
    data['simulation_id'] = simulation_id
    return data


def _build_failed_result(
    netlogo_seed: int | None,
    failure_reason: str,
    start_time: float,
) -> Result:
    """
    Builds a failure result for exceptions that escape the normal simulation path.
    """
    return Result(
        netlogo_seed=0 if netlogo_seed is None else netlogo_seed,
        evacuation_ticks=None,
        evacuation_time=round(time.time() - start_time, 2),
        success=False,
        failure_reason=failure_reason,
    )


def _recycle_netlogo_link(
    netlogo_link: pyNetLogo.NetLogoLink,
    netlogo_model_path: str,
) -> pyNetLogo.NetLogoLink:
    """
    Rebuilds a NetLogo link after a simulation-level crash so the rest of the batch can continue.
    """
    try:
        netlogo_link.kill_workspace()
    except Exception as exc:
        logger.warning("Failed to kill NetLogo workspace during recycle: %s", exc)
    return initialise_netlogo_link(netlogo_model_path)


def batch_processor(simulation_batch: list[dict[str, Any]], netlogo_model_path: str,
                    index: int, q: Queue, simulation_timeout: int) -> None:
    """
    Used to run a batch of simulations in a dedicated Process.

    It runs the simulations in the batch sequentially. It loads the netlogo model and
    runs each simulation before killing the link.

    Args:
        simulation_batch: A list of dictionaries containing the simulation id, seed and parameters.
        netlogo_model_path: The path to the NetLogo model.
        index: The index of the current batch.
        q: The queue to track the progress of the simulations.
    """
    netlogo_link = initialise_netlogo_link(netlogo_model_path)

    try:
        for simulation in simulation_batch:
            result: Result
            recycle_link = False
            start_time = time.time()
            seeded_netlogo_seed: int | None = None

            try:
                result = run_simulation(simulation['id'],
                                        simulation['seed'],
                                        simulation['params'],
                                        netlogo_link,
                                        simulation_timeout)
                seeded_netlogo_seed = result.netlogo_seed
                if result.failure_reason in {
                    NETLOGO_EXCEPTION_FAILURE_REASON,
                    EXCEPTION_FAILURE_REASON,
                }:
                    recycle_link = True
            except Exception as exc:
                logger.exception(
                    "Simulation %s crashed inside batch processor: %s",
                    simulation['id'],
                    exc,
                )
                result = _build_failed_result(
                    netlogo_seed=seeded_netlogo_seed,
                    failure_reason=EXCEPTION_FAILURE_REASON,
                    start_time=start_time,
                )
                recycle_link = True
            finally:
                # Update the queue to indicate that the simulation attempt has finished.
                q.get()

            data = _build_server_result_payload(result, simulation['id'])

            url = BASE_URL + "/put_results"
            try:
                response = requests.put(url, json=data)
                response.raise_for_status()
            except Exception as exc:
                logger.error(
                    "Failed to submit results for %s: %s",
                    simulation['id'],
                    exc,
                )

            logger.debug(f"Simulation id: {simulation['id']} finished. - Result: {result}.")

            if recycle_link:
                netlogo_link = _recycle_netlogo_link(netlogo_link, netlogo_model_path)

        logger.debug(f"Finished batch {index + 1}")
    finally:
        try:
            netlogo_link.kill_workspace()
        except Exception as exc:
            logger.warning("Failed to kill NetLogo workspace at batch end: %s", exc)


def build_batches(simulations: list[Simulation], num_cpus: int) -> list[list[dict[str, Any]]]:
    """
    Builds batches of simulations to be executed in parallel.

    It splits the list of simulations into batches and creates a list of dictionaries
    containing the simulation id, seed and parameters for each simulation in each batch.
    Objects are not passed by reference to the Process, but by value. This is why the
    simulations are converted to dictionaries.

    [[ {id: 1, seed: 123, params: {num_of_robots: 10, ...}}, ...], ...]

    Args:
        simulations: The simulations to be batched.
        num_cpus: The number of CPUs available.

    Returns:
        simulation_batches: The list of simulation batches.
    """
    # Initialize an empty list for each CPU
    simulation_batches: list[list] = [[] for _ in range(num_cpus)]
    simulations_dict = [{'id': sim.id, 'seed': sim.seed, 'params': sim.netlogo_params}
                        for sim in simulations]

    used_cores = set()
    # Assign simulations to CPUs
    for i, simulation in enumerate(simulations_dict):
        cpu_index = i % num_cpus
        simulation_batches[cpu_index].append(simulation)
        used_cores.add(cpu_index)
    # remove empty lists
    simulation_batches = [batch for batch in simulation_batches if batch]
    total_bathes_len = sum(len(batch) for batch in simulation_batches)
    logger.info(
        f"Total number of simulations to run: {total_bathes_len}. Total cores: {len(used_cores)}")
    return simulation_batches


def execute_parallel_simulations(
    simulations: list[Simulation],
    netlogo_model_path: str,
    simulation_timeout: int,
) -> None:
    """
    Executes the simulations in parallel using the available CPUs.

    It creates a Process for each core and runs (total simulations / number of cores) simulations
    in each, using the parameters from each Simulation object.

    Args:
        simulations: The simulations to be executed.
        netlogo_model_path: The path to the NetLogo model.
    """
    num_cpus = get_available_cpus()
    simulation_batches = build_batches(simulations, num_cpus)
    logger.info(f"Setting up {len(simulations)} Simulations")
    # Create a queue to track the progress of the simulations
    q = Queue()
    for _ in simulations:
        q.put(1)
    try:
        processes = []
        for index, batch in enumerate(simulation_batches):
            process = Process(target=batch_processor,
                              args=(batch, netlogo_model_path, index, q, simulation_timeout))
            processes.append(process)
            process.start()
            logger.debug(f"Started batch {index + 1} with {len(batch)} simulations, "
                         f"on processes: {process.pid}. {[sim['id'] for sim in batch]}")
        # used to track the progress of the simulations
        prev_size = len(simulations) + 1
        # used to print dots while waiting for the setting up to finish
        dot = 0
        pbar: PBar = PBar()
        while any(p.is_alive() for p in processes):
            size = q.qsize()

            # print dots while waiting for the setting up to finish
            if size == len(simulations):
                dot = print_dots(dot, len(simulations))

            # update the progress bar
            if size < len(simulations) and size != prev_size:
                prev_size = pbar.update(len(simulations), size, prev_size)

        for process in processes:
            process.join()

        q.close()
        pbar.close(len(simulations), size)
        logger.info(f"\nFinished {len(simulations) - size} simulations.")
    except Exception as e:
        logger.error(f"Exception in parallel simulation: {e}")


def build_simulation_pool(scenarios: list[Scenario]) -> list[Simulation]:
    """
    Combines all simulations from the provided scenarios into a list.

    Args:
        scenarios: The scenarios to be combined.

    Returns:
        simulations_pool: The list of all simulations.
    """
    simulations_pool = []
    for scenario in scenarios:
        logger.debug(
            f"Adding simulations for: {scenario.name}. List size {len(scenario.simulations)}")
        simulations_pool.extend(scenario.simulations)
    return simulations_pool


def video_worker(args: tuple) -> None:
    """
    Worker function to generate videos in parallel.

    Args:
        args: A tuple containing the simulation id and the path to save the video.
    """
    simulation_id, video_folder_path = args
    try:
        generate_video(simulation_id, video_folder_path)
    except Exception as e:
        logger.error(f"Error generating video for {simulation_id}. {e}")


def save_simulations_results(scenarios: list[Scenario], experiment_folder: dict) -> None:
    """
    Gather the results from each simulation and saves a csv for each scenario, in their
    respective folder under the current experiment folder.
    Then it combines all the results in a single dataFrame and saves it as a csv.

    Args:
        scenarios: The scenarios to get the results from.
        experiment_folder: A dictionary containing the paths in the experiment folder.
    """
    video_folder_path = experiment_folder['video']

    simulations_with_video: list[str] = []
    for scenario in scenarios:
        simulations_with_video.extend(scenario.simulation_ids_with_video)

    # Save the data
    try:
        finalize_checkpoint_files(scenarios, experiment_folder)
    except Exception as e:
        logger.error(f"Error saving results file: {e}")

    # save potential videos
    if simulations_with_video:
        logger.info(f"Generating videos for {simulations_with_video}")
        args_list = [(simulation_id, video_folder_path) for simulation_id in simulations_with_video]
        with Pool(processes=get_available_cpus()) as pool:
            pool.map(video_worker, args_list)


def log_execution_time(start_time: float, end_time: float) -> None:
    minutes, seconds = divmod(end_time - start_time, 60)
    logger.info(f"Experiment finished after {int(minutes)} minutes and {seconds:.2f} seconds")


def update_simulations_pool(
    simulations_pool,
    log_retry_message: bool = True,
) -> list[Simulation]:
    """
    Queries the server for the list of unfinished simulations
    and updates the next iteration of the simulations pool.

    Args:
        simulations_pool: The previous iteration of the simulations pool.

    Returns:
        new_pool: The updated simulations pool.
    """
    responses = requests.get(BASE_URL + "/get_unfinished_simulations")
    data = responses.json()
    unfinished_simulations = data['ids']
    unfinished_simulations = set(unfinished_simulations)

    new_pool = []
    for simulation in simulations_pool:
        if simulation.id in unfinished_simulations:
            new_pool.append(simulation)

    if log_retry_message and len(new_pool) != len(simulations_pool) and len(new_pool) != 0:
        logger.warning(
            "%s simulations remain unfinished after this pass. Retrying eligible runs...",
            len(unfinished_simulations),
        )

    return new_pool


def _filter_pool_to_window(
    simulations_pool: list[Simulation],
    window: dict[str, int] | None,
) -> list[Simulation]:
    if window is None:
        return simulations_pool
    window_start = int(window["start_index"])
    window_end = int(window["end_index_exclusive"])
    return [
        simulation
        for simulation in simulations_pool
        if window_start <= int(simulation.index) < window_end
    ]


def _select_sample_window(
    simulations_pool: list[Simulation],
    sample_window_size: int,
    experiment_folder: dict[str, str],
) -> tuple[list[Simulation], dict[str, int] | None]:
    unfinished_pool = update_simulations_pool(simulations_pool, log_retry_message=False)
    if not unfinished_pool:
        return [], None

    experiment_manifest = load_experiment_manifest(experiment_folder) or {}
    current_window = experiment_manifest.get("current_window")
    current_window_complete = bool(experiment_manifest.get("current_window_complete", True))
    if isinstance(current_window, dict) and not current_window_complete:
        pinned_window_pool = _filter_pool_to_window(unfinished_pool, current_window)
        if pinned_window_pool:
            return pinned_window_pool, {
                "start_index": int(current_window["start_index"]),
                "end_index_exclusive": int(current_window["end_index_exclusive"]),
            }

    window_start = min(
        (int(simulation.index) // sample_window_size) * sample_window_size
        for simulation in unfinished_pool
    )
    window_end = window_start + sample_window_size
    window_pool = [
        simulation
        for simulation in unfinished_pool
        if window_start <= int(simulation.index) < window_end
    ]
    return window_pool, {
        "start_index": int(window_start),
        "end_index_exclusive": int(window_end),
    }


def start_experiments(config: dict[str, Any],
                      scenarios: list[Scenario],
                      experiment_folder: dict[str, str],
                      sample_window_size: int | None = None) -> None:
    """
    Starts the simulations for the provided scenarios.

    It runs the simulations in parallel and saves the result in their respective Scenario objects.
    Then it combines all the results and saves them in a csv file.
    Finally, it returns the results for further analysis.

    Args:
        config: The configuration parameters for the simulations.
        scenarios: The scenarios to be executed.
        experiment_folder: A dictionary containing the paths in the experiment folder.
    """
    start_time = time.time()

    netlogo_model_path: str = config.get('netlogoModelPath', NETLOGO_FOLDER + "model.nlogo")
    simulation_timeout = int(config.get("maxSimulationTime", 120))
    simulations_pool = build_simulation_pool(scenarios)
    if sample_window_size is not None:
        current_pool, current_window = _select_sample_window(
            simulations_pool,
            sample_window_size,
            experiment_folder,
        )
        update_experiment_manifest_window(
            experiment_folder,
            current_window=current_window,
            current_window_complete=current_window is None,
        )
        if current_pool:
            execute_parallel_simulations(current_pool, netlogo_model_path, simulation_timeout)
            remaining_in_window = update_simulations_pool(current_pool)
            if remaining_in_window:
                update_experiment_manifest_window(
                    experiment_folder,
                    current_window=current_window,
                    current_window_complete=False,
                )
            else:
                update_experiment_manifest_window(
                    experiment_folder,
                    current_window=None,
                    last_completed_window=current_window,
                    current_window_complete=True,
                )
        else:
            update_experiment_manifest_window(
                experiment_folder,
                current_window=None,
                current_window_complete=True,
            )
    else:
        current_pool = update_simulations_pool(simulations_pool, log_retry_message=False)
        # Run the simulations until all are finished
        while current_pool:
            execute_parallel_simulations(current_pool, netlogo_model_path, simulation_timeout)
            current_pool = update_simulations_pool(simulations_pool)
        update_experiment_manifest_window(
            experiment_folder,
            current_window=None,
            current_window_complete=True,
        )

    save_simulations_results(scenarios, experiment_folder)
    end_time = time.time()
    log_execution_time(start_time, end_time)
