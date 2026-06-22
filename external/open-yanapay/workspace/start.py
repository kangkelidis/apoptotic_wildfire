"""
This module serves as the entry point for running experiments.
"""

import argparse
import os
import shutil
import traceback

from src.results_analysis import perform_analysis
from utils.cleanup import cleanup_workspace
from utils.helper import setup_logger
from utils.paths import (EXPERIMENT_FOLDER_STRUCT, FRAMES_FOLDER, LOGS_FOLDER,
                         RESULTS_FOLDER, WORKSPACE_FOLDER)

logger = setup_logger()
WORKSPACE_RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")


def analyse_folder(folder_name: str) -> None:
    """
    Analyse the results of a given folder, in the results folder.

    Args:
        folder_name: The folder name to analyse.
    """
    logger.info(f"Analysing folder: {folder_name}")
    perform_analysis(_build_experiment_folder_struct(folder_name))


def _ensure_experiment_folder_struct(experiment_folder: dict[str, str]) -> None:
    for path in [RESULTS_FOLDER, LOGS_FOLDER, FRAMES_FOLDER, *experiment_folder.values()]:
        os.makedirs(path, exist_ok=True)


def _build_experiment_folder_struct(folder_name_or_path: str) -> dict[str, str]:
    experiment_folder_path = folder_name_or_path
    if not os.path.isabs(experiment_folder_path):
        default_results_path = os.path.join(RESULTS_FOLDER, folder_name_or_path)
        local_results_path = os.path.join(WORKSPACE_RESULTS_FOLDER, folder_name_or_path)
        experiment_folder_path = (
            default_results_path
            if os.path.exists(default_results_path)
            else local_results_path
        )
    experiment_folder_path = os.path.abspath(experiment_folder_path)
    return {
        "path": experiment_folder_path + "/",
        "data": os.path.join(experiment_folder_path, "data") + "/",
        "img": os.path.join(experiment_folder_path, "img") + "/",
        "video": os.path.join(experiment_folder_path, "video") + "/",
    }


def run_experiment(
    experiment_folder: dict[str, str] | None = None,
    config_file_path: str | None = None,
    sample_window_size: int | None = None,
    paper_mode: str | None = None,
    metric_variants: str | None = None,
) -> None:
    """Calls the server to start the experiment and analyses the results."""
    try:
        import requests
        from src.server import BASE_URL

        terminal_size = shutil.get_terminal_size(fallback=(80, 20))

        experiment_folder = experiment_folder or EXPERIMENT_FOLDER_STRUCT
        _ensure_experiment_folder_struct(experiment_folder)
        logger.info("-" * terminal_size.columns)
        logger.info("******* Starting Experiment *******")

        url = BASE_URL + "/start"
        payload = {"experiment_folder": experiment_folder}
        if config_file_path:
            payload["config_file_path"] = config_file_path
        if sample_window_size is not None:
            payload["sample_window_size"] = int(sample_window_size)
        if paper_mode is not None:
            payload["paper_mode"] = paper_mode
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            logger.critical(response.text)
        else:
            logger.info("-" * terminal_size.columns)
            logger.info("Starting Results Analysis...")
            perform_analysis(
                experiment_folder,
                paper_mode=paper_mode,
                metric_variants=metric_variants,
            )
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: Cleaning up workspace.")
    finally:
        cleanup_workspace(WORKSPACE_FOLDER)
        logger.info("Done!")


def main() -> None:
    """
    Main function. Parses the arguments and runs the experiment or analyses a given folder.
    """
    usage_text = (
        "Usage: %(prog)s [--analyse FOLDER] [--resume FOLDER] [--config PATH] "
        "[--sample-window-size N] [--paper-mode {standard,extended}] "
        "[--metric-variants {mean,median,p90,all}]"
    )
    parser = argparse.ArgumentParser(description='Run an experiment or analyse a given folder.',
                                     usage=usage_text)
    parser.add_argument('--analyse', nargs=1, type=str,
                        help='Analyse the results in a given folder.')
    parser.add_argument('--resume', nargs=1, type=str,
                        help='Resume a previous experiment folder from saved checkpoints.')
    parser.add_argument('--config', type=str,
                        help='Override config path for a new run or a resume extension.')
    parser.add_argument('--sample-window-size', type=int,
                        help='Run only one per-scenario sample window of size N, then exit.')
    parser.add_argument('--paper-mode', choices=['standard', 'extended'],
                        help='Generate evacuation paper plots in standard or extended mode.')
    parser.add_argument('--metric-variants', choices=['mean', 'median', 'p90', 'all'],
                        help='Metric variant(s) for evacuation paper plots.')
    args = parser.parse_args()

    if args.analyse:
        folder_name = args.analyse[0]
        perform_analysis(
            _build_experiment_folder_struct(folder_name),
            paper_mode=args.paper_mode,
            metric_variants=args.metric_variants,
        )
    elif args.resume:
        experiment_folder = _build_experiment_folder_struct(args.resume[0])
        run_experiment(
            experiment_folder=experiment_folder,
            config_file_path=(
                args.config
                if args.config
                else os.path.join(experiment_folder["path"], "config.json")
            ),
            sample_window_size=args.sample_window_size,
            paper_mode=args.paper_mode,
            metric_variants=args.metric_variants,
        )
    else:
        run_experiment(
            config_file_path=args.config,
            sample_window_size=args.sample_window_size,
            paper_mode=args.paper_mode,
            metric_variants=args.metric_variants,
        )


if __name__ == "__main__":
    main()
