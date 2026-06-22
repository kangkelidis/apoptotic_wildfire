from __future__ import annotations

import ast
import copy
import csv
import json
import os
from io import StringIO
from typing import Any

import pandas as pd  # type: ignore
from src.simulation import Result, Scenario, Simulation
from utils.paths import RESULTS_CSV_FILE_NAME

PARTICIPATION_DECISION_TRACE_FILE_NAME = "participation_decision_trace.csv"
PARTICIPATION_TICK_TRACE_FILE_NAME = "participation_tick_trace.csv"
SCENARIO_MANIFEST_FILE_NAME = "scenario_manifest.csv"
EXPERIMENT_MANIFEST_FILE_NAME = "experiment_manifest.json"
RESULTS_SCHEMA_VERSION = 2
CURRENT_WINDOW_COMPLETE_KEY = "current_window_complete"
ATTEMPT_COUNTS_KEY = "attempt_counts"

NUMERIC_RESULT_COLUMNS = {
    "sample_index",
    "netlogo_seed",
    "num_of_samples",
    "num_of_robots",
    "num_of_passengers",
    "num_of_staff",
    "fall_length",
    "fall_chance",
    "robot_persuasion_factor",
    "participation_decision_interval",
    "robot_interference_delay",
    "max_netlogo_ticks",
    "room_type",
    "param_seed",
    "evacuation_ticks",
    "evacuation_time",
    "attempt_count",
    "robot_contacts",
    "duplicate_robot_contacts",
    "busy_staff_distractions",
    "interference_delay_total",
    "interference_delay_applications",
    "duplicate_delay_applications",
    "busy_staff_delay_applications",
    "active_robot_ticks",
    "reserve_robot_ticks",
    "task_committed_robot_ticks",
    "runtime_robot_interference_delay",
    "participation_decision_count",
    "participation_stay_active_decisions",
    "participation_return_to_reserve_decisions",
    "participation_reactivation_decisions",
    "participation_reserve_wait_decisions",
    "mean_active_robots",
    "peak_active_robots",
    "mean_reserve_robots",
    "peak_reserve_robots",
    "mean_task_committed_robots",
    "mean_unresolved_fallen",
    "mean_attended_fallen",
    "reserve_with_unresolved_ticks",
    "participation_param_decisionInterval",
}
BOOL_RESULT_COLUMNS = {
    "participation_debug_enabled",
    "enable_video",
    "success",
    "participation_param_debug",
}
LIST_LIKE_RESULT_COLUMNS = {
    "robot_actions",
    "robot_responses",
}

CANONICAL_BASE_RESULT_COLUMNS = [
    "simulation_id",
    "scenario",
    "sample_index",
    "netlogo_seed",
    "evacuation_ticks",
    "evacuation_time",
    "failure_reason",
    "attempt_count",
    "robot_actions",
    "robot_responses",
    "robot_contacts",
    "duplicate_robot_contacts",
    "busy_staff_distractions",
    "interference_delay_total",
    "interference_delay_applications",
    "duplicate_delay_applications",
    "busy_staff_delay_applications",
    "active_robot_ticks",
    "reserve_robot_ticks",
    "task_committed_robot_ticks",
    "runtime_robot_participation_strategy",
    "runtime_robot_interference_delay",
    "participation_decision_count",
    "participation_stay_active_decisions",
    "participation_return_to_reserve_decisions",
    "participation_reactivation_decisions",
    "participation_reserve_wait_decisions",
    "mean_active_robots",
    "peak_active_robots",
    "mean_reserve_robots",
    "peak_reserve_robots",
    "mean_task_committed_robots",
    "mean_unresolved_fallen",
    "mean_attended_fallen",
    "reserve_with_unresolved_ticks",
    "success",
]

SCENARIO_MANIFEST_BASE_COLUMNS = [
    "scenario",
    "scenario_family",
    "description",
    "strategy",
    "participation_controller",
    "netlogo_seed",
    "num_of_samples",
    "num_of_robots",
    "num_of_passengers",
    "num_of_staff",
    "fall_length",
    "fall_chance",
    "robot_persuasion_factor",
    "robot_participation_strategy",
    "robot_participation_config",
    "participation_decision_interval",
    "participation_debug_enabled",
    "robot_interference_delay",
    "max_netlogo_ticks",
    "room_type",
    "enable_video",
    "param_seed",
    "participation_param_debug",
    "participation_param_decisionInterval",
]

SCENARIO_COMPATIBILITY_EXCLUDED_COLUMNS = {
    "scenario",
    "description",
    "num_of_robots",
}


def get_results_csv_path(experiment_folder: dict[str, str]) -> str:
    return os.path.join(experiment_folder["data"], RESULTS_CSV_FILE_NAME)


def get_scenario_manifest_path(experiment_folder: dict[str, str]) -> str:
    return os.path.join(experiment_folder["data"], SCENARIO_MANIFEST_FILE_NAME)


def get_experiment_manifest_path(experiment_folder: dict[str, str]) -> str:
    return os.path.join(experiment_folder["data"], EXPERIMENT_MANIFEST_FILE_NAME)


def get_participation_decision_trace_path(experiment_folder: dict[str, str]) -> str:
    return os.path.join(experiment_folder["data"], PARTICIPATION_DECISION_TRACE_FILE_NAME)


def get_participation_tick_trace_path(experiment_folder: dict[str, str]) -> str:
    return os.path.join(experiment_folder["data"], PARTICIPATION_TICK_TRACE_FILE_NAME)


def append_dataframe_row(dataframe: pd.DataFrame, csv_path: str) -> None:
    if dataframe.empty:
        return
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    dataframe.to_csv(csv_path, mode="a", header=write_header, index=False)


def normalize_results_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    for column in NUMERIC_RESULT_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in BOOL_RESULT_COLUMNS:
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].map(
            lambda value: value
            if pd.isna(value) or isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes"}
        )

    for column in LIST_LIKE_RESULT_COLUMNS:
        if column in normalized.columns:
            normalized[column] = normalized[column].fillna("[]")

    return normalized


def _extract_scenario_family(scenario_name: str) -> str:
    return scenario_name.split("@", 1)[0]


def _serialize_manifest_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _normalize_manifest_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(_serialize_manifest_value)
    return normalize_results_dataframe(normalized)


def get_canonical_result_columns(scenarios: list[Scenario] | None = None) -> list[str]:
    del scenarios
    return list(CANONICAL_BASE_RESULT_COLUMNS)


def build_scenario_manifest_row(scenario: Scenario) -> dict[str, Any]:
    params = copy.deepcopy(scenario.netlogo_params.__dict__)
    params["param_seed"] = params["seed"]
    del params["seed"]
    params = {key: _serialize_manifest_value(value) for key, value in params.items()}

    row: dict[str, Any] = {
        "scenario": scenario.name,
        "scenario_family": _extract_scenario_family(scenario.name),
        "description": scenario.description,
        "strategy": str(scenario.adaptation_strategy) if scenario.adaptation_strategy else None,
        "participation_controller": (
            str(scenario.participation_controller)
            if scenario.participation_controller is not None
            else None
        ),
        **params,
    }
    for key, value in scenario.participation_params.items():
        row[f"participation_param_{key}"] = _serialize_manifest_value(value)
    return row


def get_canonical_scenario_manifest_columns(scenarios: list[Scenario]) -> list[str]:
    discovered_columns: set[str] = set()
    for scenario in scenarios:
        discovered_columns.update(build_scenario_manifest_row(scenario).keys())

    canonical_columns = [
        column
        for column in SCENARIO_MANIFEST_BASE_COLUMNS
        if column in discovered_columns
    ]
    extra_columns = sorted(discovered_columns - set(canonical_columns))
    return canonical_columns + extra_columns


def build_simulation_result_row(scenario: Scenario, simulation: Simulation) -> dict[str, Any]:
    result = {
        key: value
        for key, value in simulation.result.__dict__.items()
        if not key.startswith("debug_")
    }
    return {
        "simulation_id": simulation.id,
        "scenario": scenario.name,
        "sample_index": simulation.index,
        **result,
    }


def _build_canonical_row(
    scenario: Scenario,
    simulation: Simulation,
    canonical_columns: list[str],
) -> dict[str, Any]:
    row = build_simulation_result_row(scenario, simulation)
    return {column: row.get(column, None) for column in canonical_columns}


def _read_csv_header(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    return list(pd.read_csv(csv_path, nrows=0).columns)


def _has_exact_schema(csv_path: str, canonical_columns: list[str]) -> bool:
    header = _read_csv_header(csv_path)
    if not header:
        return False
    return header == canonical_columns


def _write_canonical_header(csv_path: str, canonical_columns: list[str]) -> None:
    pd.DataFrame(columns=canonical_columns).to_csv(csv_path, index=False)


def _append_canonical_row(
    row: dict[str, Any],
    csv_path: str,
    canonical_columns: list[str],
) -> None:
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=canonical_columns,
            extrasaction="ignore",
            restval="",
        )
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, None) for column in canonical_columns})


def _load_scenario_manifest(experiment_folder: dict[str, str]) -> pd.DataFrame | None:
    scenario_manifest_path = get_scenario_manifest_path(experiment_folder)
    if not os.path.exists(scenario_manifest_path) or os.path.getsize(scenario_manifest_path) == 0:
        return None
    return _normalize_manifest_dataframe(pd.read_csv(scenario_manifest_path))


def _build_manifest_signature(row: pd.Series) -> tuple[tuple[str, str], ...]:
    signature_items: list[tuple[str, str]] = []
    for column, value in row.items():
        if column in SCENARIO_COMPATIBILITY_EXCLUDED_COLUMNS:
            continue
        if pd.isna(value):
            normalized_value = ""
        else:
            normalized_value = str(value)
        signature_items.append((str(column), normalized_value))
    return tuple(sorted(signature_items))


def _validate_scenario_manifest_extension(
    existing_manifest: pd.DataFrame,
    new_manifest: pd.DataFrame,
) -> None:
    if existing_manifest.empty or new_manifest.empty:
        return

    existing_by_scenario = existing_manifest.set_index("scenario", drop=False)
    existing_families = set(existing_manifest["scenario_family"].dropna().astype(str))

    for family in existing_families:
        family_rows = existing_manifest[existing_manifest["scenario_family"] == family]
        family_signatures = {_build_manifest_signature(row) for _, row in family_rows.iterrows()}
        if len(family_signatures) > 1:
            raise ValueError(
                f"Existing scenario family {family} contains incompatible static parameters."
            )

    for _, row in new_manifest.iterrows():
        scenario_name = str(row["scenario"])
        scenario_family = str(row["scenario_family"])
        row_signature = _build_manifest_signature(row)

        if scenario_name in existing_by_scenario.index:
            existing_signature = _build_manifest_signature(existing_by_scenario.loc[scenario_name])
            if row_signature != existing_signature:
                raise ValueError(
                    f"Scenario {scenario_name} does not match the existing experiment manifest."
                )
            continue

        if scenario_family not in existing_families:
            raise ValueError(
                f"Cannot append new scenario family {scenario_family}. "
                "Only new robot counts for existing scenario families are allowed."
            )

        existing_family_rows = existing_manifest[
            existing_manifest["scenario_family"] == scenario_family
        ]
        existing_signature = _build_manifest_signature(existing_family_rows.iloc[0])
        if row_signature != existing_signature:
            raise ValueError(
                f"Scenario {scenario_name} changes static parameters for family {scenario_family}. "
                "Only num_of_robots may differ when extending an experiment."
            )


def write_scenario_manifest_file(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
) -> None:
    scenario_manifest_path = get_scenario_manifest_path(experiment_folder)
    rows = [build_scenario_manifest_row(scenario) for scenario in scenarios]
    new_manifest = _normalize_manifest_dataframe(pd.DataFrame(rows))
    existing_manifest = _load_scenario_manifest(experiment_folder)

    if existing_manifest is not None and not existing_manifest.empty:
        _validate_scenario_manifest_extension(existing_manifest, new_manifest)
        merged_manifest = pd.concat([existing_manifest, new_manifest], ignore_index=True)
        merged_manifest = merged_manifest.drop_duplicates(subset=["scenario"], keep="last")
    else:
        merged_manifest = new_manifest

    canonical_columns = list(SCENARIO_MANIFEST_BASE_COLUMNS)
    extra_columns = sorted(set(merged_manifest.columns) - set(canonical_columns))
    canonical_columns = [column for column in canonical_columns if column in merged_manifest.columns]
    canonical_columns += extra_columns
    merged_manifest = merged_manifest.reindex(columns=canonical_columns)
    merged_manifest.to_csv(scenario_manifest_path, index=False, columns=canonical_columns)


def _coerce_sample_window_size(sample_window_size: int | None) -> int | None:
    if sample_window_size is None:
        return None
    coerced = int(sample_window_size)
    if coerced <= 0:
        raise ValueError("sample_window_size must be a positive integer when provided.")
    return coerced


def build_experiment_manifest(
    scenarios: list[Scenario],
    sample_window_size: int | None,
    paper_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "staged_execution": sample_window_size is not None,
        "sample_window_size": sample_window_size,
        "paper_mode": paper_mode,
        "enabled_scenarios": [scenario.name for scenario in scenarios],
        "samples_per_scenario": {
            scenario.name: int(scenario.netlogo_params.num_of_samples)
            for scenario in scenarios
        },
        "current_window": None,
        "last_completed_window": None,
        CURRENT_WINDOW_COMPLETE_KEY: True,
        ATTEMPT_COUNTS_KEY: {},
    }


def load_experiment_manifest(experiment_folder: dict[str, str]) -> dict[str, Any] | None:
    experiment_manifest_path = get_experiment_manifest_path(experiment_folder)
    if not os.path.exists(experiment_manifest_path):
        return None
    with open(experiment_manifest_path, "r") as manifest_file:
        return json.load(manifest_file)


def save_experiment_manifest(
    experiment_folder: dict[str, str],
    experiment_manifest: dict[str, Any],
) -> None:
    experiment_manifest_path = get_experiment_manifest_path(experiment_folder)
    with open(experiment_manifest_path, "w") as manifest_file:
        json.dump(experiment_manifest, manifest_file, indent=2, sort_keys=True)


def initialize_experiment_manifest(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
    requested_sample_window_size: int | None = None,
    requested_paper_mode: str | None = None,
) -> dict[str, Any]:
    requested_sample_window_size = _coerce_sample_window_size(requested_sample_window_size)
    existing_manifest = load_experiment_manifest(experiment_folder)
    if existing_manifest is None:
        experiment_manifest = build_experiment_manifest(
            scenarios,
            requested_sample_window_size,
            requested_paper_mode,
        )
        save_experiment_manifest(experiment_folder, experiment_manifest)
        return experiment_manifest

    stored_sample_window_size = existing_manifest.get("sample_window_size")
    stored_sample_window_size = _coerce_sample_window_size(stored_sample_window_size)
    stored_paper_mode = existing_manifest.get("paper_mode")
    if requested_sample_window_size is not None and \
            requested_sample_window_size != stored_sample_window_size:
        raise ValueError(
            "Conflicting sample_window_size for resumed experiment. "
            f"Stored value: {stored_sample_window_size}, requested: {requested_sample_window_size}"
        )
    if requested_paper_mode is not None and requested_paper_mode != stored_paper_mode:
        raise ValueError(
            "Conflicting paper_mode for resumed experiment. "
            f"Stored value: {stored_paper_mode}, requested: {requested_paper_mode}"
        )

    enabled_scenarios = [scenario.name for scenario in scenarios]
    samples_per_scenario = {
        scenario.name: int(scenario.netlogo_params.num_of_samples)
        for scenario in scenarios
    }
    existing_manifest["results_schema_version"] = RESULTS_SCHEMA_VERSION
    existing_manifest["staged_execution"] = stored_sample_window_size is not None
    existing_manifest["sample_window_size"] = stored_sample_window_size
    existing_manifest["paper_mode"] = stored_paper_mode
    existing_manifest["enabled_scenarios"] = enabled_scenarios
    existing_manifest["samples_per_scenario"] = samples_per_scenario
    existing_manifest.setdefault("current_window", None)
    existing_manifest.setdefault("last_completed_window", None)
    existing_manifest.setdefault(CURRENT_WINDOW_COMPLETE_KEY, True)
    existing_manifest.setdefault(ATTEMPT_COUNTS_KEY, {})
    save_experiment_manifest(experiment_folder, existing_manifest)
    return existing_manifest


def update_experiment_manifest_window(
    experiment_folder: dict[str, str],
    current_window: dict[str, int] | None,
    last_completed_window: dict[str, int] | None = None,
    current_window_complete: bool | None = None,
) -> None:
    experiment_manifest = load_experiment_manifest(experiment_folder)
    if experiment_manifest is None:
        return
    experiment_manifest["current_window"] = current_window
    if last_completed_window is not None:
        experiment_manifest["last_completed_window"] = last_completed_window
    if current_window_complete is not None:
        experiment_manifest[CURRENT_WINDOW_COMPLETE_KEY] = current_window_complete
    save_experiment_manifest(experiment_folder, experiment_manifest)


def get_persisted_attempt_counts(experiment_folder: dict[str, str]) -> dict[str, int]:
    experiment_manifest = load_experiment_manifest(experiment_folder)
    if experiment_manifest is None:
        return {}
    raw_attempt_counts = experiment_manifest.get(ATTEMPT_COUNTS_KEY, {})
    if not isinstance(raw_attempt_counts, dict):
        return {}
    attempt_counts: dict[str, int] = {}
    for simulation_id, attempt_count in raw_attempt_counts.items():
        try:
            coerced = int(attempt_count)
        except (TypeError, ValueError):
            continue
        if coerced > 0:
            attempt_counts[str(simulation_id)] = coerced
    return attempt_counts


def set_persisted_attempt_count(
    experiment_folder: dict[str, str],
    simulation_id: str,
    attempt_count: int | None,
) -> None:
    experiment_manifest = load_experiment_manifest(experiment_folder)
    if experiment_manifest is None:
        return
    attempt_counts = experiment_manifest.get(ATTEMPT_COUNTS_KEY, {})
    if not isinstance(attempt_counts, dict):
        attempt_counts = {}
    if attempt_count is None:
        attempt_counts.pop(simulation_id, None)
    else:
        attempt_counts[simulation_id] = int(attempt_count)
    experiment_manifest[ATTEMPT_COUNTS_KEY] = attempt_counts
    save_experiment_manifest(experiment_folder, experiment_manifest)


def append_simulation_checkpoint(
    scenario: Scenario,
    simulation: Simulation,
    experiment_folder: dict[str, str],
) -> None:
    results_csv_path = get_results_csv_path(experiment_folder)
    canonical_columns = get_canonical_result_columns([scenario])
    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        _write_canonical_header(results_csv_path, canonical_columns)
    elif not _has_exact_schema(results_csv_path, canonical_columns):
        raise ValueError(
            f"Non-canonical results checkpoint schema in {results_csv_path}. "
            "Legacy folders are not supported by the future-only checkpoint writer."
        )

    _append_canonical_row(
        _build_canonical_row(scenario, simulation, canonical_columns),
        results_csv_path,
        canonical_columns,
    )

    if simulation.result.debug_participation_trace:
        decision_rows = [
            {
                "simulation_id": simulation.id,
                "scenario": scenario.name,
                "sample_index": simulation.index,
                "num_of_robots": simulation.netlogo_params.num_of_robots,
                "robot_participation_strategy": scenario.netlogo_params.robot_participation_strategy,
                **row,
            }
            for row in simulation.result.debug_participation_trace
        ]
        append_dataframe_row(
            pd.DataFrame(decision_rows),
            get_participation_decision_trace_path(experiment_folder),
        )

    if simulation.result.debug_tick_trace_csv.strip():
        tick_df = pd.read_csv(StringIO(simulation.result.debug_tick_trace_csv))
        if not tick_df.empty:
            tick_df.insert(0, "simulation_id", simulation.id)
            tick_df.insert(1, "scenario", scenario.name)
            tick_df.insert(2, "sample_index", simulation.index)
            tick_df.insert(3, "num_of_robots", simulation.netlogo_params.num_of_robots)
            tick_df.insert(
                4,
                "robot_participation_strategy",
                scenario.netlogo_params.robot_participation_strategy,
            )
            append_dataframe_row(
                tick_df,
                get_participation_tick_trace_path(experiment_folder),
            )


def load_completed_simulation_ids(experiment_folder: dict[str, str]) -> set[str]:
    results_csv_path = get_results_csv_path(experiment_folder)
    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        return set()

    try:
        results_df = pd.read_csv(results_csv_path, usecols=["simulation_id"])
    except ValueError:
        return set()
    if results_df.empty:
        return set()
    return set(results_df["simulation_id"].dropna().astype(str))


def _coerce_loaded_result_value(key: str, value: Any) -> Any:
    if pd.isna(value):
        return None
    if key in LIST_LIKE_RESULT_COLUMNS:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
            return parsed if isinstance(parsed, list) else []
        return []
    return value


def restore_checkpointed_results(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
) -> set[str]:
    repair_results_checkpoint_file(scenarios, experiment_folder)
    results_csv_path = get_results_csv_path(experiment_folder)
    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        return set()

    canonical_columns = get_canonical_result_columns(scenarios)
    if not _has_exact_schema(results_csv_path, canonical_columns):
        return set()

    results_df = pd.read_csv(results_csv_path)
    if results_df.empty or "simulation_id" not in results_df.columns:
        return set()

    results_df = normalize_results_dataframe(results_df.reindex(columns=canonical_columns))
    results_df = results_df.drop_duplicates(subset=["simulation_id"], keep="last")
    result_keys = set(Result().__dict__.keys())
    completed_ids: set[str] = set()

    for row in results_df.to_dict(orient="records"):
        simulation_id = str(row.get("simulation_id", "")).strip()
        if not simulation_id:
            continue
        try:
            simulation = Simulation.find_by_id(scenarios, simulation_id)
        except NameError:
            continue

        restored = {
            key: _coerce_loaded_result_value(key, value)
            for key, value in row.items()
            if key in result_keys
        }
        simulation.result.update(restored)
        completed_ids.add(simulation_id)

    return completed_ids


def finalize_checkpoint_files(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
) -> None:
    repair_results_checkpoint_file(scenarios, experiment_folder)
    results_csv_path = get_results_csv_path(experiment_folder)
    canonical_columns = get_canonical_result_columns(scenarios)

    if os.path.exists(results_csv_path) and os.path.getsize(results_csv_path) > 0:
        if not _has_exact_schema(results_csv_path, canonical_columns):
            return
        results_df = normalize_results_dataframe(
            pd.read_csv(results_csv_path).reindex(columns=canonical_columns)
        )
        if not results_df.empty and "simulation_id" in results_df.columns:
            results_df = results_df.drop_duplicates(subset=["simulation_id"], keep="last")
            results_df.to_csv(results_csv_path, index=False, columns=canonical_columns)
        return

    fallback_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for simulation in scenario.simulations:
            fallback_rows.append(_build_canonical_row(scenario, simulation, canonical_columns))
    append_dataframe_row(
        normalize_results_dataframe(pd.DataFrame(fallback_rows, columns=canonical_columns)),
        results_csv_path,
    )


def initialize_results_checkpoint_file(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
) -> None:
    results_csv_path = get_results_csv_path(experiment_folder)
    canonical_columns = get_canonical_result_columns(scenarios)
    if not canonical_columns:
        return

    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        _write_canonical_header(results_csv_path, canonical_columns)
    elif not _has_exact_schema(results_csv_path, canonical_columns):
        raise ValueError(
            f"Non-canonical results checkpoint schema in {results_csv_path}. "
            "Legacy folders are not supported by the future-only checkpoint writer."
        )

    write_scenario_manifest_file(scenarios, experiment_folder)
    repair_results_checkpoint_file(scenarios, experiment_folder)


def repair_results_checkpoint_file(
    scenarios: list[Scenario],
    experiment_folder: dict[str, str],
) -> None:
    results_csv_path = get_results_csv_path(experiment_folder)
    canonical_columns = get_canonical_result_columns(scenarios)
    if not canonical_columns:
        return
    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        return
    if not _has_exact_schema(results_csv_path, canonical_columns):
        return

    results_df = normalize_results_dataframe(
        pd.read_csv(results_csv_path).reindex(columns=canonical_columns)
    )
    results_df.to_csv(results_csv_path, index=False, columns=canonical_columns)
