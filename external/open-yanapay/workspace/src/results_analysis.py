"""
Results Analysis Module

This module is responsible for analysing and plotting the results of the simulation experiments.

Using https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
And https://www.statology.org/pooled-standard-deviation-calculator/
function to calculate Cohen's d for independent samples
Inspired by: https://machinelearningmastery.com/effect-size-measures-in-python/
"""

import textwrap
import os
import json

import matplotlib  # type: ignore

matplotlib.use('Agg')
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore
from src.evacuation_paper_plotter import EvacuationPaperPlotter, resolve_metric_variants
from src.experiment_checkpoint import (
    SCENARIO_MANIFEST_FILE_NAME,
    normalize_results_dataframe,
)
from src.load_config import get_target_scenario
from src.simulation import Simulation
from utils.helper import setup_logger
from utils.paths import RESULTS_CSV_FILE_NAME, RESULTS_FOLDER

try:
    import statsmodels.api as sm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in local dev envs
    sm = None

PLOT_STYLE = 'seaborn-v0_8-darkgrid'

logger = setup_logger()


def _load_experiment_results(
    csv_results_path: str,
    data_folder_path: str,
) -> pd.DataFrame:
    experiment_data = normalize_results_dataframe(pd.read_csv(csv_results_path))
    scenario_manifest_path = os.path.join(data_folder_path, SCENARIO_MANIFEST_FILE_NAME)
    if not os.path.exists(scenario_manifest_path):
        return experiment_data

    scenario_manifest = normalize_results_dataframe(pd.read_csv(scenario_manifest_path))
    if scenario_manifest.empty or 'scenario' not in scenario_manifest.columns:
        return experiment_data
    if 'scenario' not in experiment_data.columns:
        return experiment_data

    scenario_manifest = scenario_manifest.drop_duplicates(subset=['scenario'], keep='last')
    merged = experiment_data.merge(
        scenario_manifest,
        on='scenario',
        how='left',
        suffixes=('', '_manifest'),
    )
    for column in scenario_manifest.columns:
        if column == 'scenario':
            continue
        manifest_column = f"{column}_manifest"
        if manifest_column not in merged.columns:
            continue
        if column not in experiment_data.columns:
            merged[column] = merged[manifest_column]
        else:
            merged[column] = merged[column].combine_first(merged[manifest_column])
        merged = merged.drop(columns=[manifest_column])

    return normalize_results_dataframe(merged)


def _with_effective_evacuation_ticks(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    prepared['effective_evacuation_ticks'] = prepared['evacuation_ticks']

    if 'failure_reason' in prepared.columns and 'max_netlogo_ticks' in prepared.columns:
        max_ticks_mask = (
            prepared['effective_evacuation_ticks'].isna()
            & prepared['failure_reason'].fillna("").eq('max_ticks')
        )
        prepared.loc[max_ticks_mask, 'effective_evacuation_ticks'] = prepared.loc[
            max_ticks_mask, 'max_netlogo_ticks'
        ]
    elif 'max_netlogo_ticks' in prepared.columns:
        prepared['effective_evacuation_ticks'] = prepared['effective_evacuation_ticks'].fillna(
            prepared['max_netlogo_ticks']
        )

    return prepared


def _primary_metric_summary(
    data: pd.DataFrame,
    group_columns: list[str],
    value_column: str = 'effective_evacuation_ticks',
) -> pd.DataFrame:
    if value_column not in data.columns:
        return pd.DataFrame()

    prepared = data.copy()
    if 'success' not in prepared.columns:
        prepared['success'] = prepared[value_column].notna()

    summary = (
        prepared.groupby(group_columns, observed=True)
        .agg(
            runs=('simulation_id', 'size') if 'simulation_id' in prepared.columns else (value_column, 'size'),
            success_rate=('success', 'mean'),
            median_evacuation_ticks=(value_column, 'median'),
            p90_evacuation_ticks=(value_column, lambda s: s.quantile(0.9)),
            mean_evacuation_ticks=(value_column, 'mean'),
        )
        .reset_index()
    )
    return summary


def _build_per_robot_count_pareto_summary(data: pd.DataFrame) -> pd.DataFrame:
    prepared = _with_effective_evacuation_ticks(data)
    per_robot_count_summary = _primary_metric_summary(
        prepared,
        ['robot_participation_strategy', 'num_of_robots'],
    )
    if per_robot_count_summary.empty:
        return per_robot_count_summary

    cost_summary = (
        prepared.groupby(['robot_participation_strategy', 'num_of_robots'], observed=True)[
            ['active_robot_ticks']
        ]
        .median()
        .reset_index()
        .rename(columns={'active_robot_ticks': 'median_active_robot_ticks'})
    )
    per_robot_count_summary = per_robot_count_summary.merge(
        cost_summary,
        on=['robot_participation_strategy', 'num_of_robots'],
        how='left',
    )
    return per_robot_count_summary.dropna(
        subset=['median_active_robot_ticks', 'median_evacuation_ticks', 'success_rate']
    )


def build_strategy_level_pareto_summary(data: pd.DataFrame) -> pd.DataFrame:
    per_robot_count_summary = _build_per_robot_count_pareto_summary(data)
    if per_robot_count_summary.empty:
        return per_robot_count_summary

    summary = (
        per_robot_count_summary.groupby('robot_participation_strategy', observed=True)
        .agg(
            median_active_robot_ticks=('median_active_robot_ticks', 'mean'),
            median_evacuation_ticks=('median_evacuation_ticks', 'mean'),
            success_rate=('success_rate', 'mean'),
        )
        .reset_index()
    )
    return summary


def save_primary_metric_summaries(
    experiment_data: pd.DataFrame,
    data_folder: str,
) -> None:
    prepared = _with_effective_evacuation_ticks(experiment_data)

    summary_specs = [
        ('scenario', ['scenario']),
        ('strategy', ['strategy']) if 'strategy' in prepared.columns else None,
        (
            'robot_participation_strategy',
            ['robot_participation_strategy'],
        ) if 'robot_participation_strategy' in prepared.columns else None,
        (
            'robot_participation_strategy_by_num_of_robots',
            ['robot_participation_strategy', 'num_of_robots'],
        ) if {'robot_participation_strategy', 'num_of_robots'}.issubset(set(prepared.columns)) else None,
    ]

    for spec in summary_specs:
        if spec is None:
            continue
        name, group_columns = spec
        summary = _primary_metric_summary(prepared, group_columns)
        if summary.empty:
            continue
        summary.to_csv(os.path.join(data_folder, f"{name}_primary_metrics.csv"), index=False)


def _augment_with_tick_trace_costs(
    experiment_data: pd.DataFrame,
    tick_trace_data: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if tick_trace_data is None or tick_trace_data.empty:
        return experiment_data
    if 'simulation_id' not in tick_trace_data.columns or 'simulation_id' not in experiment_data.columns:
        return experiment_data

    required_trace_columns = {
        'simulation_id',
        'active_robots',
        'reserve_robots',
        'task_committed_robots',
        'unresolved_fallen',
        'attended_fallen',
    }
    if not required_trace_columns.issubset(set(tick_trace_data.columns)):
        return experiment_data

    grouped = (
        tick_trace_data.groupby('simulation_id', observed=True)
        .agg(
            active_robot_ticks=('active_robots', 'sum'),
            reserve_robot_ticks=('reserve_robots', 'sum'),
            task_committed_robot_ticks=('task_committed_robots', 'sum'),
            mean_active_robots=('active_robots', 'mean'),
            mean_reserve_robots=('reserve_robots', 'mean'),
            mean_task_committed_robots=('task_committed_robots', 'mean'),
            peak_active_robots=('active_robots', 'max'),
            peak_reserve_robots=('reserve_robots', 'max'),
            mean_unresolved_fallen=('unresolved_fallen', 'mean'),
            mean_attended_fallen=('attended_fallen', 'mean'),
        )
        .reset_index()
    )
    reserve_with_unresolved = (
        tick_trace_data.assign(
            reserve_with_unresolved=(
                (tick_trace_data['reserve_robots'] > 0)
                & (tick_trace_data['unresolved_fallen'] > 0)
            ).astype(int)
        )
        .groupby('simulation_id', observed=True)['reserve_with_unresolved']
        .sum()
        .reset_index()
        .rename(columns={'reserve_with_unresolved': 'reserve_with_unresolved_ticks'})
    )
    grouped = grouped.merge(reserve_with_unresolved, on='simulation_id', how='left')
    augmented = experiment_data.merge(grouped, on='simulation_id', how='left', suffixes=('', '_trace'))
    for column in grouped.columns:
        if column == 'simulation_id':
            continue
        trace_column = f"{column}_trace"
        if trace_column not in augmented.columns:
            continue
        if column not in experiment_data.columns:
            augmented[column] = augmented[trace_column]
        else:
            augmented[column] = augmented[column].fillna(augmented[trace_column])
        augmented = augmented.drop(columns=[trace_column])

    if {
        'robot_participation_strategy',
        'num_of_robots',
    }.issubset(set(augmented.columns)):
        effective = _with_effective_evacuation_ticks(augmented)
        always_mask = (
            augmented['robot_participation_strategy'].fillna("").eq('always')
            & augmented['num_of_robots'].notna()
        )
        if 'active_robot_ticks' in augmented.columns:
            missing_active_cost = always_mask & augmented['active_robot_ticks'].isna()
            augmented.loc[missing_active_cost, 'active_robot_ticks'] = (
                augmented.loc[missing_active_cost, 'num_of_robots']
                * effective.loc[missing_active_cost, 'effective_evacuation_ticks']
            )
        if 'reserve_robot_ticks' in augmented.columns:
            missing_reserve_cost = always_mask & augmented['reserve_robot_ticks'].isna()
            augmented.loc[missing_reserve_cost, 'reserve_robot_ticks'] = 0
        if 'mean_active_robots' in augmented.columns:
            missing_mean_active = always_mask & augmented['mean_active_robots'].isna()
            augmented.loc[missing_mean_active, 'mean_active_robots'] = augmented.loc[
                missing_mean_active, 'num_of_robots'
            ]
        if 'mean_reserve_robots' in augmented.columns:
            missing_mean_reserve = always_mask & augmented['mean_reserve_robots'].isna()
            augmented.loc[missing_mean_reserve, 'mean_reserve_robots'] = 0
        if 'peak_active_robots' in augmented.columns:
            missing_peak_active = always_mask & augmented['peak_active_robots'].isna()
            augmented.loc[missing_peak_active, 'peak_active_robots'] = augmented.loc[
                missing_peak_active, 'num_of_robots'
            ]
        if 'peak_reserve_robots' in augmented.columns:
            missing_peak_reserve = always_mask & augmented['peak_reserve_robots'].isna()
            augmented.loc[missing_peak_reserve, 'peak_reserve_robots'] = 0

    return augmented


def _get_target_scenario_for_analysis(experiment_folder_path: str) -> str:
    config_path = os.path.join(experiment_folder_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            return str(config.get("targetScenarioForAnalysis", ""))
        except Exception as exc:
            logger.warning("Failed to load analysis target from %s: %s", config_path, exc)
    return get_target_scenario()


def cohen_d_from_metrics(mean_1: float, mean_2: float, std_dev_1: float, std_dev_2: float) -> float:
    """
    Calculate Cohen's d effect size from the means and standard deviations of two samples.

    Cohen's d is a measure of the standardized difference between two means. It is calculated as the
    difference between the two means divided by the pooled standard deviation.

    Args:
        mean_1: The mean of the first sample.
        mean_2: The mean of the second sample.
        std_dev_1: The standard deviation of the first sample.
        std_dev_2: The standard deviation of the second sample.

    Returns:
        The Cohen's d effect size.
    """
    pooled_std_dev = np.sqrt((std_dev_1 ** 2 + std_dev_2 ** 2) / 2)
    return (mean_1 - mean_2) / pooled_std_dev


def calculate_sample_size(mean_1: float, mean_2: float, std_dev_1: float, std_dev_2: float,
                          alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Calculates the recommended sample size for a two-sample test.

    This function uses the `statsmodels` library to calculate the recommended sample size based on
    the provided means, standard deviations, alpha level, and desired power.

    Args:
        mean_1: The mean of the first sample.
        mean_2: The mean of the second sample.
        std_dev_1: The standard deviation of the first sample.
        std_dev_2: The standard deviation of the second sample.
        alpha: The desired alpha level (type I error rate). Defaults to 0.05.
        power: The desired statistical power. Defaults to 0.8.

    Returns:
        The recommended sample size for each group.
    """
    if sm is None:
        return 0
    analysis: sm.stats.TTestIndPower = sm.stats.TTestIndPower()
    effect_size = cohen_d_from_metrics(mean_1, mean_2, std_dev_1, std_dev_2)
    # If the results are identical, the effect size will be 0
    if effect_size == 0:
        return 0
    result = analysis.solve_power(effect_size=effect_size,
                                  alpha=alpha,
                                  power=power,
                                  alternative="two-sided")
    return result


def test_hypothesis(first_scenario_column: str,
                    second_scenario_column: str,
                    results_dataframe: pd.DataFrame,
                    experiment_folder_path: str,
                    alternative: str = "two-sided",) -> None:
    """
    Perform a Mann-Whitney U test to compare the distributions of two samples.

    This function calculates the means, standard deviations, and recommended sample sizes for the
    two samples, then performs a Mann-Whitney U test to determine if the distributions are
    significantly different. The results are printed to the console and also saved to a file.

    Args:
        first_scenario_column: The name of the column containing the first sample.
        second_scenario_column: The name of the column containing the second sample.
        results_dataframe: The DataFrame containing the sample data.
        experiment_folder_path: The path to the experiments folder.
        alternative: The alternative hypothesis, either "two-sided", "less", or "greater".
                     Defaults to "two-sided".
    """

    first_scenario_data = results_dataframe[first_scenario_column].values
    first_scenario_mean = np.mean(first_scenario_data).item()
    first_scenario_stddev = np.std(first_scenario_data).item()

    second_scenario_data = results_dataframe[second_scenario_column].values
    second_scenario_mean = np.mean(second_scenario_data).item()
    second_scenario_stddev = np.std(second_scenario_data).item()

    logger.info("{}->mean = {} std = {} len={}".format(
        first_scenario_column, first_scenario_mean,
        first_scenario_stddev, len(first_scenario_data)))
    logger.info("{}->mean = {} std = {} len={}".format(
        second_scenario_column, second_scenario_mean,
        second_scenario_stddev, len(second_scenario_data)))
    logger.info("Recommended Sample size: {}".format(
        calculate_sample_size(first_scenario_mean, second_scenario_mean, first_scenario_stddev,
                              second_scenario_stddev)))

    null_hypothesis = (
        "MANN-WHITNEY RANK TEST: The distribution of {} times is THE SAME as the "
        "distribution of {} times".format(first_scenario_column, second_scenario_column)
    )

    alternative_hypothesis = (
        "ALTERNATIVE HYPOTHESIS: the distribution underlying {} is stochastically {} "
        "than the distribution underlying {}".format(
            first_scenario_column, alternative, second_scenario_column
        )
    )

    threshold = 0.05
    u, p_value = mannwhitneyu(x=first_scenario_data, y=second_scenario_data,
                              alternative=alternative)
    logger.info("U={} , p={}".format(u, p_value))

    hypothesis_file_path = experiment_folder_path + "hypothesis_tests.txt"
    if p_value > threshold:
        logger.info("FAILS TO REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
        # save the results
        with open(hypothesis_file_path, "a") as f:
            f.write(f"p value: {p_value}\n")
            f.write("FAILS TO REJECT NULL HYPOTHESIS: {}\n".format(null_hypothesis))
            f.write(alternative_hypothesis)
            f.write("\n")
    else:
        logger.info("REJECT NULL HYPOTHESIS: {}".format(null_hypothesis))
        logger.info(alternative_hypothesis)
        # save the results
        with open(hypothesis_file_path, "a") as f:
            f.write(f"p value: {p_value}\n")
            f.write("REJECT NULL HYPOTHESIS: {}\n".format(null_hypothesis))
            f.write(alternative_hypothesis)
            f.write("\n")


def get_metrics(experiment_results: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the metrics of the experiment results.

    Args:
        experiment_results: The DataFrame containing the experiment results.

    Returns:
        The DataFrame containing the description of the results.
    """
    metrics_df = experiment_results.describe()
    median = experiment_results.median(numeric_only=True).rename('median')
    p90 = experiment_results.quantile(0.9, numeric_only=True).rename('p90')
    metrics_df = pd.concat([metrics_df, median.to_frame().T, p90.to_frame().T])
    logger.info("\n%s\n", metrics_df)
    return metrics_df


def plot_results(data_for_violin: dict[str, pd.DataFrame], img_folder: str) -> None:
    """
    Plots the results of the experiment, if the number of scenarios is less than 7.

    Args:
        data_for_violin: A dictionary containing the data to plot and name of the column.
        img_folder: The path to the image folder.
    """
    for name, violin_data in data_for_violin.items():
        if len(violin_data.columns) > 20:
            continue
        violin_width = 4
        total_fig_width = len(violin_data.columns) * violin_width
        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(total_fig_width, 10))
        plt_path = img_folder + name + "_violin_plot"

        medians = violin_data.median().sort_values(ascending=False)
        sorted_violin_data = violin_data[medians.index]

        ax = sns.violinplot(data=sorted_violin_data, order=None)
        ax.set_title(f"{name.capitalize()} Comparison")
        locs = ax.get_xticks()
        labels = [textwrap.fill(label.get_text(), 30) for label in ax.get_xticklabels()]
        ax.xaxis.set_major_locator(plt.FixedLocator(locs))
        ax.set_xticklabels(labels, ha='center')
        plt.savefig(plt_path + ".png", bbox_inches='tight', pad_inches=0)
        plt.savefig(plt_path + ".eps", bbox_inches='tight', pad_inches=0)
        plt.clf()


def process_data(experiment_data: pd.DataFrame, column: str, data_folder: str) -> pd.DataFrame:
    """
    Processes the data from the experiment results in order to plot them.

    It combines the simulation evacuation ticks from each scenario sample in a row and
    has each scenario as a column.

    Args:
        experiment_data: DataFrame with all simulations' data.
        column: The column to group the data by.
        data_folder: The path to the folder where the processed data will be saved.

    Returns:
        processed_data: DataFrame with ticks grouped by scenario.
    """
    prepared = _with_effective_evacuation_ticks(experiment_data)
    # Split 'simulation_id' to extract the simulation number
    if 'sample_index' in prepared.columns:
        prepared['sim_index'] = prepared['sample_index']
    else:
        prepared['sim_index'] = prepared['simulation_id'].apply(Simulation.get_index)
    # Pivot the DataFrame using 'sim_index' as the new index
    processed_data = prepared.pivot_table(
        index='sim_index', columns=column, values='effective_evacuation_ticks', aggfunc='median')

    processed_data_path = data_folder + column + "_processed_data.csv"
    processed_data.to_csv(processed_data_path)

    metrics = get_metrics(processed_data)
    metrics_path = data_folder + column + "_metrics.csv"
    metrics.to_csv(metrics_path)
    return processed_data


def plot_robot_actions(data: pd.DataFrame, img_folder: str) -> None:
    """
    Plots the robot actions for the different scenarios.

    Args:
        data: The DataFrame containing the experiment data.
        img_folder: The path to the image folder.
    """
    plt.style.use(PLOT_STYLE)
    plt_path = img_folder + "robot_actions.png"
    # Replace NaN with 'NoStrategy'
    data['strategy'] = data['strategy'].fillna('NoStrategy')
    strategies = data['strategy'].unique()
    true_counts = []
    false_counts = []
    call_staff_counts = []
    # count the number of times each strategy appears in the data and store it a dictionary
    strategy_counts = data['strategy'].value_counts().reindex(strategies).to_dict()

    # Data Preparation
    for strategy in strategies:
        strategy_data = data[data['strategy'] == strategy]
        true_counts.append(strategy_data['robot_responses'].apply(lambda x: 'true' in x).sum())
        false_counts.append(strategy_data['robot_responses'].apply(lambda x: 'false' in x).sum())
        call_staff_counts.append(
            strategy_data['robot_actions'].apply(lambda x: 'call-staff' in x).sum())

    # Plotting
    x = range(len(strategies))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, false_counts, width, label='Refused to help')
    ax.bar(x, true_counts, width, label='Accepted to help', bottom=false_counts)

    ax.bar([p + width for p in x],
           call_staff_counts, width, label='Call-Staff Actions', align='center')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_title('Total Robot Responses and Actions by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{strategy}\n(n:{strategy_counts[strategy]})" for strategy in strategies],
        rotation=45, ha='center')
    ax.legend()
    fig.tight_layout()
    plt.savefig(plt_path)
    plt.clf()


def plot_interference_metrics(data: pd.DataFrame, img_folder: str) -> None:
    """
    Plots aggregate interference metrics by robot count when available, otherwise by
    participation strategy.

    Args:
        data: The DataFrame containing the experiment data.
        img_folder: The path to the image folder.
    """
    required_columns = ['duplicate_robot_contacts', 'busy_staff_distractions',
                        'interference_delay_total']
    if any(column not in data.columns for column in required_columns):
        return

    metric_columns = ['duplicate_robot_contacts', 'busy_staff_distractions',
                      'interference_delay_total']
    optional_debug_columns = [
        'interference_delay_applications',
        'duplicate_delay_applications',
        'busy_staff_delay_applications',
    ]
    metric_columns.extend([column for column in optional_debug_columns if column in data.columns])

    group_column = 'robot_participation_strategy'
    chart_type = 'bar'
    title = 'Interference Metrics by Participation Strategy'
    rotation = 0
    if 'num_of_robots' in data.columns and data['num_of_robots'].nunique() > 1:
        group_column = 'num_of_robots'
        chart_type = 'line'
        title = 'Interference Metrics by Number of Robots'
        rotation = 0
    elif 'robot_participation_strategy' not in data.columns:
        return

    grouped = data.groupby(group_column, observed=True)[metric_columns].median()
    if grouped.empty:
        return

    plt.style.use(PLOT_STYLE)
    plot_kwargs = {'figsize': (10, 6)}
    if chart_type == 'line':
        plot_kwargs.update({'marker': 'o'})
    ax = grouped.plot(kind=chart_type, **plot_kwargs)
    ax.set_ylabel('Mean Count / Delay')
    ax.set_title(title)
    ax.legend(title='Metric')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(img_folder + "interference_metrics.png", bbox_inches='tight', pad_inches=0)
    plt.clf()


def plot_interference_metrics_by_participation_strategy(
    data: pd.DataFrame,
    img_folder: str,
) -> None:
    """
    Plot interference metrics separately for each participation strategy.

    When multiple robot counts exist, each strategy gets a line plot over `num_of_robots`.
    Otherwise, it gets a single bar chart of the mean interference metrics.

    Args:
        data: The DataFrame containing the experiment data.
        img_folder: The path to the image folder.
    """
    required_columns = [
        'robot_participation_strategy',
        'duplicate_robot_contacts',
        'busy_staff_distractions',
        'interference_delay_total',
    ]
    if any(column not in data.columns for column in required_columns):
        return

    strategies = data['robot_participation_strategy'].dropna().unique()
    if len(strategies) <= 1:
        return

    metric_columns = [
        'duplicate_robot_contacts',
        'busy_staff_distractions',
        'interference_delay_total',
    ]
    optional_debug_columns = [
        'interference_delay_applications',
        'duplicate_delay_applications',
        'busy_staff_delay_applications',
    ]
    metric_columns.extend([column for column in optional_debug_columns if column in data.columns])

    has_robot_count_sweep = (
        'num_of_robots' in data.columns and data['num_of_robots'].nunique() > 1
    )
    plt.style.use(PLOT_STYLE)

    for strategy in strategies:
        strategy_data = data[data['robot_participation_strategy'] == strategy].copy()
        if strategy_data.empty:
            continue

        plt.figure(figsize=(10, 6))
        if has_robot_count_sweep:
            grouped = strategy_data.groupby('num_of_robots', observed=True)[metric_columns].median()
            if grouped.empty:
                plt.close()
                continue
            ax = grouped.plot(kind='line', marker='o', ax=plt.gca())
            ax.set_xlabel('Number of Robots')
            plt.xticks(sorted(strategy_data['num_of_robots'].dropna().unique()))
        else:
            grouped = strategy_data[metric_columns].median().to_frame(name='median_value')
            ax = grouped.plot(kind='bar', legend=False, ax=plt.gca())
            ax.set_xlabel('Metric')
            plt.xticks(rotation=45, ha='right')

        ax.set_ylabel('Median Count / Delay')
        ax.set_title(f'Interference Metrics: {strategy}')
        if has_robot_count_sweep:
            ax.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(
            img_folder + f"interference_metrics_{strategy}.png",
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.clf()


def plot_num_of_robots_by_participation_strategy(data: pd.DataFrame, img_folder: str) -> None:
    """
    Plot evacuation ticks against robot count for each participation strategy.

    This produces:
    - a combined comparison plot with one line per participation strategy
    - one dedicated plot per participation strategy to make each knee easy to inspect

    Args:
        data: The DataFrame containing the experiment data.
        img_folder: The path to the image folder.
    """
    required_columns = ['num_of_robots', 'evacuation_ticks', 'robot_participation_strategy']
    if any(column not in data.columns for column in required_columns):
        return

    if data['num_of_robots'].nunique() <= 1 or data['robot_participation_strategy'].nunique() <= 1:
        return

    ordered = _with_effective_evacuation_ticks(data)
    ordered = ordered.sort_values(['robot_participation_strategy', 'num_of_robots'])
    summary = _primary_metric_summary(
        ordered,
        ['robot_participation_strategy', 'num_of_robots'],
    )
    if summary.empty:
        return
    plt.style.use(PLOT_STYLE)

    metric_specs = [
        (
            'median_evacuation_ticks',
            'Median Evacuation Ticks',
            'robot_participation_strategy_num_of_robots_comparison_median.png',
            'Median Evacuation Ticks vs Number of Robots by Participation Strategy',
            'Median Evacuation Ticks (max-ticks failures capped)',
        ),
        (
            'mean_evacuation_ticks',
            'Mean Evacuation Ticks',
            'robot_participation_strategy_num_of_robots_comparison_mean.png',
            'Mean Evacuation Ticks vs Number of Robots by Participation Strategy',
            'Mean Evacuation Ticks (max-ticks failures capped)',
        ),
    ]
    for column, _label, filename, title, ylabel in metric_specs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=summary,
            x='num_of_robots',
            y=column,
            hue='robot_participation_strategy',
            marker='o',
        )
        plt.title(title)
        plt.xlabel('Number of Robots')
        plt.ylabel(ylabel)
        plt.xticks(sorted(summary['num_of_robots'].unique()))
        plt.legend(title='Participation strategy')
        plt.tight_layout()
        plt.savefig(
            img_folder + filename,
            bbox_inches='tight',
            pad_inches=0,
        )
        if column == 'median_evacuation_ticks':
            plt.savefig(
                img_folder + "robot_participation_strategy_num_of_robots_comparison.png",
                bbox_inches='tight',
                pad_inches=0,
            )
        plt.clf()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=summary,
        x='num_of_robots',
        y='success_rate',
        hue='robot_participation_strategy',
        marker='o',
    )
    plt.title('Success Rate vs Number of Robots by Participation Strategy')
    plt.xlabel('Number of Robots')
    plt.ylabel('Success Rate')
    plt.xticks(sorted(summary['num_of_robots'].unique()))
    plt.ylim(0, 1.05)
    plt.legend(title='Participation strategy')
    plt.tight_layout()
    plt.savefig(
        img_folder + "robot_participation_strategy_success_rate.png",
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.clf()

    for strategy in summary['robot_participation_strategy'].dropna().unique():
        strategy_subset = summary[summary['robot_participation_strategy'] == strategy]
        if strategy_subset.empty:
            continue

        for column, label, filename, title, ylabel in [
            (
                'median_evacuation_ticks',
                'Median',
                f"num_of_robots_comparison_{strategy}_median.png",
                f"Median Evacuation Ticks vs Number of Robots ({strategy})",
                'Median Evacuation Ticks (max-ticks failures capped)',
            ),
            (
                'mean_evacuation_ticks',
                'Mean',
                f"num_of_robots_comparison_{strategy}_mean.png",
                f"Mean Evacuation Ticks vs Number of Robots ({strategy})",
                'Mean Evacuation Ticks (max-ticks failures capped)',
            ),
        ]:
            plt.figure(figsize=(8, 5))
            sns.lineplot(
                data=strategy_subset,
                x='num_of_robots',
                y=column,
                marker='o',
            )
            plt.title(title)
            plt.xlabel('Number of Robots')
            plt.ylabel(ylabel)
            plt.xticks(sorted(strategy_subset['num_of_robots'].unique()))
            plt.tight_layout()
            plt.savefig(
                img_folder + filename,
                bbox_inches='tight',
                pad_inches=0,
            )
            if column == 'median_evacuation_ticks':
                plt.savefig(
                    img_folder + f"num_of_robots_comparison_{strategy}.png",
                    bbox_inches='tight',
                    pad_inches=0,
                )
            plt.clf()

        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=strategy_subset,
            x='num_of_robots',
            y='success_rate',
            marker='o',
        )
        plt.title(f"Success Rate vs Number of Robots ({strategy})")
        plt.xlabel('Number of Robots')
        plt.ylabel('Success Rate')
        plt.xticks(sorted(strategy_subset['num_of_robots'].unique()))
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(
            img_folder + f"success_rate_comparison_{strategy}.png",
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.clf()


def plot_robot_tick_costs_by_participation_strategy(
    data: pd.DataFrame,
    img_folder: str,
) -> None:
    required_columns = [
        'num_of_robots',
        'robot_participation_strategy',
        'active_robot_ticks',
        'reserve_robot_ticks',
        'task_committed_robot_ticks',
    ]
    if any(column not in data.columns for column in required_columns):
        return

    if data['num_of_robots'].nunique() <= 1 or data['robot_participation_strategy'].nunique() <= 1:
        return

    metrics = [
        ('active_robot_ticks', 'Active Robot-Ticks'),
        ('reserve_robot_ticks', 'Idle Robot-Ticks'),
        ('task_committed_robot_ticks', 'Task-Committed Robot-Ticks'),
    ]
    plt.style.use(PLOT_STYLE)
    for agg_name, agg_label in [('median', 'Median'), ('mean', 'Mean')]:
        grouped = (
            data.groupby(['robot_participation_strategy', 'num_of_robots'], observed=True)[
                ['active_robot_ticks', 'reserve_robot_ticks', 'task_committed_robot_ticks']
            ]
            .agg(agg_name)
            .reset_index()
        )
        if grouped.empty:
            continue

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
        for axis, (column, label) in zip(axes, metrics):
            sns.lineplot(
                data=grouped,
                x='num_of_robots',
                y=column,
                hue='robot_participation_strategy',
                marker='o',
                ax=axis,
            )
            axis.set_title(label)
            axis.set_ylabel(f'{agg_label} Tick Cost')
            axis.legend(title='Participation strategy')

        axes[-1].set_xlabel('Number of Robots')
        axes[-1].set_xticks(sorted(grouped['num_of_robots'].unique()))
        fig.tight_layout()
        plt.savefig(
            img_folder + f"robot_tick_costs_by_participation_strategy_{agg_name}.png",
            bbox_inches='tight',
            pad_inches=0,
        )
        if agg_name == 'median':
            plt.savefig(
                img_folder + "robot_tick_costs_by_participation_strategy.png",
                bbox_inches='tight',
                pad_inches=0,
            )
        plt.clf()


def plot_evacuation_vs_active_robot_ticks(
    data: pd.DataFrame,
    img_folder: str,
) -> None:
    required_columns = [
        'robot_participation_strategy',
        'num_of_robots',
        'active_robot_ticks',
        'evacuation_ticks',
    ]
    if any(column not in data.columns for column in required_columns):
        return

    prepared = _with_effective_evacuation_ticks(data)
    prepared = prepared.dropna(subset=['effective_evacuation_ticks', 'active_robot_ticks'])
    if prepared.empty or prepared['robot_participation_strategy'].nunique() <= 1:
        return

    plt.style.use(PLOT_STYLE)
    for agg_name, agg_label in [('median', 'Median'), ('mean', 'Mean')]:
        grouped = (
            prepared.groupby(['robot_participation_strategy', 'num_of_robots'], observed=True)[
                ['active_robot_ticks', 'effective_evacuation_ticks']
            ]
            .agg(agg_name)
            .reset_index()
        )
        if grouped.empty:
            continue

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x='active_robot_ticks',
            y='effective_evacuation_ticks',
            hue='robot_participation_strategy',
            marker='o',
        )
        for row in grouped.itertuples():
            plt.annotate(
                str(int(row.num_of_robots)),
                (row.active_robot_ticks, row.effective_evacuation_ticks),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=8,
            )
        plt.title(f'{agg_label} Evacuation Time vs {agg_label} Active Robot-Ticks')
        plt.xlabel(f'{agg_label} Active Robot-Ticks')
        plt.ylabel(f'{agg_label} Evacuation Ticks')
        plt.tight_layout()
        plt.savefig(
            img_folder + f"evacuation_vs_active_robot_ticks_{agg_name}.png",
            bbox_inches='tight',
            pad_inches=0,
        )
        if agg_name == 'median':
            plt.savefig(
                img_folder + "evacuation_vs_active_robot_ticks.png",
                bbox_inches='tight',
                pad_inches=0,
            )
        plt.clf()


def plot_pareto_frontier(
    data: pd.DataFrame,
    img_folder: str,
) -> None:
    required_columns = [
        'robot_participation_strategy',
        'num_of_robots',
        'active_robot_ticks',
        'evacuation_ticks',
    ]
    if any(column not in data.columns for column in required_columns):
        return

    prepared = _with_effective_evacuation_ticks(data)
    prepared = prepared.dropna(subset=['active_robot_ticks', 'effective_evacuation_ticks'])
    if prepared.empty:
        return

    per_robot_count_summary = _build_per_robot_count_pareto_summary(prepared)
    if per_robot_count_summary.empty:
        return

    summary = build_strategy_level_pareto_summary(prepared)
    if summary.empty:
        return

    def is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
        better_or_equal = (
            (candidates['median_active_robot_ticks'] <= row['median_active_robot_ticks'])
            & (candidates['median_evacuation_ticks'] <= row['median_evacuation_ticks'])
            & (candidates['success_rate'] >= row['success_rate'])
        )
        strictly_better = (
            (candidates['median_active_robot_ticks'] < row['median_active_robot_ticks'])
            | (candidates['median_evacuation_ticks'] < row['median_evacuation_ticks'])
            | (candidates['success_rate'] > row['success_rate'])
        )
        return bool((better_or_equal & strictly_better).any())

    summary = summary.copy()
    summary['is_pareto_efficient'] = [
        not is_dominated(row, summary.drop(index=index))
        for index, row in summary.iterrows()
    ]

    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        summary['median_active_robot_ticks'],
        summary['median_evacuation_ticks'],
        c=summary['success_rate'],
        cmap='viridis',
        s=90,
        alpha=0.9,
    )
    for row in summary.itertuples():
        plt.annotate(
            f"{row.robot_participation_strategy}",
            (row.median_active_robot_ticks, row.median_evacuation_ticks),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
        )

    frontier = summary[summary['is_pareto_efficient']].sort_values(
        ['median_active_robot_ticks', 'median_evacuation_ticks']
    )
    if not frontier.empty:
        plt.plot(
            frontier['median_active_robot_ticks'],
            frontier['median_evacuation_ticks'],
            linestyle='--',
            linewidth=1.5,
            color='black',
            label='Pareto frontier',
        )
        plt.legend()

    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Success Rate')
    plt.title('Pareto Frontier: Median Evacuation Time vs Median Active Robot-Ticks')
    plt.xlabel('Median Active Robot-Ticks')
    plt.ylabel('Median Evacuation Ticks')
    plt.tight_layout()
    plt.savefig(
        img_folder + "pareto_frontier_evacuation_vs_active_robot_ticks.png",
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.clf()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        per_robot_count_summary['median_active_robot_ticks'],
        per_robot_count_summary['median_evacuation_ticks'],
        c=per_robot_count_summary['success_rate'],
        cmap='viridis',
        s=80,
        alpha=0.85,
    )
    for row in per_robot_count_summary.itertuples():
        plt.annotate(
            f"{row.robot_participation_strategy}:{int(row.num_of_robots)}",
            (row.median_active_robot_ticks, row.median_evacuation_ticks),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=8,
        )
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Success Rate')
    plt.title('Pareto Detail: Median Evacuation Time vs Median Active Robot-Ticks by Robot Count')
    plt.xlabel('Median Active Robot-Ticks')
    plt.ylabel('Median Evacuation Ticks')
    plt.tight_layout()
    plt.savefig(
        img_folder + "pareto_frontier_evacuation_vs_active_robot_ticks_by_num_of_robots.png",
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.clf()


def plot_active_vs_idle_robot_summary(data: pd.DataFrame, img_folder: str) -> None:
    """
    Plot mean active vs reserve robot counts from run-level aggregate metrics.

    Args:
        data: The DataFrame containing experiment-level results.
        img_folder: The path to the image folder.
    """
    required_columns = ['mean_active_robots', 'mean_reserve_robots']
    if any(column not in data.columns for column in required_columns):
        return

    if 'num_of_robots' not in data.columns:
        return

    group_columns = ['num_of_robots']
    if 'robot_participation_strategy' in data.columns and \
            data['robot_participation_strategy'].nunique() > 1:
        group_columns.append('robot_participation_strategy')

    grouped = (
        data.groupby(group_columns, observed=True)[required_columns]
        .median()
        .reset_index()
        .rename(columns={
            'mean_active_robots': 'Active',
            'mean_reserve_robots': 'Idle',
        })
    )
    melted = grouped.melt(
        id_vars=group_columns,
        value_vars=['Active', 'Idle'],
        var_name='robot_state',
        value_name='mean_robot_count',
    )
    if melted.empty:
        return

    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(10, 6))
    line_kwargs = {
        'data': melted,
        'x': 'num_of_robots',
        'y': 'mean_robot_count',
        'hue': 'robot_state',
        'marker': 'o',
    }
    if 'robot_participation_strategy' in group_columns:
        line_kwargs['style'] = 'robot_participation_strategy'
    sns.lineplot(**line_kwargs)
    plt.title('Median Active vs Idle Robots')
    plt.xlabel('Number of Robots')
    plt.ylabel('Median Robot Count')
    plt.xticks(sorted(melted['num_of_robots'].unique()))
    plt.tight_layout()
    plt.savefig(
        img_folder + "active_vs_idle_robots_summary.png",
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.clf()


def plot_active_vs_idle_robot_timelines(
    tick_trace_data: pd.DataFrame,
    img_folder: str,
) -> None:
    """
    Plot active vs reserve robots over time from debug tick traces.

    Generates one figure per participation strategy, with one subplot per robot count.

    Args:
        tick_trace_data: The DataFrame containing per-tick robot-state traces.
        img_folder: The path to the image folder.
    """
    required_columns = [
        'tick',
        'active_robots',
        'reserve_robots',
        'num_of_robots',
        'robot_participation_strategy',
    ]
    if any(column not in tick_trace_data.columns for column in required_columns):
        return

    if tick_trace_data.empty:
        return

    plt.style.use(PLOT_STYLE)
    strategies = tick_trace_data['robot_participation_strategy'].dropna().unique()
    for strategy in strategies:
        strategy_data = tick_trace_data[
            tick_trace_data['robot_participation_strategy'] == strategy
        ].copy()
        if strategy_data.empty:
            continue

        grouped = (
            strategy_data.groupby(['num_of_robots', 'tick'], observed=True)[
                ['active_robots', 'reserve_robots']
            ]
            .median()
            .reset_index()
            .rename(columns={
                'active_robots': 'Active',
                'reserve_robots': 'Idle',
            })
        )
        melted = grouped.melt(
            id_vars=['num_of_robots', 'tick'],
            value_vars=['Active', 'Idle'],
            var_name='robot_state',
            value_name='mean_robot_count',
        )
        robot_counts = sorted(melted['num_of_robots'].dropna().unique())
        if not robot_counts:
            continue

        fig, axes = plt.subplots(
            len(robot_counts),
            1,
            figsize=(10, max(4, 3 * len(robot_counts))),
            sharex=True,
            sharey=True,
        )
        if len(robot_counts) == 1:
            axes = [axes]

        for axis, robot_count in zip(axes, robot_counts):
            subset = melted[melted['num_of_robots'] == robot_count]
            sns.lineplot(
                data=subset,
                x='tick',
                y='mean_robot_count',
                hue='robot_state',
                marker=None,
                ax=axis,
            )
            axis.set_title(f"{strategy}: {int(robot_count)} robots")
            axis.set_ylabel('Mean Robot Count')
            axis.legend(title='State')

        axes[-1].set_xlabel('Tick')
        fig.tight_layout()
        plt.savefig(
            img_folder + f"active_vs_idle_robots_over_time_{strategy}.png",
            bbox_inches='tight',
            pad_inches=0,
        )
        plt.clf()


def plot_comparisons(experiment_data: pd.DataFrame, img_folder: str) -> None:
    """
    Plots the comparisons between differences in the dataFrame.

    Checks the dataFrame for columns that have different values and plots combinations.

    Example:
        - If the data has under num_robots values 1 and 2, the function will plot the difference
          between the evacuation_ticks for those values.

    Args:
        experiment_data: The DataFrame containing the experiment data.
        img_folder: The path to the image folder.
    """
    columns_to_check = ['robot_persuasion_factor', 'num_of_robots', 'num_of_passengers',
                        'num_of_staff', 'fall_length', 'fall_chance', 'room_type',
                        'robot_interference_delay']
    unique_columns = {
        column: experiment_data[column].unique()
        for column in columns_to_check
        if column in experiment_data.columns
    }

    for column, values in unique_columns.items():
        if len(values) > 1:
            plt.style.use(PLOT_STYLE)
            plt.figure(figsize=(10, 6))
            # Plot the column with a different color for each other column value if unique
            for other_column, other_values in unique_columns.items():
                if len(other_values) == 1:
                    continue
                # plot a column with unique values vs evacuation_ticks
                if other_column == column:
                    sns.lineplot(data=experiment_data, x=column, y='evacuation_ticks', estimator='median')
                    plt.xticks(values)
                    plt_path = img_folder + f"{column}_comparison.png"
                    plt.title(f"Median Evacuation Ticks vs {column.capitalize()}")
                    plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
                    plt.clf()
                    continue
                # plot the column but add a line for each other column with unique values
                for value in other_values:
                    subset = experiment_data[experiment_data[other_column] == value]
                    sns.lineplot(data=subset, x=column, y='evacuation_ticks',
                                 label=f"{other_column}={value}", errorbar=None, estimator='median')
                # Plot the entire dataset for this column as a dotted line
                sns.lineplot(data=experiment_data, x=column, y='evacuation_ticks',
                             label='Overall (with error band)', linestyle='--', color='grey',
                             estimator='median')
                plt.xticks(values)
                plt_path = img_folder + f"{column}({other_column})_comparison.png"
                plt.title(f"Median Evacuation Ticks vs {column.capitalize()} ({other_column})")
                plt.legend(title=other_column)
                plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
                plt.clf()

    # for each parameter with a unique value, plot the evacuation_ticks for each strategy
    strategies_df = experiment_data['strategy'].str.split('@', expand=True)
    experiment_data['strategy'] = strategies_df[0].replace(np.nan, 'NoStrategy')
    for column, values in unique_columns.items():
        if not len(values) > 1:
            continue
        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(10, 6))
        for strategy in experiment_data['strategy'].unique():
            subset = experiment_data[experiment_data['strategy'] == strategy]
            sns.lineplot(data=subset, x=column, y='evacuation_ticks',
                         label=f"{strategy}", errorbar=None, estimator='median')

        sns.lineplot(data=experiment_data, x=column, y='evacuation_ticks',
                     label='Overall (with error band)', linestyle='--', color='grey',
                     estimator='median')
        plt.xticks(values, rotation=45)
        plt_path = img_folder + f"strategy_{column}_comparison.png"
        plt.title(f"Median Evacuation Ticks vs {column} (strategy)")
        plt.legend(title='strategy')
        plt.savefig(plt_path, bbox_inches='tight', pad_inches=0)
        plt.clf()


def perform_analysis(experiment_folder: dict[str, str],
                     folder_name: Optional[str] = None,
                     paper_mode: Optional[str] = None,
                     metric_variants: Optional[str] = None) -> None:
    """
    Performs the analysis of the experiment results.

    Args:
        experiment_folder: a dictionary containing the path to the
                           experiment folder and its sub-folders.
        folder_name: the name of the folder in results containing the experiment results.
    """
    if folder_name:
        experiment_folder_path = RESULTS_FOLDER + folder_name + '/'
        imgs_folder_path = experiment_folder_path + 'img/'
        data_folder_path = experiment_folder_path + 'data/'
        csv_results_path = data_folder_path + RESULTS_CSV_FILE_NAME
    else:
        experiment_folder_path = experiment_folder['path']
        imgs_folder_path = experiment_folder['img']
        data_folder_path = experiment_folder['data']
        csv_results_path = experiment_folder['data'] + RESULTS_CSV_FILE_NAME

    experiment_data = _load_experiment_results(csv_results_path, data_folder_path)
    save_primary_metric_summaries(experiment_data, data_folder_path)
    tick_trace_csv_path = data_folder_path + "participation_tick_trace.csv"
    tick_trace_data = None
    if os.path.exists(tick_trace_csv_path):
        tick_trace_data = pd.read_csv(tick_trace_csv_path)
        if 'num_of_robots' not in tick_trace_data.columns and \
                'simulation_id' in tick_trace_data.columns and \
                'simulation_id' in experiment_data.columns and \
                'num_of_robots' in experiment_data.columns:
            tick_trace_data = tick_trace_data.merge(
                experiment_data[['simulation_id', 'num_of_robots']].drop_duplicates(),
                on='simulation_id',
                how='left',
            )
    experiment_data = _augment_with_tick_trace_costs(experiment_data, tick_trace_data)

    scenario_processed_data = process_data(experiment_data, 'scenario', data_folder_path)
    strategy_processed_data = process_data(experiment_data, 'strategy', data_folder_path)
    participation_processed_data = None
    if 'robot_participation_strategy' in experiment_data.columns:
        participation_processed_data = process_data(
            experiment_data,
            'robot_participation_strategy',
            data_folder_path,
        )

    violin_data = {
        'scenario': scenario_processed_data,
        'strategy': strategy_processed_data,
    }
    if participation_processed_data is not None:
        violin_data['robot_participation_strategy'] = participation_processed_data

    plot_results(violin_data, imgs_folder_path)

    plot_comparisons(experiment_data, imgs_folder_path)
    plot_robot_actions(experiment_data, imgs_folder_path)
    plot_interference_metrics(experiment_data, imgs_folder_path)
    plot_interference_metrics_by_participation_strategy(experiment_data, imgs_folder_path)
    plot_num_of_robots_by_participation_strategy(experiment_data, imgs_folder_path)
    plot_robot_tick_costs_by_participation_strategy(experiment_data, imgs_folder_path)
    plot_evacuation_vs_active_robot_ticks(experiment_data, imgs_folder_path)
    plot_pareto_frontier(experiment_data, imgs_folder_path)
    plot_active_vs_idle_robot_summary(experiment_data, imgs_folder_path)
    if tick_trace_data is not None:
        plot_active_vs_idle_robot_timelines(tick_trace_data, imgs_folder_path)
    if paper_mode is not None:
        paper_plotter = EvacuationPaperPlotter(
            imgs_folder_path,
            data_folder=data_folder_path,
            plot_mode=paper_mode,
        )
        paper_plotter.plot_suite(
            experiment_data,
            tick_trace_data,
            metric_variants=resolve_metric_variants(metric_variants),
        )

    target_scenario = _get_target_scenario_for_analysis(experiment_folder_path)
    scenarios = scenario_processed_data.columns.to_list()
    if target_scenario in scenarios:
        for alternative_scenario in scenarios:
            if alternative_scenario != target_scenario:
                test_hypothesis(first_scenario_column=target_scenario,
                                second_scenario_column=alternative_scenario,
                                results_dataframe=scenario_processed_data,
                                experiment_folder_path=experiment_folder_path,
                                alternative="less")
    else:
        logger.error(
            f"Cannot test. Scenario: '{target_scenario}' for analysis not in simulationScenarios," +
            " check targetScenarioForAnalysis in config.json.")
