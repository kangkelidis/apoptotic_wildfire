"""
Config Manager.

This module provides functionality to load and merge YAML configuration files.
It follows a priority chain: Base -> Preset
"""

import os

import yaml


def merge_dicts(base: dict, overrides: dict) -> dict:
    """
    Recursively merges two dictionaries.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_dir="config", preset=None) -> dict:
    """
    Loads the hierarchical configuration.

    Args:
        config_dir (str): Path to the directory containing yaml files.
        preset (str): The name of the preset to apply from presets.yaml.

    Returns:
        dict: The final merged configuration dictionary.
    """
    base_path = os.path.join(config_dir, "base.yaml")
    presets_path = os.path.join(config_dir, "presets.yaml")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base config not found at {base_path}")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if preset:
        if not os.path.exists(presets_path):
            raise FileNotFoundError(
                f"Presets file not found at {presets_path}")

        with open(presets_path, "r") as f:
            all_presets = yaml.safe_load(f) or {}

        if preset not in all_presets:
            raise ValueError(f"Preset '{preset}' not found in {presets_path}")

        config = merge_dicts(config, all_presets[preset])

    return config
