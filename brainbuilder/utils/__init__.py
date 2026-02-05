# SPDX-License-Identifier: Apache-2.0
"""libraries of common functionality for circuit building"""

import json

import yaml


def load_json(filepath):
    """Load from JSON file."""
    with open(str(filepath), "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(filepath):
    """Load from YAML file."""
    with open(str(filepath), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(filepath, data, indent=2):
    """Dump to JSON file."""
    with open(str(filepath), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def dump_yaml(filepath, data):
    """Dump to YAML file."""
    with open(str(filepath), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
