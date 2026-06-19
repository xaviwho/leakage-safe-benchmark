"""
Configuration loader utility.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.yaml

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_param(config: Dict[str, Any], *keys, default=None):
    """
    Safely get nested parameter from config.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found

    Returns:
        Parameter value or default

    Example:
        >>> get_param(config, 'simulation', 'duration', default=14)
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
