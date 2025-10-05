from pathlib import Path
import pandas as pd
import yaml


def load_config(project_path: Path | str) -> dict:
    """
    Loads configuration parameters from a YAML file.

    Parameters:
    -----------
    project_path: Path | str
        The path to the project directory containing the config.yaml file.

    Returns:
    --------
    config: dict
        A dictionary containing configuration parameters.
    """
    project_path = Path(project_path).resolve()
    config_file_path = project_path / "config.yaml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
