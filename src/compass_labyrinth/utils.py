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


def read_metadata(config: dict) -> dict:
    """
    Reads the project metadata from the Excel file specified in the configuration.

    Parameters:
    -----------
    config: dict
        The project's configuration dictionary.

    Returns:
    --------
    metadata: pd.DataFrame
        A DataFrame containing metadata information.
    """
    project_path = Path(config["project_path_full"]).resolve()
    metadata_file_path = project_path / "metadata.xlsx"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    return pd.read_excel(metadata_file_path)