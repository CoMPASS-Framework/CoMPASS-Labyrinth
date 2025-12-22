from pathlib import Path
import yaml
import os
import pandas as pd


def load_project(project_path: Path | str) -> tuple[dict, pd.DataFrame]:
    """
    Loads configuration parameters and metadata from an existing project.

    Parameters
    ----------
    project_path: Path | str
        The path to the project directory containing the config.yaml and cohort_metadata.csv files.

    Returns
    -------
    config: dict
        A dictionary containing configuration parameters.
    metadata_df: pd.DataFrame
        A DataFrame containing cohort metadata.
    """
    # Load config.yaml
    project_path = Path(project_path).resolve()
    config_file_path = project_path / "config.yaml"
    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Load metadata CSV
    metadata_file_path = project_path / "cohort_metadata.csv"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    metadata_df = pd.read_csv(metadata_file_path)

    return (config, metadata_df)


def load_cohort_metadata(config: dict) -> pd.DataFrame:
    """
    Loads cohort metadata from the CSV file specified in the project configuration.

    Parameters
    ----------
    config: dict
        The project configuration dictionary containing the path to the cohort metadata CSV.

    Returns
    -------
    metadata_df: pd.DataFrame
        A DataFrame containing cohort metadata.
    """
    project_path = Path(config["project_path_full"]).resolve()
    metadata_file_path = project_path / "cohort_metadata.csv"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")

    metadata_df = pd.read_csv(metadata_file_path)
    return metadata_df


def save_figure(
    config: dict,
    fig_name: str,
    subdir: str = "results/task_performance",
    dpi: int = 300,
    ext: str = "pdf",
):
    """
    Save the current matplotlib figure to a standardized results folder.

    Parameters
    ----------
    config : dict
        Project's configuration dictionary.
    fig_name : str
        Name of the figure file, e.g., 'Shannons_entropy' or 'Bout_Success'.
        Extension is automatically appended as defined by `ext`.
    subdir : str
        Subfolder path under BASE_PATH to save the figure.
    dpi : int
        Resolution of saved figure.
    ext : str
        File extension, e.g., 'pdf', 'png', etc.
    """
    import matplotlib.pyplot as plt

    base_path = config["project_path_full"]
    os.makedirs(os.path.join(base_path, subdir), exist_ok=True)
    save_path = os.path.join(base_path, subdir, f"{fig_name}.{ext}")
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {save_path}")
