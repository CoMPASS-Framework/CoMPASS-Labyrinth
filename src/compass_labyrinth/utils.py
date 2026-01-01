from pathlib import Path
import yaml
import os
import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point
import geopandas as gpd


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


def load_grid_positions(
    ds: xr.Dataset,
    session: str,
    grid_files_path: Path | str,
) -> xr.Dataset:
    """
    Add grid position numbers to an xarray Dataset containing pose estimation data.
    This function performs spatial joins between tracked body positions and a grid
    shapefile to determine which grid cell each position falls within at each time point.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing pose estimation data.
    session : str
        Session name (e.g., 'Session0001') to identify the correct grid file
    grid_files_path : Path | str
        Path to directory containing grid shapefiles (e.g., '{session}_grid.shp')

    Returns
    -------
    xr.Dataset
        Input dataset with added 'grid_number' data variable indicating grid cell numbers.
    """        
    # Load the Grid shapefile
    grid_files_path = Path(grid_files_path)
    grid_file = grid_files_path / f"{session}_grid.shp"
    if not grid_file.exists():
        raise FileNotFoundError(f"Grid file not found at {grid_file}")
    grid = gpd.read_file(str(grid_file))
    
    # Initialize grid numbers array with NaNs
    n_time = len(ds.time)
    n_keypoints = len(ds.keypoints)
    n_individuals = 1  # Assuming single individual 'individual_0'
    grid_numbers_array = np.full((n_time, n_keypoints, n_individuals), np.nan)
    
    # Get keypoint names
    keypoints = ds.keypoints.values
    
    # Process each keypoint
    for kp_idx, keypoint in enumerate(keypoints):
        # Extract x,y positions for this keypoint
        # position has shape (time, space, keypoints, individuals)
        xy = ds.sel(individuals='individual_0', keypoints=keypoint).position.values
        
        # Create Point geometries for each time point
        points = []
        for x, y in xy:
            if pd.notna(x) and pd.notna(y):
                points.append(Point(x, y))
            else:
                points.append(None)
        
        # Create GeoDataFrame of points
        pnt_gpd = gpd.GeoDataFrame(
            geometry=points,
            index=np.arange(len(points)),
            crs=grid.crs,
        )
        
        # Find which polygon each point is in
        pointInPolys = gpd.tools.sjoin(
            pnt_gpd,
            grid,
            predicate="within",
            how="left",
        )
        
        # Extract grid numbers
        # Use 'FID' column from grid or index_right if FID doesn't exist
        if "FID" in pointInPolys.columns:
            grid_nums = pointInPolys["FID"].values
        else:
            grid_nums = pointInPolys["index_right"].values
        
        # Store in the array
        grid_numbers_array[:, kp_idx, 0] = grid_nums
    
    # Add grid_number as a new data variable to the dataset
    ds['grid_number'] = xr.DataArray(
        grid_numbers_array,
        dims=['time', 'keypoints', 'individuals'],
        coords={
            'time': ds.time,
            'keypoints': ds.keypoints,
            'individuals': ds.individuals
        },
        attrs={
            'description': 'Grid cell number for each tracked position',
            'grid_file': str(grid_file),
            'session': session
        }
    )
    
    return ds
