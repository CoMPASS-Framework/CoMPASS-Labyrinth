from pathlib import Path
import pandas as pd
import datetime
import yaml
import shutil
import os

from .utils import load_project, load_cohort_metadata
from .behavior.pose_estimation.dlc_utils import (
    import_cohort_metadata,
    validate_metadata,
    save_first_frame,
)


def init_project(
    project_name: str,
    project_path: Path | str,
    source_data_path: Path | str,
    user_metadata_file_path: Path | str,
    trial_type: str = "Labyrinth_DSI",
    file_ext: str = ".csv",
    video_type: str = ".mp4",
    dlc_scorer: str = "DLC_resnet50_LabyrinthMar13shuffle1_1000000",
    experimental_groups: list = ["A", "B", "C", "D"],
    palette: str = "grey",
) -> tuple[dict, pd.DataFrame]:
    """
    Initializes project for the CoMPASS-Labyrinth analysis,  including:
    - Setting up directory structure
    - Copying user metadata file to project directory
    - Creating a config.yaml file with project parameters

    Parameters:
    -----------
    project_name: str
        The name of the project.
    project_path: Path | str
        The path to the project directory.
    source_data_path: Path | str
        The path to the source data directory containing videos and DLC outputs.
    user_metadata_file_path: Path | str
        The path to the user metadata Excel file.

    Returns:
    --------
    config: dict
        A dictionary containing configuration parameters.
    metadata_df: pd.DataFrame
        A DataFrame containing cohort metadata.
    """
    # Project name checks should be alphanumeric and underscores only
    if not project_name.replace("_", "").isalnum():
        raise ValueError("Project name must be alphanumeric and can only contain underscores.")
    
    # Validate source data path
    source_data_path = Path(source_data_path).resolve()
    if not source_data_path.exists():
        raise ValueError(f"Source data path {source_data_path} does not exist.")
    
    # Set up project's base path
    project_path = Path(project_path).resolve()
    project_path_full = project_path / project_name
    if not project_path_full.exists():
        project_path_full.mkdir(parents=True, exist_ok=True)
        print(f"Project path does not exist. Creating directory at {project_path_full}")
    else:
        print(f"Project already exists at {project_path_full}")
        return load_project(project_path_full)

    # Create organized directory structure
    all_dirs = {
        # Videos folder - original videos and frames
        'videos': project_path_full / 'videos',
        'videos_original': project_path_full / 'videos' / 'original_videos',
        'frames': project_path_full / 'videos' / 'frames',
        
        # Data folder - analysis inputs and outputs
        'data': project_path_full / 'data',
        'dlc_results': project_path_full / 'data' / 'dlc_results',
        'dlc_cropping': project_path_full / 'data' / 'dlc_cropping_bounds',
        'grid_files': project_path_full / 'data' / 'grid_files',
        'grid_boundaries': project_path_full / 'data' / 'grid_boundaries',
        'metadata': project_path_full / 'data' / 'metadata',
        'eeg_edfs': project_path_full / 'data' / 'processed_eeg_edfs',
        
        # Figures folder - all plots and visualizations
        'figures': project_path_full / 'figures',
        
        # CSV's folder
        'csvs': project_path_full / 'csvs',
        'csvs_individual': project_path_full / 'csvs'/ 'individual',
        'csvs_combined': project_path_full / 'csvs' / 'combined',

        # Results folders
        'results': project_path_full / 'results' ,
        'results_task_performance': project_path_full / 'results' / 'task_performance',
        'results_simulation_agent': project_path_full / 'results' / 'simulation_agent',
        'results_compass_level_1': project_path_full / 'results' / 'compass_level_1',
        'results_compass_level_2': project_path_full / 'results' / 'compass_level_2',
        'results_ephys_compass': project_path_full / 'results' / 'ephys_compass'
    }
    for _, dir_path in all_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy pose estimation outputs to project path
    pose_est_csv_path = project_path_full / "data" / "dlc_results"
    pe_files = [f.resolve() for f in source_data_path.glob(f"*{dlc_scorer}*{file_ext}")]
    pe_files = sorted(pe_files, key=lambda f: f.name)
    for file in pe_files:
        dest_file = pose_est_csv_path / file.name
        if not dest_file.exists():
            shutil.copy2(file, dest_file)

    # ------ temporary - to be removed later ------
    # COPY withGrids.csv files as well
    with_grid_files = [f.resolve() for f in source_data_path.glob(f"*withGrids*{file_ext}")]
    for file in with_grid_files:
        dest_file = pose_est_csv_path / file.name
        if not dest_file.exists():
            shutil.copy2(file, dest_file)

    if len(pe_files) == 0:
        pe_files = with_grid_files
    # ------------------------------------------------------

    # Extract session names from pose estimation files
    session_names = [f.stem.replace(f"{dlc_scorer}", "") for f in pe_files]

    # Extract bodyparts names
    df = pd.read_csv(pe_files[0], header=[0, 1, 2], skipinitialspace=True)
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    bodyparts = [bp for bp in bodyparts if bp.lower() != "bodyparts"]

    # Link videos to central video location
    video_dest_path = project_path_full / "videos" / "original_videos"
    video_files = [f.resolve() for f in source_data_path.glob(f"*{video_type}")]
    video_files = sorted(video_files, key=lambda f: f.name)
    for file in video_files:
        dest_file = video_dest_path / file.name
        if not dest_file.exists():
            os.symlink(file, dest_file)
        # Save first frame of each video for reference
        save_first_frame(
            video_path=file,
            frames_dir=project_path_full / "videos" / "frames",
        )

    # Copy the user passed metadata to the project's path
    # TODO - later on, we will like to construct this metadata file automatically, instead of requesting from user
    construct_metadata = False
    if construct_metadata:
        cohort_metadata = []
        for sess in session_names:
            cohort_metadata.append({
                "session_name": sess,
            })
    else:
        user_metadata_file_path = Path(user_metadata_file_path).resolve()
        metadata_df = import_cohort_metadata(
            metadata_path=user_metadata_file_path,
            trial_sheet_name=trial_type,
        )
        validate_metadata(metadata_df)
        cohort_metadata = metadata_df.to_dict(orient="records")

    # Save cohort metadata as CSV
    metadata_df = pd.DataFrame(cohort_metadata)
    metadata_df.to_csv(project_path_full / "cohort_metadata.csv", index=False)

    # Create config dictionary and save config.yaml in project path
    config = {
        "project_name": project_name,
        "project_path_full": str(project_path_full),
        "creation_date_time": datetime.datetime.now().isoformat(),
        "trial_type": trial_type,
        "file_ext": file_ext,
        "video_type": video_type,
        "dlc_scorer": dlc_scorer,
        "session_names": session_names,
        "bodyparts": bodyparts,
        "experimental_groups": experimental_groups,
        "palette": palette,
    }

    with open(project_path_full / "config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

    return (config, metadata_df)
