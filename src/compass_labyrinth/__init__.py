from pathlib import Path
import pandas as pd
import datetime
import yaml
import shutil

from .utils import load_config
from compass_labyrinth.behavior.pose_estimation.dlc_utils import (
    load_cohort_metadata,
    validate_metadata,
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
    bodyparts: list = [
        "nose", "belly", "sternum", "leftflank", "rightflank", "tailbase"
    ],
    experimental_groups: list = ["A", "B", "C", "D"],
    palette: str = "grey",
):
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
        return load_config(project_path_full)

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
    for dir_name, dir_path in all_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy or link videos to central video location
    # TODO

    # Copy pose estimation CSV outputs to project path
    pose_est_csv_path = project_path_full / "data" / "dlc_results"
    csv_files = [f for f in source_data_path.glob(f"*{file_ext}") if f.is_file()]
    for file in csv_files:
        dest_file = pose_est_csv_path / file.name
        if not dest_file.exists():
            shutil.copy2(file, dest_file)

    # Path for all grid based files for a particular Session
    # as part of Level-1 Post-Analysis --> Plot 1: Heatmap Representations of HMM States
    # TODO - copy grid files if available

    # Copy the user passed metadata to the project's path
    # TODO - later on, we will like to construct this metadata file automatically, instead of requesting from user
    user_metadata_file_path = Path(user_metadata_file_path).resolve()
    metadata_df = load_cohort_metadata(
        metadata_path=user_metadata_file_path,
        trial_sheet_name=trial_type,
    )
    validate_metadata(metadata_df)
    sessions_dict = metadata_df.to_dict(orient="records")

    # # =============================================================================
    # # Specific to Palop Lab, IGNORE FOR MOST CASES --------------------------------
    # # Location of original raw video locations from 2 computers 
    # # (Copy original videos to central VIDEOFILE_PATH location)
    # VIDEO_PATH_1 = ""
    # VIDEO_PATH_2 = ""
    # print(f"Location of Computer 1 Videos: {VIDEO_PATH_1}")
    # print(f"Location of Computer 2 Videos: {VIDEO_PATH_2}")

    # # DeepLabCut CONFIG PATH, if running DLC from Palop labyrinth 'supernetwork'
    # DLC_CONFIG_PATH = "" 

    # =============================================================================
    # PRE-FIXED VALUES (DO NOT EDIT)
    # =============================================================================
    
    # -------------------- REGION---------------------------#
    # Map Grid Nodes to Regions
    target_zone = [84, 85, 86]
    entry_zone = [47, 46]
    loops = [
        33,
        45,
        57,
        58,
        59,
        71,
        83,
        95,
        70,
        69,
        68,
        56,
        44,
        78,
        79,
        80,
        81,
        82,
        94,
        106,
        118,
        105,
        93,
        92,
        104,
        116,
        117,
        91,
        90,
        52,
        53,
        41,
        42,
        43,
        55,
        67,
        54,
        66,
        65,
        64,
        38,
        37,
        36,
        25,
        24,
        13,
        12,
        0,
        1,
    ]
    neutral_zone = [107, 119, 131, 143]

    # P1, P2, P3 are the 3 sections of the Reward Path
    p1 = [22, 21, 34, 20, 32, 31, 30, 29, 17, 5, 4, 3, 2, 14, 26]
    p2 = [27, 39, 51, 63, 62, 61, 60, 72, 73, 74, 75, 76, 77, 89, 101]
    p3 = [102, 103, 115, 114, 113, 125, 137, 136, 135, 123, 111, 110, 109, 108, 96, 97, 98]
    reward_path = p1 + p2 + p3

    left_dead_ends = [10, 11, 23, 35, 9, 8, 6, 7, 19, 18, 15, 16, 28, 40, 50, 49, 48]  # Left Dead Ends
    right_dead_ends = [
        128,
        129,
        130,
        142,
        141,
        140,
        139,
        127,
        126,
        138,
        87,
        88,
        100,
        112,
        124,
        99,
        122,
        134,
        121,
        133,
        132,
        120,
    ]  # Right Dead Ends
    dead_ends = left_dead_ends + right_dead_ends
    # ------------------------------------------------------#

    # -------CHOOSE REGION NAMES (KEY), MAPPED TO LIST OF GRID NODES IN THAT REGION (VALUE)--------#
    region_mapping = {
        "target_zone": target_zone,
        "entry_zone": entry_zone,
        "reward_path": reward_path,
        "dead_ends": dead_ends,
        "neutral_zone": neutral_zone,
        "loops": loops,
    }

    # -------CHOOSE REGION NAMES (KEY), MAPPED TO LENGTH OF GRID NODES IN THAT REGION (VALUE)--------#
    region_lengths = {
        "entry_zone": len(entry_zone),
        "loops": len(loops),
        "dead_ends": len(dead_ends),
        "neutral_zone": len(neutral_zone),
        "reward_path": len(reward_path),
        "target_zone": len(target_zone),
    }

    # ----------------NODE-TYPES-------------------------#
    decision_reward = [20, 32, 17, 14, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
    nondecision_reward = [34, 21, 31, 30, 4, 3, 62, 61, 73, 74, 75, 76, 102, 125, 136, 123, 97]
    corner_reward = [22, 29, 5, 2, 26, 27, 72, 101, 103, 113, 137, 135, 111, 108, 96]
    decision_nonreward = [100, 71, 12, 24, 42, 106, 92, 119]
    nondecision_nonreward = [
        35,
        23,
        18,
        15,
        28,
        49,
        127,
        140,
        141,
        129,
        126,
        122,
        121,
        99,
        112,
        45,
        58,
        70,
        69,
        83,
        56,
        44,
        13,
        38,
        52,
        64,
        65,
        54,
        55,
        78,
        79,
        80,
        81,
        94,
        91,
        90,
        104,
        131,
    ]
    corner_nonreward = [
        11,
        10,
        9,
        8,
        6,
        7,
        19,
        16,
        40,
        48,
        50,
        139,
        142,
        130,
        128,
        138,
        134,
        133,
        132,
        120,
        88,
        87,
        124,
        33,
        57,
        59,
        95,
        68,
        0,
        1,
        36,
        25,
        37,
        66,
        53,
        41,
        43,
        67,
        82,
        105,
        93,
        116,
        117,
        118,
        107,
        143,
    ]
    entry_zone = [47, 46]
    target_zone = [84, 85, 86]
    decision_3way = [20, 17, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
    decision_4way = [32, 14]

    node_type_mapping = {
        "decision_reward": decision_reward,
        "nondecision_reward": nondecision_reward,
        "corner_reward": corner_reward,
        "decision_nonreward": decision_nonreward,
        "nondecision_nonreward": nondecision_nonreward,
        "corner_nonreward": corner_nonreward,
        "entry_zone": entry_zone,
        "target_zone": target_zone,
        "decision_3way": decision_3way,
        "decision_4way": decision_4way,
    }

    # Create config dictionary and save config.yaml in project path
    config = {
        "project_name": project_name,
        "project_path_full": str(project_path_full),
        "trial_type": trial_type,
        "file_ext": file_ext,
        "video_type": video_type,
        "dlc_scorer": dlc_scorer,
        "bodyparts": bodyparts,
        "experimental_groups": experimental_groups,
        "palette": palette,
        "sessions": sessions_dict,
        "region_mapping": region_mapping,
        "region_lengths": region_lengths,
        "node_type_mapping": node_type_mapping,
    }

    with open(project_path_full / "config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

    return config
