from pathlib import Path
import shutil
import yaml

from .utils import load_config


def init_project(
    project_name: str,
    project_path: Path | str,
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

    # Set up project's base path
    project_path = Path(project_path).resolve()
    project_path_full = project_path / project_name
    if not project_path_full.exists():
        project_path_full.mkdir(parents=True, exist_ok=True)
        print(f"Project path does not exist. Creating directory at {project_path_full}")
    else:
        print(f"Project already exists at {project_path_full}")
        return load_config(project_path_full)

    # Central video location (where all videos are copied for processing)
    videofile_path = project_path_full / "videos" / "original_videos"
    if not videofile_path.exists():
        videofile_path.mkdir(parents=True, exist_ok=True)

    # Pose estimation CSV outputs filepath
    pose_est_csv_path = project_path_full / "data" / "dlc_results"
    if not pose_est_csv_path.exists():
        pose_est_csv_path.mkdir(parents=True, exist_ok=True)

    # Path for all grid based files for a particular Session
    # as part of Level-1 Post-Analysis --> Plot 1: Heatmap Representations of HMM States
    grid_path = project_path_full / "data" / "grid_files"
    if not grid_path.exists():
        grid_path.mkdir(parents=True, exist_ok=True)

    # Copy the user passed metadata to the project's path
    # TODO - later on, we will like to construct this metadata file automatically, instead of requesting from user
    user_metadata_file_path = Path(user_metadata_file_path).resolve()
    metadata_file_path = project_path_full / "metadata.xlsx"
    if not metadata_file_path.exists():
        shutil.copy(user_metadata_file_path, metadata_file_path)

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
    }

    # Save config.yaml in project path
    with open(project_path_full / "config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

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
    Target_Zone = [84, 85, 86]
    Entry_Zone = [47, 46]
    Loops = [
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
    Neutral_Zone = [107, 119, 131, 143]

    # P1, P2, P3 are the 3 sections of the Reward Path
    P1 = [22, 21, 34, 20, 32, 31, 30, 29, 17, 5, 4, 3, 2, 14, 26]
    P2 = [27, 39, 51, 63, 62, 61, 60, 72, 73, 74, 75, 76, 77, 89, 101]
    P3 = [102, 103, 115, 114, 113, 125, 137, 136, 135, 123, 111, 110, 109, 108, 96, 97, 98]
    Reward_Path = P1 + P2 + P3

    Left_Dead_Ends = [10, 11, 23, 35, 9, 8, 6, 7, 19, 18, 15, 16, 28, 40, 50, 49, 48]  # Left Dead Ends
    Right_Dead_Ends = [
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
    Dead_Ends = Left_Dead_Ends + Right_Dead_Ends
    # ------------------------------------------------------#

    # -------CHOOSE REGION NAMES (KEY), MAPPED TO LIST OF GRID NODES IN THAT REGION (VALUE)--------#
    region_mapping = {
        "Target Zone": Target_Zone,
        "Entry Zone": Entry_Zone,
        "Reward Path": Reward_Path,
        "Dead Ends": Dead_Ends,
        "Neutral Zone": Neutral_Zone,
        "Loops": Loops,
    }

    # -------CHOOSE REGION NAMES (KEY), MAPPED TO LENGTH OF GRID NODES IN THAT REGION (VALUE)--------#
    region_lengths = {
        "Entry Zone": len(Entry_Zone),
        "Loops": len(Loops),
        "Dead Ends": len(Dead_Ends),
        "Neutral Zone": len(Neutral_Zone),
        "Reward Path": len(Reward_Path),
        "Target Zone": len(Target_Zone),
    }

    # ----------------NODE-TYPES-------------------------#
    Decision_Reward = [20, 32, 17, 14, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
    NonDecision_Reward = [34, 21, 31, 30, 4, 3, 62, 61, 73, 74, 75, 76, 102, 125, 136, 123, 97]
    Corner_Reward = [22, 29, 5, 2, 26, 27, 72, 101, 103, 113, 137, 135, 111, 108, 96]
    Decision_NonReward = [100, 71, 12, 24, 42, 106, 92, 119]
    NonDecision_NonReward = [
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
    Corner_NonReward = [
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
    Entry_Zone = [47, 46]
    Target_Zone = [84, 85, 86]

    Decision_3way = [20, 17, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
    Decision_4way = [32, 14]

    return config