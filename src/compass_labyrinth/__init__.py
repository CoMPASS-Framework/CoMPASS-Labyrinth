from pathlib import Path
import pandas as pd
import datetime
import shutil
import yaml
import os
import re
from movement.io import load_poses

from .utils import load_project, update_dataset_with_grid_positions
from .behavior.pose_estimation.dlc_utils import (
    import_cohort_metadata,
    validate_metadata,
    save_first_frame,
    check_preprocessing_status,
)


def normalize_session_name(name: str) -> str:
    """
    Convert names like Session-4, Session4, Session_4, Session0004
    into Session0004.
    """
    m = re.search(r"Session[-_]?(\d+)", str(name), flags=re.IGNORECASE)
    if not m:
        return str(name)
    return f"Session{int(m.group(1)):04d}"


def init_project(
    project_name: str,
    project_path: Path | str,
    source_data_path: Path | str,
    user_metadata_file_path: Path | str,
    trial_type: str = "Labyrinth_DSI",
    file_ext: str = ".h5",
    video_type: str = ".mp4",
    sampling_rate: int = 30,
    dlc_scorer: str = "DLC_resnet50_LabyrinthMar13shuffle1_1000000",
    experimental_groups: list = ["A", "B", "C", "D"],
    palette: str = "grey",
) -> tuple[dict, pd.DataFrame]:
    """
    Initializes project for the CoMPASS-Labyrinth analysis, including:
    - Setting up directory structure
    - Copying user metadata file to project directory
    - Creating a config.yaml file with project parameters
    - Saving normalized processed pose files as Session000X_withGrids.csv
      into data/dlc_results
    - Copying existing withGrids CSV files into data/dlc_results
      with normalized names

    IMPORTANT:
    If the project folder already exists, this function prints a message and
    returns the already existing project without processing again.
    """
    if not project_name.replace("_", "").isalnum():
        raise ValueError("Project name must be alphanumeric and can only contain underscores.")

    source_data_path = Path(source_data_path).resolve()
    if not source_data_path.exists():
        raise ValueError(f"Source data path {source_data_path} does not exist.")

    # TODO - re-enable later if needed
    # check_preprocessing_status(source_data_path)

    project_path = Path(project_path).resolve()
    project_path_full = project_path / project_name

    if not project_path_full.exists():
        project_path_full.mkdir(parents=True, exist_ok=True)
        print(f"Project path does not exist. Creating directory at {project_path_full}")
    else:
        print(f"Project already exists at {project_path_full}")
        return load_project(project_path_full)

    all_dirs = {
        "videos": project_path_full / "videos",
        "videos_original": project_path_full / "videos" / "original_videos",
        "frames": project_path_full / "videos" / "frames",
        "data": project_path_full / "data",
        "dlc_results": project_path_full / "data" / "dlc_results",
        "dlc_cropping": project_path_full / "data" / "dlc_cropping_bounds",
        "grid_files": project_path_full / "data" / "grid_files",
        "grid_boundaries": project_path_full / "data" / "grid_boundaries",
        "metadata": project_path_full / "data" / "metadata",
        "eeg_edfs": project_path_full / "data" / "processed_eeg_edfs",
        "figures": project_path_full / "figures",
        "csvs": project_path_full / "csvs",
        "csvs_individual": project_path_full / "csvs" / "individual",
        "csvs_combined": project_path_full / "csvs" / "combined",
        "results": project_path_full / "results",
        "results_task_performance": project_path_full / "results" / "task_performance",
        "results_simulation_agent": project_path_full / "results" / "simulation_agent",
        "results_compass_level_1": project_path_full / "results" / "compass_level_1",
        "results_compass_level_2": project_path_full / "results" / "compass_level_2",
        "results_ephys_compass": project_path_full / "results" / "ephys_compass",
    }
    for dir_path in all_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    pose_est_dest_path = project_path_full / "data" / "dlc_results"

    # Find DLC pose-estimation files
    pe_files = sorted(
        [f.resolve() for f in source_data_path.glob(f"*{dlc_scorer}*{file_ext}")],
        key=lambda f: f.name,
    )

    print(f"source_data_path: {source_data_path}")
    print(f"pose_est_dest_path: {pose_est_dest_path}")
    print(f"Looking for pose files with pattern: *{dlc_scorer}*{file_ext}")
    print("Pose files found:", [f.name for f in pe_files])

    session_names = [
        normalize_session_name(f.stem.replace(dlc_scorer, ""))
        for f in pe_files
    ]
    print("Normalized session names:", session_names)

    bodyparts = []
    processed_sessions = set()

    # Create normalized Session000X_withGrids.csv files from DLC pose files
    for file in pe_files:
        raw_session_name = file.stem.replace(dlc_scorer, "")
        session_name = normalize_session_name(raw_session_name)
        dest_file = pose_est_dest_path / f"{session_name}_withGrids.csv"

        try:
            ds = load_poses.from_dlc_file(
                file_path=file,
                fps=sampling_rate,
            )

            # Use raw source naming to find the shapefile first
            grid_file_aux = [f for f in source_data_path.glob(f"{raw_session_name}*.shp")]

            # Fallback: also try normalized session name
            if len(grid_file_aux) == 0:
                grid_file_aux = [f for f in source_data_path.glob(f"{session_name}*.shp")]

            if len(grid_file_aux) == 0:
                raise FileNotFoundError(
                    f"Grid shapefile for session {session_name} not found. "
                    f"Tried patterns '{raw_session_name}*.shp' and '{session_name}*.shp' "
                    f"in {source_data_path}"
                )

            grid_file_path = grid_file_aux[0].resolve()
            ds = update_dataset_with_grid_positions(
                ds=ds,
                grid_file_path=grid_file_path,
            )

            # Convert to dataframe and save as CSV
            df_pose = ds.to_dataframe().reset_index()

            if "confidence" in df_pose.columns and "likelihood" not in df_pose.columns:
                df_pose = df_pose.rename(columns={"confidence": "likelihood"})

            df_pose.to_csv(dest_file, index=False)
            processed_sessions.add(session_name)
            print(f"Saved CSV: {dest_file.name}")

            if len(bodyparts) == 0 and hasattr(ds, "keypoints"):
                bodyparts = ds.keypoints.values.tolist()
            elif hasattr(ds, "keypoints"):
                if ds.keypoints.values.tolist() != bodyparts:
                    raise ValueError(
                        f"Bodyparts in file {file} do not match previously read bodyparts."
                    )

        except Exception as e:
            print(f"[WARN] Failed to process {file.name}: {e}")

    # Copy existing withGrids CSV files only for sessions not already processed
    with_grid_files = sorted(
        list(source_data_path.glob("*withGrids.csv")) +
        list(source_data_path.glob("*_withGrids.csv")),
        key=lambda f: f.name,
    )
    print("withGrids CSV files found:", [f.name for f in with_grid_files])

    for file in with_grid_files:
        try:
            normalized_session = normalize_session_name(file.stem)
            if normalized_session in processed_sessions:
                continue

            dest_file = pose_est_dest_path / f"{normalized_session}_withGrids.csv"
            shutil.copy2(file.resolve(), dest_file)
            print(f"Copied CSV: {file.name} -> {dest_file.name}")
        except Exception as e:
            print(f"[WARN] Failed to copy CSV {file.name}: {e}")

    # Copy all shape files into project grid_files folder
    grid_files_dest = project_path_full / "data" / "grid_files"
    for ext in [".shp", ".dbf", ".shx"]:
        grid_files = [f.resolve() for f in source_data_path.glob(f"*{ext}")]
        for file in grid_files:
            dest_file = grid_files_dest / file.name
            if not dest_file.exists():
                shutil.copy2(file, dest_file)

    # Link videos to project and save first frames
    video_dest_path = project_path_full / "videos" / "original_videos"
    video_files = sorted(
        [f.resolve() for f in source_data_path.glob(f"*{video_type}")],
        key=lambda f: f.name,
    )

    for file in video_files:
        dest_file = video_dest_path / file.name
        if not dest_file.exists():
            os.symlink(file, dest_file)

        save_first_frame(
            video_path=file,
            frames_dir=project_path_full / "videos" / "frames",
        )

    # Load metadata
    user_metadata_file_path = Path(user_metadata_file_path).resolve()
    metadata_df = import_cohort_metadata(
        metadata_path=user_metadata_file_path,
        trial_sheet_name=trial_type,
    )
    validate_metadata(metadata_df)
    metadata_df = pd.DataFrame(metadata_df.to_dict(orient="records"))

    # Save cohort metadata
    metadata_df.to_csv(project_path_full / "cohort_metadata.csv", index=False)

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

    return config, metadata_df