"""
DATA PREPROCESSING
Author: Shreya Bangera
Goal:
   ├── Concatenating all Pose estimation CSV files
   ├── Preprocessing all the tracking data
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import os

from compass_labyrinth.utils import load_cohort_metadata
from compass_labyrinth.constants import (
    NODE_TYPE_MAPPING,
    REGION_MAPPING,
    ADJACENCY_MATRIX,
)


##################################################################
# Concatenating all Pose Estimation results
###################################################################


def load_and_preprocess_session_data(
    filename: str,
    bp: str,
    region_mapping: dict = REGION_MAPPING,
) -> pd.DataFrame:
    """
    Loads pose estimation data from NetCDF and assigns spatial regions based on grid numbers.

    Parameters
    -----------
    filename : str
        NetCDF file path for a session.
    bp : str
        Body part name (e.g., 'sternum').
    region_mapping : dict
        Dictionary mapping region names to grid number lists.

    Returns
    --------
    pd.DataFrame
        Cleaned and region-labeled DataFrame for the session.
    """
    # Load NetCDF file
    ds = xr.open_dataset(filename)

    # Extract data for specific bodypart from individual_0
    # position has shape (time, space) where space contains [x, y]
    position = ds.sel(individuals="individual_0", keypoints=bp).position.values
    confidence = ds.sel(individuals="individual_0", keypoints=bp).confidence.values
    grid_numbers = ds.sel(individuals="individual_0", keypoints=bp).grid_number.values

    # Close dataset to free memory
    ds.close()

    # Create DataFrame
    dflin = pd.DataFrame(
        {
            "x": position[:, 0],  # x coordinates (first spatial dimension)
            "y": position[:, 1],  # y coordinates (second spatial dimension)
            "grid_number": grid_numbers,
            "likelihood": confidence,
        }
    )
    dflin["s_no"] = np.arange(1, len(dflin) + 1)

    # Filter: tracking likelihood and grid presence
    dflin = dflin.fillna(-1)
    dflin = dflin[(dflin["likelihood"] > 0.6) & (dflin["grid_number"] != -1)].copy()
    dflin.reset_index(drop=True, inplace=True)

    # Assign regions from dictionary
    dflin["region"] = "Unknown"
    for region_name, grid_list in region_mapping.items():
        dflin.loc[dflin["grid_number"].isin(grid_list), "region"] = region_name

    return dflin


def compile_mouse_sessions(
    config: dict,
    bp: str,
    region_mapping: dict = REGION_MAPPING,
) -> pd.DataFrame:
    """
    Compiles all sessions into a single DataFrame.

    Parameters
    -----------
    config : dict
        Project configuration dictionary.
    bp : str
        Body part name (e.g., 'sternum').
    region_mapping : dict
        Region name -> grid number list.

    Returns
    --------
    pd.DataFrame
        Combined session dataframe with Region, Genotype, Sex.
        (Sessions with missing files are skipped and reported.)
    """
    pose_est_filepath = Path(config["project_path_full"]) / "data" / "dlc_results"
    dlc_scorer = config["dlc_scorer"]
    cohort_metadata = load_cohort_metadata(config)

    cohort_metadata = cohort_metadata.copy()
    cohort_metadata.columns = cohort_metadata.columns.str.strip().str.lower()

    if "session" in cohort_metadata.columns:
        session_col = "session"
    elif "session #" in cohort_metadata.columns:
        session_col = "session #"
    else:
        raise ValueError(
            f"No session column found in cohort metadata. "
            f"Columns found: {list(cohort_metadata.columns)}"
        )

    genotype_col = "genotype" if "genotype" in cohort_metadata.columns else None
    sex_col = "sex" if "sex" in cohort_metadata.columns else None

    li_group = []
    missing_sessions = []

    for sess in cohort_metadata[session_col].dropna().unique():
        session_num = int(sess)
        session_id = f"Session-{session_num}"

        filename_csv_underscore = pose_est_filepath / f"{session_id}_withGrids.csv"
        filename_csv_space = pose_est_filepath / f"{session_id} withGrids.csv"
        filename_nc = pose_est_filepath / f"{session_id}.nc"

        if filename_csv_underscore.exists():
            filename = filename_csv_underscore
            filetype = "csv"
        elif filename_csv_space.exists():
            filename = filename_csv_space
            filetype = "csv"
        elif filename_nc.exists():
            filename = filename_nc
            filetype = "nc"
        else:
            missing_sessions.append(session_num)
            print(
                f"[WARN] Skipping Session {session_num}: no matching file found "
                f"(tried '{filename_csv_underscore.name}', "
                f"'{filename_csv_space.name}', '{filename_nc.name}')"
            )
            continue

        if filetype == "csv":
            df = load_and_preprocess_session_data(
                str(filename),
                bp,
                region_mapping,
            )
        else:
            df = load_and_preprocess_session_data(
                str(filename),
                bp,
                region_mapping,
            )

        df["Session"] = session_num
        li_group.append(df)

    if not li_group:
        raise RuntimeError(
            "No session files were loaded. "
            "All sessions were missing."
        )

    df_comb = pd.concat(li_group, axis=0, ignore_index=True)

    if "Grid Number" in df_comb.columns:
        df_comb["Grid Number"] = pd.to_numeric(df_comb["Grid Number"], errors="coerce")
        df_comb = df_comb.dropna(subset=["Grid Number"]).copy()
        df_comb["Grid Number"] = df_comb["Grid Number"].astype(int)

    if genotype_col is not None:
        session_to_genotype = dict(
            cohort_metadata[[session_col, genotype_col]].drop_duplicates().values
        )
        df_comb["Genotype"] = df_comb["Session"].map(session_to_genotype)

    if sex_col is not None:
        session_to_sex = dict(
            cohort_metadata[[session_col, sex_col]].drop_duplicates().values
        )
        df_comb["Sex"] = df_comb["Session"].map(session_to_sex)

    if missing_sessions:
        missing_sessions_sorted = sorted(missing_sessions)
        print(
            "\n[SUMMARY] The following sessions were skipped because no file was found:\n"
            f"{missing_sessions_sorted}\n"
            f"Total skipped: {len(missing_sessions_sorted)}"
        )

    # make all column names lowercase with underscores instead of spaces
    df_comb.columns = (
        df_comb.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df_comb


##################################################################
# OLD CSV-BASED FUNCTIONS - DEPRECATED
##################################################################


# # TODO - to be removed
# def load_and_preprocess_session_data_old(
#     filename: str,
#     bp: str,
#     DLCscorer: str,
#     region_mapping: dict = REGION_MAPPING,
# ) -> pd.DataFrame:
#     """
#     [DEPRECATED] Loads DLC-tracked session data from CSV and assigns spatial regions.
#     Use load_and_preprocess_session_data() for NetCDF files instead.

#     Parameters
#     -----------
#     filename : str
#         CSV file path for a session.
#     bp : str
#         Body part name (e.g., 'sternum').
#     DLCscorer : str
#         DLC scorer name from the CSV header.
#     region_mapping : dict
#         Dictionary mapping region names to grid number lists.

#     Returns
#     --------
#     pd.DataFrame
#         Cleaned and region-labeled DataFrame for the session.
#     """
#     dflin = pd.read_csv(filename, index_col=None, header=[0, 1, 2], skipinitialspace=True)

#     # Extract relevant columns
#     dflin = dflin.loc[
#         :, [(DLCscorer, bp, "x"), (DLCscorer, bp, "y"), (DLCscorer, bp, "grid_number"), (DLCscorer, bp, "likelihood")]
#     ]
#     dflin.columns = ["x", "y", "grid_number", "likelihood"]
#     dflin["s_no"] = np.arange(1, len(dflin) + 1)

#     # Filter: tracking likelihood and grid presence
#     dflin = dflin.fillna(-1)
#     dflin = dflin[(dflin["likelihood"] > 0.6) & (dflin["grid_number"] != -1)].copy()
#     dflin.reset_index(drop=True, inplace=True)

#     # Assign regions from dictionary
#     dflin["region"] = "Unknown"
#     for region_name, grid_list in region_mapping.items():
#         dflin.loc[dflin["grid_number"].isin(grid_list), "region"] = region_name

#     return dflin


# # TODO - to be removed
# def compile_mouse_sessions_old(
#     config: dict,
#     bp: str,
#     region_mapping: dict = REGION_MAPPING,
# ) -> pd.DataFrame:
#     """
#     [DEPRECATED] Compiles all sessions from CSV files into a single DataFrame.
#     Use compile_mouse_sessions() for NetCDF files instead.

#     Parameters
#     -----------
#     config : dict
#         Project configuration dictionary.
#     bp : str
#         Body part name (e.g., 'sternum').
#     region_mapping : dict
#         Region name → grid number list.

#     Returns
#     --------
#     pd.DataFrame
#         Combined session dataframe with region, genotype, sex.
#     """
#     pose_est_csv_filepath = Path(config["project_path_full"]) / "data" / "dlc_results"
#     dlc_scorer = config["dlc_scorer"]
#     cohort_metadata = load_cohort_metadata(config)

#     li_group = []
#     for sess in cohort_metadata["session"].unique():
#         session_name = f"Session-{int(sess)}"
#         filename = os.path.join(pose_est_csv_filepath, f"{session_name}withGrids.csv")
#         df = load_and_preprocess_session_data_old(filename, bp, dlc_scorer, region_mapping)
#         df["session"] = sess
#         li_group.append(df)

#     df_comb = pd.concat(li_group, axis=0, ignore_index=True)
#     df_comb["grid_number"] = df_comb["grid_number"].astype(int)
#     # Map genotype and sex
#     session_to_genotype = {k: g["session"].tolist() for k, g in cohort_metadata.groupby("genotype")}
#     inverse_mapping = {session: genotype for genotype, sessions in session_to_genotype.items() for session in sessions}
#     df_comb["genotype"] = df_comb["session"].map(inverse_mapping)

#     session_to_sex = dict(cohort_metadata[["session", "sex"]].values)
#     df_comb["sex"] = df_comb["session"].map(session_to_sex)
#     return df_comb


##################################################################
# Preprocessing
###################################################################


def remove_until_initial_node(
    df: pd.DataFrame,
    initial_nodes: list = [47, 46, 34, 22],
) -> pd.DataFrame:
    """
    Removes all rows in the dataframe until the first occurrence of a grid node
    in the provided initial_nodes list.

    Parameters
    ----------
    df : pd.DataFrame
        The input session dataframe.
    initial_nodes : list
        List of grid node integers to detect.

    Returns
    ------
    pd.DataFrame
        Truncated dataframe starting from the first initial node.
    """
    if df.iloc[0]["grid_number"] in initial_nodes:
        return df.copy()

    first_valid_index = df[df["grid_number"].isin(initial_nodes)].index.min()
    if pd.notna(first_valid_index):
        return df.iloc[first_valid_index:].reset_index(drop=True)

    return df.copy()


def remove_invalid_grid_transitions(
    df: pd.DataFrame,
    adjacency_matrix: pd.DataFrame = ADJACENCY_MATRIX,
) -> pd.DataFrame:
    """
    Removes rows from the dataframe where the transition between consecutive
    grid numbers is not valid (i.e., not adjacent in the adjacency matrix).

    Parameters
    ----------
    df : pd.DataFrame
        The session dataframe after initial truncation.
    adjacency_matrix : pd.DataFrame
        Square adjacency matrix with binary values.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with only valid grid transitions.
    """
    grid_numbers = list(df["grid_number"])
    drop_indices = []
    x = 0
    num = 0

    while x < len(grid_numbers) - 1:
        from_node = int(grid_numbers[x])
        to_node = int(grid_numbers[x + 1])
        col_name = f'Grid{str(to_node).replace(".0", "")}'

        if adjacency_matrix.loc[from_node, col_name] == 0:
            del grid_numbers[x + 1]
            drop_indices.append(num + 1)
        else:
            x += 1
        num += 1

    df_cleaned = df.drop(df.index[drop_indices]).reset_index(drop=True)
    return df_cleaned


def preprocess_sessions(
    df_comb: pd.DataFrame,
    adjacency_matrix: pd.DataFrame = ADJACENCY_MATRIX,
    initial_nodes: list = [47, 46, 34, 22],
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for all sessions: trims to initial nodes and removes invalid transitions.
    """
    df_comb = df_comb.copy()

    # handle column-name variants
    if "session" in df_comb.columns:
        session_col = "session"
    elif "Session" in df_comb.columns:
        session_col = "Session"
    else:
        raise KeyError(f"No session column found. Columns are: {df_comb.columns.tolist()}")

    if "grid_number" in df_comb.columns:
        grid_col = "grid_number"
    elif "Grid Number" in df_comb.columns:
        grid_col = "Grid Number"
    else:
        raise KeyError(f"No grid number column found. Columns are: {df_comb.columns.tolist()}")

    preprocessed_sessions = []

    for _, session_df in df_comb.groupby(session_col):
        session_df = session_df.reset_index(drop=True)
        session_df = remove_until_initial_node(session_df, initial_nodes)
        session_df = remove_invalid_grid_transitions(session_df, adjacency_matrix)
        preprocessed_sessions.append(session_df)

    df_all_cleaned = pd.concat(preprocessed_sessions, ignore_index=True)

    # standardize names after processing
    df_all_cleaned = df_all_cleaned.rename(
        columns={
            session_col: "session",
            grid_col: "grid_number",
        }
    )

    df_all_cleaned["session"] = df_all_cleaned["session"].astype(int)
    df_all_cleaned["grid_number"] = df_all_cleaned["grid_number"].astype(int)

    label_mapping = {
        "decision_reward": "Decision (Reward)",
        "nondecision_reward": "Non-Decision (Reward)",
        "corner_reward": "Corner (Reward)",
        "decision_nonreward": "Decision (Non-Reward)",
        "nondecision_nonreward": "Non-Decision (Non-Reward)",
        "corner_nonreward": "Corner (Non-Reward)",
        "entry_zone": "Entry Nodes",
        "target_zone": "Target Nodes",
    }

    df_all_cleaned["node_type"] = "Unlabeled"

    for var_name, label in label_mapping.items():
        node_list = NODE_TYPE_MAPPING[var_name]
        df_all_cleaned.loc[
            df_all_cleaned["grid_number"].isin(node_list), "node_type"
        ] = label

    return df_all_cleaned

#######################################################
# Velocity column creation
#######################################################


def ensure_velocity_column(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    velocity_col: str = "velocity",
    fps: float = 5,
) -> pd.DataFrame:
    """
    Adds a velocity column to the DataFrame if it doesn't already exist.
    Velocity is calculated as Euclidean displacement between frames, scaled by fps.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with coordinate data.
    x_col : str
        Name of x-coordinate column.
    y_col : str
        Name of y-coordinate column.
    velocity_col : str
        Name of the new velocity column to add.
    fps : float
        Frames per second to scale velocity to units/sec.

    Returns
    -------
    pd.DataFrame
        DataFrame with velocity column added.
    """
    if velocity_col in df.columns:
        print(f"'{velocity_col}' already exists. Skipping velocity computation.")
        return df.copy()

    if fps <= 0:
        raise ValueError("fps must be greater than 0.")

    df = df.copy()

    if "session" in df.columns:
        coords = df[[x_col, y_col, "session"]]
        velocity = (
            coords.groupby("session", group_keys=False)[[x_col, y_col]]
            .apply(lambda g: np.sqrt(g[x_col].diff() ** 2 + g[y_col].diff() ** 2) * fps)
            .fillna(0)
        )
    else:
        velocity = (np.sqrt(df[x_col].diff() ** 2 + df[y_col].diff() ** 2) * fps).fillna(0)

    df[velocity_col] = velocity
    return df


#########################################################
# Save dataframes to CSV files
#########################################################
def save_preprocessed_to_csv(
    config: dict,
    df: pd.DataFrame,
    save_combined: bool = True,
    save_individual: bool = True,
) -> None:
    """
    Saves preprocessed data to CSV files.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    df : pd.DataFrame
        Preprocessed DataFrame to save.
    save_combined : bool, optional
        If True, saves a single combined CSV file. Default is True.
    save_individual : bool, optional
        If True, saves per-session individual CSV files. Default is True.

    Returns
    -------
    None
    """
    project_path = Path(config["project_path_full"])
    csv_dir = project_path / "csvs"
    combined_dir = csv_dir / "combined"
    individual_dir = csv_dir / "individual"

    # detect session column
    if "Session" in df.columns:
        session_col = "Session"
    elif "session" in df.columns:
        session_col = "session"
    else:
        raise KeyError(f"No session column found. Columns are: {df.columns.tolist()}")

    # Create base csv directory only if needed
    if save_combined or save_individual:
        csv_dir.mkdir(parents=True, exist_ok=True)

    # Save combined file
    if save_combined:
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "Preprocessed_combined_file.csv"
        df.to_csv(combined_path, index=False)
        print(f"Saved combined file: {combined_path}")

    # Save per-session files
    if save_individual:
        individual_dir.mkdir(parents=True, exist_ok=True)

        for session_id, df_session in df.groupby(session_col):
            file_name = f"Session{int(session_id):04d}_preprocessed.csv"
            file_path = individual_dir / file_name
            df_session.to_csv(file_path, index=False)

        print(
            f"Saved {df[session_col].nunique()} individual session CSVs to: {individual_dir}"
        )