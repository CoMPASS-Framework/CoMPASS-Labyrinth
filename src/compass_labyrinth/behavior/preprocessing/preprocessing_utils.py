"""
DATA PREPROCESSING
Author: Shreya Bangera
Goal:
   +-- Concatenating all pose estimation session files
   +-- Preprocessing all the tracking data
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

from compass_labyrinth.utils import load_cohort_metadata
from compass_labyrinth.constants import (
    NODE_TYPE_MAPPING,
    REGION_MAPPING,
    ADJACENCY_MATRIX,
)


##################################################################
# Concatenating all Pose Estimation results
##################################################################


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


def load_and_preprocess_session_data(
    filename: str | Path,
    bp: str,
    region_mapping: dict,
) -> pd.DataFrame:
    """
    Load pose estimation data from CSV or NetCDF and assign region labels.
    Handles:
      1. flat CSV
      2. long-form CSV
      3. DLC multi-header wide CSV
      4. .nc files
    """
    filename = Path(filename)
    bp_lower = str(bp).strip().lower()

    if filename.suffix.lower() == ".csv":

        # ---------------------------------------------------------
        # Peek only the header first to decide how to read the file
        # ---------------------------------------------------------
        preview = pd.read_csv(filename, nrows=5, header=None, low_memory=False)

        # DLC multi-header CSV usually starts with scorer/bodyparts/coords rows
        is_dlc_multihdr = False
        if len(preview) >= 3:
            row0 = preview.iloc[0].astype(str).str.lower().tolist()
            row1 = preview.iloc[1].astype(str).str.lower().tolist()
            row2 = preview.iloc[2].astype(str).str.lower().tolist()

            if (
                any("scorer" in x for x in row0)
                and any("bodyparts" in x for x in row1)
                and any("coords" in x for x in row2)
            ):
                is_dlc_multihdr = True

        if is_dlc_multihdr:
            # -----------------------------------------------------
            # Case 3: DLC multi-header wide CSV
            # -----------------------------------------------------
            dlc_df = pd.read_csv(filename, header=[0, 1, 2], low_memory=False)

            flat_cols = []
            for col in dlc_df.columns:
                parts = [str(x).strip().lower() for x in col if str(x) != "nan"]
                flat_cols.append("_".join(parts))

            dlc_df.columns = (
                pd.Index(flat_cols)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )

            x_candidates = [
                c for c in dlc_df.columns
                if f"_{bp_lower}_x" in c or c.endswith(f"{bp_lower}_x")
            ]
            y_candidates = [
                c for c in dlc_df.columns
                if f"_{bp_lower}_y" in c or c.endswith(f"{bp_lower}_y")
            ]
            lik_candidates = [
                c for c in dlc_df.columns
                if f"_{bp_lower}_likelihood" in c or c.endswith(f"{bp_lower}_likelihood")
            ]
            grid_candidates = [
                c for c in dlc_df.columns
                if f"_{bp_lower}_grid_number" in c or c.endswith(f"{bp_lower}_grid_number")
            ]

            if not x_candidates:
                raise ValueError(
                    f"Could not find x column for bodypart '{bp}' in {filename.name}."
                )
            if not y_candidates:
                raise ValueError(
                    f"Could not find y column for bodypart '{bp}' in {filename.name}."
                )
            if not lik_candidates:
                raise ValueError(
                    f"Could not find likelihood column for bodypart '{bp}' in {filename.name}."
                )
            if not grid_candidates:
                raise ValueError(
                    f"Could not find grid_number column for bodypart '{bp}' in {filename.name}. "
                    f"Columns found: {list(dlc_df.columns)}"
                )

            dflin = pd.DataFrame({
                "x": dlc_df[x_candidates[0]],
                "y": dlc_df[y_candidates[0]],
                "grid_number": dlc_df[grid_candidates[0]],
                "likelihood": dlc_df[lik_candidates[0]],
            })

        else:
            # -----------------------------------------------------
            # Non-DLC CSV: read normally
            # -----------------------------------------------------
            dflin = pd.read_csv(filename, low_memory=False)

            dflin.columns = (
                dflin.columns.astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )

            # Case 1: flat CSV
            if {"x", "y", "grid_number", "likelihood"}.issubset(dflin.columns):
                dflin = dflin[["x", "y", "grid_number", "likelihood"]].copy()

            # Case 2: long-form CSV
            elif {"position", "grid_number", "likelihood"}.issubset(dflin.columns):
                if "space" not in dflin.columns:
                    raise ValueError(
                        f"CSV file {filename.name} has 'position' but is missing 'space'. "
                        f"Columns found: {list(dflin.columns)}"
                    )

                if "keypoints" in dflin.columns:
                    dflin = dflin[
                        dflin["keypoints"].astype(str).str.lower() == bp_lower
                    ].copy()

                if "individuals" in dflin.columns:
                    dflin = dflin[
                        dflin["individuals"].astype(str) == "individual_0"
                    ].copy()

                if dflin.empty:
                    raise ValueError(
                        f"No rows found for bodypart '{bp}' in file {filename.name}."
                    )

                pos_wide = (
                    dflin.pivot_table(
                        index="time",
                        columns="space",
                        values="position",
                        aggfunc="first",
                    )
                    .reset_index()
                )

                pos_wide.columns = [str(c).strip().lower() for c in pos_wide.columns]

                if "x" not in pos_wide.columns or "y" not in pos_wide.columns:
                    raise ValueError(
                        f"Could not extract x/y from 'space' column in {filename.name}."
                    )

                aux = (
                    dflin.groupby("time", as_index=False)[["grid_number", "likelihood"]]
                    .first()
                )

                dflin = pos_wide.merge(aux, on="time", how="left")
                dflin = dflin[["x", "y", "grid_number", "likelihood"]].copy()

            else:
                raise ValueError(
                    f"CSV file {filename.name} is missing required columns. "
                    f"Columns found: {list(dflin.columns)}"
                )

    elif filename.suffix.lower() == ".nc":
        ds = xr.open_dataset(filename)

        position = ds.sel(individuals="individual_0", keypoints=bp).position.values
        confidence = ds.sel(individuals="individual_0", keypoints=bp).likelihood.values
        grid_numbers = ds.sel(individuals="individual_0", keypoints=bp).grid_number.values

        ds.close()

        dflin = pd.DataFrame({
            "x": position[:, 0],
            "y": position[:, 1],
            "grid_number": grid_numbers,
            "likelihood": confidence,
        })

    else:
        raise ValueError(f"Unsupported file type: {filename.suffix}")

    dflin["s_no"] = np.arange(1, len(dflin) + 1)

    dflin["x"] = pd.to_numeric(dflin["x"], errors="coerce")
    dflin["y"] = pd.to_numeric(dflin["y"], errors="coerce")
    dflin["grid_number"] = pd.to_numeric(dflin["grid_number"], errors="coerce")
    dflin["likelihood"] = pd.to_numeric(dflin["likelihood"], errors="coerce")

    dflin = dflin.fillna(-1)
    dflin = dflin[
        (dflin["likelihood"] > 0.6) &
        (dflin["grid_number"] != -1)
    ].copy()

    dflin.reset_index(drop=True, inplace=True)
    dflin["grid_number"] = dflin["grid_number"].astype(int)

    dflin["region"] = "unknown"
    for region_name, grid_list in region_mapping.items():
        dflin.loc[dflin["grid_number"].isin(grid_list), "region"] = region_name.lower()

    return dflin


def compile_mouse_sessions(
    config: dict,
    bp: str,
    region_mapping: dict,
) -> pd.DataFrame:
    """
    Compile all sessions into a single dataframe.
    """
    pose_est_filepath = Path(config["project_path_full"]) / "data" / "dlc_results"
    cohort_metadata = load_cohort_metadata(config)

    cohort_metadata = cohort_metadata.copy()
    cohort_metadata.columns = (
        cohort_metadata.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    if "session" in cohort_metadata.columns:
        session_col = "session"
    elif "session_#" in cohort_metadata.columns:
        session_col = "session_#"
    elif "session#" in cohort_metadata.columns:
        session_col = "session#"
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
        session_id = f"Session{session_num:04d}"

        filename_csv_underscore = pose_est_filepath / f"{session_id}_withGrids.csv"
        filename_csv_noscore = pose_est_filepath / f"{session_id}withGrids.csv"
        filename_nc = pose_est_filepath / f"{session_id}.nc"

        if filename_csv_underscore.exists():
            filename = filename_csv_underscore
        elif filename_csv_noscore.exists():
            filename = filename_csv_noscore
        elif filename_nc.exists():
            filename = filename_nc
        else:
            missing_sessions.append(session_num)
            print(
                f"[WARN] Skipping session {session_num}: no matching file found "
                f"(tried '{filename_csv_underscore.name}', "
                f"'{filename_csv_noscore.name}', '{filename_nc.name}')"
            )
            continue

        df = load_and_preprocess_session_data(
            filename=filename,
            bp=bp,
            region_mapping=region_mapping,
        )

        df["session"] = session_num
        li_group.append(df)

    if not li_group:
        raise RuntimeError("No session files were loaded. All sessions were missing.")

    df_comb = pd.concat(li_group, axis=0, ignore_index=True)

    if "grid_number" in df_comb.columns:
        df_comb["grid_number"] = pd.to_numeric(df_comb["grid_number"], errors="coerce")
        df_comb = df_comb.dropna(subset=["grid_number"]).copy()
        df_comb["grid_number"] = df_comb["grid_number"].astype(int)

    if genotype_col is not None:
        session_to_genotype = dict(
            cohort_metadata[[session_col, genotype_col]].drop_duplicates().values
        )
        df_comb["genotype"] = df_comb["session"].map(session_to_genotype)

    if sex_col is not None:
        session_to_sex = dict(
            cohort_metadata[[session_col, sex_col]].drop_duplicates().values
        )
        df_comb["sex"] = df_comb["session"].map(session_to_sex)

    if missing_sessions:
        missing_sessions_sorted = sorted(missing_sessions)
        print(
            "\n[SUMMARY] The following sessions were skipped because no file was found:\n"
            f"{missing_sessions_sorted}\n"
            f"Total skipped: {len(missing_sessions_sorted)}"
        )

    df_comb.columns = (
        df_comb.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df_comb


def compile_mouse_sessions(
    config: dict,
    bp: str,
    region_mapping: dict = REGION_MAPPING,
) -> pd.DataFrame:
    """
    Compiles all sessions into a single DataFrame.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    bp : str
        Body part name (e.g. 'sternum').
    region_mapping : dict
        Region name -> grid number list.

    Returns
    -------
    pd.DataFrame
        Combined session dataframe with region, genotype, sex.
        Sessions with missing files are skipped and reported.
    """
    pose_est_filepath = Path(config["project_path_full"]) / "data" / "dlc_results"
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
        session_id = f"Session{session_num:04d}"

        filename_csv_underscore = pose_est_filepath / f"{session_id}_withGrids.csv"
        filename_csv_noscore = pose_est_filepath / f"{session_id}withGrids.csv"
        filename_nc = pose_est_filepath / f"{session_id}.nc"

        if filename_csv_underscore.exists():
            filename = filename_csv_underscore
        elif filename_csv_noscore.exists():
            filename = filename_csv_noscore
        elif filename_nc.exists():
            filename = filename_nc
        else:
            missing_sessions.append(session_num)
            print(
                f"[WARN] Skipping session {session_num}: no matching file found "
                f"(tried '{filename_csv_underscore.name}', "
                f"'{filename_csv_noscore.name}', '{filename_nc.name}')"
            )
            continue

        df = load_and_preprocess_session_data(
            filename=filename,
            bp=bp,
            region_mapping=region_mapping,
        )

        df["session"] = session_num
        li_group.append(df)

    if not li_group:
        raise RuntimeError(
            "No session files were loaded. All sessions were missing."
        )

    df_comb = pd.concat(li_group, axis=0, ignore_index=True)

    if "grid_number" in df_comb.columns:
        df_comb["grid_number"] = pd.to_numeric(df_comb["grid_number"], errors="coerce")
        df_comb = df_comb.dropna(subset=["grid_number"]).copy()
        df_comb["grid_number"] = df_comb["grid_number"].astype(int)

    if genotype_col is not None:
        session_to_genotype = dict(
            cohort_metadata[[session_col, genotype_col]].drop_duplicates().values
        )
        df_comb["genotype"] = df_comb["session"].map(session_to_genotype)

    if sex_col is not None:
        session_to_sex = dict(
            cohort_metadata[[session_col, sex_col]].drop_duplicates().values
        )
        df_comb["sex"] = df_comb["session"].map(session_to_sex)

    if missing_sessions:
        missing_sessions_sorted = sorted(missing_sessions)
        print(
            "\n[SUMMARY] The following sessions were skipped because no file was found:\n"
            f"{missing_sessions_sorted}\n"
            f"Total skipped: {len(missing_sessions_sorted)}"
        )

    df_comb.columns = (
        df_comb.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df_comb


##################################################################
# Preprocessing
##################################################################


def remove_until_initial_node(
    df: pd.DataFrame,
    initial_nodes: list = [47, 46, 34, 22],
) -> pd.DataFrame:
    """
    Removes all rows in the dataframe until the first occurrence of a grid node
    in the provided initial_nodes list.
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
    Removes rows where transitions between consecutive grid numbers
    are not valid based on the adjacency matrix.
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
    Full preprocessing pipeline for all sessions:
    trims to initial nodes and removes invalid transitions.
    """
    df_comb = df_comb.copy()

    if "session" not in df_comb.columns:
        raise KeyError(f"No session column found. Columns are: {df_comb.columns.tolist()}")

    if "grid_number" not in df_comb.columns:
        raise KeyError(f"No grid_number column found. Columns are: {df_comb.columns.tolist()}")

    preprocessed_sessions = []

    for _, session_df in df_comb.groupby("session"):
        session_df = session_df.reset_index(drop=True)
        session_df = remove_until_initial_node(session_df, initial_nodes)
        session_df = remove_invalid_grid_transitions(session_df, adjacency_matrix)
        preprocessed_sessions.append(session_df)

    df_all_cleaned = pd.concat(preprocessed_sessions, ignore_index=True)

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

    df_all_cleaned["node_type"] = "unlabeled"

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
    Adds a velocity column to the DataFrame if it does not already exist.
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
    """
    project_path = Path(config["project_path_full"])
    csv_dir = project_path / "csvs"
    combined_dir = csv_dir / "combined"
    individual_dir = csv_dir / "individual"

    if "session" not in df.columns:
        raise KeyError(f"No session column found. Columns are: {df.columns.tolist()}")

    if save_combined or save_individual:
        csv_dir.mkdir(parents=True, exist_ok=True)

    if save_combined:
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "preprocessed_combined_file.csv"
        df.to_csv(combined_path, index=False)
        print(f"Saved combined file: {combined_path}")

    if save_individual:
        individual_dir.mkdir(parents=True, exist_ok=True)

        for session_id, df_session in df.groupby("session"):
            file_name = f"Session{int(session_id):04d}_preprocessed.csv"
            file_path = individual_dir / file_name
            df_session.to_csv(file_path, index=False)

        print(
            f"Saved {df['session'].nunique()} individual session CSVs to: {individual_dir}"
        )