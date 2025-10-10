'''
    DATA PREPROCESSING
    Author: Shreya Bangera
    Goal: 
       ├── Concatenating all Pose estimation CSV files
       ├── Preprocessing all the tracking data
'''

import pandas as pd
import numpy as np
from pathlib import Path
import os

from compass_labyrinth.constants import (
    NODE_TYPE_MAPPING,
    REGION_MAPPING,
    ADJACENCY_MATRIX,
)


##################################################################
# Concatenating all Pose Estimation results
###################################################################

def load_and_preprocess_session_data(filename, bp, DLCscorer, region_mapping):
    """
    Loads DLC-tracked session data and assigns spatial regions based on grid numbers.

    Parameters:
    - filename (str): CSV file path for a session.
    - bp (str): Body part name (e.g., 'sternum').
    - DLCscorer (str): DLC scorer name from the CSV header.
    - region_mapping (dict): Dictionary mapping region names to grid number lists.

    Returns:
    - pd.DataFrame: Cleaned and region-labeled DataFrame for the session.
    """
    dflin = pd.read_csv(filename, index_col=None, header=[0, 1, 2], skipinitialspace=True)

    # Extract relevant columns
    dflin = dflin.loc[:, [(DLCscorer, bp, "x"),
                          (DLCscorer, bp, "y"),
                          (DLCscorer, bp, "Grid Number"),
                          (DLCscorer, bp, "likelihood")]]
    dflin.columns = ['x', 'y', 'Grid Number', 'likelihood']
    dflin['S_no'] = np.arange(1, len(dflin) + 1)

    # Filter: tracking likelihood and grid presence
    dflin = dflin.fillna(-1)
    dflin = dflin[(dflin['likelihood'] > 0.6) & (dflin['Grid Number'] != -1)].copy()
    dflin.reset_index(drop=True, inplace=True)

    # Assign regions from dictionary
    dflin['Region'] = 'Unknown'
    for region_name, grid_list in region_mapping.items():
        dflin.loc[dflin['Grid Number'].isin(grid_list), 'Region'] = region_name

    return dflin


def compile_mouse_sessions(mouseinfo, pose_est_csv_filepath, DLCscorer, bp, region_mapping=REGION_MAPPING):
    """
    Compiles all sessions into a single DataFrame.

    Parameters:
    - mouseinfo (pd.DataFrame): Metadata for mouse sessions.
    - videofile_path (str): Path to input session CSVs.
    - DLCscorer (str): DLC scorer name.
    - bp (str): Body part name (e.g., 'sternum').
    - region_mapping (dict): Region name → grid number list.

    Returns:
    - pd.DataFrame: Combined session dataframe with Region, Genotype, Sex.
    """
    li_group = []

    for sess in mouseinfo['Session #'].unique():
        session_name = f"Session-{int(sess)}"
        filename = os.path.join(pose_est_csv_filepath, f"{session_name}withGrids.csv")
        df = load_and_preprocess_session_data(filename, bp, DLCscorer, region_mapping)
        df['Session'] = sess
        li_group.append(df)

    df_comb = pd.concat(li_group, axis=0, ignore_index=True)
    df_comb['Grid Number'] = df_comb['Grid Number'].astype(int)
    # Map Genotype and Sex
    session_to_genotype = {k: g["Session #"].tolist() for k, g in mouseinfo.groupby("Genotype")}
    inverse_mapping = {session: genotype for genotype, sessions in session_to_genotype.items() for session in sessions}
    df_comb['Genotype'] = df_comb['Session'].map(inverse_mapping)

    session_to_sex = dict(mouseinfo[['Session #', 'Sex']].values)
    df_comb['Sex'] = df_comb['Session'].map(session_to_sex)

    return df_comb


##################################################################
# Preprocessing
###################################################################

def remove_until_initial_node(df, initial_nodes=[47, 46, 34, 22]):
    """
    Removes all rows in the dataframe until the first occurrence of a grid node
    in the provided initial_nodes list.

    Parameters:
    - df (pd.DataFrame): The input session dataframe.
    - initial_nodes (list): List of grid node integers to detect.

    Returns:
    - pd.DataFrame: Truncated dataframe starting from the first initial node.
    """
    if df.iloc[0]['Grid Number'] in initial_nodes:
        return df.copy()

    first_valid_index = df[df['Grid Number'].isin(initial_nodes)].index.min()
    if pd.notna(first_valid_index):
        return df.iloc[first_valid_index:].reset_index(drop=True)
    
    return df.copy()


def remove_invalid_grid_transitions(df, adjacency_matrix=ADJACENCY_MATRIX):
    """
    Removes rows from the dataframe where the transition between consecutive
    grid numbers is not valid (i.e., not adjacent in the adjacency matrix).

    Parameters:
    - df (pd.DataFrame): The session dataframe after initial truncation.
    - adjacency_matrix (pd.DataFrame): Square adjacency matrix with binary values.

    Returns:
    - pd.DataFrame: Cleaned dataframe with only valid grid transitions.
    """
    grid_numbers = list(df['Grid Number'])
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


def preprocess_sessions(df_comb, adjacency_matrix=ADJACENCY_MATRIX, initial_nodes=[47, 46, 34, 22]):
    """
    Full preprocessing pipeline for all sessions: trims to initial nodes and removes invalid transitions.

    Parameters:
    - df_comb (pd.DataFrame): Combined dataframe containing all sessions.
    - adjacency_matrix (pd.DataFrame): Grid adjacency matrix.
    - initial_nodes (list): Nodes that mark the true session start.

    Returns:
    - pd.DataFrame: Fully cleaned and combined dataframe across all sessions.
    """
    preprocessed_sessions = []

    for _, session_df in df_comb.groupby('Session'):
        session_df = session_df.reset_index(drop=True)
        session_df = remove_until_initial_node(session_df, initial_nodes)
        session_df = remove_invalid_grid_transitions(session_df, adjacency_matrix)
        preprocessed_sessions.append(session_df)

    df_all_cleaned = pd.concat(preprocessed_sessions, ignore_index=True)
    df_all_cleaned['Session'] = df_all_cleaned['Session'].astype(int)
    df_all_cleaned['Grid Number'] = df_all_cleaned['Grid Number'].astype(int)

    # Mapping of variable names to NodeType labels
    # key : value pair, key = list name (as in Initializations) & value = column value name decided by user 
    label_mapping = {
        'decision_reward': 'Decision (Reward)',
        'nondecision_reward': 'Non-Decision (Reward)',
        'corner_reward': 'Corner (Reward)',
        'decision_nonreward': 'Decision (Non-Reward)',
        'nondecision_nonreward': 'Non-Decision (Non-Reward)',
        'corner_nonreward': 'Corner (Non-Reward)',
        'entry_zone': 'Entry Nodes',
        'target_zone': 'Target Nodes'
    }
    df_all_cleaned['NodeType'] = 'Unlabeled'

    # Apply mapping to access the list by name
    # Creates the column NodeType based on Grid Numbers
    for var_name, label in label_mapping.items():
        node_list = NODE_TYPE_MAPPING[var_name]
        df_all_cleaned.loc[df_all_cleaned['Grid Number'].isin(node_list), 'NodeType'] = label

    return df_all_cleaned


#######################################################
# Velocity column creation
#######################################################

def ensure_velocity_column(df, x_col='x', y_col='y', velocity_col='Velocity', fps=5):
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

    if 'Session' in df.columns:
        coords = df[[x_col, y_col, 'Session']]
        velocity = (
            coords.groupby('Session', group_keys=False)[[x_col, y_col]]
            .apply(lambda g: np.sqrt(g[x_col].diff()**2 + g[y_col].diff()**2) * fps)
            .fillna(0)
        )
    else:
        velocity = (
            np.sqrt(df[x_col].diff()**2 + df[y_col].diff()**2) * fps
        ).fillna(0)

    df[velocity_col] = velocity
    return df


#########################################################
# Save dataframes to CSV files
#########################################################
def save_preprocessed_to_csv(config: dict, df: pd.DataFrame) -> None:
    """
    Saves Preprocessed data to CSV files

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    df : pd.DataFrame
        Preprocessed DataFrame to save.

    Returns
    -------
    None
    """
    project_path = Path(config["project_path_full"])
    csv_dir = project_path / 'csvs'
    combined_dir = csv_dir / 'combined'
    individual_dir = csv_dir / 'individual'

    # Create folders if they don’t exist
    combined_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)

    # Save combined file
    combined_path = combined_dir / 'Preprocessed_combined_file.csv'
    df.to_csv(combined_path, index=False)
    print(f"Saved combined file: {combined_path}")

    # Save per-session individual files
    for session_id, df_session in df.groupby('Session'):
        file_name = f'Session-{session_id}_preprocessed.csv'
        file_path = individual_dir / file_name
        df_session.to_csv(file_path, index=False)
    print(f"Saved {df['Session'].nunique()} individual session CSVs to: {individual_dir}")
