import pandas as pd
import numpy as np


# -------------------------------------------------------------------------------------
# Bout number assignment
# --------------------------------------------------------------------------------------
def assign_bouts_per_session(
    df: pd.DataFrame,
    terminal_values=[47],
    bout_col="Bout_ID",
) -> pd.DataFrame:
    """
    Assigns bout numbers to a DataFrame based on terminal grid node values,
    if the specified bout column does not already exist.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame with 'Grid Number' column.
    terminal_values : list
        List of grid numbers that signify end-of-bout events.
    bout_col : str
        Name of the output column to store bout numbers.

    Returns
    --------
    pd.DataFrame
        DataFrame with the specified bout column added or preserved.
    """
    if bout_col in df.columns:
        return df  # Already exists; return unchanged

    df = df.reset_index(drop=True).copy()
    bout_num = 1
    in_non_terminal_phase = False
    bout_nums = []

    for i in range(len(df)):
        current = df.iloc[i]["Grid Number"]

        if current not in terminal_values:
            in_non_terminal_phase = True
        else:
            if in_non_terminal_phase:
                in_non_terminal_phase = False  # End bout

        bout_nums.append(bout_num)

        if current in terminal_values and not in_non_terminal_phase:
            if i > 0:
                prev_vals = df.iloc[:i]["Grid Number"]
                last_terminal_idx = prev_vals[prev_vals.isin(terminal_values)].last_valid_index()

                if last_terminal_idx is not None and any(
                    ~df.iloc[last_terminal_idx + 1 : i]["Grid Number"].isin(terminal_values)
                ):
                    if i + 1 < len(df):
                        bout_num += 1

    df[bout_col] = bout_nums
    return df


# ---------------------------------------------------------------------------------------
# Create Phases from Bout Numbers
# ---------------------------------------------------------------------------------------
def build_phase_map(
    df: pd.DataFrame,
    n_phases: int,
) -> dict:
    phase_map = {}
    sessions = df.Session.unique()
    for sess in sessions:
        df_sess = df[df.Session == sess]
        unique_bouts = df_sess["Bout_ID"].dropna().unique()
        unique_bouts.sort()
        phase_chunks = np.array_split(unique_bouts, n_phases)
        for i, chunk in enumerate(phase_chunks):
            phase_map[(sess, i)] = chunk
    return phase_map
