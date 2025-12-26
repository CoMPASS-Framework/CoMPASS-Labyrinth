"""
Correlation Analysis
Author: Shreya Bangera
Goal:
    ├── Correlation Analysis of Neural features across Optimal and Non-Optimal Transitions at Decision nodes for Successful & Unsuccessful bouts

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import numpy as np
from itertools import chain, combinations, product
from scipy.stats import zscore
from scipy.stats import ttest_ind
import os
from typing import List, Tuple
from scipy.stats import wasserstein_distance
import textwrap


#######################################################
# Assign Bout Numbers
#######################################################


def assign_bout_numbers(df, grid_anchor=47):
    """
    Assign bout numbers based on visits to a specific anchor node (default: Grid Number == 47).
    """
    session_groups = [x for _, x in df.groupby("Session")]
    for dflin in session_groups:
        dflin.reset_index(drop=True, inplace=True)
        dflin["Bout_num"] = 0
        j = 1
        for i in range(len(dflin)):
            if dflin.loc[i, "Grid Number"] != grid_anchor:
                dflin.loc[i, "Bout_num"] = j
            else:
                dflin.loc[i, "Bout_num"] = 0
                j += 1
    return pd.concat(session_groups, axis=0, ignore_index=True)


#######################################################
# Normalization of Features
#######################################################


def normalize_features(df, features):
    """
    Normalize each column in 'features' to a [0, 1] range and store them as {feature}2.
    """
    for feat in features:
        df[f"{feat}2"] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())
    return df


#######################################################
# Compute Correlation Matrix
#######################################################


def get_node_combinations(nodes, size=1):
    """
    Generate combinations of decision nodes of a given size (default = 1).
    """
    return list(chain.from_iterable(combinations(nodes, r) for r in range(1, size + 1)))


def compute_decision_node_metrics(df, decision_nodes, features=["gamma", "theta", "Velocity"], grid_anchor=47):
    """
    Computes per-bout metrics including transition probabilities and feature medians for each decision node.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least columns: 'Session', 'Genotype', 'Grid Number', 'Region', and features.
    decision_nodes : list
        List of decision node grid numbers.
    features : list
        List of feature column names to normalize and summarize.
    grid_anchor : int
        Grid number that marks the start of a new bout.

    Returns
    -------
    index_df : pd.DataFrame
        Summary DataFrame with probability and feature metrics per bout.
    """
    # Assign bout numbers and normalize features
    df = assign_bout_numbers(df, grid_anchor=grid_anchor)
    df = normalize_features(df, features)

    # Generate combinations
    node_combos = get_node_combinations(decision_nodes, size=1)

    # Define columns
    prob_cols = [f"Prob_D{c}" for c in node_combos]
    opt_cols = [f"{feat}_Opt_D{c}" for c in node_combos for feat in features]
    nonopt_cols = [f"{feat}_NonOpt_D{c}" for c in node_combos for feat in features]

    # Initialize output DataFrame
    index_df = pd.DataFrame(
        columns=["Session", "Genotype", "Bout_no", "Successful_bout"] + prob_cols + opt_cols + nonopt_cols
    )

    row_idx = 0
    for _, sess_df in df.groupby("Session"):
        bouts = [x for _, x in sess_df.groupby("Bout_num")]
        if len(bouts) > 0 and bouts[0]["Bout_num"].iloc[0] == 0:
            bouts.pop(0)

        for bout_num, bout in enumerate(bouts, start=1):
            index_df.loc[row_idx, "Session"] = bout["Session"].iloc[0]
            index_df.loc[row_idx, "Genotype"] = bout["Genotype"].iloc[0]
            index_df.loc[row_idx, "Bout_no"] = bout_num
            index_df.loc[row_idx, "Successful_bout"] = (
                "Successful" if "Target Zone" in bout["Region"].values else "Unsuccessful"
            )

            for combo in node_combos:
                total, opt, opt_idx, nonopt_idx = 0, 0, [], []
                for node in combo:
                    node_locs = bout.index[bout["Grid Number"] == node].tolist()
                    for idx in node_locs:
                        future = bout.index[bout.index > idx]
                        next_diff = next(
                            (j for j in future if bout.loc[j, "Grid Number"] != bout.loc[idx, "Grid Number"]), None
                        )
                        if next_diff is not None:
                            next_region = bout.loc[next_diff, "Region"]
                            total += 1
                            if next_region == "Reward Path":
                                opt += 1
                                opt_idx.append(next_diff)
                            else:
                                nonopt_idx.append(next_diff)

                index_df.loc[row_idx, f"Prob_D{combo}"] = opt / total if total > 0 else np.nan

                # Compute feature medians
                if opt_idx:
                    opt_nodes = bout.loc[opt_idx]
                    for feat in features:
                        index_df.loc[row_idx, f"{feat}_Opt_D{combo}"] = opt_nodes[f"{feat}2"].median()
                if nonopt_idx:
                    nonopt_nodes = bout.loc[nonopt_idx]
                    for feat in features:
                        index_df.loc[row_idx, f"{feat}_NonOpt_D{combo}"] = nonopt_nodes[f"{feat}2"].median()

            row_idx += 1

    return index_df


#######################################################
# Plot Correlation heatmaps
#######################################################


def plot_feature_corr_heatmaps(index_df, feature_set, decision_nodes, successful_only=True, bout_ranges=None):
    """
    Plot correlation heatmaps of features across bout number ranges.

    Parameters
    ----------
    index_df : pd.DataFrame
        Output of `compute_decision_node_metrics`.
    feature_set : list of str
        List of features to include (e.g., ['gamma', 'theta', 'Velocity']).
    decision_nodes : list of int
        Node list used to construct column names.
    successful_only : bool
        Whether to restrict heatmaps to successful bouts only.
    """

    node_combinations = get_node_combinations(decision_nodes)
    feature_cols = [f"{feat}_Opt_D{combo}" for feat in feature_set for combo in node_combinations]

    ranges = bout_ranges
    plt.figure(figsize=(22, 18))

    for i, (start, end) in enumerate(ranges):
        subset = index_df[(index_df["Bout_no"] >= start) & (index_df["Bout_no"] <= end)]
        if successful_only:
            subset = subset[subset["Successful_bout"] == "Successful"]

        plt.subplot(5, 5, i + 1)
        corr = subset[feature_cols].corr()
        sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0)
        plt.title(f"Bout Range {start}-{end}", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.suptitle("Feature Correlation Heatmaps Across Bout Ranges", fontsize=16, y=1.02)
    plt.show()


#######################################################
# Velocity column creation
#######################################################


def ensure_velocity_column(df, x_col="x", y_col="y", velocity_col="Velocity"):
    """
    Add a Velocity column to the DataFrame if it doesn't already exist.
    Velocity is computed as Euclidean distance between successive (x, y) points.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with x/y coordinate columns.
    x_col : str
        Column name for x-coordinate.
    y_col : str
        Column name for y-coordinate.
    velocity_col : str
        Name of the velocity column to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with Velocity column added (if it wasn't already present).
    """
    if velocity_col in df.columns:
        print(f"'{velocity_col}' column already exists. Skipping velocity computation.")
        return df

    # Group by session if available, else compute over the entire DataFrame
    if "Session" in df.columns:
        df[velocity_col] = df.groupby("Session", group_keys=False).apply(
            lambda g: np.sqrt(g[x_col].diff() ** 2 + g[y_col].diff() ** 2).fillna(0)
        )
    else:
        df[velocity_col] = np.sqrt(df[x_col].diff() ** 2 + df[y_col].diff() ** 2).fillna(0)

    return df
