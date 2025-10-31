"""
TEMPORAL EVOLUTION OF STATE PROBABILITIES
Author: Shreya Bangera
Goal:
    ├── Tracks probability of being in a chosen HMM state over time bins per genotype and session-averaged plots

"""

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import median
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING


warnings.filterwarnings("ignore")


def compute_node_state_medians_over_time(
    df_hmm: pd.DataFrame,
    state_types: list,
    lower_lim: int,
    upper_lim: int,
    bin_size: int,
    decision_nodes: str = "decision_reward",
    nondecision_nodes: str = "nondecision_reward",
) -> pd.DataFrame:
    """
    Compute time-binned medians of HMM state proportions for decision and non-decision nodes.

    Parameters:
    -----------
    df_hmm : pd.DataFrame
        Dataframe with 'Genotype', 'Session', 'HMM_State', and 'Grid Number'.
    state_types : list
        List of HMM states to compute proportions for (e.g., [2])
    lower_lim : int
        Range of rows to consider per session
    upper_lim : int
        Range of rows to consider per session
    bin_size : int
        Number of rows per time bin
    decision_nodes : str
        List of Grid Number values representing decision nodes
    nondecision_nodes : str
        List of Grid Number values representing non-decision nodes

    Returns:
    --------
    pd.DataFrame
        Dataframe with median proportions per session, time bin, node type, and genotype
    """
    li_node_genotype = []
    decision_nodes_ids = NODE_TYPE_MAPPING.get(decision_nodes, [])
    nondecision_nodes_ids = NODE_TYPE_MAPPING.get(nondecision_nodes, [])
    for genotype in df_hmm["Genotype"].unique():
        sess_li = [x for _, x in df_hmm[df_hmm["Genotype"] == genotype].groupby("Session")]

        for node_type_list, node_type_label in zip(
            [decision_nodes_ids, nondecision_nodes_ids], ["Decision node", "Non-Decision node"]
        ):
            med_df = pd.DataFrame(columns=["Time_Bins", "Session", "Median_Probability"])

            row_index = 1
            for k in range(lower_lim, upper_lim, bin_size):
                for session_df in sess_li:
                    session_df = session_df.reset_index(drop=True)
                    df_subset = session_df.iloc[k : k + bin_size, :]

                    # Count by HMM state
                    st_cnt = df_subset.groupby(["Grid Number", "HMM_State"]).size().rename("cnt").reset_index()
                    gn_cnt = df_subset.groupby(["Grid Number"]).size().rename("tot").reset_index()
                    x_y = df_subset.groupby(["Grid Number"]).agg({"x": "mean", "y": "mean"}).reset_index()

                    state_count = st_cnt.merge(gn_cnt, on="Grid Number", how="left")
                    state_count["prop"] = state_count["cnt"] / state_count["tot"]
                    state_count = state_count.merge(x_y, on="Grid Number", how="left")

                    subset = state_count[
                        (state_count["HMM_State"].isin(state_types)) & (state_count["Grid Number"].isin(node_type_list))
                    ]

                    if not subset.empty:
                        med_val = median(subset["prop"])
                    else:
                        med_val = 0

                    med_df.loc[row_index, "Median_Probability"] = med_val
                    med_df.loc[row_index, "Session"] = session_df["Session"].unique()[0]
                    med_df.loc[row_index, "Time_Bins"] = k + bin_size
                    row_index += 1

            med_df = med_df[med_df["Median_Probability"] != 0]
            med_df["Node Type"] = node_type_label
            med_df["Genotype"] = genotype
            li_node_genotype.append(med_df)

    deci_df = pd.concat(li_node_genotype).reset_index(drop=True)
    deci_df["Genotype + Node Type"] = deci_df["Genotype"] + " , " + deci_df["Node Type"]
    deci_df["Time_Bins"] = deci_df["Time_Bins"].astype(int)

    return deci_df


def plot_node_state_median_curve(
    config: dict,
    deci_df: pd.DataFrame,
    figure_ylimit: tuple = (0., 1.1),
    palette: list = ['grey', 'black'],
    fig_title: str | None = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot time-binned inverse median probabilities (1 - median) for decision and non-decision nodes.

    Parameters:
    -----------
    deci_df : pd.DataFrame
        DataFrame containing 'Time_Bins', '1-Median_Probability', and 'Genotype + Node Type'
    palette : list
        List of colors to apply to each unique category in hue
    figure_ylimit : tuple
        Tuple like (0, 0.6) for y-axis limits
    fig_title : str | None
        Title of the figure
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure
        Seaborn FacetGrid object
    """

    ax = sns.catplot(
        data=deci_df,
        x="Time_Bins",
        y="Median_Probability",
        hue="Genotype + Node Type",
        kind="point",
        errorbar="se",
        capsize=0.15,
        errwidth=1.5,
        palette=palette,
        aspect=1.9,
        legend=True,  # auto legend
    )

    ax.set(ylim=figure_ylimit)
    ax.set_xlabels("Time/Frame Bins", size=12)
    ax.set_ylabels("Median Probability of HMM state", size=12)
    ax.set_xticklabels(rotation=45, size=9, color="black")
    ax.set_yticklabels(size=9, color="black")

    # Adjust legend appearance (not manual mapping!)
    if ax._legend:
        ax._legend.set_bbox_to_anchor((1.02, 1))
        ax._legend.set_frame_on(True)
        ax._legend.set_title("")
    plt.title(fig_title, fontsize=15)
    plt.tight_layout()
    
    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = (
            Path(config["project_path_full"]) / "figures" / "temporal_median_state_probability_curve.pdf"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


def get_max_session_row_bracket(
    df_combined: pd.DataFrame,
    session_col: str = "Session",
) -> int:
    """
    Finds the session with the maximum number of rows and returns the largest lower multiple of 10,000.

    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined dataframe containing multiple sessions.
    session_col : str
        Name of the column representing session ID.

    Returns:
    --------
    int
        Lower bracketed row count (e.g., 20000 if max session has 23567 rows).
    """
    session_counts = df_combined[session_col].value_counts()
    max_rows = session_counts.max()
    return int(np.floor(max_rows / 10000) * 10000)


def get_min_session_row_bracket(
    df_combined: pd.DataFrame,
    session_col: str = "Session",
) -> int:
    """
    Finds the session with the minimum number of rows and returns the largest lower multiple of 10,000.

    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined dataframe containing multiple sessions.
    session_col : str
        Name of the column representing session ID.

    Returns:
    --------
    int
        Lower bracketed row count (e.g., 10000 if min session has 10234 rows).
    """
    session_counts = df_combined[session_col].value_counts()
    min_rows = session_counts.min()
    return int(np.floor(min_rows / 10000) * 10000)
