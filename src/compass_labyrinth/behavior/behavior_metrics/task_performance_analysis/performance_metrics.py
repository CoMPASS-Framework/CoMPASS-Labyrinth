"""
BINNING-BASED REGIONAL ANALYSIS, SESSION SELECTION, AND SHANNON ENTROPY QUANTIFICATION
Author: Shreya Bangera
Goal:
    ├── Region occupancy binner
    ├── Session filtering based on performance
    ├── Shannon entropy computation
"""

import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from itertools import combinations
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import mixedlm
from scipy.stats import entropy, ttest_ind

from compass_labyrinth.constants import REGION_LENGTHS, REGION_NAMES


warnings.filterwarnings("ignore")


##################################################################
# Create time-binned dictionary
###################################################################
def get_max_session_row_bracket(
    df_combined: pd.DataFrame,
    session_col: str = "Session",
) -> int:
    """
    Finds the session with the maximum number of rows and returns the largest
    lower multiple of 10,000.

    Parameters
    -----------
    df_combined : pd.DataFrame
        Combined dataframe containing multiple sessions.
    session_col : str
        Name of the column representing session ID.

    Returns
    --------
    int
        Lower bracketed row count (e.g., 20000 if max session has 23567 rows).
    """
    session_counts = df_combined[session_col].value_counts()
    max_rows = session_counts.max()
    return int(np.floor(max_rows / 10000) * 10000)


def generate_region_heatmap_pivots(
    df: pd.DataFrame,
    lower_lim: int = 0,
    upper_lim: int = 80000,
    difference: int = 10000,
    region_columns: list = ["entry_zone", "loops", "dead_ends", "neutral_zone", "reward_path", "target_zone"],
    region_lengths: dict = REGION_LENGTHS,
) -> dict:
    """
    Create binned pivot tables for each genotype showing region occupancy over time windows.

    Parameters
    -----------
    df : pd.DataFrame
        Input DataFrame containing 'Session', 'Genotype', and 'Region' columns.
    lower_lim : int
        Start index for binning.
    upper_lim : int
        End index for binning.
    difference : int
        Bin size.
    region_columns : list
        List of region names to consider.
    region_lengths : dict
        Dictionary with total lengths for each region.

    Returns
    -----------
    pivot_dict : dict
        Dictionary with Genotype as keys and list of pivot DataFrames as values.
    """
    pivot_dict = {}
    thresh_val = int((upper_lim - lower_lim) / difference)

    for genotype in df.Genotype.unique():
        li_pivot_geno = []
        df_geno = df[df.Genotype == genotype]
        session_clus_geno = [group for _, group in df_geno.groupby("Session")]
        grouped_bins = []

        for start in range(lower_lim, upper_lim, difference):
            li_bin = []
            for df_subset in session_clus_geno:
                bin_subset = df_subset.iloc[start : start + difference]
                if not bin_subset.empty:
                    li_bin.append(bin_subset)
            if li_bin:
                grouped_bins.append(pd.concat(li_bin, axis=0, ignore_index=True))

        reg_datafr = []
        for i, df_bin in enumerate(grouped_bins):
            datafr = pd.DataFrame(columns=["Session"] + region_columns)
            for sess in df.Session.unique():
                df_sess = df_bin[df_bin.Session == sess]
                for region in region_columns:
                    count = len(df_sess[df_sess.Region == region])
                    datafr.loc[sess, region] = count
                datafr.loc[sess, "Session"] = sess
            datafr.fillna(0, inplace=True)
            datafr.set_index("Session", inplace=True)
            reg_datafr.append(datafr)

        for j in range(len(reg_datafr)):
            datafr_sub = reg_datafr[j].reset_index()
            for region in region_columns:
                region_len = region_lengths.get(region, 1) if region_lengths else 1
                datafr_sub[region] /= region_len

            melted = pd.melt(
                datafr_sub, id_vars="Session", value_vars=region_columns, var_name="Region", value_name="value"
            )
            pivoted = pd.pivot_table(melted, index="Region", columns="Session", values="value", aggfunc="mean")
            col_sums = pivoted.sum(axis=0).replace(0, np.nan)
            pivoted = pivoted.div(col_sums, axis=1)

            li_pivot_geno.append(pivoted)

        pivot_dict[genotype] = li_pivot_geno

    return pivot_dict


##################################################################
# Exclusion Criteria
###################################################################
def compute_frames_per_session(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Session").size().reset_index(name="No_of_Frames")


def compute_target_zone_usage(
    df: pd.DataFrame,
    pivot_dict: dict,
    region: str = "target_zone",
    difference: int = 10000,
) -> pd.DataFrame:
    """
    Compute target zone usage from a time-binned pivot dictionary.

    Parameters
    -----------
    df : pd.DataFrame
        DataFrame containing 'Session' and 'Genotype' columns.
    pivot_dict : dict
        Dictionary with Genotype as keys and list of pivot DataFrames as values.
    region : str
        The region to compute usage for.
    difference : int
        Bin size.

    Returns
    --------
    pd.DataFrame
        DataFrame containing target zone usage information.
    """
    usage_records = []
    for genotype in df.Genotype.unique():
        li_genotype = pivot_dict[genotype]
        for bout_idx, pivot in enumerate(li_genotype):
            for session in pivot.columns:
                usage_records.append(
                    {
                        "Genotype": genotype,
                        "Session": session,
                        "Bout": (bout_idx + 1) * difference,
                        "Target_Usage": pivot.loc[region, session],
                    }
                )
    return pd.DataFrame(usage_records)


def summarize_target_usage(
    region_target: str,
    frames_df: pd.DataFrame,
    cohort_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize target zone usage per session.

    Parameters
    -----------
    region_target : str
        The target region to summarize.
    frames_df : pd.DataFrame
        DataFrame containing frame information.
    cohort_metadata : pd.DataFrame
        DataFrame containing cohort metadata.

    Returns
    --------
    pd.DataFrame
        DataFrame containing the summary of target zone usage.
    """
    session_frames = dict(frames_df.values)
    session_sex = dict(cohort_metadata[["Session #", "Sex"]].values)
    summary = region_target.groupby(["Genotype", "Session"])["Target_Usage"].mean().reset_index()
    summary["No_of_Frames"] = summary["Session"].map(session_frames)
    summary["Sex"] = summary["Session"].map(session_sex)
    return summary


def plot_target_usage_vs_frames(
    config: dict,
    summary_df: pd.DataFrame,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot target zone usage vs number of frames.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    summary_df : pd.DataFrame
        DataFrame containing the summary of target zone usage.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    summary_df = summary_df[np.isfinite(summary_df["No_of_Frames"]) & np.isfinite(summary_df["Target_Usage"])]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=summary_df,
        x="No_of_Frames",
        y="Target_Usage",
        hue="Genotype",
        style="Sex",
        palette=["orange", "grey", "red", "blue"],
        ax=ax,
    ).collections[0].set_sizes([200])

    plt.xlabel("No. of Frames", fontsize=10)
    plt.ylabel("Mean Target Usage", fontsize=10)
    plt.ylim(0, 1)
    plt.xlim(left=0)

    for line in range(summary_df.shape[0]):
        plt.text(
            summary_df["No_of_Frames"].iloc[line] + 0.2,
            summary_df["Target_Usage"].iloc[line],
            summary_df["Session"].iloc[line],
            ha="right",
            color="black",
            size="medium",
        )
    plt.title("Target Zone Usage vs No. of Frames")
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


def exclude_low_performing_sessions(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    usage_threshold: float | None = 0.4,
    min_frames: int | None = 30000,
) -> pd.DataFrame:
    """
    Exclude sessions based on target usage and frame count thresholds.

    Parameters
    -----------
    df : pd.DataFrame
        The original DataFrame containing session data.
    summary_df : pd.DataFrame
        The summary DataFrame containing session performance metrics.
    usage_threshold : float | None
        The minimum target usage threshold for excluding sessions.
    min_frames : int | None
        The minimum number of frames threshold for excluding sessions.

    Returns
    --------
    pd.DataFrame
        The cleaned DataFrame with low-performing sessions excluded.
    """
    try:
        if usage_threshold is None:
            target_threshold = float(input("Enter minimum target usage threshold (e.g., 0.4): "))
        if min_frames is None:
            frame_threshold = int(input("Enter minimum number of frames threshold (e.g., 30000): "))
    except ValueError:
        print("Invalid input. Using default thresholds: Target Usage = 0.4, Frames = 30000")
        usage_threshold = 0.4
        min_frames = 30000

    sessions_to_exclude = (
        summary_df.loc[
            (summary_df["Target_Usage"] < usage_threshold) & (summary_df["No_of_Frames"] < min_frames), "Session"
        ]
        .unique()
        .tolist()
    )

    print(f"\nExcluding {len(sessions_to_exclude)} session(s): {sessions_to_exclude}")
    df_cleaned = df[~df["Session"].isin(sessions_to_exclude)].copy()
    return df_cleaned


def plot_target_usage_with_exclusions(
    config: dict,
    summary_df: pd.DataFrame,
    sessions_to_exclude: list,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot target zone usage vs number of frames, marking excluded sessions.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    summary_df : pd.DataFrame
        DataFrame containing the summary of target zone usage.
    sessions_to_exclude : list
        List of session IDs to exclude from the plot.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to show the figure.
    return_fig : bool
        Whether to return the figure.

    Returns
    --------
    None | plt.Figure
        The created figure, if return_fig is True.
    """
    summary_df = summary_df[np.isfinite(summary_df["No_of_Frames"]) & np.isfinite(summary_df["Target_Usage"])]
    fig, ax = plt.subplots(figsize=(10, 8))

    # Split included and excluded sessions
    included_df = summary_df[~summary_df["Session"].isin(sessions_to_exclude)]
    excluded_df = summary_df[summary_df["Session"].isin(sessions_to_exclude)]

    # Plot included points
    sns.scatterplot(
        data=included_df,
        x="No_of_Frames",
        y="Target_Usage",
        hue="Genotype",
        style="Sex",
        palette=["orange", "grey", "red", "blue"],
        ax=ax,
        s=200,
        alpha=0.9,
        legend=True,
    )

    # Plot excluded points (overlay with 'X' marker)
    sns.scatterplot(
        data=excluded_df, x="No_of_Frames", y="Target_Usage", color="black", marker="X", s=250, label="Excluded", ax=ax
    )

    # Add session labels
    for _, row in summary_df.iterrows():
        plt.text(row["No_of_Frames"] + 0.2, row["Target_Usage"], row["Session"], ha="right", fontsize=9, color="black")

    plt.xlabel("No. of Frames", fontsize=10)
    plt.ylabel("Mean Target Usage", fontsize=10)
    plt.ylim(0, 1)
    plt.xlim(left=0)
    plt.title("Target Zone Usage vs No. of Frames (Excluded Sessions Marked)")
    plt.tight_layout()
    plt.legend()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames_exclusions.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


###################################################################
# Subset the Time-Binned Dictionary based on Valid Sessions
###################################################################
def subset_pivot_dict_sessions(pivot_dict: dict, df_all_csv: pd.DataFrame) -> dict:
    """
    Subset an existing pivot_dict to only include valid sessions from df_all_csv.

    Parameters
    -----------
    pivot_dict : dict
        Original pivot_dict with all sessions.
    df_all_csv : pd.DataFrame
        Must contain 'Session' and 'Genotype' columns.

    Returns
    --------
    dict
        Filtered pivot_dict with only valid sessions per genotype.
    """
    # Map genotype to list of valid sessions
    valid_sessions_dict = {
        geno: df_all_csv[df_all_csv.Genotype == geno]["Session"].unique().tolist()
        for geno in df_all_csv["Genotype"].unique()
    }

    # Filter pivot_dict
    filtered_pivot_dict = {}
    for genotype, pivot_list in pivot_dict.items():
        valid_sessions = valid_sessions_dict.get(genotype, [])
        filtered_list = []
        for df_bin in pivot_list:
            filtered = df_bin.loc[:, df_bin.columns.intersection(valid_sessions)]
            filtered_list.append(filtered)
        filtered_pivot_dict[genotype] = filtered_list

    return filtered_pivot_dict


##################################################################
# Plot 1: Heatmap Representations
###################################################################
def plot_region_heatmaps(
    config: dict,
    pivot_dict: dict,
    group_name: str,
    lower_lim: int,
    upper_lim: int,
    difference: int,
    included_sessions: list | None = None,
    vmax: float = 0.6,
    region_desired_order: list | None = None,
    cmap: str = "viridis",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Clean and aesthetically pleasing vertically stacked heatmaps with one colorbar per bin.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    pivot_dict : dict
        Dictionary with Genotype as keys and list of pivot DataFrames as values.
    group_name : str
        The genotype or group to plot.
    lower_lim : int
        Start frame.
    upper_lim : int
        End frame.
    difference : int
        Bin size.
    included_sessions : list | None
        List of session IDs to include. If None, include all sessions.
    vmax : float
        Colorbar upper limit.
    region_desired_order : list | None
        Desired order of regions for the heatmap. If None, use default order.
    cmap : str
        Colormap name.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    None | plt.Figure
        The figure object if return_fig is True, otherwise None.
    """
    sns.set_context("notebook", font_scale=1.0)
    sns.set_style("ticks")

    region_order = region_desired_order or [
        "entry_zone",
        "loops",
        "dead_ends",
        "neutral_zone",
        "reward_path",
        "target_zone",
    ]
    n_bins = int((upper_lim - lower_lim) / difference)

    fig = plt.figure(figsize=(16, 3.5 * n_bins))
    outer = gridspec.GridSpec(n_bins, 1, hspace=0.4)

    for i in range(n_bins):
        if i >= len(pivot_dict.get(group_name, [])):
            continue

        pivot_tab = pivot_dict[group_name][i]

        if included_sessions is not None:
            valid_cols = [sess for sess in pivot_tab.columns if sess in included_sessions]
            pivot_tab = pivot_tab[valid_cols]

        # Reindex and round values
        pivot_tab = pivot_tab.reindex(region_order).fillna(np.nan)
        pivot_tab.index = pivot_tab.index.map(lambda x: REGION_NAMES.get(x, x))
        rounded = pivot_tab.round(2)

        # Create sub-grid for each bin
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], width_ratios=[20, 1], wspace=0.08)

        ax = fig.add_subplot(inner[0])
        cax = fig.add_subplot(inner[1])

        sns.heatmap(
            data=rounded,
            ax=ax,
            cbar=True,
            cbar_ax=cax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            mask=rounded.isna(),
            linewidths=0.5,
            linecolor="white",
            square=False,
        )

        # Styling
        ax.set_title(
            f"{group_name} | Bin {i+1} ({lower_lim + i * difference:,}–{lower_lim + (i+1) * difference:,})",
            fontsize=15,
            pad=8,
            weight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelrotation=0, labelsize=10)

    fig.suptitle(f"Region Occupancy Heatmaps for {group_name}", fontsize=18, weight="bold", y=0.99)
    plt.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.95)

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / f"region_heatmaps_{group_name}.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()

    if return_fig:
        return fig


#####################################################################
## Heatmap Representations across all Genotypes
#####################################################################
def plot_region_heatmaps_all_genotypes(
    config: dict,
    pivot_dict: dict,
    df_all_csv: pd.DataFrame,
    lower_lim: int,
    upper_lim: int,
    difference: int,
    vmax: float = 0.6,
    region_desired_order: list | None = None,
    cmap: str = "viridis",
    included_genotypes: list | None = None,
    figsize_per_genotype: tuple = (4.5, 2.8),
    spacing_w: float = 0.6,  # Wider horizontal space between genotypes
    spacing_h: float = 0.2,  # Tighter vertical space between bins
    show_colorbar: bool = True,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot region occupancy heatmaps for each genotype and bin:
    - Rows: bins
    - Columns: genotypes

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    pivot_dict : dict
        Dictionary with Genotype as keys and list of pivot DataFrames as values.
    df_all_csv : pd.DataFrame
        DataFrame with valid 'Genotype' and 'Session' combinations.
    lower_lim : int
        Start frame
    upper_lim : int
        End frame
    difference : int
        Bin size
    vmax : float
        Colorbar upper limit
    region_desired_order : list
        Optional order of regions
    cmap : str
        Colormap name
    included_genotypes : list
        Genotype order to include
    figsize_per_genotype : tuple
        width x height scaling per genotype
    spacing_w : float
        Space between genotype columns
    spacing_h : float
        Space between time-bin rows
    show_colorbar : bool
        If True, show colorbar in last column per row
    save_fig : bool
        If True, save the figure
    show_fig : bool
        If True, display the figure
    return_fig : bool
        If True, return the figure object

    Returns
    --------
    None | plt.Figure
        The figure object if return_fig is True, otherwise None.
    """
    sns.set_context("notebook", font_scale=1.0)
    sns.set_style("white")

    region_order = region_desired_order or [
        "entry_zone",
        "loops",
        "dead_ends",
        "neutral_zone",
        "reward_path",
        "target_zone",
    ]

    all_genotypes = list(pivot_dict.keys())
    genotypes = included_genotypes if included_genotypes is not None else all_genotypes

    n_bins = int((upper_lim - lower_lim) / difference)
    n_genos = len(genotypes)

    fig_w = figsize_per_genotype[0] * n_genos
    fig_h = figsize_per_genotype[1] * n_bins

    fig, axes = plt.subplots(nrows=n_bins, ncols=n_genos, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(n_bins):
        for j, genotype in enumerate(genotypes):
            ax = axes[i, j]

            pivot_tables = pivot_dict.get(genotype, [])
            if i >= len(pivot_tables):
                ax.axis("off")
                continue

            pivot_tab = pivot_tables[i]

            # Use only valid sessions from df_all_csv
            valid_sessions = df_all_csv.loc[df_all_csv["Genotype"] == genotype, "Session"].unique()
            pivot_tab = pivot_tab[[s for s in pivot_tab.columns if s in valid_sessions]]
            pivot_tab = pivot_tab.reindex(region_order).fillna(np.nan)
            pivot_tab.index = pivot_tab.index.map(lambda x: REGION_NAMES.get(x, x))

            sns.heatmap(
                data=pivot_tab,
                ax=ax,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                mask=pivot_tab.isna(),
                annot=False,
                linewidths=0.4,
                linecolor="white",
                cbar=(show_colorbar and j == n_genos - 1),
            )

            # Titles
            if i == 0:
                ax.set_title(genotype, fontsize=13, weight="bold")

            # Y-labels only for first column
            if j == 0:
                bin_start = lower_lim + i * difference
                bin_end = bin_start + difference
                ax.set_ylabel(f"{bin_start}-{bin_end}", fontsize=11)
                ax.set_yticklabels(pivot_tab.index, rotation=0, fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])

            # X-tick session labels (always)
            ax.set_xlabel("")
            ax.set_xticklabels(pivot_tab.columns, fontsize=9)
            ax.tick_params(axis="x", bottom=True)

    fig.suptitle("Region Occupancy Heatmaps by Genotype & Bin", fontsize=17, weight="bold", y=1.01)

    plt.subplots_adjust(wspace=spacing_w, hspace=spacing_h, top=0.95, bottom=0.05, left=0.05, right=0.95)

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "region_heatmaps_all_genotypes.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()

    if return_fig:
        return fig


##################################################################
# Shannon's Entropy
###################################################################
def compute_shannon_entropy_per_bin(
    pivot_dict: dict,
    df_all_csv: pd.DataFrame,
    bin_size: int = 10000,
) -> pd.DataFrame:
    """
    Computes Shannon entropy per bin per session.
    Uses df_all_csv for genotype mapping and ensures robust merging.

    Parameters
    -----------
    pivot_dict : dict
        Dictionary with Genotype as keys and list of pivot DataFrames as values.
    df_all_csv : pd.DataFrame
        DataFrame containing 'Session' and 'Genotype' columns.
    bin_size : int
        Size of each time bin.

    Returns
    --------
    pd.DataFrame
        DataFrame with columns: 'Session', 'Bin', 'Entropy', 'Genotype'.
    """
    entropy_records = []

    df_all_csv = df_all_csv.copy()
    df_all_csv["Session"] = df_all_csv["Session"].astype(str)

    for genotype in df_all_csv["Genotype"].unique():
        bins = pivot_dict.get(genotype, [])
        for idx, pivot_table in enumerate(bins):
            if pivot_table is None or pivot_table.empty:
                continue

            data = pivot_table.dropna(axis=1, how="any").T
            if data.empty:
                continue

            data.index = data.index.astype(str)
            prob_data = data.div(data.sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            ent_vals = entropy(prob_data.values, base=2, axis=1)

            for session_id, e in zip(data.index, ent_vals):
                if np.isfinite(e):
                    entropy_records.append({"Session": session_id, "Bin": (idx + 1) * bin_size, "Entropy": e})

    entropy_df = pd.DataFrame(entropy_records)

    # Prepare full session × bin grid
    entropy_df["Session"] = entropy_df["Session"].astype(str)
    df_all_csv["Session"] = df_all_csv["Session"].astype(str)

    all_bins = sorted(entropy_df["Bin"].unique())
    all_sessions = df_all_csv["Session"].unique()
    full_index = pd.MultiIndex.from_product([all_sessions, all_bins], names=["Session", "Bin"])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Merge in entropy and genotype
    full_df = full_df.merge(entropy_df, on=["Session", "Bin"], how="left")
    genotype_map = df_all_csv[["Session", "Genotype"]].drop_duplicates()
    full_df = full_df.merge(genotype_map, on="Session", how="left")

    # Final checks
    if "Genotype" not in full_df.columns or full_df["Genotype"].isna().all():
        raise ValueError("Genotype column is missing or all values are NaN after merge.")

    return full_df


###############################################################################
## Plot 2: Plotting Shannon's Entropy across Sessions (/Mice)
###############################################################################
def plot_entropy_over_bins(
    config: dict,
    entropy_df: pd.DataFrame,
    palette: list | None = None,
    ylim: tuple = (0, 5),
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot Shannon's entropy across bins for each genotype.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    entropy_df : pd.DataFrame
        DataFrame containing 'Session', 'Bin', 'Entropy', and 'Genotype' columns
    palette : list | None
        List of colors for genotypes.
    ylim : tuple
        Y-axis limits.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    None | plt.Figure
        The figure object if return_fig is True, otherwise None.
    """
    sns.set_style("ticks")
    g = sns.catplot(
        data=entropy_df,
        x="Bin",
        y="Entropy",
        hue="Genotype",
        kind="point",
        capsize=0.15,
        errwidth=1.5,
        errorbar="se",
        palette=palette,
        legend=True,
        aspect=1.5,
        height=5,
    )

    g._legend.set_title("Genotype")
    g._legend.set_bbox_to_anchor((1, 1))
    for text in g._legend.texts:
        text.set_fontsize(12)
    g._legend.get_title().set_fontsize(13)

    ax = g.ax
    ax.set_ylim(ylim)
    ax.set_xlabel("Frames", fontsize=15)
    ax.set_ylabel("Shannon's Entropy\n(per session per bin)", fontsize=15)
    # ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "shannon_entropy.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()

    if return_fig:
        return g


# ------------------- Repeated Measures ANOVA (within-subject: Bin) -----------------#
def run_entropy_anova(entropy_df: pd.DataFrame) -> AnovaRM | None:
    """
    Run repeated measures ANOVA using Bin as within-subject factor.
    Fills missing Entropy values with 0 (only here).

    Parameters
    -----------
    entropy_df : pd.DataFrame
        DataFrame containing 'Session', 'Bin', 'Entropy', and 'Genotype' columns

    Returns
    --------
    AnovaRM | None
        The fitted ANOVA model or None if it fails.
    """
    df_stats = entropy_df.copy()
    df_stats["Entropy"] = df_stats["Entropy"].fillna(0)

    try:
        aovrm = AnovaRM(data=df_stats, depvar="Entropy", subject="Session", within=["Bin"])
        result = aovrm.fit()
        print("Repeated Measures ANOVA (within-subject Bin):")
        print(result.summary())
        return result
    except Exception as e:
        print("ANOVA failed:", e)
        return None


# ----------------- FDR-Corrected Pairwise T-Tests (all genotype combos per bin) ----------------#
def run_fdr_pairwise_tests(entropy_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    For each bin, performs pairwise t-tests between all genotype pairs.
    Applies FDR correction across all tests.

    Parameters
    -----------
    entropy_df : pd.DataFrame
        DataFrame containing 'Session', 'Bin', 'Entropy', and 'Genotype' columns

    Returns
    --------
    pd.DataFrame | None
        DataFrame with pairwise test results.
    """
    df = entropy_df.copy()
    df["Entropy"] = df["Entropy"].fillna(0)

    all_bins = sorted(df["Bin"].dropna().unique())
    all_genotypes = df["Genotype"].dropna().unique()
    pairwise_combos = list(combinations(all_genotypes, 2))

    results = []

    for bin_val in all_bins:
        df_bin = df[df["Bin"] == bin_val]
        for g1, g2 in pairwise_combos:
            g1_vals = df_bin[df_bin["Genotype"] == g1]["Entropy"].values
            g2_vals = df_bin[df_bin["Genotype"] == g2]["Entropy"].values

            if len(g1_vals) > 1 and len(g2_vals) > 1:
                stat, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)
                results.append({"Bin": bin_val, "Group1": g1, "Group2": g2, "t-stat": stat, "raw-p": pval})

    # FDR correction
    raw_pvals = [r["raw-p"] for r in results]
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    for i, r in enumerate(results):
        r["FDR-p"] = pvals_corrected[i]
        r["Significant"] = reject[i]

    return pd.DataFrame(results)


# --------------- Mixed Effects Model (Bin × Genotype interaction per pair) --------------#
def run_mixed_model_per_genotype_pair(entropy_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    For each genotype pair, test if Bin x Genotype interaction is significant.
    Does NOT fill NaNs. Uses only complete-case rows per model.

    Parameters
    -----------
    entropy_df : pd.DataFrame
        DataFrame containing 'Session', 'Bin', 'Entropy', and 'Genotype' columns

    Returns
    --------
    result_dict : dict
        Model summaries
    interaction_table : pd.DataFrame
        p-values of interaction terms
    """
    df = entropy_df.copy()
    df = df.dropna(subset=["Entropy"])
    df["Session"] = df["Session"].astype(str)
    df["Bin"] = df["Bin"].astype("category")
    df["Genotype"] = df["Genotype"].astype("category")

    genotype_pairs = list(combinations(df["Genotype"].dropna().unique(), 2))
    result_dict = {}
    summary_rows = []

    for g1, g2 in genotype_pairs:
        df_pair = df[df["Genotype"].isin([g1, g2])].copy()
        df_pair["Genotype"] = df_pair["Genotype"].cat.remove_unused_categories()

        try:
            model = mixedlm("Entropy ~ Bin * Genotype", df_pair, groups=df_pair["Session"])
            result = model.fit()
            result_dict[(g1, g2)] = result

            interaction_pvals = {k: v for k, v in result.pvalues.items() if "Bin" in k and "Genotype" in k}

            summary_rows.append(
                {
                    "Genotype1": g1,
                    "Genotype2": g2,
                    "Interaction_pvals": interaction_pvals,
                    "Significant": any(p < 0.05 for p in interaction_pvals.values()),
                }
            )

            print(f"\n Genotype Pair: {g1} vs {g2}")
            print(result.summary())

        except Exception as e:
            print(f"\n MixedLM failed for {g1} vs {g2}: {e}")
            result_dict[(g1, g2)] = None
            summary_rows.append({"Genotype1": g1, "Genotype2": g2, "Interaction_pvals": {}, "Significant": False})

    interaction_table = pd.DataFrame(summary_rows)
    return result_dict, interaction_table


##################################################################
# Proportion of Region-based usage across Time bins
###################################################################
def compute_region_usage_over_bins(
    pivot_dict: dict,
    df_all_csv: pd.DataFrame,
    region: str,
    bin_size: int,
) -> pd.DataFrame:
    """
    Computes binned region usage across sessions for the given region.

    Parameters
    -----------
    pivot_dict : dict
        Dictionary with genotype keys and binned pivot tables.
    df_all_csv : pd.DataFrame
        DataFrame with session and genotype mapping.
    region : str
        Region to compute usage for (e.g., "Target Zone").
    bin_size : int
        Size of each bin (in frames).

    Returns
    --------
    pd.DataFrame
        Binned region usage across sessions with Genotype labels.
    """
    region_usage = []

    for genotype, bin_list in pivot_dict.items():
        for i, df_bin in enumerate(bin_list):
            if region in df_bin.index:
                subset = df_bin.loc[[region]].T.reset_index()
                subset["Bin"] = (i + 1) * bin_size
                region_usage.append(subset)

    reg_binned = pd.concat(region_usage, ignore_index=True)

    # Map Session to Genotype
    session_to_genotype = {k: g["Session"].tolist() for k, g in df_all_csv.groupby("Genotype")}
    for geno, sessions in session_to_genotype.items():
        reg_binned.loc[reg_binned["Session"].isin(sessions), "Genotype"] = geno

    return reg_binned


##################################################################
## Plot 3: Proportion of usage per Region across time
###################################################################
def plot_region_usage_over_bins(
    config: dict,
    region_data: pd.DataFrame,
    region_name: str,
    palette: list | None = None,
    ylim: tuple = (0, 1),
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
):
    """
    Plots the proportion of usage over time bins for a specific region.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    region_data : pd.DataFrame
        Output from compute_region_usage_over_bins().
    region_name : str
        Display name for the region.
    palette : list or dict
        Optional Seaborn color palette for genotypes.
    ylim : tuple
        Y-axis limits.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    None | plt.Figure
        The figure object if return_fig is True, otherwise None.
    """
    ax = sns.catplot(
        x="Bin",
        y=region_name,
        hue="Genotype",
        data=region_data,
        kind="point",
        capsize=0.15,
        errorbar="se",
        palette=palette,
        aspect=1.5,
    )

    plt.xticks(size=10, color="black")
    plt.yticks(size=12, color="black")
    plt.xlabel("Cumulative time in maze (frames)", size=15)
    plt.ylabel("Proportion of Usage", size=15)
    plt.title(REGION_NAMES[region_name], fontsize=15, weight="bold")
    ax.set(yticks=np.arange(ylim[0], ylim[1] + 0.1, 0.1))

    # Save figure
    fig = ax.figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / f"{region_name}_prop_usage.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()

    if return_fig:
        return fig


##################################################################
## Plot 4: Proportion of usage for all Regions across time
###################################################################
def plot_all_regions_usage_over_bins(
    config: dict,
    pivot_dict: dict,
    df_all_csv: pd.DataFrame,
    region_list: list,
    bin_size: int = 10000,
    palette: list | None = None,
    ylim: tuple = (0, 1),
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
):
    """
    Plots usage over bins for multiple regions in a 2x3 subplot layout with a shared legend outside.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project settings.
    pivot_dict : dict
        Dictionary with genotype keys and binned pivot tables.
    df_all_csv : pd.DataFrame
        DataFrame with session and genotype mapping.
    region_list : list
        List of regions to plot (max 6).
    bin_size : int
        Size of each bin (in frames).
    palette : list or dict
        Optional Seaborn color palette for genotypes.
    ylim : tuple
        Y-axis limits.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    None | plt.Figure
        The figure object if return_fig is True, otherwise None.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for idx, region in enumerate(region_list):
        ax = axes[idx]

        # Compute region usage
        region_data = compute_region_usage_over_bins(pivot_dict, df_all_csv, region, bin_size)

        # Plot with seaborn
        plot = sns.pointplot(
            data=region_data, x="Bin", y=region, hue="Genotype", errorbar="se", palette=palette, capsize=0.15, ax=ax
        )

        ax.set_title(REGION_NAMES[region], fontsize=14, weight="bold")
        ax.set_xlabel("Frames", fontsize=12)
        ax.set_ylabel("Usage Proportion", fontsize=12)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=10)

        # Store legend handles/labels from the first plot only
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()  # Remove legend from all subplots

    # Remove unused subplots if < 6 regions
    for j in range(len(region_list), 6):
        fig.delaxes(axes[j])

    # Shared legend outside to the right
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center right",
        fontsize=12,
        title="Genotype",
        title_fontsize=13,
        frameon=True,
    )

    plt.suptitle("Proportion of Region Usage Across Time Bins", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])  # Leave space for external legend

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_regions_prop_usage.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_fig:
        plt.show()

    if return_fig:
        return fig


# ------------------- Mixed Effects Model ----------------------#
def run_region_usage_stats_mixedlm(reg_binned: pd.DataFrame, region_col: str = "target_zone") -> None:
    """
    Mixed Effects Model (Bin x Genotype) with missing bins dropped.

    Parameters
    -----------
    reg_binned : pd.DataFrame
        DataFrame from compute_region_usage_over_bins().
    region_col : str
        Column name for the region of interest. Default is "target_zone".

    Returns
    --------
    None
    """
    # Rename column safely (avoid space)
    safe_col = region_col.replace(" ", "_")
    reg_binned = reg_binned.rename(columns={region_col: safe_col})

    # MixedLM (drop NaNs)
    df_nan = reg_binned[["Session", "Bin", "Genotype", safe_col]].dropna()
    df_nan["Bin"] = df_nan["Bin"].astype(float)
    df_nan["Genotype"] = df_nan["Genotype"].astype("category")

    print("\n=== Mixed Effects Model (missing values preserved) ===")
    try:
        model = mixedlm(f"{safe_col} ~ Bin * Genotype", data=df_nan, groups=df_nan["Session"])
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print("MixedLM error:", e)


# --------------- Pairwise Comparison with FDR Correction --------------#
def run_region_usage_stats_fdr(
    reg_binned: pd.DataFrame,
    region_col: str = "target_zone",
) -> pd.DataFrame | None:
    """
    Pairwise genotype comparisons at each bin (FDR corrected).

    Parameters
    -----------
    reg_binned : pd.DataFrame
        DataFrame from compute_region_usage_over_bins().
    region_col : str
        Column name for the region of interest. Default is "target_zone".

    Returns
    --------
    pd.DataFrame | None
        DataFrame with pairwise test results or None if an error occurs.
    """
    # ------------ Rename column safely (avoid space) -------------
    safe_col = region_col.replace(" ", "_")
    reg_binned = reg_binned.rename(columns={region_col: safe_col})

    # ------------ Pairwise t-tests at each bin (fillna(0)) -------------
    df_zero = reg_binned[["Session", "Bin", "Genotype", safe_col]].copy()
    df_zero[safe_col] = df_zero[safe_col].fillna(0)

    print("\n=== Pairwise t-tests between Genotypes at each Bin (FDR corrected) ===")
    try:
        bin_results = []
        for b in sorted(df_zero["Bin"].unique()):
            df_bin = df_zero[df_zero["Bin"] == b]
            genotypes = df_bin["Genotype"].unique()
            for g1, g2 in combinations(genotypes, 2):
                vals1 = df_bin[df_bin["Genotype"] == g1][safe_col]
                vals2 = df_bin[df_bin["Genotype"] == g2][safe_col]
                stat, pval = ttest_ind(vals1, vals2, equal_var=False)
                bin_results.append({"Bin": b, "Group1": g1, "Group2": g2, "pval": pval})

        df_stats = pd.DataFrame(bin_results)
        reject, pvals_corrected, _, _ = multipletests(df_stats["pval"], method="fdr_bh")
        df_stats["pval_fdr"] = pvals_corrected
        df_stats["significant"] = reject
        print(df_stats.to_string(index=False))
        return df_stats
    except Exception as e:
        print("Pairwise t-test error:", e)
