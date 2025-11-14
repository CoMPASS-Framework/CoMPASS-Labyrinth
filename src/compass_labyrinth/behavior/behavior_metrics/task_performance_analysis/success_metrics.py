"""
SUCCESSFUL BOUT ANALYSIS
Author: Shreya Bangera
Goal:
    ├── Cumulative successful bout analysis
    ├── Time-based successful bout analysis
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


##################################################################
## Plot 5: Cumulative Successful Bout Percentage
###################################################################
def assign_bout_indices_from_entry_node(
    navigation_df: pd.DataFrame,
    delimiter_node: int = 47,
) -> pd.DataFrame:
    """
    Assigns bout indices to each row of a session based on entries to a delimiter node (e.g., Entry node = 47).
    A new bout starts every time the delimiter node is encountered.

    Parameters:
    -----------
    navigation_df : pd.DataFrame
        DataFrame with 'Session' and 'Grid Number' columns.
    delimiter_node : int
        Grid number that marks the entry point for bouts.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an added 'Bout_ID' column.
    """
    all_sessions = []

    for _, session_data in navigation_df.groupby("Session"):
        session_data = session_data.reset_index(drop=True).copy()
        session_data["Bout_ID"] = 0
        bout_counter = 1

        for row_idx in range(len(session_data)):
            if session_data.loc[row_idx, "Grid Number"] != delimiter_node:
                session_data.loc[row_idx, "Bout_ID"] = bout_counter
            else:
                session_data.loc[row_idx, "Bout_ID"] = 0
                bout_counter += 1

        all_sessions.append(session_data)

    return pd.concat(all_sessions, ignore_index=True)


def summarize_bout_success_by_session(
    navigation_df: pd.DataFrame,
    optimal_regions: list = ["entry_zone", "reward_path", "target_zone"],
    target_region_label: list = ["target_zone"],
    min_bout_length: int = 20,
) -> pd.DataFrame:
    """
    Computes number of total, valid, successful, and perfect bouts per session.

    Parameters:
    -----------
    navigation_df : pd.DataFrame
        DataFrame with 'Session', 'Genotype', 'Region', and 'Bout_Index'.
    optimal_regions : list
        Ordered list of region labels that define a perfect bout.
    target_region_label : list
        Region considered as successful bout completion.
    min_bout_length : int
        Minimum length of frames required to count a bout as valid.

    Returns:
    --------
    summary_table : pd.DataFrame
        DataFrame summarizing bout stats by session.
    """
    summary_records = []

    for session_id, session_data in navigation_df.groupby("Session"):
        genotype = session_data["Genotype"].iloc[0]
        session_bouts = [b for _, b in session_data.groupby("Bout_ID") if b["Bout_ID"].iloc[0] != 0]

        valid_bouts = [b for b in session_bouts if len(b) > min_bout_length]
        successful_bouts = [b for b in valid_bouts if any(r in target_region_label for r in b["Region"])]
        perfect_bouts = [b for b in successful_bouts if set(optimal_regions) == set(b["Region"].unique())]

        summary_records.append(
            {
                "session": session_id,
                "genotype": genotype,
                "total_bouts": len(session_bouts),
                "valid_bouts": len(valid_bouts),
                "successful_bouts": len(successful_bouts),
                "perfect_bouts": len(perfect_bouts),
            }
        )

    summary_table = pd.DataFrame(summary_records)
    summary_table = summary_table[summary_table["total_bouts"] != 0]

    # Derived percentages
    summary_table["success_rate"] = (100 * summary_table["successful_bouts"]) / summary_table["valid_bouts"]
    summary_table["perfect_rate"] = (
        100 * summary_table["perfect_bouts"] / summary_table["successful_bouts"].replace(0, np.nan)
    )

    return summary_table


def plot_success_rate(
    config: dict,
    summary_table: pd.DataFrame,
    palette: list = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plots a barplot showing success rates across genotypes from the bout summary.

    Parameters:
    -----------
    config : dict
        Project configuration dictionary.
    summary_table : pd.DataFrame
        Output from summarize_bout_success_by_session.
    palette : list, optional
        Optional color palette for different genotypes.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    plt.figure(figsize=(4.5, 5))

    ax = sns.barplot(
        x="genotype",
        y="success_rate",
        data=summary_table,
        errorbar="se",
        width=0.7,
        err_kws={"color": "black", "linewidth": 1.5},
        capsize=0.15,
        edgecolor="black",
        palette=palette if palette else "deep",
    )

    sns.stripplot(x="genotype", y="success_rate", data=summary_table, dodge=True, color="black", size=4)

    ax.set_title("Percentage of Successful Bouts by Genotype", fontsize=15)
    ax.set_xlabel("Genotype", fontsize=13)
    ax.set_ylabel("% of Successful Bouts", fontsize=13)
    ax.set(ylim=(0, 100))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=2.5, color="black")
    plt.xticks(size=12, color="black")
    plt.yticks(size=12, color="black")

    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "cumulative_successful_bouts.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return ax.figure


# -------------------- T-TEST ON SUCCESS RATE PER GENOTYPE PAIR ---------------------#
def perform_genotype_ttests(
    summary_table: pd.DataFrame,
    rate_col: str = "success_rate",
):
    """
    Performs t-tests between genotypes on a given rate column (e.g., success_rate or perfect_rate).

    Parameters:
    -----------
    summary_table : pd.DataFrame
        DataFrame from `summarize_bout_success_by_session`
    rate_col : str, default 'success_rate'
        Column to compare across genotypes. Default is 'success_rate'.

    Returns:
    --------
    dict
        Dictionary with t-test results between each genotype pair.
    """
    results = {}

    # Unique genotype pairs
    genotypes = summary_table["genotype"].unique()
    for g1, g2 in combinations(genotypes, 2):
        data1 = summary_table[summary_table["genotype"] == g1][rate_col].dropna()
        data2 = summary_table[summary_table["genotype"] == g2][rate_col].dropna()
        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)

        results[f"{g1} vs {g2}"] = {
            "t_stat": t_stat,
            "p_value": p_val,
            "mean_1": data1.mean(),
            "mean_2": data2.mean(),
            "n_1": len(data1),
            "n_2": len(data2),
        }

    return results


##################################################################
# Plot 6: Time-based Successful Bout Percentage
###################################################################
def compute_binned_success_summary(
    df_all_csv: pd.DataFrame,
    lower_succ_lim: int = 0,
    upper_succ_lim: int = 90000,
    diff_succ: int = 5000,
    valid_bout_threshold: int = 19,
    optimal_path_regions: list[str] = ["entry_zone", "reward_path", "target_zone"],
    target_zone: str = "target_zone",
) -> pd.DataFrame:
    """
    Computes successful bout metrics per session, binned by cumulative frame index.

    Parameters:
    -----------
    df_all_csv : pd.DataFrame
        DataFrame with 'Session', 'Genotype', 'Region', 'Bout_ID', and 'Frame' columns.
    lower_succ_lim : int
        Lower limit of frames to start binning.
    upper_succ_lim : int
        Upper limit of frames to end binning.
    diff_succ : int
        Size of each frame bin.
    valid_bout_threshold : int
        Minimum number of frames for a bout to be considered valid.
    optimal_path_regions : list
        Regions defining a perfect bout.
    target_zone : str
        Region considered as successful bout completion.

    Returns:
    --------
    pd.DataFrame
        DataFrame summarizing binned successful bout metrics per session.
    """
    summary_records = []
    session_clusters = [x for _, x in df_all_csv.groupby("Session")]

    for session_subset in session_clusters:
        for k in range(lower_succ_lim, upper_succ_lim, diff_succ):
            sess_sub = session_subset[k : k + diff_succ]
            bouts_in_session = [x for _, x in sess_sub.groupby("Bout_ID")]

            sum_succ, sum_perfect, sum_valid_bouts = 0, 0, 0
            li_length_bouts, journey_length = [], []

            for bout in bouts_in_session:
                if len(bout) > valid_bout_threshold:
                    sum_valid_bouts += 1
                    if any(e in target_zone for e in bout["Region"].to_list()):
                        sum_succ += 1
                        li_length_bouts.append(len(bout["Region"]))
                        journey_length.append(len(bout))
                        if set(bout["Region"].unique()) == set(optimal_path_regions):
                            sum_perfect += 1

            summary_records.append(
                {
                    "Session": session_subset.Session.unique()[0],
                    "Genotype": session_subset["Genotype"].unique()[0],
                    "Bout_num": k + diff_succ,
                    "No_of_Bouts": len(bouts_in_session),
                    "No_Valid_bouts": sum_valid_bouts,
                    "No_of_Succ_Bouts": sum_succ,
                    "No_of_perfect_bouts": sum_perfect,
                }
            )

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df[summary_df["No_of_Bouts"] != 0]
    summary_df["Succ_bout_perc"] = (100 * summary_df["No_of_Succ_Bouts"]) / summary_df["No_Valid_bouts"]
    return summary_df


def plot_binned_success(
    config: dict,
    summary_df: pd.DataFrame,
    palette: list[str] = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plots % of successful bouts over time across genotypes.

    Parameters:
    -----------
    config : dict
        Project configuration dictionary.
    summary_df : pd.DataFrame
        DataFrame containing summary statistics for each session.
    palette : list, optional
        Optional color palette for different genotypes.
    save_fig : bool, default True
        Whether to save the figure as a PDF.
    show_fig : bool, default True
        Whether to display the figure.
    return_fig : bool, default False
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The matplotlib figure object if return_fig is True, else None.
    """
    sns.set_style("white")
    sns.set_style("ticks")

    summary_df["Bout_num"] = pd.Categorical(summary_df["Bout_num"])
    summary_df["Genotype"] = pd.Categorical(summary_df["Genotype"])
    summary_df["Succ_bout_perc"] = pd.to_numeric(summary_df["Succ_bout_perc"])

    ax = sns.catplot(
        x="Bout_num",
        y="Succ_bout_perc",
        hue="Genotype",
        data=summary_df,
        errorbar="se",
        kind="point",
        capsize=0.15,
        aspect=1.9,
        palette=palette,
    )
    plt.ylim(0, 110)
    plt.xticks(rotation=45)
    plt.xlabel("Time in maze")
    plt.title("Successful Bout % over time across genotypes")
    plt.ylabel("% of Successful Bouts")

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "time_based_successful_bouts.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return ax.figure


# ----------------- Mixed Effects Model (with NaNs kept) ----------------- #
def run_mixedlm_with_nans(summary_df: pd.DataFrame) -> None:
    print("\nRunning MixedLM with NaNs preserved...")
    model_df = summary_df.copy()
    model_df = model_df.dropna(subset=["Succ_bout_perc"])
    model = mixedlm("Succ_bout_perc ~ C(Bout_num) * C(Genotype)", model_df, groups=model_df["Session"])
    result = model.fit()
    print(result.summary())


# ----------------- Repeated Measures ANOVA (after fillna) ----------------- #
def run_repeated_measures_anova(summary_df: pd.DataFrame) -> None:
    print("\nRunning Repeated Measures ANOVA (NaNs filled with 0)...")
    anova_df = summary_df.copy()
    anova_df["Succ_bout_perc"] = anova_df["Succ_bout_perc"].fillna(0)
    try:
        aovrm = AnovaRM(anova_df, depvar="Succ_bout_perc", subject="Session", within=["Bout_num"], between=["Genotype"])
        anova_res = aovrm.fit()
        print(anova_res)
    except Exception as e:
        print(f"ANOVA failed: {e}")


# ----------------- Pairwise Multiple Comparisons (Tukey + FDR) ----------------- #
def run_pairwise_comparisons(summary_df: pd.DataFrame) -> None:
    print("\nRunning Pairwise Comparisons with Tukey HSD + FDR...")
    tukey_df = summary_df.copy()
    tukey_df["Succ_bout_perc"] = tukey_df["Succ_bout_perc"].fillna(0)
    results = []
    for bout in tukey_df["Bout_num"].unique():
        sub = tukey_df[tukey_df["Bout_num"] == bout]
        if sub["Genotype"].nunique() > 1:
            tukey = pairwise_tukeyhsd(
                endog=sub["Succ_bout_perc"],
                groups=sub["Genotype"],
                alpha=0.05,
            )
            df_tukey = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            df_tukey["Bout_num"] = bout
            results.append(df_tukey)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        # Apply FDR correction
        reject, pvals_corrected, _, _ = multipletests(all_results["p-adj"], method="fdr_bh")
        all_results["FDR_p"] = pvals_corrected
        all_results["Significant"] = reject
        print(all_results)
    else:
        print("No pairwise comparisons could be performed.")
