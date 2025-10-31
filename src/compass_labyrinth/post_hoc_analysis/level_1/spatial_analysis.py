"""
STATE DISTRIBUTIONS BY NODE-TYPE AND REGION
Author: Shreya Bangera
Goal:
    ├── Comparison of proportion of time spent in a state across Maze regions and Node types.
    ├── Allows genotype level comparisons behavioral states.
"""

from pathlib import Path
from itertools import combinations
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING


warnings.simplefilter(action="ignore", category=FutureWarning)


##################################################################
# Plot 2: Probability of Surveillance across Node Types and Regions
###################################################################
def compute_state_probability(
    df_hmm: pd.DataFrame,
    column_of_interest: str,
    values_displayed: list[str] | None = None,
    state: int = 1,
) -> pd.DataFrame:
    """
    Computes HMM state proportions by category (e.g., NodeType or Region).
    Optionally reassigns decision node labels for 3-way and 4-way decisions.

    Parameters:
    -----------
    df_hmm: pd.DataFrame
        Dataframe with 'Genotype', 'Session', 'HMM_State', and category column.
    column_of_interest: str
        'NodeType' or 'Region'
    values_displayed: Optional[List[str]]
        Categories to include and order
    state: int
        HMM_state of interest

    Returns:
    --------
    pd.DataFrame
        Dataframe with proportions per session.
    """

    df_plot = df_hmm.copy()

    # Optional reassignment of NodeType for 3-way / 4-way decisions
    decision_3way_grids = NODE_TYPE_MAPPING.get("decision_3way", [])
    decision_4way_grids = NODE_TYPE_MAPPING.get("decision_4way", [])
    if column_of_interest == "NodeType" and decision_3way_grids and decision_4way_grids:
        df_plot.loc[df_plot["Grid Number"].isin(decision_3way_grids), "NodeType"] = "3-way Decision (Reward)"
        df_plot.loc[df_plot["Grid Number"].isin(decision_4way_grids), "NodeType"] = "4-way Decision (Reward)"
        df_plot = df_plot.loc[~df_plot["NodeType"].isin(["Entry Nodes", "Target Nodes"])]

    # Compute state occurrence counts
    st_cnt = (
        df_plot.groupby(["Genotype", column_of_interest, "Session", "HMM_State"]).size().rename("cnt").reset_index()
    )
    gn_cnt = df_plot.groupby(["Genotype", column_of_interest, "Session"]).size().rename("tot").reset_index()
    state_count = st_cnt.merge(gn_cnt, on=[column_of_interest, "Genotype", "Session"], how="left")
    state_count["prop"] = state_count["cnt"] / state_count["tot"]

    # Filter for target HMM state and reorder
    state_count = state_count[state_count["HMM_State"] == state].copy()
    if values_displayed:
        state_count = state_count[state_count[column_of_interest].isin(values_displayed)].reset_index(drop=True)
        state_count[column_of_interest] = pd.Categorical(
            state_count[column_of_interest], categories=values_displayed, ordered=True
        )

    return state_count


def plot_state_probability_boxplot(
    config: dict,
    state_count_df: pd.DataFrame,
    column_of_interest: str,
    state: int = 1,
    figsize: tuple = (16, 7),
    palette: str = "Set2",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plots boxplot of HMM state probabilities by category and genotype.

    Parameters:
    state_count: pd.DataFrame
        Dataframe returned from compute_state_probability()
    column_of_interest: str
        Categorical variable on x-axis
    state: int
        HMM state used for labeling
    figsize: tuple
        Figure size
    palette: str
        Seaborn palette
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    fig : plt.Figure | None
        Matplotlib Figure object if return_fig is True, else None.
    """
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(
        x=column_of_interest,
        y="prop",
        hue="Genotype",
        data=state_count_df,
        palette=palette,
    )
    ax.set_ylabel(f"Probability of being in State {state}", fontsize=15)
    ax.set_xlabel(column_of_interest, fontsize=15)
    plt.xticks(size=11)
    plt.yticks(size=15)
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = (
            Path(config["project_path_full"]) / "figures" / f"state_{state}_probability_by_{column_of_interest}.pdf"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


##################################################################
# T-Tests per genotype combo
###################################################################
def run_pairwise_ttests(
    state_count_df: pd.DataFrame,
    column_of_interest: str = "NodeType",
) -> pd.DataFrame:
    """
    Perform pairwise t-tests between genotypes within each level of the column_of_interest.

    Parameters:
    -----------
    state_count_df: pd.DataFrame
        DataFrame returned from compute_state_probability
    column_of_interest: str
        Column over which comparisons are grouped

    Returns:
    --------
    pd.DataFrame
        Dataframe with columns: [Group, Genotype1, Genotype2, t-stat, p-value]
    """
    results = []
    groups = state_count_df[column_of_interest].dropna().unique()

    for group in groups:
        subset = state_count_df[state_count_df[column_of_interest] == group]
        genotypes = subset["Genotype"].unique()

        for g1, g2 in combinations(genotypes, 2):
            values1 = subset[subset["Genotype"] == g1]["prop"].dropna()
            values2 = subset[subset["Genotype"] == g2]["prop"].dropna()

            if len(values1) >= 2 and len(values2) >= 2:
                t_stat, p_val = ttest_ind(values1, values2, equal_var=False)
                results.append({"Group": group, "Genotype1": g1, "Genotype2": g2, "T-stat": t_stat, "P-value": p_val})

    return pd.DataFrame(results)
