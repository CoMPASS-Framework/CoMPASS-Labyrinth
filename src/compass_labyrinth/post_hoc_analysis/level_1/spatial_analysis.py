"""
STATE DISTRIBUTIONS BY NODE-TYPE AND REGION
Author: Shreya Bangera
Goal:
    ├── Comparison of proportion of time spent in a state across Maze regions and Node types.
    ├── Allows genotype level comparisons behavioral states.
"""

import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)


##################################################################
# Plot 2: Probability of Surveillance across Node Types and Regions
###################################################################
def compute_state_probability(
    df_hmm,
    column_of_interest,
    values_displayed=None,
    decision_3way=None,
    decision_4way=None,
    state=1,
):
    """
    Computes HMM state proportions by category (e.g., NodeType or Region).
    Optionally reassigns decision node labels for 3-way and 4-way decisions.

    Parameters:
    - df_hmm: pd.DataFrame with 'Genotype', 'Session', 'HMM_State', and category column
    - column_of_interest: str, 'NodeType' or 'Region'
    - values_displayed: Optional[List[str]], categories to include and order
    - decision_3way: Optional[List[int]], grid numbers for 3-way decisions (only used if column_of_interest is 'NodeType')
    - decision_4way: Optional[List[int]], grid numbers for 4-way decisions (only used if column_of_interest is 'NodeType')
    - state: int, HMM_state of interest

    Returns:
    - pd.DataFrame with proportions per session
    """

    df_plot = df_hmm.copy()

    # Optional reassignment of NodeType for 3-way / 4-way decisions
    if column_of_interest == "NodeType" and decision_3way and decision_4way:
        df_plot.loc[df_plot["Grid.Number"].isin(decision_3way), "NodeType"] = "3-way Decision (Reward)"
        df_plot.loc[df_plot["Grid.Number"].isin(decision_4way), "NodeType"] = "4-way Decision (Reward)"
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
    state_count,
    column_of_interest,
    state=1,
    figsize=(14, 7),
    palette="Set2",
):
    """
    Plots boxplot of HMM state probabilities by category and genotype.

    Parameters:
    - state_count: pd.DataFrame returned from compute_state_probability()
    - column_of_interest: str, categorical variable on x-axis
    - state: int, HMM state used for labeling
    - figsize: tuple, figure size
    - palette: seaborn palette

    Returns:
    - matplotlib Axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    ax = sns.boxplot(x=column_of_interest, y="prop", hue="Genotype", data=state_count, palette=palette)
    ax.set_ylabel(f"Probability of being in State {state}", fontsize=15)
    ax.set_xlabel(column_of_interest, fontsize=15)
    plt.xticks(size=11)
    plt.yticks(size=15)
    plt.tight_layout()

    return ax


##################################################################
# T-Tests per genotype combo
###################################################################


def run_pairwise_ttests(
    state_count_df,
    column_of_interest="NodeType",
):
    """
    Perform pairwise t-tests between genotypes within each level of the column_of_interest.

    Parameters:
    - state_count_df: pd.DataFrame returned from compute_state_probability
    - column_of_interest: str, column over which comparisons are grouped

    Returns:
    - pd.DataFrame with columns: [Group, Genotype1, Genotype2, t-stat, p-value]
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
