"""
SIMULATED AGENT MODELING AND ANALYSIS
Author: Shreya Bangera
Goal:
   ├── Simulated Agent Modeling & Visualisation
   ├── Chi Square Analysis, Visualisation
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
import math
import warnings


warnings.filterwarnings("ignore")

##################################################################
# Simulated Agent Modelling
###################################################################
def get_valid_and_optimal_transitions(
    df: pd.DataFrame,
    decision_label: str = "Decision (Reward)",
    reward_label: str = "reward_path",
) -> tuple[dict, dict]:
    """
    Extract valid and optimal transitions per session.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing navigation data.
    decision_label : str
        Label for decision points.
    reward_label : str
        Label for reward path.
    """
    valid_transitions, optimal_transitions = {}, {}

    for session, group in df.groupby("Session"):
        valid, optimal = {}, {}

        for i in range(len(group) - 1):
            if group.iloc[i]["NodeType"] == decision_label:
                current = group.iloc[i]["Grid Number"]
                nxt = group.iloc[i + 1]["Grid Number"]
                region = group.iloc[i + 1]["Region"]

                valid.setdefault(current, set()).add(nxt)
                if region == reward_label:
                    optimal.setdefault(current, set()).add(nxt)

        valid_transitions[session] = valid
        optimal_transitions[session] = optimal

    return valid_transitions, optimal_transitions


def simulate_agent_vs_actual(
    df_slice : pd.DataFrame,
    valid_dict : dict,
    optimal_dict : dict,
    n_simulations : int,
    decision_label : str = "Decision (Reward)",
) -> tuple[list, list]:
    """
    Simulate random agent transitions and compare with actual.
    
    Parameters:
    -----------
    df_slice : pd.DataFrame
        DataFrame segment for the epoch.
    valid_dict : dict
        Valid transitions for the session.
    optimal_dict : dict
        Optimal transitions for the session.
    n_simulations : int
        Number of random simulations per decision point.
    decision_label : str
        Label for decision points.

    Returns:
    --------
    tuple of lists
        Lists of actual and simulated optimal transitions (1 for optimal, 0 otherwise).
    """
    actual, simulated = [], []

    for i in range(len(df_slice) - 1):
        if df_slice.iloc[i]["NodeType"] == decision_label:
            current = df_slice.iloc[i]["Grid Number"]
            actual_next = df_slice.iloc[i + 1]["Grid Number"]

            is_actual_optimal = actual_next in optimal_dict.get(current, set())
            actual.append(1 if is_actual_optimal else 0)

            rand_results = []
            for _ in range(n_simulations):
                if current in valid_dict:
                    rand_choice = random.choice(list(valid_dict[current]))
                    is_rand_optimal = rand_choice in optimal_dict.get(current, set())
                    rand_results.append(1 if is_rand_optimal else 0)
            simulated.append(np.mean(rand_results))

    return actual, simulated


def bootstrap_distribution(
    data: list,
    n_samples: int = 10000,
) -> np.ndarray:
    """
    Generate bootstrap sample means.
    
    Parameters:
    -----------
    data : list
        Data points.
    n_samples : int
        Number of bootstrap samples.

    Returns:
    --------
    np.ndarray
        Array of bootstrap sample means.
    """
    samples = np.random.choice(data, (n_samples, len(data)), replace=True)
    return np.mean(samples, axis=1)


def compute_epoch_metrics(
    df_slice : pd.DataFrame,
    valid_dict : dict,
    optimal_dict : dict,
    n_bootstrap : int,
    n_simulations : int,
    decision_label : str = "Decision (Reward)",
) -> pd.Series:
    """
    Compute performance metrics for a single epoch of navigation.
    
    Parameters:
    -----------
    df_slice : pd.DataFrame
        DataFrame segment for the epoch.
    valid_dict : dict
        Valid transitions for the session.
    optimal_dict : dict
        Optimal transitions for the session.
    n_bootstrap : int
        Number of bootstrap samples.
    n_simulations : int
        Number of random simulations per decision point.
    decision_label : str
        Label for decision points.

    Returns:
    --------
    pd.Series
        Series with computed metrics.
    """
    if df_slice.empty or decision_label not in df_slice["NodeType"].values:
        return pd.Series(
            {
                k: np.nan
                for k in [
                    "Actual Reward Path %",
                    "Simulated Agent Reward Path %",
                    "Actual Reward Path % CI Lower",
                    "Actual Reward Path % CI Upper",
                    "Simulated Agent Reward Path % CI Lower",
                    "Simulated Agent Reward Path % CI Upper",
                    "Relative Performance",
                ]
            }
        )

    actual, simulated = simulate_agent_vs_actual(df_slice, valid_dict, optimal_dict, n_simulations, decision_label)

    if not actual or not simulated:
        return pd.Series(
            {
                k: np.nan
                for k in [
                    "Actual Reward Path %",
                    "Simulated Agent Reward Path %",
                    "Actual Reward Path % CI Lower",
                    "Actual Reward Path % CI Upper",
                    "Simulated Agent Reward Path % CI Lower",
                    "Simulated Agent Reward Path % CI Upper",
                    "Relative Performance",
                ]
            }
        )

    actual_dist = bootstrap_distribution(actual, n_bootstrap)
    simulated_dist = bootstrap_distribution(simulated, n_bootstrap)

    return pd.Series(
        {
            "Actual Reward Path %": np.mean(actual_dist),
            "Simulated Agent Reward Path %": np.mean(simulated_dist),
            "Actual Reward Path % CI Lower": np.percentile(actual_dist, 5),
            "Actual Reward Path % CI Upper": np.percentile(actual_dist, 95),
            "Simulated Agent Reward Path % CI Lower": np.percentile(simulated_dist, 5),
            "Simulated Agent Reward Path % CI Upper": np.percentile(simulated_dist, 95),
            "Relative Performance": (
                np.mean(actual_dist) / np.mean(simulated_dist) if np.mean(simulated_dist) > 0 else np.nan
            ),
        }
    )


def segment_data_by_epoch(
    df: pd.DataFrame,
    epoch_size: int,
) -> list:
    """
    Split DataFrame by genotype and session into sequential time-based epochs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing navigation data.
    epoch_size : int
        Number of rows per epoch.

    Returns:
    --------
    list of tuples
        Each tuple contains (session, epoch_number, epoch_dataframe).
    """
    epochs = []
    for (genotype, session), group in df.groupby(["Genotype", "Session"]):
        for i in range(0, len(group), epoch_size):
            segment = group.iloc[i : i + epoch_size]
            if not segment.empty:
                epochs.append((session, i // epoch_size + 1, segment))
    return epochs


def trim_to_common_epochs(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Trims the results dataframe to retain only the maximum number of epochs common across all sessions.

    Parameters:
    -----------
    df_results : pd.DataFrame
        The output of evaluate_agent_performance.
            - 'Session' (str): Column name indicating sessions.
            - 'Epoch_Number' (str): Column name indicating epoch/bin number.

    Returns:
    --------
    pd.DataFrame
        Trimmed dataframe with only common epochs.
    """
    df_trimmed = df_results.copy()

    # Ensure correct dtypes
    df_trimmed["Session"] = df_trimmed["Session"].astype(int)
    df_trimmed["Epoch Number"] = df_trimmed["Epoch Number"].astype(int)

    # Find common epochs across all sessions
    epoch_sets = df_trimmed.groupby("Session")["Epoch Number"].apply(set)
    common_epochs = set.intersection(*epoch_sets)

    if not common_epochs:
        print("Warning: No common epochs across sessions. Returning original dataframe.")
        return df_trimmed

    max_common_epoch = max(common_epochs)
    print(f" Max common epoch across all sessions: {max_common_epoch}")

    # Filter
    df_trimmed = df_trimmed[df_trimmed["Epoch Number"] <= max_common_epoch].reset_index(drop=True)
    return df_trimmed


def evaluate_agent_performance(
    df: pd.DataFrame,
    epoch_size: int,
    n_bootstrap: int,
    n_simulations: int,
    decision_label: str = "Decision (Reward)",
    reward_label: str = "reward_path",
    genotype: str | None = None,
    trim: bool = True,
) -> pd.DataFrame:
    """
    Run full evaluation pipeline for simulated agent vs. actual mouse.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing navigation data.
    epoch_size : int
        Number of rows per epoch.
    n_bootstrap : int
        Number of bootstrap samples.
    n_simulations : int
        Number of random simulations per decision point.
    decision_label : str
        Label for decision points.
    reward_label : str
        Label for reward path.
    genotype : str | None
        Genotype to filter data.
    trim : bool
        Whether to trim to common epochs across sessions.

    Returns:
    --------
    pd.DataFrame
        DataFrame with performance metrics per epoch.
    """
    df = df.copy()
    
    # Filter by genotype if specified
    if genotype is not None:
        if genotype not in df["Genotype"].unique():
            raise ValueError(f"Genotype '{genotype}' not found in DataFrame.")
        genotypes = [genotype]
    else:
        genotypes = df["Genotype"].unique()

    results = dict()
    for i, genotype in enumerate(genotypes):
        df_genotype = df.loc[df["Genotype"] == genotype]

        valid_dict, optimal_dict = get_valid_and_optimal_transitions(df_genotype, decision_label, reward_label)
        epochs = segment_data_by_epoch(df_genotype, epoch_size)

        all_results = []
        for session, epoch_num, segment in epochs:
            valid = valid_dict.get(session, {})
            optimal = optimal_dict.get(session, {})
            result = compute_epoch_metrics(segment, valid, optimal, n_bootstrap, n_simulations, decision_label)
            result["Session"] = session
            result["Epoch Number"] = epoch_num
            all_results.append(result)

        if trim:
            df_results = pd.DataFrame(all_results)
            df_results = trim_to_common_epochs(df_results)
        else:
            df_results = pd.DataFrame(all_results)

        results[genotype] = df_results

    return results


############################################################################
## Plot 1: Simulated Agent v/s Mouse Performance across Time
#############################################################################
def plot_agent_transition_performance(
    config: dict,
    evaluation_results: dict,
    genotype: str | None = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot performance comparison between actual mouse and simulated agent over time.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with project path.
    evaluation_results : dict
        Dictionary with evaluation results for each genotype.
    genotype : str | None
        Specific genotype to plot. If None, plots all genotypes.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    if genotype is not None:
        if genotype not in evaluation_results:
            raise ValueError(f"Genotype '{genotype}' not found in evaluation results.")
        genotypes = [genotype]
    else:
        genotypes = evaluation_results.keys()
    n_genotypes = len(genotypes)

    n_cols = math.ceil(np.sqrt(n_genotypes))
    n_rows = math.ceil(n_genotypes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        ax = axes[i]
        df_result = evaluation_results[genotype]

        sns.lineplot(
            data=df_result,
            x="Epoch Number",
            y="Actual Reward Path %",
            marker="o",
            label="Mouse",
            color="black",
            ax=ax,
        )
        sns.lineplot(
            data=df_result,
            x="Epoch Number",
            y="Simulated Agent Reward Path %",
            linestyle="dashed",
            label="Simulated Agent",
            color="navy",
            ax=ax,
        )

        ax.set_title(f"{genotype}: Mouse vs. Agent")
        ax.set_xlabel("Epochs (in Maze)")
        ax.set_ylabel("Reward Path Transition %")
        ax.grid(True)
        ax.legend()

    # Hide unused axes
    for j in range(len(genotypes), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Mouse vs. Simulated Agent: Reward Path Transition Proportion", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig = plt.gcf()
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_sim_agent_mouse_perf.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


##################################################################
## Plot 2: Relative Performance across Time
###################################################################
def plot_relative_agent_performance(
    config: dict,
    evaluation_results: dict,
    genotype: str | None = None,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot relative performance of mouse vs simulated agent over time.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with project path.
    evaluation_results : dict
        Dictionary with evaluation results for each genotype.
    genotype : str | None
        Specific genotype to plot. If None, plots all genotypes.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    if genotype is not None:
        if genotype not in evaluation_results:
            raise ValueError(f"Genotype '{genotype}' not found in evaluation results.")
        genotypes = [genotype]
    else:
        genotypes = evaluation_results.keys()
    n_genotypes = len(genotypes)

    n_cols = 1
    n_rows = n_genotypes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        ax = axes[i]
        df_result = evaluation_results[genotype]
        sns.lineplot(
            data=df_result,
            x="Epoch Number",
            y="Relative Performance",
            marker="o",
            color="black",
            ax=ax,
        )
        ax.axhline(
            y=1,
            color="black",
            linestyle="dashed",
            label="Simulated Agent Baseline",
        )

        ax.set_xlabel("Epochs (in Maze)")
        ax.set_ylabel("Relative Performance (Mouse / Simulated)")
        ax.set_title(f"{genotype}: Mouse vs. Simulated Agent - Relative Performance Over Time")
        ax.legend(["Relative Performance", "Simulated Agent Baseline"])
        ax.grid(True)
        plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_relative_perf.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig
        

#############################################################################
## Plot 3: Avg. Simulated Agent and Mouse Performance across Sessions(/Mice)
##############################################################################
def fit_mixed_effects_model(df_long: pd.DataFrame) -> tuple:
    """
    Fit a linear mixed-effects model comparing agent types.

    Parameters:
    -----------
    df_long : pd.DataFrame
        Long-form DataFrame with columns 'AgentType', 'Performance', and session info.

    Returns:
    --------
    tuple
        Tuple with result (Fitted model object) and p_value (P-value for AgentType effect).
    """
    model = mixedlm("Performance ~ AgentType", df_long, groups=df_long["Session"])
    result = model.fit()

    # Automatically detect which coefficient relates to the simulated agent
    coef_key = [key for key in result.pvalues.keys() if "Simulated Agent" in key]
    p_value = result.pvalues.get(coef_key[0], np.nan) if coef_key else np.nan
    return result, p_value


def plot_agent_performance_boxplot(
    df_long: pd.DataFrame,
    p_value: float,
    palette: None | list = None
) -> None:
    """
    Plot boxplot comparing actual vs simulated agent with p-value annotation.

    Parameters:
    -----------
    df_long : pd.DataFrame
        Long-form DataFrame.
    p_value : float
        P-value from mixed model.
    palette : list or None
        Color palette for the boxplot.

    Returns:
    --------
    None
    """
    plt.figure(figsize=(6, 6))
    sns.boxplot(x="AgentType", y="Performance", data=df_long, palette=palette, showfliers=False)

    plt.title(f"Performance: Mouse vs. Simulated Agent (across sessions)\n LMM p-value = {p_value:.4f}", fontsize=13)
    plt.xlabel("Agent Type", fontsize=11)
    plt.ylabel("Proportion of Optimal Transitions", fontsize=11)
    plt.xticks(ticks=[0, 1], labels=["Mouse", "Simulated Agent"], fontsize=10)
    plt.tight_layout()


##########################################################################################################################
## Avg. Simulated Agent and Mouse Performance across Sessions(/Mice) for all genotypes (when multiple genotypes)
###########################################################################################################################
# -----------------------------------------
# Reshape for MixedLM
# -----------------------------------------
def reshape_for_mixedlm(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the dataframe to long format for mixed-effects modeling.

    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame with columns 'Actual Reward Path %', 'Simulated Agent Reward Path %',
        'Session', 'Epoch Number' and 'Genotype'.

    Returns:
    --------
    pd.DataFrame
        Long-form DataFrame suitable for mixedlm.
    """
    df_long = pd.melt(
        df_results,
        id_vars=["Session", "Epoch Number", "Genotype"],
        value_vars=["Actual Reward Path %", "Simulated Agent Reward Path %"],
        var_name="AgentType",
        value_name="Performance",
    )
    df_long = df_long.dropna(subset=["Performance"])
    df_long["Session"] = df_long["Session"].astype(str)
    return df_long.reset_index(drop=True)


# -----------------------------------------
# Fit Mixed Effects Model
# -----------------------------------------
def fit_mixed_effects_model(df_long: pd.DataFrame) -> tuple:
    """
    Fit a linear mixed-effects model comparing agent types.

    Parameters:
    -----------
    df_long : pd.DataFrame
        Long-form DataFrame.

    Returns:
    --------
    tuple
        Tuple with result (Fitted model object) and p_value (P-value for AgentType effect).
    """
    model = mixedlm("Performance ~ AgentType", df_long, groups=df_long["Session"])
    result = model.fit()
    coef_key = [key for key in result.pvalues.keys() if "Simulated Agent" in key]
    p_value = result.pvalues.get(coef_key[0], np.nan) if coef_key else np.nan
    return result, p_value


# -----------------------------------------
# Plotting per Axes
# -----------------------------------------
def plot_agent_performance_boxplot_ax(
    ax: plt.Axes,
    df_long: pd.DataFrame,
    p_value: float,
    palette: list | None = None,
    genotype: str | None = None,
) -> None:
    """
    Plot a boxplot of agent performance.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib Axes object to plot on.
    df_long : pd.DataFrame
        Long-form DataFrame.
    p_value : float
        P-value from mixed model.
    palette : list or None
        Color palette for the boxplot.
    genotype : str or None
        Genotype name for the title.

    Returns:
    --------
    None
    """
    sns.boxplot(x="AgentType", y="Performance", data=df_long, palette=palette, showfliers=False, ax=ax)
    title = f"Mouse vs. Agent Performance\n{genotype} | LMM p = {p_value:.4f}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Agent Type", fontsize=10)
    ax.set_ylabel("Proportion Optimal", fontsize=10)


# -----------------------------------------
# Main Runner: Across Genotypes
# -----------------------------------------
def run_mixedlm_for_all_genotypes(
    config: dict,
    evaluation_results: dict,
    plot_palette=None,
    save_fig: bool = True,
    show_fig: bool = True,
) -> dict:
    """
    Run mixed-effects modeling and plot results for all genotypes.

    Parameters:
    -----------
    config : dict
        Configuration dictionary for this project.
    evaluation_results : dict
        Dictionary with evaluation results for each genotype.
    plot_palette : list or None
        Color palette for the boxplots.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.

    Returns:
    --------
    dict
        Dictionary with p-values for each genotype.
    """
    genotype_pvals = {}
    all_dfs_long = []

    genotypes = evaluation_results.keys()
    n_genotypes = len(genotypes)

    n_cols = math.ceil(n_genotypes**0.5)
    n_rows = math.ceil(n_genotypes / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Safe handling: ensure axs is always iterable
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, genotype in enumerate(genotypes):
        df_eval = evaluation_results[genotype]
        df_eval["Genotype"] = genotype

        df_long = reshape_for_mixedlm(df_eval)
        result, p_val = fit_mixed_effects_model(df_long)
        genotype_pvals[genotype] = p_val

        plot_agent_performance_boxplot_ax(axs[i], df_long, p_val, palette=plot_palette, genotype=genotype)
        all_dfs_long.append(df_long)

    # Hide unused axes
    for j in range(n_genotypes, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "cumulative_sim_agent_mouse_perf.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    return genotype_pvals


##################################################################
# Chi Square Analysis
###################################################################
def compute_chi_square_statistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the chi-square statistic between actual and simulated reward path usage
    for each row in the DataFrame. Also ensures 'Epoch Number' and 'Session' are integers.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns 'Actual Reward Path %' and 'Simulated Agent Reward Path %'.
    
    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with 'Chi Square Statistic' and cleaned column types.
    """
    df = df.copy()
    chi_square = ((df["Actual Reward Path %"] - df["Simulated Agent Reward Path %"]) ** 2) / df[
        "Simulated Agent Reward Path %"
    ]
    df["Chi Square Statistic"] = chi_square
    # Ensure consistent types
    if "Epoch Number" in df.columns:
        df["Epoch Number"] = df["Epoch Number"].astype(int)
    if "Session" in df.columns:
        df["Session"] = df["Session"].astype(int)
    return df


def compute_rolling_chi_square(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Compute rolling average of chi-square statistic within each session.
    
    Patameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Chi Square Statistic' column.
    window : int
        Window size for rolling average.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with 'Rolling Chi Square' column.
    """
    df = df.copy()
    df["Rolling Chi Square"] = df.groupby("Session")["Chi Square Statistic"].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    return df


def compute_cumulative_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative sum of chi-square statistic within each session.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Chi Square Statistic' column.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with 'Cumulative Chi Square' column.
    """
    df = df.copy()
    df["Cumulative Chi Square"] = df.groupby("Session")["Chi Square Statistic"].cumsum()
    return df


#############################################################################################
## Chi Square Statistic of Agents across Time
#############################################################################################
def run_chi_square_analysis(
    config: dict,
    evaluation_results: dict,
    rolling_window: int = 3,
) -> dict:
    """
    Run chi-square analysis for each genotype in the evaluation results.
    """
    genotypes = evaluation_results.keys()
    results = dict()
    for genotype in genotypes:
        df_result = evaluation_results[genotype].copy()
        df_result["Genotype"] = genotype
        df_chisq = compute_chi_square_statistic(df=df_result)
        df_chisq = compute_rolling_chi_square(df=df_chisq, window=rolling_window)
        df_chisq = compute_cumulative_chi_square(df=df_chisq)
        results[genotype] = df_chisq
    return results


def plot_chi_square_and_rolling(
    config: dict,
    chisquare_results: dict,
    epoch_col: str = "Epoch Number",
    chi_col: str = "Chi Square Statistic",
    rolling_col: str = "Rolling Chi Square",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot chi-square and rolling statistics for each genotype.

    Parameters:
    -----------
    config : dict
        Configuration dictionary for this project.
    chisquare_results : dict
        Chi-square results dictionary.
    epoch_col : str
        Column name for epochs.
    chi_col : str
        Column name for chi-square statistic.
    rolling_col : str
        Column name for rolling chi-square.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    genotypes = chisquare_results.keys()
    n_genotypes = len(genotypes)
    n_cols = 1
    n_rows = n_genotypes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        ax = axes[i]
        df_geno = chisquare_results[genotype]

        sns.barplot(data=df_geno, x=epoch_col, y=chi_col, errorbar="se", palette="viridis", ax=ax,)
        sns.lineplot(data=df_geno, x=epoch_col, y=rolling_col, color="black", lw=2, ax=ax,)

        ax.set_title(f"{genotype}: Chi-Square & Rolling")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Chi-Square")

    # Hide unused subplots
    for j in range(n_genotypes, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Chi-Square Statistic + Rolling Average by Genotype", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_chi_square_rolling.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


def plot_rolling_mean(
    config: dict,
    chisquare_results: dict,
    epoch_col: str = "Epoch Number",
    rolling_col: str = "Rolling Chi Square",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot rolling chi-square statistics for each genotype.

    Parameters:
    -----------
    config : dict
        Configuration dictionary for this project.
    chisquare_results : dict
        Chi-square results dictionary.
    epoch_col : str
        Column name for epochs.
    rolling_col : str
        Column name for rolling chi-square.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    genotypes = chisquare_results.keys()
    n_genotypes = len(genotypes)
    n_cols = 1
    n_rows = n_genotypes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        ax = axes[i]
        df_geno = chisquare_results[genotype]

        sns.barplot(
            data=df_geno,
            x=epoch_col,
            y=rolling_col,
            errorbar="se",
            palette="Blues",
            ax=ax,
        )
        ax.set_title(f"{genotype}: Rolling Chi-Square")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Rolling Stat")

    for j in range(len(genotypes), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Rolling Chi-Square by Genotype", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_average_chi_square_rolling.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


def plot_cumulative_chi_square(
    config: dict,
    chisquare_results: dict,
    epoch_col: str = "Epoch Number",
    cum_col: str = "Cumulative Chi Square",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot cumulative chi-square statistics for each genotype.

    Parameters:
    -----------
    config : dict
        Configuration dictionary for this project.
    chisquare_results : dict
        Chi-square results dictionary.
    epoch_col : str
        Column name for epochs.
    cum_col : str
        Column name for cumulative chi-square.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    genotypes = chisquare_results.keys()
    n_genotypes = len(genotypes)
    n_cols = 1
    n_rows = n_genotypes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        ax = axes[i]
        df_geno = chisquare_results[genotype]

        sns.barplot(data=df_geno, x=epoch_col, y=cum_col, errorbar="se", palette="magma", ax=ax,)
        ax.set_title(f"{genotype}: Cumulative Chi-Square")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cumulative Stat")

    for j in range(len(genotypes), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Cumulative Chi-Square by Genotype", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_cumulative_chi_square.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig
