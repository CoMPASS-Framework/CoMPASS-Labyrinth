"""
MULTI-AGENT MODELING
Author: Shreya Bangera
Goal:
   ├── Simulated Agent, Binary Agent, 3/4 way Agent Modelling
   ├── Comparsion across Agents
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from compass_labyrinth.behavior.behavior_metrics.simulation_modeling.explore_exploit_agent import trim_to_common_epochs


##################################################################
# Simulated Agent, Binary Agent, 3/4-way Agent Modelling & Comparison
###################################################################
# -------------------- Step 0: Chunking Utility -------------------- #
def split_into_epochs_multi(df, epoch_size):
    epochs = []
    for session, sess_df in df.groupby("Session"):
        for i in range(0, len(sess_df), epoch_size):
            chunk = sess_df.iloc[i : i + epoch_size]
            if not chunk.empty:
                epochs.append((session, i // epoch_size + 1, chunk))
    return epochs


# -------------------- Step 1: Transition Tracking -------------------- #
def track_valid_transitions_multi(df, decision_label, reward_label):
    session_valid = {}
    session_optimal = {}

    for session, group in df.groupby("Session"):
        valid_dict = {}
        optimal_dict = {}

        for i in range(len(group) - 1):
            if group.iloc[i]["NodeType"] == decision_label:
                curr_grid = group.iloc[i]["Grid Number"]
                next_grid = group.iloc[i + 1]["Grid Number"]
                next_region = group.iloc[i + 1]["Region"]

                valid_dict.setdefault(curr_grid, set()).add(next_grid)

                if next_region == reward_label:
                    optimal_dict.setdefault(curr_grid, set()).add(next_grid)

        session_valid[session] = valid_dict
        session_optimal[session] = optimal_dict

    return session_valid, session_optimal


# -------------------- Step 2: Simulated Agent Logic -------------------- #
def simulate_random_agent_multi(chunk, valid_dict, optimal_dict, decision_label, n_simulations):
    actual, random_perf = [], []

    for i in range(len(chunk) - 1):
        if chunk.iloc[i]["NodeType"] == decision_label:
            curr = chunk.iloc[i]["Grid Number"]
            next_actual = chunk.iloc[i + 1]["Grid Number"]

            is_opt = next_actual in optimal_dict.get(curr, set())
            actual.append(1 if is_opt else 0)

            sim_choices = [
                1 if random.choice(list(valid_dict[curr])) in optimal_dict.get(curr, set()) else 0
                for _ in range(n_simulations)
                if curr in valid_dict
            ]
            if sim_choices:
                random_perf.append(np.mean(sim_choices))

    return actual, random_perf


def simulate_binary_agent_multi(chunk, valid_dict, optimal_dict, decision_label, n_simulations):
    binary_perf = []

    for i in range(len(chunk) - 1):
        if chunk.iloc[i]["NodeType"] == decision_label:
            curr = chunk.iloc[i]["Grid Number"]
            choices = list(valid_dict.get(curr, []))

            opt = [x for x in choices if x in optimal_dict.get(curr, set())]
            non_opt = [x for x in choices if x not in opt]

            if opt and non_opt:
                sim_choices = [opt[0], non_opt[0]]
            elif len(choices) >= 2:
                sim_choices = random.sample(choices, 2)
            else:
                continue

            binary_opt = [1 if random.choice(sim_choices) in opt else 0 for _ in range(n_simulations)]
            binary_perf.append(np.mean(binary_opt))

    return binary_perf


def simulate_multiway_agent_multi(
    chunk, valid_dict, optimal_dict, decision_label, three_nodes, four_nodes, n_simulations
):
    perf = []

    for i in range(len(chunk) - 1):
        if chunk.iloc[i]["NodeType"] == decision_label:
            curr = chunk.iloc[i]["Grid Number"]

            prob = None
            if curr in three_nodes:
                prob = 1 / 3
            elif curr in four_nodes:
                prob = 1 / 4

            if prob:
                perf.append(np.mean([1 if random.random() < prob else 0 for _ in range(n_simulations)]))

    return perf


# -------------------- Step 3: Metric Evaluation -------------------- #
def bootstrap_means_multi(data, n):
    return np.mean(np.random.choice(data, (n, len(data)), replace=True), axis=1)


def evaluate_epoch_multi(
    chunk, valid_dict, optimal_dict, decision_label, reward_label, three_nodes, four_nodes, n_bootstrap, n_simulations
):

    if chunk.empty or decision_label not in chunk["NodeType"].values:
        return pd.Series(dtype="float64")  # Empty metrics

    actual, random_perf = simulate_random_agent_multi(chunk, valid_dict, optimal_dict, decision_label, n_simulations)
    binary_perf = simulate_binary_agent_multi(chunk, valid_dict, optimal_dict, decision_label, n_simulations)
    multiway_perf = simulate_multiway_agent_multi(
        chunk, valid_dict, optimal_dict, decision_label, three_nodes, four_nodes, n_simulations
    )

    if not (actual and random_perf and binary_perf and multiway_perf):
        return pd.Series(dtype="float64")

    actual_boot = bootstrap_means_multi(actual, n_bootstrap)
    random_boot = bootstrap_means_multi(random_perf, n_bootstrap)
    binary_boot = bootstrap_means_multi(binary_perf, n_bootstrap)
    multi_boot = bootstrap_means_multi(multiway_perf, n_bootstrap)

    return pd.Series(
        {
            "Actual Reward Path %": actual_boot.mean(),
            "Random Agent Reward Path %": random_boot.mean(),
            "Binary Agent Reward Path %": binary_boot.mean(),
            "Three/Four Way Agent Reward Path %": multi_boot.mean(),
            "Actual Reward Path % CI Lower": np.percentile(actual_boot, 5),
            "Actual Reward Path % CI Upper": np.percentile(actual_boot, 95),
            "Random Agent Reward Path % CI Lower": np.percentile(random_boot, 5),
            "Random Agent Reward Path % CI Upper": np.percentile(random_boot, 95),
            "Binary Agent Reward Path % CI Lower": np.percentile(binary_boot, 5),
            "Binary Agent Reward Path % CI Upper": np.percentile(binary_boot, 95),
            "Three/Four Way Agent Reward Path % CI Lower": np.percentile(multi_boot, 5),
            "Three/Four Way Agent Reward Path % CI Upper": np.percentile(multi_boot, 95),
            "Relative Performance (Actual/Random)": (
                actual_boot.mean() / random_boot.mean() if random_boot.mean() > 0 else np.nan
            ),
            "Relative Performance (Actual/Binary)": (
                actual_boot.mean() / binary_boot.mean() if binary_boot.mean() > 0 else np.nan
            ),
        }
    )


# -------------------- Step 4: Main Evaluation Wrapper -------------------- #
def evaluate_agent_performance_multi(
    df: pd.DataFrame,
    epoch_size: int,
    n_bootstrap: int,
    n_simulations: int,
    decision_label: str = "Decision (Reward)",
    reward_label: str = "reward_path",
    genotype: str | None = None,
    trim: bool = True,
    three_nodes: list | None = None,
    four_nodes: list | None = None,
):
    """
    Evaluate the performance of different agent types over multiple epochs.
    """
    if three_nodes is None:
        three_nodes = [20, 17, 39, 51, 63, 60, 77, 89, 115, 114, 110, 109, 98]
    if four_nodes is None:
        four_nodes = [32, 14]

    valid_dict_all, optimal_dict_all = track_valid_transitions_multi(df, decision_label, reward_label)
    epochs = split_into_epochs_multi(df, epoch_size)

    all_results = []
    for session, idx, chunk in epochs:
        valid_dict = valid_dict_all.get(session, {})
        optimal_dict = optimal_dict_all.get(session, {})
        metrics = evaluate_epoch_multi(
            chunk,
            valid_dict,
            optimal_dict,
            decision_label,
            reward_label,
            three_nodes,
            four_nodes,
            n_bootstrap,
            n_simulations,
        )
        metrics["Session"] = int(session)
        metrics["Epoch Number"] = int(idx)
        all_results.append(metrics)

    results = pd.DataFrame(all_results)
    if trim:
        results = trim_to_common_epochs(results)

    return results


##################################################################
## Plot 5: All Agents Comparative Performance over time
###################################################################
def plot_agent_vs_mouse_performance_multi(df_metrics, mouseinfo, genotype, figsize=(12, 6)):
    """
    Plot actual vs. simulated agent reward path performance across epochs for a specified genotype.

    Parameters:
        df_metrics (pd.DataFrame): Output from evaluate_agent_performance_multi().
        mouseinfo (pd.DataFrame): Metadata mapping sessions to genotypes.
        genotype (str): Genotype to filter (e.g., 'WT-WT').
        figsize (tuple): Size of the plot.
    """
    # --- Constants ---
    x_col = "Epoch Number"
    y_col_actual = "Actual Reward Path %"
    y_col_random = "Random Agent Reward Path %"
    y_col_binary = "Binary Agent Reward Path %"
    y_col_multi = "Three/Four Way Agent Reward Path %"
    title = "Mouse vs. Agent Reward Path Transition Proportion"

    # --- Filter sessions by genotype ---
    sessions_reqd = mouseinfo.loc[mouseinfo.Genotype == genotype, "Session #"].unique()
    df_filtered = df_metrics[df_metrics["Session"].isin(sessions_reqd)].copy()

    # --- Plot ---
    plt.figure(figsize=figsize)

    sns.lineplot(data=df_filtered, x=x_col, y=y_col_actual, marker="o", label="Mouse", color="black")
    sns.lineplot(data=df_filtered, x=x_col, y=y_col_random, linestyle="dashed", label="Random Agent", color="navy")
    sns.lineplot(data=df_filtered, x=x_col, y=y_col_binary, linestyle="dashed", label="Binary Agent", color="green")
    sns.lineplot(
        data=df_filtered, x=x_col, y=y_col_multi, linestyle="dashed", label="Three/Four Way Agent", color="maroon"
    )

    plt.xlabel("Epochs (in maze)", fontsize=12)
    plt.ylabel("Proportion of Reward Path Transitions", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Agent")
    plt.tight_layout()
    # plt.show()


###################################################################
## Plot 6: Cumulative Agent Performance
###################################################################
def plot_cumulative_agent_comparison_boxplot_multi(df_metrics, mouseinfo, genotype, figsize=(10, 6)):
    """
    Plots a boxplot comparing the cumulative reward path transition percentage
    across all sessions for the specified genotype for mouse and simulated agents.

    Parameters:
        df_metrics (pd.DataFrame): Output of `evaluate_agent_performance_multi()`.
        mouseinfo (pd.DataFrame): DataFrame with genotype and session mapping.
        genotype (str): Genotype to filter for comparison.
        show (bool): Whether to display the plot immediately.
    """
    # --- Constants ---
    metric_cols = {
        "Mouse": "Actual Reward Path %",
        "Random Agent": "Random Agent Reward Path %",
        "Binary Agent": "Binary Agent Reward Path %",
        "3/4-Way Agent": "Three/Four Way Agent Reward Path %",
    }

    # --- Filter sessions for the genotype ---
    sessions_reqd = mouseinfo.loc[mouseinfo.Genotype == genotype, "Session #"].unique()
    df_filtered = df_metrics[df_metrics["Session"].isin(sessions_reqd)].copy()

    # --- Aggregate to session level (mean across epochs) ---
    df_agg = df_filtered.groupby("Session")[[*metric_cols.values()]].mean().reset_index()

    # --- Melt for plotting ---
    df_melt = df_agg.melt(id_vars="Session", var_name="Agent", value_name="Reward Path %")
    df_melt["Agent"] = df_melt["Agent"].map({v: k for k, v in metric_cols.items()})

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.boxplot(data=df_melt, x="Agent", y="Reward Path %", palette="Set2")
    sns.stripplot(data=df_melt, x="Agent", y="Reward Path %", color="black", size=4, jitter=True, alpha=0.6)

    plt.title(
        f"Cumulative Reward Path Transition % across Sessions\nGenotype: {genotype}", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Mean Reward Path Transition %")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
