from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from hmmlearn.hmm import GMMHMM
import matplotlib.patches as patches

from .utils import (
    assign_bouts_per_session,
    build_phase_map,
)


# ============================================================
# AIC Computation
# ============================================================
def compute_aic(model, X):
    log_likelihood = model.score(X)
    n_states = model.n_components
    n_features = model.means_.shape[-1]
    n_mixtures = model.n_mix
    num_params = (
        (n_states - 1)  # start prob
        + n_states * (n_states - 1)  # transition matrix
        + n_states * n_mixtures * n_features  # means
        + n_states * n_mixtures * n_features  # covariances
        + n_states * n_mixtures  # weights
    )
    return 2 * num_params - 2 * log_likelihood


# ============================================================
# Covariance Regularization
# ============================================================
def regularize_covariances(covariances, reg_val=1e-6):
    return np.array([cov + np.eye(cov.shape[0]) * reg_val for cov in covariances])


# ============================================================
# BGMM Initialization
# ============================================================
def initialize_bgmm(n_components, reg_val, random_state=3):
    return BayesianGaussianMixture(
        n_components=n_components, covariance_type="full", reg_covar=reg_val, random_state=random_state, max_iter=500
    )


# ============================================================
# GMM-HMM Computation
# ============================================================
def initialize_gmmhmm(n_states, n_mix, means, covariances, weights, random_state=3):
    model = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="full", random_state=random_state, n_iter=500)
    model.means_ = means[:, :n_mix, :]
    model.covars_ = covariances[:, :n_mix, :, :]
    model.weights_ = weights[:, :n_mix]
    return model


# ============================================================
# Run CoMPASS
# ============================================================
def run_compass(
    config: dict,
    df: pd.DataFrame,
    features: list,
    phase_options: list = [5],
    ncomp_options: range = range(2, 4),
    k_options: range = range(2, 4),
    reg_options: list = [1e-4, 1e-5, 1e-6],
    terminal_values: list = [47],
    bout_col: str = "Bout_ID",
    patience: None | str = None,
    patience_candidates: list = [2, 3, 5, 10],
) -> tuple[pd.DataFrame, list]:
    """
    Run CoMPASS.

    Parameters:
    -----------
    config : dict
        Configuration dictionary.
    df : pd.DataFrame
        Input dataframe.
    features : list
        List of feature column names to use.
    phase_options : list, optional
        List of phase options to test (default is [5]).
    ncomp_options : range, optional
        Range of number of components to test (default is range(2, 4)).
    k_options : range, optional
        Range of k values to test (default is range(2, 4)).
    reg_options : list, optional
        List of regularization values to test (default is [1e-4, 1e-5, 1e-6]).
    terminal_values : list, optional
        List of terminal grid values (default is [47]).
    bout_col : str, optional
        Name of the bout column (default is "Bout_ID").
    patience : None or str, optional
        Patience setting for early stopping (default is None).
        Set to 'tune' if wanted to apply patience window.
    patience_candidates : list, optional
        List of patience candidates to test if patience is 'tune' (default is [2, 3, 5, 10]).
        Only used if patience is set to 'tune'.

    Returns:
    --------
    tuple
        A tuple containing:
        - pd.DataFrame: DataFrame with assigned Level 2 states.
        - list: List of all CV results for visualization.
    """
    all_results = []
    final_sess_data = []

    df = assign_bouts_per_session(df, terminal_values=terminal_values, bout_col=bout_col)

    for n_phases in phase_options:
        sessions = df.Session.unique()
        phase_labels = range(n_phases)
        phase_map = build_phase_map(df, n_phases)

        for phase_index in phase_labels:
            for test_sess in sessions:
                print(f"\n=== CV | Test: Session {test_sess} - Phase {phase_index+1}/{n_phases} ===")

                test_bouts = phase_map[(test_sess, phase_index)]
                df_test = df[(df.Session == test_sess) & (df[bout_col].isin(test_bouts))]

                train_sessions = [s for s in sessions if s != test_sess]
                df_train_pool = pd.concat(
                    [df[(df.Session == s) & (df[bout_col].isin(phase_map[(s, phase_index)]))] for s in train_sessions]
                )

                inner_sessions = df_train_pool.Session.unique()
                best_log_lik = -np.inf
                best_aic = np.inf
                best_model = None

                best_patience = patience_candidates[0] if patience == "tune" else patience
                patience_results = {}

                for test_patience in (patience_candidates if patience == "tune" else [best_patience]):
                    log_liks, aics, param_labels = [], [], []
                    no_improve = 0
                    best_inner_loglik = -np.inf

                    early_stopped = False

                    for val_sess in inner_sessions:
                        df_val = df_train_pool[df_train_pool.Session == val_sess]
                        df_train = df_train_pool[df_train_pool.Session != val_sess]

                        for ncomp in ncomp_options:
                            for k in k_options:
                                for reg_val in reg_options:
                                    try:
                                        X_train = df_train[features].values
                                        X_val = df_val[features].values

                                        gmm = initialize_bgmm(ncomp, reg_val)
                                        gmm.fit(X_train)
                                        covariances = regularize_covariances(gmm.covariances_, reg_val)

                                        means = np.tile(gmm.means_[None, :, :], (ncomp, 1, 1))
                                        covars = np.tile(covariances[None, :, :, :], (ncomp, 1, 1, 1))
                                        weights = np.tile(gmm.weights_[None, :], (ncomp, 1))

                                        model = initialize_gmmhmm(ncomp, k, means, covars, weights)
                                        model.fit(X_train)

                                        log_lik = model.score(X_val)
                                        aic = compute_aic(model, X_train)
                                        label = f"n={ncomp},k={k},r={reg_val:.0e}"

                                        log_liks.append(log_lik)
                                        aics.append(aic)
                                        param_labels.append(label)

                                        if log_lik > best_inner_loglik:
                                            best_inner_loglik = log_lik
                                            best_model = model
                                            no_improve = 0
                                        else:
                                            no_improve += 1

                                        if test_patience is not None and no_improve >= test_patience:
                                            early_stopped = True
                                            break
                                    except Exception as e:
                                        print(f"Model failed for n={ncomp}, k={k}, reg={reg_val:.0e}: {e}")
                                        continue
                                if early_stopped:
                                    break
                            if early_stopped:
                                break

                    avg_loglik = np.mean(log_liks)
                    patience_results[test_patience] = (avg_loglik, best_model, log_liks, aics, param_labels)

                # Pick best patience if tuning
                if patience == "tune":
                    best_patience, (best_avg_loglik, best_model, log_liks, aics, param_labels) = max(
                        patience_results.items(), key=lambda x: x[1][0]
                    )
                    print(f"Optimal patience for Session {test_sess}, Phase {phase_index+1}: {best_patience}")

                if best_model is not None:
                    X_test = df_test[features].values
                    df_test = df_test.copy()
                    df_test["Level_2_States"] = best_model.predict(X_test)
                    final_sess_data.append(df_test)

                tag = f"Session:{test_sess}_PhaseIndex:{phase_index+1}_NumPhases:{n_phases}_Patience:{best_patience}"
                all_results.append((tag, log_liks, aics, param_labels))

    df_hier = pd.concat(final_sess_data)

    # Save results
    save_path = Path(config["project_path_full"]) / "csvs" / "combined" / "hhmm_state_file.csv"
    df_hier.to_csv(save_path, index=False)
    print(f"HHMM state file saved at: {save_path}")

    return df_hier, all_results


# ============================================================
# CV Performance Visualization
# ============================================================
def visualize_cv_results(all_results):
    for tag, log_liks, aics, param_labels in all_results:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(len(log_liks)), y=log_liks, marker="o")
        plt.xticks(ticks=np.arange(len(param_labels)), labels=param_labels, rotation=90)
        plt.title(f"{tag} - Log-Likelihoods")
        plt.xlabel("Param Config (n,k,reg)")
        plt.ylabel("Log-Likelihood")

        plt.subplot(1, 2, 2)
        sns.lineplot(x=np.arange(len(aics)), y=aics, marker="o")
        plt.xticks(ticks=np.arange(len(param_labels)), labels=param_labels, rotation=90)
        plt.title(f"{tag} - AIC")
        plt.xlabel("Param Config (n,k,reg)")
        plt.ylabel("AIC")

        plt.tight_layout()
        plt.show()


################################################################
# Observe the raw state sequence
################################################################
def get_unique_states(df, state_col="Level_2_States"):
    """Return sorted unique states from the specified column."""
    return sorted(df[state_col].dropna().unique())


def generate_state_color_map(unique_states, palette="tab10"):
    """Assign each state a color from the selected Seaborn palette."""
    colors = sns.color_palette(palette, len(unique_states))
    return {state: colors[i] for i, state in enumerate(unique_states)}


def plot_state_sequence_for_session(
    df_session, state_col="Level_2_States", color_map=None, title_prefix="State Sequence"
):
    """Plot the state sequence using color bars for one session."""
    df_session = df_session.reset_index(drop=True).copy()
    df_session["color"] = df_session[state_col].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 3))
    for idx, row in df_session.iterrows():
        rect = patches.Rectangle((idx, 0), 1, 1, color=row["color"])
        ax.add_patch(rect)

    ax.set_xlim(df_session.index.min(), df_session.index.max() + 1)
    ax.set_yticks([])
    ax.set_title(f"{title_prefix} - Session {df_session['Session'].iloc[0]}")

    legend_handles = [patches.Patch(color=color_map[state], label=f"State {state}") for state in color_map]
    ax.legend(
        handles=legend_handles, title="States", bbox_to_anchor=(0.5, -0.15), loc="upper center", borderaxespad=0.0
    )

    plt.tight_layout()
    plt.show()


def plot_state_sequences(
    df,
    genotype="WT-WT",
    state_col="Level_2_States",
    sessions_to_plot="all",  # Can be 'all', a list of session IDs, or an int (top n)
    title_prefix="State Sequence",
):
    """Plot state sequences for specified sessions and genotype."""
    df_geno = df[df["Genotype"] == genotype]
    unique_states = get_unique_states(df_geno, state_col=state_col)
    color_map = generate_state_color_map(unique_states)

    # Determine which sessions to plot
    all_sessions = df_geno["Session"].unique()
    if isinstance(sessions_to_plot, int):
        selected_sessions = all_sessions[:sessions_to_plot]
    elif isinstance(sessions_to_plot, list):
        selected_sessions = [s for s in sessions_to_plot if s in all_sessions]
    else:
        selected_sessions = all_sessions

    # Plot each selected session
    for sess_id in selected_sessions:
        df_sess = df_geno[df_geno["Session"] == sess_id][[state_col, "Session"]]
        plot_state_sequence_for_session(df_sess, state_col=state_col, color_map=color_map, title_prefix=title_prefix)


################################################################
# Create 4 level HHMM States
################################################################
def assign_reward_orientation(
    df,
    angle_col="Targeted_Angle_smooth_abs",
    level_2_state_col="Level_2_States",
    session_col="Session",
):
    """
    Assigns reward orientation labels ('Reward Oriented' or 'Non-Reward Oriented') to Level 2 states per session,
    based on the median Targeted_Angle_smooth within each state and relative to session median.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns for session, level 2 state, and angle.
        angle_col (str): Column name representing the smoothed targeted angle.
        level_2_state_col (str): Column name for level 2 HMM states.
        session_col (str): Column name for session identifier.

    Returns:
        pd.DataFrame: Updated dataframe with 'Reward_Oriented' column.
    """
    df = df.copy()
    df["Reward_Oriented"] = np.nan

    for sess in df[session_col].unique():
        df_sess = df[df[session_col] == sess]
        state_medians = df_sess.groupby(level_2_state_col)[angle_col].median()
        session_median = df_sess[angle_col].median()

        if len(state_medians) == 2:
            reward_state = state_medians.idxmin()
            non_reward_state = state_medians.idxmax()
            df.loc[(df[session_col] == sess) & (df[level_2_state_col] == reward_state), "Reward_Oriented"] = (
                "Reward Oriented"
            )
            df.loc[(df[session_col] == sess) & (df[level_2_state_col] == non_reward_state), "Reward_Oriented"] = (
                "Non-Reward Oriented"
            )

        elif len(state_medians) == 3:
            sorted_states = state_medians.sort_values()
            reward_state = sorted_states.index[0]
            non_reward_state = sorted_states.index[2]
            middle_state = sorted_states.index[1]

            if state_medians[middle_state] <= session_median:
                middle_label = "Reward Oriented"
            else:
                middle_label = "Non-Reward Oriented"

            df.loc[(df[session_col] == sess) & (df[level_2_state_col] == reward_state), "Reward_Oriented"] = (
                "Reward Oriented"
            )
            df.loc[(df[session_col] == sess) & (df[level_2_state_col] == non_reward_state), "Reward_Oriented"] = (
                "Non-Reward Oriented"
            )
            df.loc[(df[session_col] == sess) & (df[level_2_state_col] == middle_state), "Reward_Oriented"] = (
                middle_label
            )

    return df


def assign_hhmm_state(
    df,
    level_1_state_col,
    level_2_state_col,
):
    """
    Assigns a final HHMM (Hierarchical Hidden Markov Model) state to the dataframe.
    The final HHMM state is based on the combination of level 1 and level 2 states.

    Args:
    - df: DataFrame containing the level_1_state_col and level_2_state_col columns.
    - level_1_state_col: The name of the column representing the first-level HMM state.
    - level_2_state_col: The name of the column representing the second-level state (reward-oriented or not).

    Returns:
    - df: DataFrame with an additional 'HHMM State' column indicating the final HHMM state.
    """

    # Define the conditions for assigning the HHMM states
    conds = [
        (df[level_1_state_col] == 1) & (df[level_2_state_col] == "Non-Reward Oriented"),
        (df[level_1_state_col] == 1) & (df[level_2_state_col] == "Reward Oriented"),
        (df[level_1_state_col] == 2) & (df[level_2_state_col] == "Non-Reward Oriented"),
        (df[level_1_state_col] == 2) & (df[level_2_state_col] == "Reward Oriented"),
    ]

    # Define the labels for the corresponding HHMM states
    labels = [
        "Surveillance, Non-Reward Oriented",
        "Surveillance, Reward Oriented",
        "Ambulatory, Non-Reward Oriented",
        "Ambulatory, Reward Oriented",
    ]

    # Set the default value as 'NaN' (string) to match the data type of the labels
    df["HHMM State"] = np.select(conds, labels, default="NaN")

    return df


################################################################
# HHMM State sequence
################################################################
def plot_hhmm_state_sequence(
    df,
    session_col="Session",
    state_col="HHMM State",
    session_id=None,
    title_prefix="State Sequence",
    colors=None,
):
    """
    Plots a rectangular sequence of HHMM states for a given session.

    Parameters:
        df (pd.DataFrame): DataFrame containing session and HHMM state columns.
        session_col (str): Name of the column indicating session.
        state_col (str): Name of the column containing HHMM state labels.
        session_id (optional): Specific session to plot. If None, plots all sessions.
        title_prefix (str): Custom title prefix for plots.

    Returns:
        None (displays plots)
    """
    sessions_to_plot = [session_id] if session_id is not None else df[session_col].unique()

    for sess in sessions_to_plot:
        test = df.loc[df[session_col] == sess, [state_col]].reset_index(drop=True)
        test["color"] = test[state_col].map(colors)

        fig, ax = plt.subplots(figsize=(10, 3))
        for idx, row in test.iterrows():
            rect = patches.Rectangle((idx, 0), 1, 1, color=row["color"])
            ax.add_patch(rect)

        plt.yticks([])
        plt.xlim(test.index.min(), test.index.max() + 1)
        plt.title(f"{title_prefix} - Session {sess}")
        handles = [patches.Patch(color=color, label=label) for label, color in colors.items()]
        ax.legend(handles=handles, title="States", bbox_to_anchor=(0.5, -0.15), loc="upper center", borderaxespad=0.0)
        plt.tight_layout()

        plt.show()
