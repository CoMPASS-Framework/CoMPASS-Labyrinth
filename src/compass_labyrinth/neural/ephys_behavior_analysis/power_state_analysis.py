'''
    Power-State Analysis 
    Author: Shreya Bangera 
    Goal: 
        ├── Power across spatial positions and across HHMM States 
        ├── Powers distribution for HHMM States 
        ├── Wasserstein Distance Computation
        ├── Power across States per Region / across Velocity Bins
        ├── Feature KDE per State
        ├── Gamma Trends at Decision Points across HHMM State transition points


'''

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
from scipy.stats import wasserstein_distance
import textwrap
from typing import List, Tuple, Optional
from scipy.stats import gaussian_kde
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
import matplotlib.cm as cm


###########################################################
# Gamma & Theta Power across states
###########################################################

# --------------------- Generalized Function: Prepare Data --------------------- #
def prepare_power_comparison_data(df_power, 
                                  power_cols=['gamma', 'theta'], 
                                  state_col='HMM State', 
                                  group_col='Session'):
    """
    Reshape dataframe for comparison of power bands across HMM or HHMM states.

    Parameters:
    - df_power: pd.DataFrame with state, session, and power columns.
    - power_cols: list of str, power columns to melt (default: ['gamma', 'theta'])
    - state_col: column name for state ('HMM State' or 'HHMM State')
    - group_col: column name for subject/session group (default: 'Session')

    Returns:
    - df_melted: pd.DataFrame reshaped for plotting and analysis
    """
    df_melted = df_power.melt(
        id_vars=[state_col, group_col],
        value_vars=power_cols,
        var_name='Power Type',
        value_name='Power'
    )
    df_melted.rename(columns={state_col: 'State', group_col: 'Group'}, inplace=True)
    return df_melted

# --------------------- Generalized Function: Plot --------------------- #
def plot_power_comparison_by_state(df_melted, 
                                   state_order=None, 
                                   palette=None, figure_size = None,
                                   title="Power Comparison by State"):
    """
    Plot barplot comparing power bands across behavioral states.

    Parameters:
    - df_melted: output from `prepare_power_comparison_data`
    - state_order: list of state names to order on x-axis
    - palette: optional list or dict of colors for each power type
    - title: title of the plot

    Returns:
    - ax: matplotlib axis object
    """
    plt.figure(figsize=figure_size)
    ax = sns.barplot(
        x='State', y='Power', hue='Power Type',
        data=df_melted, ci=95,
        order=state_order,
        palette=palette or ['mediumslateblue', 'palegreen']
    )
    ax.set_xlabel("State")
    ax.set_ylabel("Power")
    ax.set_title(title)
    plt.legend(title="Power Type", loc='upper right')
    #plt.ylim(0, df_melted['Power'].max() + 1)
    plt.grid(True)
    plt.tight_layout()
    return ax

# --------------------- Generalized Function: MixedLM --------------------- #
def run_mixedlm_for_power(df_melted, 
                          power_cols=['gamma', 'theta'], 
                          states_to_compare=['Ambulatory', 'Active Surveillance']):
    """
    Run and describe a mixed linear model (MixedLM) for each power type across selected states.

    Parameters:
    - df_melted: Long-format DataFrame from `prepare_power_comparison_data`.
    - power_cols: List of power types to analyze.
    - states_to_compare: List of state labels to compare.

    Returns:
    - summary_df: pd.DataFrame with model results for each power type.
    """
    results = []

    for power_type in power_cols:
        df_filtered = df_melted[
            (df_melted['Power Type'] == power_type) & 
            (df_melted['State'].isin(states_to_compare))
        ].copy()

        df_filtered['State'] = pd.Categorical(df_filtered['State'], categories=states_to_compare, ordered=True)

        try:
            model = smf.mixedlm("Power ~ C(State)", 
                                data=df_filtered, 
                                groups=df_filtered["Group"])
            result = model.fit()

            print(f"\nMixedLM Result: {power_type.upper()} Power across States")
            print(f"→ Comparing: {states_to_compare[0]} (reference) vs {states_to_compare[1]}")
            print(result.summary())

            for param_name, coef, pval, ci in zip(
                result.params.index,
                result.params.values,
                result.pvalues.values,
                result.conf_int().values
            ):
                results.append({
                    'Power Type': power_type,
                    'Parameter': param_name,
                    'Coef': coef,
                    'P-Value': pval,
                    'CI Lower': ci[0],
                    'CI Upper': ci[1]
                })

                if 'State' in param_name:
                    print(f"  ↳ Effect of '{param_name}':")
                    print(f"     Coefficient = {coef:.3f}, P-value = {pval:.4f}")
                    if pval < 0.05:
                        print("       Significant difference between states.")
                    else:
                        print("       No significant difference.\n")

        except Exception as e:
            print(f"Could not fit model for {power_type}: {e}")
            continue

    return pd.DataFrame(results)


###########################################################
# Wasserstein distance computation
###########################################################

def compute_wasserstein_permutation_test(
    df,
    label_col: str,
    label_1: str,
    label_2: str,
    value_col: str,
    session_col: str = "Session",
    node_filter_col: str = "NodeType",
    node_filter_val: str = "Decision (Reward)",
    n_permutations: int = 1000
):
    observed_distances, p_values, directions, sessions = [], [], [], []

    for sess in df[session_col].unique():
        session_data = df[df[session_col] == sess]

        # Optional filtering
        if node_filter_col and node_filter_val:
            session_data = session_data[session_data[node_filter_col] == node_filter_val]

        group1 = session_data[session_data[label_col] == label_1][value_col]
        group2 = session_data[session_data[label_col] == label_2][value_col]

        if len(group1) == 0 or len(group2) == 0:
            continue

        # Observed Wasserstein distance
        obs_dist = wasserstein_distance(group1, group2)
        observed_distances.append(obs_dist)

        # Directionality
        mean1, mean2 = group1.mean(), group2.mean()
        if mean1 > mean2:
            direction = f"{label_1} > {label_2}"
        elif mean1 < mean2:
            direction = f"{label_2} > {label_1}"
        else:
            direction = "No Shift"
        directions.append(direction)

        # Permutation test
        perm_dists = []
        for _ in range(n_permutations):
            shuffled = session_data.copy()
            shuffled[label_col] = np.random.permutation(shuffled[label_col].values)

            g1 = shuffled[shuffled[label_col] == label_1][value_col]
            g2 = shuffled[shuffled[label_col] == label_2][value_col]

            if len(g1) > 0 and len(g2) > 0:
                perm_dists.append(wasserstein_distance(g1, g2))

        p_val = np.mean(np.array(perm_dists) >= obs_dist)
        p_values.append(p_val)
        sessions.append(sess)

    return pd.DataFrame({
        "Session": sessions,
        "Observed Wasserstein Distance": observed_distances,
        "Permutation p-value": p_values,
        "Directionality": directions
    })


# ----------- Barplot with wrapped labels -----------
def plot_wasserstein_barplot(result_df):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Session", y="Observed Wasserstein Distance", data=result_df, palette="viridis")

    # Annotate with wrapped directionality text
    for i, row in result_df.iterrows():
        direction_wrapped = "\n".join(textwrap.wrap(row["Directionality"], width=18))
        ax.annotate(
            direction_wrapped,
            (i, row["Observed Wasserstein Distance"] + 0.001),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    plt.title("Wasserstein Distance per Session\n(Decision Nodes - Reward Path)", fontsize=14, weight="bold")
    plt.ylim(0,2)
    plt.xlabel("Session", fontsize=12)
    plt.ylabel("Wasserstein Distance", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------- Formatted Table Plot -----------
def plot_wasserstein_table(result_df):
    fig, ax = plt.subplots(figsize=(12, len(result_df) * 0.4 + 3))  # dynamic height
    ax.axis("off")

    table_data = result_df[["Session", "Observed Wasserstein Distance", "Formatted p-value", "Directionality"]]
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Optional: Wrap long directionality strings
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight='bold')
        if key[1] == 3 and key[0] > 0:  # Column 3 = Directionality
            val = cell.get_text().get_text()
            wrapped = "\n".join(textwrap.wrap(val, width=25))
            cell.get_text().set_text(wrapped)

    plt.title("Wasserstein Distance and Directionality Results\n(Decision Nodes - Reward Path)", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def plot_gamma_distribution_kde(
    df,
    label_col="HHMM State",
    label_1="Active Surveillance, Reward-oriented",
    label_2="Active Surveillance, Non-reward oriented",
    value_col="gamma",
    filter_node_col="NodeType",
    filter_node_val="Decision (Reward)"
):
    """
    Plot KDE distributions of a value (e.g., gamma) for two specified HHMM state labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with gamma and HHMM state labels.
    label_col : str
        Column name representing the HHMM state labels.
    label_1 : str
        First HHMM state label to compare (e.g., Reward-oriented).
    label_2 : str
        Second HHMM state label to compare (e.g., Non-reward oriented).
    value_col : str
        Column representing the value to compare (e.g., 'gamma').
    filter_node_col : str
        Column to filter for specific node type (e.g., 'NodeType').
    filter_node_val : str
        Value in filter_node_col to retain rows (e.g., 'Decision (Reward)').
    """
    df_filtered = df.copy()
    if filter_node_col in df.columns and filter_node_val is not None:
        df_filtered = df_filtered[df_filtered[filter_node_col] == filter_node_val]

    data_1 = df_filtered[df_filtered[label_col] == label_1][value_col]
    data_2 = df_filtered[df_filtered[label_col] == label_2][value_col]

    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=data_1, fill=True, label=label_1, color='blue')
    sns.kdeplot(data=data_2, fill=True, label=label_2, color='grey')

    plt.title(f'{value_col.title()} Distribution: \n{label_1} vs {label_2} ({filter_node_val})')
    plt.xlabel(value_col.title())
    plt.ylabel('Density')
    plt.legend(title='HHMM State')
    plt.tight_layout()
    plt.show()



###########################################################
# Power/Velocity by Node Type across Level 1 States
###########################################################

# ------------------------------------------
# Function 1: Filter and prepare the data
# ------------------------------------------
def prepare_node_comparison_data(df, y_col='gamma', 
                                  node_types=['Decision (Reward)', 'Non-Decision (Reward)'], 
                                  required_cols=['Session', 'Region', 'NodeType', 'HMM_State']):
    """
    Filters and prepares data for plotting and modeling power/velocity across node types.
    """
    use_cols = required_cols + [y_col]
    df_filtered = df[use_cols].copy()
    df_filtered = df_filtered[df_filtered['NodeType'].isin(node_types)]
    return df_filtered

# ------------------------------------------
# Function 2: Barplot across node types
# ------------------------------------------
def plot_power_by_nodetype(df, y_col='gamma', 
                           node_order=['Decision (Reward)', 'Non-Decision (Reward)'],
                           figsize=(6, 5), palette=None, title=None):
    """
    Creates a grouped barplot of the selected y_col (e.g., gamma, theta, velocity) 
    across node types, split by HMM state.
    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x='NodeType',
        y=y_col,
        data=df,
        hue='HMM State',
        palette=palette or sns.color_palette("Set2"),
        errorbar='se',
        order=node_order
    )
    plt.legend(title="HMM State", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Node Type', fontsize=12)
    plt.ylabel(y_col.capitalize(), fontsize=12)
    plt.title(title or f'{y_col.capitalize()} Across Node Types', fontsize=14, weight='bold')
    plt.grid(True)
    plt.tight_layout()
    return ax

# ------------------------------------------
# Function 3: Mixed Linear Model (optional)
# ------------------------------------------
def run_mixedlm_node_type_by_hmmstate(df, 
                                      y_col='gamma', 
                                      node_types=['Decision (Reward)', 'Non-Decision (Reward)']):
    """
    Fit a mixed linear model to compare power/velocity across node types separately per HMM state,
    controlling for session as a random effect.

    Parameters:
    - df: input DataFrame
    - y_col: column to model (e.g., 'gamma', 'theta', 'Velocity')
    - node_types: list of node types to compare (must contain at least two)

    Returns:
    - results_dict: dictionary of fitted models per HMM state
    """
    results_dict = {}
    hmm_states = df['HMM State'].dropna().unique()

    for hmm in hmm_states:
        df_hmm = df[df['HMM State'] == hmm].copy()
        df_hmm = df_hmm[df_hmm['NodeType'].isin(node_types)]
        df_hmm['NodeType'] = pd.Categorical(df_hmm['NodeType'], categories=node_types, ordered=True)

        if df_hmm['NodeType'].nunique() < 2:
            print(f"\nSkipping HMM State {hmm}: Not enough NodeType variation.")
            continue

        try:
            formula = f"{y_col} ~ C(NodeType)"
            model = smf.mixedlm(formula, data=df_hmm, groups=df_hmm["Session"])
            result = model.fit()

            print(f"\nMixedLM Result: {y_col.upper()} across Node Types (HMM State = {hmm})")
            print(f"→ Comparing: {node_types[0]} (reference) vs {node_types[1]}")
            print(result.summary())

            results_dict[hmm] = result

        except Exception as e:
            print(f"Could not fit model for HMM State {hmm}: {e}")

    return results_dict


#######################################################
# Velocity column creation
#######################################################

def ensure_velocity_column(df, x_col='x', y_col='y', velocity_col='Velocity'):
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
    if 'Session' in df.columns:
        df[velocity_col] = (
            df.groupby('Session', group_keys=False)
              .apply(lambda g: np.sqrt(g[x_col].diff()**2 + g[y_col].diff()**2).fillna(0))
        )
    else:
        df[velocity_col] = np.sqrt(df[x_col].diff()**2 + df[y_col].diff()**2).fillna(0)

    return df


#######################################################
# Create Velocity Bins Column
#######################################################

def create_velocity_bins(df, col_of_interest='Velocity', num_bins=5):
    """
    Assign quantile-based velocity bins per session using min-max scaling.

    Handles sessions with flat or low-variance velocity values by assigning a single bin.
    """
    def process_session(session_df):
        v = session_df[col_of_interest]
        vmin, vmax = v.min(), v.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
            scaled = pd.Series([0.5] * len(v), index=v.index)
        else:
            scaled = (v - vmin) / (vmax - vmin)
        session_df['minmax'] = scaled

        try:
            qcut_bins, edges = pd.qcut(scaled, q=num_bins, retbins=True, duplicates='drop')
            actual_bins = len(edges) - 1
            labels = [f'Bin{i+1}' for i in range(actual_bins)]
            session_df['Velocity Bins'] = pd.qcut(scaled, q=actual_bins, labels=labels, duplicates='drop')
        except ValueError:
            session_df['Velocity Bins'] = 'Bin1'
        return session_df

    df_out = df.groupby('Session', group_keys=False).apply(process_session)

    # Ensure consistent category ordering
    if df_out['Velocity Bins'].dtype.name != 'category':
        df_out['Velocity Bins'] = pd.Categorical(df_out['Velocity Bins'])

    df_out['Velocity Bins'] = df_out['Velocity Bins'].cat.reorder_categories(
        sorted(df_out['Velocity Bins'].cat.categories, key=lambda s: int(str(s).replace("Bin", ""))),
        ordered=True
    )
    return df_out



#######################################################
# Power across Level 2 States per Velocity Bin
#######################################################

def run_lmm_comparisons(df, comparisons, metric, state_col):
    """
    Run LMMs comparing state_col groups within each velocity bin.

    Returns a stats table with effect sizes and p-values per bin.
    """
    results = []

    for region, group1, group2 in comparisons:
        mask = (df['Velocity Bins'] == region) & (df[state_col].isin([group1, group2]))
        df_bin = df.loc[mask, ['Session', state_col, metric]].copy()

        g_counts = df_bin[state_col].value_counts()
        if g_counts.get(group1, 0) > 1 and g_counts.get(group2, 0) > 1:
            df_bin = df_bin.rename(columns={state_col: 'Group', metric: 'Metric'})
            df_bin['Region'] = region

            try:
                model = smf.mixedlm("Metric ~ C(Group)", df_bin, groups=df_bin["Session"])
                result = model.fit()
                coef_name = next((c for c in result.pvalues.index if 'C(Group)' in c), None)
                est = result.params.get(coef_name, np.nan)
                p_val = result.pvalues.get(coef_name, np.nan)
                signif = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            except Exception as e:
                est, p_val, signif = np.nan, np.nan, f'fit error: {e}'
        else:
            est, p_val, signif = np.nan, np.nan, 'insufficient data'

        results.append({
            'Velocity Bin': region,
            'Comparison': f"{group1} vs {group2}",
            'Effect Size': round(est, 4) if pd.notna(est) else np.nan,
            'P-Value': round(p_val, 4) if pd.notna(p_val) else np.nan,
            'Significance': signif
        })

    return pd.DataFrame(results)


def analyze_velocity_bin_comparisons(df,
                                     metric='Velocity',
                                     num_bins=5,
                                     state_col='HHMM State',
                                     node_filter='Decision (Reward)',
                                     custom_comparisons=None):
    """
    Wrapper to assign velocity bins, filter by node type, and run LMM comparisons.
    """
    df_binned = create_velocity_bins(df, col_of_interest=metric, num_bins=num_bins)
    df_binned = df_binned[df_binned['NodeType'] == node_filter].copy()
    stats_df = run_lmm_comparisons(df_binned, custom_comparisons, metric=metric, state_col=state_col)
    return stats_df

def plot_velocity_bin_barplot(df, 
                              metric='Velocity', 
                              state_col='HHMM State', 
                              node_filter='Decision (Reward)', 
                              y_max=300,
                              palette=('maroon', 'navy', 'lightblue', 'coral')):
    """
    Grouped barplot of metric by Velocity Bins and HHMM state.
    """
    dfp = df[df['NodeType'] == node_filter].copy()
    if 'Velocity Bins' not in dfp.columns:
        raise KeyError("Column 'Velocity Bins' not found. Run create_velocity_bins() first.")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        x='Velocity Bins', y=metric, hue=state_col, data=dfp,
        palette=list(palette), ax=ax, errorbar='ci', errwidth=1.5, capsize=0.05
    )

    ax.set_xlabel('Velocity Bin')
    ax.set_ylabel(metric)
    ax.set_title(f'{node_filter} - Velocity Bin Comparison')
    ax.set_ylim(0, y_max + 100)
    ax.legend(title=state_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax


#######################################################
# Feature KDE per State
#######################################################

def plot_metric_kde_by_state(df,
                              state_col='HHMM State',
                              metric='Velocity',
                              groups=None,
                              palette=None,
                              density_thresh=0.001,
                              use_dynamic_cutoff=True,
                              figsize=(15, 8)):
    """
    Plot KDE curves of a specified metric across HHMM State categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the metric and state column.
    state_col : str
        Column with group/category labels (e.g., 'HHMM State').
    metric : str
        Name of the metric to plot (e.g., 'Velocity').
    groups : list of str, optional
        Specific HHMM states to include. If None, uses all available.
    palette : dict or list, optional
        Color mapping for each group. If None, default seaborn palette is used.
    density_thresh : float
        Threshold below which density is considered negligible (used if use_dynamic_cutoff=True).
    use_dynamic_cutoff : bool
        Whether to dynamically adjust x-axis limit where all curves fall below density_thresh.
    figsize : tuple
        Size of the figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object of the plot.
    """
    if groups is None:
        groups = df[state_col].dropna().unique()

    if palette is None:
        palette = sns.color_palette("Set2", len(groups))

    if isinstance(palette, list):
        palette = dict(zip(groups, palette))

    x_grid = np.linspace(df[metric].min(), df[metric].max(), 1000)
    kde_curves = {}

    # Compute KDEs
    for group in groups:
        subset = df[df[state_col] == group][metric].dropna()
        if len(subset) > 1:
            kde = gaussian_kde(subset)
            kde_vals = kde(x_grid)
            kde_curves[group] = kde_vals

    # Determine dynamic cutoff point (optional)
    max_x_cutoff = df[metric].max()
    if use_dynamic_cutoff and kde_curves:
        all_kde_below_thresh = np.all(np.array(list(kde_curves.values())) < density_thresh, axis=0)
        below_indices = np.where(all_kde_below_thresh)[0]
        if len(below_indices) > 0:
            max_x_cutoff = x_grid[below_indices[0]]

    # Plot
    plt.figure(figsize=figsize)
    for group in groups:
        subset = df[df[state_col] == group][metric].dropna()
        sns.kdeplot(data=subset, shade=True,
                    color=palette.get(group, None), label=group)

    plt.xlim(df[metric].min(), max_x_cutoff)
    plt.xlabel(metric)
    plt.title(f"{metric} KDE by {state_col}")
    plt.legend()
    plt.tight_layout()
    return plt.gca()

#######################################################
# Normalize Features
#######################################################

def normalize_columns(df, features):
    """
    Normalize specified feature columns to [0, 1] range.

    Parameters:
    - df: DataFrame
    - features: list of column names to normalize

    Returns:
    - DataFrame with normalized columns
    """
    df = df.copy()
    for feat in features:
        min_val = df[feat].min()
        max_val = df[feat].max()
        if max_val > min_val:
            df[feat] = (df[feat] - min_val) / (max_val - min_val)
    return df


#######################################################################
# Gamma Trends at Decision Points across HHMM State transition points
#######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem


def zscore_column(df, col):
    """Z-score a column."""
    return (df[col] - df[col].mean()) / df[col].std()


def extract_aligned_windows(df, state_col, target_states, node_col, node_filter, metric, window_size):
    """
    Extract metric windows aligned before and after state onset at decision nodes.
    """
    before_all, after_all = [], []

    for sess in df['Session'].unique():
        session_data = df[df['Session'] == sess].reset_index(drop=True)

        if metric not in session_data.columns:
            continue

        session_data[metric] = zscore_column(session_data, metric)

        # Find state transitions at nodes
        state_idx = session_data[
            (session_data[state_col].isin(target_states)) &
            (session_data[node_col].isin(node_filter))
        ].index

        for idx in state_idx:
            before = session_data.iloc[max(0, idx - window_size):idx][metric].values
            after = session_data.iloc[idx:min(idx + window_size, len(session_data))][metric].values

            if len(before) == window_size and len(after) == window_size:
                before_all.append(before)
                after_all.append(after)

    return np.array(before_all), np.array(after_all)


def build_trend_dataframe(before_array, after_array, window_size, metric_label='Gamma Power (Z-scored)'):
    """
    Build a dataframe with aligned mean + SEM for plotting.
    """
    timepoints = np.linspace(-1, 1, 2 * window_size)
    mean_values = np.concatenate([before_array.mean(axis=0), after_array.mean(axis=0)])
    sem_values = np.concatenate([sem(before_array, axis=0), sem(after_array, axis=0)])

    return pd.DataFrame({
        'Time': timepoints,
        metric_label: mean_values,
        'SEM': sem_values
    })


def plot_trend(plot_df, metric_label='Gamma Power (Z-scored)', title='', y_label=None):
    """
    Plot aligned mean ± SEM of the metric.
    """
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=plot_df, x='Time', y=metric_label, linewidth=2.5, label=f'Mean {metric_label}')

    plt.fill_between(
        plot_df['Time'],
        plot_df[metric_label] - plot_df['SEM'],
        plot_df[metric_label] + plot_df['SEM'],
        alpha=0.25,
        color='C0',
        label='± SEM'
    )

    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Normalized Time', fontsize=14)
    plt.ylabel(y_label or metric_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()


def aggregate_metric_trend(df,
                           metric='gamma',
                           window_size=25,
                           states=['Ambulatory, Reward-oriented', 'Active Surveillance, Reward-oriented'],
                           node_filter=None,
                           node_col='Grid.Number',
                           state_col='HHMM State'):
    """
    Extract and plot aligned metric trends around specific HHMM states at decision nodes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least 'Session', metric, state, and node columns.
    metric : str
        Column name for the metric (e.g., 'gamma').
    window_size : int
        Number of frames before and after to extract.
    states : list of str
        HHMM states to align on.
    node_filter : list of int
        Grid.Number values defining node subset.
    node_col : str
        Name of the node/grid column.
    state_col : str
        Column containing state assignments.
    """
    if node_filter is None:
        raise ValueError("Please provide a list of decision nodes for `node_filter`.")

    before, after = extract_aligned_windows(
        df, state_col=state_col, target_states=states,
        node_col=node_col, node_filter=node_filter,
        metric=metric, window_size=window_size
    )

    if len(before) == 0 or len(after) == 0:
        print(" Warning: No valid transitions found for the given filters.")
        return

    plot_df = build_trend_dataframe(
        before, after, window_size,
        metric_label=f"{metric.capitalize()} Power (Z-scored)"
    )

    plot_trend(
        plot_df,
        metric_label=f"{metric.capitalize()} Power (Z-scored)",
        title='Trend Around Reward-Oriented States\nAt Decision Nodes',
        y_label=f'Z-scored {metric.capitalize()} Power'
    )
