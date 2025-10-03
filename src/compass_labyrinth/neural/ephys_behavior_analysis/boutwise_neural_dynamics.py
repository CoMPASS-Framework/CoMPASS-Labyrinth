'''
    Boutwise Neural Dynamics
    Author: Shreya Bangera 
    Goal: 
        ├── Normalize specified neural features (e.g., gamma, theta, velocity) to [0, 1] range for fair comparisons.
        ├── Compute bout-level median feature values and classify bouts as Valid and Successful based on behavior and region.
        ├── Visualize and statistically compare neural dynamics (gamma/theta/velocity) across Successful vs Unsuccessful bouts.
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

#####################################################################################
# Neural Dynamics across Bout types (Successful v/s Unsuccessful) Analysis
#####################################################################################

def compute_basic_bout_summary(
    df,
    features,
    condition_col='Region',
    condition_filter='Reward Path',
    target_zone='Target Zone',
    valid_bout_threshold=10
):
    """
    Computes bout-wise median metrics and assigns success/validity labels.

    Parameters:
    - df: full behavioral dataframe
    - features: list of features to summarize (must be normalized beforehand)
    - condition_col: column to apply region filtering (e.g., 'Region')
    - condition_filter: value in condition_col to include (e.g., 'Reward Path')
    - target_zone: region value to determine success (e.g., 'Target Zone')
    - valid_bout_threshold: minimum unique grid nodes for valid bout

    Returns:
    - index_df: summary dataframe per bout
    """
    index_df = pd.DataFrame(columns=['Session', 'Genotype', 'Bout_no'] + features + ['Valid_bout', 'Successful_bout'])
    j = 0
    for _, session_df in df.groupby('Session'):
        bouts = [x for _, x in session_df.groupby('Bout_num') if x['Bout_num'].iloc[0] != 0]
        for bout_no, bout in enumerate(bouts, start=1):
            row = {
                'Session': session_df['Session'].iloc[0],
                'Genotype': session_df['Genotype'].iloc[0],
                'Bout_no': bout_no
            }
            region_subset = bout[bout[condition_col] == condition_filter]
            for feat in features:
                row[feat] = region_subset[feat].median()
            row['Valid_bout'] = 'Valid' if bout['Grid.Number'].nunique() > valid_bout_threshold else 'Invalid'
            row['Successful_bout'] = 'Successful' if target_zone in bout['Region'].values else 'Unsuccessful'
            index_df.loc[j] = row
            j += 1
    return index_df


#######################################################
# Plotting
#######################################################

def plot_bout_metric_comparison(index_df, features):
    """
    Plot median features for Successful vs Unsuccessful bouts, with t-test significance.

    Parameters:
    - index_df: output from compute_basic_bout_summary
    - features: list of features to plot (must be normalized)

    Returns:
    - None (shows boxplot)
    """
    index_df = index_df.copy()
    df_melted = index_df[index_df['Valid_bout'] == 'Valid'].melt(
        id_vars='Successful_bout',
        value_vars=features,
        var_name='Feature',
        value_name='Value'
    )

    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(
        data=df_melted,
        x='Feature',
        y='Value',
        hue='Successful_bout',
        palette=['gray', 'blue'],
        showfliers=False
    )

    y_max = df_melted['Value'].max()
    y_offset = 0.05 * y_max
    for i, feat in enumerate(features):
        succ_vals = df_melted[(df_melted['Successful_bout'] == 'Successful') & (df_melted['Feature'] == feat)]['Value']
        unsucc_vals = df_melted[(df_melted['Successful_bout'] == 'Unsuccessful') & (df_melted['Feature'] == feat)]['Value']
        if len(succ_vals) > 1 and len(unsucc_vals) > 1:
            t_stat, p_val = ttest_ind(succ_vals, unsucc_vals, nan_policy='omit')
            star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            x1, x2 = i - 0.2, i + 0.2
            y_pos = y_max + (i + 1) * y_offset
            ax.plot([x1, x1, x2, x2], [y_pos - y_offset, y_pos, y_pos, y_pos - y_offset], color='black', lw=1.5)
            ax.text((x1 + x2) / 2, y_pos, star, ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Successful vs Unsuccessful Bouts\n(Median Normalized Features)')
    plt.ylabel('Normalized Median Values')
    plt.xlabel('')
    plt.legend(title='Bout Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
