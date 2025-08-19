'''
    BOUT-TYPE BASED STATE COMPARISONS    
    Author: Shreya Bangera 
    Goal: 
        ├── Classifies bouts as successful or unsuccessful based on target reach.
        ├── Computes and compares HMM state proportions across these bout types.

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from shapely.geometry import Point, LineString
from scipy.interpolate import make_interp_spline
from collections import Counter
from plotly.colors import sample_colorscale
from itertools import combinations
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

##################################################################
# Plot 4: Surveillance Probability by Bout Type
###################################################################

def assign_bout_indices(df: pd.DataFrame, delimiter_node: int = 47) -> pd.DataFrame:
    df = df.copy()
    updated = []

    for _, sess_df in df.groupby('Session'):
        sess_df = sess_df.reset_index(drop=True)
        bout_id = 1
        bout_indices = []

        for _, row in sess_df.iterrows():
            if row['Grid.Number'] == delimiter_node:
                bout_id += 1
            bout_indices.append(bout_id)

        sess_df['Bout_Index'] = bout_indices
        updated.append(sess_df)

    return pd.concat(updated, ignore_index=True)


def compute_surveillance_probabilities(df_hmm, decision_nodes):
    records = []

    for session_id, sess_df in df_hmm.groupby('Session'):
        genotype = sess_df['Genotype'].unique()[0]
        bouts = list(sess_df.groupby('Bout_Index'))[1:]  # skip incomplete first bout

        for bout_num, (_, bout_df) in enumerate(bouts, 1):
            success = 'Successful' if 'Target Zone' in bout_df['Region'].values else 'Unsuccessful'
            state_probs = bout_df[bout_df['Grid.Number'].isin(decision_nodes)]['HMM_State'].value_counts(normalize=True)
            prob_state_1 = state_probs.get(1, np.nan)

            records.append({
                'Session': session_id,
                'Genotype': genotype,
                'Bout_no': bout_num,
                'Successful_bout': success,
                'Probability_1': prob_state_1
            })

    return pd.DataFrame(records)


def plot_surveillance_by_bout(index_df, ylim, figure_size=(8, 3), palette=['grey']):
    plt.figure(figsize=figure_size)
    
    genotypes = sorted(index_df['Genotype'].unique())
    
    ax = sns.barplot(
        x='Successful_bout',
        y='Probability_1',
        hue='Genotype',
        data=index_df,
        errorbar='se',
        capsize=0.1,
        errwidth=1.6,
        palette=palette,
        edgecolor='black'
    )

    plt.xlabel('Bout Type', fontsize=12)
    plt.ylabel('Surveillance Probability (State 1)', fontsize=12)
    plt.title('Surveillance Probability across Bout Types', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, ylim)
    plt.legend(title='Genotype', frameon=True, loc='upper right')
    plt.tight_layout()
    plt.show()

def test_within_genotype_success(index_df):
    """
    T-tests between Successful vs Unsuccessful bouts for each genotype.
    """
    results = []

    for genotype in sorted(index_df['Genotype'].unique()):
        df = index_df[index_df['Genotype'] == genotype]
        s = df[df['Successful_bout'] == 'Successful']['Probability_1'].dropna()
        u = df[df['Successful_bout'] == 'Unsuccessful']['Probability_1'].dropna()

        if len(s) >= 2 and len(u) >= 2:
            stat, pval = ttest_ind(s, u, equal_var=False)
            results.append({
                'Genotype': genotype,
                'Group 1': 'Successful',
                'Group 2': 'Unsuccessful',
                'T-stat': stat,
                'P-value': pval
            })

    return pd.DataFrame(results)

def test_across_genotypes_per_bout(index_df, bout_type='Successful'):
    """
    T-tests between genotypes for either Successful or Unsuccessful bouts.
    """
    results = []

    df = index_df[index_df['Successful_bout'] == bout_type]
    genotypes = sorted(df['Genotype'].unique())

    for g1, g2 in combinations(genotypes, 2):
        vals1 = df[df['Genotype'] == g1]['Probability_1'].dropna()
        vals2 = df[df['Genotype'] == g2]['Probability_1'].dropna()

        if len(vals1) >= 2 and len(vals2) >= 2:
            stat, pval = ttest_ind(vals1, vals2, equal_var=False)
            results.append({
                'Bout Type': bout_type,
                'Genotype 1': g1,
                'Genotype 2': g2,
                'T-stat': stat,
                'P-value': pval
            })

    return pd.DataFrame(results)



def run_within_genotype_mixedlm_with_fdr(index_df):
    """
    Run a mixed-effects model per genotype comparing Successful vs Unsuccessful bouts,
    with Session as a random effect. Applies FDR correction.

    Returns:
    - DataFrame with Genotype, Effect size, raw P-value, FDR P-value, and significance flag.
    """
    results = []
    genotypes = index_df['Genotype'].unique()

    for genotype in genotypes:
        df_sub = index_df[index_df['Genotype'] == genotype].copy()

        # Ensure sufficient sessions
        if df_sub['Session'].nunique() < 2:
            continue

        # Ensure both bout types are present
        bout_counts = df_sub['Successful_bout'].value_counts()
        if not all(x in bout_counts.index for x in ['Successful', 'Unsuccessful']):
            continue

        try:
            # Proper data typing
            df_sub['Successful_bout'] = pd.Categorical(df_sub['Successful_bout'], categories=['Unsuccessful', 'Successful'])
            df_sub['Session'] = df_sub['Session'].astype(str)

            # Fit model
            model = mixedlm("Probability_1 ~ Successful_bout", data=df_sub, groups=df_sub["Session"])
            result = model.fit()

            # Extract stats
            term_name = next((k for k in result.params.keys() if 'Successful_bout' in k), None)
            coef = result.params.get(term_name, np.nan)
            pval = result.pvalues.get(term_name, np.nan)

        except Exception as e:
            coef = np.nan
            pval = np.nan

        results.append({
            "Genotype": genotype,
            "Effect: Successful vs Unsuccessful": coef,
            "P-value": pval
        })

    result_df = pd.DataFrame(results)

    # Apply FDR correction if valid
    if not result_df.empty and result_df['P-value'].notna().sum() > 0:
        reject, pvals_corrected, _, _ = multipletests(result_df['P-value'], method='fdr_bh')
        result_df['FDR P-value'] = pvals_corrected
        result_df['Significant (FDR < 0.05)'] = reject
    else:
        result_df['FDR P-value'] = np.nan
        result_df['Significant (FDR < 0.05)'] = False

    return result_df
