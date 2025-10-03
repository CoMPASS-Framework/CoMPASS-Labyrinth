'''
    SUCCESSFUL BOUT ANALYSIS
    Author: Shreya Bangera 
    Goal: 
        ├── Cumulative successful bout analysis
        ├── Time-based successful bout analysis
'''
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import entropy
import logging
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

##################################################################
## Plot 5: Cumulative Successful Bout Percentage
###################################################################

def assign_bout_indices_from_entry_node(navigation_df: pd.DataFrame, delimiter_node: int = 47) -> pd.DataFrame:
    """
    Assigns bout indices to each row of a session based on entries to a delimiter node (e.g., Entry node = 47).
    A new bout starts every time the delimiter node is encountered.

    Parameters:
    - navigation_df: DataFrame with 'Session' and 'Grid Number' columns.
    - delimiter_node: Grid number that marks the entry point for bouts.

    Returns:
    - DataFrame with an added 'Bout_ID' column.
    """
    all_sessions = []

    for _, session_data in navigation_df.groupby('Session'):
        session_data = session_data.reset_index(drop=True).copy()
        session_data['Bout_ID'] = 0
        bout_counter = 1

        for row_idx in range(len(session_data)):
            if session_data.loc[row_idx, 'Grid Number'] != delimiter_node:
                session_data.loc[row_idx, 'Bout_ID'] = bout_counter
            else:
                session_data.loc[row_idx, 'Bout_ID'] = 0
                bout_counter += 1

        all_sessions.append(session_data)

    return pd.concat(all_sessions, ignore_index=True)


def summarize_bout_success_by_session(navigation_df: pd.DataFrame,
                                      optimal_regions: list = ['Entry Zone', 'Reward Path', 'Target Zone'],
                                      target_region_label: list = ['Target Zone'],
                                      min_bout_length: int = 20) -> pd.DataFrame:
    """
    Computes number of total, valid, successful, and perfect bouts per session.

    Parameters:
    - navigation_df: DataFrame with 'Session', 'Genotype', 'Region', and 'Bout_Index'.
    - optimal_regions: Ordered list of region labels that define a perfect bout.
    - target_region_label: Region considered as successful bout completion.
    - min_bout_length: Minimum length of frames required to count a bout as valid.

    Returns:
    - summary_table: DataFrame summarizing bout stats by session.
    """
    summary_records = []

    for session_id, session_data in navigation_df.groupby('Session'):
        genotype = session_data['Genotype'].iloc[0]
        session_bouts = [b for _, b in session_data.groupby('Bout_ID') if b['Bout_ID'].iloc[0] != 0]

        valid_bouts = [b for b in session_bouts if len(b) > min_bout_length]
        successful_bouts = [b for b in valid_bouts if any(r in target_region_label for r in b['Region'])]
        perfect_bouts = [b for b in successful_bouts if set(optimal_regions) == set(b['Region'].unique())]

        summary_records.append({
            'Session': session_id,
            'Genotype': genotype,
            'Total_Bouts': len(session_bouts),
            'Valid_Bouts': len(valid_bouts),
            'Successful_Bouts': len(successful_bouts),
            'Perfect_Bouts': len(perfect_bouts)
        })

    summary_table = pd.DataFrame(summary_records)
    summary_table = summary_table[summary_table['Total_Bouts'] != 0]

    # Derived percentages
    summary_table['Success_Rate'] = (100 * summary_table['Successful_Bouts']) / summary_table['Valid_Bouts']
    summary_table['Perfect_Rate'] = (
        100 * summary_table['Perfect_Bouts'] / summary_table['Successful_Bouts'].replace(0, np.nan)
    )

    return summary_table


def plot_success_rate(summary_table: pd.DataFrame, palette: list = None) -> None:
    """
    Plots a barplot showing success rates across genotypes from the bout summary.

    Parameters:
    - summary_table: Output from summarize_bout_success_by_session.
    - palette: Optional color palette for different genotypes.
    """
    plt.figure(figsize=(4.5, 5))

    ax = sns.barplot(
        x='Genotype',
        y='Success_Rate',
        data=summary_table,
        errorbar='se',
        width=0.7,
        err_kws={'color': 'black', 'linewidth': 1.5},
        capsize=0.15,
        edgecolor='black',
        palette=palette if palette else 'deep'
    )

    sns.stripplot(
        x='Genotype',
        y='Success_Rate',
        data=summary_table,
        dodge=True,
        color='black',
        size=4
    )

    ax.set_title('Percentage of Successful Bouts by Genotype', fontsize=15)
    ax.set_xlabel('Genotype', fontsize=13)
    ax.set_ylabel('% of Successful Bouts', fontsize=13)
    ax.set(ylim=(0, 100))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=2.5, color='black')
    plt.xticks(size=12, color='black')
    plt.yticks(size=12, color='black')

    plt.tight_layout()
    #plt.show()

#-------------------- T-TEST ON SUCCESS RATE PER GENOTYPE PAIR ---------------------#

def perform_genotype_ttests(summary_table: pd.DataFrame, rate_col: str = 'Success_Rate'):
    """
    Performs t-tests between genotypes on a given rate column (e.g., Success_Rate or Perfect_Rate).

    Parameters:
    - summary_table: DataFrame from `summarize_bout_success_by_session`
    - rate_col: Column to compare (default 'Success_Rate')

    Returns:
    - Dictionary with t-test results between each genotype pair.
    """
    from itertools import combinations
    results = {}

    # Unique genotype pairs
    genotypes = summary_table['Genotype'].unique()
    for g1, g2 in combinations(genotypes, 2):
        data1 = summary_table[summary_table['Genotype'] == g1][rate_col].dropna()
        data2 = summary_table[summary_table['Genotype'] == g2][rate_col].dropna()
        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)

        results[f"{g1} vs {g2}"] = {
            "t_stat": t_stat,
            "p_value": p_val,
            "mean_1": data1.mean(),
            "mean_2": data2.mean(),
            "n_1": len(data1),
            "n_2": len(data2)
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
    optimal_path_regions: list[str] = ['Entry Zone', 'Reward Path', 'Target Zone'],
    target_zone: str = 'Target Zone'
) -> pd.DataFrame:
    """
    Computes successful bout metrics per session, binned by cumulative frame index.
    """
    summary_records = []
    session_clusters = [x for _, x in df_all_csv.groupby('Session')]

    for session_subset in session_clusters:
        for k in range(lower_succ_lim, upper_succ_lim, diff_succ):
            sess_sub = session_subset[k:k + diff_succ]
            bouts_in_session = [x for _, x in sess_sub.groupby('Bout_ID')]

            sum_succ, sum_perfect, sum_valid_bouts = 0, 0, 0
            li_length_bouts, journey_length = [], []

            for bout in bouts_in_session:
                if len(bout) > valid_bout_threshold:
                    sum_valid_bouts += 1
                    if any(e in target_zone for e in bout['Region'].to_list()):
                        sum_succ += 1
                        li_length_bouts.append(len(bout['Region']))
                        journey_length.append(len(bout))
                        if set(bout['Region'].unique()) == set(optimal_path_regions):
                            sum_perfect += 1

            summary_records.append({
                'Session': session_subset.Session.unique()[0],
                'Genotype': session_subset['Genotype'].unique()[0],
                'Bout_num': k + diff_succ,
                'No_of_Bouts': len(bouts_in_session),
                'No_Valid_bouts': sum_valid_bouts,
                'No_of_Succ_Bouts': sum_succ,
                'No_of_perfect_bouts': sum_perfect
            })

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df[summary_df['No_of_Bouts'] != 0]
    summary_df['Succ_bout_perc'] = (100 * summary_df['No_of_Succ_Bouts']) / summary_df['No_Valid_bouts']
    return summary_df


def plot_binned_success(summary_df: pd.DataFrame, palette: list[str]) -> None:
    """
    Plots % of successful bouts over time across genotypes.
    """
    sns.set_style('white')
    sns.set_style('ticks')

    summary_df['Bout_num'] = pd.Categorical(summary_df['Bout_num'])
    summary_df['Genotype'] = pd.Categorical(summary_df['Genotype'])
    summary_df['Succ_bout_perc'] = pd.to_numeric(summary_df['Succ_bout_perc'])

    ax = sns.catplot(
        x="Bout_num",
        y="Succ_bout_perc",
        hue='Genotype',
        data=summary_df,
        errorbar='se',
        kind="point",
        capsize=.15,
        aspect=1.9,
        palette=palette
    )
    plt.ylim(0, 110)
    plt.xticks(rotation=45)
    plt.xlabel('Time in maze')
    plt.title('Successful Bout % over time across genotypes')
    plt.ylabel('% of Successful Bouts')
    #plt.show()


# ----------------- Mixed Effects Model (with NaNs kept) ----------------- #
def run_mixedlm_with_nans(summary_df: pd.DataFrame):
    print("\nRunning MixedLM with NaNs preserved...")

    model_df = summary_df.copy()
    model_df = model_df.dropna(subset=['Succ_bout_perc'])

    model = mixedlm("Succ_bout_perc ~ C(Bout_num) * C(Genotype)", 
                    model_df, 
                    groups=model_df["Session"])
    result = model.fit()
    print(result.summary())

# ----------------- Repeated Measures ANOVA (after fillna) ----------------- #
def run_repeated_measures_anova(summary_df: pd.DataFrame):
    print("\nRunning Repeated Measures ANOVA (NaNs filled with 0)...")

    anova_df = summary_df.copy()
    anova_df['Succ_bout_perc'] = anova_df['Succ_bout_perc'].fillna(0)

    try:
        aovrm = AnovaRM(
            anova_df, 
            depvar='Succ_bout_perc', 
            subject='Session', 
            within=['Bout_num'],
            between=['Genotype']
        )
        anova_res = aovrm.fit()
        print(anova_res)
    except Exception as e:
        print(f"ANOVA failed: {e}")

# ----------------- Pairwise Multiple Comparisons (Tukey + FDR) ----------------- #
def run_pairwise_comparisons(summary_df: pd.DataFrame):
    print("\nRunning Pairwise Comparisons with Tukey HSD + FDR...")

    tukey_df = summary_df.copy()
    tukey_df['Succ_bout_perc'] = tukey_df['Succ_bout_perc'].fillna(0)

    results = []
    for bout in tukey_df['Bout_num'].unique():
        sub = tukey_df[tukey_df['Bout_num'] == bout]
        if sub['Genotype'].nunique() > 1:
            tukey = pairwise_tukeyhsd(endog=sub['Succ_bout_perc'],
                                      groups=sub['Genotype'],
                                      alpha=0.05)
            df_tukey = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            df_tukey['Bout_num'] = bout
            results.append(df_tukey)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        # Apply FDR correction
        reject, pvals_corrected, _, _ = multipletests(all_results['p-adj'], method='fdr_bh')
        all_results['FDR_p'] = pvals_corrected
        all_results['Significant'] = reject
        print(all_results)
    else:
        print("No pairwise comparisons could be performed.")




