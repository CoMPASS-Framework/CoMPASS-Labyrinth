'''
    TRAJECTORY DEVIATION AND MOVEMENT DYNAMICS ACROSS BOUTS
    Author: Shreya Bangera 
    Goal: 
       ├── Deviation from Reward Path metric across bouts
       ├── Velocity across bouts
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
from math import ceil, sqrt


##################################################################
# Plot 7: Deviation from Reward Path & Velocity Analysis
###################################################################

# Ensures velocity column exists
def ensure_velocity_column(df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y', frame_rate: float = 5.0) -> pd.DataFrame:
    """
    Ensure the DataFrame contains a 'Velocity' column. If not, it is computed as the Euclidean distance
    between consecutive (x, y) coordinates multiplied by the frame rate.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column name for x coordinates.
        y_col (str): Column name for y coordinates.
        frame_rate (float): Sampling rate (Hz) to convert framewise displacement into velocity.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Velocity' column.
    """
    if "Velocity" not in df.columns:
        if x_col in df.columns and y_col in df.columns:
            dx = df[x_col].diff()
            dy = df[y_col].diff()
            displacement = np.sqrt(dx**2 + dy**2)
            df["Velocity"] = displacement * frame_rate
        else:
            raise ValueError(f"Missing '{x_col}' or '{y_col}' columns required to compute velocity.")
    return df

# Assign Bout Indices
def assign_bout_indices_from_entry_node(df: pd.DataFrame, delimiter_node: int = 47) -> pd.DataFrame:
    df = df.copy()
    all_sessions = []
    for _, session_data in df.groupby('Session'):
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

# Ensure Bout indices exist
def ensure_bout_indices(df: pd.DataFrame, delimiter_node: int = 47) -> pd.DataFrame:
    if 'Bout_ID' not in df.columns:
        return assign_bout_indices_from_entry_node(df, delimiter_node)
    return df.copy()

# Exponential decay model
def exp_decreasing(x, a, b, c):
    return a * np.exp(-b * x) + c

# Compute deviation and velocity per bout
def compute_deviation_velocity(df, key_regions=['Entry Zone', 'Reward Path', 'Target Zone']):
    sessioncluster = [x for _, x in df.groupby('Session')]
    records = []
    for session_df in sessioncluster:
        ind = 1
        bouts_in_session = [x for _, x in session_df.groupby('Bout_ID')]
        if bouts_in_session:
            bouts_in_session.pop(0)
        for bout in bouts_in_session:
            if len(bout) > 0:
                records.append({
                    'Ind_no': ind,
                    'Session': session_df['Session'].iloc[0],
                    'Genotype': session_df['Genotype'].iloc[0],
                    'Deviation': len(bout.loc[~bout.Region.isin(key_regions), 'Grid Number']) / len(bout),
                    'Velocity': bout['Velocity'].mean()
                })
                ind += 1
    return pd.DataFrame(records)

# Process deviation and velocity (normalize, smooth, fit curves)
def process_deviation_velocity(index_df, genotype):
    df = index_df.dropna(subset=['Deviation', 'Velocity'])
    df = df[df['Genotype'] == genotype].copy()

    # Normalize and smooth
    robust_scaler = RobustScaler()
    df['Velocity_robust_scaled'] = robust_scaler.fit_transform(df[['Velocity']])
    
    qt = QuantileTransformer(output_distribution='uniform')
    df['Velocity_normalized'] = qt.fit_transform(df[['Velocity_robust_scaled']])
    df['Velocity_smooth_normalized'] = gaussian_filter1d(df['Velocity_normalized'], sigma=2)
    df['Deviation_smooth'] = gaussian_filter1d(df['Deviation'], sigma=2)

    # Curve fitting
    x_vals = df['Ind_no'].values
    y_dev = df['Deviation_smooth'].values
    y_vel = df['Velocity_smooth_normalized'].values
    
    params_dev, _ = curve_fit(exp_decreasing, x_vals, y_dev, p0=[1, 0.01, 1], maxfev=10000)
    params_vel, _ = curve_fit(exp_decreasing, x_vals, y_vel, p0=[1, 0.01, 1], maxfev=10000)

    return df, params_dev, params_vel

# Plotting function
def plot_deviation_velocity_fit(df, params_dev, params_vel, genotype,max_bouts=None):
    x_vals = df['Ind_no'].values
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
    y_fit_dev = exp_decreasing(x_fit, *params_dev)
    y_fit_vel = exp_decreasing(x_fit, *params_vel)

    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Ind_no', y='Deviation_smooth', data=df, label='Smoothed Deviation',
                 linewidth=2, alpha=0.8, color='blue')
    sns.lineplot(x='Ind_no', y='Velocity_smooth_normalized', data=df,
                 label='Normalized Smoothed Velocity', linewidth=2, alpha=0.8, color='green')

    plt.plot(x_fit, y_fit_dev, '--', color='red', label='Exponential Fit (Deviation)')
    plt.plot(x_fit, y_fit_vel, '--', color='purple', label='Exponential Fit (Velocity)')

    plt.title(f'Deviation from Reward Path and Velocity across Bouts- {genotype}', fontsize=16)
    plt.xlabel('Bout Number', fontsize=14)
    plt.ylabel('Deviation / Normalized Velocity', fontsize=14)
    plt.ylim(0, 1)
    if max_bouts:
        plt.xlim(0, max_bouts)
    else:
        plt.xlim(0, df['Ind_no'].max() + 5)
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    #plt.show()


##################################################################
# Deviation from Reward Path and Velocity for all Genotypes
##################################################################

def plot_deviation_velocity_all(index_df, max_bouts=None):
    """
    Creates a grid of subplots (auto-arranged) for each genotype showing smoothed
    deviation from reward path and velocity with exponential fits.
    """
    genotypes = index_df['Genotype'].unique()
    n = len(genotypes)
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=False, sharey=True)
    axes = axes.flatten()

    for i, genotype in enumerate(genotypes):
        df, params_dev, params_vel = process_deviation_velocity(index_df, genotype)
        ax = axes[i]

        # Data
        x_vals = df['Ind_no'].values
        y_dev = df['Deviation_smooth'].values
        y_vel = df['Velocity_smooth_normalized'].values
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 1000)
        y_fit_dev = exp_decreasing(x_fit, *params_dev)
        y_fit_vel = exp_decreasing(x_fit, *params_vel)

        # Plot smoothed data
        sns.lineplot(x=x_vals, y=y_dev, ax=ax, label='Smoothed Deviation',
                     color='blue', linewidth=2)
        sns.lineplot(x=x_vals, y=y_vel, ax=ax, label='Smoothed Velocity',
                     color='green', linewidth=2)

        # Plot exponential fits
        ax.plot(x_fit, y_fit_dev, '--', color='red', label='Exp Fit (Deviation)')
        ax.plot(x_fit, y_fit_vel, '--', color='purple', label='Exp Fit (Velocity)')

        ax.set_title(f'{genotype}', fontsize=14, weight='bold')
        ax.set_xlabel('Bout Number', fontsize=12)
        ax.set_ylabel('Deviation / Velocity', fontsize=12)
        if max_bouts:
            ax.set_xlim(0, max_bouts)
        else:
            ax.set_xlim(0, df['Ind_no'].max() + 5)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9, frameon=False)
        ax.grid(True)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Deviation from Reward Path and Velocity per Genotype', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show()
