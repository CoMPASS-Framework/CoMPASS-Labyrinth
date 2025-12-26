"""
UMAPS
Author: Shreya Bangera
Goal:
    ├── UMAP: Reveal structure in neural features across latent behavioral states
    ├── UMAPS - Temporal Progression: Capture how neural dynamics evolve over time around decision points

"""

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
from matplotlib.cm import ScalarMappable


#######################################################
# UMAPS
#######################################################


def prepare_sliding_window_data(
    df, window_size=20, feature_col="gamma", state_col="HHMM State", grid_col="Grid Number", node_list=None
):
    gamma_windows = []
    state_labels = []

    encoder = OneHotEncoder(sparse_output=False)
    state_encoded = encoder.fit_transform(df[[state_col]])
    middle_offset = window_size // 2

    for i in range(len(df) - window_size):
        middle_idx = i + middle_offset

        if node_list and df[grid_col].iloc[middle_idx] not in node_list:
            continue

        gamma_segment = df[feature_col].values[i : i + window_size]
        state_segment = state_encoded[middle_idx]
        combined = np.concatenate([gamma_segment, state_segment])
        gamma_windows.append(combined)
        state_labels.append(df[state_col].iloc[middle_idx])

    return np.array(gamma_windows), state_labels


def plot_umap_3d(embedding, labels, state_color_map, title="3D UMAP"):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for state, color in state_color_map.items():
        mask = np.array(labels) == state
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], label=state, color=color, s=10, alpha=0.5
        )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("UMAP-1", fontsize=13, labelpad=10)
    ax.set_ylabel("UMAP-2", fontsize=13, labelpad=10)
    ax.set_zlabel("UMAP-3", fontsize=13, labelpad=10)

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.legend(loc="upper left", fontsize=10, title="HHMM State")
    plt.tight_layout()
    plt.show()


#######################################################
# UMAPS - Temporal Progression
#######################################################

from matplotlib import cm
from matplotlib.cm import ScalarMappable


def add_time_index(df, session_col="Session"):
    df = df.copy()
    df["T_Index"] = df.groupby(session_col).cumcount() + 1
    return df


def prepare_sliding_window_data_temporal(
    df, window_size=20, gamma_col="gamma", state_col="HHMM State", grid_col="Grid Number", decision_nodes=None
):
    gamma_windows = []
    state_labels = []
    t_indices = []

    encoder = OneHotEncoder(sparse_output=False)
    encoded_states = encoder.fit_transform(df[[state_col]])
    middle_offset = window_size // 2

    for i in range(len(df) - window_size):
        middle_idx = i + middle_offset
        if decision_nodes is not None and df[grid_col].iloc[middle_idx] not in decision_nodes:
            continue

        gamma_segment = df[gamma_col].values[i : i + window_size]
        state_segment = encoded_states[middle_idx]
        combined = np.concatenate([gamma_segment, state_segment])

        gamma_windows.append(combined)
        state_labels.append(df[state_col].iloc[middle_idx])
        t_indices.append(df["T_Index"].iloc[middle_idx])

    return np.array(gamma_windows), state_labels, np.array(t_indices)


def compute_umap_embedding_temporal(
    df,
    window_size=20,
    gamma_col="gamma",
    state_col="HHMM State",
    grid_col="Grid Number",
    decision_nodes=None,
    session_filter=None,
):
    if session_filter is not None:
        df = df[df["Session"] == session_filter].reset_index(drop=True)

    df = add_time_index(df)
    df[gamma_col] = (df[gamma_col] - df[gamma_col].mean()) / df[gamma_col].std()

    window_data, labels, t_indices = prepare_sliding_window_data_temporal(
        df, window_size, gamma_col, state_col, grid_col, decision_nodes
    )

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", n_components=3, random_state=42)
    embedding = reducer.fit_transform(window_data)

    return embedding, labels, t_indices


def plot_umap_embedding_temporal_3d(embedding, labels, t_indices, title="Temporal UMAP"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    t_indices_int = t_indices.astype(int)
    cmap = cm.get_cmap("viridis", t_indices_int.max() + 1)
    colors = cmap(t_indices_int)

    for label in np.unique(labels):
        mask = np.array(labels) == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], label=label, color=colors[mask], alpha=0.85, s=6
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    sm = ScalarMappable(cmap=cmap)
    sm.set_array(t_indices_int)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label("Time Index", fontsize=12)

    ax.legend(loc="upper right", fontsize=9, title="HHMM State")
    plt.tight_layout()
    plt.show()
