from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import mlab
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter


# ----------------------------
# KDE Plot per Session
# ----------------------------
def plot_kde_per_session(df, best_sigma, kde_col="KDE"):
    for session in df["Session"].unique():
        sess_df = df[df["Session"] == session]
        if len(sess_df) < 5:
            continue

        x_vals, y_vals = sess_df["x"], sess_df["y"]
        kde_vals = sess_df[kde_col].to_numpy()

        x_min, x_max = x_vals.min() - 0.5, x_vals.max() + 0.5
        y_min, y_max = y_vals.min() - 0.5, y_vals.max() + 0.5
        x_grid = np.linspace(x_min, x_max, 150)
        y_grid = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x_grid, y_grid)

        grid_kde = mlab.griddata(x_vals, y_vals, kde_vals, x_grid, y_grid, interp="linear")
        grid_kde = np.nan_to_num(grid_kde)

        scaled = MinMaxScaler((0, 3)).fit_transform(grid_kde.reshape(-1, 1)).reshape(grid_kde.shape)

        plt.figure(figsize=(8, 7))
        plt.imshow(scaled, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="viridis", interpolation="bicubic")
        plt.scatter(x_vals, y_vals, c="white", s=6, edgecolor="black", linewidth=0.2, alpha=0.7)
        plt.title(f"KDE Spatial Map – Session {session} (σ = {best_sigma})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.colorbar(label="Normalized KDE Density [0–3]")
        plt.tight_layout()
        plt.show()


# ----------------------------
# Compute Smoothed Spatial Embedding
# ----------------------------
def compute_spatial_embedding(df: pd.DataFrame, sigma: float = 2) -> pd.DataFrame:
    df = df.copy()
    df["spatial_value_raw"] = 1 - (df["Value"] / df["Value"].max())

    x_max = int(df["x"].max()) + 1
    y_max = int(df["y"].max()) + 1
    grid = np.full((x_max, y_max), np.nan)

    for _, row in df.iterrows():
        grid[int(row["x"]), int(row["y"])] = row["spatial_value_raw"]

    smoothed_grid = gaussian_filter(np.nan_to_num(grid, nan=0), sigma=sigma)
    smoothed_grid[np.isnan(grid)] = np.nan

    df["spatial_embedding"] = df.apply(lambda row: smoothed_grid[int(row["x"]), int(row["y"])], axis=1)

    return df


# ----------------------------
## Create Embedding Grid
# ----------------------------
def create_embedding_grid(df: pd.DataFrame, value_column: str = "spatial_embedding") -> np.ndarray:
    x_max = int(df["x"].max()) + 1
    y_max = int(df["y"].max()) + 1
    embedding_grid = np.full((x_max, y_max), np.nan)

    for _, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        embedding_grid[x, y] = row[value_column]

    return np.transpose(embedding_grid)


# ----------------------------
## Plot Spatial Embedding Heatmap
# ----------------------------
def plot_spatial_embedding(
    config: dict,
    embedding_grid: np.ndarray,
    title: str = "Spatial Embedding Heatmap",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot a heatmap of the spatial embedding grid.

    Parameters:
    -----------
    config : dict
        Configuration dictionary for this project.
    embedding_grid : np.ndarray
        2D array representing the spatial embedding values.
    title : str
        Title of the heatmap.
    save_fig : bool
        Whether to save the figure to disk.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.
    
    Returns:
    --------
    plt.Figure or None
        The matplotlib Figure object if return_fig is True, else None.
    """
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(
        embedding_grid,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Spatial Embedding (Smoothed)"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "spatial_embedding_heatmap.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


#######################################################
# Data Stream Analysis
#######################################################
def normalize_features(df, feature_cols):
    """
    Normalize specified feature columns to [0, 1] range.
    """
    df = df.copy()
    for feat in feature_cols:
        min_val = df[feat].min()
        max_val = df[feat].max()
        if max_val > min_val:
            df[feat] = (df[feat] - min_val) / (max_val - min_val)
    return df


def compute_detailed_bout_summary(
    df,
    feature_cols,
    node_filter="Decision (Reward)",
    state_col="HMM State",
    target_zone="Target Zone",
    valid_bout_threshold=10,
    bout_col="Bout_ID",
):
    """
    Compute per-bout median values of features and success/validity flags.
    """
    df = normalize_features(df, feature_cols)
    cols = ["Session", "Genotype", "Bout_no"] + feature_cols + ["Valid_bout", "Successful_bout", "Probability_1"]
    index_df = pd.DataFrame(columns=cols)
    session_clusters = [x for _, x in df.groupby("Session")]

    j = 0
    for session_df in session_clusters:
        bouts = [x for _, x in session_df.groupby(bout_col)]
        if len(bouts) > 1:
            bouts = bouts[1:]  # skip bout 0
        boutnum = 1
        prob_list = []
        for bout in bouts:
            subset = bout[bout["NodeType"] == node_filter]
            row = {
                "Session": session_df["Session"].iloc[0],
                "Genotype": session_df["Genotype"].iloc[0],
                "Bout_no": boutnum,
            }
            for feat in feature_cols:
                row[feat] = subset[feat].median()

            prob = subset[state_col].value_counts(normalize=True).get(1, np.nan)
            prob_list.append(prob)
            row["Probability_1"] = np.nanmedian(prob_list)

            if bout["Grid Number"].nunique() > valid_bout_threshold:
                row["Valid_bout"] = "Valid"
            row["Successful_bout"] = "Successful" if target_zone in bout["Region"].values else "Unsuccessful"

            index_df.loc[j] = row
            boutnum += 1
            j += 1

    return index_df


def plot_measures_by_bout_type(index_df, feature_cols=None):
    """
    Plot violin + box + swarm plots for angular measures by bout type.
    """
    for col in feature_cols:
        index_df[col] = pd.to_numeric(index_df[col], errors="coerce")

    df_melted = index_df.melt(id_vars=["Session", "Successful_bout"], value_vars=feature_cols)
    df_melted = df_melted.groupby(["Session", "Successful_bout", "variable"])["value"].median().reset_index()

    plt.figure(figsize=(8, 6))
    palette = {"Successful": "cornflowerblue", "Unsuccessful": "grey"}

    sns.violinplot(
        data=df_melted,
        x="variable",
        y="value",
        hue="Successful_bout",
        palette=palette,
        split=True,
        inner=None,
        linewidth=1.2,
        alpha=0.8,
    )

    sns.boxplot(
        data=df_melted,
        x="variable",
        y="value",
        hue="Successful_bout",
        palette=["blue", "black"],
        width=0.3,
        showcaps=True,
        boxprops={"zorder": 2, "facecolor": "none"},
        whiskerprops={"zorder": 2},
        medianprops={"zorder": 3},
    )

    sns.stripplot(
        data=df_melted,
        x="variable",
        y="value",
        hue="Successful_bout",
        palette=["blue", "black"],
        dodge=True,
        alpha=0.7,
        jitter=True,
        size=4,
        marker="o",
    )

    plt.xlabel("")
    plt.ylabel("Standardized Median Values", fontsize=12)
    plt.title("Measures by Bout Type at Decision Nodes", fontsize=16, fontweight="bold")
    plt.legend(title="Bout Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(ticks=range(len(feature_cols)), labels=feature_cols, fontsize=15)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
