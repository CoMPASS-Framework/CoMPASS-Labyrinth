"""
HEATMAP REPRESENTATIONS
Author: Shreya Bangera
Goal:
    ├── Grid-wise probability heatmaps showing the proportion of time spent in a specific HMM state across all sessions.
    ├── Interactive version of the visualization available too.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import warnings

from compass_labyrinth.constants import NODE_TYPE_MAPPING


warnings.simplefilter(action="ignore", category=FutureWarning)


##################################################################
# Plot 1: Heatmap Representations of HMM States
###################################################################
def compute_state_proportion(
    df: pd.DataFrame,
    genotype_name: str,
    hmm_state: int = 2,
) -> pd.DataFrame:
    """
    Compute state proportions and return filtered dataframe for given genotype and HMM state.

    Parameters
    -----------
    df : pd.DataFrame
        DataFrame containing columns: 'Genotype', 'Grid Number', 'HMM_State', 'x', 'y'.
    genotype_name : str
        Name of the genotype to filter.
    hmm_state : int
        HMM state to filter.

    Returns
    --------
    pd.DataFrame
        DataFrame with computed proportions for the specified genotype and HMM state.
    """
    st_cnt = df.groupby(["Genotype", "Grid Number", "HMM_State"]).size().rename("cnt").reset_index()
    gn_cnt = df.groupby(["Genotype", "Grid Number"]).size().rename("tot").reset_index()
    x_y = df.groupby(["Genotype", "Grid Number"]).agg({"x": "mean", "y": "mean"}).reset_index()
    state_count = st_cnt.merge(gn_cnt, on=["Genotype", "Grid Number"], how="left")
    state_count["prop"] = state_count["cnt"] / state_count["tot"]
    state_count = state_count.merge(x_y, on=["Genotype", "Grid Number"], how="left")

    return state_count[(state_count.Genotype == genotype_name) & (state_count["HMM_State"] == hmm_state)].reset_index(
        drop=True
    )


def create_grid_geodata(config: dict, grid_filename: str) -> gpd.GeoDataFrame:
    """Load the shapefile grid as GeoDataFrame."""
    gridfile = Path(config["project_path_full"]) / "data" / "grid_files" / grid_filename
    return gpd.read_file(gridfile)


def map_points_to_grid(df_points: pd.DataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map mean x, y points to the grid."""
    points = [Point(xy) for xy in zip(df_points["x"], df_points["y"])]
    pnt_gpd = gpd.GeoDataFrame(geometry=points, index=np.arange(len(points)), crs=grid.crs)
    pointInPolys = gpd.tools.sjoin(pnt_gpd, grid, predicate="within", how="left")
    return pointInPolys


def merge_state_proportions_to_grid(
    grid: gpd.GeoDataFrame,
    df_props: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Merge state proportions to shapefile grid polygons."""
    prop_by_grid = df_props[["Grid Number", "prop"]].copy()
    prop_by_grid = prop_by_grid.rename(columns={"prop": "State1_Proportion"})
    return grid.merge(
        prop_by_grid,
        left_on="FID",
        right_on="Grid Number",
        how="left",
    )


def plot_grid_heatmap(
    config: dict,
    grid: gpd.GeoDataFrame,
    genotype_name: str,
    highlight_grids: str | None = None,
    target_grids: str | None = None,
    cmap: str = "RdBu",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot grid heatmap for state proportions.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    grid : gpd.GeoDataFrame
        GeoDataFrame of the grid with state proportions merged.
    genotype_name : str
        Name of the genotype to be plotted.
    highlight_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    cmap : str
        Colormap to use for the heatmap.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : plt.Figure | None
        Matplotlib Figure object if return_fig is True, else None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    grid.plot(
        column="State1_Proportion",
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        vmin=0,
        vmax=1,
        edgecolor="0.8",
        legend=True,
        cax=cax,
        alpha=1.0,
    )

    if highlight_grids is not None:
        highlight_grids_values = NODE_TYPE_MAPPING[highlight_grids]
        edge_subset = grid[grid["FID"].isin(highlight_grids_values)]
        edge_subset.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2)
    if target_grids is not None:
        target_grids_values = NODE_TYPE_MAPPING[target_grids]
        target_nodes = grid[grid["FID"].isin(target_grids_values)]
        target_nodes.plot(ax=ax, edgecolor="yellow", facecolor="none", linewidth=5)

    ax.set_title(f"{genotype_name} (Sternum Tracking)", fontsize=14)
    plt.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / f"{genotype_name}_grid_heatmap.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


##################################################################
# Heatmap Representations of HMM States for all genotypes
###################################################################
def plot_all_genotype_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    grid_filename: str,
    highlight_grids: str | None = None,
    target_grids: str | None = None,
    hmm_state: int = 1,
    cmap: str = "RdBu",
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot grid heatmaps for all genotypes.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'Genotype' and 'Grid Number' columns.
    grid_filename : str
        Filename of the grid shapefile.
    highlight_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    hmm_state : int
        HMM state to filter.
    cmap : str
        Colormap to use for the heatmap.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : plt.Figure | None
        Matplotlib Figure object if return_fig is True, else None.
    """
    genotypes = sorted(df_hmm["Genotype"].unique())
    n_genotypes = len(genotypes)
    n_cols = math.ceil(n_genotypes**0.5)
    n_rows = math.ceil(n_genotypes / n_cols)

    grid = create_grid_geodata(config, grid_filename)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, genotype in enumerate(genotypes):
        # Step 1: Compute state proportions
        state_df = compute_state_proportion(df_hmm, genotype, hmm_state)

        # Step 2: Map mean (x, y) points to grid polygons
        pointInPolys = map_points_to_grid(state_df, grid)

        # Step 3: Merge with grid
        grid_mapped = merge_state_proportions_to_grid(grid, state_df)

        # Step 4: Plot on subplot
        ax = axs[i]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        grid_mapped.plot(
            column="State1_Proportion",
            cmap=cmap,
            linewidth=0.8,
            ax=ax,
            vmin=0,
            vmax=1,
            edgecolor="0.8",
            legend=True,
            cax=cax,
            alpha=1.0,
        )

        if highlight_grids is not None:
            highlight_grids_values = NODE_TYPE_MAPPING[highlight_grids]
            edge_subset = grid_mapped[grid_mapped["FID"].isin(highlight_grids_values)]
            edge_subset.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2)

        if target_grids is not None:
            target_grids_values = NODE_TYPE_MAPPING[target_grids]
            target_subset = grid_mapped[grid_mapped["FID"].isin(target_grids_values)]
            target_subset.plot(ax=ax, edgecolor="yellow", facecolor="none", linewidth=5)

        ax.set_title(f"{genotype} (Sternum Tracking)", fontsize=12)
        ax.axis("off")

    # Hide unused axes
    for j in range(len(genotypes), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"HMM State {hmm_state} Proportion Across Genotypes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_grid_heatmap.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig


##################################################################
# Plot 1: Heatmap Representations of HMM States - INTERACTIVE MAP
###################################################################
def get_grid_centroids(grid):
    grid["x"] = grid.geometry.centroid.x
    grid["y"] = grid.geometry.centroid.y
    return {row["FID"]: (row["x"], row["y"]) for _, row in grid.iterrows()}


def overlay_trajectory_lines_plotly(
    fig: go.Figure,
    df_hmm: pd.DataFrame,
    genotype_name: str,
    grid_centroids: dict,
    top_percent: float = 0.1,
):
    """
    Overlay trajectory lines on the interactive heatmap.

    Parameters
    -----------
    fig : go.Figure
        Plotly Figure object to overlay lines on.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'Genotype' and 'Grid Number' columns.
    genotype_name : str
        Name of the genotype to filter.
    grid_centroids : dict
        Dictionary mapping grid numbers to their centroid coordinates.
    top_percent : float
        Top percentage of transitions to visualize.

    Returns
    --------
    None
    """
    df_geno = df_hmm[df_hmm["Genotype"] == genotype_name].copy()
    df_geno["Grid.Next"] = df_geno["Grid Number"].shift(-1)
    df_geno["Grid.Prev"] = df_geno["Grid Number"]
    transitions = df_geno[["Grid.Prev", "Grid.Next"]].dropna()
    transitions = transitions[transitions["Grid.Prev"] != transitions["Grid.Next"]].astype(int)
    transitions["pair"] = list(zip(transitions["Grid.Prev"], transitions["Grid.Next"]))
    trans_counts = transitions["pair"].value_counts()

    top_n = int(len(trans_counts) * top_percent)
    trans_counts = trans_counts.iloc[:top_n]

    max_freq = trans_counts.max()
    min_freq = trans_counts.min()

    for (g1, g2), freq in trans_counts.items():
        if g1 in grid_centroids and g2 in grid_centroids:
            pt1, pt2 = grid_centroids[g1], grid_centroids[g2]
            if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                continue
            # Smooth interpolation
            x_vals = np.array([pt1[0], (pt1[0] + pt2[0]) / 2, pt2[0]])
            y_vals = np.array([pt1[1], (pt1[1] + pt2[1]) / 2 + 2, pt2[1]])  # add slight curve
            spline = make_interp_spline([0, 1, 2], np.column_stack([x_vals, y_vals]), k=2)
            smooth = spline(np.linspace(0, 2, 20))
            xs, ys = smooth[:, 0], smooth[:, 1]

            line_alpha = np.interp(freq, [min_freq, max_freq], [0.2, 1.0])
            line_width = np.interp(freq, [min_freq, max_freq], [1, 10])

            fig.add_trace(
                go.Scatter(
                    x=xs.tolist(),
                    y=ys.tolist(),
                    mode="lines",
                    line=dict(color="black", width=line_width),
                    opacity=line_alpha,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


# ---------------- Plotly Heatmap ---------------- #
def plot_interactive_heatmap(
    config: dict,
    grid_mapped: gpd.GeoDataFrame,
    genotype_name: str,
    decision_grids: str | None = None,
    target_grids: str | None = None,
    entry_node_id: int | None = 47,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | go.Figure:
    """
    Plot interactive heatmap for state proportions.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    grid_mapped : gpd.GeoDataFrame
        GeoDataFrame of the grid with state proportions merged.
    genotype_name : str
        Name of the genotype to be plotted.
    decision_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    entry_node_id : int
        ID of the entry node (default is 47).
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : go.Figure | None
        Plotly Figure object if return_fig is True, else None.
    """
    fig = go.Figure()

    # Add grid cells as filled polygons
    for _, row in grid_mapped.iterrows():
        if row.geometry is None or row["State1_Proportion"] is None:
            continue
        x = list(row.geometry.exterior.xy[0])
        y = list(row.geometry.exterior.xy[1])
        color_val = row["State1_Proportion"]
        fillcolor = sample_colorscale("RdBu", [color_val])[0]  # RdBu reversed = blue to red
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                # fillcolor=f'rgba({255 * (1 - row["State1_Proportion"])},0,{255 * row["State1_Proportion"]},0.7)',
                fillcolor=fillcolor,
                line=dict(width=0.5, color="gray"),
                hoverinfo="text",
                text=f"Grid: {row['FID']}<br>Proportion: {row['State1_Proportion']:.2f}",
                showlegend=False,
            )
        )

    # Highlight decision nodes in black
    for fid in NODE_TYPE_MAPPING.get(decision_grids, []):
        if fid in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == fid, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=2),
                    name="Decision Node",
                    showlegend=False,
                )
            )

    # Highlight decision nodes in black
    if entry_node_id is not None:
        if entry_node_id in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == entry_node_id, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=5),
                    name="Entry Node",
                    showlegend=False,
                )
            )

    # Highlight target nodes in yellow
    for fid in NODE_TYPE_MAPPING.get(target_grids, []):
        if fid in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == fid, "geometry"].values[0]
            x = list(poly.exterior.xy[0])
            y = list(poly.exterior.xy[1])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="yellow", width=5),
                    name="Target Node",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{genotype_name}: Grid Heatmap with Trajectories",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(scaleanchor="x", showgrid=False, visible=False),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Add annotations above Grid Number 47 and below 84
    for fid, label, valign in [(47, "entry_zone", "top"), (84, "target_zone", "bottom")]:
        if fid in grid_mapped["FID"].values:
            poly = grid_mapped.loc[grid_mapped["FID"] == fid, "geometry"].values[0]
            centroid_x = poly.centroid.x
            centroid_y = poly.centroid.y

            offset = 55 if valign == "top" else -55

            fig.add_annotation(
                x=centroid_x,
                y=centroid_y + offset,
                text=label,
                showarrow=False,
                font=dict(color="black", size=14, family="Arial"),
                xanchor="center",
            )

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / f"{genotype_name}_interactive_grid_heatmap.html"
        fig.write_html(save_path)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        fig.show()

    # Return figure
    if return_fig:
        return fig


###################################################################
# Interactive Heatmaps for multiple genotypes
###################################################################
def plot_all_genotype_interactive_heatmaps(
    config: dict,
    df_hmm: pd.DataFrame,
    grid_filename: str,
    hmm_state: int = 2,
    decision_grids: str | None = None,
    target_grids: str | None = None,
    entry_node_id: int | None = 47,
    top_percent: float = 0.1,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | go.Figure:
    """
    Plot interactive grid heatmaps for all genotypes.

    Parameters
    -----------
    config : dict
        Configuration dictionary containing project details.
    df_hmm : pd.DataFrame
        DataFrame containing HMM data with 'Genotype' and 'Grid Number' columns.
    grid_filename : str
        Filename of the grid shapefile.
    hmm_state : int
        HMM state to filter.
    decision_grids : str | None
        Node type to highlight (e.g., "decision_reward").
    target_grids : str | None
        Node type to mark as target (e.g., "target_zone").
    entry_node_id : int | None
        ID of the entry node (default is 47).
    top_percent : float
        Top percentage of transitions to visualize.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns
    --------
    fig : go.Figure | None
        Plotly Figure object if return_fig is True, else None.
    """
    genotypes = sorted(df_hmm["Genotype"].unique())
    n = len(genotypes)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    grid = create_grid_geodata(config, grid_filename)
    grid_centroids = get_grid_centroids(grid)

    # Create subplot figure
    subplot_fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=genotypes,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    subplot_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if subplot_idx >= n:
                break

            genotype = genotypes[subplot_idx]
            subplot_idx += 1
            row = i + 1
            col = j + 1

            # 1. Compute state proportions
            state_df = compute_state_proportion(df_hmm, genotype, hmm_state)

            # 2. Merge into grid
            grid_mapped = merge_state_proportions_to_grid(grid, state_df)

            # 3. Generate interactive heatmap figure
            fig = plot_interactive_heatmap(
                config=config,
                grid_mapped=grid_mapped,
                decision_grids=decision_grids,
                target_grids=target_grids,
                genotype_name=genotype,
                entry_node_id=entry_node_id,
                save_fig=False,
                show_fig=False,
                return_fig=True,
            )

            # 4. Add trajectory lines
            overlay_trajectory_lines_plotly(
                fig=fig,
                df_hmm=df_hmm,
                genotype_name=genotype,
                grid_centroids=grid_centroids,
                top_percent=top_percent,
            )

            # 5. Add traces to subplot
            for trace in fig.data:
                subplot_fig.add_trace(trace, row=row, col=col)

    # Final layout improvements
    subplot_fig.update_layout(
        title_text=f"Genotype-wise Grid Heatmaps (HMM State {hmm_state})",
        height=600 * n_rows,
        width=600 * n_cols,
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=10),
        title_x=0.5,
    )

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_interactive_grid_heatmap.html"
        subplot_fig.write_html(save_path)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        subplot_fig.show()

    # Return figure
    if return_fig:
        return subplot_fig
