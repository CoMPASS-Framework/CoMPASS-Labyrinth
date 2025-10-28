import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_step_and_angle_distributions(
    config: dict,
    df: pd.DataFrame,
    save_fig: bool = True,
    show_fig: bool = True,
    return_fig: bool = False,
) -> None | plt.Figure:
    """
    Plot histograms for the Step Length Distribution and Turning Angle Distribution.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing project settings.
    df : pd.DataFrame
        DataFrame containing step and angle data.
    save_fig : bool
        Whether to save the figure.
    show_fig : bool
        Whether to display the figure.
    return_fig : bool
        Whether to return the figure object.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram: Step Length Distribution
    sns.histplot(
        df["step"],
        bins=range(0, int(df["step"].max()) + 20, 20),
        kde=False,
        ax=ax1,
    )
    ax1.set_title("Step Length Distribution")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Count")

    # Histogram: Turning Angle Distribution (in radians)
    sns.histplot(
        df["angle"],
        bins=int((df["angle"].max() - df["angle"].min()) / 0.1),
        kde=False,
        ax=ax2,
    )
    ax2.set_title("Turning Angle Distribution")
    ax2.set_xlabel("Angle (rad)")
    ax2.set_ylabel("Count")

    fig.tight_layout()

    # Save figure
    if save_fig:
        save_path = Path(config["project_path_full"]) / "figures" / "step_and_angle_distribution.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    # Show figure
    if show_fig:
        plt.show()

    # Return figure
    if return_fig:
        return fig
