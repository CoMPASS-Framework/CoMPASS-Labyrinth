import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class TestCompasPosthoc:

    def test_compass_posthoc_heatmaps(self, create_project_fixture):
        from compass_labyrinth.post_hoc_analysis.level_1 import plot_all_genotype_heatmaps

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")

        grid_filename = "Session-3 grid.shp"

        fig = plot_all_genotype_heatmaps(
            config=config,
            df_hmm=df_hmm,
            grid_filename=grid_filename,
            highlight_grids="decision_reward",
            target_grids="target_zone",
            hmm_state=2,
            cmap='RdBu',
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        save_path = project_path / "figures" / "all_genotypes_grid_heatmap.pdf"
        assert save_path.exists()
