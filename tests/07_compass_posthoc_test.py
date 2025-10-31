import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class TestCompasPosthocAnalysis:

    def test_compass_posthoc_heatmaps(self, create_project_fixture):
        from compass_labyrinth.post_hoc_analysis.level_1 import (
            plot_all_genotype_heatmaps,
            plot_all_genotype_interactive_heatmaps,
        )

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

        interactive_fig = plot_all_genotype_interactive_heatmaps(
            config=config,
            df_hmm=df_hmm,
            grid_filename=grid_filename,
            hmm_state=2,
            decision_grids="decision_reward",
            target_grids="target_zone",
            top_percent=.1,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(interactive_fig, go.Figure)
        assert (project_path / "figures" / "all_genotypes_interactive_grid_heatmap.html").exists()

    def test_compass_posthoc_surveillance(self, create_project_fixture):
        from compass_labyrinth.post_hoc_analysis.level_1 import (
            compute_state_probability,
            plot_state_probability_boxplot,
            run_pairwise_ttests,
        )

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")

        column_of_interest = 'NodeType'
        values_displayed = [
            '3-way Decision (Reward)', '4-way Decision (Reward)','Non-Decision (Reward)', 
            'Decision (Non-Reward)', 'Non-Decision (Non-Reward)',
            'Corner (Reward)', 'Corner (Non-Reward)'
        ]
        state = 1

        # Step 1: Compute proportions
        state_count_df = compute_state_probability(
            df_hmm=df_hmm,
            column_of_interest=column_of_interest,
            values_displayed=values_displayed,
            state=state,
        )

        # Step 2: Plot boxplot
        fig = plot_state_probability_boxplot(
            config=config,
            state_count_df=state_count_df,
            column_of_interest=column_of_interest,
            state=state,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        save_path = Path(config["project_path_full"]) / "figures" / f"state_{state}_probability_by_{column_of_interest}.pdf"
        assert save_path.exists()

        ttest_results = run_pairwise_ttests(
            state_count_df=state_count_df,
            column_of_interest=column_of_interest,
        )
        assert isinstance(ttest_results, pd.DataFrame)
        assert not ttest_results.empty
