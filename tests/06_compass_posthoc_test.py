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

    def test_compass_posthoc_temporal_analysis(self, create_project_fixture):
        from compass_labyrinth.post_hoc_analysis.level_1 import (
            get_max_session_row_bracket,
            get_min_session_row_bracket,
            compute_node_state_medians_over_time,
            plot_node_state_median_curve,
        )

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")

        lower_limit = 0
        upper_limit = get_max_session_row_bracket(df_hmm)
        threshold =  get_min_session_row_bracket(df_hmm)  # Only show bins where all sessions are present
        bin_size = 2000
        palette = ['grey', 'black']
        figure_ylimit = (0.6, 1.1)

        # Step 1: Compute median probability of being in State 1 across time bins
        deci_df = compute_node_state_medians_over_time(
            df_hmm=df_hmm,
            state_types=[2],
            lower_lim=lower_limit,
            upper_lim=upper_limit,
            bin_size=bin_size
        )

        # Step 2: Optional filter to only plot early session bins
        deci_df = deci_df.loc[deci_df.Time_Bins < threshold]

        # Step 3: Plot time-evolving median probability curves
        fig = plot_node_state_median_curve(
            config=config,
            deci_df=deci_df,
            palette=palette,
            figure_ylimit=figure_ylimit,
            fig_title = 'Median Probability of Ambulatory State',
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        save_path = (
            Path(config["project_path_full"]) / "figures" / "temporal_median_state_probability_curve.pdf"
        )
        assert save_path.exists()

    def test_compass_posthoc_surveillance_analysis(self, create_project_fixture):
        from compass_labyrinth.post_hoc_analysis.level_1 import (
            assign_bout_indices,
            compute_surveillance_probabilities,
            plot_surveillance_by_bout,
            run_within_genotype_mixedlm_with_fdr,
            test_across_genotypes_per_bout,
        )

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")

        # Assign Bout Numbers 
        df_hmm = assign_bout_indices(
            df=df_hmm,
            delimiter_node=47,
        )
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "Bout_Index" in df_hmm.columns

        # Compute surveillance probability at Decision nodes by Bout type
        index_df, median_df = compute_surveillance_probabilities(
            df_hmm=df_hmm,
            decision_nodes="decision_reward",
        )
        assert isinstance(median_df, pd.DataFrame)
        assert not median_df.empty
        for col in ["Genotype", "Session", "Successful_bout", "Probability_1"]:
            assert col in median_df.columns

        # Barplot to depict the above with ttest-ind pvalue
        fig = plot_surveillance_by_bout(
            config=config,
            median_df=median_df,
            ylim=0.6,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        save_path = (
            Path(config["project_path_full"]) / "figures" / "surveillance_probability_by_bout.pdf"
        )
        assert save_path.exists()

        # LMM for same genotype comparison across Bout types
        df_within = run_within_genotype_mixedlm_with_fdr(median_df)
        assert isinstance(df_within, pd.DataFrame)

        # T-test across genotypes under Unsuccessful Bouts
        df_across_unsuccess = test_across_genotypes_per_bout(median_df, bout_type='Unsuccessful')
        assert isinstance(df_across_unsuccess, pd.DataFrame)
