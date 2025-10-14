import pytest
from pathlib import Path
import pandas as pd


class TestPerformanceMetrics:

    def test_time_binned_dict(self, create_time_binned_dict):
        assert isinstance(create_time_binned_dict, dict)
        assert len(create_time_binned_dict) > 0
        for key, v in create_time_binned_dict.items():
            assert isinstance(v, list)
            assert len(v) > 0
            for item in v:
                assert isinstance(item, pd.DataFrame)
                assert not item.empty

    def test_plot_target_usage_vs_frames(self, create_project_fixture, task_performance):
        config, cohort_metadata = create_project_fixture
        fig_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames.png"
        assert fig_path.exists()

    def test_plot_target_usage_with_exclusions(self, create_project_fixture, task_performance):
        config, cohort_metadata = create_project_fixture
        fig_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames_exclusions.png"
        assert fig_path.exists()

    def test_task_performance(self, task_performance):
        df_all_csv, pivot_dict = task_performance
        assert isinstance(df_all_csv, pd.DataFrame)
        assert not df_all_csv.empty
        assert isinstance(pivot_dict, dict)
        assert len(pivot_dict) > 0
        for key, v in pivot_dict.items():
            assert isinstance(v, list)
            assert len(v) > 0
            for item in v:
                assert isinstance(item, pd.DataFrame)
                assert not item.empty

    def test_plot_region_heatmaps_by_genotype(self, create_project_fixture, task_performance):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            get_max_session_row_bracket,
            plot_region_heatmaps,
            plot_region_heatmaps_all_genotypes,
        )

        config, cohort_metadata = create_project_fixture
        df_all_csv, pivot_dict = task_performance

        GENOTYPE_DISP = 'WT'
        LOWER_LIMIT = 0       # lower limit for bins 
        BIN_SIZE = 10000      # bin size for the heatmap plot
        VMAX = 0.6            # max range on colorbar
        UPPER_LIMIT = get_max_session_row_bracket(df_all_csv)  # upper limit for bins 

        # Store valid sessions post exclusion, specific to the genotype/group wanting to visualize
        valid_sessions = df_all_csv[df_all_csv.Genotype == GENOTYPE_DISP]['Session'].unique().tolist()
        if len(valid_sessions) == 0:
            raise ValueError("Valid sessions list is empty! Choose a valid Genotype.")

        # Plot the region-based heatmap
        plot_region_heatmaps(
            config=config,
            pivot_dict=pivot_dict,
            group_name=GENOTYPE_DISP,
            lower_lim=LOWER_LIMIT,
            upper_lim=UPPER_LIMIT,
            difference=BIN_SIZE,
            vmax=VMAX,
            included_sessions=valid_sessions,
            save_fig=True,
            show_fig=False,
            return_fig=False,
        )

        fig_path = Path(config["project_path_full"]) / "figures" / f"region_heatmaps_{GENOTYPE_DISP}.pdf"
        assert fig_path.exists()    

        # Plot all genotypes together
        included_genotypes = ['WT']  # add more genotypes
        UPPER_LIMIT = get_max_session_row_bracket(df_all_csv) # upper limit for bins 

        plot_region_heatmaps_all_genotypes(
            config=config,
            pivot_dict=pivot_dict,
            df_all_csv=df_all_csv, 
            lower_lim=LOWER_LIMIT,
            upper_lim=UPPER_LIMIT,
            difference=BIN_SIZE,
            included_genotypes=included_genotypes,
            spacing_w=0.2,
            spacing_h=0.15,
            show_colorbar=True,
            vmax=VMAX,
            save_fig=True,
            show_fig=False,
            return_fig=False,
        )   

        fig_path = Path(config["project_path_full"]) / "figures" / "region_heatmaps_all_genotypes.pdf"
        assert fig_path.exists()

    def test_shannon_entropy(self, create_project_fixture, shannon_entropy):
        config, _ = create_project_fixture
        entropy_df = shannon_entropy
        assert isinstance(entropy_df, pd.DataFrame)
        assert not entropy_df.empty
        required_columns = ['Session', 'Genotype', 'Bin', 'Entropy']
        for col in required_columns:
            assert col in entropy_df.columns

        # Check if the plot file was created
        fig_path = Path(config["project_path_full"]) / "figures" / "shannon_entropy.pdf"
        assert fig_path.exists()
