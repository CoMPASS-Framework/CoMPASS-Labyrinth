import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

        GENOTYPE_DISP = "WT"
        LOWER_LIMIT = 0  # lower limit for bins
        BIN_SIZE = 10000  # bin size for the heatmap plot
        VMAX = 0.6  # max range on colorbar
        UPPER_LIMIT = get_max_session_row_bracket(df_all_csv)  # upper limit for bins

        # Store valid sessions post exclusion, specific to the genotype/group wanting to visualize
        valid_sessions = df_all_csv[df_all_csv.Genotype == GENOTYPE_DISP]["Session"].unique().tolist()
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
        included_genotypes = ["WT", "KO"]
        UPPER_LIMIT = get_max_session_row_bracket(df_all_csv)  # upper limit for bins

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
        required_columns = ["Session", "Genotype", "Bin", "Entropy"]
        for col in required_columns:
            assert col in entropy_df.columns

        # Check if the plot file was created
        fig_path = Path(config["project_path_full"]) / "figures" / "shannon_entropy.pdf"
        assert fig_path.exists()

    def test_statistical_tests(self, create_project_fixture, shannon_entropy):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            run_entropy_anova,
            run_fdr_pairwise_tests,
            run_mixed_model_per_genotype_pair,
        )

        # Repeated Measures ANOVA
        anova_result = run_entropy_anova(shannon_entropy)
        assert isinstance(anova_result.anova_table, pd.DataFrame)
        assert not anova_result.anova_table.empty

        # Pairwise t-tests + FDR correction (per bin, per genotype pair)
        fdr_results = run_fdr_pairwise_tests(shannon_entropy)
        assert isinstance(fdr_results, pd.DataFrame)
        assert not fdr_results.empty

        # Run per-pair mixed models
        mixed_results, interaction_table = run_mixed_model_per_genotype_pair(shannon_entropy)
        assert isinstance(mixed_results, dict)
        assert isinstance(interaction_table, pd.DataFrame)
        assert not interaction_table.empty

    def test_region_usage_over_bins(self, create_project_fixture, task_performance):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            compute_region_usage_over_bins,
            plot_region_usage_over_bins,
            run_region_usage_stats_mixedlm,
            run_region_usage_stats_fdr,
            plot_all_regions_usage_over_bins,
        )

        config, _ = create_project_fixture
        df_all_csv, pivot_dict = task_performance

        REGION = "target_zone"
        BIN_SIZE = 10000

        region_usage_df = compute_region_usage_over_bins(
            pivot_dict=pivot_dict,
            df_all_csv=df_all_csv,
            region=REGION,
            bin_size=BIN_SIZE,
        )

        assert isinstance(region_usage_df, pd.DataFrame)
        assert not region_usage_df.empty
        required_columns = ["Session", "Genotype", "Bin", REGION]
        for col in required_columns:
            assert col in region_usage_df.columns
        
        fig = plot_region_usage_over_bins(
            config=config,
            region_data=region_usage_df,
            region_name=REGION,
            ylim=(0, 1),
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )

        assert isinstance(fig, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / f"{REGION}_prop_usage.pdf"
        assert fig_path.exists()

        # Mixed Effects Model
        run_region_usage_stats_mixedlm(region_usage_df, region_col=REGION)

        # Pairwise t-tests + FDR correction
        fdr_results = run_region_usage_stats_fdr(region_usage_df, region_col=REGION)
        assert isinstance(fdr_results, pd.DataFrame)
        assert not fdr_results.empty

        # Proportion of usage across all Regions
        region_list = ['entry_zone', 'loops','dead_ends', 'neutral_zone', 'reward_path', 'target_zone']
        fig_all = plot_all_regions_usage_over_bins(
            config=config,
            pivot_dict=pivot_dict,
            df_all_csv=df_all_csv,
            region_list=region_list,
            bin_size=BIN_SIZE,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_all, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_regions_prop_usage.pdf"
        assert fig_path.exists()

    def test_bout_success(self, create_project_fixture, bout_level_metrics):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            summarize_bout_success_by_session,
            plot_success_rate,
            perform_genotype_ttests,
        )

        config, _ = create_project_fixture
        df_all_csv_wbouts = bout_level_metrics

        assert isinstance(df_all_csv_wbouts, pd.DataFrame)
        assert not df_all_csv_wbouts.empty

        # Compute success summary statistics
        bout_summary = summarize_bout_success_by_session(df_all_csv_wbouts)
        assert isinstance(bout_summary, pd.DataFrame)
        assert not bout_summary.empty

        # Plot % of successful bouts per genotype
        fig = plot_success_rate(
            config=config,
            summary_table=bout_summary,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / "cumulative_successful_bouts.pdf"
        assert fig_path.exists()

        # Perform t-tests between genotypes
        ttest_results_success = perform_genotype_ttests(bout_summary, rate_col='success_rate')
        assert isinstance(ttest_results_success, dict)
        assert len(ttest_results_success) > 0

    def test_time_based_successful_bouts(self, create_project_fixture, bout_level_metrics):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            compute_binned_success_summary,
            plot_binned_success,
            run_mixedlm_with_nans,
            run_repeated_measures_anova,
            run_pairwise_comparisons,
        )

        config, _ = create_project_fixture
        df_all_csv_wbouts = bout_level_metrics

        BIN_SIZE = 5000
        LOWER_BOUND = 0
        UPPER_BOUND = 90000

        summary_df = compute_binned_success_summary(
            df_all_csv=df_all_csv_wbouts,
            lower_succ_lim=LOWER_BOUND,
            upper_succ_lim=UPPER_BOUND,
            diff_succ=BIN_SIZE,
            valid_bout_threshold=19,
            optimal_path_regions=['entry_zone', 'reward_path', 'target_zone'],
            target_zone='target_zone'
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty

        fig = plot_binned_success(
            config=config,
            summary_df=summary_df,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / "time_based_successful_bouts.pdf"
        assert fig_path.exists()

        # Statistical Tests
        # Run Mixed Linear Model (preserves NaNs)
        #run_mixedlm_with_nans(summary_df)

        # Run Repeated Measures ANOVA (NaNs filled with 0)
        run_repeated_measures_anova(summary_df)

        # Run Pairwise Comparisons (with FDR correction)
        run_pairwise_comparisons(summary_df)

    def test_deviation_from_reward_path_and_velocity(self, create_project_fixture, task_performance):
        from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import (
            ensure_velocity_column,
            ensure_bout_indices,
            compute_deviation_velocity,
            process_deviation_velocity,
            plot_deviation_velocity_fit,
            plot_deviation_velocity_all,
        )

        config, _ = create_project_fixture
        df_all_csv, _ = task_performance

        # Ensures velocity column exists
        df_all_csv_wvelocity = ensure_velocity_column(
            df=df_all_csv,
            frame_rate=5.0,
        )
        assert isinstance(df_all_csv_wvelocity, pd.DataFrame)
        assert not df_all_csv_wvelocity.empty
        assert "Velocity" in df_all_csv_wvelocity.columns

        # Ensure Bout Index column Bout_ID exists
        df_all_csv_wbouts = ensure_bout_indices(
            df=df_all_csv_wvelocity,
            delimiter_node=47,
        )
        assert isinstance(df_all_csv_wbouts, pd.DataFrame)
        assert not df_all_csv_wbouts.empty
        assert "Bout_ID" in df_all_csv_wbouts.columns

        # Compute deviation and velocity per bout
        df_deviation = compute_deviation_velocity(df=df_all_csv_wbouts)
        assert isinstance(df_deviation, pd.DataFrame)
        assert not df_deviation.empty
        for col in ['ind_no', 'session', 'genotype', 'deviation', 'velocity']:
            assert col in df_deviation.columns

        # Process deviation and velocity (normalize, smooth, fit curves)
        GENOTYPE = 'WT'
        df_processed, params_dev, params_vel = process_deviation_velocity(
            index_df=df_deviation,
            genotype=GENOTYPE,
        )
        assert isinstance(df_processed, pd.DataFrame)
        assert not df_processed.empty
        for col in ['velocity_robust_scaled', 'velocity_normalized', 'velocity_smooth_normalized', 'deviation_smooth']:
            assert col in df_processed.columns
        assert isinstance(params_dev, (list, np.ndarray))
        assert isinstance(params_vel, (list, np.ndarray))

        # Plot deviation and velocity with fitted curves for a specific genotype
        fig = plot_deviation_velocity_fit(
            config=config,
            df=df_processed,
            params_dev=params_dev,
            params_vel=params_vel,
            genotype=GENOTYPE,
            max_bouts=200,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / f"{GENOTYPE}_deviation_velocity_metric.pdf"
        assert fig_path.exists()

        # Plot deviation and velocity with fitted curves for all genotypes
        fig_all = plot_deviation_velocity_all(
            config=config,
            index_df=df_deviation,
            max_bouts=200,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_all, plt.Figure)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_deviation_velocity_metric.pdf"
        assert fig_path.exists()

