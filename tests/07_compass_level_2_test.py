import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class TestCompassLevel2:

    def test_compass_level_2(self, create_project_fixture, hmm_results_fixture):
        from compass_labyrinth.compass.level_2 import (
            loso_kde_cv,
            compute_kde_scaled,
            assign_reference_info,
            compute_angle_deviation,
            compute_value_distance,
            compute_spatial_embedding,
            create_embedding_grid,
            plot_spatial_embedding,
            run_compass,
            visualize_cv_results,
            assign_reward_orientation,
            assign_hhmm_state,
            plot_state_sequences,
            plot_hhmm_state_sequence,
        )

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hmm = pd.read_csv(project_path / "results" / "compass_level_1" / "data_with_states.csv")

        # Define bandwidth search space
        smoothing_factors = [0.5, 2, 4]

        # Run LOSO CV to get best sigma
        best_sigma = loso_kde_cv(df_hmm, smoothing_factors)

        # Compute KDE using optimal sigma
        df_hmm = compute_kde_scaled(df_hmm, best_sigma)
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "kde" in df_hmm.columns

        # Reference Info
        df_hmm = assign_reference_info(df_hmm)

        # Compute angle deviation
        ROLLING_WINDOW = 5
        df_hmm = compute_angle_deviation(df_hmm, rolling_window=ROLLING_WINDOW)
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "targeted_angle_abs" in df_hmm.columns
        assert "targeted_angle_smooth_abs" in df_hmm.columns

        # Run the value distance computation pipeline
        df_hmm = compute_value_distance(df_hmm, center_grids=[84, 85])
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "vb_distance" in df_hmm.columns

        # Compute Smoothed Spatial Embedding
        df_smoothed = compute_spatial_embedding(df_hmm, sigma=2)

        # Convert Smoothed Data to Grid Format
        embedding_grid = create_embedding_grid(df_smoothed)

        # Visualize Spatial Embedding as Heatmap
        fig_0 = plot_spatial_embedding(
            config=config,
            embedding_grid=embedding_grid,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_0, plt.Figure)
        plt.close(fig_0)
        save_path = project_path / "figures" / "spatial_embedding_heatmap.pdf"
        assert save_path.exists()

        # Run CoMPASS Level 2
        features = ["hmm_state", "vb_distance", "targeted_angle_smooth_abs", "kde"]
        for f in features:
            assert f in df_hmm.columns

        df_hier, cv_results = run_compass(
            config=config,
            df=df_hmm,
            features=features,
            phase_options=[5],
            ncomp_options=[2],
            k_options=[2],
            reg_options=[1e-4],
            terminal_values=[47],
            bout_col="bout_id",
            patience=None,
        )
        assert isinstance(df_hier, pd.DataFrame)
        assert not df_hier.empty
        assert isinstance(cv_results, list)
        assert len(cv_results) > 0
        save_path = project_path / "csvs" / "combined" / "hhmm_state_file.csv"
        assert save_path.exists()

        # Visualize CV Results
        all_figs = visualize_cv_results(
            config=config,
            all_results=cv_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(all_figs, list)
        assert len(all_figs) == len(cv_results)
        for fig in all_figs:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

        # Assign reward orientation based on session-specific angle medians
        df_hier = assign_reward_orientation(
            df_hier,
            angle_col="targeted_angle_smooth_abs",
            level_2_state_col="level_2_states",
            session_col="session",
        )

        # Then assign the final HHMM state
        df_hier = assign_hhmm_state(
            df_hier,
            level_1_state_col="hmm_state",
            level_2_state_col="reward_oriented",
        )

        assert isinstance(df_hier, pd.DataFrame)
        assert not df_hier.empty
        assert "reward_oriented" in df_hier.columns
        assert "hhmm_state" in df_hier.columns

        # Plot state sequences for all sessions
        all_figs_2 = plot_state_sequences(
            config=config,
            df=df_hier,
            genotype="WT",
            state_col="level_2_states",
            sessions_to_plot="all",
            title_prefix="State Sequence",
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(all_figs_2, list)
        for fig in all_figs_2:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

        # Plot HHMM state sequences for all sessions
        all_figs_3 = plot_hhmm_state_sequence(
            config=config,
            df=df_hier,
            session_col="session",
            state_col="hhmm_state",
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(all_figs_3, list)
        for fig in all_figs_3:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
