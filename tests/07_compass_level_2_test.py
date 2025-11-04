import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class TestCompassLevel2:

    def test_compass_level_2_0(self, create_project_fixture):
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
        assert "KDE" in df_hmm.columns

        # Reference Info
        df_hmm = assign_reference_info(df_hmm)

        # Compute angle deviation
        ROLLING_WINDOW = 5
        df_hmm = compute_angle_deviation(df_hmm, rolling_window=ROLLING_WINDOW)
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "Targeted_Angle_abs" in df_hmm.columns
        assert "Targeted_Angle_smooth_abs" in df_hmm.columns

        # Run the value distance computation pipeline
        df_hmm = compute_value_distance(df_hmm, center_grids=[84, 85])
        assert isinstance(df_hmm, pd.DataFrame)
        assert not df_hmm.empty
        assert "VB_Distance" in df_hmm.columns

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
        features = ['HMM_State','VB_Distance','Targeted_Angle_smooth_abs','KDE']
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
            bout_col='Bout_ID',
            patience=None,
        )
        assert isinstance(df_hier, pd.DataFrame)
        assert not df_hier.empty
        assert isinstance(cv_results, list)
        assert len(cv_results) > 0
        save_path = project_path / "results" / "csvs" / "combined" / "hhmm_state_file.csv"
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

    def test_compass_level_2_1(self, create_project_fixture):
        from compass_labyrinth.compass.level_2 import (
            assign_reward_orientation,    
            assign_hhmm_state,
        )

        config, _ = create_project_fixture
        project_path = Path(config["project_path_full"])
        df_hier = pd.read_csv(project_path / "results" / "csvs" / "combined" / "hhmm_state_file.csv")

        # Assign reward orientation based on session-specific angle medians
        df_hier = assign_reward_orientation(
            df_hier,
            angle_col='Targeted_Angle_smooth_abs',
            level_2_state_col='Level_2_States',
            session_col='Session',
        )

        # Then assign the final HHMM state
        df_hier = assign_hhmm_state(
            df_hier,
            level_1_state_col='HMM_State',
            level_2_state_col='Reward_Oriented',
        )

        assert isinstance(df_hier, pd.DataFrame)
        assert not df_hier.empty
        assert "Reward_Oriented" in df_hier.columns
        assert "HHMM State" in df_hier.columns