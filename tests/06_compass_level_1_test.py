import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class TestCompasLevel1:

    def test_compass_level_1_df(self, compass_level_1_fixture):
        assert isinstance(compass_level_1_fixture, pd.DataFrame)
        assert not compass_level_1_fixture.empty

    def test_compass_level_1_plots(self, create_project_fixture, compass_level_1_fixture):
        from compass_labyrinth.compass.level_1 import plot_step_and_angle_distributions

        config, cohort_metadata = create_project_fixture

        fig = plot_step_and_angle_distributions(
            config=config,
            df=compass_level_1_fixture,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        save_path = Path(config["project_path_full"]) / "figures" / "step_and_angle_distribution.pdf"
        assert save_path.exists()

    def test_run_compass_fit_hmm(self, compass_level_1_fixture):
        from compass_labyrinth.compass.level_1 import fit_best_hmm
        import numpy as np

        ## TODO - run and create asserts once the function is optimized

        # res = fit_best_hmm(
        #     preproc_df=compass_level_1_fixture,
        #     n_states=2,
        #     n_iter=5,
        #     opt_methods=("L-BFGS-B",),
        #     use_abs_angle=(True, False),
        #     stationary_flag="auto",
        #     use_data_driven_ranges=True,
        #     angle_mean_biased=(np.pi/2, 0.0),
        #     session_col="Session",
        #     seed=123,
        #     show_progress=True
        # )