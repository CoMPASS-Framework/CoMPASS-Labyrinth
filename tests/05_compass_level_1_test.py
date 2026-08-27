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

    def test_run_compass_fit_hmm(self, create_project_fixture, hmm_results_fixture):
        from compass_labyrinth.compass.level_1 import (
            GammaHMM,
            print_hmm_summary,
        )

        config, _ = create_project_fixture

        res = hmm_results_fixture
        assert hasattr(res, "model")
        assert isinstance(res.model, GammaHMM)

        assert hasattr(res, "summary")
        assert isinstance(res.summary, dict)

        assert hasattr(res, "records")
        assert isinstance(res.records, pd.DataFrame)
        assert not res.records.empty

        assert hasattr(res, "data")
        assert isinstance(res.data, pd.DataFrame)
        assert not res.data.empty

        print_hmm_summary(
            model_summary=res.summary,
            model=res.model,
        )

        results_path = Path(config["project_path_full"]) / "results" / "compass_level_1"
        assert results_path.exists()

        model_path = results_path / "fitted_model.joblib"
        assert model_path.exists()

        records_path = results_path / "model_selection_records.csv"
        assert records_path.exists()

        data_path = results_path / "data_with_states.csv"
        assert data_path.exists()

        summary_path = results_path / "model_summary.json"
        assert summary_path.exists()
