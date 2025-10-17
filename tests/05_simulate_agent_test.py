from logging import config
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TestPerformanceMetrics:

    def test_simulate_agent_df(self, simulate_agent_fixture):
        assert isinstance(simulate_agent_fixture, dict)
        for genotype, df in simulate_agent_fixture.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    def test_agent_transition_performance(self, create_project_fixture, simulate_agent_fixture):
        from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import (
            plot_agent_transition_performance,
            plot_relative_agent_performance,
            run_mixedlm_for_all_genotypes,
        )
        
        config, cohort_metadata = create_project_fixture
        sim_results = simulate_agent_fixture

        fig_1 = plot_agent_transition_performance(
            config=config,
            evaluation_results=sim_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_1, plt.Figure)
        plt.close(fig_1)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_sim_agent_mouse_perf.pdf"
        assert fig_path.exists()

        fig_2 = plot_relative_agent_performance(
            config=config,
            evaluation_results=sim_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_2, plt.Figure)
        plt.close(fig_2)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_relative_perf.pdf"
        assert fig_path.exists()

        pvals = run_mixedlm_for_all_genotypes(
            config=config,
            evaluation_results=sim_results,
            save_fig=True,
            show_fig=False,
        )
        assert isinstance(pvals, dict)
        fig_path = Path(config["project_path_full"]) / "figures" / "cumulative_sim_agent_mouse_perf.pdf"
        assert fig_path.exists()
