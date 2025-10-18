import pytest
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class TestSimulateAgent:

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

    def test_chi_square_analysis(self, create_project_fixture, simulate_agent_fixture):
        from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import (
            run_chi_square_analysis,
            plot_chi_square_and_rolling,
            plot_rolling_mean,
            plot_cumulative_chi_square,
        )

        config, cohort_metadata = create_project_fixture
        sim_results = simulate_agent_fixture

        chisq_results = run_chi_square_analysis(
            config=config,
            evaluation_results=sim_results,
            rolling_window=3,
        )
        assert isinstance(chisq_results, dict)

        fig_1 = plot_chi_square_and_rolling(
            config=config,
            chisquare_results=chisq_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_1, plt.Figure)
        plt.close(fig_1)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_chi_square_rolling.pdf"
        assert fig_path.exists()

        fig_2 = plot_rolling_mean(
            config=config,
            chisquare_results=chisq_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_2, plt.Figure)
        plt.close(fig_2)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_average_chi_square_rolling.pdf"
        assert fig_path.exists()

        fig_3 = plot_cumulative_chi_square(
            config=config,
            chisquare_results=chisq_results,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig_3, plt.Figure)
        plt.close(fig_3)
        fig_path = Path(config["project_path_full"]) / "figures" / "all_genotypes_cumulative_chi_square.pdf"
        assert fig_path.exists()

