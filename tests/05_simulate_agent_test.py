from logging import config
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TestPerformanceMetrics:

    def test_simulate_agent_df(self, simulate_agent_fixture):
        assert isinstance(simulate_agent_fixture, pd.DataFrame)
        assert not simulate_agent_fixture.empty

    def test_agent_transition_performance(self, create_project_fixture, simulate_agent_fixture):
        from compass_labyrinth.behavior.behavior_metrics.simulation_modeling import plot_agent_transition_performance
        
        config, cohort_metadata = create_project_fixture
        df_sim = simulate_agent_fixture
        
        GENOTYPE = "WT"

        fig = plot_agent_transition_performance(
            config=config,
            df_result=df_sim,
            genotype=GENOTYPE,
            save_fig=True,
            show_fig=False,
            return_fig=True,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        fig_path = Path(config["project_path_full"]) / "figures" / f"{GENOTYPE}_sim_agent_mouse_perf.pdf"
        assert fig_path.exists()
