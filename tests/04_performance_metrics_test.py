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

    def test_plot_target_usage_vs_frames(self, create_project_fixture, exclusion_criteria):
        config, cohort_metadata = create_project_fixture
        fig_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames.png"
        assert fig_path.exists()

    def test_plot_target_usage_with_exclusions(self, create_project_fixture, exclusion_criteria):
        config, cohort_metadata = create_project_fixture
        fig_path = Path(config["project_path_full"]) / "figures" / "target_usage_vs_frames_exclusions.png"
        assert fig_path.exists()

    def test_exclusion_criteria(self, exclusion_criteria):
        assert isinstance(exclusion_criteria, pd.DataFrame)
        assert not exclusion_criteria.empty
