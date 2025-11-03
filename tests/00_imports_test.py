import pytest


class TestMainModuleImports:
    """Test that main modules can be imported."""

    def test_import_compass_labyrinth(self):
        """Test importing the main package."""
        try:
            import compass_labyrinth

            assert compass_labyrinth is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth: {e}")

    def test_import_behavior_module(self):
        """Test importing the behavior module."""
        try:
            import compass_labyrinth.behavior

            assert compass_labyrinth.behavior is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior: {e}")

    def test_import_compass_module(self):
        """Test importing the compass module."""
        try:
            import compass_labyrinth.compass

            assert compass_labyrinth.compass is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.compass: {e}")

    def test_import_neural_module(self):
        """Test importing the neural module."""
        try:
            import compass_labyrinth.neural

            assert compass_labyrinth.neural is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.neural: {e}")

    def test_import_post_hoc_analysis_module(self):
        """Test importing the post_hoc_analysis module."""
        try:
            import compass_labyrinth.post_hoc_analysis

            assert compass_labyrinth.post_hoc_analysis is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.post_hoc_analysis: {e}")


class TestBehaviorSubmoduleImports:
    """Test that behavior submodules can be imported."""

    def test_import_pose_estimation(self):
        """Test importing pose_estimation submodule."""
        try:
            import compass_labyrinth.behavior.pose_estimation

            assert compass_labyrinth.behavior.pose_estimation is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.pose_estimation: {e}")

    def test_import_preprocessing(self):
        """Test importing preprocessing submodule."""
        try:
            import compass_labyrinth.behavior.preprocessing

            assert compass_labyrinth.behavior.preprocessing is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.preprocessing: {e}")

    def test_import_behavior_metrics(self):
        """Test importing behavior_metrics submodule."""
        try:
            import compass_labyrinth.behavior.behavior_metrics

            assert compass_labyrinth.behavior.behavior_metrics is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.behavior_metrics: {e}")


class TestCompassSubmoduleImports:
    """Test that compass submodules can be imported."""

    def test_import_compass_level_2(self):
        """Test importing compass level_2 submodule."""
        try:
            import compass_labyrinth.compass.level_2

            assert compass_labyrinth.compass.level_2 is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.compass.level_2: {e}")


class TestNeuralSubmoduleImports:
    """Test that neural submodules can be imported."""

    def test_import_eeg_preprocessing(self):
        """Test importing eeg_preprocessing submodule."""
        try:
            import compass_labyrinth.neural.eeg_preprocessing

            assert compass_labyrinth.neural.eeg_preprocessing is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.neural.eeg_preprocessing: {e}")

    def test_import_ephys_behavior_analysis(self):
        """Test importing ephys_behavior_analysis submodule."""
        try:
            import compass_labyrinth.neural.ephys_behavior_analysis

            assert compass_labyrinth.neural.ephys_behavior_analysis is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.neural.ephys_behavior_analysis: {e}")


class TestPostHocAnalysisSubmoduleImports:
    """Test that post_hoc_analysis submodules can be imported."""

    def test_import_level_1(self):
        """Test importing post_hoc_analysis level_1 submodule."""
        try:
            import compass_labyrinth.post_hoc_analysis.level_1

            assert compass_labyrinth.post_hoc_analysis.level_1 is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.post_hoc_analysis.level_1: {e}")


class TestUtilityModuleImports:
    """Test that utility modules can be imported."""

    def test_import_dlc_utils(self):
        """Test importing dlc_utils module."""
        try:
            import compass_labyrinth.behavior.pose_estimation.dlc_utils

            assert compass_labyrinth.behavior.pose_estimation.dlc_utils is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.pose_estimation.dlc_utils: {e}")

    def test_import_grid_utils(self):
        """Test importing grid_utils module."""
        try:
            import compass_labyrinth.behavior.pose_estimation.grid_utils

            assert compass_labyrinth.behavior.pose_estimation.grid_utils is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.pose_estimation.grid_utils: {e}")

    def test_import_preprocessing_utils(self):
        """Test importing preprocessing_utils module."""
        try:
            import compass_labyrinth.behavior.preprocessing.preprocessing_utils

            assert compass_labyrinth.behavior.preprocessing.preprocessing_utils is not None
        except ImportError as e:
            pytest.fail(f"Failed to import compass_labyrinth.behavior.preprocessing.preprocessing_utils: {e}")


class TestPackageDataImports:
    """Test that package data can be imported and loaded correctly."""

    def test_import_adjacency_matrix(self):
        """Test importing ADJACENCY_MATRIX from package data."""
        try:
            from compass_labyrinth.constants import ADJACENCY_MATRIX
            import pandas as pd

            assert ADJACENCY_MATRIX is not None
            assert isinstance(ADJACENCY_MATRIX, pd.DataFrame)
            assert ADJACENCY_MATRIX.shape == (144, 144), f"Expected shape (144, 144), got {ADJACENCY_MATRIX.shape}"
        except ImportError as e:
            pytest.fail(f"Failed to import ADJACENCY_MATRIX: {e}")
        except AssertionError as e:
            pytest.fail(f"ADJACENCY_MATRIX validation failed: {e}")

    def test_import_value_function(self):
        """Test importing VALUE_FUNCTION from package data."""
        try:
            from compass_labyrinth.constants import VALUE_FUNCTION
            import pandas as pd

            assert VALUE_FUNCTION is not None
            assert isinstance(VALUE_FUNCTION, pd.DataFrame)
            assert VALUE_FUNCTION.shape == (144, 2), f"Expected shape (144, 2), got {VALUE_FUNCTION.shape}"

            # Check column names
            expected_columns = ["Grid Number", "Value"]
            assert (
                list(VALUE_FUNCTION.columns) == expected_columns
            ), f"Expected columns {expected_columns}, got {list(VALUE_FUNCTION.columns)}"
        except ImportError as e:
            pytest.fail(f"Failed to import VALUE_FUNCTION: {e}")
        except AssertionError as e:
            pytest.fail(f"VALUE_FUNCTION validation failed: {e}")
