"""
Test suite for dlc_utils module functions.

Tests the availability and basic functionality of functions from
compass_labyrinth.behavior.pose_estimation.dlc_utils module.
"""

import pytest
import inspect
from pathlib import Path


class TestDLCUtilsImports:
    """Test that dlc_utils functions can be imported."""

    def test_import_dlc_utils_module(self):
        """Test that the dlc_utils module can be imported."""
        try:
            from compass_labyrinth.behavior.pose_estimation import dlc_utils
            assert dlc_utils is not None
        except ImportError as e:
            pytest.fail(f"Failed to import dlc_utils: {e}")

    def test_import_cohort_metadata_exists(self):
        """Test that import_cohort_metadata function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'import_cohort_metadata')
        assert callable(dlc_utils.import_cohort_metadata)

    def test_validate_metadata_exists(self):
        """Test that validate_metadata function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'validate_metadata')
        assert callable(dlc_utils.validate_metadata)

    def test_display_metadata_summary_exists(self):
        """Test that display_metadata_summary function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'display_metadata_summary')
        assert callable(dlc_utils.display_metadata_summary)

    def test_create_organized_directory_structure_exists(self):
        """Test that create_organized_directory_structure function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'create_organized_directory_structure')
        assert callable(dlc_utils.create_organized_directory_structure)

    def test_batch_save_first_frames_exists(self):
        """Test that batch_save_first_frames function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'batch_save_first_frames')
        assert callable(dlc_utils.batch_save_first_frames)

    def test_get_labyrinth_boundary_and_cropping_exists(self):
        """Test that get_labyrinth_boundary_and_cropping function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'get_labyrinth_boundary_and_cropping')
        assert callable(dlc_utils.get_labyrinth_boundary_and_cropping)

    def test_get_grid_coordinates_exists(self):
        """Test that get_grid_coordinates function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'get_grid_coordinates')
        assert callable(dlc_utils.get_grid_coordinates)

    def test_batch_create_grids_exists(self):
        """Test that batch_create_grids function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'batch_create_grids')
        assert callable(dlc_utils.batch_create_grids)
