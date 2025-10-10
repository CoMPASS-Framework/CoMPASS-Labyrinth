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

    def test_load_cohort_metadata_exists(self):
        """Test that load_cohort_metadata function exists."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        assert hasattr(dlc_utils, 'load_cohort_metadata')
        assert callable(dlc_utils.load_cohort_metadata)

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


class TestDLCUtilsFunctionSignatures:
    """Test that dlc_utils functions have expected signatures."""

    def test_load_cohort_metadata_signature(self):
        """Test load_cohort_metadata function signature."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        sig = inspect.signature(dlc_utils.load_cohort_metadata)
        params = list(sig.parameters.keys())
        
        assert 'metadata_path' in params, "Missing 'metadata_path' parameter"
        assert 'trial_sheet_name' in params, "Missing 'trial_sheet_name' parameter"

    def test_validate_metadata_signature(self):
        """Test validate_metadata function signature."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        sig = inspect.signature(dlc_utils.validate_metadata)
        params = list(sig.parameters.keys())
        
        assert 'df' in params, "Missing 'df' parameter"

    def test_create_organized_directory_structure_signature(self):
        """Test create_organized_directory_structure function signature."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        sig = inspect.signature(dlc_utils.create_organized_directory_structure)
        params = list(sig.parameters.keys())
        
        assert 'base_path' in params, "Missing 'base_path' parameter"

    def test_batch_save_first_frames_signature(self):
        """Test batch_save_first_frames function signature."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        sig = inspect.signature(dlc_utils.batch_save_first_frames)
        params = list(sig.parameters.keys())
        
        assert 'mouseinfo_df' in params, "Missing 'mouseinfo_df' parameter"
        assert 'video_directory' in params, "Missing 'video_directory' parameter"
        assert 'frames_directory' in params, "Missing 'frames_directory' parameter"

    def test_get_grid_coordinates_signature(self):
        """Test get_grid_coordinates function signature."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        sig = inspect.signature(dlc_utils.get_grid_coordinates)
        params = list(sig.parameters.keys())
        
        assert 'posList' in params, "Missing 'posList' parameter"
        assert 'num_squares' in params, "Missing 'num_squares' parameter"
        assert 'grid_files_directory' in params, "Missing 'grid_files_directory' parameter"
        assert 'session' in params, "Missing 'session' parameter"


class TestDLCUtilsDocstrings:
    """Test that dlc_utils functions have docstrings."""

    def test_load_cohort_metadata_has_docstring(self):
        """Test that load_cohort_metadata has a docstring."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        assert dlc_utils.load_cohort_metadata.__doc__ is not None
        assert len(dlc_utils.load_cohort_metadata.__doc__.strip()) > 0

    def test_validate_metadata_has_docstring(self):
        """Test that validate_metadata has a docstring."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        assert dlc_utils.validate_metadata.__doc__ is not None
        assert len(dlc_utils.validate_metadata.__doc__.strip()) > 0

    def test_create_organized_directory_structure_has_docstring(self):
        """Test that create_organized_directory_structure has a docstring."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        assert dlc_utils.create_organized_directory_structure.__doc__ is not None
        assert len(dlc_utils.create_organized_directory_structure.__doc__.strip()) > 0


class TestDLCUtilsModuleMetadata:
    """Test dlc_utils module metadata and structure."""

    def test_module_has_docstring(self):
        """Test that the module itself has a docstring."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        assert dlc_utils.__doc__ is not None
        assert len(dlc_utils.__doc__.strip()) > 0
        assert "DeepLabCut" in dlc_utils.__doc__ or "DLC" in dlc_utils.__doc__

    def test_module_author_info(self):
        """Test that module contains author information."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        # Check if docstring contains author info
        assert "Author" in dlc_utils.__doc__ or "author" in dlc_utils.__doc__.lower()

    def test_count_exported_functions(self):
        """Test that module exports expected number of functions."""
        from compass_labyrinth.behavior.pose_estimation import dlc_utils
        
        # Count public functions (not starting with _)
        public_functions = [
            name for name in dir(dlc_utils)
            if callable(getattr(dlc_utils, name)) and not name.startswith('_')
        ]
        
        # Should have a reasonable number of utility functions
        assert len(public_functions) >= 10, (
            f"Expected at least 10 public functions, found {len(public_functions)}"
        )


class TestDLCUtilsImportDirect:
    """Test direct import of specific functions."""

    def test_direct_import_load_cohort_metadata(self):
        """Test direct import of load_cohort_metadata."""
        try:
            from compass_labyrinth.behavior.pose_estimation.dlc_utils import load_cohort_metadata
            assert load_cohort_metadata is not None
            assert callable(load_cohort_metadata)
        except ImportError as e:
            pytest.fail(f"Failed to directly import load_cohort_metadata: {e}")

    def test_direct_import_validate_metadata(self):
        """Test direct import of validate_metadata."""
        try:
            from compass_labyrinth.behavior.pose_estimation.dlc_utils import validate_metadata
            assert validate_metadata is not None
            assert callable(validate_metadata)
        except ImportError as e:
            pytest.fail(f"Failed to directly import validate_metadata: {e}")

    def test_direct_import_create_organized_directory_structure(self):
        """Test direct import of create_organized_directory_structure."""
        try:
            from compass_labyrinth.behavior.pose_estimation.dlc_utils import (
                create_organized_directory_structure
            )
            assert create_organized_directory_structure is not None
            assert callable(create_organized_directory_structure)
        except ImportError as e:
            pytest.fail(f"Failed to directly import create_organized_directory_structure: {e}")

    def test_direct_import_batch_save_first_frames(self):
        """Test direct import of batch_save_first_frames."""
        try:
            from compass_labyrinth.behavior.pose_estimation.dlc_utils import batch_save_first_frames
            assert batch_save_first_frames is not None
            assert callable(batch_save_first_frames)
        except ImportError as e:
            pytest.fail(f"Failed to directly import batch_save_first_frames: {e}")

    def test_direct_import_get_grid_coordinates(self):
        """Test direct import of get_grid_coordinates."""
        try:
            from compass_labyrinth.behavior.pose_estimation.dlc_utils import get_grid_coordinates
            assert get_grid_coordinates is not None
            assert callable(get_grid_coordinates)
        except ImportError as e:
            pytest.fail(f"Failed to directly import get_grid_coordinates: {e}")
