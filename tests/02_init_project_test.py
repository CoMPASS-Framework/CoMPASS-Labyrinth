"""
Test suite for init_project function and test_project fixture.

This module demonstrates the usage of the test_project fixture
and validates the init_project function behavior.
"""

import pytest
from pathlib import Path


class TestInitProject:

    def test_fixture_creates_project(self, test_create_project):
        """Test that the fixture successfully creates a project."""
        assert isinstance(test_create_project, dict)

    def test_project_path_exists(self, test_create_project):
        """Test that the project path exists."""
        project_path = Path(test_create_project['project_path_full'])
        assert project_path.exists()
        assert project_path.is_dir()

    def test_config_yaml_created(self, test_create_project):
        """Test that config.yaml file is created."""
        config_path = Path(test_create_project['project_path_full']) / 'config.yaml'
        assert config_path.exists()
        assert config_path.is_file()

    def test_config_contents(self, test_create_project):
        """Test that config contains expected fields."""
        config = test_create_project
        
        # Check required fields
        assert 'project_name' in config
        assert 'project_path_full' in config
        assert 'creation_date_time' in config
        assert 'trial_type' in config
        assert 'session_names' in config
        assert 'bodyparts' in config


class TestLoadProject:

    def test_load_project_function(self, test_create_project):
        """Test that load_project can load the created project."""
        from compass_labyrinth.utils import load_project

        project_path = test_create_project['project_path_full']
        loaded_config = load_project(project_path)
        
        assert isinstance(loaded_config, dict)

    def test_loaded_config_matches(self, test_create_project):
        """Test that loaded config matches the original config."""
        from compass_labyrinth.utils import load_project
        
        original_config = test_create_project
        project_path = test_create_project['project_path_full']
        loaded_config = load_project(project_path)
        
        # Check key fields match
        assert loaded_config['project_name'] == original_config['project_name']
        assert loaded_config['trial_type'] == original_config['trial_type']
        assert loaded_config['session_names'] == original_config['session_names']
