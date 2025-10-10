"""Shared fixtures for tests."""

import pytest
from pathlib import Path
import shutil


@pytest.fixture(scope="session")
def test_create_project(tmp_path_factory):
    """
    Session-scoped fixture that creates a test project using init_project.
    
    This fixture runs ONCE per test session and is shared across all tests.
    This is efficient for read-only tests but means tests should not modify
    the project state.
    
    This fixture:
    - Creates a temporary source data directory with test CSV files
    - Copies metadata from tests/assets/
    - Initializes a project using init_project
    - Returns project configuration and paths for testing
    
    The project is created in a temporary directory that is automatically
    cleaned up after all tests complete.
    
    Returns
    -------
    dict
        Project's configuration dict from init_project
    pd.DataFrame
        Cohort metadata DataFrame from init_project
    """
    from compass_labyrinth import init_project
    
    # Create temporary directory using tmp_path_factory for session scope
    tmp_path = tmp_path_factory.mktemp("test_session")
    
    # Define paths
    assets_dir = Path(__file__).parent / "assets"
    source_data_path = tmp_path / "source_data"
    source_data_path.mkdir()
    
    # Copy CSV files from assets to source directory
    csv_files = [
        "Session0003_withGrids.csv",
        "Session0004_withGrids.csv",
    ]
    
    for csv_file in csv_files:
        src = assets_dir / csv_file
        dst = source_data_path / csv_file
        shutil.copy2(src, dst)

    # Copy metadata file
    metadata_src = assets_dir / "WT_DSI_Labyrinth_Metadata.csv"
    metadata_dst = source_data_path / "WT_DSI_Labyrinth_Metadata.csv"
    shutil.copy2(metadata_src, metadata_dst)
    
    # # Create mock video files (empty files with .mp4 extension)
    # video_files = [
    #     "Session0003.mp4",
    #     "Session0004.mp4",
    # ]
    
    # for video_file in video_files:
    #     video_path = source_data_path / video_file
    #     video_path.touch()  # Create empty file
    
    # Initialize the project
    config, metadata_df = init_project(
        project_name="test_project",
        project_path=tmp_path,
        source_data_path=source_data_path,
        user_metadata_file_path=metadata_dst,
        trial_type="Labyrinth_DSI",
        file_ext=".csv",
        video_type=".mp4",
        dlc_scorer="DLC_resnet50_LabyrinthMar13shuffle1_1000000",
    )

    yield config, metadata_df
