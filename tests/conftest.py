"""Shared fixtures for tests."""

import pytest
import shutil
from pathlib import Path
import pandas as pd


@pytest.fixture(scope="session")
def create_project_fixture(tmp_path_factory):
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
        "Session-3withGrids.csv",
        "Session-4withGrids.csv",
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


@pytest.fixture(scope="session")
def compiled_sessions_df(create_project_fixture):
    """
    Provides compiled session data.
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all sessions
    """
    from compass_labyrinth.behavior.preprocessing import compile_mouse_sessions
    
    config, cohort_metadata = create_project_fixture
    df_comb = compile_mouse_sessions(
        config=config,
        bp='sternum',
    )
    return df_comb


@pytest.fixture(scope="session")
def preprocessed_sessions_df(compiled_sessions_df):
    """
    Provides preprocessed session data.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with all sessions
    """
    from compass_labyrinth.behavior.preprocessing import preprocess_sessions

    df_all_csv = preprocess_sessions(df_comb=compiled_sessions_df)

    return df_all_csv


@pytest.fixture(scope="session")
def velocity_column_df(preprocessed_sessions_df):
    """
    Provides preprocessed session data with velocity column.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with velocity column added
    """
    from compass_labyrinth.behavior.preprocessing import ensure_velocity_column

    df_with_velocity = ensure_velocity_column(
        df=preprocessed_sessions_df,
        fps=5,
    )

    return df_with_velocity


@pytest.fixture(scope="session")
def save_preprocessed_data(create_project_fixture, velocity_column_df):
    """
    Saves preprocessed data to CSV files.
    
    Returns
    -------
    None
    """
    from compass_labyrinth.behavior.preprocessing import save_preprocessed_to_csv

    config, _ = create_project_fixture
    save_preprocessed_to_csv(config=config, df=velocity_column_df)

    return True


@pytest.fixture(scope="session")
def create_time_binned_dict(create_project_fixture, save_preprocessed_data):
    """
    Creates time-binned data dictionary.
    
    Returns
    -------
    dict
        Dictionary with time-binned DataFrames for each session
    """
    from compass_labyrinth.behavior.behavior_metrics.task_performance_analysis import generate_region_heatmap_pivots

    config, cohort_metadata = create_project_fixture
        
    # Import combined CSV
    filepath = Path(config["project_path_full"]) / "csvs" / "combined" / "Preprocessed_combined_file.csv"
    df_all_csv = pd.read_csv(filepath)

    pivot_dict = generate_region_heatmap_pivots(
        df=df_all_csv,
        lower_lim=0,        # Start of time window
        upper_lim=80000,    # End of time window
        difference=10000,   # Bin width in timepoints
    )

    return pivot_dict