import pytest
from pathlib import Path
import pandas as pd


class TestPreprocessing:

    def test_compile_sessions(self, compiled_sessions_df):
        assert isinstance(compiled_sessions_df, pd.DataFrame)
        assert not compiled_sessions_df.empty

    def test_preprocess_sessions(self, preprocessed_sessions_df):
        assert isinstance(preprocessed_sessions_df, pd.DataFrame)
        assert not preprocessed_sessions_df.empty

    def test_velocity_column(self, velocity_column_df):
        assert isinstance(velocity_column_df, pd.DataFrame)
        assert not velocity_column_df.empty
        assert 'Velocity' in velocity_column_df.columns
        assert velocity_column_df['Velocity'].dtype in [float, int]

    def test_save_preprocessed_to_csv(self, create_project_fixture, save_preprocessed_data):
        assert save_preprocessed_data
        config, _ = create_project_fixture
        project_path = Path(config['project_path_full'])
        csv_dir = project_path / 'csvs'
        combined_dir = csv_dir / 'combined'
        individual_dir = csv_dir / 'individual'

        assert csv_dir.exists()
        assert combined_dir.exists()
        assert individual_dir.exists()

        individual_files = list(individual_dir.glob('*.csv'))
        assert len(individual_files) > 0

        combined_file = combined_dir / 'Preprocessed_combined_file.csv'
        assert combined_file.exists()
    