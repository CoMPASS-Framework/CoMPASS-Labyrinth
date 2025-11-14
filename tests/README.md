# CoMPASS-Labyrinth Test Suite

This directory contains the test suite for the CoMPASS-Labyrinth project.

## Test Organization

- `00_init_project_test.py` - Tests for project initialization and the `test_project` fixture
- `01_data_preprocessing_test.py` - Tests for data preprocessing
- `03_performance_metrics_test.py` - Tests for assessing performance metrics
- `04_simulate_agent_test.py` - Tests simulation agents assessment
- `05_compass_level_1_test.py` - Tests for compass level-1 model
- `06_compass_posthoc_test.py` - Tests for compass level-1 post analysis
- `07_compass_level_2_test.py` - Tests for compass level-2 model
- `conftest.py` - Shared pytest fixtures

## Test Fixtures

### `test_project` Fixture

The `test_project` fixture in `conftest.py` provides a fully initialized test project for use in tests. This fixture:

**Setup:**
- Creates a temporary source data directory
- Copies test CSV files from `tests/assets/`
- Creates mock video files
- Copies metadata file
- Initializes a complete project using `init_project`

**Returns:**
A dictionary containing:
- `config`: The project configuration dict
- `project_path`: Path to the created project directory
- `source_data_path`: Path to the source data directory

**Usage Example:**
```python
def test_my_feature(test_project):
    """Test using the test_project fixture."""
    config = test_project['config']
    project_path = test_project['project_path']
    
    # Use the project for testing
    assert config['project_name'] == 'test_project'
    assert (project_path / 'config.yaml').exists()
```

**Features:**
- **Isolated**: Each test gets its own temporary project
- **Auto-cleanup**: Temporary directories are automatically removed after tests
- **Realistic**: Uses actual test data from `tests/assets/`
- **Reusable**: Can be used by any test that needs a project

## Test Assets

The `tests/assets/` directory contains:
- `Session-3withGrids.csv` - Test DLC output CSV
- `Session-4withGrids.csv` - Test DLC output CSV
- `WT_DSI_Labyrinth_Metadata.csv` - Test metadata file

## Running Tests

Run all tests:
```bash
conda run -n compass-labyrinth python -m pytest tests/ -v
```

Run specific test file:
```bash
conda run -n compass-labyrinth python -m pytest tests/03_init_project_test.py -v
```

Run specific test:
```bash
conda run -n compass-labyrinth python -m pytest tests/03_init_project_test.py::TestProjectFixture::test_fixture_creates_project -v
```

## Adding New Tests

When adding tests that need a project:

1. Import the fixture by adding `test_project` as a parameter to your test function
2. Access the project configuration and paths from the returned dictionary
3. Write your test logic using the initialized project

Example:
```python
def test_new_feature(test_project):
    """Test a new feature using the project fixture."""
    config = test_project['config']
    # Your test code here
    assert some_condition
