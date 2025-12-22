# Getting Started

This guide will help you install and set up CoMPASS-Labyrinth on your system.

## Prerequisites

- **Python**: 3.11 or higher
- **R**: version 4.4.0 or lower

## Installation

You can install CoMPASS-Labyrinth using either Conda or pip.

### Option A: Using Conda (Recommended)

1. **Clone the repository**

   ```bash
   git clone https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth.git
   cd CoMPASS-Labyrinth
   ```

2. **Create Python environment**

   ```bash
   conda env create -f environment.yml
   conda activate compass-labyrinth
   ```

   This automatically installs the package with all dependencies from `pyproject.toml`.

3. **Initialize R environment**

   ```bash
   Rscript R/init_renv.R
   ```

   This installs all required R packages using renv. First run may take several minutes.

### Option B: Using pip

1. **Clone the repository**

   ```bash
   git clone https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth.git
   cd CoMPASS-Labyrinth
   ```

2. **Create Python virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package with dependencies**

   ```bash
   # Core dependencies only
   pip install -e .
   
   # Or with development tools (includes Jupyter)
   pip install -e ".[dev]"
   
   # Or with documentation tools
   pip install -e ".[docs]"
   
   # Or install all optional dependencies
   pip install -e ".[dev,tests,docs]"
   ```

4. **Initialize R environment**

   ```bash
   Rscript R/init_renv.R
   ```

## Verify Installation

To verify that the installation was successful, you can run:

```python
import compass_labyrinth
print(compass_labyrinth.__version__)
```

Or run the test suite:

```bash
pytest tests/
```

## Next Steps

Now that you have CoMPASS-Labyrinth installed, you can:

- Explore the [Tutorials](tutorials.md) to learn how to use the framework
- Read the [User Guide](user-guide/index.md) for detailed usage instructions
- Check the [API Reference](api/index.md) for detailed documentation

## Dependency Management Notes

- **Python packages**: Managed via `pyproject.toml`
- **R packages**: Managed via `renv`
- When you start R in this directory, renv will automatically activate

## Troubleshooting

!!! warning "Common Issues"
    - **R version compatibility**: Make sure you're using R version 4.4.0 or lower
    - **Python version**: Requires Python 3.11 or higher
    - **Missing dependencies**: If you encounter import errors, try reinstalling with `pip install -e ".[dev]"`

For more help, please [open an issue](https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth/issues) on GitHub.
