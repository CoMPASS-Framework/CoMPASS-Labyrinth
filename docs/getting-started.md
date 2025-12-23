# Getting Started

This guide will help you install and set up CoMPASS-Labyrinth on your system.

## Prerequisites

- **Python**: 3.11 or higher

## Installation

You can install CoMPASS-Labyrinth directly with pip:

```bash
pip install compass-labyrinth
```

or from source for development:

```bash
git clone https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth.git
cd CoMPASS-Labyrinth

# Core dependencies only
pip install -e .

# Or with development tools
pip install -e ".[dev]"

# Or with documentation tools
pip install -e ".[docs]"

# Or install all optional dependencies
pip install -e ".[dev,tests,docs]"
```

You can also use Conda to manage your environment:

```bash
conda env create -f environment.yml
conda activate compass-labyrinth
```

## Next Steps

Now that you have CoMPASS-Labyrinth installed, you can:

- Explore the [Tutorials](tutorials.md) to learn how to use the framework
- Read the [User Guide](user-guide/index.md) for detailed usage instructions
- Check the [API Reference](api/index.md) for detailed documentation

For more help, please [open an issue](https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth/issues) on GitHub.
