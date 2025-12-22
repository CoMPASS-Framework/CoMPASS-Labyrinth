# Contributing to CoMPASS-Labyrinth

Thank you for your interest in contributing to CoMPASS-Labyrinth! This document provides guidelines for contributing to the project.

## Getting Started

### Setting Up Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/CoMPASS-Labyrinth.git
   cd CoMPASS-Labyrinth
   ```

2. **Create a development environment**

   ```bash
   conda env create -f environment.yml
   conda activate compass-labyrinth
   ```

   Or with pip:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,tests,docs]"
   ```

## Development Workflow

### Code Style

We use `black` for Python code formatting:

```bash
black src/ tests/
```

Configuration is in `pyproject.toml` with a line length of 120.

### Running Tests

Before submitting changes, make sure all tests pass:

```bash
pytest tests/
```

To run tests with coverage:

```bash
pytest tests/ --cov=compass_labyrinth --cov-report=html
```

### Documentation

#### Building Documentation Locally

1. **Install documentation dependencies**

   ```bash
   pip install -e ".[docs]"
   ```

2. **Serve documentation locally**

   ```bash
   mkdocs serve --watch src/
   ```

   The documentation will be available at `http://127.0.0.1:8000`

3. **Build static documentation**

   ```bash
   mkdocs build
   ```

#### Adding to Documentation

- Documentation source files are in the `docs/` directory
- Edit markdown files directly
- API documentation is auto-generated from docstrings using mkdocstrings
- Use numpy-style docstrings in your code

### Submitting Changes

1. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, commented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**

   ```bash
   git add .
   git commit -m "Clear description of your changes"
   ```

4. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Describe your changes clearly
   - Reference any related issues

## Code Review Process

- All submissions require review before merging
- Reviewers will provide feedback and may request changes
- Once approved, maintainers will merge your PR

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant error messages and logs

Use the [GitHub issue tracker](https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth/issues).

## Feature Requests

We welcome feature requests! Please:

- Check if a similar request already exists
- Clearly describe the feature and its use case
- Explain how it would benefit the project

## Questions?

If you have questions about contributing:

- Open a [GitHub discussion](https://github.com/CoMPASS-Framework/CoMPASS-Labyrinth/discussions)
- Check existing documentation and tutorials
- Reach out to the maintainers

## License

By contributing to CoMPASS-Labyrinth, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project's documentation and release notes.

Thank you for helping make CoMPASS-Labyrinth better! 🎉
