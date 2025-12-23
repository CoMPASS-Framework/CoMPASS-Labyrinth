# Documentation Development Guide

This directory contains the source files for the CoMPASS-Labyrinth documentation, built with MkDocs.

## Setup for Local Development

### 1. Install Documentation Dependencies

```bash
pip install -e ".[docs]"
```

This installs:
- `mkdocs` - Core documentation builder
- `mkdocs-material` - Modern Material Design theme
- `mkdocstrings[python]` - Auto-generate API docs from docstrings
- `mkdocs-jupyter` - Embed Jupyter notebooks in docs

### 2. Serve Documentation Locally

```bash
mkdocs serve --watch ./src --watch ./docs --watch ./mkdocs.yml
```

This starts a local development server at `http://127.0.0.1:8000` with live-reload functionality. Any changes you make to the documentation files will automatically refresh in your browser.

### 3. Build Static Documentation

```bash
mkdocs build
```

This generates the static HTML site in the `site/` directory.

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started.md          # Installation guide
├── tutorials.md                # Tutorial links
├── contributing.md             # Contributing guide
├── user-guide/
│   └── index.md               # User guide (placeholder)
├── framework/
│   └── index.md               # Framework overview (placeholder)
└── api/
    └── index.md               # API reference (placeholder)
```

## Adding API Documentation

To add auto-generated API documentation for a module, create a new markdown file in `docs/api/` with:

```markdown
# Module Name

::: module.path.here
    options:
      show_source: true
      docstring_style: numpy
```

For example, to document the `prep_data` module:

```markdown
# Data Preparation

::: compass_labyrinth.compass.level_1.prep_data
```

## Writing Documentation

### Markdown Extensions

The documentation supports several markdown extensions:

- **Admonitions**: `!!! note`, `!!! warning`, `!!! tip`, `!!! info`
- **Code highlighting**: Fenced code blocks with language specification
- **Tables**: Standard markdown tables
- **Math equations**: LaTeX-style math (if needed, add extension)

### Example Admonition

```markdown
!!! tip "Pro Tip"
    This is a helpful tip for users!
```

### Example Code Block

````markdown
```python
import compass_labyrinth
project = compass_labyrinth.init_project("my_project")
```
````

## Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the `.github/workflows/docs.yaml` workflow.

The deployed site will be available at:
`https://compass-framework.github.io/CoMPASS-Labyrinth/`

## Configuration

The documentation is configured in `mkdocs.yml` at the root of the repository. Key sections:

- `site_name`, `site_description` - Basic site metadata
- `nav` - Navigation structure
- `theme` - Material theme configuration
- `plugins` - mkdocstrings, search, jupyter support
- `markdown_extensions` - Enhanced markdown features

## Tips

1. **Keep pages focused**: Each page should cover one topic thoroughly
2. **Use cross-references**: Link between related pages
3. **Add examples**: Code examples make documentation easier to follow
4. **Update as you code**: Documentation is most accurate when updated alongside code changes
5. **Use docstrings**: Well-written docstrings enable automatic API documentation

## Need Help?

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
