#!/usr/bin/env python
"""
Utility script to convert all Jupyter notebooks in /tutorials to markdown files in /docs/tutorials.

This script uses jupyter nbconvert to convert .ipynb files to markdown format with embedded images
and no code prompts.
"""

import os
import subprocess
from pathlib import Path


def convert_notebooks():
    """Convert all Jupyter notebooks from tutorials/ to markdown in docs/tutorials/."""
    
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    tutorials_dir = project_root / "tutorials"
    output_dir = project_root / "docs" / "tutorials"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .ipynb files in tutorials directory
    notebooks = sorted(tutorials_dir.glob("*.ipynb"))
    
    if not notebooks:
        print(f"No notebooks found in {tutorials_dir}")
        return
    
    print(f"Found {len(notebooks)} notebook(s) to convert:\n")
    
    # Track conversion results
    successful = []
    failed = []
    
    # Convert each notebook
    for notebook in notebooks:
        print(f"Converting: {notebook.name}...")
        
        # Build the jupyter nbconvert command
        cmd = [
            "jupyter",
            "nbconvert",
            "--to", "markdown",
            str(notebook),
            "--embed-images",
            "--no-prompt",
            f"--output-dir={output_dir}"
        ]
        
        try:
            # Run the conversion command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  ✓ Successfully converted {notebook.name}")
            successful.append(notebook.name)
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to convert {notebook.name}")
            print(f"    Error: {e.stderr}")
            failed.append(notebook.name)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total notebooks: {len(notebooks)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nMarkdown files saved to: {output_dir}")
    
    if failed:
        print(f"\nFailed conversions:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    convert_notebooks()
