#!/usr/bin/env python3
"""Setup ML project with uv package manager.

This script initializes a new ML project with uv, configures dependencies,
and sets up development tools (ruff, mypy, pytest).
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv() -> bool:
    """Install uv package manager."""
    logger.info("Installing uv...")
    try:
        subprocess.run(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["sh"],
            input=subprocess.run(
                ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
                capture_output=True,
                check=True,
            ).stdout,
            check=True,
        )
        logger.info("✓ uv installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install uv: {e}")
        return False


def initialize_project(project_dir: Path, project_name: str) -> bool:
    """Initialize uv project."""
    logger.info(f"Initializing project: {project_name}")

    try:
        subprocess.run(
            ["uv", "init", "--name", project_name],
            cwd=project_dir,
            check=True,
        )
        logger.info("✓ Project initialized")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize project: {e}")
        return False


def add_ml_dependencies(project_dir: Path, cuda: bool = True) -> bool:
    """Add ML dependencies."""
    logger.info("Adding ML dependencies...")

    deps = [
        "torch>=2.1",
        "pytorch-lightning>=2.1",
        "hydra-core>=1.3",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
    ]

    try:
        # Add main dependencies
        if cuda:
            # Add PyTorch with CUDA
            subprocess.run(
                ["uv", "add", "torch", "--index-url",
                 "https://download.pytorch.org/whl/cu121"],
                cwd=project_dir,
                check=True,
            )
            deps.remove("torch>=2.1")

        subprocess.run(
            ["uv", "add"] + deps,
            cwd=project_dir,
            check=True,
        )

        logger.info("✓ ML dependencies added")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add dependencies: {e}")
        return False


def add_dev_dependencies(project_dir: Path) -> bool:
    """Add development dependencies."""
    logger.info("Adding development dependencies...")

    dev_deps = [
        "pytest>=7.0",
        "pytest-cov>=4.0",
        "ruff>=0.1",
        "mypy>=1.0",
        "pre-commit>=3.0",
    ]

    try:
        subprocess.run(
            ["uv", "add", "--dev"] + dev_deps,
            cwd=project_dir,
            check=True,
        )
        logger.info("✓ Development dependencies added")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to add dev dependencies: {e}")
        return False


def create_config_files(project_dir: Path) -> bool:
    """Create configuration files."""
    logger.info("Creating configuration files...")

    # ruff.toml
    ruff_config = """line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "LOG", "G"]
ignore = ["ANN101", "ANN102"]

[lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["ANN", "S101"]
"""

    (project_dir / "ruff.toml").write_text(ruff_config)
    logger.info("✓ Created ruff.toml")

    # Add pytest and mypy config to pyproject.toml
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "a") as f:
            f.write(
                """
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["--cov=src", "--cov-report=html", "--cov-report=term"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
"""
            )
        logger.info("✓ Updated pyproject.toml with pytest and mypy config")

    return True


def create_project_structure(project_dir: Path) -> bool:
    """Create ML project directory structure."""
    logger.info("Creating project structure...")

    directories = [
        "src/models",
        "src/data",
        "src/utils",
        "tests",
        "configs/model",
        "configs/data",
        "configs/trainer",
        "configs/logger",
        "configs/experiment",
        "notebooks",
        "scripts",
        "logs",
    ]

    for dir_path in directories:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/data/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
    ]

    for init_file in init_files:
        (project_dir / init_file).touch()

    logger.info("✓ Project structure created")
    return True


@app.command()
def setup(
    project_dir: Path = typer.Option(Path.cwd(), help="Project directory"),
    project_name: str = typer.Option("ml-project", help="Project name"),
    cuda: bool = typer.Option(True, help="Install PyTorch with CUDA support"),
    skip_install: bool = typer.Option(False, help="Skip uv installation check"),
) -> None:
    """Setup ML project with uv."""
    logger.info(f"Setting up ML project with uv: {project_name}")

    # Check/install uv
    if not skip_install and not check_uv_installed():
        logger.warning("uv not found, installing...")
        if not install_uv():
            logger.error("Failed to install uv")
            raise typer.Exit(1)

    # Initialize project
    if not initialize_project(project_dir, project_name):
        raise typer.Exit(1)

    # Add dependencies
    if not add_ml_dependencies(project_dir, cuda=cuda):
        raise typer.Exit(1)

    if not add_dev_dependencies(project_dir):
        raise typer.Exit(1)

    # Create configs
    if not create_config_files(project_dir):
        raise typer.Exit(1)

    # Create structure
    if not create_project_structure(project_dir):
        raise typer.Exit(1)

    # Sync dependencies
    logger.info("Installing dependencies...")
    try:
        subprocess.run(["uv", "sync"], cwd=project_dir, check=True)
        logger.info("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync dependencies: {e}")
        raise typer.Exit(1)

    logger.info("\n✓ ML project setup complete!")
    logger.info(f"  Project directory: {project_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. cd into project directory")
    logger.info("  2. Run: uv run python -c 'import torch; print(torch.cuda.is_available())'")
    logger.info("  3. Start developing your ML project!")


if __name__ == "__main__":
    app()
