#!/usr/bin/env python3
"""Setup ML project with uv package manager.

This script initializes a new ML project with uv, configures dependencies,
and sets up development tools (ruff, mypy, pytest).
"""

import logging
from pathlib import Path
import subprocess
from typing import Annotated

import typer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    else:
        return True


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
    except subprocess.CalledProcessError:
        logger.exception("Failed to install uv")
        return False
    else:
        logger.info("✓ uv installed successfully")
        return True


def initialize_project(project_dir: Path, project_name: str) -> bool:
    """Initialize uv project."""
    logger.info("Initializing project: %s", project_name)

    try:
        subprocess.run(
            ["uv", "init", "--name", project_name],
            cwd=project_dir,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("Failed to initialize project")
        return False
    else:
        logger.info("✓ Project initialized")
        return True


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
                ["uv", "add", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"],
                cwd=project_dir,
                check=True,
            )
            deps.remove("torch>=2.1")

        subprocess.run(
            ["uv", "add", *deps],
            cwd=project_dir,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("Failed to add dependencies")
        return False
    else:
        logger.info("✓ ML dependencies added")
        return True


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
            ["uv", "add", "--dev", *dev_deps],
            cwd=project_dir,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("Failed to add dev dependencies")
        return False
    else:
        logger.info("✓ Development dependencies added")
        return True


def create_config_files(project_dir: Path) -> bool:
    """Create configuration files."""
    logger.info("Creating configuration files...")

    # ruff.toml
    ruff_config = """line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "LOG", "G"]

[lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["ANN", "S101"]
"""

    (project_dir / "ruff.toml").write_text(ruff_config)
    logger.info("✓ Created ruff.toml")

    # Add pytest and mypy config to pyproject.toml
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with pyproject_path.open("a") as f:
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
    project_dir: Annotated[Path | None, typer.Option(help="Project directory")] = None,
    project_name: Annotated[str | None, typer.Option(help="Project name")] = None,
    cuda: Annotated[bool, typer.Option(help="Install PyTorch with CUDA support")] = True,
    skip_install: Annotated[bool, typer.Option(help="Skip uv installation check")] = False,
) -> None:
    """Setup ML project with uv."""
    # Set defaults within function to avoid B008
    if project_dir is None:
        project_dir = Path.cwd()
    if project_name is None:
        project_name = "ml-project"

    logger.info("Setting up ML project with uv: %s", project_name)

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
    except subprocess.CalledProcessError:
        logger.exception("Failed to sync dependencies")
        raise typer.Exit(1) from None
    else:
        logger.info("✓ Dependencies installed")

    logger.info("\n✓ ML project setup complete!")
    logger.info("  Project directory: %s", project_dir)
    logger.info("\nNext steps:")
    logger.info("  1. cd into project directory")
    logger.info("  2. Run: uv run python -c 'import torch; print(torch.cuda.is_available())'")
    logger.info("  3. Start developing your ML project!")


if __name__ == "__main__":
    app()
