#!/usr/bin/env python3
"""Setup ML project with pixi package manager.

This script initializes a new ML project with pixi for ML projects requiring
CUDA/GPU dependencies. Pixi uses conda channels for better ML package management.
"""

import logging
from pathlib import Path
import subprocess
from typing import Annotated

import typer


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


def check_pixi_installed() -> bool:
    """Check if pixi is installed."""
    try:
        subprocess.run(["pixi", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    else:
        return True


def install_pixi() -> bool:
    """Install pixi package manager."""
    logger.info("Installing pixi...")
    try:
        subprocess.run(
            ["curl", "-fsSL", "https://pixi.sh/install.sh"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["sh"],
            input=subprocess.run(
                ["curl", "-fsSL", "https://pixi.sh/install.sh"],
                capture_output=True,
                check=True,
            ).stdout,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("Failed to install pixi")
        return False
    else:
        logger.info("✓ pixi installed successfully")
        return True


def initialize_pixi_project(project_dir: Path, project_name: str) -> bool:
    """Initialize pixi project with pixi.toml."""
    logger.info("Initializing pixi project: %s", project_name)

    pixi_toml = f"""[project]
name = "{project_name}"
version = "0.1.0"
channels = ["pytorch", "nvidia", "conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
python = "3.10.*"
pytorch = ">=2.1"
pytorch-cuda = "12.1.*"
pytorch-lightning = ">=2.1"
hydra-core = ">=1.3"
numpy = "*"
pandas = "*"
matplotlib = "*"
scikit-learn = "*"

[feature.dev.dependencies]
pytest = "*"
pytest-cov = "*"
ruff = "*"
mypy = "*"
jupyterlab = "*"

[tasks]
train = "python src/train.py"
test = "pytest tests/"
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "mypy src/"
"""

    (project_dir / "pixi.toml").write_text(pixi_toml)
    logger.info("✓ Created pixi.toml")
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

    # pyproject.toml for mypy and pytest config
    pyproject_toml = """[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["--cov=src", "--cov-report=html"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
"""

    (project_dir / "pyproject.toml").write_text(pyproject_toml)
    logger.info("✓ Created pyproject.toml")

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
    skip_install: Annotated[bool, typer.Option(help="Skip pixi installation check")] = False,
) -> None:
    """Setup ML project with pixi."""
    # Set defaults within function to avoid B008
    if project_dir is None:
        project_dir = Path.cwd()
    if project_name is None:
        project_name = "ml-project"

    logger.info("Setting up ML project with pixi: %s", project_name)

    # Check/install pixi
    if not skip_install and not check_pixi_installed():
        logger.warning("pixi not found, installing...")
        if not install_pixi():
            logger.error("Failed to install pixi")
            raise typer.Exit(1)

    # Initialize project
    if not initialize_pixi_project(project_dir, project_name):
        raise typer.Exit(1)

    # Create configs
    if not create_config_files(project_dir):
        raise typer.Exit(1)

    # Create structure
    if not create_project_structure(project_dir):
        raise typer.Exit(1)

    # Install dependencies
    logger.info("Installing dependencies with pixi...")
    try:
        subprocess.run(["pixi", "install"], cwd=project_dir, check=True)
    except subprocess.CalledProcessError:
        logger.exception("Failed to install dependencies")
        raise typer.Exit(1) from None
    else:
        logger.info("✓ Dependencies installed")

    logger.info("\n✓ ML project setup complete!")
    logger.info("  Project directory: %s", project_dir)
    logger.info("\nNext steps:")
    logger.info("  1. cd into project directory")
    logger.info("  2. Run: pixi run python -c 'import torch; print(torch.cuda.is_available())'")
    logger.info("  3. Activate environment: pixi shell")
    logger.info("  4. Start developing your ML project!")


if __name__ == "__main__":
    app()
