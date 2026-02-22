#!/usr/bin/env python3
"""Comprehensive ML project validation."""

import argparse
import importlib
import logging
from pathlib import Path
import subprocess
import sys


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ProjectValidator:
    """Validate ML project structure and configuration."""

    def __init__(self, project_dir: Path = Path.cwd()):
        self.project_dir = project_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting ML project validation...")

        self.check_structure()
        self.check_configs()
        self.check_code_quality()
        self.check_dependencies()
        self.check_gpu()

        # Report results
        if self.warnings:
            logger.warning(f"Found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")

        if self.errors:
            logger.error(f"Found {len(self.errors)} error(s):")
            for error in self.errors:
                logger.error(f"  ❌ {error}")
            return False

        logger.info("✅ All validation checks passed!")
        return True

    def check_structure(self):
        """Check project structure."""
        logger.info("Checking project structure...")

        # Required directories
        required_dirs = ["src", "configs"]
        for dir_name in required_dirs:
            dir_path = self.project_dir / dir_name
            if not dir_path.exists():
                self.errors.append(f"Missing directory: {dir_name}/")
            else:
                logger.info(f"  ✓ Found {dir_name}/")

        # Recommended directories
        recommended_dirs = ["tests", "src/models", "src/data"]
        for dir_name in recommended_dirs:
            dir_path = self.project_dir / dir_name
            if not dir_path.exists():
                self.warnings.append(f"Missing recommended directory: {dir_name}/")
            else:
                logger.info(f"  ✓ Found {dir_name}/")

        # Required files
        required_files = ["src/train.py", "configs/config.yaml"]
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                self.errors.append(f"Missing file: {file_path}")
            else:
                logger.info(f"  ✓ Found {file_path}")

        # Package manager
        has_pyproject = (self.project_dir / "pyproject.toml").exists()
        has_pixi = (self.project_dir / "pixi.toml").exists()
        if not (has_pyproject or has_pixi):
            self.warnings.append("No package manager config found (pyproject.toml or pixi.toml)")
        else:
            pkg_manager = "pyproject.toml" if has_pyproject else "pixi.toml"
            logger.info(f"  ✓ Found {pkg_manager}")

    def check_configs(self):
        """Validate configuration files."""
        logger.info("Validating configuration files...")

        configs_dir = self.project_dir / "configs"
        if not configs_dir.exists():
            self.errors.append("configs/ directory not found")
            return

        # Check YAML syntax
        import yaml

        yaml_count = 0
        for yaml_file in configs_dir.rglob("*.yaml"):
            yaml_count += 1
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
                logger.info(f"  ✓ {yaml_file.relative_to(self.project_dir)}")
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {yaml_file}: {e}")

        if yaml_count == 0:
            self.warnings.append("No YAML files found in configs/")

        # Test Hydra config composition
        try:
            from hydra import compose, initialize_config_dir

            config_dir = str((self.project_dir / "configs").absolute())
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name="config")

            logger.info("  ✓ Config composition successful")

            # Check required fields
            required_fields = ["model", "data", "trainer"]
            for field in required_fields:
                if field not in cfg:
                    self.warnings.append(f"Missing recommended config field: {field}")
                else:
                    logger.info(f"  ✓ Config has {field}")

        except Exception as e:
            self.errors.append(f"Config composition failed: {e}")

    def check_code_quality(self):
        """Check code quality with ruff."""
        logger.info("Checking code quality...")

        # Check if ruff is available
        try:
            result = subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.warnings.append("Ruff not available")
                return
        except FileNotFoundError:
            self.warnings.append("Ruff not installed (install with: uv add ruff)")
            return

        # Run ruff check
        src_paths = []
        if (self.project_dir / "src").exists():
            src_paths.append(str(self.project_dir / "src"))
        if (self.project_dir / "tests").exists():
            src_paths.append(str(self.project_dir / "tests"))

        if not src_paths:
            self.warnings.append("No source directories to lint")
            return

        result = subprocess.run(
            ["ruff", "check", *src_paths],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("  ✓ Ruff checks passed")
        else:
            self.warnings.append("Ruff found code quality issues (run: ruff check --fix)")

    def check_dependencies(self):
        """Check required dependencies."""
        logger.info("Checking dependencies...")

        required = [
            ("torch", "PyTorch"),
            ("pytorch_lightning", "PyTorch Lightning"),
            ("hydra", "Hydra"),
        ]

        optional = [
            ("wandb", "Weights & Biases"),
            ("tensorboard", "TensorBoard"),
            ("torch_geometric", "PyTorch Geometric"),
        ]

        # Check required
        for module_name, display_name in required:
            try:
                mod = importlib.import_module(module_name.replace("-", "_"))
                version = getattr(mod, "__version__", "unknown")
                logger.info(f"  ✓ {display_name} {version}")
            except ImportError:
                self.errors.append(f"Required package not installed: {display_name}")

        # Check optional
        for module_name, display_name in optional:
            try:
                mod = importlib.import_module(module_name.replace("-", "_"))
                version = getattr(mod, "__version__", "unknown")
                logger.info(f"  ✓ {display_name} {version} (optional)")
            except ImportError:
                pass  # Optional packages don't trigger warnings

    def check_gpu(self):
        """Check GPU availability."""
        logger.info("Checking GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"  ✓ CUDA available: {torch.version.cuda}")
                logger.info(f"  ✓ GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"    GPU {i}: {gpu_name}")
            else:
                self.warnings.append("No GPU detected (CPU training will be slow)")
        except ImportError:
            self.errors.append("PyTorch not installed, cannot check GPU")


def main() -> int:
    """Run validation."""
    parser = argparse.ArgumentParser(description="Validate ML project structure and configuration")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project directory to validate (default: current directory)",
    )

    args = parser.parse_args()

    validator = ProjectValidator(args.project_dir)
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
