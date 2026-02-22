#!/usr/bin/env python3
"""Experiment registry for tracking ML experiments."""

from datetime import datetime
import json
from pathlib import Path


def log_experiment(
    name: str,
    config_path: str,
    metrics: dict,
    hyperparameters: dict,
    status: str = "completed",
    tags: list[str] | None = None,
    notes: str = "",
    runtime: str = "",
    gpu_count: int = 1,
    registry_path: Path | None = None,
) -> str:
    """Log experiment to registry.

    Args:
        name: Experiment name
        config_path: Path to config file
        metrics: Dictionary of metrics (best_val_acc, best_val_loss, etc.)
        hyperparameters: Dictionary of hyperparameters
        status: Experiment status (running, completed, failed)
        tags: List of tags for organization
        notes: Additional notes
        runtime: Runtime string (e.g., "2h 34m")
        gpu_count: Number of GPUs used
        registry_path: Path to registry file (default: logs/experiment_registry.json)

    Returns:
        Experiment ID (e.g., "exp_001")
    """
    if registry_path is None:
        registry_path = Path("logs/experiment_registry.json")

    # Load existing registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry = {"experiments": []}

    # Generate experiment ID
    exp_id = f"exp_{len(registry['experiments']) + 1:03d}"

    # Create experiment entry
    experiment = {
        "id": exp_id,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "status": status,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "runtime": runtime,
        "gpu_count": gpu_count,
        "tags": tags or [],
        "notes": notes,
    }

    # Add to registry
    registry["experiments"].append(experiment)

    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"✅ Logged experiment {exp_id}: {name}")
    return exp_id


def get_experiment(exp_id: str, registry_path: Path | None = None) -> dict | None:
    """Get experiment by ID.

    Args:
        exp_id: Experiment ID
        registry_path: Path to registry file

    Returns:
        Experiment dictionary or None if not found
    """
    if registry_path is None:
        registry_path = Path("logs/experiment_registry.json")

    if not registry_path.exists():
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    for exp in registry["experiments"]:
        if exp["id"] == exp_id:
            return exp

    return None


def list_experiments(
    tags: list[str] | None = None,
    status: str | None = None,
    registry_path: Path | None = None,
) -> list[dict]:
    """List experiments with optional filtering.

    Args:
        tags: Filter by tags (any match)
        status: Filter by status
        registry_path: Path to registry file

    Returns:
        List of experiment dictionaries
    """
    if registry_path is None:
        registry_path = Path("logs/experiment_registry.json")

    if not registry_path.exists():
        return []

    with open(registry_path) as f:
        registry = json.load(f)

    experiments = registry["experiments"]

    # Filter by tags
    if tags:
        experiments = [
            exp for exp in experiments if any(tag in exp.get("tags", []) for tag in tags)
        ]

    # Filter by status
    if status:
        experiments = [exp for exp in experiments if exp.get("status") == status]

    return experiments


def update_experiment_status(exp_id: str, status: str, registry_path: Path | None = None) -> bool:
    """Update experiment status.

    Args:
        exp_id: Experiment ID
        status: New status
        registry_path: Path to registry file

    Returns:
        True if updated, False if experiment not found
    """
    if registry_path is None:
        registry_path = Path("logs/experiment_registry.json")

    if not registry_path.exists():
        return False

    with open(registry_path) as f:
        registry = json.load(f)

    # Find and update experiment
    for exp in registry["experiments"]:
        if exp["id"] == exp_id:
            exp["status"] = status
            exp["updated_at"] = datetime.now().isoformat()

            # Save registry
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            return True

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment registry utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.add_argument("--status", help="Filter by status")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get experiment by ID")
    get_parser.add_argument("exp_id", help="Experiment ID")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update experiment status")
    update_parser.add_argument("exp_id", help="Experiment ID")
    update_parser.add_argument("status", help="New status")

    args = parser.parse_args()

    if args.command == "list":
        experiments = list_experiments(tags=args.tags, status=args.status)
        for exp in experiments:
            print(
                f"{exp['id']}: {exp['name']} [{exp['status']}] - "
                f"Val Acc: {exp['metrics'].get('best_val_acc', 'N/A')}"
            )

    elif args.command == "get":
        exp = get_experiment(args.exp_id)
        if exp:
            print(json.dumps(exp, indent=2))
        else:
            print(f"Experiment {args.exp_id} not found")

    elif args.command == "update":
        if update_experiment_status(args.exp_id, args.status):
            print(f"Updated {args.exp_id} to {args.status}")
        else:
            print(f"Experiment {args.exp_id} not found")
