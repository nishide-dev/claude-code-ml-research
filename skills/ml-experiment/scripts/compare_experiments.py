#!/usr/bin/env python3
"""Compare multiple ML experiments side by side."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compare_experiments(exp_ids: list[str], registry_path: Path | None = None):
    """Compare multiple experiments side by side.

    Args:
        exp_ids: List of experiment IDs to compare
        registry_path: Path to registry file
    """
    if registry_path is None:
        registry_path = Path("logs/experiment_registry.json")

    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        return

    # Load registry
    with open(registry_path) as f:
        registry = json.load(f)

    # Get experiments
    experiments = [e for e in registry["experiments"] if e["id"] in exp_ids]

    if not experiments:
        print("No experiments found with given IDs")
        return

    # Create comparison DataFrame
    data = []
    for exp in experiments:
        metrics = exp.get("metrics", {})
        hparams = exp.get("hyperparameters", {})

        row = {
            "ID": exp["id"],
            "Name": exp["name"],
            "Status": exp.get("status", "unknown"),
            "Val Acc": metrics.get("best_val_acc", float("nan")),
            "Val Loss": metrics.get("best_val_loss", float("nan")),
            "Train Loss": metrics.get("final_train_loss", float("nan")),
            "Epochs": metrics.get("epochs_trained", "N/A"),
            "LR": hparams.get("lr", "N/A"),
            "Batch Size": hparams.get("batch_size", "N/A"),
            "Optimizer": hparams.get("optimizer", "N/A"),
            "Runtime": exp.get("runtime", "N/A"),
            "GPUs": exp.get("gpu_count", "N/A"),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Print comparison table
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # Find best experiment
    if not df["Val Acc"].isna().all():
        best_idx = df["Val Acc"].idxmax()
        best_exp = df.loc[best_idx]
        print(f"\n✨ Best experiment: {best_exp['ID']} - {best_exp['Name']}")
        print(f"   Val Acc: {best_exp['Val Acc']:.4f}")
        print(f"   Val Loss: {best_exp['Val Loss']:.4f}")

    # Plot comparison
    plot_comparison(df, registry_path.parent / "experiment_comparison.png")


def plot_comparison(df: pd.DataFrame, output_path: Path):
    """Plot experiment comparison.

    Args:
        df: Comparison DataFrame
        output_path: Output path for plot
    """
    # Filter out NaN values for plotting
    df_plot = df[~df["Val Acc"].isna()].copy()

    if len(df_plot) == 0:
        print("No valid metrics to plot")
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Validation Accuracy
    ax = axes[0, 0]
    colors = ["#2ecc71" if i == df_plot["Val Acc"].idxmax() else "#3498db" for i in df_plot.index]
    ax.bar(df_plot["Name"], df_plot["Val Acc"], color=colors)
    ax.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # 2. Validation Loss
    ax = axes[0, 1]
    colors = ["#2ecc71" if i == df_plot["Val Loss"].idxmin() else "#3498db" for i in df_plot.index]
    ax.bar(df_plot["Name"], df_plot["Val Loss"], color=colors)
    ax.set_title("Validation Loss", fontsize=14, fontweight="bold")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # 3. Learning Rate vs Accuracy
    ax = axes[1, 0]
    df_lr = df_plot[df_plot["LR"] != "N/A"].copy()
    if len(df_lr) > 0:
        df_lr["LR"] = pd.to_numeric(df_lr["LR"])
        ax.scatter(df_lr["LR"], df_lr["Val Acc"], s=100, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Learning Rate vs Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Annotate points
        for _, row in df_lr.iterrows():
            ax.annotate(
                row["ID"],
                (row["LR"], row["Val Acc"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # 4. Batch Size vs Accuracy
    ax = axes[1, 1]
    df_bs = df_plot[df_plot["Batch Size"] != "N/A"].copy()
    if len(df_bs) > 0:
        df_bs["Batch Size"] = pd.to_numeric(df_bs["Batch Size"])
        ax.scatter(df_bs["Batch Size"], df_bs["Val Acc"], s=100, alpha=0.6)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Batch Size vs Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Annotate points
        for _, row in df_bs.iterrows():
            ax.annotate(
                row["ID"],
                (row["Batch Size"], row["Val Acc"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ML experiments")
    parser.add_argument("exp_ids", nargs="+", help="Experiment IDs to compare")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("logs/experiment_registry.json"),
        help="Path to experiment registry",
    )

    args = parser.parse_args()

    compare_experiments(args.exp_ids, args.registry)


if __name__ == "__main__":
    main()
