#!/usr/bin/env python3
"""Generate comprehensive ML training debug report."""

import argparse
from pathlib import Path

import pandas as pd
import torch


def generate_debug_report(log_dir: Path, output_file: Path | None = None):
    """Generate comprehensive debug report for ML training.

    Args:
        log_dir: Directory containing training logs and metrics
        output_file: Optional output file path for report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ML TRAINING DEBUG REPORT")
    report_lines.append("=" * 70)

    # 1. System Info
    report_lines.append("\n## System Information")
    report_lines.append(f"PyTorch version: {torch.__version__}")
    report_lines.append(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        report_lines.append(f"CUDA version: {torch.version.cuda}")
        report_lines.append(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            report_lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Memory usage
        report_lines.append("\n## GPU Memory Usage")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            report_lines.append(f"GPU {i}:")
            report_lines.append(f"  Allocated: {allocated:.2f} GB")
            report_lines.append(f"  Reserved:  {reserved:.2f} GB")
            report_lines.append(f"  Total:     {total:.2f} GB")
            report_lines.append(f"  Usage:     {allocated / total * 100:.1f}%")

    # 2. Training Metrics
    metrics_file = log_dir / "metrics.csv"
    if metrics_file.exists():
        report_lines.append("\n## Training Metrics")
        metrics = pd.read_csv(metrics_file)

        if "epoch" in metrics.columns:
            report_lines.append(f"Total epochs: {metrics['epoch'].max()}")

        # Latest metrics
        if "train_loss" in metrics.columns:
            latest_train = metrics["train_loss"].iloc[-1]
            report_lines.append(f"Latest train loss: {latest_train:.6f}")

        if "val_loss" in metrics.columns:
            latest_val = metrics["val_loss"].iloc[-1]
            best_val = metrics["val_loss"].min()
            report_lines.append(f"Latest val loss:   {latest_val:.6f}")
            report_lines.append(f"Best val loss:     {best_val:.6f}")

            # Overfitting check
            if "train_loss" in metrics.columns:
                overfit_ratio = latest_val / latest_train
                report_lines.append(f"Overfit ratio:     {overfit_ratio:.2f}")
                if overfit_ratio > 1.5:
                    report_lines.append("  ⚠️  WARNING: Possible overfitting detected!")
                elif overfit_ratio > 1.2:
                    report_lines.append("  ℹ️  INFO: Some overfitting present")
                else:
                    report_lines.append("  ✅ GOOD: No significant overfitting")

        # Accuracy metrics
        if "train_acc" in metrics.columns:
            report_lines.append(f"Latest train acc:  {metrics['train_acc'].iloc[-1]:.4f}")
        if "val_acc" in metrics.columns:
            report_lines.append(f"Latest val acc:    {metrics['val_acc'].iloc[-1]:.4f}")
            report_lines.append(f"Best val acc:      {metrics['val_acc'].max():.4f}")

        # Loss trends
        report_lines.append("\n## Loss Trends")
        if "train_loss" in metrics.columns and len(metrics) > 10:
            recent_losses = metrics["train_loss"].tail(10)

            # Check if decreasing
            if recent_losses.is_monotonic_decreasing:
                report_lines.append("✅ Training loss is consistently decreasing")
            else:
                # Calculate trend
                first_half = recent_losses[:5].mean()
                second_half = recent_losses[5:].mean()
                if second_half < first_half:
                    report_lines.append("✅ Training loss is decreasing (with fluctuations)")
                elif second_half > first_half * 1.1:
                    report_lines.append("⚠️  WARNING: Training loss is increasing!")
                else:
                    report_lines.append("ℹ️  Training loss has plateaued")

        # Check for NaN
        if "train_loss" in metrics.columns:
            if metrics["train_loss"].isna().any():
                nan_epoch = metrics[metrics["train_loss"].isna()]["epoch"].iloc[0]
                report_lines.append(f"\n❌ ERROR: NaN loss detected at epoch {nan_epoch}")

    else:
        report_lines.append("\n## Training Metrics")
        report_lines.append(f"❌ Metrics file not found: {metrics_file}")

    # 3. Checkpoint Analysis
    checkpoint_dir = log_dir / "checkpoints"
    if checkpoint_dir.exists():
        report_lines.append("\n## Checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        report_lines.append(f"Number of checkpoints: {len(checkpoints)}")

        if checkpoints:
            # Analyze latest checkpoint
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            report_lines.append(f"Latest checkpoint: {latest_ckpt.name}")

            try:
                ckpt = torch.load(latest_ckpt, map_location="cpu")

                # Check for NaN in weights
                nan_count = 0
                inf_count = 0
                extreme_count = 0

                for name, param in ckpt["state_dict"].items():
                    if torch.isnan(param).any():
                        nan_count += 1
                        report_lines.append(f"  ❌ NaN in weights: {name}")

                    if torch.isinf(param).any():
                        inf_count += 1
                        report_lines.append(f"  ❌ Inf in weights: {name}")

                    # Check for extreme values
                    max_val = param.abs().max().item()
                    if max_val > 1e6:
                        extreme_count += 1
                        report_lines.append(f"  ⚠️  Extreme values in {name}: max={max_val:.2e}")

                if nan_count == 0 and inf_count == 0 and extreme_count == 0:
                    report_lines.append("  ✅ Checkpoint weights look healthy")

            except Exception as e:
                report_lines.append(f"  ⚠️  Error loading checkpoint: {e}")

    # 4. Recommendations
    report_lines.append("\n## Recommendations")

    recommendations = []

    # Based on metrics
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)

        # High overfitting
        if "val_loss" in metrics.columns and "train_loss" in metrics.columns:
            overfit_ratio = metrics["val_loss"].iloc[-1] / metrics["train_loss"].iloc[-1]
            if overfit_ratio > 1.5:
                recommendations.append("- Add regularization: increase dropout, add weight decay")
                recommendations.append("- Use data augmentation")
                recommendations.append("- Enable early stopping")
                recommendations.append("- Reduce model capacity")

        # Not decreasing
        if "train_loss" in metrics.columns and len(metrics) > 10:
            recent = metrics["train_loss"].tail(10)
            if recent.mean() > metrics["train_loss"].head(5).mean():
                recommendations.append("- Learning rate may be too high")
                recommendations.append("- Try gradient clipping")
                recommendations.append("- Check for NaN in data")

        # Plateau
        if "train_loss" in metrics.columns and len(metrics) > 20:
            recent = metrics["train_loss"].tail(10)
            if recent.std() < 0.001:
                recommendations.append("- Loss has plateaued")
                recommendations.append("- Try reducing learning rate")
                recommendations.append("- Consider using learning rate scheduler")

    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            usage = allocated / total

            if usage > 0.9:
                recommendations.append(f"- GPU {i} memory usage very high ({usage * 100:.1f}%)")
                recommendations.append(
                    "  Consider reducing batch size or using gradient accumulation"
                )

    if recommendations:
        for rec in recommendations:
            report_lines.append(rec)
    else:
        report_lines.append("✅ No major issues detected")

    # 5. Footer
    report_lines.append("\n" + "=" * 70)

    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file if requested
    if output_file:
        output_file.write_text(report_text)
        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate ML training debug report")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report (default: print to stdout)",
    )

    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: Log directory not found: {args.log_dir}")
        return 1

    generate_debug_report(args.log_dir, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
