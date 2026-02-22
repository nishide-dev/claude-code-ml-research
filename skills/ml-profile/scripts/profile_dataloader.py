#!/usr/bin/env python3
"""Profile DataLoader to find optimal num_workers and identify bottlenecks."""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt


def profile_dataloader(dataloader, num_batches: int = 100):
    """Measure data loading speed.

    Args:
        dataloader: DataLoader to profile
        num_batches: Number of batches to measure

    Returns:
        Average time per batch in seconds
    """
    times = []

    print(f"Profiling DataLoader ({num_batches} batches)...")

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start = time.time()
        _ = batch  # Access data
        elapsed = time.time() - start
        times.append(elapsed)

    if not times:
        print("No batches loaded")
        return 0.0

    avg_time = sum(times) / len(times)
    throughput = 1.0 / avg_time if avg_time > 0 else 0

    print("\nData Loading Profile:")
    print(f"  Avg time/batch: {avg_time * 1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} batches/sec")
    print(f"  Total batches: {len(times)}")

    return avg_time


def find_optimal_num_workers(
    dataset,
    batch_size: int = 32,
    max_workers: int = 16,
    num_batches: int = 50,
    output_path: Path = Path("num_workers_profile.png"),
):
    """Find optimal num_workers for DataLoader.

    Args:
        dataset: Dataset to test
        batch_size: Batch size to use
        max_workers: Maximum number of workers to test
        num_batches: Number of batches per test
        output_path: Path to save plot

    Returns:
        Optimal number of workers
    """
    from torch.utils.data import DataLoader

    print(f"Finding optimal num_workers (batch_size={batch_size})...")
    results = []

    for num_workers in range(max_workers + 1):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        avg_time = profile_dataloader(loader, num_batches=num_batches)
        results.append((num_workers, avg_time))
        print(f"  num_workers={num_workers}: {avg_time * 1000:.2f}ms/batch")

    # Plot results
    workers, times = zip(*results)

    plt.figure(figsize=(10, 6))
    plt.plot(workers, times, marker="o", linewidth=2, markersize=8)
    plt.xlabel("num_workers", fontsize=12)
    plt.ylabel("Time per batch (s)", fontsize=12)
    plt.title("DataLoader Performance vs num_workers", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark optimal
    optimal_idx = times.index(min(times))
    optimal_workers = workers[optimal_idx]
    plt.axvline(
        optimal_workers, color="r", linestyle="--", alpha=0.7, label=f"Optimal: {optimal_workers}"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Plot saved to: {output_path}")

    print(f"\n🎯 Optimal num_workers: {optimal_workers}")
    print(f"   Best time: {times[optimal_idx] * 1000:.2f}ms/batch")

    return optimal_workers


def main():
    parser = argparse.ArgumentParser(description="Profile DataLoader performance")
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name (default: config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to test (default: 32)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum num_workers to test (default: 16)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches per test (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("num_workers_profile.png"),
        help="Output path for plot (default: num_workers_profile.png)",
    )

    args = parser.parse_args()

    # Load config and create dataset
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)

    # Create datamodule and setup
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    dataset = datamodule.train_dataset

    # Find optimal workers
    find_optimal_num_workers(
        dataset=dataset,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_batches=args.num_batches,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
