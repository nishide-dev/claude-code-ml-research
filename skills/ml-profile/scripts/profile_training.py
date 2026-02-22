#!/usr/bin/env python3
"""Profile complete training run to identify bottlenecks."""

import argparse
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.profilers import PyTorchProfiler
import torch


def profile_training(
    config_name: str = "config", max_steps: int = 100, output_dir: Path = Path("./profiler_output")
):
    """Profile a short training run.

    Args:
        config_name: Hydra config name to use
        max_steps: Number of training steps to profile
        output_dir: Directory for profiler output
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    print(f"Starting training profile ({max_steps} steps)...")

    # Load config
    config_dir = Path.cwd() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    # Create model and data
    print("Instantiating model and datamodule...")
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)

    # Setup profiler
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path("./tb_logs/profiler")
    tb_dir.mkdir(parents=True, exist_ok=True)

    profiler = PyTorchProfiler(
        dirpath=str(output_dir),
        filename="training_profile",
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(tb_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Train with profiling
    print("Running profiled training...")
    trainer = Trainer(
        profiler=profiler,
        max_steps=max_steps,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, datamodule)

    print("\n✅ Profiling complete!")
    print(f"  Profile saved to: {output_dir}/")
    print(f"  View in TensorBoard: tensorboard --logdir={tb_dir.parent}")
    print("\n  In TensorBoard:")
    print("  1. Navigate to PROFILE tab")
    print("  2. See GPU/CPU utilization over time")
    print("  3. Analyze operator-level performance")


def main():
    parser = argparse.ArgumentParser(description="Profile ML training performance")
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name (default: config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps to profile (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./profiler_output"),
        help="Output directory for profiler results (default: ./profiler_output)",
    )

    args = parser.parse_args()

    profile_training(
        config_name=args.config_name,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
