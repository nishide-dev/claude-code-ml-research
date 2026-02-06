---
name: model-export
description: Export trained PyTorch models to various formats (ONNX, TorchScript, TensorRT) and upload to model registries (Hugging Face Hub, MLflow)
---

# ML Model Export

Export trained PyTorch models to various formats for deployment (ONNX, TorchScript, TensorRT) and upload to model registries.

## Overview

Model export enables:

- Cross-platform deployment (ONNX)
- Production serving (TorchScript)
- Optimized inference (TensorRT, OpenVINO)
- Model sharing (Hugging Face Hub, MLflow)
- Mobile deployment (TorchScript Mobile, ONNX)

## Export Formats

### 1. ONNX (Open Neural Network Exchange)

**Benefits:**

- Cross-framework compatibility (PyTorch → TensorFlow, etc.)
- Hardware optimization (CPUs, GPUs, NPUs)
- Industry standard for model interchange
- Supported by ONNX Runtime, TensorRT, OpenVINO

**Export to ONNX:**

```python
import torch
from pathlib import Path

def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_axes: dict = None,
):
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Example input shape for tracing
        opset_version: ONNX opset version (17 recommended)
        dynamic_axes: Dynamic dimensions for flexible input shapes
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    # Default dynamic axes for batch size
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✓ Model exported to ONNX: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
```

**Validate ONNX Export:**

```python
import onnx
import onnxruntime as ort
import numpy as np

def validate_onnx(onnx_path: Path, pytorch_model: torch.nn.Module):
    """Validate ONNX export matches PyTorch output."""
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Create inference session
    ort_session = ort.InferenceSession(str(onnx_path))

    # Compare outputs
    dummy_input = torch.randn(1, 3, 224, 224)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX output
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Compare
    np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
    print("✓ ONNX output matches PyTorch output")
```

**Optimize ONNX:**

```python
import onnx
from onnxruntime.transformers import optimizer

def optimize_onnx(input_path: Path, output_path: Path):
    """Optimize ONNX model for inference."""
    # Load model
    model = onnx.load(str(input_path))

    # Apply optimizations
    from onnx import optimizer as onnx_optimizer

    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
    ]

    optimized_model = onnx_optimizer.optimize(model, passes)

    # Save optimized model
    onnx.save(optimized_model, str(output_path))
    print(f"✓ Optimized ONNX model saved to: {output_path}")

    # Compare sizes
    original_size = input_path.stat().st_size / 1024**2
    optimized_size = output_path.stat().st_size / 1024**2
    reduction = (1 - optimized_size / original_size) * 100

    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Optimized size: {optimized_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")
```

### 2. TorchScript

**Benefits:**

- Native PyTorch format
- C++ deployment without Python
- Mobile deployment (iOS, Android)
- Optimized execution
- No external dependencies

**Export with Tracing:**

```python
import torch

def export_torchscript_trace(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
):
    """Export using torch.jit.trace (for models with fixed control flow)."""
    model.eval()
    device = next(model.parameters()).device
    example_input = torch.randn(input_shape).to(device)

    # Trace model
    traced_model = torch.jit.trace(model, example_input)

    # Optimize for inference
    traced_model = torch.jit.freeze(traced_model)

    # Save
    traced_model.save(str(output_path))
    print(f"✓ TorchScript model (traced) saved to: {output_path}")
```

**Export with Scripting:**

```python
def export_torchscript_script(
    model: torch.nn.Module,
    output_path: Path,
):
    """Export using torch.jit.script (for models with dynamic control flow)."""
    model.eval()

    # Script model (analyzes Python code)
    scripted_model = torch.jit.script(model)

    # Optimize
    scripted_model = torch.jit.freeze(scripted_model)

    # Save
    scripted_model.save(str(output_path))
    print(f"✓ TorchScript model (scripted) saved to: {output_path}")
```

**Load and Use TorchScript:**

```python
def load_torchscript(model_path: Path):
    """Load TorchScript model for inference."""
    model = torch.jit.load(str(model_path))
    model.eval()

    # Inference
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)

    print(f"✓ TorchScript model loaded successfully")
    return model
```

**TorchScript Mobile:**

```python
from torch.utils.mobile_optimizer import optimize_for_mobile

def export_torchscript_mobile(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
):
    """Export for mobile deployment (iOS, Android)."""
    model.eval()
    example_input = torch.randn(input_shape)

    # Trace
    traced_model = torch.jit.trace(model, example_input)

    # Optimize for mobile
    optimized_model = optimize_for_mobile(traced_model)

    # Save
    optimized_model._save_for_lite_interpreter(str(output_path))
    print(f"✓ Mobile TorchScript model saved to: {output_path}")
```

### 3. TensorRT (NVIDIA GPUs)

**Benefits:**

- Optimized inference on NVIDIA GPUs
- Up to 10x speedup
- Automatic kernel fusion and precision calibration
- Supports FP32, FP16, INT8

**Export to TensorRT (via ONNX):**

```python
import tensorrt as trt

def convert_onnx_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",  # fp32, fp16, int8
    max_batch_size: int = 1,
):
    """Convert ONNX to TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")

    # Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB

    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration data

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)

    # Serialize and save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"✓ TensorRT engine saved to: {engine_path}")
```

### 4. Hugging Face Hub Upload

**Upload to Hub:**

```python
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface_hub(
    model_path: Path,
    repo_name: str,
    token: str = None,
    model_card: str = None,
):
    """Upload model to Hugging Face Hub.

    Args:
        model_path: Path to model file or directory
        repo_name: Repository name (username/model-name)
        token: HF token (or set HUGGING_FACE_HUB_TOKEN env var)
        model_card: Optional model card content
    """
    api = HfApi()

    # Create repo
    try:
        create_repo(repo_name, token=token, exist_ok=True)
        print(f"✓ Repository created: {repo_name}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")

    # Upload model
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=repo_name,
        token=token,
    )

    # Upload model card if provided
    if model_card:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token,
        )

    print(f"✓ Model uploaded to: https://huggingface.co/{repo_name}")
```

**Create Model Card:**

```python
def generate_model_card(
    model_name: str,
    task: str,
    metrics: dict,
    training_data: str,
) -> str:
    """Generate Hugging Face model card."""
    return f"""---
language: en
tags:
- pytorch
- pytorch-lightning
- {task}
license: mit
---

# {model_name}

## Model Description

This model was trained for {task}.

## Training Data

{training_data}

## Performance Metrics

{"".join(f"- {k}: {v}\\n" for k, v in metrics.items())}

## Usage

```python
import torch

# Load model
model = torch.load("model.pt")
model.eval()

# Inference
output = model(input_tensor)
```

## Citation

```bibtex
@misc{{{model_name.lower().replace(" ", "_")},
  author = {{Your Name}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/...}}
}}
```

"""

### 5. MLflow Model Registry

**Log Model to MLflow:**

```python
import mlflow
import mlflow.pytorch

def log_model_to_mlflow(
    model: torch.nn.Module,
    model_name: str,
    metrics: dict,
    artifacts: dict = None,
):
    """Log model and metrics to MLflow."""
    with mlflow.start_run():
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)

        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name,
        )

        print(f"✓ Model logged to MLflow: {model_name}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
```

## Complete Export Script

```python
#!/usr/bin/env python3
"""Complete model export script."""

import torch
from pathlib import Path
from typing import Optional

def export_model(
    checkpoint_path: Path,
    output_dir: Path,
    formats: list[str] = ["onnx", "torchscript"],
    input_shape: tuple = (1, 3, 224, 224),
):
    """Export model to multiple formats.

    Args:
        checkpoint_path: Path to Lightning checkpoint
        output_dir: Directory to save exported models
        formats: List of formats to export (onnx, torchscript, mobile)
        input_shape: Example input shape
    """
    from pytorch_lightning import LightningModule

    # Load model from checkpoint
    model = LightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to requested formats
    if "onnx" in formats:
        onnx_path = output_dir / "model.onnx"
        export_to_onnx(model, onnx_path, input_shape)
        validate_onnx(onnx_path, model)

        # Optimize ONNX
        onnx_opt_path = output_dir / "model_optimized.onnx"
        optimize_onnx(onnx_path, onnx_opt_path)

    if "torchscript" in formats:
        ts_path = output_dir / "model.pt"
        export_torchscript_trace(model, ts_path, input_shape)

    if "mobile" in formats:
        mobile_path = output_dir / "model_mobile.ptl"
        export_torchscript_mobile(model, mobile_path, input_shape)

    print(f"\n✓ All exports completed!")
    print(f"  Output directory: {output_dir}")

    # List exported files
    print("\nExported files:")
    for file in output_dir.iterdir():
        size = file.stat().st_size / 1024**2
        print(f"  - {file.name} ({size:.2f} MB)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python export.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    output_dir = Path("exported_models")

    export_model(checkpoint_path, output_dir)
```

## Usage

```bash
# Export model from Lightning checkpoint
python scripts/export_model.py checkpoints/best.ckpt

# Export to specific formats
python scripts/export_model.py checkpoints/best.ckpt --formats onnx torchscript

# Upload to Hugging Face
python scripts/upload_to_hub.py \
    --model exported_models/model.onnx \
    --repo username/model-name \
    --token $HF_TOKEN
```

## Deployment Examples

**ONNX Runtime Inference:**

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": input_data})
```

**TorchScript Inference:**

```python
import torch

# Load model
model = torch.jit.load("model.pt")

# Inference
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
```

## Success Criteria

- [ ] Model exported to required formats
- [ ] Exported models validated (output matches PyTorch)
- [ ] Models optimized for inference
- [ ] Documentation generated
- [ ] Models uploaded to registry (if needed)
- [ ] Deployment examples provided

Your models are ready for production!
