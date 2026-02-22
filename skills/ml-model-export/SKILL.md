---
name: ml-model-export
description: Export trained PyTorch models to various formats (ONNX, TorchScript, TensorRT) and upload to model registries (Hugging Face Hub, MLflow). Use when deploying models, sharing trained weights, or preparing for production inference.
disable-model-invocation: true
---

# ML Model Export

Export trained PyTorch models to various formats for deployment and sharing.

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
    """Export PyTorch model to ONNX format."""
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
from onnx import optimizer as onnx_optimizer

def optimize_onnx(input_path: Path, output_path: Path):
    """Optimize ONNX model for inference."""
    model = onnx.load(str(input_path))

    # Apply optimizations
    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
    ]

    optimized_model = onnx_optimizer.optimize(model, passes)
    onnx.save(optimized_model, str(output_path))

    # Report sizes
    original_size = input_path.stat().st_size / 1024**2
    optimized_size = output_path.stat().st_size / 1024**2
    reduction = (1 - optimized_size / original_size) * 100

    print(f"✓ Optimized ONNX saved: {output_path}")
    print(f"  Size reduction: {reduction:.1f}%")
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
def export_torchscript_trace(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
):
    """Export using torch.jit.trace (for fixed control flow)."""
    model.eval()
    device = next(model.parameters()).device
    example_input = torch.randn(input_shape).to(device)

    # Trace model
    traced_model = torch.jit.trace(model, example_input)

    # Optimize for inference
    traced_model = torch.jit.freeze(traced_model)

    # Save
    traced_model.save(str(output_path))
    print(f"✓ TorchScript (traced) saved: {output_path}")
```

**Export with Scripting:**

```python
def export_torchscript_script(
    model: torch.nn.Module,
    output_path: Path,
):
    """Export using torch.jit.script (for dynamic control flow)."""
    model.eval()

    # Script model (analyzes Python code)
    scripted_model = torch.jit.script(model)

    # Optimize
    scripted_model = torch.jit.freeze(scripted_model)

    # Save
    scripted_model.save(str(output_path))
    print(f"✓ TorchScript (scripted) saved: {output_path}")
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
    print(f"✓ Mobile TorchScript saved: {output_path}")
```

### 3. TensorRT (NVIDIA GPUs)

**Benefits:**

- Optimized inference on NVIDIA GPUs
- Up to 10x speedup
- Automatic kernel fusion
- Supports FP32, FP16, INT8

**Convert ONNX to TensorRT:**

```python
import tensorrt as trt

def convert_onnx_to_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",  # fp32, fp16, int8
):
    """Convert ONNX to TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX model")

    # Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB

    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)

    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)

    # Save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"✓ TensorRT engine saved: {engine_path}")
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
    """Upload model to Hugging Face Hub."""
    api = HfApi()

    # Create repo
    create_repo(repo_name, token=token, exist_ok=True)
    print(f"✓ Repository created: {repo_name}")

    # Upload model
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=repo_name,
        token=token,
    )

    # Upload model card
    if model_card:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token,
        )

    print(f"✓ Uploaded to: https://huggingface.co/{repo_name}")
```

**Generate Model Card:**

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

model = torch.load("model.pt")
model.eval()
output = model(input_tensor)
```

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

Use the automated export script:

```bash
python scripts/export_model.py checkpoints/best.ckpt
```

See `scripts/export_model.py` for implementation.

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

✅ Your models are ready for production!
