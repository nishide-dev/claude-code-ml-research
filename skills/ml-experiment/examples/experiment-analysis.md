# Experiment Analysis Workflows

Comprehensive guide for analyzing ML experiments, from single runs to multi-experiment comparisons.

## Single Experiment Analysis

### 1. Quick Analysis

Analyze a single experiment directory:

```bash
python scripts/analyze_experiment.py logs/2026-02-22/14-30-22/
```

**Output:**

```text
=== Experiment Analysis ===

Best Val Accuracy:    0.8921 (epoch 42)
Best Val Loss:        0.2984 (epoch 42)
Final Train Loss:     0.1456
Epochs Trained:       45/50
Final Learning Rate:  0.000123

Training Time:        2h 34m 18s
GPU Memory Peak:      8.2 GB
Samples/Second:       1834.5
```

### 2. Detailed Training Curves

Generate comprehensive training visualizations:

```python
from scripts.analyze_experiment import analyze_training_curves

analyze_training_curves(
    log_dir="logs/2026-02-22/14-30-22/",
    metrics=["train/loss", "val/loss", "val/acc", "learning_rate"],
    save_path="analysis/training_curves.png"
)
```

**Generates:**

- 4-panel plot with all metrics
- Smoothed curves with confidence bands
- Epoch markers for best checkpoints
- Learning rate schedule overlay

### 3. Loss Landscape Analysis

Visualize loss landscape around final checkpoint:

```python
from scripts.loss_landscape import plot_loss_landscape

plot_loss_landscape(
    checkpoint_path="logs/2026-02-22/14-30-22/checkpoints/best.ckpt",
    data_loader=val_dataloader,
    n_points=50,
    save_path="analysis/loss_landscape.png"
)
```

**Output:**

- 2D contour plot of loss surface
- Identifies flat vs sharp minima
- Helps understand generalization

## Multi-Experiment Comparison

### 1. Compare Multiple Experiments

```bash
python scripts/compare_experiments.py exp_001 exp_002 exp_003 exp_004
```

**Table Output:**

```text
ID      Name                    Val Acc  Val Loss  LR      Batch  Epochs  Runtime  GPU
exp_001 baseline_resnet50       0.8765   0.3241   0.001   128    45      2h 34m   2
exp_002 resnet50_larger_batch   0.8921   0.2984   0.002   256    42      3h 12m   4
exp_003 resnet50_dropout        0.8843   0.3121   0.001   128    48      2h 45m   2
exp_004 resnet50_augmentation   0.8998   0.2756   0.001   128    50      2h 52m   2
```

### 2. Visual Comparison

Generate comparison plots:

```python
from scripts.compare_experiments import plot_comparison

plot_comparison(
    experiment_ids=["exp_001", "exp_002", "exp_003", "exp_004"],
    metrics=["val/acc", "val/loss"],
    save_path="analysis/comparison.png"
)
```

**Generates:**

- Side-by-side bar charts
- Error bars (if multiple runs)
- Statistical significance markers
- Pareto frontier highlighting

### 3. Hyperparameter Impact Analysis

Analyze effect of hyperparameters:

```python
from scripts.hyperparameter_analysis import analyze_hp_impact

analyze_hp_impact(
    experiment_ids=["exp_001", "exp_002", "exp_005", "exp_006"],
    hyperparameters=["learning_rate", "batch_size", "dropout"],
    target_metric="val/acc",
    save_path="analysis/hp_impact.png"
)
```

**Output:**

- Scatter plots: HP value vs metric
- Correlation coefficients
- Optimal HP ranges highlighted
- Interaction effects (2D heatmaps)

## Ablation Study Analysis

### 1. Component Ablation

Analyze impact of removing components:

```bash
python scripts/ablation_analysis.py \
  --baseline exp_001 \
  --ablations exp_010,exp_011,exp_012 \
  --components "dropout,batch_norm,skip_connections"
```

**Output Table:**

```text
Component          Val Acc  Delta    Impact
Baseline (All)     0.8921   -        -
- Dropout          0.8765   -0.0156  Medium
- Batch Norm       0.8234   -0.0687  High
- Skip Connections 0.7543   -0.1378  Critical
```

### 2. Incremental Addition

Analyze impact of adding components:

```python
from scripts.ablation_analysis import incremental_analysis

incremental_analysis(
    baseline_exp="exp_020",  # Minimal model
    incremental_exps=["exp_021", "exp_022", "exp_023"],
    components=["residual", "attention", "normalization"],
    metric="val/acc"
)
```

**Visualization:**

- Line plot showing incremental improvements
- Cumulative impact bars
- Marginal gain per component

## Statistical Analysis

### 1. Multiple Runs Statistical Test

Compare experiments with multiple runs:

```python
from scripts.statistical_analysis import compare_with_stats

compare_with_stats(
    exp_group_1=["exp_001_run1", "exp_001_run2", "exp_001_run3"],
    exp_group_2=["exp_002_run1", "exp_002_run2", "exp_002_run3"],
    metric="val/acc",
    test="t-test"  # or "mann-whitney", "bootstrap"
)
```

**Output:**

```text
Group 1 (Baseline):     0.8921 ± 0.0034
Group 2 (Improved):     0.8998 ± 0.0028

T-statistic:            4.532
P-value:                0.0012
Effect size (Cohen's d): 0.76
Conclusion:             Significantly better (p < 0.05)
```

### 2. Confidence Intervals

Calculate confidence intervals across runs:

```python
from scripts.statistical_analysis import confidence_intervals

confidence_intervals(
    experiment_runs=["exp_001_run1", "exp_001_run2", "exp_001_run3"],
    metric="val/acc",
    confidence_level=0.95
)
```

**Output:**

```text
Mean:              0.8921
Std Dev:           0.0034
95% CI:            [0.8853, 0.8989]
Median:            0.8918
IQR:               [0.8896, 0.8945]
```

## Hyperparameter Optimization Analysis

### 1. Optuna Study Analysis

Analyze Optuna optimization results:

```python
import optuna

study = optuna.load_study(
    study_name="model_optimization",
    storage="sqlite:///optuna.db"
)

# Visualize optimization history
optuna.visualization.plot_optimization_history(study).show()

# Parameter importance
optuna.visualization.plot_param_importances(study).show()

# Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study).show()

# Slice plot (1D effects)
optuna.visualization.plot_slice(study).show()
```

### 2. W&B Sweeps Analysis

Analyze W&B hyperparameter sweeps:

```python
import wandb

api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")

# Get all runs
runs = sweep.runs

# Find best run
best_run = min(runs, key=lambda r: r.summary.get("val/loss", float("inf")))

print(f"Best run: {best_run.name}")
print(f"Best config: {best_run.config}")
print(f"Best val_loss: {best_run.summary['val/loss']}")

# Plot parameter importance
# (requires wandb parallel coordinates visualization)
```

## Learning Curve Analysis

### 1. Training Dynamics

Analyze learning dynamics:

```python
from scripts.learning_curve_analysis import analyze_dynamics

analyze_dynamics(
    log_dir="logs/2026-02-22/14-30-22/",
    save_path="analysis/learning_dynamics.png"
)
```

**Generates:**

- Loss curves (train/val) with convergence rate
- Gradient norm over time
- Learning rate schedule
- Generalization gap (train vs val)

### 2. Overfitting Detection

Detect and visualize overfitting:

```python
from scripts.learning_curve_analysis import detect_overfitting

overfitting_metrics = detect_overfitting(
    log_dir="logs/2026-02-22/14-30-22/",
    threshold=0.05  # Max acceptable train-val gap
)

print(f"Overfitting detected: {overfitting_metrics['is_overfitting']}")
print(f"Generalization gap: {overfitting_metrics['gap']:.4f}")
print(f"Early stop epoch: {overfitting_metrics['early_stop_epoch']}")
```

### 3. Convergence Analysis

Analyze convergence behavior:

```python
from scripts.learning_curve_analysis import analyze_convergence

convergence_info = analyze_convergence(
    log_dir="logs/2026-02-22/14-30-22/",
    metric="val/loss",
    patience=10
)

print(f"Converged: {convergence_info['converged']}")
print(f"Convergence epoch: {convergence_info['epoch']}")
print(f"Convergence rate: {convergence_info['rate']:.4f}")
```

## Model Performance Analysis

### 1. Per-Class Performance

Analyze performance breakdown by class:

```python
from scripts.model_analysis import analyze_per_class_performance

per_class_metrics = analyze_per_class_performance(
    checkpoint_path="logs/2026-02-22/14-30-22/checkpoints/best.ckpt",
    data_loader=test_dataloader,
    num_classes=10
)

# Print results
for class_id, metrics in per_class_metrics.items():
    print(f"Class {class_id}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
```

### 2. Confusion Matrix

Generate confusion matrix:

```python
from scripts.model_analysis import plot_confusion_matrix

plot_confusion_matrix(
    checkpoint_path="logs/2026-02-22/14-30-22/checkpoints/best.ckpt",
    data_loader=test_dataloader,
    class_names=["cat", "dog", "bird", ...],
    save_path="analysis/confusion_matrix.png"
)
```

### 3. Error Analysis

Analyze failure cases:

```python
from scripts.model_analysis import analyze_errors

error_analysis = analyze_errors(
    checkpoint_path="logs/2026-02-22/14-30-22/checkpoints/best.ckpt",
    data_loader=test_dataloader,
    top_k=20  # Top 20 worst predictions
)

# Save worst predictions
for i, error in enumerate(error_analysis["worst_predictions"]):
    print(f"Sample {i}: True={error['true']}, Pred={error['pred']}, Conf={error['conf']:.4f}")
    # Optionally save images
```

## Compute Efficiency Analysis

### 1. Training Efficiency

Analyze compute efficiency:

```python
from scripts.efficiency_analysis import analyze_training_efficiency

efficiency_metrics = analyze_training_efficiency(
    log_dir="logs/2026-02-22/14-30-22/"
)

print(f"Samples/second: {efficiency_metrics['throughput']:.2f}")
print(f"GPU utilization: {efficiency_metrics['gpu_util_avg']:.1f}%")
print(f"Memory efficiency: {efficiency_metrics['memory_efficiency']:.1f}%")
print(f"Time to best model: {efficiency_metrics['time_to_best']}")
```

### 2. Scaling Analysis

Analyze scaling with compute:

```bash
python scripts/scaling_analysis.py \
  --experiments exp_001,exp_002,exp_003,exp_004 \
  --resource gpu_count \
  --metric throughput
```

**Output:**

- Scaling curve (resources vs throughput)
- Scaling efficiency (ideal vs actual)
- Cost-performance analysis

## Report Generation

### 1. Automated Report

Generate comprehensive markdown report:

```bash
python scripts/generate_report.py \
  --experiments exp_001,exp_002,exp_003 \
  --output reports/experiment_report.md \
  --include-plots
```

**Report includes:**

- Experiment overview table
- Training curves for each experiment
- Comparison plots
- Statistical analysis
- Best hyperparameters
- Recommendations

### 2. Export to CSV

Export all results for external analysis:

```bash
python scripts/export_results.py \
  --experiments all \
  --output results.csv \
  --include-hyperparameters
```

## Best Practices

### Analysis Checklist

- [ ] Analyze training curves for all experiments
- [ ] Compare against baseline with statistical tests
- [ ] Check for overfitting (train-val gap)
- [ ] Analyze per-class performance
- [ ] Review failure cases
- [ ] Measure compute efficiency
- [ ] Document insights and recommendations

### Common Pitfalls

**Comparing single runs:**

- Always run multiple seeds for statistical significance
- Report mean ± std dev

**Ignoring compute cost:**

- Track GPU hours, not just accuracy
- Consider cost-performance tradeoffs

**Cherry-picking results:**

- Report all experiments, including failures
- Document why experiments failed

**Not analyzing failures:**

- Failure analysis is as valuable as success
- Learn what doesn't work

## Example Analysis Workflow

### Complete Analysis Pipeline

```bash
# 1. List all experiments
python scripts/list_experiments.py --tags optimization

# 2. Compare top experiments
python scripts/compare_experiments.py exp_045 exp_046 exp_047 exp_048

# 3. Statistical analysis (multiple runs)
python scripts/statistical_analysis.py \
  --groups "exp_045_run*" "exp_046_run*" \
  --metric val/acc

# 4. Generate visualizations
python scripts/plot_comparison.py \
  --experiments exp_045,exp_046,exp_047,exp_048 \
  --output analysis/comparison.png

# 5. Analyze best model
python scripts/analyze_experiment.py logs/exp_046/

# 6. Generate report
python scripts/generate_report.py \
  --experiments exp_045,exp_046,exp_047,exp_048 \
  --output reports/optimization_results.md

# 7. Export for presentation
python scripts/export_results.py \
  --experiments exp_045,exp_046,exp_047,exp_048 \
  --output results.csv \
  --format latex  # For LaTeX tables
```

## Success Criteria

Comprehensive experiment analysis should answer:

- ✓ Which approach works best and why?
- ✓ Is the improvement statistically significant?
- ✓ What hyperparameters matter most?
- ✓ Where does the model fail?
- ✓ Is the model overfitting?
- ✓ Is training compute-efficient?
- ✓ Can results be reproduced?

Analysis complete and insights documented!
