# Hybrid Loss Quick Start Guide

## Overview

The hybrid loss system smoothly transitions from **L2 loss** (better for early training) to **MGIoU loss** (better for final refinement) during polygon training.

## Basic Usage

### 1. Simple Training with Hybrid Loss

```bash
yolo train \
    model=yolo11n-polygon.yaml \
    data=coco128.yaml \
    epochs=100 \
    use_hybrid=True
```

This uses **default settings**:
- **Schedule**: Cosine (smooth transition)
- **Alpha start**: 0.9 (90% L2, 10% MGIoU at epoch 0)
- **Alpha end**: 0.2 (20% L2, 80% MGIoU at epoch 99)

### 2. Custom Alpha Range

```bash
yolo train \
    model=yolo11n-polygon.yaml \
    data=coco128.yaml \
    epochs=100 \
    use_hybrid=True \
    alpha_start=1.0 \
    alpha_end=0.0
```

This transitions from **100% L2** to **100% MGIoU**.

### 3. Different Schedules

**Linear schedule** (constant rate of change):
```bash
yolo train model=yolo11n-polygon.yaml data=coco128.yaml use_hybrid=True alpha_schedule=linear
```

**Step schedule** (sudden switch at epoch 50):
```bash
yolo train model=yolo11n-polygon.yaml data=coco128.yaml use_hybrid=True alpha_schedule=step
```

**Cosine schedule** (default - smooth transition):
```bash
yolo train model=yolo11n-polygon.yaml data=coco128.yaml use_hybrid=True alpha_schedule=cosine
```

## Monitoring Training

### Real-time Console Monitor

```bash
# In terminal 1: Start training
yolo train model=yolo11n-polygon.yaml data=coco128.yaml use_hybrid=True project=runs/hybrid name=exp1

# In terminal 2: Monitor in real-time
python monitor_hybrid_training.py --run runs/hybrid/exp1 --console
```

**Output example:**
```
================================================================================
Epoch 25
================================================================================
  Alpha:          0.7955 (L2: 79.6%, MGIoU: 20.4%)

  Losses:
    Polygon:      0.036994
    Box:          0.028456
    Class:        0.012345

  Validation:
    mAP@50-95:    0.4523
    mAP@50:       0.6789

  Polygon Loss Breakdown:
    L2 contrib:   0.029427 (79.6%)
    MGIoU contrib:0.007567 (20.4%)
```

### GUI Monitor with Plots

```bash
python monitor_hybrid_training.py --run runs/hybrid/exp1
```

Shows 4 live plots:
1. Alpha schedule progression
2. Loss components over time
3. Validation mAP metrics
4. L2 vs MGIoU contributions

## Benchmarking Strategies

Compare **L2-only**, **Two-stage**, and **Hybrid** on your dataset:

```bash
python benchmark_polygon_losses.py \
    --data your_data.yaml \
    --model yolo11n-polygon.yaml \
    --epochs 100 \
    --device 0
```

**Outputs:**
- `runs/polygon_benchmark/benchmark_results/benchmark_results.csv` - Metrics comparison
- `runs/polygon_benchmark/benchmark_results/training_time_comparison.png`
- `runs/polygon_benchmark/benchmark_results/map_comparison.png`
- `runs/polygon_benchmark/benchmark_results/loss_comparison.png`

**Run specific strategies only:**
```bash
# Only compare L2 and Hybrid
python benchmark_polygon_losses.py --data coco128.yaml --strategies l2 hybrid

# Only run Two-stage
python benchmark_polygon_losses.py --data coco128.yaml --strategies two_stage
```

## Testing the Implementation

Run the comprehensive test suite:

```bash
conda run -n mgiou python test_hybrid_loss.py
```

**Tests:**
- ✅ Alpha scheduling (cosine, linear, step)
- ✅ PolygonLoss modes (L2, MGIoU, Hybrid)
- ✅ Gradient flow and backpropagation
- ✅ Epoch passing mechanism
- ✅ v8PolygonLoss integration
- ✅ Gradient normalization via EMA

## When to Use Each Strategy

### Use **L2 Only** when:
- Dataset has very few samples (<100 images)
- Polygons are simple shapes (rectangles, triangles)
- Training time is critical

### Use **Two-Stage** when:
- You want manual control over switching point
- You can monitor training and decide when to switch
- Traditional approach is preferred

### Use **Hybrid** when:
- Dataset is medium to large (>500 images)
- Polygons are complex/irregular
- You want smooth, automatic transition
- **Recommended for most use cases**

## Common Issues

### Alpha not changing?
Check that `use_hybrid=True` is set and `epochs > 1`.

### NaN losses?
Try reducing `alpha_end` to keep more L2 stabilization:
```bash
yolo train ... use_hybrid=True alpha_end=0.3
```

### Training too slow?
Two-stage hybrid avoids computing both losses:
- Set `alpha_start=1.0` and `alpha_end=0.0` for pure transition
- Or use `alpha_schedule=step` to compute only one loss at a time

## Advanced Configuration

### Custom schedule for long training

For 300 epochs with delayed MGIoU:
```bash
yolo train \
    model=yolo11s-polygon.yaml \
    data=large_dataset.yaml \
    epochs=300 \
    use_hybrid=True \
    alpha_start=0.95 \
    alpha_end=0.1 \
    alpha_schedule=cosine
```

### Aggressive early MGIoU

For fine-tuning pre-trained model:
```bash
yolo train \
    model=best_weights.pt \
    data=your_data.yaml \
    epochs=50 \
    use_hybrid=True \
    alpha_start=0.5 \
    alpha_end=0.0 \
    alpha_schedule=linear
```

## Files Modified

The hybrid loss implementation spans these files:
1. `ultralytics/utils/loss.py` - Core PolygonLoss and v8PolygonLoss classes
2. `ultralytics/cfg/default.yaml` - Configuration parameters
3. `ultralytics/nn/tasks.py` - PolygonModel initialization
4. `ultralytics/models/yolo/polygon/train.py` - PolygonTrainer epoch callback

All changes are **backward compatible** - existing code works without modification.

## Getting Help

- **Full documentation**: See `POLYGON_TRAINING_GUIDE.md`
- **Test suite**: Run `python test_hybrid_loss.py` to verify installation
- **Monitor tool**: `python monitor_hybrid_training.py --help`
- **Benchmark tool**: `python benchmark_polygon_losses.py --help`

## Quick Reference

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `use_hybrid` | `False` | `True`, `False` | Enable hybrid loss |
| `alpha_schedule` | `cosine` | `cosine`, `linear`, `step` | Transition schedule |
| `alpha_start` | `0.9` | 0.0-1.0 | Initial L2 weight |
| `alpha_end` | `0.2` | 0.0-1.0 | Final L2 weight |
| `use_mgiou` | `False` | `True`, `False` | Pure MGIoU (overridden by hybrid) |

**Alpha interpretation:**
- `alpha = 1.0` → 100% L2 loss
- `alpha = 0.5` → 50% L2, 50% MGIoU
- `alpha = 0.0` → 100% MGIoU loss
