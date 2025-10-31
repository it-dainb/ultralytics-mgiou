# YOLO Polygon Training Guide: Fixing Full-Image Predictions

## Problem Summary

When training YOLO polygon models with `use_mgiou=True`, you may encounter:
- ❌ Full-image bounding box predictions after 100 epochs
- ❌ High mAP@50 (~0.9-0.95) but very low mAP@50-95 (~0.01-0.05)
- ❌ Loss plateauing around epochs 20-30 (~0.2-0.25)
- ❌ Model stuck in degenerate local minimum

## Root Cause

**MGIoU loss complexity causes gradient flow issues:**
- MGIoU uses 6+ transformation steps (normalization → SAT projection → GIoU computation)
- Gradients diluted by 10-100x compared to simple L2 loss
- Model finds "degenerate solution": full-image boxes achieve IoU > 0.5
- Weak gradients cannot escape this local minimum

**Key Insight from Pose vs Polygon Analysis:**
- Pose head uses identical `* 2.0` decoding formula but works perfectly
- Difference is NOT the decoding formula but the loss function
- Pose uses simple L2 loss with strong gradients
- Polygon uses complex MGIoU with weak gradients

## Solution: Three Training Strategies

### Strategy 1: L2 Loss Only (Recommended for Most Cases)

**Use this if:** You need accurate polygon predictions without geometric IoU optimization.

```bash
yolo train \
    model=yolo11-polygon.yaml \
    data=your_data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    use_mgiou=False  # Use simple L2 loss
```

**Expected results:**
- ✅ Polygon predictions will be actual shapes (not full-image boxes)
- ✅ `polygon_loss` decreases steadily to <0.4 by epoch 50
- ✅ `mAP@50-95` reaches >0.1 (ideally >0.3)
- ✅ Training converges properly without plateauing

**Monitoring metrics:**
```bash
# Check training logs for:
# - polygon_loss should decrease from ~1.0 to <0.4
# - mAP@50-95 should increase steadily
# - No loss plateau at epochs 20-30
```

---

### Strategy 2: Two-Stage Training (Best of Both Worlds)

**Use this if:** You need MGIoU's geometric awareness after learning basic shapes.

#### **Stage 1: Learn with L2 Loss (50-70 epochs)**

```bash
yolo train \
    model=yolo11-polygon.yaml \
    data=your_data.yaml \
    epochs=70 \
    imgsz=640 \
    batch=8 \
    use_mgiou=False \
    project=runs/polygon \
    name=stage1_l2
```

**Wait for:**
- ✅ `mAP@50-95` reaches >0.2
- ✅ `polygon_loss` stabilizes at <0.3
- ✅ Visual inspection shows proper polygon shapes

#### **Stage 2: Refine with MGIoU (30-50 epochs)**

```bash
yolo train \
    model=runs/polygon/stage1_l2/weights/best.pt \
    data=your_data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    use_mgiou=True \
    project=runs/polygon \
    name=stage2_mgiou \
    lr0=0.001  # Lower learning rate for refinement
```

**Expected results:**
- ✅ Further improvement in geometric accuracy
- ✅ Better handling of irregular polygon shapes
- ✅ Slight improvement in mAP@50-95 (0.05-0.1 gain)

**Why this works:**
- Stage 1 escapes degenerate local minimum with strong L2 gradients
- Stage 2 refines predictions with geometric-aware MGIoU gradients
- Model already knows basic shapes, so weak MGIoU gradients are sufficient

---

### Strategy 3: Hybrid Loss with Dynamic Scheduling (Recommended for Advanced Users)

**Use this if:** You want the best of both worlds in a single training run with automatic L2→MGIoU transition.

**NEW: Now available via CLI!** No code modification required.

```bash
yolo train \
    model=yolo11n-polygon.yaml \
    data=your_data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    use_hybrid=True \
    alpha_schedule=cosine \
    alpha_start=0.9 \
    alpha_end=0.2
```

**How it works:**
- Combines L2 and MGIoU losses with dynamic weighting: `loss = α × L2 + (1-α) × MGIoU`
- Alpha (L2 weight) automatically transitions from 0.9 → 0.2 over training
- Early epochs: Strong L2 gradients prevent degenerate solutions (α=0.9)
- Late epochs: Geometric MGIoU refinement takes over (α=0.2)
- Gradient normalization ensures balanced contribution from both losses

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_hybrid` | False | Enable hybrid loss (overrides `use_mgiou`) |
| `alpha_schedule` | "cosine" | Schedule type: "cosine", "linear", or "step" |
| `alpha_start` | 0.9 | Initial L2 weight (high for strong early gradients) |
| `alpha_end` | 0.2 | Final L2 weight (low to favor MGIoU refinement) |

**Schedule visualization:**

```
Alpha (L2 weight) over 100 epochs:

Cosine:  0.9 ━━━━━━━╮           ╭━━━━━━━ 0.2
                    ╰━━━━━━━━━━━╯
         Strong L2 → Smooth transition → Strong MGIoU

Linear:  0.9 ━━━━━━━━━━━━━━━━━━━╲ 0.2
         Steady L2 decrease

Step:    0.9 ━━━━━┓     ┏━━━━┓     0.2
                  ┗━━━━━┛    ┗━━━━━━
         Epoch 0-49: α=0.9
         Epoch 50-74: α=0.55
         Epoch 75-99: α=0.2
```

**Expected results:**
- ✅ Proper polygon shapes (no full-image predictions)
- ✅ Better geometric accuracy than pure L2
- ✅ Faster convergence than two-stage training
- ✅ Single training run (no need for Stage 1 → Stage 2)
- ✅ `polygon_loss` decreases steadily to <0.3
- ✅ `mAP@50-95` reaches >0.3-0.35

**When to use each schedule:**

- **Cosine** (recommended): Smooth transition, best for most cases
- **Linear**: Steady decay, good for longer training (>150 epochs)
- **Step**: Discrete jumps, mimics two-stage training automatically

**Advanced tuning:**

```bash
# More aggressive MGIoU (better geometric accuracy, slower convergence)
yolo train ... use_hybrid=True alpha_start=0.8 alpha_end=0.1

# More conservative (faster early convergence, less geometric refinement)
yolo train ... use_hybrid=True alpha_start=0.95 alpha_end=0.3

# Longer transition period
yolo train ... use_hybrid=True alpha_schedule=linear epochs=150
```

**Monitoring hybrid training:**

During training, you can see alpha values in logs:
```
Epoch 0:   α=0.900, L2=0.52, MGIoU=0.34
Epoch 25:  α=0.795, L2=0.41, MGIoU=0.28
Epoch 50:  α=0.544, L2=0.35, MGIoU=0.22
Epoch 75:  α=0.297, L2=0.30, MGIoU=0.18
Epoch 99:  α=0.200, L2=0.28, MGIoU=0.15
```

**Advantages:**
- ✅ Single training run (no manual stage switching)
- ✅ Automatic L2→MGIoU transition
- ✅ Gradient normalization prevents MGIoU from being overwhelmed
- ✅ Combines benefits of both loss functions
- ✅ Available via CLI (no code changes needed)

**Testing:**
Run the test suite to verify hybrid loss implementation:
```bash
python test_hybrid_loss.py
```

Expected output: All 6 tests should pass, with visualizations saved to `alpha_schedule_comparison.png` and `loss_mode_comparison.png`.

---

## Diagnostic Checklist

### Before Training

- [ ] Verify polygon annotations are correct (normalized coordinates)
- [ ] Check dataset YAML has correct `np` (number of polygon points)
- [ ] Ensure model YAML matches your task (e.g., `yolo11n-polygon.yaml`)

### During Training

Monitor these metrics every 10 epochs:

| Metric | L2 Mode Expected | MGIoU Mode (If Used) |
|--------|------------------|----------------------|
| `polygon_loss` | Decrease from ~1.0 to <0.4 | Should not plateau at 0.2-0.25 |
| `mAP@50` | Increase to >0.8 | Increase to >0.8 |
| `mAP@50-95` | Increase to >0.2 | Increase to >0.15 |
| Visual predictions | Proper polygon shapes | Not full-image boxes |

### After Training

```bash
# Run validation
yolo val \
    model=runs/polygon/train/weights/best.pt \
    data=your_data.yaml \
    save_json=True

# Check predictions visually
yolo predict \
    model=runs/polygon/train/weights/best.pt \
    source=path/to/test/images \
    save=True
```

**Expected visual results:**
- ✅ Polygons follow object boundaries
- ✅ Vertices positioned at corners/key points
- ✅ No full-image bounding boxes
- ✅ Proper shape variation across different objects

---

## Quick Reference

### Command Comparison

```bash
# ❌ DON'T: MGIoU from scratch (causes full-image predictions)
yolo train model=model.yaml data=data.yaml use_mgiou=True

# ✅ DO: L2 loss only
yolo train model=model.yaml data=data.yaml use_mgiou=False

# ✅ DO: Two-stage training
# Stage 1: L2
yolo train model=model.yaml data=data.yaml use_mgiou=False epochs=70 name=stage1
# Stage 2: MGIoU refinement
yolo train model=runs/polygon/stage1/weights/best.pt data=data.yaml use_mgiou=True epochs=50 name=stage2 lr0=0.001
```

### Loss Function Comparison

| Loss Type | Gradient Strength | Convergence | Geometric Awareness | Use Case |
|-----------|------------------|-------------|---------------------|----------|
| **L2** | ✅ Strong | ✅ Reliable | ⚠️ Limited | Initial training, simple shapes |
| **MGIoU** | ⚠️ Weak | ❌ Unstable from scratch | ✅ Excellent | Refinement, complex shapes |
| **Hybrid** | ✅ Strong + geometric | ✅ Good | ✅ Good | Single-run training (advanced) |

---

## Troubleshooting

### Issue: Still getting full-image predictions with L2 loss

**Possible causes:**
1. Incorrect polygon annotations (not normalized, wrong format)
2. Dataset YAML missing or incorrect `np` parameter
3. Model loading pretrained weights trained with MGIoU

**Solutions:**
```bash
# Verify annotations
python diagnose_full_image_predictions.py  # Use diagnostic script from repo

# Train from scratch (no pretrained weights)
yolo train model=yolo11n-polygon.yaml data=data.yaml pretrained=False use_mgiou=False

# Check dataset YAML
# Ensure it has: np: <number_of_polygon_points>
```

### Issue: L2 training works but MGIoU refinement fails

**Symptoms:**
- Stage 1 (L2) produces proper polygons
- Stage 2 (MGIoU) reverts to full-image boxes

**Solutions:**
```bash
# Use lower learning rate in Stage 2
yolo train model=stage1/best.pt data=data.yaml use_mgiou=True lr0=0.001 lrf=0.001

# Use shorter MGIoU refinement
yolo train model=stage1/best.pt data=data.yaml use_mgiou=True epochs=30

# Freeze backbone during MGIoU refinement
yolo train model=stage1/best.pt data=data.yaml use_mgiou=True freeze=10
```

### Issue: mAP@50-95 still low even with proper shapes

**Possible causes:**
- Polygon vertices not precise enough
- Need more training epochs
- Learning rate too high/low

**Solutions:**
```bash
# Train longer
yolo train ... epochs=150

# Adjust learning rate
yolo train ... lr0=0.005 lrf=0.005

# Use cosine LR scheduler
yolo train ... cos_lr=True
```

---

## Performance Expectations

### L2 Loss Mode

| Epochs | polygon_loss | mAP@50 | mAP@50-95 | Status |
|--------|-------------|--------|-----------|--------|
| 0-20 | 1.0 → 0.6 | 0.3 → 0.6 | 0.05 → 0.15 | Learning basic shapes |
| 20-50 | 0.6 → 0.4 | 0.6 → 0.8 | 0.15 → 0.25 | Refining predictions |
| 50-100 | 0.4 → 0.3 | 0.8 → 0.9 | 0.25 → 0.35 | Fine-tuning |

### Two-Stage Training

**Stage 1 (L2, 70 epochs):**
- Final polygon_loss: ~0.35
- Final mAP@50: ~0.85
- Final mAP@50-95: ~0.28

**Stage 2 (MGIoU, 50 epochs):**
- Final polygon_loss: ~0.30
- Final mAP@50: ~0.88
- Final mAP@50-95: ~0.35
- Improvement: +0.03-0.07 in mAP@50-95

---

## References

- **Analysis Document**: `POLYGON_VS_POSE_ANALYSIS.md` - Detailed comparison of Pose vs Polygon loss
- **Gradient Flow Analysis**: `GRADIENT_FLOW_FIX.md` - MGIoU gradient flow issues
- **Diagnostic Script**: `diagnose_full_image_predictions.py` - Tool for debugging predictions
- **Implementation**: `ultralytics/utils/loss.py:794-870` - PolygonLoss class

---

## Summary

1. **Default to L2 loss** (`use_mgiou=False`) for reliable training
2. **Use two-stage training** if you need MGIoU's geometric awareness
3. **Never start training from scratch with MGIoU** - it causes degenerate predictions
4. **Monitor polygon_loss and mAP@50-95** - they should increase steadily, not plateau
5. **Visual inspection is critical** - check that predictions show actual polygon shapes

**Quick start command:**
```bash
yolo train model=yolo11-polygon.yaml data=your_data.yaml use_mgiou=False epochs=100
```
