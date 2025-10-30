# NaN in Model Predictions - Diagnostic and Fix Guide

## Error Message
```
RuntimeError: NaN detected in pred_kpts at PolygonLoss.forward input
Shape: torch.Size([10, 4, 2])
NaN count: 80
```

## Root Cause
The model's prediction head is outputting NaN values **before** they reach the loss function. This is a **training instability issue**, not a loss function bug.

## Why NaNs Appear (100% of predictions are NaN)
When ALL predictions are NaN, it indicates severe training instability:

1. **Gradient Explosion** ⚠️ Most Common
   - Gradients become inf during backprop
   - Weights become NaN after weight update
   - All subsequent forward passes produce NaN

2. **Loss Function Instability**
   - Previous iteration produced NaN loss
   - NaN gradients backprop to model weights
   - Weights become NaN, model broken

3. **Learning Rate Too High**
   - Large weight updates cause overflow
   - Weights exceed float32 range

4. **Numerical Overflow in Activations**
   - exp(large_number) → inf
   - Operations on inf → NaN

## Diagnostic Steps

### Step 1: Check When NaN First Appears
We added NaN checks at multiple points:

```python
# In v8PoseLoss.__call__ (lines 1186-1188)
_check_nan_tensor(pred_scores, "pred_scores", "after permute")
_check_nan_tensor(pred_distri, "pred_distri", "after permute")  
_check_nan_tensor(pred_kpts, "pred_kpts (raw)", "after permute")

# After decoding (line 1206)
_check_nan_tensor(pred_kpts, "pred_kpts (decoded)", "after kpts_decode")

# In PolygonLoss.forward (line 839)
_check_nan_tensor(pred_kpts, "pred_kpts", "PolygonLoss.forward input")
```

**Enable debug mode:**
```bash
export ULTRALYTICS_DEBUG_NAN=1
python train.py
```

This will show EXACTLY where NaN first appears.

### Step 2: Check Model Weights
```python
import torch

# Load your model
model = torch.load('your_model.pt')

# Check for NaN in weights
nan_weights = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_count = torch.isnan(param).sum().item()
        nan_weights.append((name, nan_count, param.numel()))
        print(f"❌ NaN in {name}: {nan_count}/{param.numel()}")

if nan_weights:
    print(f"\n⚠️ Found NaN in {len(nan_weights)} weight tensors!")
    print("→ Model weights are corrupted. Need to restart from checkpoint.")
else:
    print("✓ All weights are clean (no NaN)")
```

### Step 3: Check Loss History
```python
# Look at your training logs
# If you see patterns like:
#   Epoch 1: loss=0.5234
#   Epoch 2: loss=0.4891
#   Epoch 3: loss=nan  ← Loss became NaN
#   Epoch 4: loss=nan  ← Everything NaN after this
```

## Solutions

### Solution 1: Gradient Clipping (Recommended)
Prevents gradient explosion by limiting gradient magnitude.

**Add to your training config:**
```python
from torch.nn.utils import clip_grad_norm_

# In your training loop, after loss.backward():
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=10.0)  # Start with 10.0
optimizer.step()
```

**Or in Ultralytics config:**
```yaml
# Add to your training yaml
gradient_clip: 10.0  # Maximum gradient norm
```

### Solution 2: Reduce Learning Rate
```yaml
# In your training config
lr0: 0.001  # Reduce from 0.01
lrf: 0.0001  # Final learning rate
```

### Solution 3: Use More Stable Optimizer
```python
# AdamW is more stable than SGD for some tasks
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001,
    eps=1e-8  # Numerical stability
)
```

### Solution 4: Mixed Precision Training (If using AMP)
```python
# If using automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Unscale before gradient clipping
clip_grad_norm_(model.parameters(), max_norm=10.0)
scaler.step(optimizer)
scaler.update()
```

### Solution 5: Restart from Last Good Checkpoint
If weights are corrupted:
```bash
# Find last checkpoint before NaN
ls runs/train/exp/weights/

# Restart training from good checkpoint
python train.py --resume runs/train/exp/weights/epoch50.pt
```

### Solution 6: Add Batch Normalization Safety
If using custom layers:
```python
# Ensure BatchNorm has eps for stability
nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
```

### Solution 7: Check Data for Inf/NaN
```python
# In your dataloader
def check_batch(batch):
    for key, val in batch.items():
        if torch.is_tensor(val):
            if torch.isnan(val).any():
                raise ValueError(f"NaN in batch[{key}]")
            if torch.isinf(val).any():
                raise ValueError(f"Inf in batch[{key}]")
    return batch
```

## Recommended Training Configuration

For stable training with MGIoU polygon loss:

```yaml
# training_config.yaml

# Learning rate
lr0: 0.001              # Initial learning rate (conservative)
lrf: 0.0001             # Final learning rate
momentum: 0.9           # SGD momentum
weight_decay: 0.0005    # L2 regularization

# Training stability
gradient_clip: 10.0     # Prevent gradient explosion
warmup_epochs: 3        # Gradual LR warmup
warmup_momentum: 0.8
warmup_bias_lr: 0.01

# Optimizer
optimizer: 'AdamW'      # More stable than SGD for some cases

# Mixed precision (if using)
amp: True               # But with gradient clipping

# Loss weights (if using custom weights)
box: 7.5               # Box loss weight
cls: 0.5               # Classification loss weight  
kpt: 5.0               # Keypoint loss weight
mgiou: 1.0             # MGIoU weight (don't set too high initially)
```

## Prevention Strategy

### 1. Start Conservative
```python
# Start with proven stable settings
lr = 0.001
gradient_clip = 10.0
use_mgiou = False  # Start with L2, switch to MGIoU after stable

# After 10-20 epochs of stability:
use_mgiou = True
lr = 0.003  # Gradually increase if needed
```

### 2. Monitor Training
```python
# Log key metrics
- Gradient norms per layer
- Loss values (all components)
- Weight statistics (mean, std)
- NaN/Inf checks per epoch
```

### 3. Early NaN Detection
```python
# Stop training immediately when NaN detected
if torch.isnan(loss):
    print("❌ NaN loss detected! Saving checkpoint...")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'checkpoint_before_nan.pt')
    raise RuntimeError("NaN loss - training stopped")
```

## Debug Checklist

When NaN appears:

- [ ] Enable `ULTRALYTICS_DEBUG_NAN=1` to see where NaN first appears
- [ ] Check if NaN is in raw model output or only after processing
- [ ] Inspect model weights for NaN/Inf
- [ ] Review loss curve - when did it start?
- [ ] Check if gradient clipping is enabled
- [ ] Verify learning rate isn't too high
- [ ] Ensure data has no NaN/Inf values
- [ ] Try reducing batch size (sometimes helps with memory/precision)
- [ ] Test with smaller model first (rule out architecture issues)

## Quick Fix Checklist

Try these in order:

1. ✅ Add gradient clipping: `gradient_clip: 10.0`
2. ✅ Reduce learning rate by 10x: `lr0: 0.001`
3. ✅ Restart from last good checkpoint
4. ✅ Switch to AdamW optimizer
5. ✅ Reduce batch size if using large batches
6. ✅ Disable mixed precision temporarily (`amp: False`)
7. ✅ Start with L2 loss, switch to MGIoU after stable

## Related Files
- `ultralytics/utils/loss.py` (lines 1186-1188, 1206) - NaN detection points
- `diagnose_nan_source.py` - Diagnostic script
- `NAN_PREVENTION_GUIDE.md` - General NaN prevention in loss functions

## Summary

The NaN in predictions is a **training stability issue**, not a bug in PolygonLoss or MGIoUPoly. The loss function correctly detected the problem. Fix it by:

1. **Add gradient clipping** (most important)
2. **Reduce learning rate**
3. **Restart from checkpoint** if weights are corrupted

The diagnostic checks we added will pinpoint exactly where the NaN originates.
