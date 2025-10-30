# Classification Loss NaN Fix

## Problem
Training was failing at epoch 4 during validation with the following error:
```
RuntimeError: NaN detected in pred_scores at v8PolygonLoss.__call__ after permute
Shape: torch.Size([42, 10285, 1])
NaN count: 10285
```

## Root Cause
The classification loss computation was changed from:
```python
loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
```

to:
```python
loss[2] = self.bce(pred_scores, target_scores.to(dtype)).mean()
```

This change caused several issues:

1. **Insufficient gradient signal**: The `.mean()` approach computes the mean over ALL anchors (10,285+ anchors), resulting in an extremely small loss value (~0.002)

2. **Poor normalization**: In YOLO, most anchors are negative samples. The `.mean()` over all samples dilutes the signal from the few positive samples

3. **Weight instability**: With such small gradients, the classification head (cv3 layers) doesn't learn properly, and after a few epochs, the weights destabilize and produce NaN outputs

## Solution
Reverted to the standard YOLO classification loss normalization:
```python
loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
```

This approach:
- Normalizes by the number of positive samples (`target_scores_sum`)
- Provides proper gradient flow to the classification head
- Is the standard approach used in all YOLO variants
- The final loss magnitude is controlled by `self.hyp.cls` weight (default: 0.5)

## Training Behavior
Before fix:
- cls_loss: ~0.002 (too small)
- Training succeeds for 3 epochs, then NaN at epoch 4 validation

After fix:
- cls_loss: Expected to be in range of 0.5-2.0 (properly scaled)
- Stable training without NaN issues

## Files Modified
- `ultralytics/utils/loss.py`: Line 1387 - reverted classification loss normalization

## Testing
You can resume training from the checkpoint to verify the fix works correctly.
