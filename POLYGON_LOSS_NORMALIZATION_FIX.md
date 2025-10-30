# Polygon Loss Normalization Fix

## Problem

The polygon loss in `v8PolygonLoss.calculate_polygon_loss()` was not being normalized correctly, causing it to scale improperly with batch size. This led to extremely slow training convergence for polygon predictions.

## Root Cause Analysis

### Issue 1: No Batch Normalization

**Location**: `ultralytics/utils/loss.py:1421-1443` (before fix)

The `calculate_polygon_loss()` method accumulated polygon losses across all images in the batch but never divided by the number of images:

```python
polys_loss = torch.zeros(1, device=self.device)

for i in range(pred_poly.shape[0]):  # Loop over batch
    fg_mask_i = masks[i]
    if fg_mask_i.sum():
        # ... calculations ...
        poly_loss, _ = self.polygon_loss(pred_poly_i, gt_poly_scaled, poly_mask, area)
        polys_loss += poly_loss  # Accumulates without dividing by batch size!

return polys_loss  # Returns sum, not average
```

**Result**: For batch_size=8, polygon loss was 8× larger than intended.

### Issue 2: Batch Size Multiplication at Return

**Location**: `ultralytics/utils/loss.py:1380`

The `v8PolygonLoss.__call__()` method multiplies the entire loss vector by batch_size before returning:

```python
return loss * batch_size, loss.detach()
```

**Combined Effect**: 
- Polygon loss accumulates across batch → ×batch_size
- Then gets multiplied by batch_size again → ×batch_size
- **Total scaling: ×batch_size²**

Example with batch_size=8:
- Per-image polygon loss: ~1.44 (11.5 / 8)
- After accumulation: 11.5 (×8)
- After return multiplication: 92.0 (×8 again)
- **Effective scaling: ×64 instead of ×8**

### Comparison with Classification Loss

After our previous fix, classification loss uses `mean()` reduction (line 1353):

```python
loss[2] = self.bce(pred_scores, target_scores.to(dtype)).mean()
```

This gives:
- Classification loss: ~0.8 (mean across all 67,200 anchors)
- After batch_size multiplication: 6.4 (0.8 × 8)
- **Correct scaling: ×8**

## The Fix

**Modified**: `ultralytics/utils/loss.py:1421-1450`

Added batch normalization to `calculate_polygon_loss()`:

```python
batch_idx = batch_idx.flatten()
polys_loss = torch.zeros(1, device=self.device)
num_images_with_fg = 0  # NEW: Track number of images with foreground

for i in range(pred_poly.shape[0]):
    fg_mask_i = masks[i]
    if fg_mask_i.sum():
        num_images_with_fg += 1  # NEW: Increment counter
        target_gt_idx_i = target_gt_idx[i][fg_mask_i]
        gt_matching_bs = batch_idx[target_gt_idx_i].long()
        gt_poly_scaled = gt_poly[gt_matching_bs]
        
        gt_poly_scaled[..., 0] /= stride_tensor[i]
        gt_poly_scaled[..., 1] /= stride_tensor[i]
        
        area = xyxy2xywh(target_bboxes[i][fg_mask_i])[:, 2:].prod(1, keepdim=True)
        pred_poly_i = pred_poly[i][fg_mask_i]
        poly_mask = torch.full_like(gt_poly_scaled[..., 0], True)
        poly_loss, _ = self.polygon_loss(pred_poly_i, gt_poly_scaled, poly_mask, area)
        polys_loss += poly_loss

# NEW: Normalize by number of images with foreground instances
# This ensures loss doesn't scale linearly with batch size
# The final loss * batch_size multiplication at return (line 1380) will then give correct scaling
if num_images_with_fg > 0:
    polys_loss = polys_loss / num_images_with_fg

return polys_loss
```

## Why This Works

### Before Fix (WRONG)

With batch_size=8, assuming each image has similar polygon loss ~1.44:

1. **Accumulation**: `polys_loss = 1.44 + 1.44 + ... (8 times) = 11.5`
2. **Return**: `loss[1] = 11.5 × 8 = 92.0`
3. **Weighted**: `loss[1] = 92.0 × 12.0 (hyp.polygon) = 1104.0`

### After Fix (CORRECT)

With batch_size=8:

1. **Accumulation**: `polys_loss = 1.44 + 1.44 + ... (8 times) = 11.5`
2. **Normalization**: `polys_loss = 11.5 / 8 = 1.44`
3. **Return**: `loss[1] = 1.44 × 8 = 11.5`
4. **Weighted**: `loss[1] = 11.5 × 12.0 (hyp.polygon) = 138.0`

Now the polygon loss scales correctly with batch_size (×8), just like classification loss.

## Impact on Training

### Loss Magnitude Comparison

With `hyp.polygon=12.0` and `hyp.cls=0.5`, after the fix:

| Loss Component | Raw Value | Weight | Weighted × batch_size | Relative Scale |
|----------------|-----------|--------|----------------------|----------------|
| Classification | 0.8       | 0.5    | 3.2                  | 1×            |
| Polygon        | 1.44      | 12.0   | 138.0                | 43×           |

The polygon loss is now ~43× larger than classification, which is much more reasonable than the previous ~345× imbalance.

### Expected Training Improvement

1. **Faster convergence**: Polygon loss gradients are now properly scaled
2. **Stable loss values**: Loss no longer depends on batch size
3. **Better gradient balance**: Polygon and classification losses can learn together

### Gradient Flow

With the previous gradient ratio of 11,600:1 (cls:poly) caused by high classification loss, and the new classification fix bringing it to ~13:1, this polygon normalization fix further improves training by ensuring the polygon loss component itself is correctly normalized.

## Recommended Hyperparameter Adjustments

After this fix, you may want to adjust the loss weights since the effective polygon loss is now ~8× smaller (for batch_size=8):

**Current weights** (from training logs):
```
box=7.5, cls=0.5, polygon=12.0
```

**Suggested adjustments**:
- Option 1 (maintain relative balance): Keep as-is, the fix corrects the scaling
- Option 2 (boost classification): Increase `cls` to 30-50 for stronger classification focus
- Option 3 (balanced approach): `cls=20.0, polygon=12.0` for ~1:1 weighted ratio

## Testing

To verify the fix works correctly, compare losses with different batch sizes:

```python
# With batch_size=8
# Expected: polygon loss ~1.44 (per-image average)

# With batch_size=16  
# Expected: polygon loss ~1.44 (same per-image average)
```

The per-image polygon loss should remain constant regardless of batch size.

## Related Fixes

This fix builds on the classification loss normalization fix:

1. **Classification Loss Fix** (`CLS_LOSS_NORMALIZATION_FIX.md`): Changed from `sum()/num_positives` to `mean()` to prevent extremely high classification loss
2. **Polygon Loss Fix** (this document): Normalized polygon loss accumulation to prevent batch size scaling issues

Together, these fixes ensure all loss components are properly normalized and can train effectively.

## Files Modified

- `ultralytics/utils/loss.py:1421-1450` - Added batch normalization to `calculate_polygon_loss()`

## Verification

Run training and observe:
- Polygon loss should start around 1-2 (not 11-12)
- Polygon loss should decrease steadily each epoch
- Loss values should be independent of batch size
