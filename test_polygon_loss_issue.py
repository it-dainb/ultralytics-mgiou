"""
Test to verify if there's a double-mean issue in PolygonLoss.
"""

import torch
from ultralytics.utils.loss import MGIoUPoly, PolygonLoss

print("="*80)
print("Testing PolygonLoss for Double-Mean Issue")
print("="*80)

torch.manual_seed(42)

# Test 1: MGIoUPoly directly with reduction="mean"
print("\nTest 1: MGIoUPoly with reduction='mean'")
mgiou_mean = MGIoUPoly(reduction="mean")
pred = torch.randn(4, 4, 2)
target = torch.randn(4, 4, 2)
weights = torch.tensor([10.0, 20.0, 30.0, 40.0])

loss_mean = mgiou_mean(pred, target, weight=weights)
print(f"  Loss (reduction='mean'): {loss_mean.item():.6f}")
print(f"  Loss shape: {loss_mean.shape}")

# Test 2: MGIoUPoly with reduction="none" 
print("\nTest 2: MGIoUPoly with reduction='none'")
mgiou_none = MGIoUPoly(reduction="none")
loss_none = mgiou_none(pred, target, weight=weights)
print(f"  Loss (reduction='none'): {loss_none}")
print(f"  Loss shape: {loss_none.shape}")
print(f"  Manual mean of none: {loss_none.mean().item():.6f}")

# Test 3: PolygonLoss implementation
print("\nTest 3: PolygonLoss with MGIoU mode")
poly_loss = PolygonLoss(use_mgiou=True)

# Check what reduction mode it uses internally
print(f"  Internal MGIoUPoly reduction: {poly_loss.mgiou_loss.reduction}")

pred_kpts = pred.unsqueeze(-1)  # Add dummy dimension for 3D
pred_kpts = torch.cat([pred, torch.zeros(4, 4, 1)], dim=-1)  # [4, 4, 3]
gt_kpts = torch.cat([target, torch.zeros(4, 4, 1)], dim=-1)
kpt_mask = torch.ones(4, 4)
area = weights.unsqueeze(-1)  # Use weights as areas

total_loss, mgiou_loss = poly_loss(pred_kpts, gt_kpts, kpt_mask, area)
print(f"  PolygonLoss total_loss: {total_loss.item():.6f}")
print(f"  PolygonLoss mgiou_loss: {mgiou_loss.item():.6f}")

# Test 4: Reproduce PolygonLoss logic step by step
print("\nTest 4: Reproduce PolygonLoss logic")
pred_poly = pred_kpts[..., :2]  # [4, 4, 2]
gt_poly = gt_kpts[..., :2]
weights_poly = area.squeeze(-1)  # [4]

print(f"  pred_poly shape: {pred_poly.shape}")
print(f"  gt_poly shape: {gt_poly.shape}")
print(f"  weights_poly shape: {weights_poly.shape}")
print(f"  weights_poly values: {weights_poly}")

# This is what PolygonLoss does (line 843)
mgiou_losses = poly_loss.mgiou_loss(pred_poly, gt_poly, weight=weights_poly)
print(f"  mgiou_losses from self.mgiou_loss(): {mgiou_losses}")
print(f"  mgiou_losses shape: {mgiou_losses.shape}")

# Then it does .mean() again (line 845)
total_loss_computed = mgiou_losses.mean()
print(f"  total_loss after .mean(): {total_loss_computed}")

print("\n" + "="*80)
print("Analysis")
print("="*80)

# The issue: PolygonLoss creates MGIoUPoly with reduction="mean" (line 812)
# Then calls .mean() on the result (line 845), but since reduction="mean", 
# the result is already a scalar, so calling .mean() again is redundant but not wrong

if mgiou_losses.shape == torch.Size([]):
    print("  ✓ mgiou_losses is already a scalar (reduction='mean' applied)")
    print("  ✓ Calling .mean() on a scalar just returns the scalar")
    print("  ✓ No double-mean issue - just redundant but harmless")
else:
    print("  ✗ mgiou_losses is not a scalar!")
    print("  ✗ This would cause double-mean issue")
