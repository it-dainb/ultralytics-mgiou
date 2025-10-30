"""
Diagnostic script to investigate why polygon loss is not decreasing.

This script will help identify issues with:
1. Polygon predictions from the model
2. MGIoU loss computation
3. Gradient flow through polygon branch
4. Normalization effects from NaN fixes
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import MGIoUPoly, PolygonLoss


def test_mgiou_basic():
    """Test basic MGIoU computation with simple shapes."""
    print("=" * 80)
    print("TEST 1: Basic MGIoU Computation")
    print("=" * 80)
    
    # Create simple squares
    pred = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # unit square
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],  # 2x2 square
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # same as pred
        [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]],  # shifted square
    ], dtype=torch.float32)
    
    mgiou = MGIoUPoly(reduction="none")
    loss = mgiou(pred, target)
    
    print(f"Predictions:\n{pred}")
    print(f"Targets:\n{target}")
    print(f"MGIoU Loss: {loss}")
    print(f"Expected: [0.0 (perfect match), >0 (shifted)]")
    print()


def test_mgiou_with_weights():
    """Test MGIoU with area-based weights (as used in training)."""
    print("=" * 80)
    print("TEST 2: MGIoU with Area Weights")
    print("=" * 80)
    
    # Small and large polygons
    pred = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # area = 1
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],  # area = 100
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[0.1, 0.1], [1.1, 0.1], [1.1, 1.1], [0.1, 1.1]],  # slightly off
        [[1.0, 1.0], [11.0, 1.0], [11.0, 11.0], [1.0, 11.0]],  # slightly off
    ], dtype=torch.float32)
    
    weights = torch.tensor([1.0, 100.0])  # weights proportional to area
    
    mgiou = MGIoUPoly(reduction="mean")
    loss_unweighted = mgiou(pred, target)
    loss_weighted = mgiou(pred, target, weight=weights)
    
    print(f"Unweighted loss: {loss_unweighted:.6f}")
    print(f"Weighted loss: {loss_weighted:.6f}")
    print(f"Large polygon has {weights[1]/weights[0]:.0f}x more weight")
    print()


def test_polygon_loss_layer():
    """Test the PolygonLoss layer as used in training."""
    print("=" * 80)
    print("TEST 3: PolygonLoss Layer (use_mgiou=True)")
    print("=" * 80)
    
    # Simulate batch of predictions (N samples, 4 vertices, 2 coords)
    pred_kpts = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
    ], dtype=torch.float32, requires_grad=True)
    
    gt_kpts = torch.tensor([
        [[0.1, 0.1], [1.1, 0.1], [1.1, 1.1], [0.1, 1.1]],
        [[0.2, 0.2], [2.2, 0.2], [2.2, 2.2], [0.2, 2.2]],
        [[0.3, 0.3], [3.3, 0.3], [3.3, 3.3], [0.3, 3.3]],
    ], dtype=torch.float32)
    
    kpt_mask = torch.ones(3, 4, dtype=torch.bool)
    area = torch.tensor([[1.0], [4.0], [9.0]], dtype=torch.float32)
    
    polygon_loss = PolygonLoss(use_mgiou=True)
    total_loss, mgiou_component = polygon_loss(pred_kpts, gt_kpts, kpt_mask, area)
    
    print(f"Predicted polygons shape: {pred_kpts.shape}")
    print(f"Total loss: {total_loss:.6f}")
    print(f"MGIoU component: {mgiou_component:.6f}")
    
    # Test gradients
    total_loss.backward()
    print(f"Gradient norm: {pred_kpts.grad.norm():.6f}")
    print(f"Gradient mean: {pred_kpts.grad.mean():.6f}")
    print(f"Gradient std: {pred_kpts.grad.std():.6f}")
    print()


def test_gradient_flow():
    """Test if gradients flow correctly through MGIoU."""
    print("=" * 80)
    print("TEST 4: Gradient Flow Through MGIoU")
    print("=" * 80)
    
    # Create a scenario where we expect gradients
    pred = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ], dtype=torch.float32, requires_grad=True)
    
    target = torch.tensor([
        [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]],  # shifted by 0.5
    ], dtype=torch.float32)
    
    mgiou = MGIoUPoly(reduction="mean")
    
    # Forward pass
    loss = mgiou(pred, target)
    print(f"Initial loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"Gradient shape: {pred.grad.shape}")
    print(f"Gradient values:\n{pred.grad}")
    print(f"Gradient norm: {pred.grad.norm():.6f}")
    
    # Simulate gradient descent step
    with torch.no_grad():
        pred_new = pred - 0.1 * pred.grad
    
    # Check if loss decreased
    pred_new.requires_grad = True
    loss_new = mgiou(pred_new, target)
    print(f"\nAfter gradient step:")
    print(f"New loss: {loss_new.item():.6f}")
    print(f"Loss change: {(loss_new.item() - loss.item()):.6f}")
    print(f"Expected: negative (loss should decrease)")
    print()


def test_nan_to_num_gradient():
    """Test if nan_to_num preserves gradients."""
    print("=" * 80)
    print("TEST 5: Gradient Preservation with nan_to_num")
    print("=" * 80)
    
    # Create tensor with normal values
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Apply nan_to_num (should be identity for normal values)
    y = torch.nan_to_num(x, nan=0.0)
    
    # Simple loss
    loss = (y ** 2).mean()
    loss.backward()
    
    print(f"Input: {x}")
    print(f"After nan_to_num: {y}")
    print(f"Gradients: {x.grad}")
    print(f"Expected: [0.67, 1.33, 2.0] (2*x/3)")
    print()


def test_normalization_effect():
    """Test effect of per-image normalization on loss magnitude."""
    print("=" * 80)
    print("TEST 6: Normalization Effect")
    print("=" * 80)
    
    # Simulate losses from multiple images
    batch_size = 8
    per_image_loss = torch.tensor([0.67] * batch_size)
    
    # Old method: sum without normalization
    old_loss = per_image_loss.sum()
    old_weighted = old_loss * batch_size * 12.0  # hyp.polygon = 12
    
    # New method: normalize by num_images, then multiply by batch_size
    new_loss = per_image_loss.sum() / batch_size
    new_weighted = new_loss * batch_size * 12.0  # hyp.polygon = 12
    
    print(f"Per-image loss: {per_image_loss[0]:.2f}")
    print(f"\nOld method (no normalization):")
    print(f"  Raw loss: {old_loss:.2f}")
    print(f"  After batch_size multiplication: {old_loss * batch_size:.2f}")
    print(f"  After weight multiplication: {old_weighted:.2f}")
    print(f"\nNew method (with normalization):")
    print(f"  Raw loss: {new_loss:.2f}")
    print(f"  After batch_size multiplication: {new_loss * batch_size:.2f}")
    print(f"  After weight multiplication: {new_weighted:.2f}")
    print(f"\nDifference: {old_weighted / new_weighted:.1f}x")
    print()


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("POLYGON LOSS INVESTIGATION")
    print("=" * 80 + "\n")
    
    test_mgiou_basic()
    test_mgiou_with_weights()
    test_polygon_loss_layer()
    test_gradient_flow()
    test_nan_to_num_gradient()
    test_normalization_effect()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Based on the training logs you provided:
- polygon_loss fluctuates around 0.65-0.73 and does NOT decrease
- mgiou_loss shows as 0.6913, 0.6788, etc. (same as polygon_loss when use_mgiou=True)
- Box and cls losses show some learning, but polygon doesn't improve

Possible issues to investigate:
1. Are polygon predictions actually changing? (Check model weights)
2. Is MGIoU loss saturating? (Check if GIoU values are constant)
3. Is learning rate too low for polygon head?
4. Is the normalization fix causing gradient scale issues?
5. Are NaN prevention measures too aggressive? (nan_to_num might clip gradients)

Next steps:
1. Add logging to track polygon prediction changes over epochs
2. Monitor GIoU values (not just loss) to see if shapes are improving
3. Check gradient magnitudes for polygon head vs other components
4. Try training without use_mgiou to compare behavior
5. Verify that polygon ground truth data is correct and scaled properly
    """)


if __name__ == "__main__":
    main()
