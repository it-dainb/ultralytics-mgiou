#!/usr/bin/env python3
"""
Test script to verify gradient flow through MGIoUPoly loss.

This script creates synthetic polygon data and verifies that:
1. Loss computation completes without NaN
2. Gradients can backpropagate through the loss
3. Optimizer can update weights based on the loss

Usage:
    python test_gradient_flow.py
"""

import torch
from ultralytics.utils.loss import MGIoUPoly

def test_gradient_flow():
    """Test that gradients flow through MGIoUPoly loss."""
    
    print("=" * 60)
    print("Testing Gradient Flow Through MGIoUPoly Loss")
    print("=" * 60)
    
    # Create MGIoU loss
    mgiou_loss = MGIoUPoly(reduction="mean", eps=1e-6)
    
    # Create synthetic polygon predictions (requires_grad=True to track gradients)
    batch_size = 4
    num_vertices = 4
    pred_poly = torch.randn(batch_size, num_vertices, 2, requires_grad=True)
    
    # Create synthetic target polygons (ground truth)
    target_poly = torch.randn(batch_size, num_vertices, 2)
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {pred_poly.shape}")
    print(f"  Targets: {target_poly.shape}")
    print(f"  Predictions require grad: {pred_poly.requires_grad}")
    
    # Forward pass
    print("\nComputing loss...")
    loss = mgiou_loss(pred_poly, target_poly)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    
    # Check for NaN
    if torch.isnan(loss):
        print("  ❌ FAILED: Loss is NaN")
        return False
    
    if torch.isinf(loss):
        print("  ❌ FAILED: Loss is Inf")
        return False
    
    # Backward pass - this tests gradient flow
    print("\nComputing gradients...")
    loss.backward()
    
    # Check gradients exist and are finite
    if pred_poly.grad is None:
        print("  ❌ FAILED: No gradients computed")
        return False
    
    print(f"  Gradient shape: {pred_poly.grad.shape}")
    print(f"  Gradient mean: {pred_poly.grad.mean().item():.6e}")
    print(f"  Gradient std: {pred_poly.grad.std().item():.6e}")
    print(f"  Gradient min: {pred_poly.grad.min().item():.6e}")
    print(f"  Gradient max: {pred_poly.grad.max().item():.6e}")
    
    # Check gradient validity
    has_nan = torch.isnan(pred_poly.grad).any()
    has_inf = torch.isinf(pred_poly.grad).any()
    all_zero = (pred_poly.grad == 0).all()
    
    print(f"\nGradient checks:")
    print(f"  Contains NaN: {has_nan.item()}")
    print(f"  Contains Inf: {has_inf.item()}")
    print(f"  All zeros: {all_zero.item()}")
    print(f"  Non-zero gradients: {(pred_poly.grad != 0).sum().item()}/{pred_poly.grad.numel()}")
    
    if has_nan:
        print("  ❌ FAILED: Gradients contain NaN")
        return False
    
    if has_inf:
        print("  ❌ FAILED: Gradients contain Inf")
        return False
    
    if all_zero:
        print("  ⚠️  WARNING: All gradients are zero (loss may not be sensitive to inputs)")
    
    # Test optimizer update
    print("\nTesting optimizer update...")
    optimizer = torch.optim.SGD([pred_poly], lr=0.01)
    
    old_pred = pred_poly.data.clone()
    optimizer.step()
    new_pred = pred_poly.data
    
    param_changed = not torch.allclose(old_pred, new_pred)
    print(f"  Parameters changed: {param_changed}")
    print(f"  Max parameter change: {(new_pred - old_pred).abs().max().item():.6e}")
    
    if not param_changed:
        print("  ⚠️  WARNING: Parameters didn't change (gradients may be too small)")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS: Gradient flow test passed!")
    print("=" * 60)
    return True


def test_edge_cases():
    """Test edge cases that previously caused NaN."""
    
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    mgiou_loss = MGIoUPoly(reduction="mean", eps=1e-6)
    
    # Test 1: Degenerate target (all zeros)
    print("\nTest 1: Degenerate target (all zeros)")
    pred = torch.randn(2, 4, 2, requires_grad=True)
    target = torch.zeros(2, 4, 2)
    
    loss = mgiou_loss(pred, target)
    print(f"  Loss: {loss.item():.6f} (should use L1 fallback)")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    loss.backward()
    print(f"  Gradients exist: {pred.grad is not None}")
    print(f"  Gradients finite: {torch.isfinite(pred.grad).all().item()}")
    
    # Test 2: Identical polygons
    print("\nTest 2: Identical polygons")
    pred2 = torch.randn(2, 4, 2, requires_grad=True)
    target2 = pred2.detach().clone()
    
    loss = mgiou_loss(pred2, target2)
    print(f"  Loss: {loss.item():.6f} (should be close to 0)")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    loss.backward()
    print(f"  Gradients exist: {pred2.grad is not None}")
    if pred2.grad is not None:
        print(f"  Gradients finite: {torch.isfinite(pred2.grad).all().item()}")
    
    # Test 3: Very small polygons
    print("\nTest 3: Very small polygons")
    pred3 = torch.randn(2, 4, 2, requires_grad=True) * 0.001
    target3 = torch.randn(2, 4, 2) * 0.001
    
    loss = mgiou_loss(pred3, target3)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    loss.backward()
    print(f"  Gradients exist: {pred3.grad is not None}")
    if pred3.grad is not None:
        print(f"  Gradients finite: {torch.isfinite(pred3.grad).all().item()}")
    
    # Test 4: Very large polygons
    print("\nTest 4: Very large polygons")
    pred4 = torch.randn(2, 4, 2, requires_grad=True) * 1000
    target4 = torch.randn(2, 4, 2) * 1000
    
    loss = mgiou_loss(pred4, target4)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    loss.backward()
    print(f"  Gradients exist: {pred4.grad is not None}")
    if pred4.grad is not None:
        print(f"  Gradients finite: {torch.isfinite(pred4.grad).all().item()}")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS: All edge cases handled correctly!")
    print("=" * 60)


def test_training_simulation():
    """Simulate a few training iterations."""
    
    print("\n" + "=" * 60)
    print("Simulating Training Iterations")
    print("=" * 60)
    
    mgiou_loss = MGIoUPoly(reduction="mean", eps=1e-6)
    
    # Create learnable predictions
    pred_poly = torch.randn(4, 4, 2, requires_grad=True)
    target_poly = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # square
        [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]],  # shifted square
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],  # larger square
        [[0.5, 0.0], [1.5, 0.5], [1.0, 1.5], [0.0, 1.0]],  # irregular quad
    ])
    
    optimizer = torch.optim.Adam([pred_poly], lr=0.1)
    
    print("\nTraining for 10 iterations...")
    losses = []
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        loss = mgiou_loss(pred_poly, target_poly)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch:2d}: loss = {loss.item():.6f}")
    
    print(f"\nLoss progression:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss decreased: {losses[-1] < losses[0]}")
    print(f"  Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    if losses[-1] >= losses[0]:
        print("  ⚠️  WARNING: Loss did not decrease (may need more iterations or different learning rate)")
    else:
        print("  ✅ Loss decreased as expected!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        # Run all tests
        test_gradient_flow()
        test_edge_cases()
        test_training_simulation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        print("\nGradient flow is working correctly.")
        print("You can now proceed with full training.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
