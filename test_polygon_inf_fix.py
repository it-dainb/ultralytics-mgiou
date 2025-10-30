"""
Test polygon decoding safety checks to prevent Inf values.

This test verifies that the polygons_decode function properly handles:
1. Extreme raw prediction values
2. Values that would overflow to Inf after multiplication
3. Gradient preservation through clamping operations
"""

import torch
import sys
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.nn.modules.head import Polygon


def test_extreme_predictions():
    """Test that extreme predictions are safely clamped."""
    print("=" * 60)
    print("Test 1: Extreme Raw Predictions")
    print("=" * 60)
    
    # Create a polygon head
    poly_head = Polygon(nc=80, np=4, ch=(128, 256, 512))
    poly_head.eval()
    
    # Simulate extreme raw predictions (as would come from cv4 layers)
    bs = 2
    num_anchors = 8400  # typical for YOLO
    npoly = 8  # 4 vertices * 2 coords
    
    # Create extreme values that would overflow without clamping
    raw_polys = torch.zeros(bs, npoly, num_anchors)
    
    # Test case 1: Values at the edge of acceptable range
    raw_polys[0, :, 0:100] = 45.0
    raw_polys[0, :, 100:200] = -45.0
    
    # Test case 2: Values beyond acceptable range (would cause Inf)
    raw_polys[1, :, 0:100] = 100.0  # Should be clamped to 50
    raw_polys[1, :, 100:200] = -100.0  # Should be clamped to -50
    
    # Test case 3: Already at Inf
    raw_polys[1, :, 200:300] = float('inf')
    raw_polys[1, :, 300:400] = float('-inf')
    
    print(f"Input shape: {raw_polys.shape}")
    print(f"Input range: [{raw_polys[torch.isfinite(raw_polys)].min():.2f}, {raw_polys[torch.isfinite(raw_polys)].max():.2f}]")
    print(f"Input has Inf: {torch.isinf(raw_polys).any().item()}")
    print(f"Input has NaN: {torch.isnan(raw_polys).any().item()}")
    
    # Decode polygons
    decoded = poly_head.polygons_decode(bs, raw_polys)
    
    print(f"\nOutput shape: {decoded.shape}")
    print(f"Output range: [{decoded.min():.2f}, {decoded.max():.2f}]")
    print(f"Output has Inf: {torch.isinf(decoded).any().item()}")
    print(f"Output has NaN: {torch.isnan(decoded).any().item()}")
    
    # Verify no Inf in output
    assert not torch.isinf(decoded).any(), "Output contains Inf values!"
    assert not torch.isnan(decoded).any(), "Output contains NaN values!"
    
    print("\n✓ Test passed: No Inf or NaN in decoded polygons")
    return True


def test_gradient_flow():
    """Test that gradients still flow through clamping."""
    print("\n" + "=" * 60)
    print("Test 2: Gradient Flow Through Clamping")
    print("=" * 60)
    
    poly_head = Polygon(nc=80, np=4, ch=(128, 256, 512))
    poly_head.eval()
    
    bs = 1
    num_anchors = 100
    npoly = 8
    
    # Create predictions that require gradient
    raw_polys = torch.randn(bs, npoly, num_anchors, requires_grad=True)
    
    # Add some extreme values
    with torch.no_grad():
        raw_polys[:, :, 0:10] = 60.0  # Will be clamped
        raw_polys[:, :, 10:20] = -60.0  # Will be clamped
    
    print(f"Input range: [{raw_polys.min():.2f}, {raw_polys.max():.2f}]")
    
    # Decode
    decoded = poly_head.polygons_decode(bs, raw_polys)
    
    # Compute a simple loss
    loss = decoded.sum()
    
    print(f"Loss: {loss.item():.2f}")
    
    # Backprop
    loss.backward()
    
    # Check gradients
    has_grad = raw_polys.grad is not None
    if has_grad:
        grad_finite = torch.isfinite(raw_polys.grad).all()
        grad_nonzero = (raw_polys.grad.abs() > 0).any()
        
        print(f"\nGradient exists: {has_grad}")
        print(f"Gradient is finite: {grad_finite.item()}")
        print(f"Gradient is non-zero: {grad_nonzero.item()}")
        print(f"Gradient range: [{raw_polys.grad.min():.6f}, {raw_polys.grad.max():.6f}]")
        
        assert grad_finite, "Gradients contain Inf/NaN!"
        assert grad_nonzero, "Gradients are all zero!"
        
        print("\n✓ Test passed: Gradients flow correctly through clamping")
        return True
    else:
        print("\n✗ Test failed: No gradients!")
        return False


def test_realistic_scenario():
    """Test with realistic prediction values from a partially trained model."""
    print("\n" + "=" * 60)
    print("Test 3: Realistic Scenario")
    print("=" * 60)
    
    poly_head = Polygon(nc=80, np=4, ch=(128, 256, 512))
    poly_head.eval()
    
    bs = 4
    num_anchors = 8400
    npoly = 8
    
    # Simulate realistic predictions (centered around 0, stddev ~5)
    raw_polys = torch.randn(bs, npoly, num_anchors) * 5.0
    
    # Add a few outliers (simulating numerical instability)
    outlier_indices = torch.randint(0, num_anchors, (20,))
    raw_polys[0, :, outlier_indices] = 75.0  # Should be clamped
    
    print(f"Input shape: {raw_polys.shape}")
    print(f"Input mean: {raw_polys.mean():.2f}, std: {raw_polys.std():.2f}")
    print(f"Input range: [{raw_polys.min():.2f}, {raw_polys.max():.2f}]")
    print(f"Outliers > 50: {(raw_polys.abs() > 50).sum().item()}")
    
    # Decode
    decoded = poly_head.polygons_decode(bs, raw_polys)
    
    print(f"\nOutput shape: {decoded.shape}")
    print(f"Output range: [{decoded.min():.2f}, {decoded.max():.2f}]")
    print(f"Output has Inf: {torch.isinf(decoded).any().item()}")
    print(f"Output has NaN: {torch.isnan(decoded).any().item()}")
    
    # Verify results
    assert not torch.isinf(decoded).any(), "Output contains Inf!"
    assert not torch.isnan(decoded).any(), "Output contains NaN!"
    assert decoded.abs().max() < 1e5, "Output exceeds maximum bound!"
    
    print("\n✓ Test passed: Realistic predictions handled correctly")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("POLYGON INF FIX VERIFICATION")
    print("=" * 60)
    print("\nTesting polygon decoding safety checks...")
    print("This verifies the fix in ultralytics/nn/modules/head.py:464-481\n")
    
    try:
        test_extreme_predictions()
        test_gradient_flow()
        test_realistic_scenario()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe polygon decode function successfully:")
        print("  1. Clamps extreme raw predictions to [-50, 50]")
        print("  2. Clamps decoded coordinates to [-1e5, 1e5]")
        print("  3. Preserves gradient flow through clamping")
        print("  4. Prevents Inf propagation to downstream operations")
        print("\nTraining validation should now complete without Inf errors.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
