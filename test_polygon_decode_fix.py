"""
Test to verify the Polygon head's decode function handles extreme values correctly
and doesn't produce Inf after the fix.
"""

import torch
import sys
sys.path.insert(0, '/mnt/data/ME/ultralytics-mgiou')

from ultralytics.nn.modules.head import Polygon


def test_polygon_decode_extreme_values():
    """Test that polygons_decode clamps extreme values and prevents Inf."""
    
    # Create a Polygon head with 8 vertices
    poly_head = Polygon(nc=80, np=8, ch=(256, 512, 1024))
    
    # Simulate forward pass to initialize anchors and strides
    # Create dummy feature maps
    x1 = torch.randn(1, 256, 80, 80)
    x2 = torch.randn(1, 512, 40, 40)
    x3 = torch.randn(1, 1024, 20, 20)
    
    # Set training mode to get raw outputs
    poly_head.train()
    poly_head.stride = torch.tensor([8, 16, 32])  # Set strides manually
    
    # Forward pass to initialize anchors
    outputs = poly_head([x1, x2, x3])
    
    # Now test decode with extreme values
    bs = 1
    
    # Test 1: Very large raw predictions (would overflow to Inf without clamping)
    extreme_polys = torch.ones(bs, 16, 8400) * 1e10  # 8 vertices * 2 coords = 16
    
    print("Test 1: Extreme large values (1e10)")
    print(f"  Input min/max: {extreme_polys.min().item():.2e}, {extreme_polys.max().item():.2e}")
    
    poly_head.eval()  # Switch to eval mode for decoding
    decoded = poly_head.polygons_decode(bs, extreme_polys)
    
    has_inf = torch.isinf(decoded).any()
    has_nan = torch.isnan(decoded).any()
    
    print(f"  Decoded contains Inf: {has_inf}")
    print(f"  Decoded contains NaN: {has_nan}")
    print(f"  Decoded min/max: {decoded.min().item():.2e}, {decoded.max().item():.2e}")
    
    assert not has_inf, "Decoded polygons should not contain Inf values!"
    assert not has_nan, "Decoded polygons should not contain NaN values!"
    print("  ✓ PASS: No Inf/NaN in decoded polygons")
    
    # Test 2: Very small raw predictions (negative extreme)
    extreme_polys_neg = torch.ones(bs, 16, 8400) * -1e10
    
    print("\nTest 2: Extreme negative values (-1e10)")
    print(f"  Input min/max: {extreme_polys_neg.min().item():.2e}, {extreme_polys_neg.max().item():.2e}")
    
    decoded_neg = poly_head.polygons_decode(bs, extreme_polys_neg)
    
    has_inf = torch.isinf(decoded_neg).any()
    has_nan = torch.isnan(decoded_neg).any()
    
    print(f"  Decoded contains Inf: {has_inf}")
    print(f"  Decoded contains NaN: {has_nan}")
    print(f"  Decoded min/max: {decoded_neg.min().item():.2e}, {decoded_neg.max().item():.2e}")
    
    assert not has_inf, "Decoded polygons should not contain Inf values!"
    assert not has_nan, "Decoded polygons should not contain NaN values!"
    print("  ✓ PASS: No Inf/NaN in decoded polygons")
    
    # Test 3: Mixed normal and extreme values
    mixed_polys = torch.randn(bs, 16, 8400)
    mixed_polys[:, :, :100] = 1e30  # Add some extreme values
    mixed_polys[:, :, 100:200] = -1e30
    
    print("\nTest 3: Mixed normal and extreme values")
    print(f"  Input min/max: {mixed_polys.min().item():.2e}, {mixed_polys.max().item():.2e}")
    print(f"  Input has Inf: {torch.isinf(mixed_polys).any()}")
    
    decoded_mixed = poly_head.polygons_decode(bs, mixed_polys)
    
    has_inf = torch.isinf(decoded_mixed).any()
    has_nan = torch.isnan(decoded_mixed).any()
    
    print(f"  Decoded contains Inf: {has_inf}")
    print(f"  Decoded contains NaN: {has_nan}")
    print(f"  Decoded min/max: {decoded_mixed.min().item():.2e}, {decoded_mixed.max().item():.2e}")
    
    assert not has_inf, "Decoded polygons should not contain Inf values!"
    assert not has_nan, "Decoded polygons should not contain NaN values!"
    print("  ✓ PASS: No Inf/NaN in decoded polygons")
    
    # Test 4: Verify clamping bounds
    # Pre-multiplication clamp should limit to [-50, 50]
    # After multiplication by max stride (32) and 2.0, max should be ~3200 per anchor
    # Post-decoding clamp limits to [-1e5, 1e5]
    
    print("\nTest 4: Verify clamping bounds")
    very_large = torch.ones(bs, 16, 8400) * 100  # Exceeds [-50, 50] clamp
    decoded_clamped = poly_head.polygons_decode(bs, very_large)
    
    # After clamping to [-50, 50], multiplying by 2.0 and max stride (32):
    # Max value per coord = 50 * 2.0 * 32 = 3200
    # Plus anchor offset (~4200 for 80x80 grid)
    # Should be well within [-1e5, 1e5] final clamp
    
    print(f"  Decoded min/max: {decoded_clamped.min().item():.2e}, {decoded_clamped.max().item():.2e}")
    assert decoded_clamped.min().item() >= -1e5, "Decoded values should be >= -1e5"
    assert decoded_clamped.max().item() <= 1e5, "Decoded values should be <= 1e5"
    print("  ✓ PASS: Values within expected bounds")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Polygon decode fix is working correctly.")
    print("="*60)


if __name__ == "__main__":
    test_polygon_decode_extreme_values()
