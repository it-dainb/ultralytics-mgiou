#!/usr/bin/env python3
"""
Test extreme edge cases that can occur during actual training.

This script simulates scenarios that caused NaN in production:
1. Very large coordinate values (from feature map scaling)
2. Nearly collinear polygons
3. Extreme aspect ratios
4. Polygons with vertices very close together
"""

import torch
import sys
import os

# Enable debug mode to catch NaN
os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"

# Force reload
if "ultralytics.utils.loss" in sys.modules:
    del sys.modules["ultralytics.utils.loss"]

from ultralytics.utils.loss import MGIoUPoly, PolygonLoss

def test_extreme_coordinates():
    """Test with very large coordinate values (scaled feature maps)."""
    print("=" * 80)
    print("Test: Extreme Coordinate Values (Feature Map Scale)")
    print("=" * 80)
    
    # Simulate coordinates at 640x640 scale (common in YOLO)
    pred = torch.tensor([
        [[100.5, 200.3], [500.2, 201.1], [499.8, 400.7], [101.2, 399.5]],
        [[10.1, 10.2], [630.5, 11.3], [629.9, 630.1], [10.5, 628.8]],
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[105.0, 205.0], [505.0, 205.0], [505.0, 405.0], [105.0, 405.0]],
        [[15.0, 15.0], [625.0, 15.0], [625.0, 625.0], [15.0, 625.0]],
    ], dtype=torch.float32)
    
    mgiou_loss = MGIoUPoly(reduction="mean")
    
    try:
        loss = mgiou_loss(pred, target)
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ No NaN with extreme coordinates")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def test_nearly_collinear():
    """Test with nearly collinear vertices (thin polygons)."""
    print("=" * 80)
    print("Test: Nearly Collinear Vertices (Thin Polygons)")
    print("=" * 80)
    
    # Very thin rectangles (almost lines)
    pred = torch.tensor([
        [[0.0, 0.0], [100.0, 0.0], [100.0, 0.01], [0.0, 0.01]],  # Extremely thin
        [[0.0, 0.0], [50.0, 0.001], [50.0, 0.002], [0.0, 0.001]],  # Almost collinear
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[0.0, 0.0], [100.0, 0.0], [100.0, 0.1], [0.0, 0.1]],
        [[0.0, 0.0], [50.0, 0.01], [50.0, 0.02], [0.0, 0.01]],
    ], dtype=torch.float32)
    
    mgiou_loss = MGIoUPoly(reduction="mean")
    
    try:
        loss = mgiou_loss(pred, target)
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ No NaN with nearly collinear vertices")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def test_extreme_aspect_ratio():
    """Test with extreme aspect ratios."""
    print("=" * 80)
    print("Test: Extreme Aspect Ratios")
    print("=" * 80)
    
    # Very long and thin polygons
    pred = torch.tensor([
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 0.1], [0.0, 0.1]],  # 10000:1 ratio
        [[0.0, 0.0], [0.1, 0.0], [0.1, 500.0], [0.0, 500.0]],  # 1:5000 ratio
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[1.0, 0.05], [999.0, 0.05], [999.0, 0.15], [1.0, 0.15]],
        [[0.05, 1.0], [0.15, 1.0], [0.15, 499.0], [0.05, 499.0]],
    ], dtype=torch.float32)
    
    mgiou_loss = MGIoUPoly(reduction="mean")
    
    try:
        loss = mgiou_loss(pred, target)
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ No NaN with extreme aspect ratios")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def test_vertices_very_close():
    """Test with vertices that are very close but not identical."""
    print("=" * 80)
    print("Test: Vertices Very Close Together")
    print("=" * 80)
    
    # Vertices separated by tiny amounts
    eps = 1e-7
    pred = torch.tensor([
        [[0.0, 0.0], [1.0, eps], [1.0 + eps, 1.0], [eps, 1.0]],
        [[10.0, 10.0], [11.0 + eps, 10.0 + eps], [11.0, 11.0 + eps], [10.0 + eps, 11.0]],
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]],
    ], dtype=torch.float32)
    
    mgiou_loss = MGIoUPoly(reduction="mean")
    
    try:
        loss = mgiou_loss(pred, target)
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ No NaN with very close vertices")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def test_mixed_scales():
    """Test with mixed scale polygons in same batch."""
    print("=" * 80)
    print("Test: Mixed Scale Polygons in Batch")
    print("=" * 80)
    
    # Mix of very small, normal, and very large polygons
    pred = torch.tensor([
        [[0.0, 0.0], [0.001, 0.0], [0.001, 0.001], [0.0, 0.001]],  # Tiny
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],  # Normal
        [[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]],  # Large
    ], dtype=torch.float32)
    
    target = torch.tensor([
        [[0.0001, 0.0001], [0.0011, 0.0001], [0.0011, 0.0011], [0.0001, 0.0011]],
        [[0.1, 0.1], [10.1, 0.1], [10.1, 10.1], [0.1, 10.1]],
        [[1.0, 1.0], [1001.0, 1.0], [1001.0, 1001.0], [1.0, 1001.0]],
    ], dtype=torch.float32)
    
    mgiou_loss = MGIoUPoly(reduction="mean")
    
    try:
        loss = mgiou_loss(pred, target)
        print(f"  ✓ Loss computed: {loss.item():.6f}")
        print(f"  ✓ No NaN with mixed scale polygons")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def test_polygon_loss_integration():
    """Test PolygonLoss with extreme cases."""
    print("=" * 80)
    print("Test: PolygonLoss Integration with Extreme Cases")
    print("=" * 80)
    
    polygon_loss = PolygonLoss(use_mgiou=True)
    
    # Simulate actual training data with 3D keypoints
    pred_kpts = torch.tensor([
        [[100.5, 200.3, 1.0], [500.2, 201.1, 1.0], [499.8, 400.7, 1.0], [101.2, 399.5, 1.0]],
        [[0.001, 0.001, 1.0], [0.002, 0.001, 1.0], [0.002, 0.002, 1.0], [0.001, 0.002, 1.0]],
    ], dtype=torch.float32)
    
    gt_kpts = torch.tensor([
        [[105.0, 205.0, 1.0], [505.0, 205.0, 1.0], [505.0, 405.0, 1.0], [105.0, 405.0, 1.0]],
        [[0.0015, 0.0015, 1.0], [0.0025, 0.0015, 1.0], [0.0025, 0.0025, 1.0], [0.0015, 0.0025, 1.0]],
    ], dtype=torch.float32)
    
    kpt_mask = torch.ones((2, 4), dtype=torch.float32)
    area = torch.tensor([[40000.0], [0.000001]], dtype=torch.float32)  # Very different areas
    
    try:
        total_loss, mgiou_loss_val = polygon_loss(pred_kpts, gt_kpts, kpt_mask, area)
        print(f"  ✓ PolygonLoss computed: {total_loss.item():.6f}")
        print(f"  ✓ No NaN in integrated polygon loss")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    return True

def main():
    """Run all extreme edge case tests."""
    print("\n" + "=" * 80)
    print("EXTREME EDGE CASE TESTS FOR TRAINING")
    print("=" * 80)
    print()
    
    tests = [
        ("Extreme Coordinates", test_extreme_coordinates),
        ("Nearly Collinear", test_nearly_collinear),
        ("Extreme Aspect Ratio", test_extreme_aspect_ratio),
        ("Vertices Very Close", test_vertices_very_close),
        ("Mixed Scales", test_mixed_scales),
        ("PolygonLoss Integration", test_polygon_loss_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed_count}/{total_count} tests passed")
    print()
    
    if passed_count == total_count:
        print("🎉 ALL EXTREME EDGE CASE TESTS PASSED!")
        print("The implementation is robust against training scenarios.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
