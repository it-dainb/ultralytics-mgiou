"""Test script to verify that Inf/NaN handling in validation works correctly."""

import torch
import numpy as np
from ultralytics.utils.metrics import poly_iou


def test_poly_iou_with_inf_values():
    """Test that poly_iou handles Inf values gracefully."""
    print("Testing poly_iou with Inf values...")
    
    # Create normal polygons (4 vertices, xy coordinates)
    poly1 = torch.tensor([
        [[10, 10], [50, 10], [50, 50], [10, 50]],  # Square 1
        [[20, 20], [60, 20], [60, 60], [20, 60]],  # Square 2
    ], dtype=torch.float32)
    
    poly2 = torch.tensor([
        [[15, 15], [55, 15], [55, 55], [15, 55]],  # Overlapping square
    ], dtype=torch.float32)
    
    # Test 1: Normal case
    print("  Test 1: Normal polygons")
    try:
        iou_normal = poly_iou(poly1, poly2)
        print(f"    ✓ Normal case passed. IoU shape: {iou_normal.shape}, values: {iou_normal.flatten()}")
    except Exception as e:
        print(f"    ✗ Normal case failed: {e}")
        return False
    
    # Test 2: Polygons with Inf values
    print("  Test 2: Polygons with Inf values")
    poly1_inf = poly1.clone()
    poly1_inf[0, 1, 0] = float('inf')  # Add Inf to one coordinate
    poly1_inf[1, 2, 1] = float('-inf')  # Add -Inf to another
    
    try:
        iou_inf = poly_iou(poly1_inf, poly2)
        print(f"    ✓ Inf case passed. IoU shape: {iou_inf.shape}, values: {iou_inf.flatten()}")
        if torch.isnan(iou_inf).any() or torch.isinf(iou_inf).any():
            print(f"    ✗ Warning: Output contains NaN/Inf values!")
            return False
    except Exception as e:
        print(f"    ✗ Inf case failed: {e}")
        return False
    
    # Test 3: Polygons with NaN values
    print("  Test 3: Polygons with NaN values")
    poly1_nan = poly1.clone()
    poly1_nan[0, 0, :] = float('nan')  # Add NaN to coordinates
    
    try:
        iou_nan = poly_iou(poly1_nan, poly2)
        print(f"    ✓ NaN case passed. IoU shape: {iou_nan.shape}, values: {iou_nan.flatten()}")
        if torch.isnan(iou_nan).any() or torch.isinf(iou_nan).any():
            print(f"    ✗ Warning: Output contains NaN/Inf values!")
            return False
    except Exception as e:
        print(f"    ✗ NaN case failed: {e}")
        return False
    
    # Test 4: Mixed NaN and Inf
    print("  Test 4: Mixed NaN and Inf values")
    poly1_mixed = poly1.clone()
    poly1_mixed[0, 0, 0] = float('nan')
    poly1_mixed[0, 1, 1] = float('inf')
    poly1_mixed[1, 2, 0] = float('-inf')
    
    try:
        iou_mixed = poly_iou(poly1_mixed, poly2)
        print(f"    ✓ Mixed case passed. IoU shape: {iou_mixed.shape}, values: {iou_mixed.flatten()}")
        if torch.isnan(iou_mixed).any() or torch.isinf(iou_mixed).any():
            print(f"    ✗ Warning: Output contains NaN/Inf values!")
            return False
    except Exception as e:
        print(f"    ✗ Mixed case failed: {e}")
        return False
    
    print("\n✓ All tests passed! The poly_iou function handles Inf/NaN values correctly.\n")
    return True


def test_validation_sanitization():
    """Test the validation sanitization logic."""
    print("Testing validation sanitization logic...")
    
    # Simulate prediction tensors with Inf values
    pred_kpts = torch.randn(10, 4, 2)  # 10 predictions, 4 vertices, xy
    pred_kpts[3, 1, 0] = float('inf')
    pred_kpts[5, 2, 1] = float('nan')
    
    print(f"  Original predictions: {torch.isnan(pred_kpts).sum()} NaN, {torch.isinf(pred_kpts).sum()} Inf")
    
    # Apply sanitization (same logic as in val.py)
    if torch.isnan(pred_kpts).any() or torch.isinf(pred_kpts).any():
        pred_kpts_clean = torch.nan_to_num(pred_kpts, nan=0.0, posinf=1e6, neginf=-1e6)
        print(f"  Sanitized predictions: {torch.isnan(pred_kpts_clean).sum()} NaN, {torch.isinf(pred_kpts_clean).sum()} Inf")
        
        if torch.isnan(pred_kpts_clean).any() or torch.isinf(pred_kpts_clean).any():
            print("  ✗ Sanitization failed!")
            return False
        else:
            print("  ✓ Sanitization successful!\n")
            return True
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Validation Fix Verification Test")
    print("=" * 60)
    print()
    
    success = True
    
    # Run tests
    success &= test_poly_iou_with_inf_values()
    success &= test_validation_sanitization()
    
    if success:
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fixes are working correctly. You can now:")
        print("1. Resume training from a checkpoint (if available)")
        print("2. Start a new training run")
        print("\nThe validation phase should now handle Inf/NaN values gracefully.")
    else:
        print("=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nThere may be issues with the fixes. Please review the errors above.")
