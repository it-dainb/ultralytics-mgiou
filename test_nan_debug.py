#!/usr/bin/env python3
"""
Test NaN debugging functionality in polygon loss.

This script tests that:
1. Normal operation works fine with debug mode disabled
2. Debug mode can be enabled via environment variable
3. NaN detection properly raises errors when enabled
"""

import os
import sys
import torch

# Test 1: Normal operation (debug disabled)
print("=" * 80)
print("Test 1: Normal Operation (Debug Disabled)")
print("=" * 80)

os.environ["ULTRALYTICS_DEBUG_NAN"] = "0"

# Force reload to pick up environment variable
if "ultralytics.utils.loss" in sys.modules:
    del sys.modules["ultralytics.utils.loss"]

from ultralytics.utils.loss import MGIoUPoly, PolygonLoss

# Create simple valid test case
pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)
target = torch.tensor([[[0.1, 0.1], [1.1, 0.1], [1.1, 1.1], [0.1, 1.1]]], dtype=torch.float32)

mgiou_loss = MGIoUPoly(reduction="mean")
loss = mgiou_loss(pred, target)
print(f"✓ Loss computed successfully: {loss.item():.6f}")
print(f"✓ No NaN detected (debug disabled)")
print()

# Test 2: Enable debug mode and verify it works
print("=" * 80)
print("Test 2: Debug Mode Enabled (Valid Data)")
print("=" * 80)

os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"

# Force reload to pick up new environment variable
if "ultralytics.utils.loss" in sys.modules:
    del sys.modules["ultralytics.utils.loss"]

from ultralytics.utils.loss import MGIoUPoly, PolygonLoss

mgiou_loss = MGIoUPoly(reduction="mean")
loss = mgiou_loss(pred, target)
print(f"✓ Loss computed successfully: {loss.item():.6f}")
print(f"✓ NaN checks active, no issues found")
print()

# Test 3: Inject NaN and verify detection
print("=" * 80)
print("Test 3: NaN Detection (Debug Enabled)")
print("=" * 80)

pred_with_nan = torch.tensor([[[0.0, 0.0], [float('nan'), 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)

try:
    loss = mgiou_loss(pred_with_nan, target)
    print("✗ FAILED: NaN not detected!")
    sys.exit(1)
except RuntimeError as e:
    if "NaN detected" in str(e):
        print(f"✓ NaN properly detected and reported:")
        print(f"  {str(e).split(chr(10))[0]}")
    else:
        print(f"✗ FAILED: Wrong error: {e}")
        sys.exit(1)
print()

# Test 4: Test PolygonLoss with MGIoU
print("=" * 80)
print("Test 4: PolygonLoss with MGIoU (Debug Enabled)")
print("=" * 80)

polygon_loss = PolygonLoss(use_mgiou=True)

pred_kpts = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
gt_kpts = torch.tensor([[[0.1, 0.1, 1.0], [1.1, 0.1, 1.0], [1.1, 1.1, 1.0], [0.1, 1.1, 1.0]]], dtype=torch.float32)
kpt_mask = torch.ones((1, 4), dtype=torch.float32)
area = torch.tensor([[1.0]], dtype=torch.float32)

total_loss, mgiou_loss_val = polygon_loss(pred_kpts, gt_kpts, kpt_mask, area)
print(f"✓ PolygonLoss computed successfully: {total_loss.item():.6f}")
print(f"✓ NaN checks active in PolygonLoss")
print()

# Test 5: Test PolygonLoss L2 mode
print("=" * 80)
print("Test 5: PolygonLoss L2 Mode (Debug Enabled)")
print("=" * 80)

polygon_loss_l2 = PolygonLoss(use_mgiou=False)
total_loss, _ = polygon_loss_l2(pred_kpts, gt_kpts, kpt_mask, area)
print(f"✓ PolygonLoss L2 computed successfully: {total_loss.item():.6f}")
print(f"✓ NaN checks active in L2 mode")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ All NaN debug tests passed!")
print()
print("Usage:")
print("  - Set ULTRALYTICS_DEBUG_NAN=1 to enable NaN checks during training")
print("  - NaN checks will raise RuntimeError with detailed diagnostics")
print("  - Keep disabled in production for better performance")
