"""Test enhanced debug output with detailed intermediate values."""
import os
import torch

# Enable debug mode
os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"

from ultralytics.utils.loss import MGIoUPoly

print("="*80)
print("Testing Enhanced Debug Output")
print("="*80)

# Create a case that might cause NaN (extreme values)
mgiou = MGIoUPoly(reduction="mean", loss_weight=1.0)

# Create polygons with one that will produce NaN in intermediate computation
pred = torch.tensor([
    [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],  # Normal polygon
    [[0.0, 0.0], [1e-10, 0.0], [1e-10, 1e-10], [0.0, 1e-10]],  # Tiny polygon (may cause issues)
]).float()

target = torch.tensor([
    [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
    [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]],
]).float()

# Inject NaN into one polygon to trigger debug output
pred[1, 0, 0] = float('nan')

print("\nTesting with NaN in prediction:")
print(f"pred has NaN: {torch.isnan(pred).any()}")

try:
    loss = mgiou(pred, target)
    print(f"Loss: {loss.item():.6f}")
except RuntimeError as e:
    print("\n✓ Enhanced debug output captured:")
    print("-" * 80)
    print(str(e))
    print("-" * 80)
    
print("\n" + "="*80)
print("Enhanced debug test completed!")
print("="*80)
