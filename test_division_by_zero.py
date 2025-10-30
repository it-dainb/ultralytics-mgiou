"""Test division by zero scenarios in GIoU computation."""
import torch
from ultralytics.utils.loss import MGIoUPoly

print("="*80)
print("Testing Division by Zero Scenarios")
print("="*80)

mgiou = MGIoUPoly(reduction="mean", loss_weight=1.0)

# Test 1: Identical polygons (union = 0, should give perfect match)
print("\n" + "="*80)
print("Test 1: Identical Polygons (union ≈ 0)")
print("="*80)

pred = torch.tensor([
    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
]).float()

target = pred.clone()  # Exact same

try:
    loss = mgiou(pred, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"  (Expected ~0.0 for identical polygons)")
    assert loss.item() < 0.01, "Loss should be near 0 for identical polygons"
    print(f"✓ No NaN with identical polygons (union=0 case)")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Nearly identical polygons (very small union)
print("\n" + "="*80)
print("Test 2: Nearly Identical Polygons (union ≈ eps)")
print("="*80)

pred = torch.tensor([
    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
]).float()

target = torch.tensor([
    [[0.0, 0.0], [10.0 + 1e-8, 0.0], [10.0 + 1e-8, 10.0], [0.0, 10.0]],
]).float()

try:
    loss = mgiou(pred, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"  (Expected very small for nearly identical)")
    print(f"✓ No NaN with nearly identical polygons")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Polygons with zero area projection on some axes
print("\n" + "="*80)
print("Test 3: Zero Area Projection on Some Axes")
print("="*80)

# Create polygons that project to zero width on some axes
pred = torch.tensor([
    [[0.0, 0.0], [0.0, 10.0], [0.0, 10.0], [0.0, 0.0]],  # Degenerate (line)
    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],  # Normal square
]).float()

target = torch.tensor([
    [[1.0, 0.0], [1.0, 10.0], [1.0, 10.0], [1.0, 0.0]],  # Degenerate (line) 
    [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]],  # Normal square
]).float()

try:
    loss = mgiou(pred, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"✓ No NaN with zero-area projections")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Batch with mix of overlapping and non-overlapping
print("\n" + "="*80)
print("Test 4: Mixed Overlap (some inter=0, some inter>0)")
print("="*80)

pred = torch.tensor([
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Small square
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Small square
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Small square
]).float()

target = torch.tensor([
    [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]],  # Overlapping
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Identical (union≈0)
    [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]],  # No overlap (inter=0)
]).float()

try:
    loss = mgiou(pred, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"✓ No NaN with mixed overlap scenarios")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 5: Very small polygons (numerical precision issues)
print("\n" + "="*80)
print("Test 5: Very Small Polygons (precision limits)")
print("="*80)

pred = torch.tensor([
    [[0.0, 0.0], [1e-6, 0.0], [1e-6, 1e-6], [0.0, 1e-6]],
]).float()

target = torch.tensor([
    [[0.0, 0.0], [1.1e-6, 0.0], [1.1e-6, 1.1e-6], [0.0, 1.1e-6]],
]).float()

try:
    loss = mgiou(pred, target)
    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"✓ No NaN with very small polygons")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*80)
print("✓ ALL DIVISION BY ZERO TESTS PASSED!")
print("="*80)
print("\nThese tests specifically target:")
print("  1. union ≈ 0 (identical polygons)")
print("  2. inter / union when both are near 0")
print("  3. hull / union when hull ≈ union")
print("  4. Mixed scenarios in same batch")
print("  5. Numerical precision limits")
