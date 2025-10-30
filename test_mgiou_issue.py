"""
Test to understand why loss is not decreasing after NaN fix.

The issue: Replacing NaN with zeros may cause:
1. All zero axes -> all zero projections -> zero IoU -> max loss
2. Incorrect gradients due to replacement operations
3. Masking preventing learning
"""

import torch
import torch.nn as nn

# Simulate the issue
def test_nan_replacement_impact():
    print("=" * 70)
    print("TEST 1: Impact of NaN replacement with zeros")
    print("=" * 70)
    
    # Case 1: Normal polygon (should work)
    pred_normal = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ], requires_grad=True)
    
    target_normal = torch.tensor([
        [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
    ])
    
    # Case 2: Polygon with NaN (what happens after replacement)
    pred_with_nan = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [float('nan'), float('nan')], [0.0, 1.0]]
    ], requires_grad=True)
    
    # After replacement (what our fix does)
    pred_replaced = pred_with_nan.clone()
    pred_replaced = torch.where(torch.isnan(pred_replaced), torch.zeros_like(pred_replaced), pred_replaced)
    
    print("\nOriginal polygon with NaN:")
    print(pred_with_nan)
    print("\nAfter replacement with zeros:")
    print(pred_replaced)
    print("\nProblem: Now we have duplicate vertices at (0,0)!")
    print("This creates degenerate edges and zero-length normals")
    
    # Case 3: What if all predictions are bad?
    print("\n" + "=" * 70)
    print("TEST 2: All vertices collapsed (common in early training)")
    print("=" * 70)
    
    pred_collapsed = torch.tensor([
        [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
    ])
    
    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])
    
    print("Predicted (all same):", pred_collapsed)
    print("Target:", target)
    
    # Compute edges
    edges_pred = pred_collapsed.roll(-1, dims=1) - pred_collapsed
    edges_target = target.roll(-1, dims=1) - target
    
    print("\nPrediction edges (all zero!):", edges_pred)
    print("Target edges:", edges_target)
    
    # Compute edge lengths
    edge_lengths_pred = torch.norm(edges_pred, dim=-1, keepdim=True)
    edge_lengths_target = torch.norm(edges_target, dim=-1, keepdim=True)
    
    print("\nPrediction edge lengths:", edge_lengths_pred.squeeze())
    print("Target edge lengths:", edge_lengths_target.squeeze())
    
    # What happens with mask?
    eps = 1e-6
    mask_pred = (edge_lengths_pred.squeeze(-1) > eps)
    mask_target = (edge_lengths_target.squeeze(-1) > eps)
    
    print("\nPrediction mask (valid edges):", mask_pred)
    print("Target mask (valid edges):", mask_target)
    print("\nProblem: All prediction edges are invalid!")
    
    # What's the fix doing?
    all_invalid = ~mask_pred.any(dim=1)
    print("\nAll prediction edges invalid?", all_invalid)
    
    if all_invalid.any():
        mask_pred[all_invalid, 0] = True
        print("After fix: Force first edge valid:", mask_pred)
        print("But the edge is still zero-length! This creates a zero normal.")
    
    print("\n" + "=" * 70)
    print("TEST 3: Impact on loss calculation")
    print("=" * 70)
    
    # When all normals are zero or invalid, what's the GIoU?
    # Projection = polygon @ normals.T
    # If normals are all zeros -> projection is all zeros
    # min=0, max=0 for both pred and target
    # inter = min(0,0) - max(0,0) = 0
    # hull = max(0,0) - min(0,0) = 0
    # IoU = 0 / eps ≈ 0
    # GIoU = 0 - penalty ≈ -1 (worst case)
    # Loss = (1 - (-1)) / 2 = 1.0
    
    print("When normals are all zeros:")
    print("- Projections: all zeros")
    print("- inter: 0")
    print("- hull: 0")
    print("- IoU: 0 / eps ≈ 0")
    print("- GIoU: -1 (maximum penalty)")
    print("- Loss: (1 - (-1)) / 2 = 1.0")
    print("\nBUT: Gradient is zero because we used .where() to replace NaN!")
    print("The network can't learn because gradients are blocked.")

if __name__ == "__main__":
    test_nan_replacement_impact()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The NaN fixes have two critical problems:

1. GRADIENT BLOCKING:
   Using torch.where() to replace NaN breaks the gradient flow.
   The network can't learn from these samples.

2. DEGENERATE GEOMETRY:
   Replacing NaN with zeros creates duplicate vertices, which leads to:
   - Zero-length edges
   - Zero normals
   - Zero projections
   - Invalid IoU calculations
   - Maximum loss but no learning signal

BETTER SOLUTION:
Instead of replacing NaN with zeros, we should:
1. Detect NaN early (at input)
2. Skip these samples entirely (don't compute loss for them)
3. Or use L1 fallback loss (which already exists in the code)
4. Let the network learn from valid samples only
5. Add gradient clipping to prevent NaN from occurring

The current approach of "NaN in -> 0 out" creates valid numbers but
destroys the learning signal. The loss appears finite but gradients are zero.
""")
