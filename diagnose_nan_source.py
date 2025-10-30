"""Diagnose source of NaN in predictions"""
import torch
import torch.nn as nn

# Simulate the kpts_decode process
def kpts_decode(anchor_points, pred_kpts):
    """Decode predicted keypoints to image coordinates."""
    print(f"\n=== kpts_decode ===")
    print(f"anchor_points: shape={anchor_points.shape}, has_nan={torch.isnan(anchor_points).any()}")
    print(f"  min={anchor_points.min():.4f}, max={anchor_points.max():.4f}")
    print(f"pred_kpts input: shape={pred_kpts.shape}, has_nan={torch.isnan(pred_kpts).any()}")
    if torch.isnan(pred_kpts).any():
        print(f"  ⚠️ NaN count in input: {torch.isnan(pred_kpts).sum()}/{pred_kpts.numel()}")
    else:
        print(f"  min={pred_kpts.min():.4f}, max={pred_kpts.max():.4f}")
    
    y = pred_kpts.clone()
    print(f"After clone: has_nan={torch.isnan(y).any()}")
    
    y[..., :2] *= 2.0
    print(f"After *= 2.0: has_nan={torch.isnan(y).any()}")
    if torch.isnan(y).any():
        print(f"  ⚠️ NaN count: {torch.isnan(y).sum()}/{y.numel()}")
    
    y[..., 0] += anchor_points[:, [0]] - 0.5
    print(f"After x += anchor: has_nan={torch.isnan(y).any()}")
    if torch.isnan(y).any():
        print(f"  ⚠️ NaN count: {torch.isnan(y).sum()}/{y.numel()}")
    
    y[..., 1] += anchor_points[:, [1]] - 0.5
    print(f"After y += anchor: has_nan={torch.isnan(y).any()}")
    if torch.isnan(y).any():
        print(f"  ⚠️ NaN count: {torch.isnan(y).sum()}/{y.numel()}")
    
    return y

print("=" * 80)
print("Testing kpts_decode with normal inputs")
print("=" * 80)

# Test 1: Normal case
anchor_points = torch.randn(100, 2) * 10 + 50
pred_kpts = torch.randn(100, 4, 2) * 0.5
decoded = kpts_decode(anchor_points, pred_kpts)
print(f"\n✓ Result: has_nan={torch.isnan(decoded).any()}")

print("\n" + "=" * 80)
print("Testing kpts_decode with NaN in pred_kpts")
print("=" * 80)

# Test 2: NaN in pred_kpts
pred_kpts_nan = torch.full((100, 4, 2), float('nan'))
decoded_nan = kpts_decode(anchor_points, pred_kpts_nan)
print(f"\n✗ Result: has_nan={torch.isnan(decoded_nan).any()}")

print("\n" + "=" * 80)
print("Testing kpts_decode with inf in pred_kpts")
print("=" * 80)

# Test 3: Inf in pred_kpts
pred_kpts_inf = torch.full((100, 4, 2), float('inf'))
decoded_inf = kpts_decode(anchor_points, pred_kpts_inf)
print(f"\n✗ Result: has_nan={torch.isnan(decoded_inf).any()}")

print("\n" + "=" * 80)
print("Potential sources of NaN in neural network output:")
print("=" * 80)
print("""
1. **Gradient explosion** - weights become inf/nan during training
2. **Numerical instability in activations** - log(0), sqrt(-1), etc.
3. **Division by zero** - in normalization layers
4. **Overflow in operations** - exp(large_number)
5. **Bad initialization** - starting with nan/inf weights
6. **Loss becoming NaN** - propagates back to weights
7. **Learning rate too high** - causes gradient explosion

Common fixes:
- Check model weights: model.state_dict() for nan/inf
- Add gradient clipping: torch.nn.utils.clip_grad_norm_()
- Reduce learning rate
- Check batch normalization layers
- Verify loss function is stable
- Add NaN checks after each layer during forward pass
""")
