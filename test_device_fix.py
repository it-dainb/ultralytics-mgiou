"""Test device consistency fix for MGIoUPoly"""
import torch
from ultralytics.utils.loss import MGIoUPoly

print("=" * 80)
print("Testing Device Consistency Fix")
print("=" * 80)

# Test on CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Create loss
loss_fn = MGIoUPoly(reduction="mean", adaptive_convex_pow=True)
print(f"Loss function created on CPU (buffers)")

# Create sample data on CUDA
batch_size = 4
num_vertices = 4
pred = torch.randn(batch_size, num_vertices, 2, device=device, requires_grad=True)
target = torch.randn(batch_size, num_vertices, 2, device=device)
weights = torch.ones(batch_size, device=device)

print(f"\nInput tensors:")
print(f"  pred device: {pred.device}")
print(f"  target device: {target.device}")
print(f"  weights device: {weights.device}")

print(f"\nBuffer devices before forward:")
print(f"  _mgiou_ema device: {loss_fn._mgiou_ema.device}")
print(f"  _prev_pow device: {loss_fn._prev_pow.device}")

# Forward pass
try:
    loss = loss_fn(pred, target, weight=weights)
    print(f"\n✓ Forward pass successful!")
    print(f"  loss value: {loss.item():.6f}")
    print(f"  loss device: {loss.device}")
    
    print(f"\nBuffer devices after forward:")
    print(f"  _mgiou_ema device: {loss_fn._mgiou_ema.device}")
    print(f"  _prev_pow device: {loss_fn._prev_pow.device}")
    
    # Backward pass
    loss.backward()
    print(f"\n✓ Backward pass successful!")
    print(f"  pred.grad device: {pred.grad.device}")
    print(f"  grad mean: {pred.grad.mean().item():.6e}")
    
    print("\n" + "=" * 80)
    print("✅ DEVICE CONSISTENCY TEST PASSED!")
    print("=" * 80)
    
except RuntimeError as e:
    print(f"\n❌ Error: {e}")
    print("\n" + "=" * 80)
    print("❌ DEVICE CONSISTENCY TEST FAILED!")
    print("=" * 80)
