#!/usr/bin/env python3
"""
Diagnostic script to understand why polygon loss isn't decreasing.
Hooks into the loss computation to track gradient magnitudes and NaN replacements.
"""

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import MGIoUPoly

# Global counters
nan_replacement_counts = {
    'proj1': 0,
    'proj2': 0,
    'inter': 0,
    'hull': 0,
    'iou_term': 0,
    'penalty_term': 0,
    'giou1d': 0,
}

def diagnose_mgiou():
    """Test MGIoU with realistic training-like data."""
    print("="*70)
    print("Diagnosing MGIoU Polygon Loss")
    print("="*70)
    
    # Simulate training batch
    batch_size = 8
    num_vertices = 4
    
    # Create somewhat realistic predictions (normalized coordinates)
    # Model output might be random initially
    pred = torch.randn(batch_size, num_vertices, 2) * 0.3  # Small random values
    pred.requires_grad = True
    
    # Target polygons (ground truth)
    target = torch.tensor([
        [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],  # Valid square
        [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],  # Valid square
        [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],  # Valid square
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Degenerate (all zeros)
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],  # Degenerate (point)
        [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],  # Valid square
        [[0.1, 0.1], [0.1, 0.1], [0.2, 0.2], [0.2, 0.2]],  # Very small
        [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],  # Valid square
    ])
    
    stride = torch.ones(batch_size, 1) * 32.0  # Typical stride value
    
    print("\nInput Statistics:")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"  Pred mean: {pred.mean().item():.4f}")
    print(f"  Target shape: {target.shape}")
    print(f"  Target valid polygons: {(target.abs().sum(dim=(1,2)) > 0.01).sum().item()}/{batch_size}")
    
    # Initialize loss
    loss_fn = MGIoUPoly()
    
    # Forward pass
    print("\nComputing loss...")
    loss = loss_fn(pred, target, stride)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    # Backward pass
    print("\nComputing gradients...")
    loss.backward()
    
    if pred.grad is not None:
        grad = pred.grad
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient mean: {grad.mean().item():.6e}")
        print(f"  Gradient std: {grad.std().item():.6e}")
        print(f"  Gradient range: [{grad.min().item():.6e}, {grad.max().item():.6e}]")
        
        # Check gradient magnitude
        grad_norm = grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.6e}")
        
        # Count non-zero gradients
        non_zero = (grad.abs() > 1e-10).sum().item()
        total = grad.numel()
        print(f"  Non-zero gradients: {non_zero}/{total} ({100*non_zero/total:.1f}%)")
        
        # Analysis
        print("\nGradient Analysis:")
        if grad_norm < 1e-6:
            print("  ⚠️  WARNING: Gradient norm is VERY small!")
            print("      This could explain why loss isn't decreasing.")
            print("      Possible causes:")
            print("      - Too many NaN replacements with 0")
            print("      - Degenerate polygons dominating the batch")
            print("      - Loss function saturated")
        elif grad_norm < 1e-3:
            print("  ⚠️  Gradient norm is small but present")
            print("      Learning might be very slow")
        else:
            print("  ✅ Gradient norm looks reasonable")
    else:
        print("  ❌ ERROR: No gradients computed!")
    
    print("\n" + "="*70)


def test_with_cls_loss():
    """Simulate the effect of high classification loss on polygon gradients."""
    print("\n" + "="*70)
    print("Testing Gradient Competition: Polygon vs Classification Loss")
    print("="*70)
    
    # Create dummy network output
    poly_pred = torch.randn(8, 4, 2, requires_grad=True)
    poly_target = torch.rand(8, 4, 2)
    stride = torch.ones(8, 1) * 32.0
    
    cls_pred = torch.randn(8, 10, requires_grad=True)  # 10 classes
    cls_target = torch.randint(0, 10, (8,))
    
    # Compute losses
    poly_loss_fn = MGIoUPoly()
    poly_loss = poly_loss_fn(poly_pred, poly_target, stride)
    
    cls_loss = F.cross_entropy(cls_pred, cls_target)
    
    # Scale cls_loss to match what we see in training (~650)
    cls_loss = cls_loss * 250  # Amplify to ~650
    
    print(f"\nLoss values:")
    print(f"  Polygon loss: {poly_loss.item():.2f}")
    print(f"  Classification loss: {cls_loss.item():.2f}")
    print(f"  Ratio (cls/poly): {(cls_loss/poly_loss).item():.1f}x")
    
    # Combined loss (as done in training)
    total_loss = poly_loss + cls_loss
    
    print(f"\nBackpropagating combined loss...")
    total_loss.backward()
    
    # Check gradient magnitudes
    poly_grad_norm = poly_pred.grad.norm().item()
    cls_grad_norm = cls_pred.grad.norm().item()
    
    print(f"\nGradient norms:")
    print(f"  Polygon gradient norm: {poly_grad_norm:.6e}")
    print(f"  Classification gradient norm: {cls_grad_norm:.6e}")
    print(f"  Ratio (cls/poly): {(cls_grad_norm/poly_grad_norm):.1f}x")
    
    print("\nAnalysis:")
    if cls_grad_norm / poly_grad_norm > 100:
        print("  ⚠️  WARNING: Classification gradients DOMINATE polygon gradients!")
        print("      The polygon loss signal is being drowned out.")
        print("      Suggestions:")
        print("      1. Increase polygon loss weight")
        print("      2. Fix classification loss (it's abnormally high)")
        print("      3. Use gradient clipping per-loss component")
    
    print("="*70)


if __name__ == '__main__':
    diagnose_mgiou()
    test_with_cls_loss()
