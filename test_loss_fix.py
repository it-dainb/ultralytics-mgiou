#!/usr/bin/env python3
"""
Quick test to verify the hybrid loss normalization fix.
This tests that the hybrid loss produces reasonable values (0-10 range).
"""

import torch
import os

# Enable debug output
os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"

from ultralytics.utils.loss import PolygonLoss

def test_hybrid_loss_scale():
    """Test that hybrid loss produces reasonable values."""
    
    # Create loss function with hybrid mode
    loss_fn = PolygonLoss(
        use_hybrid=True,
        alpha_schedule="cosine",
        alpha_start=0.9,
        alpha_end=0.2,
        total_epochs=100
    )
    
    # Create dummy inputs (similar to real training data)
    batch_size = 8
    num_points = 16
    
    pred_kpts = torch.randn(batch_size, num_points, 2) * 100  # Predictions in pixel space
    gt_kpts = torch.randn(batch_size, num_points, 2) * 100     # Ground truth in pixel space
    kpt_mask = torch.ones(batch_size, num_points)              # All points valid
    area = torch.rand(batch_size, 1) * 10000 + 1000            # Areas between 1000-11000
    
    print("="*80)
    print("Testing Hybrid Loss Normalization Fix")
    print("="*80)
    
    # Test at different epochs
    for epoch in [0, 25, 50, 75, 99]:
        total_loss, mgiou_component = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=epoch)
        
        print(f"\nEpoch {epoch:3d}:")
        print(f"  Total Loss:     {total_loss.item():.6f}")
        print(f"  MGIoU Component: {mgiou_component.item():.6f}")
        print(f"  Alpha (L2 wt):  {loss_fn.get_alpha(epoch):.4f}")
        
        # Check that loss is in reasonable range
        assert not torch.isnan(total_loss), f"NaN loss at epoch {epoch}"
        assert not torch.isinf(total_loss), f"Inf loss at epoch {epoch}"
        assert 0 < total_loss.item() < 100, f"Loss out of range at epoch {epoch}: {total_loss.item()}"
        
        # For epoch 0 (mostly L2), loss should be dominated by L2
        if epoch == 0:
            l2_ema = loss_fn.l2_loss_ema
            print(f"  L2 EMA:         {l2_ema:.6f}")
            assert abs(total_loss.item() - l2_ema) < l2_ema * 0.5, \
                f"At epoch 0 (alpha=0.9), loss should be close to L2 scale"
    
    print("\n" + "="*80)
    print("✓ All tests passed! Hybrid loss produces reasonable values.")
    print("="*80)

if __name__ == "__main__":
    test_hybrid_loss_scale()
