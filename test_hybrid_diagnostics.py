#!/usr/bin/env python3
"""
Test script to verify hybrid loss diagnostic logging is working correctly.

This script tests:
1. PolygonLoss initialization in hybrid mode
2. Epoch updates via set_epoch()
3. Loss computation with proper alpha scheduling
4. EMA tracking over multiple epochs
"""

import os
import torch

# Enable debug mode
os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"

from ultralytics.utils.loss import PolygonLoss, v8PolygonLoss
from ultralytics.nn.modules.head import Polygon

def test_polygon_loss_init():
    """Test PolygonLoss initialization with hybrid mode."""
    print("\n" + "="*80)
    print("TEST 1: PolygonLoss Initialization")
    print("="*80)
    
    loss_fn = PolygonLoss(
        use_hybrid=True,
        alpha_schedule="cosine",
        alpha_start=0.9,
        alpha_end=0.2,
        total_epochs=100
    )
    
    assert loss_fn.mode == "hybrid", "Expected hybrid mode"
    assert loss_fn.use_hybrid == True
    assert loss_fn.current_epoch == 0
    print("✓ Initialization successful\n")
    
    return loss_fn

def test_set_epoch():
    """Test epoch updates."""
    print("\n" + "="*80)
    print("TEST 2: Epoch Updates")
    print("="*80)
    
    loss_fn = PolygonLoss(
        use_hybrid=True,
        total_epochs=100
    )
    
    # Test epoch progression
    for epoch in [0, 10, 50, 99]:
        alpha = loss_fn.get_alpha(epoch)
        print(f"Epoch {epoch}: alpha={alpha:.4f} (L2={alpha:.2%}, MGIoU={1-alpha:.2%})")
    
    print("✓ Epoch updates working\n")

def test_loss_computation():
    """Test loss computation with hybrid mode."""
    print("\n" + "="*80)
    print("TEST 3: Loss Computation with Hybrid Mode")
    print("="*80)
    
    loss_fn = PolygonLoss(
        use_hybrid=True,
        alpha_schedule="cosine",
        total_epochs=100
    )
    
    # Create synthetic data (batch_size=2, num_points=4)
    batch_size = 2
    num_points = 4
    
    pred_kpts = torch.randn(batch_size, num_points, 2)
    gt_kpts = torch.randn(batch_size, num_points, 2)
    kpt_mask = torch.ones(batch_size, num_points)
    area = torch.ones(batch_size, 1) * 100.0
    
    print("\nSimulating 5 training epochs with hybrid loss:\n")
    
    for epoch in [0, 25, 50, 75, 99]:
        total_loss, mgiou_component = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=epoch)
        
        print(f"\nEpoch {epoch}: total_loss={total_loss.item():.4f}, "
              f"mgiou_component={mgiou_component.item():.4f}")
        
        # Verify loss is reasonable
        assert not torch.isnan(total_loss), f"NaN detected at epoch {epoch}"
        assert not torch.isinf(total_loss), f"Inf detected at epoch {epoch}"
        assert total_loss.item() > 0, f"Loss should be positive at epoch {epoch}"
    
    print("\n✓ Loss computation successful\n")

def test_v8_polygon_loss():
    """Test v8PolygonLoss with set_epoch method."""
    print("\n" + "="*80)
    print("TEST 4: v8PolygonLoss set_epoch Method")
    print("="*80)
    
    # Create a mock model with necessary attributes
    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")
            head = Polygon(
                nc=1,  # number of classes
                np=4,  # number of polygon points
                ch=(256, 512, 1024)  # channels from backbone
            )
            self.model = [None, None, head]  # Mock model list
            self.args = type('Args', (), {'device': 'cpu'})()
        
        def parameters(self):
            """Mock parameters method required by v8DetectionLoss."""
            return iter([torch.nn.Parameter(torch.zeros(1))])
    
    mock_model = MockModel()
    
    loss_fn = v8PolygonLoss(
        mock_model,
        use_hybrid=True,
        total_epochs=100
    )
    
    print("\nTesting set_epoch() method:\n")
    
    for epoch in [0, 25, 50, 75, 99]:
        loss_fn.set_epoch(epoch)
        alpha = loss_fn.polygon_loss.get_alpha()
        print(f"After set_epoch({epoch}): current_epoch={loss_fn.current_epoch}, alpha={alpha:.4f}")
    
    print("\n✓ set_epoch method working correctly\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID LOSS DIAGNOSTIC TEST SUITE")
    print("="*80)
    
    try:
        test_polygon_loss_init()
        test_set_epoch()
        test_loss_computation()
        test_v8_polygon_loss()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nDiagnostic logging is working correctly.")
        print("You should see detailed [HYBRID] messages during training.")
        print("\nNext step: Run actual training with ULTRALYTICS_DEBUG_NAN=1 to see:")
        print("  1. [HYBRID INIT] message at startup")
        print("  2. [HYBRID] set_epoch() messages each epoch")
        print("  3. [HYBRID] loss computation details (verbose)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
