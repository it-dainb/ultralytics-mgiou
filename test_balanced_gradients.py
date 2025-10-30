"""
Test that classification loss normalization produces balanced gradients.

This test verifies that after normalizing the classification loss by the number
of anchors, the gradient magnitudes are comparable between classification and
polygon losses, allowing both to contribute meaningfully to training.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from types import SimpleNamespace

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.utils.loss import v8PolygonLoss


class MockDetectHead(nn.Module):
    """Mock detection head for testing."""
    def __init__(self, nc=1, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.poly_shape = (32,)  # Tuple: (16 points * 2 coords,)
        
class MockModel(nn.Module):
    """Mock YOLO model for testing."""
    def __init__(self, nc=1):
        super().__init__()
        self.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            polygon=2.5,
        )
        self.model = [MockDetectHead(nc=nc)]
        # Need at least one parameter for device detection
        self.dummy = nn.Parameter(torch.zeros(1))


def test_balanced_gradients():
    """Test that cls and polygon losses have comparable gradient magnitudes."""
    print("\n" + "="*70)
    print("Testing Balanced Gradient Magnitudes After Normalization")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create mock model
    model = MockModel(nc=1).to(device)
    
    # Create loss function with MGIoU
    loss_fn = v8PolygonLoss(model=model, use_mgiou=True)
    
    # Realistic scenario: 1 class, 8400 anchors, ~16 objects per batch
    batch_size = 8
    num_anchors = 8400
    num_classes = 1
    num_objects_per_image = 2
    total_objects = batch_size * num_objects_per_image
    
    print(f"Scenario:")
    print(f"  Batch size: {batch_size}")
    print(f"  Anchors per image: {num_anchors}")
    print(f"  Objects per image: {num_objects_per_image}")
    print(f"  Total objects: {total_objects}")
    print(f"  Anchor:Object ratio: {num_anchors}:{num_objects_per_image} = {num_anchors//num_objects_per_image}:1\n")
    
    # Create dummy model predictions
    # preds format: [feats, pred_poly] where feats is a list of feature maps
    # feats needs to be split into pred_distri and pred_scores
    pred_distri = torch.randn(batch_size, num_anchors, 64, device=device, requires_grad=True)
    pred_scores = torch.randn(batch_size, num_anchors, num_classes, device=device, requires_grad=True)
    pred_poly = torch.randn(batch_size, num_anchors, 32, device=device, requires_grad=True)  # 16 points * 2 coords
    
    # Concatenate pred_distri and pred_scores, then reshape for feature maps
    # The loss expects feats as list of tensors with shape (B, no, H*W) where no = nc + reg_max*4
    no = num_classes + 16 * 4  # nc + reg_max * 4
    combined = torch.cat([pred_distri, pred_scores], dim=2).permute(0, 2, 1)  # (B, no, num_anchors)
    # Split into 3 feature levels (matching stride [8, 16, 32])
    feat_splits = [combined[:, :, :2800], combined[:, :, 2800:5600], combined[:, :, 5600:]]
    feats = [f.view(batch_size, no, -1) for f in feat_splits]
    
    # preds should be [feats, pred_poly]
    preds = [feats, pred_poly.permute(0, 2, 1)]  # pred_poly needs to be (B, 32, num_anchors)
    
    # Create dummy batch data
    batch_idx = torch.cat([torch.full((num_objects_per_image,), i, device=device) for i in range(batch_size)])
    cls = torch.zeros(total_objects, device=device)
    bboxes = torch.rand(total_objects, 4, device=device) * 0.5 + 0.25  # Random boxes in [0.25, 0.75]
    polygons = torch.rand(total_objects, 16, 2, device=device) * 0.5 + 0.25  # Random polygons
    
    batch = {
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'polygons': polygons,
    }
    
    # Forward pass
    
    try:
        loss, _ = loss_fn(preds, batch)
        total_loss = loss.sum()
        
        # Backward pass
        total_loss.backward()
        
        # Analyze gradients
        cls_grad_magnitude = pred_scores.grad.abs().mean().item()
        poly_grad_magnitude = pred_poly.grad.abs().mean().item()
        
        print("Loss Components:")
        print(f"  Box loss:        {loss[0].item():.4f}")
        print(f"  Polygon loss:    {loss[1].item():.4f}")
        print(f"  Classification:  {loss[2].item():.4f}")
        print(f"  DFL loss:        {loss[3].item():.4f}")
        print(f"  Total loss:      {total_loss.item():.4f}\n")
        
        print("Gradient Magnitudes:")
        print(f"  Classification:  {cls_grad_magnitude:.6f}")
        print(f"  Polygon:         {poly_grad_magnitude:.6f}")
        
        # Calculate balance ratio
        if poly_grad_magnitude > 0:
            ratio = cls_grad_magnitude / poly_grad_magnitude
            print(f"  Ratio (cls/poly): {ratio:.1f}:1\n")
            
            # Check if gradients are balanced (within 2 orders of magnitude)
            if ratio < 100:
                print("✅ PASS: Gradients are reasonably balanced!")
                print(f"   Classification gradients are only {ratio:.1f}x larger than polygon gradients.")
                print("   Both losses can contribute meaningfully to training.\n")
                return True
            else:
                print("❌ FAIL: Gradients are still imbalanced!")
                print(f"   Classification gradients are {ratio:.1f}x larger than polygon gradients.")
                print("   Polygon loss will still be dominated during training.\n")
                return False
        else:
            print("⚠️  WARNING: Polygon gradients are zero!\n")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cls_loss_magnitude():
    """Test that classification loss has reasonable magnitude (not 600+)."""
    print("="*70)
    print("Testing Classification Loss Magnitude")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create mock model
    model = MockModel(nc=1).to(device)
    
    loss_fn = v8PolygonLoss(model=model, use_mgiou=True)
    
    # Same realistic scenario
    batch_size = 8
    num_anchors = 8400
    num_classes = 1
    num_objects_per_image = 2
    total_objects = batch_size * num_objects_per_image
    
    # Create predictions in correct format
    pred_distri = torch.randn(batch_size, num_anchors, 64, device=device)
    pred_scores = torch.randn(batch_size, num_anchors, num_classes, device=device)
    pred_poly = torch.randn(batch_size, num_anchors, 32, device=device)
    
    no = num_classes + 16 * 4
    combined = torch.cat([pred_distri, pred_scores], dim=2).permute(0, 2, 1)
    feat_splits = [combined[:, :, :2800], combined[:, :, 2800:5600], combined[:, :, 5600:]]
    feats = [f.view(batch_size, no, -1) for f in feat_splits]
    preds = [feats, pred_poly.permute(0, 2, 1)]
    
    batch_idx = torch.cat([torch.full((num_objects_per_image,), i, device=device) for i in range(batch_size)])
    cls = torch.zeros(total_objects, device=device)
    bboxes = torch.rand(total_objects, 4, device=device) * 0.5 + 0.25
    polygons = torch.rand(total_objects, 16, 2, device=device) * 0.5 + 0.25
    
    batch = {
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'polygons': polygons,
    }
    
    try:
        loss, _ = loss_fn(preds, batch)
        cls_loss = loss[2].item()
        
        print(f"Classification loss: {cls_loss:.4f}\n")
        
        # Before normalization, cls_loss was ~600-710
        # After normalization, it should be more reasonable (< 10)
        if cls_loss < 10:
            print("✅ PASS: Classification loss has reasonable magnitude!")
            print(f"   Loss is {cls_loss:.4f}, which is comparable to other loss components.\n")
            return True
        else:
            print("❌ FAIL: Classification loss is still too high!")
            print(f"   Loss is {cls_loss:.4f}, expected < 10.\n")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Testing Classification Loss Normalization Fix")
    print("#"*70)
    
    # Run tests
    test1_passed = test_cls_loss_magnitude()
    test2_passed = test_balanced_gradients()
    
    # Summary
    print("="*70)
    print("Test Summary")
    print("="*70)
    print(f"Classification loss magnitude:  {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Gradient balance:               {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print()
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! The normalization fix is working correctly.")
        print("\nNext steps:")
        print("1. Run training to verify polygon loss now decreases")
        print("2. Monitor that classification still works (P, R, mAP improve)")
        print()
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Further tuning may be needed.")
        print()
        sys.exit(1)
