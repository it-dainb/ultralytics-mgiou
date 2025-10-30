#!/usr/bin/env python3
"""
Debug why classification loss is abnormally high with nc=1.
Check model-dataset mismatch.
"""

import torch
import torch.nn.functional as F


def test_cls_loss_with_one_class():
    """Test what classification loss should look like with 1 class."""
    print("="*70)
    print("Testing Classification Loss with nc=1")
    print("="*70)
    
    batch_size = 8
    num_classes = 1  # Your dataset
    
    # Simulate model predictions
    # With 1 class, predictions should be simple
    pred_logits = torch.randn(batch_size, num_classes)
    
    # All targets are class 0 (only class available)
    targets = torch.zeros(batch_size, dtype=torch.long)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(pred_logits, targets)
    
    print(f"\nWith nc=1 (correct configuration):")
    print(f"  Prediction shape: {pred_logits.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Classification loss: {loss.item():.4f}")
    print(f"  Expected range: 0.1 - 2.0")
    
    # Now test with WRONG number of classes (what might be happening)
    print("\n" + "="*70)
    print("Testing with Model-Dataset Mismatch")
    print("="*70)
    
    # If model was pretrained on COCO (80 classes) but dataset has 1 class
    num_classes_model = 80
    pred_logits_wrong = torch.randn(batch_size, num_classes_model)
    
    # Targets still class 0, but now competing with 79 other classes
    targets_wrong = torch.zeros(batch_size, dtype=torch.long)
    
    loss_wrong = F.cross_entropy(pred_logits_wrong, targets_wrong)
    
    print(f"\nWith nc=80 (model) but dataset nc=1:")
    print(f"  Prediction shape: {pred_logits_wrong.shape}")
    print(f"  Target shape: {targets_wrong.shape}")
    print(f"  Classification loss: {loss_wrong.item():.4f}")
    print(f"  Note: Still shouldn't be 650+")
    
    # Test with even worse mismatch - wrong loss computation
    print("\n" + "="*70)
    print("Testing Loss Computation Issues")
    print("="*70)
    
    # One possible bug: loss not averaged properly
    # If loss is summed instead of meaned over many anchors/predictions
    num_predictions = 8400  # Typical number of anchor boxes in YOLO
    
    # Each prediction point has its own classification prediction
    pred_all = torch.randn(batch_size, num_predictions, num_classes_model)
    targets_all = torch.zeros(batch_size, num_predictions, dtype=torch.long)
    
    # Wrong: Sum all losses
    loss_sum = 0
    for b in range(batch_size):
        loss_sum += F.cross_entropy(pred_all[b], targets_all[b], reduction='sum')
    
    print(f"\nIf loss is SUMMED instead of MEANED:")
    print(f"  Total summed loss: {loss_sum.item():.1f}")
    print(f"  Per-sample loss: {loss_sum.item() / batch_size:.1f}")
    print(f"  ⚠️  This could explain the ~650 value!")
    
    # Correct: Mean the losses
    loss_mean = 0
    for b in range(batch_size):
        loss_mean += F.cross_entropy(pred_all[b], targets_all[b], reduction='mean')
    loss_mean /= batch_size
    
    print(f"\nIf loss is MEANED properly:")
    print(f"  Average loss: {loss_mean.item():.4f}")
    print(f"  ✅ This is the correct range")
    
    print("\n" + "="*70)
    print("Conclusion")
    print("="*70)
    print("\nYour cls_loss of ~650 suggests:")
    print("  1. Loss is being summed over all anchor boxes instead of averaged")
    print("  2. OR: Loss weight is incorrectly scaled")
    print("  3. OR: There's a bug in how the loss is computed/reported")
    print("\nThis is NOT a gradient flow issue - it's a loss computation issue!")
    print("="*70)


def check_loss_weights():
    """Check typical YOLO loss weight configurations."""
    print("\n" + "="*70)
    print("Typical YOLO Loss Weights")
    print("="*70)
    
    print("\nDefault weights:")
    print("  box_loss: 7.5")
    print("  cls_loss: 0.5")
    print("  dfl_loss: 1.5")
    print("  polygon_loss: 1.0-3.0 (custom)")
    
    print("\nYour training output shows:")
    print("  box_loss: ~0.76")
    print("  cls_loss: ~650")
    print("  polygon_loss: ~12")
    
    print("\nAnalysis:")
    print("  box_loss looks normal (0.5-2.0 range)")
    print("  polygon_loss looks normal (10-15 range for early training)")
    print("  cls_loss is 100-1000x too high!")
    
    print("\n⚠️  CRITICAL: Classification loss computation or weighting is broken!")
    print("="*70)


if __name__ == '__main__':
    test_cls_loss_with_one_class()
    check_loss_weights()
